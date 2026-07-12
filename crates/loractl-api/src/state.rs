//! Run registry, per-run event history, and the write/read paths.
//!
//! Concurrency model (see ADR-0003): every event is serialized **once** on
//! the blocking training thread and pushed into a per-run history
//! (`Vec<StoredEvent>` behind a `std::sync::Mutex`); a `tokio::sync::watch`
//! channel is the wake signal. Subscribers each hold their own cursor over
//! the history — one code path serves live, late, and finished runs with
//! zero event loss. The mutex is never held across an `.await`: writers are
//! sync code on the blocking thread, readers copy-then-release.

use crate::{ApiConfig, TrainerFactory};
use loractl_core::{TrainConfig, TrainEvent};
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::watch;

/// One event, serialized once at emission time.
///
/// Pre-serialization removes serialize-at-edge panics from the SSE path and
/// makes replaying to N clients a string clone.
#[derive(Debug, Clone)]
pub struct StoredEvent {
    /// The event's `type` discriminator (doubles as the SSE `event:` name).
    pub kind: &'static str,
    /// The event's exact JSON wire form.
    pub json: String,
}

/// A frame handed to a subscriber: the history index plus the stored event.
#[derive(Debug, Clone)]
pub struct StoredFrame {
    /// 0-based per-run history index (the SSE `id:` field).
    pub index: usize,
    pub kind: &'static str,
    pub json: String,
}

/// Append-only per-run event history.
pub struct RunLog {
    pub events: Vec<StoredEvent>,
    /// Set exactly once by the supervisor after the trainer thread joins.
    /// Once true, `events` is final.
    pub done: bool,
}

/// One training run: its history plus the subscriber wake channel.
pub struct Run {
    pub log: Mutex<RunLog>,
    pub notify: watch::Sender<()>,
}

impl Run {
    fn new() -> Self {
        Self {
            log: Mutex::new(RunLog {
                events: Vec::new(),
                done: false,
            }),
            notify: watch::Sender::new(()),
        }
    }
}

/// The run registry: the live map plus the completion order that bounds it.
///
/// Both fields move together under one lock, so `runs` and `completed` can
/// never disagree about which ids are retained.
#[derive(Default)]
struct Registry {
    runs: HashMap<u64, Arc<Run>>,
    /// Ids of **finished** runs in completion order, oldest first. This is the
    /// eviction queue: a run is only a candidate once it lands here, which is
    /// exactly what makes in-flight runs un-evictable (#36).
    completed: VecDeque<u64>,
    /// Runs registered but not yet retired by the supervisor. Bounded by
    /// `ApiConfig::max_concurrent_runs` (#37) — each in-flight run owns a real
    /// compute thread on the blocking pool.
    in_flight: usize,
}

/// Shared application state: the run registry, the server config, and the
/// trainer seam.
pub struct AppState {
    runs: Mutex<Registry>,
    next_id: AtomicU64,
    pub factory: TrainerFactory,
    pub config: ApiConfig,
    /// The canonical directory every run's output is confined under (#37).
    /// Canonicalized once at startup so containment checks compare two
    /// symlink-free absolute paths.
    pub output_base: PathBuf,
}

impl AppState {
    pub fn new(factory: TrainerFactory, config: ApiConfig) -> anyhow::Result<Self> {
        let output_base = crate::paths::canonical_base(&config.output_base)?;
        Ok(Self {
            runs: Mutex::new(Registry::default()),
            next_id: AtomicU64::new(1),
            factory,
            config,
            output_base,
        })
    }

    /// Registers a new run and returns `(id, run)`, or `None` when the
    /// concurrent-run cap is saturated (the handler renders that as `429`).
    ///
    /// The cap check and the insert happen under **one** lock acquisition, so
    /// N simultaneous `POST /runs` cannot all observe a free slot and all take
    /// it. Ids are sequential from 1 and process-local (not stable across
    /// restarts — documented contract).
    pub fn register_run(&self) -> Option<(u64, Arc<Run>)> {
        let mut registry = self.runs.lock().unwrap();
        if registry.in_flight >= self.config.max_concurrent_runs {
            return None;
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let run = Arc::new(Run::new());
        registry.runs.insert(id, Arc::clone(&run));
        registry.in_flight += 1;
        Some((id, run))
    }

    /// Marks a run finished: frees its concurrency slot and enforces the
    /// completed-run cap (#36).
    ///
    /// Called exactly once per run, by the supervisor, **before** the terminal
    /// wake — so a subscriber whose stream has closed is guaranteed to observe
    /// the post-eviction registry, which is what makes eviction deterministically
    /// testable rather than a race.
    ///
    /// Eviction drops the `Arc<Run>` from the map only; a subscriber already
    /// streaming an evicted run holds its own `Arc` and finishes undisturbed.
    pub fn complete_run(&self, id: u64) {
        let mut registry = self.runs.lock().unwrap();
        registry.in_flight = registry.in_flight.saturating_sub(1);
        registry.completed.push_back(id);
        while registry.completed.len() > self.config.run_retention {
            let Some(evicted) = registry.completed.pop_front() else {
                break;
            };
            registry.runs.remove(&evicted);
        }
    }

    pub fn get_run(&self, id: u64) -> Option<Arc<Run>> {
        self.runs.lock().unwrap().runs.get(&id).cloned()
    }
}

/// The API-layer terminal event (the 7th `type`; core owns the other six).
#[derive(Serialize)]
struct FailedEvent {
    r#type: &'static str,
    error: String,
}

fn failed_event(error: String) -> StoredEvent {
    let json = serde_json::to_string(&FailedEvent {
        r#type: "failed",
        error,
    })
    .expect("FailedEvent is always serializable");
    StoredEvent {
        kind: "failed",
        json,
    }
}

fn kind_of(event: &TrainEvent) -> &'static str {
    match event {
        TrainEvent::Started { .. } => "started",
        TrainEvent::Step { .. } => "step",
        TrainEvent::Checkpoint { .. } => "checkpoint",
        TrainEvent::Sample { .. } => "sample",
        TrainEvent::Warning { .. } => "warning",
        TrainEvent::Finished { .. } => "finished",
    }
}

/// Write path (called from the sink on the blocking thread): serialize →
/// lock → push → unlock → wake. Every operation is sync and non-blocking;
/// history *is* the buffer, so there is no backpressure onto training.
fn push_event(run: &Run, event: &TrainEvent) {
    let (kind, json) = match serde_json::to_string(event) {
        Ok(s) => (kind_of(event), s),
        Err(_) => (
            "warning",
            r#"{"type":"warning","message":"unserializable event"}"#.into(),
        ),
    };
    let mut log = run.log.lock().unwrap();
    log.events.push(StoredEvent { kind, json });
    drop(log);
    run.notify.send_replace(());
}

/// Runs the trainer on the blocking pool under a supervisor task.
///
/// The supervisor is the choke point for every terminal path: trainer `Err`
/// and trainer panic both converge into an API-layer `failed` event; the
/// `done` flip is **always** followed by a wake on every arm (including
/// `Ok`) — the missed-wake fix that keeps subscribers from parking forever.
/// It is also where a run leaves the in-flight set and becomes evictable
/// (`complete_run`).
pub fn spawn_run(
    state: Arc<AppState>,
    id: u64,
    run: Arc<Run>,
    config: TrainConfig,
    mut trainer: Box<dyn loractl_core::Trainer + Send>,
) {
    tokio::spawn(async move {
        let sink_run = Arc::clone(&run);
        let joined = tokio::task::spawn_blocking(move || {
            // cli.rs `train()` parity: the caller creates the output dir.
            std::fs::create_dir_all(&config.output.dir).map_err(|e| {
                anyhow::anyhow!("creating output dir {}: {e}", config.output.dir.display())
            })?;
            let mut sink = move |event: TrainEvent| push_event(&sink_run, &event);
            trainer.train(&config, &mut sink)
        })
        .await;
        {
            let mut log = run.log.lock().unwrap();
            match joined {
                Ok(Ok(_)) => {} // trainer emitted Finished itself
                Ok(Err(e)) => log.events.push(failed_event(format!("{e:#}"))),
                Err(_join) => log.events.push(failed_event("trainer panicked".into())),
            }
            log.done = true;
        }
        // Retire the run BEFORE the terminal wake: a subscriber that observes
        // the closed stream has, by that ordering, also observed the eviction.
        state.complete_run(id);
        // MISSED-WAKE FIX: wake on EVERY arm, incl. Ok — a subscriber that
        // consumed the Finished wake and re-parked before `done` was set
        // would otherwise hang forever.
        run.notify.send_replace(());
    });
}

/// Read path: replay from index 0, then live-tail until the run is done.
///
/// Subscribes to the wake channel **before** the first snapshot so no write
/// can slip between snapshot and park. Spurious wakes cost one empty
/// iteration; a snapshot with `done == true` is final, so the stream closes
/// after draining it.
pub fn subscribe(run: Arc<Run>) -> impl futures_core::Stream<Item = StoredFrame> + Send {
    async_stream::stream! {
        let mut rx = run.notify.subscribe();
        let mut cursor = 0usize;
        loop {
            let (batch, done) = {
                let log = run.log.lock().unwrap();
                (log.events[cursor..].to_vec(), log.done)
            };
            for event in batch {
                let frame = StoredFrame {
                    index: cursor,
                    kind: event.kind,
                    json: event.json,
                };
                cursor += 1;
                yield frame;
            }
            if done {
                break;
            }
            if rx.changed().await.is_err() {
                // Sender dropped (cannot happen while we hold the Run, but
                // handled anyway): one final drain, then close.
                let batch = {
                    let log = run.log.lock().unwrap();
                    log.events[cursor..].to_vec()
                };
                for event in batch {
                    let frame = StoredFrame {
                        index: cursor,
                        kind: event.kind,
                        json: event.json,
                    };
                    cursor += 1;
                    yield frame;
                }
                break;
            }
        }
    }
}
