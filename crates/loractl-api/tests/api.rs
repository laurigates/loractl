//! Integration tests for the loractl-api HTTP/SSE surface (design §5).
//!
//! All offline, no ports: requests go through `tower::ServiceExt::oneshot`,
//! which is sound because every stream terminates (supervisor guarantee).
//! Every SSE drain is wrapped in a 5 s timeout so a broken close path fails
//! slow-red instead of hanging CI.

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use loractl_api::{ApiConfig, TrainerFactory};
use loractl_core::{MockTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use tokio::time::timeout;
use tower::ServiceExt;

const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static DIR_SEQ: AtomicU64 = AtomicU64::new(0);

/// The relative output dir a test config asks for. Client-supplied `output.dir`
/// is confined under the server's base (#37), so tests name a *relative* path
/// and the run lands at `base/RUN_DIR`.
const RUN_DIR: &str = "out";

/// A unique per-test output **base** under the system temp dir, created and
/// canonicalized (as the server does), so the server's `create_dir_all` side
/// effect never lands in the repo and every containment assertion compares
/// symlink-free absolute paths.
fn test_base(tag: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!(
        "loractl-api-test-{tag}-{}-{}",
        std::process::id(),
        DIR_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&base).expect("create test output base");
    base.canonicalize().expect("canonicalize test output base")
}

fn config_for(base: &std::path::Path) -> ApiConfig {
    ApiConfig {
        output_base: base.to_path_buf(),
        ..ApiConfig::default()
    }
}

/// An app on the default server config (retention 32, 4 concurrent runs —
/// neither binds in a test that starts a handful of runs).
fn app_with(factory: TrainerFactory, base: &std::path::Path) -> Router {
    loractl_api::app(factory, config_for(base)).expect("app builds")
}

/// An app whose registry retains only `run_retention` completed runs.
fn app_retaining(factory: TrainerFactory, base: &std::path::Path, run_retention: usize) -> Router {
    loractl_api::app(
        factory,
        ApiConfig {
            run_retention,
            ..config_for(base)
        },
    )
    .expect("app builds")
}

/// An app that admits at most `max_concurrent_runs` simultaneous runs.
fn app_limited(
    factory: TrainerFactory,
    base: &std::path::Path,
    max_concurrent_runs: usize,
) -> Router {
    loractl_api::app(
        factory,
        ApiConfig {
            max_concurrent_runs,
            ..config_for(base)
        },
    )
    .expect("app builds")
}

fn mock_app(base: &std::path::Path) -> Router {
    let factory: TrainerFactory = Arc::new(|_| Box::new(MockTrainer));
    app_with(factory, base)
}

/// A minimal valid config as JSON (same schema as the YAML file). `dir` is
/// relative to the server's output base.
fn config_json(dir: &str, steps: u64, checkpoint_every: u64) -> serde_json::Value {
    serde_json::json!({
        "steps": steps,
        "model": { "base": "test-base" },
        "lora": {},
        "dataset": { "path": "unused-dataset" },
        "output": { "dir": dir, "checkpoint_every": checkpoint_every }
    })
}

/// A config naming an explicit `output.dir` / `output.name` — the two fields
/// the confinement check (#37) governs.
fn config_json_output(dir: &str, name: &str) -> serde_json::Value {
    serde_json::json!({
        "steps": 1,
        "model": { "base": "test-base" },
        "lora": {},
        "dataset": { "path": "unused-dataset" },
        "output": { "dir": dir, "name": name, "checkpoint_every": 100 }
    })
}

fn post_runs(body: String) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/runs")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("valid request")
}

fn get_events(id: u64) -> Request<Body> {
    Request::builder()
        .uri(format!("/runs/{id}/events"))
        .body(Body::empty())
        .expect("valid request")
}

async fn body_string(body: Body) -> String {
    let bytes = timeout(DRAIN_TIMEOUT, body.collect())
        .await
        .expect("body drain timed out — stream failed to close")
        .expect("body read failed")
        .to_bytes();
    String::from_utf8(bytes.to_vec()).expect("utf-8 body")
}

/// One parsed SSE event (a block with at least one `data:` line).
#[derive(Debug, Clone, PartialEq)]
struct SseEvent {
    id: Option<String>,
    event: Option<String>,
    data: String,
}

/// Parses SSE text into events. Blocks without `data:` (keep-alive comment
/// frames — lines starting with `:`) are skipped, per the client contract.
fn parse_sse(text: &str) -> Vec<SseEvent> {
    let mut out = Vec::new();
    for block in text.split("\n\n") {
        let mut id = None;
        let mut event = None;
        let mut data_lines: Vec<String> = Vec::new();
        for line in block.lines() {
            if line.starts_with(':') {
                continue; // comment / keep-alive
            } else if let Some(rest) = line.strip_prefix("id:") {
                id = Some(rest.trim_start().to_string());
            } else if let Some(rest) = line.strip_prefix("event:") {
                event = Some(rest.trim_start().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.trim_start().to_string());
            }
        }
        if !data_lines.is_empty() {
            out.push(SseEvent {
                id,
                event,
                data: data_lines.join("\n"),
            });
        }
    }
    out
}

/// Parses only the *complete* SSE blocks in a partial buffer (everything up
/// to the last `\n\n`).
fn parse_complete_sse(buffer: &str) -> Vec<SseEvent> {
    match buffer.rfind("\n\n") {
        Some(end) => parse_sse(&buffer[..end + 2]),
        None => Vec::new(),
    }
}

/// The `type` discriminator of each event's JSON payload.
fn types(events: &[SseEvent]) -> Vec<String> {
    events
        .iter()
        .map(|e| {
            let value: serde_json::Value = serde_json::from_str(&e.data).expect("valid JSON data");
            value["type"].as_str().expect("type field").to_string()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test trainers
// ---------------------------------------------------------------------------

/// Emits `Started`, then fails.
struct FailingTrainer;

impl Trainer for FailingTrainer {
    fn train(
        &mut self,
        _: &TrainConfig,
        sink: &mut dyn FnMut(TrainEvent),
    ) -> anyhow::Result<PathBuf> {
        sink(TrainEvent::Started { total_steps: 1 });
        anyhow::bail!("boom")
    }
}

/// Panics immediately.
struct PanickingTrainer;

impl Trainer for PanickingTrainer {
    fn train(&mut self, _: &TrainConfig, _: &mut dyn FnMut(TrainEvent)) -> anyhow::Result<PathBuf> {
        panic!("kaboom")
    }
}

/// Returns `Ok` without emitting any events (violates the trainer contract).
struct SilentTrainer;

impl Trainer for SilentTrainer {
    fn train(&mut self, _: &TrainConfig, _: &mut dyn FnMut(TrainEvent)) -> anyhow::Result<PathBuf> {
        Ok(PathBuf::from("unused"))
    }
}

/// Emits `Started` then a tight burst of `Step`s then `Finished`, with NO
/// yielding between events — the write path outruns any subscriber, so many
/// events land in the history between two of the subscriber's wakes. Used to
/// pin that `subscribe` re-snapshots the *full* tail on each wake (no event
/// loss when writes coalesce into fewer `watch` notifications).
struct BurstTrainer {
    steps: u64,
}

impl Trainer for BurstTrainer {
    fn train(
        &mut self,
        config: &TrainConfig,
        sink: &mut dyn FnMut(TrainEvent),
    ) -> anyhow::Result<PathBuf> {
        sink(TrainEvent::Started {
            total_steps: self.steps,
        });
        for step in 1..=self.steps {
            sink(TrainEvent::Step {
                step,
                loss: 1.0 / step as f32,
                lr: 1e-4,
            });
        }
        let adapter_path = config.output.dir.join("lora.safetensors");
        sink(TrainEvent::Finished {
            adapter_path: adapter_path.clone(),
        });
        Ok(adapter_path)
    }
}

/// A gate the test opens to release a parked trainer thread.
#[derive(Clone, Default)]
struct Gate(Arc<(Mutex<bool>, Condvar)>);

impl Gate {
    fn open(&self) {
        let (lock, cvar) = &*self.0;
        *lock.lock().unwrap() = true;
        cvar.notify_all();
    }

    fn wait(&self) {
        let (lock, cvar) = &*self.0;
        let mut open = lock.lock().unwrap();
        while !*open {
            open = cvar.wait(open).unwrap();
        }
    }
}

/// Opens the gate when dropped. Declared at the top of every gated test so
/// that a panic *before* the manual `open()` still releases the parked
/// trainer thread — otherwise the unwinding test drops the tokio `Runtime`,
/// whose blocking-pool shutdown blocks forever on the thread stuck in
/// `Gate::wait`, wedging the whole `cargo test` run instead of failing red.
struct OpenOnDrop(Gate);

impl Drop for OpenOnDrop {
    fn drop(&mut self) {
        self.0.open();
    }
}

/// Parks on `start_gate` before emitting anything, emits `Started`, parks on
/// `finish_gate`, then emits steps + `Finished`. The two gates let a test
/// pin *live* tailing deterministically: a subscriber proven parked while
/// `start_gate` is closed can only receive `Started` via the tail path.
struct GatedTrainer {
    start_gate: Gate,
    finish_gate: Gate,
}

impl Trainer for GatedTrainer {
    fn train(
        &mut self,
        config: &TrainConfig,
        sink: &mut dyn FnMut(TrainEvent),
    ) -> anyhow::Result<PathBuf> {
        self.start_gate.wait();
        sink(TrainEvent::Started { total_steps: 2 });
        self.finish_gate.wait();
        sink(TrainEvent::Step {
            step: 1,
            loss: 1.0,
            lr: 0.1,
        });
        sink(TrainEvent::Step {
            step: 2,
            loss: 0.5,
            lr: 0.1,
        });
        let adapter_path = config.output.dir.join("gated.safetensors");
        sink(TrainEvent::Finished {
            adapter_path: adapter_path.clone(),
        });
        Ok(adapter_path)
    }
}

// ---------------------------------------------------------------------------
// Tests (numbering per the M5 design doc §5)
// ---------------------------------------------------------------------------

/// Test 2: POST contract — 201 with id + events_url; ids are sequential.
#[tokio::test(flavor = "multi_thread")]
async fn post_runs_returns_id_and_events_url() {
    let base = test_base("post-contract");
    let app = mock_app(&base);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, 1, 1).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    let body = body_string(response.into_body()).await;
    assert_eq!(body, r#"{"id":1,"events_url":"/runs/1/events"}"#);

    let response = app
        .oneshot(post_runs(config_json(RUN_DIR, 1, 1).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    let body = body_string(response.into_body()).await;
    assert_eq!(body, r#"{"id":2,"events_url":"/runs/2/events"}"#);

    let _ = std::fs::remove_dir_all(&base);
}

/// Test 3 (AC1 e2e): the full mock run streams started → steps/checkpoints
/// → finished, with consecutive `id:`s and `event:` matching the JSON type.
#[tokio::test(flavor = "multi_thread")]
async fn sse_streams_started_through_finished_for_mock_run() {
    let base = test_base("e2e");
    let app = mock_app(&base);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, 5, 2).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    let response = app.oneshot(get_events(1)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["content-type"].to_str().unwrap(),
        "text/event-stream"
    );

    let events = parse_sse(&body_string(response.into_body()).await);
    assert_eq!(
        types(&events),
        vec![
            "started",
            "step",
            "step",
            "checkpoint",
            "step",
            "step",
            "checkpoint",
            "step",
            "finished"
        ]
    );

    // First event carries the planned step count.
    let started: serde_json::Value = serde_json::from_str(&events[0].data).unwrap();
    assert_eq!(started["total_steps"], 5);

    // Steps arrive in order; checkpoints at the configured cadence.
    let steps: Vec<u64> = events
        .iter()
        .filter(|e| e.event.as_deref() == Some("step"))
        .map(|e| {
            serde_json::from_str::<serde_json::Value>(&e.data).unwrap()["step"]
                .as_u64()
                .unwrap()
        })
        .collect();
    assert_eq!(steps, vec![1, 2, 3, 4, 5]);
    let checkpoints: Vec<u64> = events
        .iter()
        .filter(|e| e.event.as_deref() == Some("checkpoint"))
        .map(|e| {
            serde_json::from_str::<serde_json::Value>(&e.data).unwrap()["step"]
                .as_u64()
                .unwrap()
        })
        .collect();
    assert_eq!(checkpoints, vec![2, 4]);

    // Last event is finished with the adapter path RESOLVED under the output
    // base — the request asked for the relative `RUN_DIR`, and what the trainer
    // (and therefore the wire) sees is the confined absolute path.
    let last = events.last().unwrap();
    assert_eq!(last.event.as_deref(), Some("finished"));
    let finished: serde_json::Value = serde_json::from_str(&last.data).unwrap();
    assert_eq!(
        finished["adapter_path"].as_str().unwrap(),
        base.join(RUN_DIR)
            .join("lora.safetensors")
            .to_str()
            .unwrap()
    );

    // SSE ids are 0..n consecutive; event names match the JSON type.
    for (i, event) in events.iter().enumerate() {
        assert_eq!(event.id.as_deref(), Some(i.to_string().as_str()));
        let value: serde_json::Value = serde_json::from_str(&event.data).unwrap();
        assert_eq!(event.event.as_deref(), value["type"].as_str());
    }

    let _ = std::fs::remove_dir_all(&base);
}

/// Test 4: replaying a finished run is byte-for-byte deterministic.
#[tokio::test(flavor = "multi_thread")]
async fn late_subscriber_after_finish_replays_full_history() {
    let base = test_base("replay");
    let app = mock_app(&base);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, 3, 2).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    let first = app.clone().oneshot(get_events(1)).await.unwrap();
    let first_body = body_string(first.into_body()).await;
    assert_eq!(
        types(&parse_sse(&first_body)).last().map(String::as_str),
        Some("finished")
    );

    let second = app.oneshot(get_events(1)).await.unwrap();
    let second_body = body_string(second.into_body()).await;
    assert_eq!(first_body, second_body);

    let _ = std::fs::remove_dir_all(&base);
}

/// Test 5 (missed-wake sentinel): the subscriber is first *proven* parked on
/// an empty history (a short poll comes up empty while `start_gate` is still
/// closed), so the `started` frame it then receives can only have arrived
/// via the live tail — not connect-time replay. A mutant that replays at
/// connect and batch-drains only at `done` fails red here, deterministically,
/// regardless of how the connect race schedules.
#[tokio::test(flavor = "multi_thread")]
async fn live_tail_delivers_events_before_run_completes() {
    let base = test_base("live-tail");
    let start_gate = Gate::default();
    let finish_gate = Gate::default();
    // Failure guards: any panic below releases the parked trainer thread.
    let _start_guard = OpenOnDrop(start_gate.clone());
    let _finish_guard = OpenOnDrop(finish_gate.clone());
    let factory_start = start_gate.clone();
    let factory_finish = finish_gate.clone();
    let factory: TrainerFactory = Arc::new(move |_| {
        Box::new(GatedTrainer {
            start_gate: factory_start.clone(),
            finish_gate: factory_finish.clone(),
        })
    });
    let app = app_with(factory, &base);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, 2, 100).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    let response = app.oneshot(get_events(1)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let mut buffer = String::new();

    // Prove the subscriber is parked on an EMPTY history: the trainer has
    // emitted nothing yet (start gate closed — keep-alive comments are 15 s
    // apart), so polling the body must stay pending until the timeout.
    assert!(
        timeout(Duration::from_millis(100), body.frame())
            .await
            .is_err(),
        "no frame should arrive before the trainer is released"
    );

    // Release `Started` only — the run is still parked far from done, so the
    // frame below is necessarily a live-tail delivery.
    start_gate.open();
    while parse_complete_sse(&buffer).is_empty() {
        let frame = timeout(DRAIN_TIMEOUT, body.frame())
            .await
            .expect("started frame timed out — live tail is broken")
            .expect("stream ended before any event")
            .expect("body read failed");
        if let Some(data) = frame.data_ref() {
            buffer.push_str(std::str::from_utf8(data).unwrap());
        }
    }
    let early = parse_complete_sse(&buffer);
    assert_eq!(early[0].event.as_deref(), Some("started"));

    // Release the trainer; the rest of the run must flow to the same stream.
    finish_gate.open();
    loop {
        let frame = timeout(DRAIN_TIMEOUT, body.frame())
            .await
            .expect("tail drain timed out — stream failed to close");
        match frame {
            Some(result) => {
                let frame = result.expect("body read failed");
                if let Some(data) = frame.data_ref() {
                    buffer.push_str(std::str::from_utf8(data).unwrap());
                }
            }
            None => break,
        }
    }

    let events = parse_sse(&buffer);
    assert_eq!(types(&events), vec!["started", "step", "step", "finished"]);

    let _ = std::fs::remove_dir_all(&base);
}

/// Test 6: a failing trainer yields exactly [started, failed], then closes.
#[tokio::test(flavor = "multi_thread")]
async fn failing_trainer_emits_failed_event_then_closes() {
    let base = test_base("failing");
    let factory: TrainerFactory = Arc::new(|_| Box::new(FailingTrainer));
    let app = app_with(factory, &base);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, 1, 1).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    let response = app.oneshot(get_events(1)).await.unwrap();
    let events = parse_sse(&body_string(response.into_body()).await);
    assert_eq!(types(&events), vec!["started", "failed"]);
    let failed: serde_json::Value = serde_json::from_str(&events[1].data).unwrap();
    assert!(
        failed["error"].as_str().unwrap().contains("boom"),
        "failed event should carry the error chain, got: {}",
        events[1].data
    );

    let _ = std::fs::remove_dir_all(&base);
}

/// Test 7: a panicking trainer is contained — the stream ends with a
/// `failed` event mentioning the panic, and the app keeps serving.
#[tokio::test(flavor = "multi_thread")]
async fn panicking_trainer_emits_failed_and_app_keeps_serving() {
    let base = test_base("panicking");
    let factory: TrainerFactory = Arc::new(|_| Box::new(PanickingTrainer));
    let app = app_with(factory, &base);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, 1, 1).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    let response = app.clone().oneshot(get_events(1)).await.unwrap();
    let events = parse_sse(&body_string(response.into_body()).await);
    let last = events.last().expect("terminal event");
    assert_eq!(last.event.as_deref(), Some("failed"));
    let failed: serde_json::Value = serde_json::from_str(&last.data).unwrap();
    assert!(
        failed["error"].as_str().unwrap().contains("panicked"),
        "failed event should mention the panic, got: {}",
        last.data
    );

    // Panic containment: a follow-up POST on the same app still works.
    let response = app
        .oneshot(post_runs(config_json(RUN_DIR, 1, 1).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    let body = body_string(response.into_body()).await;
    assert!(body.contains(r#""id":2"#));

    let _ = std::fs::remove_dir_all(&base);
}

/// Test 8: a client disconnecting mid-stream does not kill the run.
#[tokio::test(flavor = "multi_thread")]
async fn client_disconnect_does_not_kill_run() {
    let base = test_base("disconnect");
    let start_gate = Gate::default();
    let finish_gate = Gate::default();
    // Failure guards: any panic below releases the parked trainer thread.
    let _start_guard = OpenOnDrop(start_gate.clone());
    let _finish_guard = OpenOnDrop(finish_gate.clone());
    let factory_start = start_gate.clone();
    let factory_finish = finish_gate.clone();
    let factory: TrainerFactory = Arc::new(move |_| {
        Box::new(GatedTrainer {
            start_gate: factory_start.clone(),
            finish_gate: factory_finish.clone(),
        })
    });
    let app = app_with(factory, &base);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, 2, 100).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    // Let `Started` flow; the trainer then parks mid-run on the finish gate.
    start_gate.open();

    // Connect, read one frame, then drop the body (client disconnect).
    let response = app.clone().oneshot(get_events(1)).await.unwrap();
    let mut body = response.into_body();
    let _ = timeout(DRAIN_TIMEOUT, body.frame())
        .await
        .expect("first frame timed out")
        .expect("stream ended early")
        .expect("body read failed");
    drop(body);

    // The run still trains to completion.
    finish_gate.open();
    let response = app.oneshot(get_events(1)).await.unwrap();
    let events = parse_sse(&body_string(response.into_body()).await);
    assert_eq!(types(&events), vec!["started", "step", "step", "finished"]);

    let _ = std::fs::remove_dir_all(&base);
}

/// Test 9: concurrent runs are independent — distinct ids, own sequences.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_runs_are_independent() {
    let base = test_base("concurrent");

    let app = mock_app(&base);

    let response = app
        .clone()
        .oneshot(post_runs(config_json("a", 3, 100).to_string()))
        .await
        .unwrap();
    let body = body_string(response.into_body()).await;
    assert!(body.contains(r#""id":1"#));
    let response = app
        .clone()
        .oneshot(post_runs(config_json("b", 5, 100).to_string()))
        .await
        .unwrap();
    let body = body_string(response.into_body()).await;
    assert!(body.contains(r#""id":2"#));

    let events_a = parse_sse(
        &body_string(
            app.clone()
                .oneshot(get_events(1))
                .await
                .unwrap()
                .into_body(),
        )
        .await,
    );
    let events_b =
        parse_sse(&body_string(app.oneshot(get_events(2)).await.unwrap().into_body()).await);

    let started_a: serde_json::Value = serde_json::from_str(&events_a[0].data).unwrap();
    assert_eq!(started_a["total_steps"], 3);
    assert_eq!(
        types(&events_a).last().map(String::as_str),
        Some("finished")
    );

    let started_b: serde_json::Value = serde_json::from_str(&events_b[0].data).unwrap();
    assert_eq!(started_b["total_steps"], 5);
    assert_eq!(
        types(&events_b).last().map(String::as_str),
        Some("finished")
    );

    let _ = std::fs::remove_dir_all(&base);
}

/// Test 10: a trainer that returns Ok without emitting Finished still gets
/// its stream closed (no terminal event) instead of hanging subscribers.
#[tokio::test(flavor = "multi_thread")]
async fn trainer_ok_without_finished_still_closes_stream() {
    let base = test_base("silent");
    let factory: TrainerFactory = Arc::new(|_| Box::new(SilentTrainer));
    let app = app_with(factory, &base);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, 1, 1).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    // The drain completing at all (inside the timeout) is the assertion that
    // matters; the stream must close with zero events, no terminal.
    let response = app.oneshot(get_events(1)).await.unwrap();
    let events = parse_sse(&body_string(response.into_body()).await);
    assert!(events.is_empty(), "expected no events, got: {events:?}");

    let _ = std::fs::remove_dir_all(&base);
}

/// Test 11: unknown run id → 404 with a JSON error body.
#[tokio::test(flavor = "multi_thread")]
async fn events_for_unknown_run_is_404() {
    let base = test_base("unknown-run");
    let response = mock_app(&base).oneshot(get_events(999)).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_string(response.into_body()).await;
    assert_eq!(body, r#"{"error":"unknown run id"}"#);

    let _ = std::fs::remove_dir_all(&base);
}

/// Test 12: an invalid config is 422 and creates no run.
#[tokio::test(flavor = "multi_thread")]
async fn post_invalid_config_is_422_and_creates_no_run() {
    let base = test_base("invalid-config");
    let app = mock_app(&base);

    let response = app
        .clone()
        .oneshot(post_runs("{}".to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);

    let response = app.oneshot(get_events(1)).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let _ = std::fs::remove_dir_all(&base);
}

/// Test 13 (burst contract — regression guard for #45): a large, tight burst
/// of events followed by a terminal must be delivered to the subscriber with
/// ZERO loss, in order, and the stream must close.
///
/// This targets the "missed-wakeup under load" hypothesis directly. The write
/// path (`push_event`) pushes each event then calls `watch::Sender::send_replace(())`;
/// because `watch` is edge-triggered on a monotonic version counter (not the
/// `()` value) and its `changed()` registers-then-checks, multiple sends between
/// two subscriber wakes collapse into one notification WITHOUT loss — the
/// subscriber re-snapshots the full `events[cursor..]` tail on every wake. If a
/// future refactor made the read path lossy (e.g. one event per wake, or a
/// value-based wake), this test fails: the delivered count would drop below
/// `N + 2` or the ids would skip. A hang would trip the 5 s `body_string`
/// timeout (fail slow-red, never wedge CI).
#[tokio::test(flavor = "multi_thread")]
async fn burst_of_events_is_delivered_without_loss_and_closes() {
    const N: u64 = 512;
    let base = test_base("burst");
    let factory: TrainerFactory = Arc::new(|_| Box::new(BurstTrainer { steps: N }));
    let app = app_with(factory, &base);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, N, 10_000).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    let response = app.oneshot(get_events(1)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let events = parse_sse(&body_string(response.into_body()).await);

    // Zero loss: exactly Started + N Steps + Finished, in that shape.
    assert_eq!(
        events.len() as u64,
        N + 2,
        "burst must deliver every event: expected {} got {}",
        N + 2,
        events.len()
    );
    let kinds = types(&events);
    assert_eq!(kinds.first().map(String::as_str), Some("started"));
    assert_eq!(kinds.last().map(String::as_str), Some("finished"));
    assert_eq!(
        kinds[1..kinds.len() - 1]
            .iter()
            .filter(|k| *k == "step")
            .count() as u64,
        N,
        "every one of the {N} Step events must be present"
    );

    // In order and gap-free: SSE ids are 0..N+1 consecutive, and the Step
    // numbers are 1..=N in order (a lossy read path would skip).
    for (i, event) in events.iter().enumerate() {
        assert_eq!(event.id.as_deref(), Some(i.to_string().as_str()));
    }
    let step_nums: Vec<u64> = events
        .iter()
        .filter(|e| e.event.as_deref() == Some("step"))
        .map(|e| {
            serde_json::from_str::<serde_json::Value>(&e.data).unwrap()["step"]
                .as_u64()
                .unwrap()
        })
        .collect();
    assert_eq!(step_nums, (1..=N).collect::<Vec<_>>());

    let _ = std::fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// Run retention / eviction (#36)
// ---------------------------------------------------------------------------

/// Starts a mock run and returns once the POST is accepted (`201`). The run
/// then trains to completion on its own on the blocking pool; its *terminal*
/// state is observed separately, because draining a run's stream is unreliable
/// at low retention — a completed run can self-evict before any subscriber
/// attaches, so the drain would race a legitimate `404`.
async fn start_run(app: &Router) {
    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, 1, 100).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
}

/// Starts a mock run and drains its stream to the terminal `finished` event.
/// Sound only where the run is guaranteed to stay in the registry across the
/// POST → GET window (retention high enough that its own completion does not
/// displace it) — otherwise use [`start_run`] + [`wait_for_events_status`].
async fn run_to_finished(app: &Router, id: u64) {
    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, 1, 100).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    let response = app.clone().oneshot(get_events(id)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let events = parse_sse(&body_string(response.into_body()).await);
    assert_eq!(
        types(&events).last().map(String::as_str),
        Some("finished"),
        "run {id} should have completed"
    );
}

/// Polls `GET /runs/{id}/events` until its status is `want`, bounded by
/// `DRAIN_TIMEOUT`.
///
/// Eviction is genuinely asynchronous: it runs on the runtime a beat *after* a
/// run's stream closes (the supervisor makes `done` observable just before
/// `complete_run` finishes removing the run from the registry), so "drain to
/// finished, then assert evicted" races the eviction. Both target states are
/// monotonic — a completed-then-evicted run stays `404`; a retained run stays
/// `200` until something newer displaces it — so polling to the expected
/// status is race-free and fails slow-red on a stuck run rather than hanging.
async fn wait_for_events_status(app: &Router, id: u64, want: StatusCode) {
    let deadline = std::time::Instant::now() + DRAIN_TIMEOUT;
    loop {
        let got = events_status(app, id).await;
        if got == want {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "run {id}: expected {want} within {DRAIN_TIMEOUT:?}, still {got}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Status of `GET /runs/{id}/events` without draining the body (so an
/// in-flight run can be probed without blocking on its stream).
async fn events_status(app: &Router, id: u64) -> StatusCode {
    let response = app.clone().oneshot(get_events(id)).await.unwrap();
    response.status()
}

/// #36: the runs map is bounded — completed runs beyond `run_retention` are
/// evicted oldest-first, and an evicted run's events are a 404.
///
/// Reverting the eviction makes run 1 a live 200 here, so this fails red.
#[tokio::test(flavor = "multi_thread")]
async fn completed_runs_are_evicted_beyond_retention() {
    let base = test_base("evict");
    let factory: TrainerFactory = Arc::new(|_| Box::new(MockTrainer));
    let app = app_retaining(factory, &base, 2);

    // Retention 2: a run does not self-evict on its own completion, so it stays
    // in the registry across the POST → GET window — draining to `finished` is
    // sound here.
    run_to_finished(&app, 1).await;
    run_to_finished(&app, 2).await;

    // Two completed runs, retention 2: nothing evicted yet.
    assert_eq!(events_status(&app, 1).await, StatusCode::OK);
    assert_eq!(events_status(&app, 2).await, StatusCode::OK);

    // The third completion pushes the queue over the cap: run 1 (the
    // oldest-completed) is evicted, the two newest survive. Eviction runs a
    // beat after run 3's stream closes, so poll for it rather than racing it.
    run_to_finished(&app, 3).await;
    wait_for_events_status(&app, 1, StatusCode::NOT_FOUND).await;
    assert_eq!(events_status(&app, 2).await, StatusCode::OK);
    assert_eq!(events_status(&app, 3).await, StatusCode::OK);

    let _ = std::fs::remove_dir_all(&base);
}

/// #36: an **in-flight** run is never evicted, however tight the retention.
///
/// With `run_retention = 0` every completed run is dropped the instant it
/// finishes — yet run 1, parked mid-training, must still be streamable. A
/// naive "evict the oldest run when the map grows" policy (ignoring `done`)
/// would kill the live run and fail this test red.
#[tokio::test(flavor = "multi_thread")]
async fn in_flight_runs_are_never_evicted() {
    let base = test_base("evict-in-flight");
    let start_gate = Gate::default();
    let finish_gate = Gate::default();
    // Failure guards: any panic below releases the parked trainer thread.
    let _start_guard = OpenOnDrop(start_gate.clone());
    let _finish_guard = OpenOnDrop(finish_gate.clone());

    // The first run parks mid-training; every later run is a fast mock.
    let spawned = AtomicU64::new(0);
    let factory_start = start_gate.clone();
    let factory_finish = finish_gate.clone();
    let factory: TrainerFactory = Arc::new(move |_| {
        if spawned.fetch_add(1, Ordering::Relaxed) == 0 {
            Box::new(GatedTrainer {
                start_gate: factory_start.clone(),
                finish_gate: factory_finish.clone(),
            })
        } else {
            Box::new(MockTrainer)
        }
    });
    let app = app_retaining(factory, &base, 0);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, 2, 100).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    start_gate.open(); // let it emit `started`, then park on the finish gate

    // Two mock runs complete and, at retention 0, self-evict the instant they
    // finish — a completed run is dropped before any subscriber can attach, so
    // observe the eviction by polling the run's endpoint to 404 rather than
    // draining its (racy) stream.
    start_run(&app).await;
    start_run(&app).await;
    wait_for_events_status(&app, 2, StatusCode::NOT_FOUND).await;
    wait_for_events_status(&app, 3, StatusCode::NOT_FOUND).await;

    // ...while the in-flight run is untouched: parked mid-training, never
    // `done`, so never a candidate — however tight the retention.
    assert_eq!(
        events_status(&app, 1).await,
        StatusCode::OK,
        "an unfinished run must survive eviction of completed runs"
    );

    // Subscribe to run 1 *before* releasing it, so the drain cannot lose the
    // same pre-attach race the mocks above hit: attached while it is still
    // parked, the subscriber holds its own `Arc` and drains to `finished` even
    // as the run self-evicts on completion.
    let response = app.clone().oneshot(get_events(1)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body();
    finish_gate.open();
    let events = parse_sse(&body_string(body).await);
    assert_eq!(types(&events), vec!["started", "step", "step", "finished"]);

    // Having finished, run 1 is now a candidate and evicts (retention 0).
    wait_for_events_status(&app, 1, StatusCode::NOT_FOUND).await;

    let _ = std::fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// Output-path confinement and the concurrency cap (#37)
// ---------------------------------------------------------------------------

/// Posts a config and returns `(status, body)`.
async fn post_config(app: &Router, config: serde_json::Value) -> (StatusCode, String) {
    let response = app
        .clone()
        .oneshot(post_runs(config.to_string()))
        .await
        .unwrap();
    let status = response.status();
    (status, body_string(response.into_body()).await)
}

/// #37: a traversal, an absolute path, or an escaping `output.name` is a `400`
/// — and, crucially, **nothing is written outside the base**.
///
/// Before the fix each of these reached `create_dir_all` verbatim, so a client
/// could materialize directories (and later `.safetensors`/`.json` files)
/// anywhere the process could reach. The filesystem assertion is what pins the
/// actual vulnerability; the status code alone would still pass if the path
/// were rejected *after* the directory had been created.
#[tokio::test(flavor = "multi_thread")]
async fn output_paths_escaping_the_base_are_rejected() {
    let base = test_base("confine");
    let app = mock_app(&base);
    let outside = base.parent().expect("temp dir").join("loractl-escaped");
    let _ = std::fs::remove_dir_all(&outside);

    // Traversal out of the base, absolute path, and an escaping name — the
    // three shapes issue #37 calls out.
    let rejected = [
        config_json_output("../loractl-escaped", "lora"),
        config_json_output("../../..", "lora"),
        config_json_output("/tmp/loractl-escaped-abs", "lora"),
        config_json_output(".", "../loractl-escaped/evil"),
        config_json_output("out", "/tmp/loractl-escaped-abs"),
    ];
    for config in rejected {
        let (status, body) = post_config(&app, config.clone()).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "must reject {config}, got body: {body}"
        );
        let error: serde_json::Value = serde_json::from_str(&body).expect("JSON error body");
        assert!(
            error["error"].is_string(),
            "400 must carry a clear error message, got: {body}"
        );
    }

    // No run was ever registered for a rejected request (no id burned, no
    // stream to attach to), and nothing landed on disk outside the base.
    let response = app.clone().oneshot(get_events(1)).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert!(
        !outside.exists(),
        "a rejected request must not create {} — that is the vulnerability",
        outside.display()
    );
    assert!(!std::path::Path::new("/tmp/loractl-escaped-abs").exists());

    // Positive control: the same shape of request, confined, is accepted and
    // resolves under the base.
    let (status, body) = post_config(&app, config_json_output("nested/deep", "lora")).await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "confined path must be accepted"
    );
    assert!(
        body.contains(r#""id":1"#),
        "the first id is not burned by rejections"
    );

    let response = app.oneshot(get_events(1)).await.unwrap();
    let events = parse_sse(&body_string(response.into_body()).await);
    let finished: serde_json::Value =
        serde_json::from_str(&events.last().expect("terminal event").data).unwrap();
    let adapter = std::path::Path::new(finished["adapter_path"].as_str().expect("adapter_path"));
    assert!(
        adapter.starts_with(&base),
        "the run must write under the base, got {}",
        adapter.display()
    );

    let _ = std::fs::remove_dir_all(&base);
    let _ = std::fs::remove_dir_all(&outside);
}

/// #37: `POST /runs` is unauthenticated and each run occupies a real compute
/// thread, so the number of *simultaneous* runs is capped — past it the server
/// says `429` instead of accepting unbounded work.
///
/// Removing the cap makes the second POST a 201 here, so this fails red.
#[tokio::test(flavor = "multi_thread")]
async fn saturated_concurrency_cap_returns_429() {
    let base = test_base("cap");
    let start_gate = Gate::default();
    let finish_gate = Gate::default();
    // Failure guards: any panic below releases the parked trainer thread.
    let _start_guard = OpenOnDrop(start_gate.clone());
    let _finish_guard = OpenOnDrop(finish_gate.clone());

    // Run 1 parks mid-training and holds the only slot; later runs are mocks.
    let spawned = AtomicU64::new(0);
    let factory_start = start_gate.clone();
    let factory_finish = finish_gate.clone();
    let factory: TrainerFactory = Arc::new(move |_| {
        if spawned.fetch_add(1, Ordering::Relaxed) == 0 {
            Box::new(GatedTrainer {
                start_gate: factory_start.clone(),
                finish_gate: factory_finish.clone(),
            })
        } else {
            Box::new(MockTrainer)
        }
    });
    let app = app_limited(factory, &base, 1);

    let (status, _) = post_config(&app, config_json(RUN_DIR, 2, 100)).await;
    assert_eq!(status, StatusCode::CREATED);
    start_gate.open(); // run 1 is now genuinely in flight, holding the slot

    let (status, body) = post_config(&app, config_json(RUN_DIR, 2, 100)).await;
    assert_eq!(
        status,
        StatusCode::TOO_MANY_REQUESTS,
        "a saturated server must refuse, not queue"
    );
    let error: serde_json::Value = serde_json::from_str(&body).expect("JSON error body");
    assert!(
        error["error"]
            .as_str()
            .expect("error message")
            .contains("concurrent"),
        "429 must explain itself, got: {body}"
    );

    // The refused request registered nothing: no id was burned.
    assert_eq!(events_status(&app, 2).await, StatusCode::NOT_FOUND);

    // Finishing run 1 frees the slot (the supervisor retires it before the
    // terminal wake, so a drained stream proves the slot is back).
    finish_gate.open();
    let response = app.clone().oneshot(get_events(1)).await.unwrap();
    let events = parse_sse(&body_string(response.into_body()).await);
    assert_eq!(types(&events), vec!["started", "step", "step", "finished"]);

    let (status, body) = post_config(&app, config_json(RUN_DIR, 1, 100)).await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "a completed run must release its slot, got: {body}"
    );
    // The refused request burned NO id: this run is 2, not 3. `register_run`
    // must consume `next_id` only *after* passing the cap guard — a mutant
    // that hoists `next_id.fetch_add` above the guard would make this run 3
    // and fail here. (The 404 above only proves no Run was inserted, which
    // holds either way.)
    assert!(
        body.contains(r#""id":2"#),
        "the 429-refused request must not consume an id — expected id 2, got: {body}"
    );

    let _ = std::fs::remove_dir_all(&base);
}

// ---------------------------------------------------------------------------
// Bearer-token auth (#62)
// ---------------------------------------------------------------------------

/// The configured token in every auth test. Deliberately contains a space's
/// worth of neighbors (dashes) but no whitespace: the header grammar is
/// `Bearer <token>` split on the first space.
const TEST_TOKEN: &str = "s3cret-loractl-token";

/// The exact 401 body — a golden, like the 404's. One message for missing,
/// malformed, and wrong credentials: the response must not reveal how close
/// the caller got.
const UNAUTHORIZED_BODY: &str = r#"{"error":"missing or invalid bearer token"}"#;

/// An app whose every endpoint requires `Authorization: Bearer TEST_TOKEN`.
fn app_with_auth(base: &std::path::Path) -> Router {
    let factory: TrainerFactory = Arc::new(|_| Box::new(MockTrainer));
    loractl_api::app(
        factory,
        ApiConfig {
            api_token: Some(TEST_TOKEN.to_string()),
            ..config_for(base)
        },
    )
    .expect("app builds")
}

/// `POST /runs` carrying an arbitrary `Authorization` header value.
fn post_runs_authed(body: String, authorization: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/runs")
        .header("content-type", "application/json")
        .header("authorization", authorization)
        .body(Body::from(body))
        .expect("valid request")
}

/// `GET /runs/{id}/events` carrying an arbitrary `Authorization` header value.
fn get_events_authed(id: u64, authorization: &str) -> Request<Body> {
    Request::builder()
        .uri(format!("/runs/{id}/events"))
        .header("authorization", authorization)
        .body(Body::empty())
        .expect("valid request")
}

/// #62 AC: with a token configured, a request with no `Authorization` header
/// is `401` on BOTH endpoints — and the rejected POST registers nothing.
#[tokio::test(flavor = "multi_thread")]
async fn missing_token_is_401_on_both_endpoints() {
    let base = test_base("auth-missing");
    let app = app_with_auth(&base);

    // POST without credentials: 401, the golden JSON body, and the
    // WWW-Authenticate challenge RFC 6750 requires.
    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, 1, 1).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response.headers()["www-authenticate"].to_str().unwrap(),
        "Bearer"
    );
    assert_eq!(body_string(response.into_body()).await, UNAUTHORIZED_BODY);

    // GET without credentials: same 401 — events carry run config and
    // resolved output paths, so the read side is gated too.
    let response = app.clone().oneshot(get_events(1)).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(body_string(response.into_body()).await, UNAUTHORIZED_BODY);

    // The rejected POST left no run behind: an authenticated probe of id 1
    // is a 404 (never issued), not a 200 — and this also proves the correct
    // token passes the gate.
    let response = app
        .oneshot(get_events_authed(1, &format!("Bearer {TEST_TOKEN}")))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let _ = std::fs::remove_dir_all(&base);
}

/// #62 AC: wrong or malformed credentials are `401` — wrong token, prefix,
/// extension, wrong scheme, scheme-only, bare token. None of them burn an id.
#[tokio::test(flavor = "multi_thread")]
async fn wrong_or_malformed_credentials_are_401() {
    let base = test_base("auth-wrong");
    let app = app_with_auth(&base);

    let truncated = &TEST_TOKEN[..TEST_TOKEN.len() - 1];
    let bad = [
        "Bearer wrong-token".to_string(),
        format!("Bearer {truncated}"),   // prefix of the real token
        format!("Bearer {TEST_TOKEN}x"), // real token extended
        format!("Bearer  {TEST_TOKEN}"), // doubled separator
        format!("bearer{TEST_TOKEN}"),   // no separator at all
        format!("Basic {TEST_TOKEN}"),   // wrong scheme
        "Bearer".to_string(),            // scheme, no token
        "Bearer ".to_string(),           // scheme, empty token
        TEST_TOKEN.to_string(),          // bare token, no scheme
    ];
    for authorization in &bad {
        let response = app
            .clone()
            .oneshot(post_runs_authed(
                config_json(RUN_DIR, 1, 1).to_string(),
                authorization,
            ))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "must reject Authorization: {authorization:?}"
        );
        assert_eq!(body_string(response.into_body()).await, UNAUTHORIZED_BODY);

        let response = app
            .clone()
            .oneshot(get_events_authed(1, authorization))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "events endpoint must reject Authorization: {authorization:?}"
        );
    }

    // No rejected request burned an id: the first authenticated run is id 1.
    let response = app
        .oneshot(post_runs_authed(
            config_json(RUN_DIR, 1, 1).to_string(),
            &format!("Bearer {TEST_TOKEN}"),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    let body = body_string(response.into_body()).await;
    assert_eq!(body, r#"{"id":1,"events_url":"/runs/1/events"}"#);

    let _ = std::fs::remove_dir_all(&base);
}

/// #62 AC: the correct token proceeds end-to-end — the run trains and its
/// stream drains to `finished` through the same gate.
#[tokio::test(flavor = "multi_thread")]
async fn correct_token_runs_end_to_end() {
    let base = test_base("auth-ok");
    let app = app_with_auth(&base);

    let response = app
        .clone()
        .oneshot(post_runs_authed(
            config_json(RUN_DIR, 2, 100).to_string(),
            &format!("Bearer {TEST_TOKEN}"),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    let response = app
        .oneshot(get_events_authed(1, &format!("Bearer {TEST_TOKEN}")))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let events = parse_sse(&body_string(response.into_body()).await);
    assert_eq!(types(&events).first().map(String::as_str), Some("started"));
    assert_eq!(types(&events).last().map(String::as_str), Some("finished"));

    let _ = std::fs::remove_dir_all(&base);
}

/// The auth *scheme* is case-insensitive (RFC 9110 §11.1); the token is not.
#[tokio::test(flavor = "multi_thread")]
async fn bearer_scheme_is_case_insensitive_but_token_is_not() {
    let base = test_base("auth-case");
    let app = app_with_auth(&base);

    for scheme in ["bearer", "BEARER", "BeArEr"] {
        let response = app
            .clone()
            .oneshot(post_runs_authed(
                config_json(RUN_DIR, 1, 1).to_string(),
                &format!("{scheme} {TEST_TOKEN}"),
            ))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::CREATED,
            "scheme {scheme:?} must be accepted"
        );
    }

    let response = app
        .oneshot(post_runs_authed(
            config_json(RUN_DIR, 1, 1).to_string(),
            &format!("Bearer {}", TEST_TOKEN.to_uppercase()),
        ))
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::UNAUTHORIZED,
        "the token itself is case-sensitive"
    );

    let _ = std::fs::remove_dir_all(&base);
}

/// #62 AC: with NO token configured, nothing changes — the endpoints stay
/// open and an `Authorization` header, present or garbage, is ignored.
#[tokio::test(flavor = "multi_thread")]
async fn no_token_configured_leaves_the_api_open() {
    let base = test_base("auth-off");
    let app = mock_app(&base); // ApiConfig::default(): api_token = None

    let response = app
        .clone()
        .oneshot(post_runs(config_json(RUN_DIR, 1, 1).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    // A stray Authorization header on an auth-less server is not an error.
    let response = app
        .oneshot(post_runs_authed(
            config_json(RUN_DIR, 1, 1).to_string(),
            "Bearer whatever",
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    let _ = std::fs::remove_dir_all(&base);
}

/// #62 (the enforced-invariant half): an unauthenticated server may only
/// bind loopback. The check runs against the *actually bound* IP in `main`;
/// this pins the pure decision function it delegates to.
#[test]
fn unauthenticated_server_requires_loopback_bind() {
    use loractl_api::enforce_loopback_or_token;
    use std::net::IpAddr;

    let loopbacks: [IpAddr; 3] = [
        "127.0.0.1".parse().unwrap(),
        "::1".parse().unwrap(),
        // IPv4-mapped loopback: what a dual-stack `::` listener reports for a
        // 127.0.0.1 client-side bind — canonicalized before the check.
        "::ffff:127.0.0.1".parse().unwrap(),
    ];
    for ip in loopbacks {
        assert!(
            enforce_loopback_or_token(ip, false).is_ok(),
            "loopback {ip} must not require a token"
        );
    }

    let public: [IpAddr; 4] = [
        "0.0.0.0".parse().unwrap(),
        "::".parse().unwrap(),
        "192.168.1.10".parse().unwrap(),
        "2001:db8::1".parse().unwrap(),
    ];
    for ip in public {
        let err = enforce_loopback_or_token(ip, false)
            .expect_err("non-loopback without a token must refuse to serve");
        assert!(
            err.to_string().contains("LORACTL_API_TOKEN"),
            "the refusal must name the fix, got: {err}"
        );
        // The same bind with a token configured is fine.
        assert!(enforce_loopback_or_token(ip, true).is_ok());
    }
}
