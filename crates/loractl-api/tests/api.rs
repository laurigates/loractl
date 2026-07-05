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
use loractl_api::TrainerFactory;
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

/// A unique per-test output dir under the system temp dir, so the server's
/// `create_dir_all` side effect never lands in the repo.
fn unique_output_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "loractl-api-test-{tag}-{}-{}",
        std::process::id(),
        DIR_SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

fn mock_app() -> Router {
    let factory: TrainerFactory = Arc::new(|| Box::new(MockTrainer));
    loractl_api::app(factory)
}

/// A minimal valid config as JSON (same schema as the YAML file).
fn config_json(dir: &std::path::Path, steps: u64, checkpoint_every: u64) -> serde_json::Value {
    serde_json::json!({
        "steps": steps,
        "model": { "base": "test-base" },
        "lora": {},
        "dataset": { "path": "unused-dataset" },
        "output": { "dir": dir, "checkpoint_every": checkpoint_every }
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
    let dir = unique_output_dir("post-contract");
    let app = mock_app();

    let response = app
        .clone()
        .oneshot(post_runs(config_json(&dir, 1, 1).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    let body = body_string(response.into_body()).await;
    assert_eq!(body, r#"{"id":1,"events_url":"/runs/1/events"}"#);

    let response = app
        .oneshot(post_runs(config_json(&dir, 1, 1).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    let body = body_string(response.into_body()).await;
    assert_eq!(body, r#"{"id":2,"events_url":"/runs/2/events"}"#);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Test 3 (AC1 e2e): the full mock run streams started → steps/checkpoints
/// → finished, with consecutive `id:`s and `event:` matching the JSON type.
#[tokio::test(flavor = "multi_thread")]
async fn sse_streams_started_through_finished_for_mock_run() {
    let dir = unique_output_dir("e2e");
    let app = mock_app();

    let response = app
        .clone()
        .oneshot(post_runs(config_json(&dir, 5, 2).to_string()))
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

    // Last event is finished with the configured adapter path.
    let last = events.last().unwrap();
    assert_eq!(last.event.as_deref(), Some("finished"));
    let finished: serde_json::Value = serde_json::from_str(&last.data).unwrap();
    assert_eq!(
        finished["adapter_path"].as_str().unwrap(),
        dir.join("lora.safetensors").to_str().unwrap()
    );

    // SSE ids are 0..n consecutive; event names match the JSON type.
    for (i, event) in events.iter().enumerate() {
        assert_eq!(event.id.as_deref(), Some(i.to_string().as_str()));
        let value: serde_json::Value = serde_json::from_str(&event.data).unwrap();
        assert_eq!(event.event.as_deref(), value["type"].as_str());
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Test 4: replaying a finished run is byte-for-byte deterministic.
#[tokio::test(flavor = "multi_thread")]
async fn late_subscriber_after_finish_replays_full_history() {
    let dir = unique_output_dir("replay");
    let app = mock_app();

    let response = app
        .clone()
        .oneshot(post_runs(config_json(&dir, 3, 2).to_string()))
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

    let _ = std::fs::remove_dir_all(&dir);
}

/// Test 5 (missed-wake sentinel): the subscriber is first *proven* parked on
/// an empty history (a short poll comes up empty while `start_gate` is still
/// closed), so the `started` frame it then receives can only have arrived
/// via the live tail — not connect-time replay. A mutant that replays at
/// connect and batch-drains only at `done` fails red here, deterministically,
/// regardless of how the connect race schedules.
#[tokio::test(flavor = "multi_thread")]
async fn live_tail_delivers_events_before_run_completes() {
    let dir = unique_output_dir("live-tail");
    let start_gate = Gate::default();
    let finish_gate = Gate::default();
    // Failure guards: any panic below releases the parked trainer thread.
    let _start_guard = OpenOnDrop(start_gate.clone());
    let _finish_guard = OpenOnDrop(finish_gate.clone());
    let factory_start = start_gate.clone();
    let factory_finish = finish_gate.clone();
    let factory: TrainerFactory = Arc::new(move || {
        Box::new(GatedTrainer {
            start_gate: factory_start.clone(),
            finish_gate: factory_finish.clone(),
        })
    });
    let app = loractl_api::app(factory);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(&dir, 2, 100).to_string()))
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

    let _ = std::fs::remove_dir_all(&dir);
}

/// Test 6: a failing trainer yields exactly [started, failed], then closes.
#[tokio::test(flavor = "multi_thread")]
async fn failing_trainer_emits_failed_event_then_closes() {
    let dir = unique_output_dir("failing");
    let factory: TrainerFactory = Arc::new(|| Box::new(FailingTrainer));
    let app = loractl_api::app(factory);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(&dir, 1, 1).to_string()))
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

    let _ = std::fs::remove_dir_all(&dir);
}

/// Test 7: a panicking trainer is contained — the stream ends with a
/// `failed` event mentioning the panic, and the app keeps serving.
#[tokio::test(flavor = "multi_thread")]
async fn panicking_trainer_emits_failed_and_app_keeps_serving() {
    let dir = unique_output_dir("panicking");
    let factory: TrainerFactory = Arc::new(|| Box::new(PanickingTrainer));
    let app = loractl_api::app(factory);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(&dir, 1, 1).to_string()))
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
        .oneshot(post_runs(config_json(&dir, 1, 1).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    let body = body_string(response.into_body()).await;
    assert!(body.contains(r#""id":2"#));

    let _ = std::fs::remove_dir_all(&dir);
}

/// Test 8: a client disconnecting mid-stream does not kill the run.
#[tokio::test(flavor = "multi_thread")]
async fn client_disconnect_does_not_kill_run() {
    let dir = unique_output_dir("disconnect");
    let start_gate = Gate::default();
    let finish_gate = Gate::default();
    // Failure guards: any panic below releases the parked trainer thread.
    let _start_guard = OpenOnDrop(start_gate.clone());
    let _finish_guard = OpenOnDrop(finish_gate.clone());
    let factory_start = start_gate.clone();
    let factory_finish = finish_gate.clone();
    let factory: TrainerFactory = Arc::new(move || {
        Box::new(GatedTrainer {
            start_gate: factory_start.clone(),
            finish_gate: factory_finish.clone(),
        })
    });
    let app = loractl_api::app(factory);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(&dir, 2, 100).to_string()))
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

    let _ = std::fs::remove_dir_all(&dir);
}

/// Test 9: concurrent runs are independent — distinct ids, own sequences.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_runs_are_independent() {
    let dir_a = unique_output_dir("concurrent-a");
    let dir_b = unique_output_dir("concurrent-b");
    let app = mock_app();

    let response = app
        .clone()
        .oneshot(post_runs(config_json(&dir_a, 3, 100).to_string()))
        .await
        .unwrap();
    let body = body_string(response.into_body()).await;
    assert!(body.contains(r#""id":1"#));
    let response = app
        .clone()
        .oneshot(post_runs(config_json(&dir_b, 5, 100).to_string()))
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

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

/// Test 10: a trainer that returns Ok without emitting Finished still gets
/// its stream closed (no terminal event) instead of hanging subscribers.
#[tokio::test(flavor = "multi_thread")]
async fn trainer_ok_without_finished_still_closes_stream() {
    let dir = unique_output_dir("silent");
    let factory: TrainerFactory = Arc::new(|| Box::new(SilentTrainer));
    let app = loractl_api::app(factory);

    let response = app
        .clone()
        .oneshot(post_runs(config_json(&dir, 1, 1).to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    // The drain completing at all (inside the timeout) is the assertion that
    // matters; the stream must close with zero events, no terminal.
    let response = app.oneshot(get_events(1)).await.unwrap();
    let events = parse_sse(&body_string(response.into_body()).await);
    assert!(events.is_empty(), "expected no events, got: {events:?}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Test 11: unknown run id → 404 with a JSON error body.
#[tokio::test(flavor = "multi_thread")]
async fn events_for_unknown_run_is_404() {
    let response = mock_app().oneshot(get_events(999)).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_string(response.into_body()).await;
    assert_eq!(body, r#"{"error":"unknown run id"}"#);
}

/// Test 12: an invalid config is 422 and creates no run.
#[tokio::test(flavor = "multi_thread")]
async fn post_invalid_config_is_422_and_creates_no_run() {
    let app = mock_app();

    let response = app
        .clone()
        .oneshot(post_runs("{}".to_string()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);

    let response = app.oneshot(get_events(1)).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}
