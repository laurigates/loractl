//! `loractl-api` — the HTTP/SSE renderer over `loractl-core`.
//!
//! This crate is exactly what core's design rule promises: **another renderer
//! over the same event pipeline**. Where the CLI turns [`TrainEvent`]s into a
//! progress bar, this crate serializes the *same* events as JSON over
//! Server-Sent Events. It contains no training logic and renders nothing
//! itself.
//!
//! Two endpoints only (M5 scope):
//!
//! - `POST /runs` — start a training run from a JSON [`TrainConfig`].
//! - `GET /runs/{id}/events` — SSE stream: full replay from event 0, then
//!   live tail until the run's terminal event (`finished` or `failed`).
//!
//! The wire contract (event shapes, SSE framing, lifecycle rules) is
//! documented for GUI authors in `docs/api/events.md`; the core event shapes
//! are pinned byte-for-byte by `loractl-core/tests/event_json.rs`.
//!
//! [`TrainEvent`]: loractl_core::TrainEvent
//! [`TrainConfig`]: loractl_core::TrainConfig

mod routes;
mod state;

use axum::Router;
use loractl_core::Trainer;
use std::sync::Arc;

/// Builds a fresh trainer for each `POST /runs`.
///
/// The seam that keeps the API testable offline: `main.rs` injects the one
/// real `BurnTrainer` line, tests inject mocks. Note `Trainer` has no `Send`
/// supertrait — the bound compiles because current impls are unit structs; a
/// future `!Send` trainer breaks this seam at compile time.
pub type TrainerFactory = Arc<dyn Fn() -> Box<dyn Trainer + Send> + Send + Sync>;

/// Builds the router with all routes wired to a fresh, empty run registry.
pub fn app(factory: TrainerFactory) -> Router {
    routes::router(Arc::new(state::AppState::new(factory)))
}
