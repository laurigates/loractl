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
#![warn(missing_docs)]

mod routes;
mod state;

use anyhow::Context;
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

/// How many *completed* runs the registry retains by default.
const DEFAULT_RUN_RETENTION: usize = 32;

/// Server-level knobs. `main` sources these from the environment
/// ([`ApiConfig::from_env`]); tests construct them explicitly so no test ever
/// depends on ambient process state.
///
/// This is deliberately *not* [`TrainConfig`] — a `TrainConfig` describes one
/// run and arrives in the request body; an `ApiConfig` describes the server
/// and is never client-controlled.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Upper bound on the number of **completed** runs kept in the registry.
    /// The oldest-completed run beyond this cap is evicted (its events are
    /// dropped and `GET /runs/{id}/events` becomes `404`). In-flight runs are
    /// never evicted, whatever this is set to.
    ///
    /// Env: `LORACTL_RUN_RETENTION` (default 32).
    pub run_retention: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            run_retention: DEFAULT_RUN_RETENTION,
        }
    }
}

impl ApiConfig {
    /// Reads the config from the environment, falling back to the defaults.
    ///
    /// A present-but-unparseable value is a hard error — never a silent
    /// fallback to the default, which would leave the operator believing a
    /// limit is in force that is not.
    pub fn from_env() -> anyhow::Result<Self> {
        Ok(Self {
            run_retention: env_usize("LORACTL_RUN_RETENTION", DEFAULT_RUN_RETENTION)?,
        })
    }
}

/// Reads a `usize` from `key`, or `default` when the var is unset.
fn env_usize(key: &str, default: usize) -> anyhow::Result<usize> {
    match std::env::var(key) {
        Ok(raw) => raw
            .trim()
            .parse()
            .with_context(|| format!("{key} must be a non-negative integer, got {raw:?}")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(anyhow::Error::new(e).context(format!("reading {key}"))),
    }
}

/// Builds the router with all routes wired to a fresh, empty run registry.
pub fn app(factory: TrainerFactory, config: ApiConfig) -> Router {
    routes::router(Arc::new(state::AppState::new(factory, config)))
}
