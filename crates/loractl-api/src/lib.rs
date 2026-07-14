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

mod paths;
mod routes;
mod state;

use anyhow::Context;
use axum::Router;
use loractl_core::Trainer;
use std::path::PathBuf;
use std::sync::Arc;

/// Builds a fresh trainer for each `POST /runs`.
///
/// The seam that keeps the API testable offline: `main.rs` injects
/// [`loractl_core::select_trainer`] (which routes on the run's
/// `model.base` — hence the `&TrainConfig` parameter), tests inject mocks
/// that ignore the config. Note `Trainer` has no `Send` supertrait — the
/// bound compiles because current impls are unit structs; a future `!Send`
/// trainer breaks this seam at compile time.
pub type TrainerFactory =
    Arc<dyn Fn(&loractl_core::TrainConfig) -> Box<dyn Trainer + Send> + Send + Sync>;

/// How many *completed* runs the registry retains by default.
const DEFAULT_RUN_RETENTION: usize = 32;

/// How many runs may train at once by default.
const DEFAULT_MAX_CONCURRENT_RUNS: usize = 4;

/// Where client-supplied output paths are confined, by default.
const DEFAULT_OUTPUT_BASE: &str = "./runs";

/// Server-level knobs. `main` sources these from the environment
/// ([`ApiConfig::from_env`]); tests construct them explicitly so no test ever
/// depends on ambient process state.
///
/// This is deliberately *not* [`TrainConfig`] — a `TrainConfig` describes one
/// run and arrives in the (unauthenticated) request body; an `ApiConfig`
/// describes the server, is set by the operator, and is never
/// client-controlled. The security properties of the API rest on that split.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Upper bound on the number of **completed** runs kept in the registry.
    /// The oldest-completed run beyond this cap is evicted (its events are
    /// dropped and `GET /runs/{id}/events` becomes `404`). In-flight runs are
    /// never evicted, whatever this is set to.
    ///
    /// Env: `LORACTL_RUN_RETENTION` (default 32).
    pub run_retention: usize,

    /// The directory every run's output is confined under (#37). A request's
    /// `output.dir` is resolved *relative to this*; absolute paths and `..`
    /// components are rejected with `400`, as are symlinks escaping it.
    ///
    /// Env: `LORACTL_OUTPUT_BASE` (default `./runs`).
    pub output_base: PathBuf,

    /// Upper bound on simultaneously-training runs; `POST /runs` returns `429`
    /// while saturated. Each run occupies a blocking-pool thread doing real
    /// compute, so an unbounded count is a trivial resource-exhaustion vector
    /// on an unauthenticated endpoint.
    ///
    /// Env: `LORACTL_MAX_CONCURRENT_RUNS` (default 4).
    pub max_concurrent_runs: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            run_retention: DEFAULT_RUN_RETENTION,
            output_base: PathBuf::from(DEFAULT_OUTPUT_BASE),
            max_concurrent_runs: DEFAULT_MAX_CONCURRENT_RUNS,
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
            output_base: match std::env::var("LORACTL_OUTPUT_BASE") {
                Ok(raw) => PathBuf::from(raw),
                Err(std::env::VarError::NotPresent) => PathBuf::from(DEFAULT_OUTPUT_BASE),
                Err(e) => {
                    return Err(anyhow::Error::new(e).context("reading LORACTL_OUTPUT_BASE"));
                }
            },
            max_concurrent_runs: env_usize(
                "LORACTL_MAX_CONCURRENT_RUNS",
                DEFAULT_MAX_CONCURRENT_RUNS,
            )?,
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
///
/// Fallible because the output base is created and canonicalized here: a base
/// the server cannot own is a misconfiguration that must fail on boot, not
/// per-request.
pub fn app(factory: TrainerFactory, config: ApiConfig) -> anyhow::Result<Router> {
    Ok(routes::router(Arc::new(state::AppState::new(
        factory, config,
    )?)))
}
