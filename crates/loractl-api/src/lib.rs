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
//! Both sit behind an optional bearer-token gate (`LORACTL_API_TOKEN`, #62);
//! unset, the API is open and loopback-only binding is enforced.
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

    /// Optional bearer token gating **every** endpoint (#62).
    ///
    /// `None` (the env var unset) leaves the API open — the zero-config
    /// localhost dev loop, unchanged. `Some` requires each request to carry
    /// `Authorization: Bearer <token>`; a missing, malformed, or mismatched
    /// value is `401`. The token compare is constant-time, and the read side
    /// (`GET /runs/{id}/events`) is gated too — events carry run configuration
    /// and resolved output paths.
    ///
    /// Env: `LORACTL_API_TOKEN` (unset by default). Present-but-empty is a
    /// hard startup error, never "auth off": an operator who set the variable
    /// believes auth is in force.
    pub api_token: Option<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            run_retention: DEFAULT_RUN_RETENTION,
            output_base: PathBuf::from(DEFAULT_OUTPUT_BASE),
            max_concurrent_runs: DEFAULT_MAX_CONCURRENT_RUNS,
            api_token: None,
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
            api_token: parse_api_token(std::env::var("LORACTL_API_TOKEN"))?,
        })
    }
}

/// Interprets the raw `LORACTL_API_TOKEN` read: unset → auth off, a value →
/// the token, present-but-empty → a hard error (same "never a silent
/// fallback" stance as [`env_usize`]). Split from the env read so the
/// decision table is unit-testable without touching process state.
fn parse_api_token(raw: Result<String, std::env::VarError>) -> anyhow::Result<Option<String>> {
    match raw {
        Ok(token) if token.is_empty() => anyhow::bail!(
            "LORACTL_API_TOKEN is set but empty — set a real token, or unset it to disable auth"
        ),
        Ok(token) => Ok(Some(token)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(anyhow::Error::new(e).context("reading LORACTL_API_TOKEN")),
    }
}

/// Refuses to run an **unauthenticated** server on a non-loopback address
/// (#62): with no token configured, reachability is the only guard, so
/// loopback-only is enforced rather than assumed.
///
/// `main` calls this with the *actually bound* IP (`listener.local_addr()`),
/// not the configured string — ground truth, immune to hostname-resolution
/// surprises. IPv4-mapped IPv6 addresses are canonicalized first so
/// `::ffff:127.0.0.1` counts as loopback.
pub fn enforce_loopback_or_token(
    bind_ip: std::net::IpAddr,
    token_configured: bool,
) -> anyhow::Result<()> {
    if token_configured || bind_ip.to_canonical().is_loopback() {
        return Ok(());
    }
    anyhow::bail!(
        "refusing to serve without authentication on non-loopback address {bind_ip}: \
         set LORACTL_API_TOKEN, or bind LORACTL_API_ADDR to a loopback address"
    )
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

#[cfg(test)]
mod tests {
    use super::parse_api_token;
    use std::env::VarError;

    #[test]
    fn api_token_unset_means_auth_off() {
        assert_eq!(parse_api_token(Err(VarError::NotPresent)).unwrap(), None);
    }

    #[test]
    fn api_token_value_enables_auth() {
        assert_eq!(
            parse_api_token(Ok("s3cret".into())).unwrap(),
            Some("s3cret".into())
        );
    }

    #[test]
    fn api_token_empty_is_a_hard_error_not_auth_off() {
        let err = parse_api_token(Ok(String::new())).unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "the error must say why it refused, got: {err}"
        );
    }

    #[test]
    fn api_token_non_unicode_is_a_hard_error() {
        let raw = Err(VarError::NotUnicode("\u{fffd}".into()));
        let err = parse_api_token(raw).unwrap_err();
        assert!(err.to_string().contains("LORACTL_API_TOKEN"));
    }
}
