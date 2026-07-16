//! The two HTTP endpoints (M5 scope): `POST /runs` and `GET /runs/{id}/events`.

use crate::state::{self, AppState};
use axum::Router;
use axum::extract::{Path, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use futures_core::Stream;
use loractl_core::TrainConfig;
use serde::Serialize;
use std::convert::Infallible;
use std::sync::Arc;
use subtle::ConstantTimeEq;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/runs", post(create_run))
        .route("/runs/{id}/events", get(run_events))
        // The auth gate wraps EVERY route (#62): a future endpoint added above
        // is authenticated by default, never open by omission.
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            require_bearer_token,
        ))
        .with_state(state)
}

/// Auth gate (#62): a no-op when no token is configured; otherwise every
/// request must present `Authorization: Bearer <token>` before it reaches a
/// handler — so a rejected request does no work (no body deserialization, no
/// registry lookup) and leaves no trace (no run, no id burned).
async fn require_bearer_token(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let Some(expected) = state.config.api_token.as_deref() else {
        return next.run(request).await;
    };
    if bearer_token_matches(request.headers().get(header::AUTHORIZATION), expected) {
        return next.run(request).await;
    }
    // One message for missing, malformed, and wrong: the response must not
    // tell a caller how close they got.
    let mut response = error_response(StatusCode::UNAUTHORIZED, "missing or invalid bearer token");
    response
        .headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    response
}

/// True iff `header` is exactly `Bearer <expected>`. The scheme match is
/// case-insensitive (RFC 9110 §11.1); the token bytes are compared in
/// constant time, so a mismatch's timing does not reveal how many leading
/// bytes matched. (`ct_eq` short-circuits only on *length*, which an
/// attacker cannot exploit without the bytes themselves.)
fn bearer_token_matches(header: Option<&HeaderValue>, expected: &str) -> bool {
    let Some(value) = header.and_then(|v| v.to_str().ok()) else {
        return false;
    };
    let Some((scheme, token)) = value.split_once(' ') else {
        return false;
    };
    scheme.eq_ignore_ascii_case("bearer") && bool::from(token.as_bytes().ct_eq(expected.as_bytes()))
}

/// `201 {"id":1,"events_url":"/runs/1/events"}` — a typed struct (not
/// `json!`) so the field order is stable for the golden test.
#[derive(Serialize)]
struct CreatedRun {
    id: u64,
    events_url: String,
}

#[derive(Serialize)]
struct ApiError {
    error: String,
}

fn error_response(status: StatusCode, error: impl Into<String>) -> Response {
    (
        status,
        Json(ApiError {
            error: error.into(),
        }),
    )
        .into_response()
}

/// Starts a run. An invalid body is rejected by the `Json` extractor (422)
/// before this handler runs, so no run is ever registered for it.
///
/// Two admission gates gate it further, both because the endpoint is
/// unauthenticated **by default** (#37 — the optional bearer gate, #62, is
/// [`require_bearer_token`] above):
///
/// 1. **Path confinement** — the request's `output.dir`/`output.name` become
///    real filesystem writes, so they are resolved under the server's output
///    base and rejected (`400`) if they escape it. The config the trainer runs
///    with carries the *resolved* dir, never the client's raw string, so no
///    later code path can be handed the unvalidated value by mistake.
/// 2. **Concurrency cap** — each run occupies a blocking thread doing real
///    compute; past the cap the request is refused (`429`) rather than queued,
///    so a client learns immediately instead of timing out.
///
/// Both reject *before* `register_run`, so a refused request leaves no run
/// behind and burns no id.
async fn create_run(
    State(state): State<Arc<AppState>>,
    Json(mut config): Json<TrainConfig>,
) -> Response {
    match crate::paths::confine_output(&state.output_base, &config.output.dir, &config.output.name)
    {
        Ok(dir) => config.output.dir = dir,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, error),
    }

    let Some((id, run)) = state.register_run() else {
        return error_response(
            StatusCode::TOO_MANY_REQUESTS,
            format!(
                "too many concurrent runs (limit {}); retry when a run finishes",
                state.config.max_concurrent_runs
            ),
        );
    };

    let trainer = (state.factory)(&config);
    state::spawn_run(Arc::clone(&state), id, run, config, trainer);
    (
        StatusCode::CREATED,
        Json(CreatedRun {
            id,
            events_url: format!("/runs/{id}/events"),
        }),
    )
        .into_response()
}

/// SSE stream: full replay from event 0, then live tail. Frames carry
/// `id:` = history index and `event:` = the JSON `type` discriminator;
/// keep-alive comment lines flow during long gaps.
async fn run_events(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    let Some(run) = state.get_run(id) else {
        return error_response(StatusCode::NOT_FOUND, "unknown run id");
    };
    let frames = state::subscribe(run);
    let stream = async_stream::stream! {
        let mut frames = std::pin::pin!(frames);
        while let Some(frame) =
            std::future::poll_fn(|cx| frames.as_mut().poll_next(cx)).await
        {
            yield Ok::<_, Infallible>(
                Event::default()
                    .id(frame.index.to_string())
                    .event(frame.kind)
                    .data(frame.json),
            );
        }
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
