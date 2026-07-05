//! The two HTTP endpoints (M5 scope): `POST /runs` and `GET /runs/{id}/events`.

use crate::state::{self, AppState};
use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use futures_core::Stream;
use loractl_core::TrainConfig;
use serde::Serialize;
use std::convert::Infallible;
use std::sync::Arc;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/runs", post(create_run))
        .route("/runs/{id}/events", get(run_events))
        .with_state(state)
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
    error: &'static str,
}

/// Starts a run. An invalid body is rejected by the `Json` extractor (422)
/// before this handler runs, so no run is ever registered for it.
async fn create_run(
    State(state): State<Arc<AppState>>,
    Json(config): Json<TrainConfig>,
) -> impl IntoResponse {
    let (id, run) = state.register_run();
    let trainer = (state.factory)();
    state::spawn_run(run, config, trainer);
    (
        StatusCode::CREATED,
        Json(CreatedRun {
            id,
            events_url: format!("/runs/{id}/events"),
        }),
    )
}

/// SSE stream: full replay from event 0, then live tail. Frames carry
/// `id:` = history index and `event:` = the JSON `type` discriminator;
/// keep-alive comment lines flow during long gaps.
async fn run_events(State(state): State<Arc<AppState>>, Path(id): Path<u64>) -> Response {
    let Some(run) = state.get_run(id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: "unknown run id",
            }),
        )
            .into_response();
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
