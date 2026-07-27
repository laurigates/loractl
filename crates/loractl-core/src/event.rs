//! The event stream a [`Trainer`](crate::Trainer) emits as it runs.
//!
//! This enum is the seam between the pipeline and whatever surfaces it. The
//! CLI turns these into a progress bar; the API serializes them as JSON.
//! Keep the variants presentation-agnostic — they describe *what happened*,
//! never *how to display it*.
//!
//! The JSON wire shapes (internally tagged via `type`, snake_case) are part
//! of core's public contract: they are pinned byte-for-byte by the golden
//! test in `tests/event_json.rs` and documented for consumers in
//! `docs/api/events.md`.

use serde::Serialize;
use std::path::PathBuf;

/// A single progress signal from a training run.
///
/// Serializes as an internally tagged JSON object (`{"type":"step",...}`);
/// the exact shapes are pinned by the `train_event_wire_shapes` golden test
/// and documented in `docs/api/events.md`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TrainEvent {
    /// Emitted once at the start, carrying the planned step count.
    Started {
        /// Total number of optimization steps the run intends to perform.
        total_steps: u64,
    },

    /// Progress through a long setup or background phase that is not an
    /// optimization step — the one-time dataset encode, a multi-gigabyte
    /// checkpoint load, LoRA injection. Emitted zero or more times per phase;
    /// consumers that only track steps may ignore it entirely.
    ///
    /// `name` is drawn from a small **stable vocabulary** — a consumer may key
    /// on these, so they change only deliberately:
    ///
    /// | `name` | The phase it reports |
    /// |---|---|
    /// | `encode` | The one-time dataset encode (VAE latents + caption conditioning into the cache). Countable, one report per example. |
    /// | `dataset` | Reading the prepared cache back, plus the resulting example/bucket/batch summary. |
    /// | `load` | A checkpoint load — the VAE, the text encoder, the MMDiT. |
    /// | `quantize` | Building and streaming the int8/int4 frozen-base skeleton. Countable, one report per base-linear site. |
    /// | `merge` | Folding an optional training adapter into the frozen base. |
    /// | `inject` | Building the LoRA adapter set across the matched sites. |
    ///
    /// Countable phases are throttled to roughly 100 reports each, so a
    /// consumer must treat `done` as monotonic-but-sparse, never as +1 per
    /// event. Phases report *setup*: they are emitted before the first
    /// [`Step`](TrainEvent::Step), never between steps.
    Phase {
        /// Stable machine-readable phase id — one of `"encode"`, `"dataset"`,
        /// `"load"`, `"quantize"`, `"merge"`, `"inject"` (see the table above).
        name: String,
        /// Human-readable detail for this update, e.g. `"MMDiT (13.1 GiB)"`.
        detail: String,
        /// Completed units within this phase, when the phase is countable.
        #[serde(skip_serializing_if = "Option::is_none")]
        done: Option<u64>,
        /// Total units within this phase, when known.
        #[serde(skip_serializing_if = "Option::is_none")]
        total: Option<u64>,
    },

    /// Emitted once per optimization step.
    Step {
        /// 1-based index of the step that just completed.
        step: u64,
        /// Training loss measured on this step.
        loss: f32,
        /// Learning rate applied on this step.
        lr: f64,
    },

    /// A checkpoint was written to disk.
    Checkpoint {
        /// Step at which the checkpoint was taken.
        step: u64,
        /// Path of the `.safetensors` checkpoint just written.
        path: PathBuf,
    },

    /// A validation sample was written to disk.
    Sample {
        /// Step at which the validation sample was taken.
        step: u64,
        /// Path of the sample JSON just written.
        path: PathBuf,
    },

    /// A non-fatal issue worth surfacing to the operator.
    ///
    /// A struct variant (not a newtype) so the wire shape is a flat object
    /// like every other variant — serde cannot internally tag a newtype
    /// `String` variant.
    Warning {
        /// Human-readable description of the non-fatal issue.
        message: String,
    },

    /// Emitted once when the run completes; carries the final adapter path.
    Finished {
        /// Path of the final trained adapter written to disk.
        adapter_path: PathBuf,
    },
}
