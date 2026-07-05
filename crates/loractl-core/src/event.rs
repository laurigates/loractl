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
    Started { total_steps: u64 },

    /// Emitted once per optimization step.
    Step { step: u64, loss: f32, lr: f64 },

    /// A checkpoint was written to disk.
    Checkpoint { step: u64, path: PathBuf },

    /// A validation sample was written to disk.
    Sample { step: u64, path: PathBuf },

    /// A non-fatal issue worth surfacing to the operator.
    ///
    /// A struct variant (not a newtype) so the wire shape is a flat object
    /// like every other variant — serde cannot internally tag a newtype
    /// `String` variant.
    Warning { message: String },

    /// Emitted once when the run completes; carries the final adapter path.
    Finished { adapter_path: PathBuf },
}
