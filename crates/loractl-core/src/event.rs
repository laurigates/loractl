//! The event stream a [`Trainer`](crate::Trainer) emits as it runs.
//!
//! This enum is the seam between the pipeline and whatever surfaces it. The
//! CLI turns these into a progress bar; a future API would stream them as
//! JSON. Keep the variants presentation-agnostic — they describe *what
//! happened*, never *how to display it*.

use std::path::PathBuf;

/// A single progress signal from a training run.
#[derive(Debug, Clone)]
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
    Warning(String),

    /// Emitted once when the run completes; carries the final adapter path.
    Finished { adapter_path: PathBuf },
}
