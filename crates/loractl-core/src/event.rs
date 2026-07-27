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
    /// `name` is a closed vocabulary ([`PhaseName`]) a consumer may key on.
    ///
    /// Countable phases are throttled to roughly 100 reports each — with one
    /// deliberate exception: an `encode` **cache miss** reports every example,
    /// because a single miss is minutes of encoder work and is precisely what
    /// the operator is waiting on. (Cache *hits* are throttled like everything
    /// else.) So `done` is monotonic but **sparse and irregular**: treat it as
    /// an absolute snapshot, never as +1 per event, and drive a progress bar by
    /// assigning `done`/`total` rather than incrementing. A countable phase
    /// always closes with a terminal `done == total` report.
    ///
    /// Phases report *setup*: they are emitted before the first
    /// [`Step`](TrainEvent::Step), never between steps.
    Phase {
        /// Which phase this reports — a closed set, not a free string.
        name: PhaseName,
        /// Human-readable detail for this update, e.g. `"MMDiT (13.1 GiB)"`.
        detail: String,
        /// Progress within the phase, when it is countable at all.
        ///
        /// Flattened, so the wire carries `done`/`total` as plain sibling
        /// fields of `name`/`detail` (and omits both when `None`) — the pair
        /// is bundled in the *type* only, to make "a `total` with no `done`"
        /// unrepresentable rather than merely undocumented.
        #[serde(flatten)]
        counters: Option<PhaseCounters>,
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

/// Which setup phase a [`TrainEvent::Phase`] reports.
///
/// A closed set rather than a `String`: the vocabulary is part of the wire
/// contract a consumer keys on, so the compiler — not a test that happens to
/// execute the right path — is what keeps a new trainer from inventing
/// `"encodee"`. Serializes as the same snake_case strings it always did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseName {
    /// The one-time dataset encode: VAE latents + caption conditioning into
    /// the cache. Countable. A cache *miss* reports every example (each is
    /// minutes of encoder work); hits are throttled.
    Encode,
    /// Reading the prepared cache back, plus the resulting example/bucket/batch
    /// summary.
    Dataset,
    /// A checkpoint load — the VAE, the text encoder, the MMDiT.
    Load,
    /// Building and streaming the int8/int4 frozen-base skeleton. Countable,
    /// one report per base-linear site.
    Quantize,
    /// Folding an optional training adapter into the frozen base.
    Merge,
    /// Building the LoRA adapter set across the matched sites.
    Inject,
}

impl PhaseName {
    /// The canonical wire token — the same string this serializes to.
    ///
    /// Not a rendering choice (that would belong in a front-end): it is the
    /// identifier itself, exposed so a consumer can label or key on it without
    /// round-tripping through serde.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Encode => "encode",
            Self::Dataset => "dataset",
            Self::Load => "load",
            Self::Quantize => "quantize",
            Self::Merge => "merge",
            Self::Inject => "inject",
        }
    }
}

impl std::fmt::Display for PhaseName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Progress within a countable [`TrainEvent::Phase`].
///
/// Both fields are required: bundling them is the whole point, since two
/// independent `Option`s permit a `total` with no `done` — a state no emitter
/// produces and no consumer knows how to render. Flattened onto the event, so
/// this changes the Rust type without changing the JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PhaseCounters {
    /// Units completed so far — an **absolute snapshot**, monotonic but sparse
    /// and irregular. Never treat it as +1 per event.
    pub done: u64,
    /// Total units in this phase. A countable phase closes at `done == total`.
    pub total: u64,
}

impl PhaseCounters {
    /// Counters for a phase of `total` units with `done` complete.
    pub fn new(done: u64, total: u64) -> Self {
        Self { done, total }
    }
}
