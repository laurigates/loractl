//! Declarative training configuration.
//!
//! A run is fully described by a [`TrainConfig`], normally deserialized from a
//! YAML file (see `config/examples/`). Front-ends may override individual
//! fields (e.g. a `--lr` flag), but the config — not code paths — is the
//! source of truth for what a run does.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;

/// Everything needed to run one training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Number of optimization steps to run.
    #[serde(default = "defaults::steps")]
    pub steps: u64,

    /// RNG seed, for reproducible runs.
    #[serde(default)]
    pub seed: u64,

    /// The base model to adapt.
    pub model: ModelConfig,

    /// LoRA adapter hyperparameters.
    pub lora: LoraConfig,

    /// Where the training data lives.
    pub dataset: DatasetConfig,

    /// Optimizer settings (has sensible defaults).
    #[serde(default)]
    pub optim: OptimConfig,

    /// Output/checkpointing settings (has sensible defaults).
    #[serde(default)]
    pub output: OutputConfig,

    /// Compute backend + device (defaults to the ndarray CPU backend, so an
    /// existing config with no `compute:` block runs exactly as before).
    #[serde(default)]
    pub compute: ComputeConfig,
}

/// The base model being adapted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Hub id or local path to the base model (e.g. a safetensors directory).
    pub base: String,
}

/// LoRA adapter shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition (the `r` in A·B).
    #[serde(default = "defaults::rank")]
    pub rank: u32,

    /// Scaling factor; the effective update is `(alpha / rank) · B·A`.
    #[serde(default = "defaults::alpha")]
    pub alpha: f32,

    /// Dropout applied to the adapter input during training.
    #[serde(default)]
    pub dropout: f32,

    /// Which base-model layers to inject adapters into, as path patterns.
    ///
    /// Each [`TargetSpec`] regex is matched against a base model's injectable
    /// site paths (e.g. `transformer\.h\.\d+\.attn\.c_attn`); a site matching any
    /// pattern gets a delta, sized by that pattern's `rank`/`alpha` override or
    /// the top-level `rank`/`alpha`. Empty by default, so an existing YAML with
    /// no `targets:` deserializes unchanged (the single-target M2–M4 paths do not
    /// consult it).
    #[serde(default)]
    pub targets: Vec<TargetSpec>,
}

/// A LoRA injection target: a module-path pattern plus optional per-target
/// `rank`/`alpha` overrides.
///
/// `pattern` is a regex matched against a base model's injectable site paths.
/// `rank`/`alpha`, when `Some`, override the top-level [`LoraConfig`] values for
/// the sites this pattern matches; when `None` the global values apply.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TargetSpec {
    /// Regex matched against injectable site paths.
    pub pattern: String,
    /// Per-target rank override (falls back to [`LoraConfig::rank`]).
    #[serde(default)]
    pub rank: Option<u32>,
    /// Per-target alpha override (falls back to [`LoraConfig::alpha`]).
    #[serde(default)]
    pub alpha: Option<f32>,
}

/// Where and how to read training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Folder of images alongside same-named `.txt` caption files.
    pub path: PathBuf,

    /// Target resolution for bucketing/resizing.
    #[serde(default = "defaults::resolution")]
    pub resolution: u32,
}

/// Optimizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OptimConfig {
    /// Learning rate.
    pub lr: f64,
    /// Decoupled weight decay (AdamW-style).
    pub weight_decay: f64,
}

impl Default for OptimConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            weight_decay: 0.0,
        }
    }
}

/// Output directory, adapter name, and checkpoint cadence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Directory to write checkpoints, samples, and the final adapter into.
    pub dir: PathBuf,
    /// Base name of the final adapter file (no extension).
    pub name: String,
    /// Write a checkpoint every N steps.
    pub checkpoint_every: u64,
    /// Write a validation sample every N steps (0 = disabled).
    pub sample_every: u64,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("output"),
            name: String::from("lora"),
            checkpoint_every: 250,
            sample_every: 0,
        }
    }
}

/// Which compute backend a run executes on (M7, #18).
///
/// The enum and its `Deserialize` are **always compiled** — never behind a
/// `#[cfg]` — so a config naming a backend that this binary was not built with
/// still parses; the trainer then fails loudly at run time rather than silently
/// falling back to CPU. Only the dispatch arms in
/// [`burn_trainer`](crate::burn_trainer) are feature-gated.
///
/// `ndarray` is the always-available CPU backend and the default, which is what
/// keeps `cargo test` / CI offline and GPU-free.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    /// Portable CPU backend — always available; the offline/CI/default backend.
    #[default]
    Ndarray,
    /// wgpu (Metal on macOS, Vulkan/DX12 elsewhere). Requires `--features wgpu`.
    Wgpu,
    /// CUDA (NVIDIA). Requires `--features cuda`; not runnable on macOS.
    Cuda,
    /// libtorch. Requires `--features tch`; needs a linked libtorch binary.
    Tch,
}

// No `clap::ValueEnum` derive here on purpose: it would pull `clap` into
// `loractl-core`, breaking the "core never imports clap" invariant. The CLI
// parses the flag through this `FromStr` via clap's `value_parser` instead.
impl FromStr for BackendKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "ndarray" => Ok(Self::Ndarray),
            "wgpu" => Ok(Self::Wgpu),
            "cuda" => Ok(Self::Cuda),
            "tch" | "libtorch" => Ok(Self::Tch),
            other => Err(format!("unknown backend {other:?} (ndarray|wgpu|cuda|tch)")),
        }
    }
}

// Deserialize through `FromStr` (not the derive) so the YAML and env layers
// accept exactly what the `--backend` flag does — case-insensitive, plus the
// `libtorch` alias — and report the same clear error. This keeps all three
// config layers (YAML → env → flag) an interchangeable surface for the one
// value, rather than the derive's case-sensitive, alias-less, opaque-"unknown
// variant"-error matching. `Serialize` stays derived (writes lowercase), so a
// round-trip is stable.
impl<'de> Deserialize<'de> for BackendKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Compute backend + device selection.
///
/// `#[serde(default)]` plus the derived `Default` (`{ backend: Ndarray,
/// device: 0 }`) means every existing YAML/JSON — which carries no `compute:`
/// block — deserializes exactly as before onto the ndarray CPU backend, so
/// `just test` and CI stay offline (acceptance #2 of #18).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ComputeConfig {
    /// The compute backend to run on.
    pub backend: BackendKind,
    /// Device ordinal (GPU index). Ignored by ndarray; on wgpu `0` selects the
    /// default/best GPU (the only verified path on a single-GPU Apple-Silicon
    /// Mac).
    pub device: usize,
}

/// Field defaults that can't be expressed with `Default::default()` alone.
mod defaults {
    pub fn steps() -> u64 {
        1000
    }
    pub fn rank() -> u32 {
        16
    }
    pub fn alpha() -> f32 {
        16.0
    }
    pub fn resolution() -> u32 {
        512
    }
}
