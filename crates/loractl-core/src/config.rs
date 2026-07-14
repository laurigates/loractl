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

    /// Which training objective this run drives (M8, #19). Defaults to
    /// `classification` (the M2 demo), so every pre-M8 config runs unchanged.
    #[serde(default)]
    pub task: TaskKind,

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

    /// Rectified-flow sampler settings — only consulted when
    /// `task: flow-matching` (defaults to the SD3/kohya production values).
    #[serde(default)]
    pub flow: FlowConfig,
}

/// The base model being adapted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// `"synthetic"` (the M2 demo trainer), `"mnist"` (the demo trainer's
    /// opt-in real-MNIST path, `--features mnist`), or a local path to a
    /// Krea-2-Raw-layout checkpoint directory (`raw.safetensors`,
    /// `text_encoder/model.safetensors`, `tokenizer/tokenizer.json`,
    /// `vae/diffusion_pytorch_model.safetensors`) — which routes the run to
    /// the M14 [`DiffusionTrainer`](crate::DiffusionTrainer). The routing
    /// itself is [`select_trainer`](crate::select_trainer).
    pub base: String,
    /// Which architecture the checkpoint directory holds (M14). Explicit
    /// rather than inferred from tensor shapes — a config mistake should be
    /// a clear error, not a creative misinterpretation of a checkpoint.
    #[serde(default)]
    pub variant: ModelVariant,
}

/// The known Krea 2 architecture variants (M14).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelVariant {
    /// The real ~12B `krea/Krea-2-Raw` stack.
    #[default]
    Krea2,
    /// The dimension-matched tiny test bundle
    /// (`reference/krea2_reference.py`).
    TinyKrea2,
}

// FromStr + Deserialize-through-FromStr: the same layer-parity pattern as
// `BackendKind`/`Precision`.
impl FromStr for ModelVariant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "krea2" | "krea-2" => Ok(Self::Krea2),
            "tiny-krea2" | "tinykrea2" => Ok(Self::TinyKrea2),
            other => Err(format!(
                "unknown model variant {other:?} (krea2|tiny-krea2)"
            )),
        }
    }
}

impl<'de> Deserialize<'de> for ModelVariant {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
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

    /// Examples per training batch (per bucket — batches never mix buckets).
    /// M14; the synthetic tasks ignore it.
    #[serde(default = "defaults::batch_size")]
    pub batch_size: u32,
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

/// Numeric precision of the backend's float element type (M13, #24).
///
/// `F16` halves the resident weight memory — the knob that fits the ~12B
/// Krea 2 base (~49 GB in f32, ~24.6 GB in f16) on a 48 GiB host. Only the
/// wgpu backend supports it (Metal/Vulkan f16 kernels); selecting `F16` on
/// any other backend fails loudly, never a silent fallback — the M7 rule.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// Full precision (every backend).
    #[default]
    F32,
    /// Half precision (wgpu only).
    F16,
}

// No `clap::ValueEnum` here for the same reason as `BackendKind`: core never
// imports clap; the CLI routes its `--precision` flag through this `FromStr`.
impl FromStr for Precision {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "full" => Ok(Self::F32),
            "f16" | "fp16" | "half" => Ok(Self::F16),
            other => Err(format!("unknown precision {other:?} (f32|f16)")),
        }
    }
}

// Deserialize through `FromStr` for the same layer-parity reasons as
// `BackendKind` (case-insensitive + aliases, identical errors in YAML, env,
// and flag form).
impl<'de> Deserialize<'de> for Precision {
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
/// `just test` and CI stay offline (acceptance #2 of #18). The M13 memory
/// knobs (`precision`, `grad_checkpointing`) default off the same way.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ComputeConfig {
    /// The compute backend to run on.
    pub backend: BackendKind,
    /// Device ordinal (GPU index). Ignored by ndarray; on wgpu `0` selects the
    /// default/best GPU (the only verified path on a single-GPU Apple-Silicon
    /// Mac).
    pub device: usize,
    /// Float precision (M13). `f16` halves weight memory; wgpu only.
    pub precision: Precision,
    /// Recompute activations during the backward pass instead of storing
    /// them (M13) — burn's `BalancedCheckpointing`. Slower per step,
    /// substantially less activation memory; numerically identical
    /// (recomputation replays the same ops).
    pub grad_checkpointing: bool,
}

/// Which training objective a run drives (M8, #19).
///
/// Like [`BackendKind`], the enum and its `Deserialize` are **always
/// compiled** and route through `FromStr`, so the YAML, env, and `--task` flag
/// layers accept the same spellings (case-insensitive; `flow-matching`,
/// `flow_matching`, `flowmatching`, or the short `flow` alias) and report the
/// same clear error. The derived `Serialize` is kebab-case so `FlowMatching`
/// round-trips as `"flow-matching"` — a plain lowercase rename would emit
/// `"flowmatching"`, which stays accepted by the `FromStr` only as a
/// belt-and-braces spelling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskKind {
    /// The M2 synthetic/MNIST LoRA-MLP classifier demo — the default.
    #[default]
    Classification,
    /// The M8 rectified-flow (flow-matching) v-prediction objective on a
    /// synthetic latent toy. See [`crate::flow`] for the pinned math.
    FlowMatching,
}

// No `clap::ValueEnum` derive here on purpose (same reasoning as
// [`BackendKind`]): the CLI parses `--task` through this `FromStr` via clap's
// `value_parser`, keeping the vocabulary defined once in core.
impl FromStr for TaskKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "classification" => Ok(Self::Classification),
            "flow-matching" | "flow_matching" | "flowmatching" | "flow" => Ok(Self::FlowMatching),
            other => Err(format!(
                "unknown task {other:?} (classification|flow-matching)"
            )),
        }
    }
}

// Deserialize through `FromStr` (not the derive) so the YAML and env layers
// accept exactly what the `--task` flag does — see [`BackendKind`]'s
// `Deserialize` for the full rationale. `Serialize` stays derived (writes
// kebab-case), so a round-trip is stable.
impl<'de> Deserialize<'de> for TaskKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Rectified-flow / flow-matching hyperparameters (M8, #19).
///
/// Controls the logit-normal timestep sampler and the constant shift transform
/// (see [`crate::flow`]). Only consulted when [`TrainConfig::task`] is
/// [`TaskKind::FlowMatching`]. `#[serde(default)]` plus the hand-written
/// `Default` means every existing YAML/JSON — which carries no `flow:` block —
/// deserializes exactly as before, onto the SD3 Eq. 19 sampler defaults and
/// the kohya/SD3-scheduler production shift.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FlowConfig {
    /// Mean of the logit-normal timestep distribution (`u ~ N(mean, std)`,
    /// `t = sigmoid(u)`). SD3 Eq. 19 default: `0.0`.
    pub logit_mean: f64,
    /// Standard deviation of the logit-normal timestep distribution. SD3
    /// Eq. 19 default: `1.0`.
    pub logit_std: f64,
    /// Constant timestep shift `t' = shift·t / (1 + (shift − 1)·t)`;
    /// `shift > 1` pushes `t` toward 1 (noise). kohya/SD3-scheduler
    /// production default: `3.0`.
    pub shift: f64,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            logit_mean: 0.0,
            logit_std: 1.0,
            shift: 3.0,
        }
    }
}

/// Field defaults that can't be expressed with `Default::default()` alone.
mod defaults {
    pub fn steps() -> u64 {
        1000
    }
    pub fn batch_size() -> u32 {
        1
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
