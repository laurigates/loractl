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
    ///
    /// The four `*_path` overrides below let each component live **outside**
    /// this directory — so a ComfyUI install's scattered layout
    /// (`models/diffusion_models/…`, `models/text_encoders/…`,
    /// `models/vae/…`) works with no restructuring, duplicate files, or
    /// symlinks. When every component is overridden, `base` is used only as
    /// the root that relative overrides join onto and (until each is
    /// overridden) for the default component locations.
    pub base: String,
    /// Which architecture the checkpoint directory holds (M14). Explicit
    /// rather than inferred from tensor shapes — a config mistake should be
    /// a clear error, not a creative misinterpretation of a checkpoint.
    #[serde(default)]
    pub variant: ModelVariant,
    /// Optional denoiser filename **within `base`** overriding the variant
    /// default (`raw.safetensors` / `turbo.safetensors`) — e.g. a local
    /// scaled-fp8 repack like `krea2_turbo_fp8_scaled.safetensors` (M15,
    /// #82). fp8-vs-bf16 handling is auto-detected from the file header,
    /// not from this name. Env override: `LORACTL_MODEL__CHECKPOINT`
    /// (figment's `__` nesting — no CLI flag needed). Superseded by
    /// [`denoiser`](Self::denoiser) when that full-path override is set.
    #[serde(default)]
    pub checkpoint: Option<String>,
    /// Full path to the **denoiser** file, pointing directly at a scattered
    /// ComfyUI file (e.g. `models/diffusion_models/krea2/…fp8_scaled.safetensors`)
    /// instead of `base/<variant default | checkpoint>`. Absolute paths are
    /// used verbatim; relative paths join onto `base`. fp8-vs-bf16 (including
    /// ComfyUI's scaled-fp8 with `comfy_quant` markers) is auto-detected from
    /// the header. Env: `LORACTL_MODEL__DENOISER`.
    #[serde(default)]
    pub denoiser: Option<PathBuf>,
    /// Full path to the **text encoder** file (Qwen3-VL), overriding
    /// `base/text_encoder/model.safetensors`. Absolute verbatim; relative
    /// joins onto `base`. Env: `LORACTL_MODEL__TEXT_ENCODER`.
    #[serde(default)]
    pub text_encoder: Option<PathBuf>,
    /// Full path to the **VAE** file (Qwen-Image), overriding
    /// `base/vae/diffusion_pytorch_model.safetensors`. Absolute verbatim;
    /// relative joins onto `base`. Env: `LORACTL_MODEL__VAE`.
    #[serde(default)]
    pub vae: Option<PathBuf>,
    /// Full path to the **tokenizer** `tokenizer.json`, overriding
    /// `base/tokenizer/tokenizer.json`. Absolute verbatim; relative joins
    /// onto `base`. Env: `LORACTL_MODEL__TOKENIZER`. A ComfyUI install ships
    /// no tokenizer file — when neither this override nor the base-dir file
    /// exists, the model-invariant Qwen3-VL tokenizer is fetched once and
    /// cached (see `hf::fetch_qwen3vl_tokenizer`), so the ComfyUI flow needs
    /// nothing set here. An override that names a **missing** file is an
    /// error, never a silent fetch.
    #[serde(default)]
    pub tokenizer: Option<PathBuf>,
    /// Optional path to a LoRA **training adapter** (`.safetensors`) merged into
    /// the frozen base at load, before LoRA injection — the Krea-2-Turbo
    /// assistant-LoRA seam (#83). For each targeted site the base weight is
    /// updated `W += (alpha/rank)·B·A`, nudging the distilled turbo weights back
    /// toward a raw-like state for training (ai-toolkit's distillation-aware
    /// turbo recipe). Absolute paths are used verbatim; relative paths join onto
    /// `base`. Env: `LORACTL_MODEL__TRAINING_ADAPTER`. Rejected together with
    /// `compute.quant` (the merge needs a full-precision base). The trained
    /// adapter still deploys on **plain** turbo — same interop contract as
    /// ai-toolkit, which inverts this merge before export. See
    /// [`crate::training_adapter`].
    #[serde(default)]
    pub training_adapter: Option<PathBuf>,
}

/// The known Krea 2 architecture variants (M14).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelVariant {
    /// The real ~12B `krea/Krea-2-Raw` stack.
    #[default]
    Krea2,
    /// Krea-2-Turbo (M15, #82) — architecturally identical to
    /// [`Krea2`](Self::Krea2) (same 430 tensor keys, same configs); differs
    /// only in the default denoiser filename (`turbo.safetensors`) and the
    /// typical checkpoint dtype (bf16 official, scaled-fp8 community
    /// repacks — auto-detected from the file header at load).
    Krea2Turbo,
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
            "krea2-turbo" | "krea2turbo" | "turbo" => Ok(Self::Krea2Turbo),
            "tiny-krea2" | "tinykrea2" => Ok(Self::TinyKrea2),
            other => Err(format!(
                "unknown model variant {other:?} (krea2|krea2-turbo|tiny-krea2)"
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
    /// candle-core (Metal kernels on macOS) — an independent kernel stack
    /// from wgpu/cubecl, and the only backend offering **bf16** on Metal.
    /// Requires `--features candle`.
    Candle,
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
            "candle" => Ok(Self::Candle),
            "cuda" => Ok(Self::Cuda),
            "tch" | "libtorch" => Ok(Self::Tch),
            other => Err(format!(
                "unknown backend {other:?} (ndarray|wgpu|candle|cuda|tch)"
            )),
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
    /// bfloat16 — f32's exponent range in 16 bits, the dtype most modern
    /// checkpoints ship in (candle backend only; Metal via candle-core).
    Bf16,
}

// No `clap::ValueEnum` here for the same reason as `BackendKind`: core never
// imports clap; the CLI routes its `--precision` flag through this `FromStr`.
impl FromStr for Precision {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "full" => Ok(Self::F32),
            "f16" | "fp16" | "half" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            other => Err(format!("unknown precision {other:?} (f32|f16|bf16)")),
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

/// Frozen-base quantization of the diffusion trainer's MMDiT (#96 — the #24
/// follow-up).
///
/// Both variants load the ~12.8B Krea 2 base as weight-only, per-block
/// symmetric quantization ([`quant::quant_scheme`](crate::quant::quant_scheme))
/// while the LoRA adapters train in f32 — the QLoRA pattern — so the resident
/// base fits a 24 GB GPU:
///
/// - `Int8` (`Q8S`) cuts the base to ~1/4 of f32 (~14 GB on the real model);
///   a training step's forward+backward working set then peaks near 24 GB.
/// - `Int4` (`Q4S`) cuts the base to ~1/8 of f32 (~8 GB): 4-bit weights + the
///   same f32 per-block scales. It does NOT shrink the transient f32 dequant
///   working set (dequant is identical), but halving the *resident* base frees
///   ~6 GB of headroom — enough to keep a step's peak well under 24 GB.
///
/// Like [`Precision`], the enum and its `Deserialize` are **always compiled**
/// and route through `FromStr`; the diffusion trainer's guard matrix restricts
/// both quant modes to the numerically-validated `(ndarray, f32)` /
/// `(cuda, f32)` combos and fails loudly everywhere else (wgpu is untested,
/// candle/tch have no quant q-ops) — never a silent fallback.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Quant {
    /// No quantization — the frozen base loads in the backend's float dtype
    /// (the default; every existing config keeps its current behavior).
    #[default]
    None,
    /// Weight-only per-block symmetric int8 for the frozen base (`quant.rs`).
    Int8,
    /// Weight-only per-block symmetric int4 (`Q4S`) for the frozen base
    /// (`quant.rs`) — ~1/8 of f32, halving int8's resident base.
    Int4,
}

// No `clap::ValueEnum` here for the same reason as `BackendKind`/`Precision`:
// core never imports clap; the CLI routes its `--quant` flag through this
// `FromStr`.
impl FromStr for Quant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "none" | "off" | "f32" => Ok(Self::None),
            "int8" | "i8" => Ok(Self::Int8),
            "int4" | "i4" => Ok(Self::Int4),
            other => Err(format!("unknown quant {other:?} (none|int8|int4)")),
        }
    }
}

// Deserialize through `FromStr` for the same layer-parity reasons as
// `BackendKind`/`Precision` (case-insensitive + aliases, identical errors in
// YAML, env, and flag form).
impl<'de> Deserialize<'de> for Quant {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Default for [`ComputeConfig::dequant_chunk_mib`]: 512 MiB splits only the
/// largest weight dequant — `tproj.fc` `[36864, 6144]` ≈ 864 MiB f32 on the
/// real krea2 config — while leaving the ~36 MiB trunk tiles (and every tiny
/// fixture) single-chunk. Note: ADR-0005's sweep also recorded a recurring
/// 1,576,693,760-byte failing allocation that matches NO single weight's
/// dequant size; whether the default threshold moves the real step's peak is
/// exactly what the on-box `step-probe` rerun measures (lower thresholds,
/// e.g. 64, split the SwiGLU/attention weights too).
pub const DEFAULT_DEQUANT_CHUNK_MIB: u32 = 512;

/// Compute backend + device selection.
///
/// `#[serde(default)]` plus the hand-written `Default` (`{ backend: Ndarray,
/// device: 0, .. }`) means every existing YAML/JSON — which carries no
/// `compute:` block — deserializes exactly as before onto the ndarray CPU
/// backend, so `just test` and CI stay offline (acceptance #2 of #18). The
/// M13 memory knobs (`precision`, `grad_checkpointing`) default off the same
/// way. (`Default` is hand-written, not derived, because `dequant_chunk_mib`
/// defaults to [`DEFAULT_DEQUANT_CHUNK_MIB`], not `0` — and the struct-level
/// `#[serde(default)]` constructs a missing `compute:` block from
/// `Self::default()`, so a derived all-zeros default would silently disable
/// chunking.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// them. Slower per step, substantially less activation memory;
    /// numerically identical (recomputation replays the same ops). On the
    /// diffusion trainer this is **block-level** checkpointing (#134): the
    /// trunk forward runs graph-free storing only block inputs, and each
    /// block replays on its own small graph in backward — the measured route
    /// to fitting the int4 real-model step in 24 GB (ADR-0005 Addendum 2).
    /// Incompatible with `lora.dropout > 0` (a replay would redraw masks).
    /// On the synthetic `BurnTrainer` it remains burn's
    /// `BalancedCheckpointing` (M13).
    pub grad_checkpointing: bool,
    /// Frozen-base quantization (#96). `int8` loads the diffusion trainer's
    /// MMDiT base as per-block symmetric int8 (~1/4 the f32 weight memory);
    /// `int4` uses symmetric int4 (~1/8 the f32 weight memory, halving int8's
    /// resident base to fit a 24 GB step); `none` (default) keeps the
    /// full-precision base. Restricted to `(ndarray|cuda, f32)` by the trainer
    /// guard — every other combination fails loudly.
    #[serde(default)]
    pub quant: Quant,
    /// Chunked-dequant threshold in MiB (#128) — applies to the quant path
    /// only (`quant: int8|int4`; ignored otherwise). A quantized frozen-base
    /// weight whose f32 size exceeds this threshold is stored (and
    /// dequantized, forward and backward) as row chunks each at or below the
    /// threshold, so no full-weight f32 transient ever materializes. `0`
    /// disables chunking. The default ([`DEFAULT_DEQUANT_CHUNK_MIB`], 512)
    /// splits only the largest (`tproj`-class) weights; smaller values
    /// (e.g. `16`) split the ~36 MiB trunk tiles too. The chunked
    /// QUANTIZATION is bit-identical to whole-tensor quantization (the quant
    /// blocks live within rows — see `quant::dequant_chunk_rows`), and the
    /// forward is bit-identical on ndarray (pinned by tests); on GPU
    /// backends the per-chunk matmul may tile differently, so expect
    /// f32-rounding-level equivalence there. Backward-gradient accumulation
    /// order also differs across chunk ops.
    pub dequant_chunk_mib: u32,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            backend: BackendKind::default(),
            device: 0,
            precision: Precision::default(),
            grad_checkpointing: false,
            quant: Quant::default(),
            dequant_chunk_mib: DEFAULT_DEQUANT_CHUNK_MIB,
        }
    }
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

/// How the flow-matching timestep shift is chosen (#84).
///
/// Like [`BackendKind`]/[`TaskKind`], the enum and its `Deserialize` are
/// always compiled and route through `FromStr`, so the YAML and env layers
/// accept the same spellings (case-insensitive) and report the same clear
/// error.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShiftMode {
    /// The constant [`FlowConfig::shift`] for every batch — the kohya/SD3
    /// `discrete_flow_shift` behavior, and the backward-compatible default.
    #[default]
    Constant,
    /// Resolution-dependent shift: per batch, `shift = exp(μ)` with `μ`
    /// linear in the image-token count through the
    /// ([`FlowConfig::base_image_seq_len`], [`FlowConfig::base_shift`]) and
    /// ([`FlowConfig::max_image_seq_len`], [`FlowConfig::max_shift`]) anchor
    /// points — the FLUX-family dynamic shifting ai-toolkit trains Krea 2
    /// with ([`FlowConfig::shift`] is then unused). Requires image latents,
    /// so only the diffusion trainer accepts it; the synthetic flow toy
    /// bails loudly.
    Resolution,
}

// No `clap::ValueEnum` derive here on purpose (same reasoning as
// [`BackendKind`]): the vocabulary lives once in core. There is no CLI flag
// for this knob yet — the YAML and `LORACTL_FLOW__SHIFT_MODE` env layers are
// the surface.
impl FromStr for ShiftMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "constant" => Ok(Self::Constant),
            "resolution" => Ok(Self::Resolution),
            other => Err(format!(
                "unknown shift mode {other:?} (constant|resolution)"
            )),
        }
    }
}

// Deserialize through `FromStr` (not the derive) so the YAML and env layers
// accept exactly the same spellings — see [`BackendKind`]'s `Deserialize` for
// the full rationale. `Serialize` stays derived (writes kebab-case), so a
// round-trip is stable.
impl<'de> Deserialize<'de> for ShiftMode {
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
/// Controls the logit-normal timestep sampler and the shift transform (see
/// [`crate::flow`]). Only consulted when [`TrainConfig::task`] is
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
    /// production default: `3.0`. This is a **LINEAR** shift (`exp(μ)`), and
    /// it is consulted only under [`ShiftMode::Constant`].
    pub shift: f64,
    /// Whether the shift is the constant [`Self::shift`] or the per-batch
    /// resolution-dependent `exp(μ)` (#84). Default: constant.
    pub shift_mode: ShiftMode,
    /// μ-line anchor: the image-token count at which `μ = base_shift`.
    /// Krea 2 / ai-toolkit `scheduler_config.base_image_seq_len`: `256`.
    pub base_image_seq_len: usize,
    /// μ-line anchor: the image-token count at which `μ = max_shift`.
    /// Krea 2 / ai-toolkit `scheduler_config.max_image_seq_len`: `6400`
    /// (FLUX uses `4096`). The line is extrapolated, never clamped, outside
    /// the anchors — matching diffusers' `calculate_shift`.
    pub max_image_seq_len: usize,
    /// μ at `base_image_seq_len`. **A LOG shift** (the linear shift is
    /// `exp(μ)`), unlike [`Self::shift`] — the field names follow
    /// diffusers/ai-toolkit (`base_shift`/`max_shift` there are μ values
    /// too). Krea 2: `0.5`.
    pub base_shift: f64,
    /// μ at `max_image_seq_len`. A LOG shift, like [`Self::base_shift`].
    /// Krea 2: `1.15`.
    pub max_shift: f64,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            logit_mean: 0.0,
            logit_std: 1.0,
            shift: 3.0,
            shift_mode: ShiftMode::Constant,
            // Krea 2's dynamic-shift anchors, per ai-toolkit's krea2.py
            // scheduler_config (see crate::flow's module docs).
            base_image_seq_len: 256,
            max_image_seq_len: 6400,
            base_shift: 0.5,
            max_shift: 1.15,
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
