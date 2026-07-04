//! Declarative training configuration.
//!
//! A run is fully described by a [`TrainConfig`], normally deserialized from a
//! YAML file (see `config/examples/`). Front-ends may override individual
//! fields (e.g. a `--lr` flag), but the config — not code paths — is the
//! source of truth for what a run does.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
    /// Write a validation sample every N steps during training (0 = disabled).
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
