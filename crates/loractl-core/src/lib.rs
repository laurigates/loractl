//! `loractl-core` — the training pipeline behind loractl.
//!
//! This crate is deliberately free of CLI and I/O-presentation concerns. It
//! defines the pipeline's contract and the ML building blocks behind it:
//!
//! - [`TrainConfig`] — the run's declarative schema (deserialized from YAML).
//! - [`TrainEvent`] — the stream a trainer emits as it works.
//! - [`Trainer`] — the contract a concrete backend implements.
//! - [`LoraLinear`] — the frozen-base, low-rank adapter a real trainer learns.
//! - [`LoraAdapters`] — a name-keyed set of low-rank deltas injected across a
//!   base module tree (milestone 6), with a kohya-ss [`export`] path.
//! - [`LoraMlp`] — the tiny LoRA-adapted classifier the real trainer trains.
//! - [`BurnTrainer`] — the real, burn-backed trainer (milestone 2).
//! - [`Gpt2`] — a hand-built GPT-2 that loads real HF safetensors weights,
//!   with forward-pass parity vs. PyTorch (milestone 3).
//! - [`adapter`] — safetensors adapter save/load (milestone 4).
//! - [`sample`] — deterministic sampling from a trained adapter (milestone 4).
//! - [`flow`] — the rectified-flow (flow-matching) objective's math: the
//!   data↔noise interpolation, the v-prediction target, and the logit-normal
//!   + shift timestep sampler (milestone 8).
//! - [`QwenVae`] — the Qwen-Image latent VAE (Krea 2's autoencoder): images
//!   ↔ normalized f8/16-channel latents, with encode/decode parity vs.
//!   diffusers (milestone 9).
//! - [`Qwen3VlEncoder`]/[`Qwen3VlConditioner`] — Krea 2's caption
//!   conditioner: a frozen, text-only Qwen3-VL trunk emitting the 12-layer
//!   hidden-state stack the MMDiT cross-attends to (milestone 10).
//! - [`Mmdit`] — the Krea 2 single-stream MMDiT denoiser itself, with forward
//!   parity vs the official implementation and the M6 LoRA attach across its
//!   trunk projections (milestone 11).
//! - [`dataset`] — the image dataset pipeline: kohya-style folder scanning,
//!   aspect-ratio bucketing, and one-time latent/conditioning caching
//!   (milestone 12).
//! - [`DiffusionTrainer`] — the end-to-end Krea 2 LoRA trainer composing all
//!   of the above, exporting ComfyUI-loadable kohya-ss adapters
//!   (milestone 14).
//!
//! The design rule that keeps a GUI honest: **core emits events, the
//! caller renders them.** A trainer never draws a progress bar and never
//! `println!`s. The `loractl` CLI renders events as a terminal progress bar;
//! the `loractl-api` HTTP server serializes the *same* events as SSE/JSON
//! (milestone 5; wire contract in `docs/api/events.md`). Both are just
//! different renderers over one pipeline.
//!
//! ## Compute backend (M7)
//!
//! The trainer's compute backend is selected at run time from
//! [`TrainConfig::compute`] ([`ComputeConfig`]/[`BackendKind`]). `ndarray` (CPU)
//! is always compiled and is the default, so `cargo test` / CI stay offline;
//! GPU backends (`wgpu`, `cuda`, `tch`) are opt-in cargo features and dispatched
//! at run time inside [`BurnTrainer`], leaving every front-end seam untouched.

// burn's cubecl/wgpu backends generate deeply-nested associated-type chains that
// overflow the default recursion limit of 128 once the `wgpu` feature compiles.
// Inert for the default ndarray build; bump higher if a `--features wgpu` build
// ever reports a recursion-limit overflow.
#![recursion_limit = "256"]
// Core is the pipeline's public contract, and its front-ends read these docs;
// keep every public item documented. `just lint` runs clippy with `-D
// warnings`, so once the existing gaps are filled this warn is effectively a
// hard gate against undocumented additions.
#![warn(missing_docs)]

pub mod adapter;
pub mod adapters;
pub mod burn_trainer;
pub mod config;
pub mod dataset;
pub mod diffusion_trainer;
pub mod event;
pub mod export;
pub mod flow;
pub mod gpt2;
pub mod lora;
pub mod mmdit;
pub mod model;
pub mod qwen3vl;
pub mod qwen_vae;
pub mod sample;
pub mod train;

pub use adapters::{LoraAdapters, LoraSite, build_adapters};
pub use burn_trainer::BurnTrainer;
pub use config::{
    BackendKind, ComputeConfig, FlowConfig, ModelVariant, Precision, TaskKind, TrainConfig,
};
pub use diffusion_trainer::DiffusionTrainer;
pub use event::TrainEvent;
pub use export::{ExportFormat, export_adapters};
pub use gpt2::{Gpt2, Gpt2Config, Gpt2Trace};
pub use lora::{LoraDelta, LoraLinear};
pub use mmdit::{Mmdit, MmditConfig};
pub use model::LoraMlp;
pub use qwen_vae::{QwenVae, QwenVaeConfig};
pub use qwen3vl::{Qwen3VlConditioner, Qwen3VlConfig, Qwen3VlEncoder};
pub use train::{MockTrainer, Trainer, select_trainer};

// Re-exported so `loractl-cli` can name the concrete inference backend/device
// (`sample()` in `cli.rs`) without needing its own direct `burn` dependency —
// keeping the CLI's `Cargo.toml` from having to track burn's version/features
// a second time in lockstep with this crate's.
pub use burn::backend::NdArray;
pub use burn::tensor::Device;
