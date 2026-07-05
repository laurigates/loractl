//! `loractl-core` — the training pipeline behind loractl.
//!
//! This crate is deliberately free of CLI and I/O-presentation concerns. It
//! defines the pipeline's contract and the ML building blocks behind it:
//!
//! - [`TrainConfig`] — the run's declarative schema (deserialized from YAML).
//! - [`TrainEvent`] — the stream a trainer emits as it works.
//! - [`Trainer`] — the contract a concrete backend implements.
//! - [`LoraLinear`] — the frozen-base, low-rank adapter a real trainer learns.
//! - [`LoraMlp`] — the tiny LoRA-adapted classifier the real trainer trains.
//! - [`BurnTrainer`] — the real, burn-backed trainer (milestone 2).
//! - [`Gpt2`] — a hand-built GPT-2 that loads real HF safetensors weights,
//!   with forward-pass parity vs. PyTorch (milestone 3).
//! - [`adapter`] — safetensors adapter save/load (milestone 4).
//! - [`sample`] — deterministic sampling from a trained adapter (milestone 4).
//!
//! The design rule that keeps a future GUI honest: **core emits events, the
//! caller renders them.** A trainer never draws a progress bar and never
//! `println!`s. The `loractl` CLI renders events as a terminal progress bar;
//! a future HTTP API would serialize the *same* events as SSE/JSON. Both are
//! just different renderers over one pipeline.

pub mod adapter;
pub mod burn_trainer;
pub mod config;
pub mod event;
pub mod gpt2;
pub mod lora;
pub mod model;
pub mod sample;
pub mod train;

pub use burn_trainer::BurnTrainer;
pub use config::TrainConfig;
pub use event::TrainEvent;
pub use gpt2::{Gpt2, Gpt2Config, Gpt2Trace};
pub use lora::LoraLinear;
pub use model::LoraMlp;
pub use train::{MockTrainer, Trainer};

// Re-exported so `loractl-cli` can name the concrete inference backend/device
// (`sample()` in `cli.rs`) without needing its own direct `burn` dependency —
// keeping the CLI's `Cargo.toml` from having to track burn's version/features
// a second time in lockstep with this crate's.
pub use burn::backend::NdArray;
pub use burn::tensor::Device;
