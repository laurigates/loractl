//! `loractl-core` — the training pipeline behind loractl.
//!
//! This crate is deliberately free of CLI and I/O-presentation concerns. It
//! defines the pipeline's contract and the ML building blocks behind it:
//!
//! - [`TrainConfig`] — the run's declarative schema (deserialized from YAML).
//! - [`TrainEvent`] — the stream a trainer emits as it works.
//! - [`Trainer`] — the contract a concrete backend implements.
//! - [`LoraLinear`] — the frozen-base, low-rank adapter a real trainer learns.
//!
//! The design rule that keeps a future GUI honest: **core emits events, the
//! caller renders them.** A trainer never draws a progress bar and never
//! `println!`s. The `loractl` CLI renders events as a terminal progress bar;
//! a future HTTP API would serialize the *same* events as SSE/JSON. Both are
//! just different renderers over one pipeline.

pub mod config;
pub mod event;
pub mod lora;
pub mod train;

pub use config::TrainConfig;
pub use event::TrainEvent;
pub use lora::LoraLinear;
pub use train::{MockTrainer, Trainer};
