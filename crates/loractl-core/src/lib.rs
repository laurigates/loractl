//! `loractl-core` — the training pipeline behind loractl.
//!
//! This crate is deliberately free of CLI and I/O-presentation concerns. It
//! defines three things and nothing else:
//!
//! - [`TrainConfig`] — the run's declarative schema (deserialized from YAML).
//! - [`TrainEvent`] — the stream a trainer emits as it works.
//! - [`Trainer`] — the contract a concrete backend implements.
//!
//! The design rule that keeps a future GUI honest: **core emits events, the
//! caller renders them.** A trainer never draws a progress bar and never
//! `println!`s. The `loractl` CLI renders events as a terminal progress bar;
//! a future HTTP API would serialize the *same* events as SSE/JSON. Both are
//! just different renderers over one pipeline.

pub mod config;
pub mod event;
pub mod train;

pub use config::TrainConfig;
pub use event::TrainEvent;
pub use train::{MockTrainer, Trainer};
