//! The training contract and a dependency-free stand-in.
//!
//! [`Trainer`] is the interface every backend implements. [`MockTrainer`]
//! satisfies it with zero ML so the whole event → render pipeline can be
//! exercised today. Milestone 2 adds a burn-backed trainer behind this same
//! trait — the CLI won't change, it'll just get a different `impl Trainer`.

use crate::burn_trainer::BurnTrainer;
use crate::config::TrainConfig;
use crate::diffusion_trainer::DiffusionTrainer;
use crate::event::TrainEvent;
use anyhow::Result;
use std::path::PathBuf;

/// A training backend.
///
/// Implementors run the job described by `config` and report progress by
/// calling `sink` with [`TrainEvent`]s. They must not render progress or
/// write to stdout/stderr themselves — that's the caller's job. The returned
/// path is the final adapter written to disk.
pub trait Trainer {
    /// Runs the job described by `config`, reporting progress by calling
    /// `sink` with [`TrainEvent`]s, and returns the path of the final adapter
    /// written to disk. Implementors must not render or write to
    /// stdout/stderr — surfacing events is the caller's responsibility.
    fn train(&mut self, config: &TrainConfig, sink: &mut dyn FnMut(TrainEvent)) -> Result<PathBuf>;
}

/// Picks the concrete trainer for a run from `model.base` — the single
/// routing seam both front-ends (the `loractl` CLI and `loractl-api`) call,
/// so the mapping cannot drift between them.
///
/// - `"synthetic"` and `"mnist"` are [`BurnTrainer`]'s documented bases: the
///   offline LoRA-MLP demo (M2) and its opt-in real-MNIST path
///   (`--features mnist`; without the feature the trainer warns and falls
///   back to the demo).
/// - Anything else is treated as a path to a Krea-2-Raw-layout checkpoint
///   directory and routes to the M14 [`DiffusionTrainer`].
///
/// Routing is pinned by `tests/trainer_routing.rs`, which discriminates the
/// arms by their observable behavior (the demo's event stream, the mnist
/// fallback warning, the diffusion trainer's flow-matching bail).
pub fn select_trainer(config: &TrainConfig) -> Box<dyn Trainer + Send> {
    match config.model.base.as_str() {
        "synthetic" | "mnist" => Box::new(BurnTrainer),
        _ => Box::new(DiffusionTrainer),
    }
}

/// A stand-in trainer that exercises the event pipeline without any ML.
///
/// It runs the configured number of steps, emits a smoothly decaying
/// synthetic loss, checkpoints on the configured cadence, and returns the
/// final adapter path. Everything it "writes" is only reported via events —
/// it performs no disk I/O — so it's safe to run anywhere.
pub struct MockTrainer;

impl Trainer for MockTrainer {
    fn train(&mut self, config: &TrainConfig, sink: &mut dyn FnMut(TrainEvent)) -> Result<PathBuf> {
        let total = config.steps.max(1);
        sink(TrainEvent::Started { total_steps: total });

        let checkpoint_every = config.output.checkpoint_every.max(1);
        for step in 1..=total {
            // Synthetic loss: exponential decay toward a noise floor, with a
            // small deterministic wobble so the rendered number looks alive.
            let progress = step as f32 / total as f32;
            let loss = 2.0 * (-3.0 * progress).exp() + 0.02 * (step as f32 * 0.3).sin().abs();
            sink(TrainEvent::Step {
                step,
                loss,
                lr: config.optim.lr,
            });

            if step % checkpoint_every == 0 {
                let path = config
                    .output
                    .dir
                    .join(format!("checkpoint-{step}.safetensors"));
                sink(TrainEvent::Checkpoint { step, path });
            }
        }

        let adapter_path = config
            .output
            .dir
            .join(format!("{}.safetensors", config.output.name));
        sink(TrainEvent::Finished {
            adapter_path: adapter_path.clone(),
        });
        Ok(adapter_path)
    }
}
