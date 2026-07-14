//! Regression proof that `lora.dropout` actually reaches training
//! (issue #44).
//!
//! `lora.dropout` was declared and documented but never applied — no `Dropout`
//! layer existed in the adapter path, so any value was silently ignored. Every
//! other `BurnTrainer` test sets `dropout: 0.0`, so the drop was invisible.
//!
//! This drives the *public* `BurnTrainer` twice over the seeded synthetic
//! classification demo — identical in every field except `lora.dropout` (0.0
//! vs 0.5) — and asserts the emitted loss streams **differ**. Because dropout
//! is applied only on the autodiff (training) backend and draws a Bernoulli
//! mask over the adapter input, a non-zero dropout perturbs the trajectory; if
//! it were ignored (the bug), the two seeded runs would emit a byte-identical
//! loss sequence. Black-box (observes only the `TrainEvent` stream). Offline,
//! milliseconds.

use loractl_core::config::{
    ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig, OutputConfig,
    TaskKind,
};
use loractl_core::{BurnTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::PathBuf;

/// A unique temp output dir, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("loractl-{tag}-{}-{nanos}", std::process::id()));
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run the synthetic classification demo with the given adapter `dropout`, seed
/// 42, and collect the per-step loss stream.
fn run_losses(dropout: f32, out: &TempDir) -> Vec<f32> {
    let config = TrainConfig {
        steps: 40,
        seed: 42,
        task: TaskKind::Classification,
        model: ModelConfig {
            base: "synthetic".into(),
            variant: Default::default(),
        },
        lora: LoraConfig {
            rank: 8,
            alpha: 16.0,
            dropout,
            targets: vec![],
        },
        dataset: DatasetConfig {
            path: PathBuf::from("unused"),
            resolution: 28,
            batch_size: 1,
        },
        optim: OptimConfig {
            lr: 0.01,
            weight_decay: 0.0,
        },
        output: OutputConfig {
            dir: out.0.clone(),
            name: "adapter".into(),
            checkpoint_every: 10_000,
            sample_every: 0,
        },
        compute: ComputeConfig::default(),
        flow: FlowConfig::default(),
    };

    let mut losses = Vec::new();
    let mut trainer = BurnTrainer;
    trainer
        .train(&config, &mut |event| {
            if let TrainEvent::Step { loss, .. } = event {
                losses.push(loss);
            }
        })
        .expect("training run succeeds");
    losses
}

#[test]
fn dropout_changes_the_loss_trajectory() {
    // Two sequential runs in one test function (one process): `BurnTrainer`
    // reseeds the global RNG at the start of each run, so both see identical
    // synthetic data and identical init — the ONLY difference is dropout.
    let out_none = TempDir::new("dropout-none");
    let out_active = TempDir::new("dropout-active");

    let losses_none = run_losses(0.0, &out_none);
    let losses_active = run_losses(0.5, &out_active);

    assert_eq!(
        losses_none.len(),
        losses_active.len(),
        "both runs should emit the same number of Step events"
    );
    assert!(
        losses_none.iter().all(|l| l.is_finite()) && losses_active.iter().all(|l| l.is_finite()),
        "all losses must be finite"
    );

    // The kill assertion: if `dropout` were ignored, these two seeded,
    // otherwise-identical runs would produce a byte-identical loss sequence.
    let differs = losses_none
        .iter()
        .zip(&losses_active)
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        differs,
        "dropout=0.5 must change the loss trajectory vs 0.0 — identical streams \
         mean dropout is being ignored.\n none:   {losses_none:?}\n active: {losses_active:?}"
    );
}
