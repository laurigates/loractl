//! Regression proof that `optim.weight_decay` actually reaches the optimizer
//! (issue #43).
//!
//! `weight_decay` was declared, documented, and deserialized but never wired
//! into the Adam(W) constructor, so every value was silently ignored. Every
//! other `BurnTrainer` test sets `weight_decay: 0.0`, so the drop was invisible.
//!
//! This drives the *public* `BurnTrainer` twice over the seeded synthetic
//! classification demo — identical in every field except `weight_decay` (0.0 vs
//! a large 1.0) — and asserts the emitted loss streams **differ**. Because the
//! seed, data, and every other knob are identical, weight decay is the only
//! possible source of divergence: if it were ignored (the bug), the two runs
//! would emit a byte-identical loss sequence. Observing only the `TrainEvent`
//! stream keeps this a black-box test (no reaching into the model). Offline,
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

/// Run the synthetic classification demo with the given `weight_decay`, seed 42,
/// and collect the per-step loss stream.
fn run_losses(weight_decay: f64, out: &TempDir) -> Vec<f32> {
    let config = TrainConfig {
        steps: 40,
        seed: 42,
        task: TaskKind::Classification,
        model: ModelConfig {
            base: "synthetic".into(),
            variant: Default::default(),
            checkpoint: None,
            denoiser: None,
            text_encoder: None,
            vae: None,
            tokenizer: None,
        },
        lora: LoraConfig {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            targets: vec![],
        },
        dataset: DatasetConfig {
            path: PathBuf::from("unused"),
            resolution: 28,
            batch_size: 1,
        },
        optim: OptimConfig {
            lr: 0.01,
            weight_decay,
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
fn weight_decay_changes_the_loss_trajectory() {
    // Two sequential runs in one test function (one process): `BurnTrainer`
    // reseeds the global RNG at the start of each run, so both see identical
    // synthetic data and identical init — the ONLY difference is weight_decay.
    let out_none = TempDir::new("wd-none");
    let out_strong = TempDir::new("wd-strong");

    let losses_none = run_losses(0.0, &out_none);
    let losses_strong = run_losses(1.0, &out_strong);

    assert_eq!(
        losses_none.len(),
        losses_strong.len(),
        "both runs should emit the same number of Step events"
    );
    assert!(
        losses_none.iter().all(|l| l.is_finite()) && losses_strong.iter().all(|l| l.is_finite()),
        "all losses must be finite"
    );

    // The kill assertion: if `weight_decay` were ignored, these two seeded,
    // otherwise-identical runs would produce a byte-identical loss sequence.
    // A strong decay (1.0) must visibly perturb the trajectory.
    let differs = losses_none
        .iter()
        .zip(&losses_strong)
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        differs,
        "weight_decay=1.0 must change the loss trajectory vs 0.0 — identical \
         streams mean weight_decay is being ignored.\n none:   {losses_none:?}\n strong: {losses_strong:?}"
    );
}
