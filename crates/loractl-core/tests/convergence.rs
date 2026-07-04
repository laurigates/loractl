//! End-to-end convergence proof for the default (synthetic) `BurnTrainer` path
//! (issue #1, criterion 2 — the always-run, offline half).
//!
//! This drives the *public* `BurnTrainer` over its seeded synthetic dataset as a
//! black box — observing only the `TrainEvent` stream, never reaching into the
//! model — and asserts that real training happens: the loss stream trends
//! decisively downward, the run brackets its work with `Started`/`Finished`, and
//! the returned adapter record actually exists on disk. Fast and network-free;
//! the MNIST convergence proof on *real* data is the feature-gated `#[ignore]`
//! test in `mnist_lora.rs`.

use loractl_core::config::{DatasetConfig, LoraConfig, ModelConfig, OptimConfig, OutputConfig};
use loractl_core::{BurnTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::PathBuf;

/// A unique temp output dir so concurrent test runs don't collide or litter the
/// repo. Removed on drop.
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

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len() as f32
}

#[test]
fn synthetic_training_converges() {
    let steps = 120u64;
    let out = TempDir::new("conv");
    let config = TrainConfig {
        steps,
        seed: 42,
        model: ModelConfig {
            base: "synthetic".into(),
        },
        lora: LoraConfig {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
        },
        dataset: DatasetConfig {
            path: PathBuf::from("unused"),
            resolution: 28,
        },
        optim: OptimConfig {
            lr: 0.01,
            weight_decay: 0.0,
        },
        output: OutputConfig {
            dir: out.0.clone(),
            name: "adapter".into(),
            // Larger than `steps` so no mid-run checkpoints — keeps the test fast
            // while still exercising the final-adapter write.
            checkpoint_every: 10_000,
        },
    };

    let mut losses = Vec::new();
    let mut started = false;
    let mut finished_path = None;

    let mut trainer = BurnTrainer;
    let adapter = trainer
        .train(&config, &mut |event| match event {
            TrainEvent::Started { total_steps } => {
                assert_eq!(total_steps, steps);
                started = true;
            }
            TrainEvent::Step { loss, .. } => losses.push(loss),
            TrainEvent::Finished { adapter_path } => finished_path = Some(adapter_path),
            _ => {}
        })
        .expect("training run succeeds");

    assert!(started, "a Started event must be emitted");
    assert_eq!(
        finished_path.as_ref(),
        Some(&adapter),
        "the Finished event's path must equal the returned adapter path"
    );
    assert!(
        adapter.exists(),
        "the final adapter record must exist on disk at {}",
        adapter.display()
    );
    assert_eq!(
        losses.len(),
        steps as usize,
        "exactly one Step event per step"
    );
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "all losses must be finite"
    );

    // Black-box convergence: the mean of the last third of losses must be well
    // below the mean of the first third. Well-separated Gaussian blobs + a
    // rank-8 LoRA readout under Adam clear this with wide margin.
    let third = losses.len() / 3;
    let first = mean(&losses[..third]);
    let last = mean(&losses[losses.len() - third..]);
    assert!(
        last < 0.7 * first,
        "loss should trend down: first-third mean {first:.4}, last-third mean {last:.4}"
    );
}
