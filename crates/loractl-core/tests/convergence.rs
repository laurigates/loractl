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

mod support;
use support::TempDir;

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
            // Validation samples are exercised by `validation_samples_are_written_and_reported`
            // below; disabled here to keep this test focused on convergence.
            sample_every: 0,
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
    // Regression pin for the M4 (#3) migration off the old `.mpk` full-model
    // checkpoint format onto adapter-only `.safetensors`.
    assert_eq!(
        adapter.extension().and_then(|e| e.to_str()),
        Some("safetensors"),
        "the final adapter must be a .safetensors file, not the old .mpk format"
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

/// Issue #3 (M4), acceptance criterion 3: in-training validation samples are
/// gated by `output.sample_every` and reported via `TrainEvent::Sample`.
///
/// Drives the real `BurnTrainer` with `sample_every` set to a divisor of
/// `steps` and asserts, as a black box: a `Sample` event fires at every
/// multiple of `sample_every` (and only there), the `sample-{step}.json`
/// file it names actually exists, and its contents are a parseable report
/// naming the same step plus a predicted class and logits. This is the only
/// automated coverage of `burn_trainer.rs`'s `sample_due` gate and
/// `TrainEvent::Sample` wiring — `adapter_roundtrip.rs` never calls
/// `BurnTrainer::train` at all, so it does not exercise this path.
#[test]
fn validation_samples_are_written_and_reported() {
    let steps = 10u64;
    let sample_every = 5u64;
    let out = TempDir::new("sample");
    let config = TrainConfig {
        steps,
        seed: 7,
        model: ModelConfig {
            base: "synthetic".into(),
        },
        lora: LoraConfig {
            rank: 4,
            alpha: 8.0,
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
            // No mid-run checkpoints — keep this test focused on sampling.
            checkpoint_every: 10_000,
            sample_every,
        },
    };

    let mut sample_events = Vec::new();
    let mut trainer = BurnTrainer;
    trainer
        .train(&config, &mut |event| {
            if let TrainEvent::Sample { step, path } = event {
                sample_events.push((step, path));
            }
        })
        .expect("training run succeeds");

    assert_eq!(
        sample_events
            .iter()
            .map(|(step, _)| *step)
            .collect::<Vec<_>>(),
        vec![5, 10],
        "a Sample event must fire at exactly every multiple of sample_every"
    );

    for (step, path) in &sample_events {
        assert!(
            path.exists(),
            "sample-{step}.json must actually exist on disk at {}",
            path.display()
        );
        let contents = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("reading validation sample report {}: {e}", path.display()));
        let report: serde_json::Value = serde_json::from_str(&contents)
            .unwrap_or_else(|e| panic!("parsing validation sample report {}: {e}", path.display()));
        assert_eq!(
            report["step"].as_u64(),
            Some(*step),
            "report must name its own step"
        );
        assert!(
            report["predicted_class"].is_u64(),
            "report must contain a predicted_class: {report}"
        );
        assert!(
            report["logits"].as_array().is_some_and(|l| !l.is_empty()),
            "report must contain non-empty logits: {report}"
        );
    }
}
