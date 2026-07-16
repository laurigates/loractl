//! Coverage for `BurnTrainer`'s periodic in-training validation-sample path
//! (issue #3, milestone 4 — acceptance criterion 3: `TrainEvent::Sample` +
//! `sample-{step}.json`).
//!
//! `convergence.rs` and `mnist_lora.rs` both set `output.sample_every: 0`,
//! which *disables* exactly the `want_sample` branch in
//! `crates/loractl-core/src/burn_trainer.rs` this test exercises — so without
//! it, a broken `TrainEvent::Sample` emission, a corrupted
//! `sample-{step}.json`, or an off-by-one in the `step % sample_every` gate
//! would pass `cargo test` cleanly.

use loractl_core::config::{
    ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig, OutputConfig,
    TaskKind,
};
use loractl_core::{BurnTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::PathBuf;

/// A unique temp output dir so concurrent test runs don't collide or litter the
/// repo. Removed on drop — same convention as `convergence.rs`.
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

#[test]
fn periodic_validation_samples_are_emitted_and_written() {
    let steps = 10u64;
    let sample_every = 5u64;
    let out = TempDir::new("validation-sample");
    let config = TrainConfig {
        steps,
        seed: 1,
        // Validation sampling is classification-specific (the flow task bails
        // on sample_every > 0 — see flow_task.rs).
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
            rank: 4,
            alpha: 8.0,
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
            weight_decay: 0.0,
        },
        output: OutputConfig {
            dir: out.0.clone(),
            name: "adapter".into(),
            // Larger than `steps` so no checkpoints fire — isolates the
            // sample path from the (already-covered) checkpoint path.
            checkpoint_every: 10_000,
            sample_every,
        },
        // Default (ndarray) backend — the offline sample-path test stays on CPU.
        compute: ComputeConfig::default(),
        // Unused by the classification task.
        flow: FlowConfig::default(),
    };

    let mut sample_events: Vec<(u64, PathBuf)> = Vec::new();
    let mut trainer = BurnTrainer;
    trainer
        .train(&config, &mut |event| {
            if let TrainEvent::Sample { step, path } = event {
                sample_events.push((step, path));
            }
        })
        .expect("training run succeeds");

    let expected_steps: Vec<u64> = (1..=steps).filter(|s| s % sample_every == 0).collect();
    assert_eq!(
        sample_events.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
        expected_steps,
        "TrainEvent::Sample must fire at exactly the step % sample_every == 0 steps"
    );

    for (step, path) in &sample_events {
        assert_eq!(
            *path,
            out.0.join(format!("sample-{step}.json")),
            "the emitted path must match the file `BurnTrainer` actually writes"
        );
        let json = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("sample-{step}.json must exist and be readable: {e}"));
        let report: serde_json::Value =
            serde_json::from_str(&json).expect("sample-{step}.json must contain valid JSON");
        assert_eq!(
            report["step"].as_u64(),
            Some(*step),
            "the report's `step` field must match the emitted event's step"
        );
        assert!(
            report["predicted_class"].is_u64(),
            "sample report must contain a numeric `predicted_class`"
        );
        assert!(
            report["logits"].as_array().is_some_and(|l| !l.is_empty()),
            "sample report must contain a non-empty `logits` array"
        );
    }
}
