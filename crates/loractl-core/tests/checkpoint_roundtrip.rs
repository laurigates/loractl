//! Regression proof that `BurnTrainer` actually writes reloadable checkpoints
//! at the configured cadence (issue #46).
//!
//! Every other `BurnTrainer` test sets `checkpoint_every: 10_000` over ≤150
//! steps, so the `want_checkpoint` arm in `burn_trainer.rs` never fires: the
//! `checkpoint-{step}.safetensors` path format, the mid-run `save_adapter`
//! call, and the `TrainEvent::Checkpoint` emission were all completely
//! untested (`event_json.rs` builds a `Checkpoint` event by hand — schema
//! only; the API tests drive `MockTrainer`, which writes no files). A mutant
//! renaming the checkpoint path, dropping the emission, or corrupting the
//! `.valid()` snapshot would pass the whole suite.
//!
//! This drives the *public* `BurnTrainer` with a small `checkpoint_every` and
//! asserts the checkpoints are emitted at the right steps, exist on disk, and
//! reload via `load_adapter` into a working model. A second test exercises the
//! combined `want_checkpoint || want_sample` branch (both firing at the same
//! step). Offline, milliseconds.

use loractl_core::adapter;
use loractl_core::config::{
    ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig, OutputConfig,
    TaskKind,
};
use loractl_core::{BurnTrainer, Device, NdArray, TrainConfig, TrainEvent, Trainer};
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

fn base_config(out: &TempDir, steps: u64, checkpoint_every: u64, sample_every: u64) -> TrainConfig {
    TrainConfig {
        steps,
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
            training_adapter: None,
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
            weight_decay: 0.0,
        },
        output: OutputConfig {
            dir: out.0.clone(),
            name: "adapter".into(),
            checkpoint_every,
            sample_every,
        },
        compute: ComputeConfig::default(),
        flow: FlowConfig::default(),
    }
}

#[test]
fn checkpoints_are_written_at_cadence_and_reload() {
    // steps=4, checkpoint_every=2 → checkpoints at steps 2 and 4.
    let out = TempDir::new("ckpt");
    let config = base_config(&out, 4, 2, 0);

    let mut checkpoints: Vec<(u64, PathBuf)> = Vec::new();
    let mut trainer = BurnTrainer;
    trainer
        .train(&config, &mut |event| {
            if let TrainEvent::Checkpoint { step, path } = event {
                checkpoints.push((step, path));
            }
        })
        .expect("training run succeeds");

    // Emitted at exactly the configured cadence.
    let steps: Vec<u64> = checkpoints.iter().map(|(s, _)| *s).collect();
    assert_eq!(
        steps,
        vec![2, 4],
        "checkpoint events must fire at steps 2 and 4"
    );

    // The path format is `checkpoint-{step}.safetensors`, the files exist, and
    // each reloads into a working model whose forward is finite — proving the
    // mid-run `save_adapter` wrote a valid adapter + sidecar.
    let device: Device<NdArray> = Default::default();
    for (step, path) in &checkpoints {
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            format!("checkpoint-{step}.safetensors"),
            "checkpoint path format"
        );
        assert!(
            path.exists(),
            "checkpoint file must exist: {}",
            path.display()
        );

        let reloaded = adapter::load_adapter::<NdArray>(path, &device)
            .unwrap_or_else(|e| panic!("checkpoint {} must reload: {e:#}", path.display()));
        let probe =
            burn::tensor::Tensor::<NdArray, 2>::zeros([1, reloaded.fc1.weight.dims()[0]], &device);
        let logits = reloaded.forward(probe);
        assert!(
            logits.into_data().iter::<f32>().all(|v| v.is_finite()),
            "reloaded checkpoint {} must forward to finite logits",
            path.display()
        );
    }
}

#[test]
fn checkpoint_and_sample_at_the_same_step_both_write() {
    // checkpoint_every == sample_every == 2, steps=2 → at step 2 the combined
    // `want_checkpoint || want_sample` branch fires both writes off the single
    // `.valid()` snapshot. Pins that neither clobbers the other.
    let out = TempDir::new("ckpt-sample");
    let config = base_config(&out, 2, 2, 2);

    let mut checkpoint_steps = Vec::new();
    let mut sample_steps = Vec::new();
    let mut trainer = BurnTrainer;
    trainer
        .train(&config, &mut |event| match event {
            TrainEvent::Checkpoint { step, .. } => checkpoint_steps.push(step),
            TrainEvent::Sample { step, .. } => sample_steps.push(step),
            _ => {}
        })
        .expect("training run succeeds");

    assert_eq!(checkpoint_steps, vec![2], "one checkpoint at step 2");
    assert_eq!(sample_steps, vec![2], "one sample at step 2");
    assert!(
        out.0.join("checkpoint-2.safetensors").exists(),
        "checkpoint file must exist alongside the sample"
    );
    assert!(
        out.0.join("sample-2.json").exists(),
        "sample file must exist alongside the checkpoint"
    );
}
