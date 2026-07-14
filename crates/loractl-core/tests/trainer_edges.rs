//! Small `BurnTrainer` contract/edge coverage from the test-strategy review (#49):
//!
//! - **H10** the `lr` emitted in each `Step` equals `config.optim.lr` (no test
//!   read it — a mutant emitting `lr: 0.0` passed);
//! - **H10** the ndarray `device != 0` warning arm fires (all tests used
//!   `device: 0`);
//! - **H13** `steps: 0` is clamped to at least one step (`run_training`'s
//!   `config.steps.max(1)`), so the trainer still runs rather than emitting
//!   nothing or panicking.
//!
//! Black-box: observes only the `TrainEvent` stream. Offline, milliseconds.

use loractl_core::config::{
    BackendKind, ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig,
    OutputConfig, TaskKind,
};
use loractl_core::{BurnTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::PathBuf;

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

fn config(out: &TempDir, steps: u64, lr: f64, compute: ComputeConfig) -> TrainConfig {
    TrainConfig {
        steps,
        seed: 42,
        task: TaskKind::Classification,
        model: ModelConfig {
            base: "synthetic".into(),
            variant: Default::default(),
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
            lr,
            weight_decay: 0.0,
        },
        output: OutputConfig {
            dir: out.0.clone(),
            name: "adapter".into(),
            checkpoint_every: 10_000,
            sample_every: 0,
        },
        compute,
        flow: FlowConfig::default(),
    }
}

#[test]
fn step_events_carry_the_configured_lr() {
    let out = TempDir::new("edge-lr");
    let lr = 0.0123;
    let cfg = config(&out, 5, lr, ComputeConfig::default());

    let mut lrs = Vec::new();
    let mut trainer = BurnTrainer;
    trainer
        .train(&cfg, &mut |event| {
            if let TrainEvent::Step { lr, .. } = event {
                lrs.push(lr);
            }
        })
        .expect("training run succeeds");

    assert_eq!(lrs.len(), 5, "one Step per step");
    assert!(
        lrs.iter().all(|&l| l == lr),
        "every Step must carry config.optim.lr = {lr}, got {lrs:?}"
    );
}

#[test]
fn ndarray_nonzero_device_emits_a_warning() {
    let out = TempDir::new("edge-warn");
    // ndarray ignores the device index; selecting a non-zero one must warn (not
    // silently pretend to use GPU ordinal 1).
    let cfg = config(
        &out,
        2,
        0.01,
        ComputeConfig {
            backend: BackendKind::Ndarray,
            device: 1,
            ..ComputeConfig::default()
        },
    );

    let mut warnings = Vec::new();
    let mut trainer = BurnTrainer;
    trainer
        .train(&cfg, &mut |event| {
            if let TrainEvent::Warning { message } = event {
                warnings.push(message);
            }
        })
        .expect("training run succeeds");

    assert!(
        warnings.iter().any(|m| m.contains("ignores device index")),
        "ndarray with device != 0 must warn, got warnings: {warnings:?}"
    );
}

#[test]
fn zero_steps_is_clamped_to_at_least_one() {
    let out = TempDir::new("edge-zero-steps");
    let cfg = config(&out, 0, 0.01, ComputeConfig::default());

    let mut steps = Vec::new();
    let mut finished = false;
    let mut trainer = BurnTrainer;
    trainer
        .train(&cfg, &mut |event| match event {
            TrainEvent::Step { step, .. } => steps.push(step),
            TrainEvent::Finished { .. } => finished = true,
            _ => {}
        })
        .expect("training run with steps=0 must not panic");

    assert!(
        !steps.is_empty(),
        "steps=0 must be clamped to at least one real step"
    );
    assert!(finished, "the run must still finish");
}
