//! Pins core's [`select_trainer`] routing — the single `model.base` →
//! trainer factory both front-ends call (review: the routing predicate must
//! not regress the documented BurnTrainer bases, and must be covered by
//! tests rather than living untested in `main.rs`/`cli.rs`).
//!
//! The factory returns `Box<dyn Trainer>`, so the arms are discriminated by
//! observable behavior, not type names:
//!
//! - `"synthetic"` completes the offline LoRA-MLP demo run (a
//!   `DiffusionTrainer` would bail on `task != flow-matching` first);
//! - `"mnist"` without `--features mnist` reaches [`BurnTrainer`]'s
//!   documented fallback warning — proving the mnist base still routes to
//!   the demo trainer, not to the diffusion path;
//! - any other base (a checkpoint-directory path) hits the diffusion
//!   trainer's distinctive rectified-flow bail under the default
//!   classification task.

use loractl_core::config::{
    ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig, OutputConfig,
    TaskKind,
};
use loractl_core::{TrainConfig, TrainEvent, select_trainer};
use std::path::PathBuf;
use std::sync::Mutex;

/// See `adapter_roundtrip.rs`: the ndarray RNG is a process-global static, so
/// tests that run the seeded BurnTrainer serialize against each other.
static RNG_LOCK: Mutex<()> = Mutex::new(());

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
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn config(base: &str, out: &TempDir) -> TrainConfig {
    TrainConfig {
        steps: 2,
        seed: 42,
        task: TaskKind::Classification,
        model: ModelConfig {
            base: base.into(),
            variant: Default::default(),
        },
        lora: LoraConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            targets: vec![],
        },
        dataset: DatasetConfig {
            path: PathBuf::from("unused"),
            resolution: 32,
            batch_size: 1,
        },
        optim: OptimConfig {
            lr: 0.01,
            weight_decay: 0.0,
        },
        output: OutputConfig {
            dir: out.0.clone(),
            name: "routing".into(),
            checkpoint_every: 10_000,
            sample_every: 0,
        },
        compute: ComputeConfig::default(),
        flow: FlowConfig::default(),
    }
}

/// `base: synthetic` routes to the BurnTrainer demo: the run completes under
/// the default classification task and emits one Step per step — the
/// diffusion trainer would have bailed before its first event.
#[test]
fn synthetic_base_routes_to_the_demo_trainer() {
    let _guard = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let out = TempDir::new("route-synth");
    let config = config("synthetic", &out);

    let mut steps = 0u64;
    let mut trainer = select_trainer(&config);
    trainer
        .train(&config, &mut |event| {
            if let TrainEvent::Step { .. } = event {
                steps += 1;
            }
        })
        .expect("the synthetic demo must run under the factory");
    assert_eq!(steps, config.steps, "one Step per step through the demo");
}

/// `base: mnist` stays a BurnTrainer base. Built without `--features mnist`
/// (the default test build) the demo trainer emits its documented fallback
/// warning — which only BurnTrainer produces; the diffusion trainer would
/// instead error on `task != flow-matching`. Cfg-gated off under the mnist
/// feature, where this base would download the real dataset.
#[cfg(not(feature = "mnist"))]
#[test]
fn mnist_base_routes_to_the_demo_trainer() {
    let _guard = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let out = TempDir::new("route-mnist");
    let config = config("mnist", &out);

    let mut fallback_seen = false;
    let mut trainer = select_trainer(&config);
    trainer
        .train(&config, &mut |event| {
            if let TrainEvent::Warning { message } = &event
                && message.contains("falling back to the synthetic demo")
            {
                fallback_seen = true;
            }
        })
        .expect("the mnist base must reach BurnTrainer, not the diffusion bail");
    assert!(
        fallback_seen,
        "BurnTrainer's no-feature mnist fallback warning proves the routing arm"
    );
}

/// Any other base is a checkpoint-directory path for the diffusion trainer:
/// under the default classification task it must hit that trainer's
/// distinctive rectified-flow bail (validated before the path is touched, so
/// no fixture is needed).
#[test]
fn directory_base_routes_to_the_diffusion_trainer() {
    let out = TempDir::new("route-diffusion");
    let config = config("./some/krea2/checkpoint", &out);

    let err = select_trainer(&config)
        .train(&config, &mut |_| {})
        .expect_err("classification task must be refused by the diffusion trainer");
    assert!(
        err.to_string().contains("rectified-flow"),
        "expected the diffusion trainer's flow-matching bail, got: {err}"
    );
}
