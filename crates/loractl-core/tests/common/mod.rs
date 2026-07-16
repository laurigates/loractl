//! Shared GPU-smoke helpers: the synthetic smoke `TrainConfig` builder and the
//! backend-agnostic run-and-assert driver. Consumed by `wgpu_smoke.rs` and
//! `cuda_smoke.rs` — each of those is fully feature-gated at file level, so
//! this module is only ever compiled when a GPU feature is on and never in the
//! default/offline build.

use loractl_core::config::{
    ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig, OutputConfig,
    TaskKind,
};
use loractl_core::{BurnTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::PathBuf;

/// Build the smoke's TrainConfig for the given compute selection.
pub fn smoke_config(compute: ComputeConfig, tag: &str, steps: u64) -> (TrainConfig, PathBuf) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!("loractl-smoke-{tag}-{nanos}"));

    let config = TrainConfig {
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
            dir: out_dir.clone(),
            name: "gpu-adapter".into(),
            // Larger than `steps` so no mid-run checkpoints fire.
            checkpoint_every: 10_000,
            sample_every: 0,
        },
        // The whole point: run on the caller-selected GPU backend.
        compute,
        // Unused by the classification task.
        flow: FlowConfig::default(),
    };
    (config, out_dir)
}

/// Drive one training run and apply the portability assertions (finite,
/// decreasing loss; one Step per step; adapter written).
pub fn run_smoke(config: &TrainConfig, steps: u64) -> PathBuf {
    let mut losses = Vec::new();
    let mut started_total = None;
    let mut step_count = 0u64;
    let mut finished_path = None;

    let mut trainer = BurnTrainer;
    let adapter = trainer
        .train(config, &mut |event| match event {
            TrainEvent::Started { total_steps } => started_total = Some(total_steps),
            TrainEvent::Step { loss, .. } => {
                step_count += 1;
                losses.push(loss);
            }
            TrainEvent::Finished { adapter_path } => finished_path = Some(adapter_path),
            _ => {}
        })
        .expect("GPU training run should complete end-to-end");

    // Started announced the configured length.
    assert_eq!(started_total, Some(steps), "Started total_steps mismatch");
    // Exactly one Step event per step.
    assert_eq!(step_count, steps, "expected one Step event per step");
    // Every loss finite — a broken GPU kernel dispatch surfaces as NaN/Inf.
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "non-finite loss on the GPU backend: {losses:?}"
    );
    // Loss decreased — a LOOSE portability bound (deliberately not
    // convergence.rs's 0.7, and never compared to the ndarray numerics golden).
    let first = losses.first().copied().expect("at least one loss");
    let last = losses.last().copied().expect("at least one loss");
    assert!(
        last < 0.9 * first,
        "loss should decrease: first={first}, last={last}"
    );

    // End-to-end proof: the GPU run actually wrote the adapter to disk.
    let adapter = finished_path.unwrap_or(adapter);
    assert!(
        adapter.exists(),
        "adapter file should exist at {}",
        adapter.display()
    );
    adapter
}
