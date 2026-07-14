//! GPU portability smoke (M7, #18).
//!
//! Double-gated so it can NEVER run in CI: the whole file is `#![cfg(feature =
//! "wgpu")]` (not compiled in the default/offline build), and the test is
//! `#[ignore]`d (skipped even under `cargo test --features wgpu`). Run it
//! explicitly on a Metal-capable Mac:
//!
//! ```text
//! just test-wgpu
//! ```
//!
//! This is a PORTABILITY check, not a numerics-golden target — per ADR-0001,
//! GPU float-reduction order differs from ndarray, so the bit-exact parity
//! tests stay offline on ndarray. Here we only prove the real `BurnTrainer`
//! runs end-to-end on `Autodiff<Wgpu>` (Metal on this Mac): it emits one `Step`
//! per step with finite, decreasing loss and writes an adapter to disk — the
//! local evidence for acceptance criterion #1 ("a training run executes
//! end-to-end on a GPU backend selected from config").
#![cfg(feature = "wgpu")]

use burn::backend::Wgpu;
use loractl_core::config::{
    BackendKind, ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig,
    OutputConfig, Precision, TaskKind,
};
use loractl_core::{BurnTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::PathBuf;

/// Build the smoke's TrainConfig for the given compute selection.
fn smoke_config(compute: ComputeConfig, tag: &str, steps: u64) -> (TrainConfig, PathBuf) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!("loractl-wgpu-{tag}-{nanos}"));

    let config = TrainConfig {
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
            lr: 0.01,
            weight_decay: 0.0,
        },
        output: OutputConfig {
            dir: out_dir.clone(),
            name: "wgpu-adapter".into(),
            // Larger than `steps` so no mid-run checkpoints fire.
            checkpoint_every: 10_000,
            sample_every: 0,
        },
        // The whole point: run on wgpu (Metal on this Mac), device 0.
        compute,
        // Unused by the classification task.
        flow: FlowConfig::default(),
    };
    (config, out_dir)
}

/// Drive one training run and apply the portability assertions (finite,
/// decreasing loss; one Step per step; adapter written).
fn run_smoke(config: &TrainConfig, steps: u64) -> PathBuf {
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
        .expect("wgpu training run should complete end-to-end");

    // Started announced the configured length.
    assert_eq!(started_total, Some(steps), "Started total_steps mismatch");
    // Exactly one Step event per step.
    assert_eq!(step_count, steps, "expected one Step event per step");
    // Every loss finite — a broken GPU kernel dispatch surfaces as NaN/Inf.
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "non-finite loss on wgpu: {losses:?}"
    );
    // Loss decreased — a LOOSE portability bound (deliberately not
    // convergence.rs's 0.7, and never compared to the ndarray numerics golden).
    let first = losses.first().copied().expect("at least one loss");
    let last = losses.last().copied().expect("at least one loss");
    assert!(
        last < 0.9 * first,
        "loss should decrease on wgpu: first={first}, last={last}"
    );
    // OBSERVED (Apple Silicon / Metal, seed 42, 120 steps): record first->last
    // here after the first local run, like mnist_lora.rs documents its numbers.

    // End-to-end proof: the GPU run actually wrote the adapter to disk.
    let adapter = finished_path.unwrap_or(adapter);
    assert!(
        adapter.exists(),
        "adapter file should exist at {}",
        adapter.display()
    );
    adapter
}

#[test]
#[ignore = "requires a GPU (Metal on Apple Silicon); run via `just test-wgpu`"]
fn wgpu_training_smoke() {
    let steps = 120u64;
    let compute = ComputeConfig {
        backend: BackendKind::Wgpu,
        ..ComputeConfig::default()
    };
    let (config, out_dir) = smoke_config(compute, "smoke", steps);
    let adapter = run_smoke(&config, steps);

    // Reload the adapter on the wgpu device and forward once. This exercises the
    // reseed -> reconstruct -> forward lazy-Param path (see
    // .claude/rules/burn-lazy-param-init.md) on the GPU, proving eager `.val()`
    // materialization of the frozen base fires off-ndarray too.
    let device = Default::default();
    let reloaded = loractl_core::adapter::load_adapter::<Wgpu>(&adapter, &device)
        .expect("reload adapter on wgpu");
    let out = loractl_core::sample::run_sample(&reloaded, 0, &device).expect("sample on wgpu");
    assert!(
        out.logits.iter().all(|l| l.is_finite()),
        "reloaded wgpu logits should be finite"
    );

    let _ = std::fs::remove_dir_all(&out_dir);
}

/// The M13 (#24) memory knobs on real Metal: `precision: f16` (halved weight
/// memory — the knob that fits the ~12B base on this host) combined with
/// `grad_checkpointing: true` (recompute activations) must train end-to-end.
/// Portability only — f16 numerics are looser, so the shared loose
/// loss-decrease bound applies and nothing is compared to the f32 goldens.
#[test]
#[ignore = "requires a GPU (Metal on Apple Silicon); run via `just test-wgpu`"]
fn wgpu_f16_checkpointing_smoke() {
    let steps = 120u64;
    let compute = ComputeConfig {
        backend: BackendKind::Wgpu,
        precision: Precision::F16,
        grad_checkpointing: true,
        ..ComputeConfig::default()
    };
    let (config, out_dir) = smoke_config(compute, "f16-ckpt", steps);
    let adapter = run_smoke(&config, steps);

    // The f16 run's adapter must be USABLE, not merely present: reload it on
    // the same-precision backend and forward once (the supported round-trip
    // — burn-store preserves the file's f16 dtype, so same-precision reload
    // is the contract; cross-precision reload is undefined until a cast pass
    // exists).
    let device = Default::default();
    let reloaded =
        loractl_core::adapter::load_adapter::<Wgpu<burn::tensor::f16>>(&adapter, &device)
            .expect("reload f16 adapter on wgpu<f16>");
    let out = loractl_core::sample::run_sample(&reloaded, 0, &device).expect("sample on wgpu<f16>");
    assert!(
        out.logits.iter().all(|l| l.is_finite()),
        "reloaded f16 logits should be finite"
    );

    let _ = std::fs::remove_dir_all(&out_dir);
}
