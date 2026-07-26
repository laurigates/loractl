//! Shared integration-test helpers.
//!
//! Two groups, with different compile profiles:
//!
//! - **Numeric comparison** ([`max_abs_diff`], [`cosine`], [`assert_stage`]) —
//!   used by the staged-parity suite (`*_parity.rs`, `*_real.rs`), which is
//!   partly always-compiled and partly behind the opt-in real-weights features.
//! - **GPU smoke** ([`smoke_config`], [`run_smoke`]) — used by `wgpu_smoke.rs`
//!   and `cuda_smoke.rs`, each fully feature-gated at file level.
//!
//! Every integration test is its own crate, so Cargo compiles this module
//! separately into each consumer, and each consumer uses only a subset of it.
//! That makes unused items the norm here rather than a smell — hence the
//! blanket allow, which is what keeps `just lint`'s `-D warnings` green.
#![allow(dead_code)]

use loractl_core::config::{
    ComputeConfig, DatasetConfig, LoraConfig, OptimConfig, OutputConfig, TaskKind,
};
use loractl_core::{BurnTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::PathBuf;

/// Peak absolute elementwise difference between two flattened tensors.
///
/// Panics on a length mismatch rather than comparing a prefix: two staged
/// activations of different lengths mean the shapes diverged, which is a
/// bigger failure than any tolerance breach and should not be reported as one.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Cosine similarity, accumulated in f64.
///
/// The tolerance-free backstop in the parity suite: it is scale-invariant, so
/// it catches a structurally wrong result (permuted heads, a dropped residual)
/// that a generous absolute tolerance would wave through.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

/// Assert one stage of a staged-parity comparison, reporting the margin.
///
/// The `eprintln!` is deliberate: on a green run `cargo test -- --nocapture`
/// shows how much headroom each stage had, which is what tells you a tolerance
/// is drifting toward its limit *before* it starts failing.
pub fn assert_stage(name: &str, got: &[f32], want: &[f32], tol: f32) {
    let diff = max_abs_diff(got, want);
    assert!(diff <= tol, "{name}: max|Δ| = {diff:e} exceeds tol {tol:e}",);
    eprintln!("{name}: max|Δ| = {diff:e} (tol {tol:e})");
}

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
        lora: LoraConfig {
            rank: 8,
            ..Default::default()
        },
        dataset: DatasetConfig {
            resolution: 28,
            ..Default::default()
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
        ..Default::default()
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
