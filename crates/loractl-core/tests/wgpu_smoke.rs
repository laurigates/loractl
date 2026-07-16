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
//!
//! The config builder and run-and-assert driver live in `common/mod.rs`,
//! shared with the cuda sibling (`cuda_smoke.rs`).
#![cfg(feature = "wgpu")]

mod common;

use burn::backend::Wgpu;
use common::{run_smoke, smoke_config};
use loractl_core::config::{BackendKind, ComputeConfig, Precision};

#[test]
#[ignore = "requires a GPU (Metal on Apple Silicon); run via `just test-wgpu`"]
fn wgpu_training_smoke() {
    let steps = 120u64;
    let compute = ComputeConfig {
        backend: BackendKind::Wgpu,
        ..ComputeConfig::default()
    };
    let (config, out_dir) = smoke_config(compute, "wgpu", steps);
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
    let (config, out_dir) = smoke_config(compute, "wgpu-f16-ckpt", steps);
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
