//! CUDA portability smoke — the cuda sibling of `wgpu_smoke.rs` (M7, #18).
//!
//! Double-gated so it can NEVER run in CI: the whole file is `#![cfg(feature =
//! "cuda")]` (cuda is never compiled in CI or on macOS — burn-cuda needs nvcc
//! at build time), and the test is `#[ignore]`d. Run it explicitly on a
//! Linux + NVIDIA host with the CUDA toolkit:
//!
//! ```text
//! just test-cuda
//! ```
//!
//! Same contract as the wgpu smoke: a PORTABILITY check, never a
//! numerics-golden target. `grad_checkpointing: true` rides along
//! deliberately — on the diffusion path it exercises the #134 block-level
//! checkpointed step (`block_ckpt::checkpointed_step`) on cuda in the same
//! run, and the plain (non-checkpointed) cuda backward is covered
//! independently by the `grad_compare` example's cuda arms, so a failure here
//! with a clean `grad_compare` points at checkpointing, not the backend.
#![cfg(feature = "cuda")]

mod common;

use burn::backend::Cuda;
use common::{run_smoke, smoke_config};
use loractl_core::config::{BackendKind, ComputeConfig};

#[test]
#[ignore = "requires an NVIDIA GPU (CUDA toolkit at build time); run via `just test-cuda`"]
fn cuda_training_smoke() {
    let steps = 120u64;
    let compute = ComputeConfig {
        backend: BackendKind::Cuda,
        grad_checkpointing: true,
        ..ComputeConfig::default()
    };
    let (config, out_dir) = smoke_config(compute, "cuda", steps);
    let adapter = run_smoke(&config, steps);

    // Reload the adapter on the cuda device and forward once — the same
    // reseed -> reconstruct -> forward lazy-Param path the wgpu smoke pins
    // (see .claude/rules/burn-lazy-param-init.md), on the cuda backend.
    let device = Default::default();
    let reloaded = loractl_core::adapter::load_adapter::<Cuda>(&adapter, &device)
        .expect("reload adapter on cuda");
    let out = loractl_core::sample::run_sample(&reloaded, 0, &device).expect("sample on cuda");
    assert!(
        out.logits.iter().all(|l| l.is_finite()),
        "reloaded cuda logits should be finite"
    );

    let _ = std::fs::remove_dir_all(&out_dir);
}
