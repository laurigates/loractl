//! Integration test for the `loractl sample` subcommand (issue #3, M4,
//! acceptance criterion 2 — "`loractl sample` produces real output, no
//! longer bailing").
//!
//! `crates/loractl-core/tests/adapter_roundtrip.rs` covers the underlying
//! `adapter::load_adapter` / `sample::run_sample` *library* calls directly,
//! but nothing anywhere invokes the CLI binary's own `Sample` command path —
//! this test drives the compiled `loractl` executable so a regression purely
//! inside `cli.rs`'s `sample()` function (device/backend wiring, the
//! `with_context` error wrapping, or the `println!` output formatting) would
//! be caught here, rather than compiling and passing `just test` untouched.

use loractl_core::adapter;
use loractl_core::{Device, LoraMlp, NdArray, TaskKind};
use std::process::Command;

#[test]
fn sample_subcommand_prints_real_output() {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("loractl-cli-sample-{}-{nanos}", std::process::id()));
    let adapter_path = dir.join("adapter.safetensors");

    // Build and save a tiny adapter directly through the library. This test
    // is about exercising the CLI's `sample` wiring end-to-end, not about
    // producing a "real" trained adapter — that round-trip is
    // `adapter_roundtrip.rs`'s job. `save_adapter` creates `dir` itself.
    let device: Device<NdArray> = Default::default();
    let model = LoraMlp::<NdArray>::new(8, 6, 4, 2, 8.0, 0.0, &device);
    adapter::save_adapter(&model, &adapter_path, 99, TaskKind::Classification)
        .expect("save a tiny adapter for the CLI to load");

    let exe = env!("CARGO_BIN_EXE_loractl");
    let output = Command::new(exe)
        .args([
            "sample",
            adapter_path.to_str().expect("adapter path is valid UTF-8"),
            "--prompt",
            "hello world",
        ])
        .output()
        .expect("run `loractl sample`");

    assert!(
        output.status.success(),
        "`loractl sample` should exit 0; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("predicted class:"),
        "stdout should report a predicted class, got:\n{stdout}"
    );
    assert!(
        stdout.contains("top logits:"),
        "stdout should report the top logits, got:\n{stdout}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
