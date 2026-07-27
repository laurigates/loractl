//! Pins that `init_telemetry` actually **consumes** the resolved `Verbosity`.
//!
//! `level_for`/`resolve_verbosity`/`flag_directives` are pure and unit-tested,
//! but nothing else proves the filter they produce reaches the subscriber.
//! Reverting `init_telemetry` to `EnvFilter::from_default_env()` — the pre-fix
//! code that swallowed every `TrainEvent::Warning` with `RUST_LOG` unset — left
//! the whole suite green before this test existed, so the load-bearing
//! behaviour (a flagless run says *something*) rested on a manual smoke run no
//! gate replayed.
//!
//! `train()` is unreachable from a test — the crate has no `[lib]` target — so
//! this spawns the real binary, which is also the only way to observe what the
//! subscriber writes (the `fmt` layer renders to **stdout**, alongside the
//! CLI's own `println!`s).

use std::process::{Command, Output};

/// The synthetic demo's `TrainEvent::Warning`, rendered by the CLI at WARN.
const WARN_MARKER: &str = "BurnTrainer trains a synthetic LoRA-MLP classifier demo";

fn train(args: &[&str], tag: &str) -> (Output, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "loractl-verbosity-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let config = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../config/examples/lora.yaml"
    );
    let output = Command::new(env!("CARGO_BIN_EXE_loractl"))
        .args(args)
        .args(["train", config, "--steps", "1"])
        // The harness may inherit `RUST_LOG`; this test is about the flag path.
        .env_remove("RUST_LOG")
        // No `--output-dir` flag exists, so steer the writes via the env layer.
        .env("LORACTL_OUTPUT__DIR", &dir)
        .output()
        .expect("run `loractl train`");
    (output, dir)
}

/// Everything the run wrote, both streams — the `fmt` layer renders to stdout,
/// but a future change of writer must not silently pass this test.
fn console(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

#[test]
fn default_run_surfaces_trainer_warnings() {
    let (out, dir) = train(&[], "default");
    let console = console(&out);
    assert!(
        out.status.success(),
        "train should exit 0; output:\n{console}"
    );
    assert!(
        console.contains(WARN_MARKER),
        "a flagless run must print trainer warnings — the default console \
         filter is WARN, not `EnvFilter::from_default_env()` (which enables \
         nothing when RUST_LOG is unset); output was:\n{console}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn quiet_suppresses_warnings() {
    // The other direction: without this, a filter hard-wired to WARN — which
    // ignores the resolved verbosity just as thoroughly — would also pass the
    // test above.
    let (out, dir) = train(&["-q"], "quiet");
    let console = console(&out);
    assert!(
        out.status.success(),
        "train -q should exit 0; output:\n{console}"
    );
    assert!(
        !console.contains(WARN_MARKER),
        "-q must suppress warnings; output was:\n{console}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
