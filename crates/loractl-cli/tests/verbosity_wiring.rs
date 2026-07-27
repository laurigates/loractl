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
    train_with_env(args, tag, &[])
}

fn train_with_env(
    args: &[&str],
    tag: &str,
    extra_env: &[(&str, &str)],
) -> (Output, std::path::PathBuf) {
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
    let mut command = Command::new(env!("CARGO_BIN_EXE_loractl"));
    command
        .args(args)
        .args(["train", config, "--steps", "1"])
        // The harness may inherit `RUST_LOG`; this test is about the flag path.
        .env_remove("RUST_LOG")
        // No `--output-dir` flag exists, so steer the writes via the env layer.
        .env("LORACTL_OUTPUT__DIR", &dir);
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let output = command.output().expect("run `loractl train`");
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

/// `-v` must actually raise the level for **loractl's own** events.
///
/// `flag_directives` emits the target `loractl`, which is the `[[bin]]` name
/// (what `module_path!` roots at) — not the package name `loractl-cli`. An
/// external review argued the directive therefore matches nothing and `-v` is
/// inert; it is wrong, but nothing in the suite could settle it, because the
/// unit test only asserts the directive *string* and never that the filter
/// matches a real event. This pair does: a run with `-v` must show the INFO
/// checkpoint line, and the same run without `-v` must not.
///
/// A checkpoint every step is the synthetic path's only INFO — the WARN marker
/// would pass at either level and prove nothing.
const INFO_MARKER: &str = "checkpoint";
const CHECKPOINT_EVERY_STEP: (&str, &str) = ("LORACTL_OUTPUT__CHECKPOINT_EVERY", "1");

#[test]
fn verbose_raises_the_level_for_loractl_targets() {
    let (out, dir) = train_with_env(&["-v"], "verbose", &[CHECKPOINT_EVERY_STEP]);
    let console = console(&out);
    assert!(
        out.status.success(),
        "train -v should exit 0; output:\n{console}"
    );
    assert!(
        console.contains(INFO_MARKER),
        "-v must surface loractl's own INFO events — if this fails, the \
         `loractl=` target in `flag_directives` no longer matches the binary's \
         module path and the whole verbosity ladder is inert; output was:\n{console}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn default_run_hides_info() {
    // The kill-test for the pair above: without this, a filter stuck at INFO
    // (or TRACE) would satisfy `verbose_raises_the_level_for_loractl_targets`
    // while `-v` did nothing at all.
    let (out, dir) = train_with_env(&[], "default-info", &[CHECKPOINT_EVERY_STEP]);
    let console = console(&out);
    assert!(
        !console.contains(INFO_MARKER),
        "a flagless run must stay at WARN and hide INFO; output was:\n{console}"
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
