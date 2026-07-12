//! Architectural gate: `loractl-core` emits events, never renders (issue #48).
//!
//! The project's load-bearing invariant is that core reports progress only
//! through the `&mut dyn FnMut(TrainEvent)` sink — it never writes to
//! stdout/stderr and never imports `clap`. That is what makes "a GUI can be
//! built separately over the same core" real. Until now the rule lived only in
//! doc comments; a stray `println!("loss={loss}")` in the training loop would
//! compile and pass the whole suite.
//!
//! This is a deterministic grep gate (per the offload-to-a-substrate
//! principle): it scans `loractl-core/src/**` for rendering tokens on
//! non-comment lines and fails with the exact `file:line` if any appear. Prose
//! mentions in doc comments (e.g. lib.rs's "…never `println!`s") are excluded
//! because they start with `//`.

use std::fs;
use std::path::Path;

/// Tokens that indicate rendering to stdout/stderr — forbidden anywhere in
/// core's own code (front-ends render; core emits `TrainEvent`s).
const FORBIDDEN: &[&str] = &[
    "println!",
    "print!",
    "eprintln!",
    "eprint!",
    "dbg!",
    "io::stdout",
    "io::stderr",
];

#[test]
fn core_never_renders_to_stdout_or_stderr() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut violations = Vec::new();
    scan(&src, &mut violations);
    assert!(
        violations.is_empty(),
        "loractl-core must not render — it emits TrainEvents through the sink; \
         front-ends (CLI/API) render. Found rendering tokens:\n{}",
        violations.join("\n")
    );
}

/// Also pin the sibling half of the invariant: core must not depend on `clap`.
#[test]
fn core_never_imports_clap() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut violations = Vec::new();
    for path in rs_files(&src) {
        let text = fs::read_to_string(&path).expect("read source");
        for (i, line) in text.lines().enumerate() {
            let code = code_part(line);
            if code.contains("use clap") || code.contains("clap::") {
                violations.push(format!("{}:{}", path.display(), i + 1));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "loractl-core must not import clap (the CLI owns argument parsing). \
         Found:\n{}",
        violations.join("\n")
    );
}

/// Returns the code portion of a line: empty for a full-line comment, else the
/// text before any inline `//` comment. Core has no `://` in code lines, so the
/// naive split is safe and keeps doc-comment prose (which mentions these
/// tokens deliberately) from tripping the gate.
fn code_part(line: &str) -> &str {
    if line.trim_start().starts_with("//") {
        return "";
    }
    line.split("//").next().unwrap_or(line)
}

fn scan(dir: &Path, out: &mut Vec<String>) {
    for path in rs_files(dir) {
        let text = fs::read_to_string(&path).expect("read source");
        for (i, line) in text.lines().enumerate() {
            let code = code_part(line);
            if let Some(pat) = FORBIDDEN.iter().find(|p| code.contains(**p)) {
                out.push(format!("{}:{}: contains `{}`", path.display(), i + 1, pat));
            }
        }
    }
}

fn rs_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).expect("read src dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            files.extend(rs_files(&path));
        } else if path.extension().is_some_and(|e| e == "rs") {
            files.push(path);
        }
    }
    files
}
