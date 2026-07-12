//! Confinement of client-supplied output paths to a server-owned base dir (#37).
//!
//! `POST /runs` takes an unauthenticated `TrainConfig` whose `output.dir` /
//! `output.name` are turned straight into filesystem paths by the trainer
//! (`create_dir_all`, then `.safetensors`/`.json` writes). Left unvalidated
//! that is an arbitrary-file-write primitive: `"dir": "../../.."`, or any
//! absolute path, writes wherever the process can reach.
//!
//! So the request never names a path — it names a location *inside* a base
//! directory the operator chose (`LORACTL_OUTPUT_BASE`). [`confine_output`]
//! resolves the request's relative path against that base and returns the
//! absolute path the trainer may use, or an error the handler renders as a
//! `400`. The rule it enforces:
//!
//! 1. `output.dir` is **relative** and contains **no `..` component**. This is
//!    a check over `Path::components()`, not a substring scan — a substring
//!    scan both misses (`a/../../b` normalizes differently per-platform) and
//!    over-rejects (a legitimate `..foo` directory name).
//! 2. `output.name` is a single plain filename component: no separators, no
//!    `..`, not empty. It is joined onto the dir by core, so it escapes just
//!    as well as the dir does.
//! 3. The joined path, once resolved, is still **under the base** — the
//!    belt-and-braces check that catches any normalization gap in (1).
//! 4. **No symlink escapes**: the deepest already-existing ancestor of the
//!    target is canonicalized (which resolves symlinks) and must *still* be
//!    under the canonical base. Plain `canonicalize()` on the target itself
//!    cannot do this job — it fails outright on a path that does not exist
//!    yet, which is the normal case for a fresh run's output dir.

use std::path::{Component, Path, PathBuf};

/// Resolves a request's `output.dir` / `output.name` to an absolute path
/// confined under `base`, which MUST already be canonicalized (see
/// [`canonical_base`]).
///
/// Returns the directory the run may write into, or a client-facing message
/// explaining the rejection.
pub fn confine_output(base: &Path, dir: &Path, name: &str) -> Result<PathBuf, String> {
    validate_name(name)?;
    let relative = relative_components(dir)?;
    let resolved = base.join(relative);

    // (3) Component-wise containment — `starts_with` compares path components,
    // so it cannot be fooled the way a string prefix can (`/base-evil`).
    if !resolved.starts_with(base) {
        return Err(escape_message());
    }

    // (4) Symlink escape: canonicalize the deepest ancestor that exists. A
    // symlinked component anywhere along the path resolves here, so a
    // pre-planted `base/link -> /etc` is caught even though `base/link/x`
    // does not exist yet.
    let anchor = deepest_existing_ancestor(&resolved);
    let canonical_anchor = anchor
        .canonicalize()
        .map_err(|e| format!("resolving output.dir {}: {e}", dir.display()))?;
    if !canonical_anchor.starts_with(base) {
        return Err(escape_message());
    }

    Ok(resolved)
}

/// Creates the output base if absent and canonicalizes it, so every later
/// containment check compares two symlink-free absolute paths.
///
/// Done once at startup: a base that cannot be created is a misconfigured
/// server, which should fail loudly on boot rather than 500 per request.
pub fn canonical_base(base: &Path) -> anyhow::Result<PathBuf> {
    use anyhow::Context;
    std::fs::create_dir_all(base)
        .with_context(|| format!("creating output base {}", base.display()))?;
    base.canonicalize()
        .with_context(|| format!("resolving output base {}", base.display()))
}

fn escape_message() -> String {
    String::from("output.dir must stay within the server's output base directory")
}

/// (1) Rejects absolute paths and `..` components, returning the path's
/// normalized relative form (`.` segments dropped).
fn relative_components(dir: &Path) -> Result<PathBuf, String> {
    let mut relative = PathBuf::new();
    for component in dir.components() {
        match component {
            Component::Normal(part) => relative.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(String::from("output.dir must not contain `..` components"));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(String::from(
                    "output.dir must be a relative path, not an absolute one",
                ));
            }
        }
    }
    Ok(relative)
}

/// (2) `output.name` is joined onto the dir and given an extension by core, so
/// it must be one plain filename component.
fn validate_name(name: &str) -> Result<(), String> {
    let mut components = Path::new(name).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(()),
        _ => Err(String::from(
            "output.name must be a plain file name: no path separators, no `..`, not empty",
        )),
    }
}

/// The deepest ancestor of `path` that exists on disk.
///
/// `symlink_metadata` (not `exists()`) so a **dangling** symlink counts as
/// existing: it is then canonicalized, which fails, and the request is
/// rejected — where `exists()` would have skipped past it to a safe parent and
/// waved the escape through.
///
/// Terminates: `base` itself is an ancestor of every `resolved` we pass here
/// and is created at startup.
fn deepest_existing_ancestor(path: &Path) -> PathBuf {
    path.ancestors()
        .find(|ancestor| std::fs::symlink_metadata(ancestor).is_ok())
        .unwrap_or(path)
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "loractl-paths-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        canonical_base(&dir).expect("temp base")
    }

    #[test]
    fn relative_dir_resolves_under_the_base() {
        let base = base();
        let resolved = confine_output(&base, Path::new("run-1/out"), "lora").expect("accepted");
        assert_eq!(resolved, base.join("run-1").join("out"));
        assert!(resolved.starts_with(&base));
    }

    #[test]
    fn curdir_segments_are_normalized_away() {
        let base = base();
        let resolved = confine_output(&base, Path::new("./a/./b"), "lora").expect("accepted");
        assert_eq!(resolved, base.join("a").join("b"));
    }

    #[test]
    fn parent_dir_components_are_rejected() {
        let base = base();
        for dir in ["..", "../../etc", "a/../../..", "a/../b"] {
            let error = confine_output(&base, Path::new(dir), "lora")
                .expect_err("`..` must be rejected regardless of position");
            assert!(
                error.contains(".."),
                "unexpected message for {dir}: {error}"
            );
        }
    }

    #[test]
    fn a_dir_name_merely_starting_with_dots_is_not_a_parent_ref() {
        // The component check must not degenerate into a substring scan: a
        // directory literally named `..foo` is legal and stays under the base.
        let base = base();
        let resolved = confine_output(&base, Path::new("..foo"), "lora").expect("accepted");
        assert_eq!(resolved, base.join("..foo"));
    }

    #[test]
    fn absolute_dirs_are_rejected() {
        let base = base();
        let error = confine_output(&base, Path::new("/etc/loractl"), "lora").expect_err("absolute");
        assert!(error.contains("relative"), "unexpected message: {error}");
    }

    #[test]
    fn names_that_are_not_plain_components_are_rejected() {
        let base = base();
        for name in ["../evil", "a/b", "/abs", "..", ""] {
            confine_output(&base, Path::new("out"), name)
                .expect_err("output.name must be a single plain component");
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_dirs_pointing_outside_the_base_are_rejected() {
        let base = base();
        let outside = std::env::temp_dir().join(format!("loractl-outside-{}", std::process::id()));
        std::fs::create_dir_all(&outside).expect("outside dir");
        let link = base.join("escape-hatch");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&outside, &link).expect("symlink");

        // Lexically `base/escape-hatch/x` is under the base; only resolving
        // the symlink reveals that it is not.
        let error = confine_output(&base, Path::new("escape-hatch/x"), "lora")
            .expect_err("symlink escape must be rejected");
        assert!(error.contains("output base"), "unexpected message: {error}");

        let _ = std::fs::remove_file(&link);
        let _ = std::fs::remove_dir_all(&outside);
    }
}
