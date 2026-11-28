//! Confinement of client-supplied paths to server-owned base dirs: output
//! paths (#37) and `resume.from` (#180) under `LORACTL_OUTPUT_BASE`, dataset
//! and model paths under `LORACTL_INPUT_BASE` (#224, see [`confine_inputs`]).
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

use loractl_core::TrainConfig;
use std::path::{Component, Path, PathBuf};

/// Resolves a request's `output.dir` / `output.name` to an absolute path
/// confined under `base`, which MUST already be canonicalized (see
/// [`canonical_base`]).
///
/// Returns the directory the run may write into, or a client-facing message
/// explaining the rejection.
pub fn confine_output(base: &Path, dir: &Path, name: &str) -> Result<PathBuf, String> {
    validate_name(name)?;
    let relative = relative_components(dir, "output.dir")?;
    contain(base, base.join(relative), dir, "output.dir", "output")
}

/// (3) + (4): `resolved` must be under `base` component-wise and, once the
/// deepest existing ancestor's symlinks are resolved, still under it.
fn contain(
    base: &Path,
    resolved: PathBuf,
    requested: &Path,
    field: &str,
    base_name: &str,
) -> Result<PathBuf, String> {
    let escape = || format!("{field} must stay within the server's {base_name} base directory");
    // (3) Component-wise containment — `starts_with` compares path components,
    // so it cannot be fooled the way a string prefix can (`/base-evil`).
    if !resolved.starts_with(base) {
        return Err(escape());
    }
    // (4) Symlink escape: canonicalize the deepest ancestor that exists. A
    // symlinked component anywhere along the path resolves here, so a
    // pre-planted `base/link -> /etc` is caught even though `base/link/x`
    // does not exist yet.
    let anchor = deepest_existing_ancestor(&resolved);
    let canonical_anchor = anchor
        .canonicalize()
        .map_err(|e| format!("resolving {field} {}: {e}", requested.display()))?;
    if !canonical_anchor.starts_with(base) {
        return Err(escape());
    }
    Ok(resolved)
}

/// Creates a base directory (output or input) if absent and canonicalizes it,
/// so every later containment check compares two symlink-free absolute paths.
///
/// Done once at startup: a base that cannot be created is a misconfigured
/// server, which should fail loudly on boot rather than 500 per request.
pub fn canonical_base(base: &Path) -> anyhow::Result<PathBuf> {
    use anyhow::Context;
    std::fs::create_dir_all(base)
        .with_context(|| format!("creating base directory {}", base.display()))?;
    base.canonicalize()
        .with_context(|| format!("resolving base directory {}", base.display()))
}

/// (1) Rejects absolute paths and `..` components, returning the path's
/// normalized relative form (`.` segments dropped). `field` names the request
/// field in the client-facing message.
fn relative_components(path: &Path, field: &str) -> Result<PathBuf, String> {
    let mut relative = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => relative.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(format!("{field} must not contain `..` components"));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!(
                    "{field} must be a relative path, not an absolute one"
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

/// Resolves a request's `resume.from` under `base`, with the same rules
/// [`confine_output`] applies to `output.dir` (#180).
///
/// `resume.from` is the *read* twin of `output.dir`: an unauthenticated
/// `POST /runs` names a file the server then opens and parses, and the parse
/// errors describe tensor names and shapes. Without this it is an
/// arbitrary-file-read oracle over anything the process can reach — a hole in
/// a boundary that is otherwise sealed. `output.name`'s single-component rule
/// deliberately does NOT apply: resuming `checkpoints/run-3/checkpoint-50.safetensors`
/// from a sibling run's directory is the recovery workflow this exists for.
pub fn confine_resume(base: &Path, from: &Path) -> Result<PathBuf, String> {
    confine_read(base, from, "resume.from", "output")
}

/// Resolves a client-supplied **read** path under `base`: relative, no `..`,
/// not empty, and no symlink escape. The shared rule behind `resume.from` and
/// the input fields of [`confine_inputs`].
fn confine_read(base: &Path, path: &Path, field: &str, base_name: &str) -> Result<PathBuf, String> {
    let relative = relative_components(path, field)?;
    if relative.as_os_str().is_empty() {
        return Err(format!("{field} must name a path, not an empty one"));
    }
    contain(base, base.join(relative), path, field, base_name)
}

/// Resolves every client-supplied path the trainer **reads** under the input
/// base (`LORACTL_INPUT_BASE`, #224): `dataset.path`, `model.base` (unless it
/// is a built-in demo name), and the `model.*` file overrides.
///
/// Same rules and same reason as `resume.from`: the endpoint is
/// unauthenticated by default, and load errors stream back over SSE, so an
/// unconfined read path is an existence and parse-error oracle over anything
/// the process can reach. Models and datasets get their own base because they
/// legitimately live outside the output tree.
///
/// Core joins a relative `model.*` override onto `model.base`; that meaning is
/// kept, so an override is confined as `model.base/override` and handed to the
/// trainer resolved. `model.checkpoint` is a file name core joins onto the
/// base, so it is validated the same way and left as given. An empty
/// `dataset.path` names nothing to read and is left alone.
///
/// On success every rewritten field holds an absolute path under `base`.
pub fn confine_inputs(base: &Path, config: &mut TrainConfig) -> Result<(), String> {
    const BASE: &str = "input";
    if !config.dataset.path.as_os_str().is_empty() {
        config.dataset.path = confine_read(base, &config.dataset.path, "dataset.path", BASE)?;
    }

    let model = &mut config.model;
    let root = if loractl_core::is_builtin_demo_base(&model.base) {
        PathBuf::new()
    } else {
        let requested = PathBuf::from(&model.base);
        let resolved = confine_read(base, &requested, "model.base", BASE)?;
        model.base = resolved
            .to_str()
            .ok_or_else(|| String::from("model.base must resolve to a UTF-8 path"))?
            .to_owned();
        requested
    };
    if let Some(checkpoint) = &model.checkpoint {
        confine_read(base, &root.join(checkpoint), "model.checkpoint", BASE)?;
    }
    for (field, slot) in [
        ("model.denoiser", &mut model.denoiser),
        ("model.text_encoder", &mut model.text_encoder),
        ("model.vae", &mut model.vae),
        ("model.tokenizer", &mut model.tokenizer),
        ("model.training_adapter", &mut model.training_adapter),
    ] {
        if let Some(path) = slot.as_deref() {
            *slot = Some(confine_read(base, &root.join(path), field, BASE)?);
        }
    }
    Ok(())
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

    /// `resume.from` is a *read* path but the same confinement applies: a
    /// request that could name `/etc/shadow` would get its parse error back as
    /// a 400 body. Mirrors `absolute_dir_is_rejected` / `parent_traversal_is_rejected`.
    #[test]
    fn resume_from_is_confined_like_the_output_dir() {
        let base = base();
        let resolved =
            confine_resume(&base, Path::new("run-1/checkpoint-50.safetensors")).expect("accepted");
        assert_eq!(resolved, base.join("run-1/checkpoint-50.safetensors"));

        for bad in ["/etc/shadow", "../../../etc/shadow", ".."] {
            let error = confine_resume(&base, Path::new(bad))
                .expect_err("{bad} must be rejected as a resume source");
            assert!(error.contains("resume.from"), "{bad}: {error}");
        }
        assert!(confine_resume(&base, Path::new("")).is_err());
    }

    // ---- #224: dataset and model paths are reads, confined like resume.from --

    fn diffusion_config() -> loractl_core::TrainConfig {
        let mut config = loractl_core::TrainConfig::default();
        config.model.base = "models/krea2".into();
        config.dataset.path = "datasets/fox".into();
        config
    }

    #[test]
    fn dataset_and_model_paths_resolve_under_the_input_base() {
        let base = base();
        let mut config = diffusion_config();
        config.model.denoiser = Some("raw-fp8.safetensors".into());
        config.model.text_encoder = Some("te/model.safetensors".into());
        config.model.vae = Some("vae.safetensors".into());
        config.model.tokenizer = Some("tok/tokenizer.json".into());
        config.model.training_adapter = Some("assist.safetensors".into());
        config.model.checkpoint = Some("turbo.safetensors".into());
        confine_inputs(&base, &mut config).expect("all relative, all inside");

        assert_eq!(config.dataset.path, base.join("datasets/fox"));
        let root = base.join("models/krea2");
        assert_eq!(config.model.base, root.to_str().unwrap());
        // Relative overrides keep core's meaning (joined onto `model.base`),
        // and are handed to the trainer already resolved.
        assert_eq!(
            config.model.denoiser,
            Some(root.join("raw-fp8.safetensors"))
        );
        assert_eq!(
            config.model.text_encoder,
            Some(root.join("te/model.safetensors"))
        );
        assert_eq!(config.model.vae, Some(root.join("vae.safetensors")));
        assert_eq!(
            config.model.tokenizer,
            Some(root.join("tok/tokenizer.json"))
        );
        assert_eq!(
            config.model.training_adapter,
            Some(root.join("assist.safetensors"))
        );
        assert_eq!(
            config.model.checkpoint.as_deref(),
            Some("turbo.safetensors")
        );
    }

    #[test]
    fn every_input_field_rejects_absolute_and_parent_paths() {
        let base = base();
        type Set = fn(&mut loractl_core::TrainConfig, &str);
        let fields: [(&str, Set); 8] = [
            ("dataset.path", |c, v| c.dataset.path = v.into()),
            ("model.base", |c, v| c.model.base = v.into()),
            ("model.denoiser", |c, v| c.model.denoiser = Some(v.into())),
            ("model.text_encoder", |c, v| {
                c.model.text_encoder = Some(v.into())
            }),
            ("model.vae", |c, v| c.model.vae = Some(v.into())),
            ("model.tokenizer", |c, v| c.model.tokenizer = Some(v.into())),
            ("model.training_adapter", |c, v| {
                c.model.training_adapter = Some(v.into())
            }),
            ("model.checkpoint", |c, v| {
                c.model.checkpoint = Some(v.into())
            }),
        ];
        for (field, set) in fields {
            for bad in ["/etc/shadow", "../../etc/shadow", "a/../../b"] {
                let mut config = diffusion_config();
                set(&mut config, bad);
                let error = confine_inputs(&base, &mut config)
                    .expect_err(&format!("{field} = {bad} must be rejected"));
                assert!(error.contains(field), "{field} = {bad}: {error}");
            }
        }
    }

    #[test]
    fn builtin_demo_bases_are_not_paths() {
        let base = base();
        for demo in ["synthetic", "mnist"] {
            let mut config = diffusion_config();
            config.model.base = demo.into();
            confine_inputs(&base, &mut config).expect("a demo base names no file");
            assert_eq!(config.model.base, demo);
        }
        // An empty dataset path names nothing to read and is left alone.
        let mut config = diffusion_config();
        config.model.base = "synthetic".into();
        config.dataset.path = PathBuf::new();
        confine_inputs(&base, &mut config).expect("empty dataset path");
        assert_eq!(config.dataset.path, PathBuf::new());
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_dataset_escaping_the_input_base_is_rejected() {
        let base = base();
        let link = base.join("escape-dataset");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink("/etc", &link).unwrap();
        let mut config = diffusion_config();
        config.dataset.path = "escape-dataset".into();
        let error = confine_inputs(&base, &mut config).expect_err("symlink out of the base");
        assert!(error.contains("dataset.path"), "{error}");
        let _ = std::fs::remove_file(&link);
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
