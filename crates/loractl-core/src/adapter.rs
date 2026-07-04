//! Adapter-only safetensors persistence (milestone 4, #3).
//!
//! [`save_adapter`] / [`load_adapter`] replace milestone 2's
//! `NamedMpkFileRecorder` full-model checkpoints: they persist **only the
//! trainable LoRA factors** — nothing else — in the industry-standard
//! `.safetensors` format, plus a small JSON sidecar that makes the file
//! self-describing enough to reconstruct the rest.
//!
//! ## Tensor naming scheme
//!
//! Only these two tensors are ever written, at their natural burn module
//! path (see [`LORA_TENSOR_REGEX`]):
//!
//! | Path                 | Meaning                                      |
//! |-----------------------|----------------------------------------------|
//! | `fc2.lora_a.weight`    | Down-projection `A`, `hidden -> rank`         |
//! | `fc2.lora_b.weight`    | Up-projection `B`, `rank -> out`              |
//!
//! The frozen base (`fc1.weight`, `fc1.bias`, `fc2.base.weight`,
//! `fc2.base.bias`) is **never** written. That is the entire point of a LoRA
//! "adapter" as distinct from a full model checkpoint — persisting the base
//! would just re-invent milestone 2's `.mpk` checkpoint with extra steps.
//! This mirrors the *pattern* of community LoRA conventions (HF PEFT names
//! its trainable factors `lora_A`/`lora_B`) without claiming literal PEFT
//! interop: `LoraMlp` is a synthetic classifier, not a downloadable public
//! base model, so there is no shared base checkpoint to actually be
//! interoperable *with*.
//!
//! ## The sidecar JSON, and why
//!
//! A LoRA-only safetensors file is not self-describing on its own: reloading
//! it needs the frozen base's *exact* weights back, plus the adapter's shape
//! (rank, alpha, dimensions). `burn-store` 0.21's
//! [`SafetensorsStore::metadata`](burn_store::SafetensorsStore) can *write*
//! custom safetensors `__metadata__` entries, but — verified against the
//! crate source (`burn-store-0.21.0/src/safetensors/store.rs`) — there is
//! **no public API to read that metadata back** after opening a file for
//! loading; the only accessor, `get_metadata`, is a private inherent method
//! used internally by the store itself. So instead of round-tripping through
//! embedded metadata, loractl writes a `<path>.json` sidecar ([`AdapterMeta`])
//! next to the `.safetensors` file, holding exactly what's needed to
//! reconstruct the rest:
//!
//! - `seed` — the training run's RNG seed. Re-seeding the device with this
//!   value reproduces `fc1`/`fc2.base`'s Kaiming init **bit-identically**,
//!   because burn's RNG is deterministic per seed and [`LoraMlp::new`] is
//!   called immediately after seeding, before any other draws (see
//!   `burn_trainer.rs`'s module docs for the exact ordering this depends on).
//! - `rank`/`alpha`/`d_in`/`hidden`/`out` — the exact shape to pass back into
//!   [`LoraMlp::new`].
//!
//! This is the same two-file shape as HF PEFT's own
//! `adapter_model.safetensors` + `adapter_config.json` convention — arguably
//! *more* interoperable in spirit than a custom embedded-metadata scheme,
//! since it's a plain, framework-agnostic JSON file any tool can read without
//! even touching the safetensors library.
//!
//! See `docs/adrs/0002-adapter-format-and-sample-semantics.md` for the full
//! decision record.

use crate::model::LoraMlp;
use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn_store::{ModuleSnapshot, PathFilter, SafetensorsStore};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Regex matching exactly the two trainable LoRA factors — `fc2.lora_a.weight`
/// and `fc2.lora_b.weight` — and nothing else. See the [module docs](self).
const LORA_TENSOR_REGEX: &str = r"\.lora_(a|b)\.weight$";

/// Everything needed to reconstruct a [`LoraMlp`]'s frozen base and adapter
/// shape from an adapter-only safetensors file. Persisted as a `<path>.json`
/// sidecar — see the [module docs](self) for why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterMeta {
    /// The training run's RNG seed; re-seeding the device with this value
    /// reproduces the frozen base (`fc1`, `fc2.base`) bit-identically.
    pub seed: u64,
    /// LoRA rank (`fc2.lora_a`'s output width / `fc2.lora_b`'s input width).
    pub rank: u32,
    /// LoRA alpha (`scaling = alpha / rank`).
    pub alpha: f32,
    /// Model input width (`fc1`'s input width).
    pub d_in: usize,
    /// Hidden width (`fc1`'s output width / `fc2`'s input width).
    pub hidden: usize,
    /// Output width / class count (`fc2`'s output width).
    pub out: usize,
}

/// The sidecar path for a given adapter file: `<path>.json`, appended rather
/// than substituted so `lora.safetensors` maps to `lora.safetensors.json`
/// (using `with_extension` here would replace `.safetensors` and lose the
/// qualifier).
fn sidecar_path(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".json");
    PathBuf::from(os)
}

/// Save `model`'s trainable LoRA factors to `path` (a `.safetensors` file)
/// plus a `<path>.json` sidecar describing how to reconstruct the rest.
///
/// `seed` must be the training run's seed (see the [module docs](self) for
/// why that's what makes the file self-describing). `rank`/`alpha`/the
/// model's dimensions are derived from `model` itself — nothing here is a
/// hardcoded constant.
pub fn save_adapter<B: Backend>(model: &LoraMlp<B>, path: &Path, seed: u64) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating adapter output dir {}", parent.display()))?;
    }

    let rank = model.fc2.lora_a.weight.dims()[1];
    let alpha = (model.fc2.scaling * rank as f64) as f32;
    let d_in = model.fc1.weight.dims()[0];
    let hidden = model.fc1.weight.dims()[1];
    let out = model.fc2.base.weight.dims()[1];

    let filter = PathFilter::new().with_regex(LORA_TENSOR_REGEX);
    let mut store = SafetensorsStore::from_file(path)
        .filter(filter)
        .overwrite(true);
    model
        .save_into(&mut store)
        .with_context(|| format!("saving LoRA adapter to {}", path.display()))?;

    let meta = AdapterMeta {
        seed,
        rank: rank as u32,
        alpha,
        d_in,
        hidden,
        out,
    };
    let meta_json =
        serde_json::to_string_pretty(&meta).context("serializing adapter metadata sidecar")?;
    std::fs::write(sidecar_path(path), meta_json)
        .with_context(|| format!("writing adapter metadata sidecar for {}", path.display()))?;

    Ok(())
}

/// Load a [`LoraMlp`] from an adapter-only safetensors file plus its
/// `<path>.json` sidecar.
///
/// The frozen base is reconstructed bit-identically by re-seeding `device`
/// with the persisted training seed *before* constructing the model — see
/// the [module docs](self). Only `fc2.lora_a.weight` / `fc2.lora_b.weight`
/// are actually read from `path`; the rest of the freshly constructed model
/// (`fc1`, `fc2.base`) is left as the re-seeded init, which is exactly the
/// original frozen base.
pub fn load_adapter<B: Backend>(path: &Path, device: &B::Device) -> Result<LoraMlp<B>> {
    let meta_json = std::fs::read_to_string(sidecar_path(path))
        .with_context(|| format!("reading adapter metadata sidecar for {}", path.display()))?;
    let meta: AdapterMeta =
        serde_json::from_str(&meta_json).context("parsing adapter metadata sidecar")?;

    // Seed FIRST, then construct immediately — mirrors `BurnTrainer::train`'s
    // ordering exactly, which is what makes the frozen base reproducible.
    B::seed(device, meta.seed);
    let mut model = LoraMlp::<B>::new(
        meta.d_in,
        meta.hidden,
        meta.out,
        meta.rank as usize,
        meta.alpha as f64,
        device,
    );

    // `allow_partial`: the frozen base tensors are legitimately absent from
    // an adapter-only file — that's expected, not an error.
    let mut store = SafetensorsStore::from_file(path).allow_partial(true);
    let result = model
        .load_from(&mut store)
        .with_context(|| format!("loading LoRA adapter from {}", path.display()))?;
    anyhow::ensure!(
        result.errors.is_empty(),
        "adapter load at {} produced errors: {:?}",
        path.display(),
        result.errors
    );
    anyhow::ensure!(
        result.applied.len() == 2,
        "expected exactly 2 LoRA tensors applied from {}, got {}: {:?}",
        path.display(),
        result.applied.len(),
        result.applied
    );

    Ok(model)
}
