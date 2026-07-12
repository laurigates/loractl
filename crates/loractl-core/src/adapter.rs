//! Safetensors adapter save/load — milestone 4 (#3), acceptance a & d.
//!
//! Persists and restores [`LoraMlp`]'s trainable LoRA factors as a real
//! `.safetensors` file, replacing the burn-native `.mpk` stopgap from
//! milestone 2. See [`BurnTrainer`](crate::burn_trainer::BurnTrainer) for how
//! a run's checkpoints and final adapter are written, and
//! [`crate::sample`] for what an adapter is used *for* once reloaded.
//!
//! ## Tensor-naming scheme
//!
//! Only the **trainable** LoRA factors are persisted, at their natural burn
//! module path:
//!
//! | Tensor               | Shape            | Trainable |
//! |-----------------------|------------------|-----------|
//! | `fc2.lora_a.weight`   | `[hidden, rank]` | yes       |
//! | `fc2.lora_b.weight`   | `[rank, out]`    | yes       |
//!
//! `fc2.lora_a`/`fc2.lora_b` are bias-less [`Linear`](burn::nn::Linear)s (see
//! [`LoraLinear`](crate::lora::LoraLinear)), so there are no `.bias` tensors
//! to account for. The frozen base (`fc1`, `fc2.base`) is **never**
//! serialized — see "Reconstructing the frozen base" below for why that is
//! sound rather than lossy.
//!
//! This mirrors the *pattern* of community LoRA conventions (Hugging Face
//! PEFT's `lora_A`/`lora_B` naming) without claiming literal interop:
//! `LoraMlp` isn't attached to a downloadable public base model, so there is
//! no PEFT checkpoint to actually be compatible *with* — only the naming
//! pattern and the adapter-only shape are recognizable to anyone who has seen
//! a PEFT adapter on disk.
//!
//! ## A JSON sidecar, not embedded safetensors metadata
//!
//! `burn-store` 0.21's `SafetensorsStore::metadata(key, value)` is
//! **write-only** — there is no public method to read a safetensors file's
//! `__metadata__` header back out after opening it for a load (the crate's
//! own metadata getter is private; see `safetensors/store.rs`'s
//! `get_metadata`). So the information needed to *reconstruct* a model before
//! applying the adapter tensors (`seed`, `rank`, `alpha`, layer widths) is
//! written to a small sidecar file, `<path-with-extension>.json`, instead —
//! reliable, inspectable with any JSON tool, and close in spirit to how HF's
//! PEFT ships `adapter_config.json` alongside `adapter_model.safetensors`.
//!
//! ## Reconstructing the frozen base
//!
//! [`load_adapter`] never reads `fc1`/`fc2.base` from disk. Instead it
//! reseeds burn's global RNG with the training run's original seed
//! (`B::seed(device, meta.seed)`) *before* constructing a fresh [`LoraMlp`] —
//! the same determinism trick [`BurnTrainer`](crate::burn_trainer::BurnTrainer)
//! itself relies on. Because burn's RNG is deterministic given a seed,
//! [`LoraMlp::new`]'s Kaiming-initialized `fc1` and `fc2.base` come out
//! bit-identical to the original run, so the adapter file only needs to carry
//! the two tensors that actually diverged from their initial values.

use crate::config::TaskKind;
use crate::model::LoraMlp;
use anyhow::{Result, ensure};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_store::{ModuleSnapshot, PathFilter, SafetensorsStore};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Sidecar metadata written alongside an adapter's `.safetensors` file.
///
/// Everything [`load_adapter`] needs to reconstruct the frozen base and size
/// the LoRA factors before applying the saved tensors on top.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterMeta {
    /// The training run's RNG seed. Reseeding with this value before
    /// constructing the model regenerates the frozen `fc1`/`fc2.base`
    /// bit-identically to the run that produced this adapter.
    pub seed: u64,
    /// LoRA rank (`fc2.lora_a`'s output width / `fc2.lora_b`'s input width).
    pub rank: u32,
    /// LoRA alpha (`scaling = alpha / rank`).
    pub alpha: f32,
    /// Input width of `fc1`.
    pub d_in: usize,
    /// Hidden width (`fc1`'s output width / `fc2`'s input width).
    pub hidden: usize,
    /// Output width (`fc2`'s output width / number of classes).
    pub out: usize,
    /// Which training task produced this adapter (M8, #19) — a classifier and
    /// a flow-matching velocity net share the `LoraMlp` shape, and downstream
    /// consumers (the `sample` refusal in [`crate::sample::sample_adapter`])
    /// must be able to tell them apart. `#[serde(default)]` (=
    /// `Classification`) keeps every pre-M8 sidecar parsing.
    #[serde(default)]
    pub task: TaskKind,
}

/// The sidecar path for a given adapter path: the whole filename with a
/// literal `.json` appended.
///
/// Deliberately NOT `Path::with_extension`, which would *replace* the
/// `.safetensors` suffix rather than append to it (`lora.safetensors` would
/// become `lora.json`, not `lora.safetensors.json`).
fn sidecar_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".json");
    PathBuf::from(name)
}

/// `true` if every element of `tensor` is finite (not `NaN`/`±Inf`).
///
/// Used to guard both ends of adapter persistence: [`save_adapter`] refuses to
/// write a NaN/Inf-poisoned adapter (e.g. from a training run that diverged
/// under an unstable learning rate), and [`load_adapter`] refuses to hand back
/// a model reconstructed from a corrupted/hand-edited `.safetensors` file —
/// both fail with a clear error at the I/O boundary rather than deferring the
/// failure to whatever later calls `model.forward(...)` (e.g. `run_sample`).
fn all_finite<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> bool {
    tensor.to_data().iter::<f32>().all(|v: f32| v.is_finite())
}

/// Save `model`'s trainable LoRA factors to `path` (a `.safetensors` file)
/// plus a `<path>.json` sidecar describing how to reconstruct the frozen
/// base.
///
/// `seed` MUST be the training run's original RNG seed — [`load_adapter`]
/// needs it to regenerate `fc1`/`fc2.base` bit-identically. `task` records
/// which training objective produced the adapter (see [`AdapterMeta::task`]).
/// Creates `path`'s parent directory if it doesn't exist yet; overwrites an
/// existing file at `path`.
pub fn save_adapter<B: Backend>(
    model: &LoraMlp<B>,
    path: &Path,
    seed: u64,
    task: TaskKind,
) -> Result<()> {
    ensure!(
        all_finite(&model.fc2.lora_a.weight.val()) && all_finite(&model.fc2.lora_b.weight.val()),
        "refusing to save adapter to {} — LoRA weights contain non-finite (NaN/Inf) \
         values, most likely because training diverged",
        path.display()
    );

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| {
            anyhow::anyhow!("creating adapter output dir {}: {e}", parent.display())
        })?;
    }

    // Inclusive filter: `PathFilter::new()` matches nothing by default, and
    // `with_regex` OR's in this one pattern, so only the two LoRA tensors are
    // collected — the frozen base never touches the file.
    let filter = PathFilter::new().with_regex(r"\.lora_(a|b)\.weight$");
    let mut store = SafetensorsStore::from_file(path)
        .filter(filter)
        .overwrite(true);
    model
        .save_into(&mut store)
        .map_err(|e| anyhow::anyhow!("writing adapter to {}: {e}", path.display()))?;

    let rank = model.fc2.lora_a.weight.dims()[1];
    let meta = AdapterMeta {
        seed,
        rank: rank as u32,
        alpha: (model.fc2.scaling * rank as f64) as f32,
        d_in: model.fc1.weight.dims()[0],
        hidden: model.fc1.weight.dims()[1],
        out: model.fc2.base.weight.dims()[1],
        task,
    };
    let json = serde_json::to_string_pretty(&meta)
        .map_err(|e| anyhow::anyhow!("serializing adapter sidecar: {e}"))?;
    let sidecar = sidecar_path(path);
    std::fs::write(&sidecar, json)
        .map_err(|e| anyhow::anyhow!("writing adapter sidecar {}: {e}", sidecar.display()))?;

    Ok(())
}

/// Read the JSON sidecar for the adapter at `path` (without touching the
/// tensor file). Shared by [`load_adapter`] and the task check in
/// [`crate::sample::sample_adapter`].
pub(crate) fn read_meta(path: &Path) -> Result<AdapterMeta> {
    let sidecar = sidecar_path(path);
    let json = std::fs::read_to_string(&sidecar)
        .map_err(|e| anyhow::anyhow!("reading adapter sidecar {}: {e}", sidecar.display()))?;
    serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("parsing adapter sidecar {}: {e}", sidecar.display()))
}

/// Load an adapter previously written by [`save_adapter`]: reconstruct the
/// frozen base deterministically from the sidecar's seed and shape, then
/// apply the two saved LoRA tensors on top.
///
/// # Side effect: this reseeds burn's global RNG
///
/// `load_adapter` calls `B::seed(device, meta.seed)`, which mutates burn's
/// **process-global** per-backend RNG. That is not an accident and it is not
/// a leak to be tidied away — **it is the mechanism** by which the frozen base
/// is reproduced (see "Reconstructing the frozen base" in the [module
/// docs](self)). The adapter file deliberately carries only the two LoRA
/// tensors; `fc1`/`fc2.base` are regenerated by replaying the training run's
/// seeded RNG through [`LoraMlp::new`]. Reseeding is what makes that replay
/// land on the same values, and it must happen *before* construction because
/// burn's `Param` draws its random init on first access
/// (`.claude/rules/burn-lazy-param-init.md`).
///
/// Consequences, so the next reader doesn't "fix" this:
///
/// - Any RNG stream in flight on the same backend+device is **clobbered** by a
///   load. Callers that care must load before they start drawing.
/// - Tests that touch the global RNG around a load serialize on an `RNG_LOCK`
///   mutex (`tests/adapter_roundtrip.rs`, `tests/multi_adapter_step.rs`,
///   `tests/flow_convergence.rs`) — cargo runs a file's tests as threads in one
///   process, so the reseed is genuinely shared state there.
/// - Threading an explicit seed/RNG through [`LoraMlp::new`] would remove the
///   global write, but it is a real API change rippling through
///   [`crate::adapters`], [`crate::burn_trainer`], and [`crate::sample`] — for a
///   problem that only bites tests, which already handle it. Reviewed and
///   **deliberately kept** (#49 P2).
pub fn load_adapter<B: Backend>(path: &Path, device: &B::Device) -> Result<LoraMlp<B>> {
    let meta = read_meta(path)?;

    // Reseed BEFORE constructing the model so the freshly initialized
    // `fc1`/`fc2.base` come out bit-identical to the training run that
    // produced this adapter — see the module docs' "Reconstructing the
    // frozen base" section.
    B::seed(device, meta.seed);
    let mut model = LoraMlp::<B>::new(
        meta.d_in,
        meta.hidden,
        meta.out,
        meta.rank as usize,
        meta.alpha as f64,
        // Dropout is identity at inference (non-autodiff backend), and dropout
        // prob is not persisted in the sidecar, so reload with 0.0. A reloaded
        // adapter is used for sampling; continued training is not yet wired.
        0.0,
        device,
    );

    let mut store = SafetensorsStore::from_file(path).allow_partial(true);
    let result = model
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("loading adapter from {}: {e}", path.display()))?;
    ensure!(
        result.errors.is_empty(),
        "adapter load errors: {:?}",
        result.errors
    );
    // Exactly the 2 LoRA tensors are expected — NOT `result.missing.is_empty()`:
    // the 4 frozen-base tensors are legitimately absent from this adapter-only
    // file (see ADR-0002). Asserting `missing` empty here would be the wrong
    // field, the same documented footgun ADR-0001 calls out for `unused`.
    ensure!(
        result.applied.len() == 2,
        "expected exactly 2 applied LoRA tensors, got {}: {:?}",
        result.applied.len(),
        result.applied
    );
    ensure!(
        all_finite(&model.fc2.lora_a.weight.val()) && all_finite(&model.fc2.lora_b.weight.val()),
        "adapter at {} contains non-finite (NaN/Inf) LoRA weights — the file may be corrupted",
        path.display()
    );

    Ok(model)
}
