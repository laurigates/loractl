//! The end-to-end Krea 2 LoRA trainer (M14, #25) — the payoff the M6–M13
//! chain exists to reach.
//!
//! [`DiffusionTrainer`] is a second [`Trainer`] implementation composing the
//! whole stack: the M12 dataset pipeline caches M9 VAE latents and M10
//! conditioner stacks once; the training loop then drives the M11 MMDiT with
//! the M8 rectified-flow objective (`x_t = (1−t)·x₀ + t·ε`, target
//! `v = ε − x₀`, logit-normal + shifted timesteps), stepping **only** the M6
//! LoRA adapters injected across the trunk. Checkpoints and the final
//! artifact are the M6 **kohya-ss export** — ComfyUI-loadable at every save
//! point. The M13 memory knobs (`compute.precision: f16`,
//! `compute.grad_checkpointing`) apply through the same dispatch as
//! [`BurnTrainer`](crate::BurnTrainer).
//!
//! ## Checkpoint directory layout ([`ModelConfig::base`])
//!
//! The `krea/Krea-2-Raw` HF snapshot layout, consumed as-is:
//!
//! ```text
//! <base>/raw.safetensors                          the MMDiT (variant default; see below)
//! <base>/text_encoder/model.safetensors           Qwen3-VL (vision tower dropped at load)
//! <base>/tokenizer/tokenizer.json                 the Qwen tokenizer
//! <base>/vae/diffusion_pytorch_model.safetensors  the Qwen-Image VAE
//! ```
//!
//! The denoiser filename is the variant default — `raw.safetensors` for
//! `krea2`/`tiny-krea2`, `turbo.safetensors` for `krea2-turbo` — unless
//! `model.checkpoint` names another file within `base` (M15, #82); see
//! [`denoiser_filename`]. Scaled-fp8 repacks are auto-detected from the
//! file header and load through [`load_fp8_module`].
//!
//! Each component can also live **outside** `base` via the
//! `model.{denoiser,text_encoder,vae,tokenizer}` path overrides — so a
//! ComfyUI install's scattered `models/{diffusion_models,text_encoders,vae}/…`
//! layout works with no restructuring, duplicate files, or symlinks (an
//! absolute override is used verbatim, a relative one joins onto `base`; see
//! [`resolve_component`]).
//!
//! [`ModelVariant`] names the architecture explicitly (`krea2` |
//! `krea2-turbo` | `tiny-krea2`) — a config mistake is a clear error, never
//! a creative shape inference.
//!
//! ## Memory sequencing & encode precision
//!
//! The VAE and text encoder run only during dataset preparation (everything
//! they produce is cached by M12), so they are loaded, used, and **dropped
//! before the MMDiT loads** — peak memory holds either the encoders or the
//! denoiser, never both.
//!
//! The encode phase **always runs on the CPU ndarray backend in f32,
//! regardless of `compute.backend`/`compute.precision`** — the exact path
//! the M9/M10 parity tests pin against diffusers/transformers. Two observed
//! failure modes force this:
//!
//! - **f16 overflow**: the frozen Qwen-family encoders exceed f16's numeric
//!   range on the real weights (activation overflow turned every cached
//!   latent and conditioning tensor Inf/NaN);
//! - **wgpu f32 corruption**: burn 0.21's wgpu kernels progressively
//!   corrupted *sequential* encoder outputs (the first caption encoded
//!   clean, later identical calls degraded to ~1e32 magnitudes and then
//!   NaN — a buffer-reuse/kernel bug beneath this crate, not model math).
//!
//! The encoders never benefit from the GPU knobs anyway — they run once,
//! alone, are cached, and are dropped; the f16 knob exists to fit the
//! *MMDiT*. The cache fingerprint carries an `enc32` marker so caches
//! written by the earlier reduced-precision encode path are invalidated
//! rather than silently reused.
//!
//! Like every trainer, this emits [`TrainEvent`]s through the sink and never
//! renders; the front-ends changed only at their constructor seam (a
//! two-armed factory on `model.base`).

use crate::adapters::{LoraAdapters, build_adapters};
use crate::config::{
    BackendKind, ModelConfig, ModelVariant, Precision, Quant, TaskKind, TrainConfig,
};
use crate::dataset::prepare_dataset;
use crate::event::TrainEvent;
use crate::export::{ExportFormat, export_adapters, import_adapters};
use crate::flow::{interpolate, sample_timesteps, velocity_target};
use crate::mmdit::{BaseLinear, Mmdit, MmditConfig, krea2_positions, patchify};
use crate::quant::{QuantBackend, quantize_linear_weight};
use crate::qwen_vae::{QwenVae, QwenVaeConfig};
use crate::qwen3vl::{Qwen3VlConditioner, Qwen3VlConfig, Qwen3VlEncoder};
use crate::train::Trainer;
use anyhow::{Context, Result, anyhow, bail};
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, Param};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{DType, Distribution, Element, ElementConversion, Tensor};
use burn_store::{
    KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    TensorSnapshot,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;

#[cfg(feature = "cuda")]
use burn::backend::{Cuda, cuda::CudaDevice};

/// The per-variant architecture bundle: (MMDiT, encoder, VAE, caption budget).
fn variant_configs(variant: ModelVariant) -> (MmditConfig, Qwen3VlConfig, QwenVaeConfig, usize) {
    match variant {
        // Turbo is architecturally identical to Krea2 (same 430 tensor
        // keys); the variants differ only in the default denoiser filename.
        ModelVariant::Krea2 | ModelVariant::Krea2Turbo => (
            MmditConfig::krea2(),
            Qwen3VlConfig::krea2_4b(),
            QwenVaeConfig::qwen_image(),
            512,
        ),
        ModelVariant::TinyKrea2 => (
            MmditConfig::tiny_krea2(),
            Qwen3VlConfig::tiny(),
            QwenVaeConfig::tiny(),
            16,
        ),
    }
}

/// The denoiser filename inside [`ModelConfig::base`]: an explicit
/// `model.checkpoint` always wins; otherwise the variant default
/// (`raw.safetensors` for Krea2/TinyKrea2, `turbo.safetensors` for
/// Krea2Turbo).
pub fn denoiser_filename(model: &ModelConfig) -> &str {
    match &model.checkpoint {
        Some(name) => name.as_str(),
        None => match model.variant {
            ModelVariant::Krea2 | ModelVariant::TinyKrea2 => "raw.safetensors",
            ModelVariant::Krea2Turbo => "turbo.safetensors",
        },
    }
}

/// Resolve a component path from an optional override against the base dir.
///
/// The seam that makes a scattered ComfyUI layout work with no restructuring:
/// an **absolute** override is used verbatim (point straight at
/// `…/ComfyUI/models/vae/…`), a **relative** override joins onto `base` (so a
/// `base` of the ComfyUI `models/` root plus `vae/qwen/…` reads cleanly), and
/// **no** override falls back to the historical `base/<default_rel>` layout —
/// so every pre-existing snapshot-dir config resolves byte-identically.
fn resolve_component(override_path: Option<&Path>, base: &Path, default_rel: &str) -> PathBuf {
    match override_path {
        Some(p) if p.is_absolute() => p.to_path_buf(),
        Some(p) => base.join(p),
        None => base.join(default_rel),
    }
}

/// The denoiser path: the full-path [`denoiser`](ModelConfig::denoiser)
/// override wins (absolute verbatim, relative onto `base`); otherwise
/// `base/<denoiser_filename>` (which still honors `checkpoint`).
fn denoiser_path(model: &ModelConfig, base: &Path) -> PathBuf {
    match &model.denoiser {
        Some(p) if p.is_absolute() => p.clone(),
        Some(p) => base.join(p),
        None => base.join(denoiser_filename(model)),
    }
}

/// The text-encoder path — override or `base/text_encoder/model.safetensors`.
fn text_encoder_path(model: &ModelConfig, base: &Path) -> PathBuf {
    resolve_component(
        model.text_encoder.as_deref(),
        base,
        "text_encoder/model.safetensors",
    )
}

/// The VAE path — override or `base/vae/diffusion_pytorch_model.safetensors`.
fn vae_path(model: &ModelConfig, base: &Path) -> PathBuf {
    resolve_component(
        model.vae.as_deref(),
        base,
        "vae/diffusion_pytorch_model.safetensors",
    )
}

/// The tokenizer path — override or `base/tokenizer/tokenizer.json`. (Fetching
/// the model-invariant Qwen3-VL tokenizer when neither exists is a later
/// milestone; today an absent file surfaces as a clear load error.)
fn tokenizer_path(model: &ModelConfig, base: &Path) -> PathBuf {
    resolve_component(model.tokenizer.as_deref(), base, "tokenizer/tokenizer.json")
}

/// Dynamic loss scaling (see the optimizer step). The initial factor lifts
/// f16 gradients out of underflow across the 28-block backward; on
/// non-finite gradients the step is skipped and the scale halves, and after
/// [`SCALE_GROWTH_INTERVAL`] consecutive clean steps it doubles again —
/// the standard mixed-precision sawtooth. Both directions were observed on
/// the real model: at S=1 gradients underflow to exactly zero (adapters
/// never train), and a static S=16384 overflowed around step 13 as the
/// growing LoRA path amplified early-layer gradients past f16's max.
const INITIAL_LOSS_SCALE: f32 = 16384.0;
/// Ceiling/floor for the dynamic scale.
const MAX_LOSS_SCALE: f32 = 65536.0;
/// See [`INITIAL_LOSS_SCALE`].
const MIN_LOSS_SCALE: f32 = 1.0;
/// Clean steps before the scale doubles.
const SCALE_GROWTH_INTERVAL: u32 = 50;

/// The cache fingerprint for a variant's encoder outputs. The `enc32`
/// marker records that the encode phase ran in f32 — caches produced by the
/// earlier (numerically broken) reduced-precision encode path carry the old
/// unmarked fingerprint and are invalidated by this rename. Turbo maps onto
/// the `krea2` fingerprint: it shares Krea2's encoders (the same files in
/// the same `base`), so their caches are interchangeable — and the emitted
/// strings stay byte-identical to the pre-M15 `{variant:?}`-derived form,
/// keeping existing caches valid.
pub fn encoder_fingerprint(variant: ModelVariant, max_length: usize) -> String {
    let arch = match variant {
        ModelVariant::Krea2 | ModelVariant::Krea2Turbo => "krea2",
        ModelVariant::TinyKrea2 => "tinykrea2",
    };
    format!("{arch}-ml{max_length}-enc32")
}

/// The Krea 2 LoRA trainer — see the [module docs](self).
pub struct DiffusionTrainer;

impl Trainer for DiffusionTrainer {
    fn train(&mut self, config: &TrainConfig, sink: &mut dyn FnMut(TrainEvent)) -> Result<PathBuf> {
        // Validate before any backend work, identically on every backend.
        if config.task != TaskKind::FlowMatching {
            bail!(
                "the diffusion trainer trains the rectified-flow objective only; \
                 set task: flow-matching (got {:?})",
                config.task
            );
        }
        if config.output.sample_every > 0 {
            bail!(
                "output.sample_every is classification-specific; the diffusion \
                 trainer has no sample path — set it to 0"
            );
        }
        if config.lora.targets.is_empty() {
            bail!(
                "lora.targets is empty — nothing would train. Add at least one \
                 pattern, e.g. `targets: [{{ pattern: \"blocks\\\\.\" }}]` to adapt \
                 every MMDiT trunk projection"
            );
        }

        // Started frames the whole run, encode phase included, so SSE/bar
        // consumers see the run begin before the (potentially long) one-time
        // dataset encode rather than after it.
        sink(TrainEvent::Started {
            total_steps: config.steps.max(1),
        });
        std::fs::create_dir_all(&config.output.dir)
            .with_context(|| format!("creating output dir {}", config.output.dir.display()))?;

        // Frozen-base int8 quantization (#96) is validated only on the two
        // numerically-clean f32 paths — ndarray (the offline/CI path) and cuda
        // (the real 24 GB run) — because burn's int8 q-ops need those backends
        // and its non-f32 autodiff is broken on GPU (burn#5162). Gate it here,
        // BEFORE the backend/precision match below, so the quant-specific
        // message wins over the more generic ones (e.g. wgpu+int8 says "use
        // f16", not "wgpu not built"). Every illegal combo fails loudly — never
        // a silent full-precision or wrong-backend load. Compiled always (a
        // pure config check), like the rest of this validation.
        if config.compute.quant == Quant::Int8 {
            match config.compute.backend {
                BackendKind::Ndarray | BackendKind::Cuda => {}
                BackendKind::Wgpu => bail!(
                    "int8 base quantization is validated on cuda and ndarray; wgpu is untested — \
                     use compute.precision: f16 for wgpu memory savings"
                ),
                BackendKind::Candle | BackendKind::Tch => bail!(
                    "compute.quant = int8 is not supported on the {:?} backend (no quantized \
                     matmul q-ops); use ndarray (offline/CI) or cuda (the 24 GB real run)",
                    config.compute.backend
                ),
            }
            if config.compute.precision != Precision::F32 {
                bail!(
                    "quantization dequantizes to f32; set compute.precision to f32 \
                     (got compute.precision = {:?} with compute.quant = int8)",
                    config.compute.precision
                );
            }
        }

        // Backend validity first, so a misconfigured run fails before the
        // (potentially long) encode phase rather than after it.
        match (config.compute.backend, config.compute.precision) {
            (BackendKind::Ndarray, p) if p != Precision::F32 => bail!(
                "compute.precision = {p:?} is not supported on the ndarray backend \
                 (f16 needs wgpu, bf16 needs candle)"
            ),
            (BackendKind::Wgpu, Precision::Bf16) => bail!(
                "compute.precision = bf16 is only supported on the candle backend \
                 (selected backend: wgpu — burn-wgpu has no bf16 support)"
            ),
            (BackendKind::Candle, Precision::F16) => bail!(
                "compute.precision = f16 on candle is not wired; use bf16 (same \
                 memory, f32-like range) or f32"
            ),
            #[cfg(not(feature = "wgpu"))]
            (BackendKind::Wgpu, _) => bail!(
                "config selected the 'wgpu' backend but this binary was built without it; \
                 rebuild with `--features wgpu`"
            ),
            #[cfg(not(feature = "candle"))]
            (BackendKind::Candle, _) => bail!(
                "config selected the 'candle' backend but this binary was built without it; \
                 rebuild with `--features candle`"
            ),
            #[cfg(not(feature = "cuda"))]
            (BackendKind::Cuda, _) => bail!(
                "config selected the 'cuda' backend but this binary was built without it; \
                 rebuild with `--features cuda` on a Linux+NVIDIA host (CUDA toolkit \
                 required). cuda is not runnable on macOS"
            ),
            // cuda is wired f32-only: burn's f16 autodiff produces exactly-zero
            // adapter gradients on cuda — the same defect as Metal, validated
            // on the RTX 4090 (tracel-ai/burn#5162, examples/grad_compare.rs).
            #[cfg(feature = "cuda")]
            (BackendKind::Cuda, p) if p != Precision::F32 => bail!(
                "compute.precision = {p:?} is not supported on the cuda backend — burn's \
                 non-f32 autodiff is broken on cuda (exactly-zero adapter gradients, \
                 tracel-ai/burn#5162); set compute.precision to f32"
            ),
            (BackendKind::Tch, _) => bail!(
                "the diffusion trainer currently wires ndarray, wgpu, candle, and cuda; \
                 tch lands once it can be verified on real hardware"
            ),
            _ => {}
        }

        // The one-time dataset encode ALWAYS runs on the CPU ndarray backend
        // in f32 — the parity-proven encoder path. See the module docs: f16
        // overflows the Qwen encoders' range, and burn 0.21's wgpu f32
        // kernels corrupted sequential encoder outputs progressively (clean
        // → ~1e32 magnitudes → NaN across identical calls). The cache makes
        // this a one-time cost per dataset.
        encode_phase::<NdArray>(config, Default::default())?;

        match config.compute.backend {
            BackendKind::Ndarray => {
                dispatch_checkpointing::<NdArray>(config, Default::default(), sink)
            }
            #[cfg(feature = "wgpu")]
            BackendKind::Wgpu => match config.compute.precision {
                Precision::F32 => dispatch_checkpointing::<Wgpu>(config, Default::default(), sink),
                Precision::F16 => dispatch_checkpointing::<Wgpu<burn::tensor::f16>>(
                    config,
                    Default::default(),
                    sink,
                ),
                Precision::Bf16 => unreachable!("validated above"),
            },
            // burn deprecates burn-candle in favor of burn-cubecl — but
            // cubecl's Metal kernels are precisely what produces the NaN
            // gradients this arm exists to dodge (examples/grad_compare.rs:
            // candle-metal bf16 matches CPU ground truth where both wgpu
            // arms NaN). Revisit when a burn release fixes wgpu autodiff on
            // Apple Silicon.
            #[cfg(feature = "candle")]
            #[allow(deprecated)]
            BackendKind::Candle => {
                let device = burn::backend::candle::CandleDevice::metal(config.compute.device);
                match config.compute.precision {
                    Precision::F32 => {
                        dispatch_checkpointing::<burn::backend::Candle>(config, device, sink)
                    }
                    Precision::Bf16 => dispatch_checkpointing::<
                        burn::backend::Candle<burn::tensor::bf16>,
                    >(config, device, sink),
                    Precision::F16 => unreachable!("validated above"),
                }
            }
            // f32-only by the guard above (burn#5162); like candle, the
            // device is constructed explicitly from the config's ordinal.
            // The first fully-clean GPU configuration (grad ratio 1.00 vs
            // ndarray at every adapter site — PR #94's 4090 validation).
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => {
                dispatch_checkpointing::<Cuda>(config, CudaDevice::new(config.compute.device), sink)
            }
            _ => unreachable!("backend validated above"),
        }
    }
}

/// The one-time dataset encode: load the frozen encoders on an **f32**
/// backend, run the M12 cache pass, and drop everything — the cache on disk
/// is the only output. The training phase re-reads it on its own backend
/// and never touches the encoders (its cache-miss closures bail).
///
/// Encoder loading is **lazy** — the closures load a model on their first
/// cache miss. A fully warm cache therefore never loads the encoders at
/// all (on the real model that skips a ~16 GB f32 text-encoder load per
/// warm rerun), pinned by the e2e's encoders-deleted warm-rerun test.
fn encode_phase<B: burn::tensor::backend::Backend>(
    config: &TrainConfig,
    device: B::Device,
) -> Result<()> {
    let base = PathBuf::from(&config.model.base);
    let (_, enc_cfg, vae_cfg, max_length) = variant_configs(config.model.variant);
    let fingerprint = encoder_fingerprint(config.model.variant, max_length);

    let mut vae: Option<QwenVae<B>> = None;
    let mut conditioner: Option<Qwen3VlConditioner<B>> = None;
    prepare_dataset::<B>(
        &config.dataset,
        &fingerprint,
        &device,
        |image| {
            if vae.is_none() {
                vae = Some(
                    load_module(
                        QwenVae::<B>::init(vae_cfg.clone(), &device),
                        &vae_path(&config.model, &base),
                        &QwenVae::<B>::key_remap(),
                        false,
                        None,
                        "VAE",
                    )?
                    .no_grad(),
                );
            }
            Ok(vae.as_ref().expect("just initialized").encode(image))
        },
        |caption| {
            if conditioner.is_none() {
                let encoder = load_module(
                    Qwen3VlEncoder::<B>::init(enc_cfg.clone(), &device),
                    &text_encoder_path(&config.model, &base),
                    &[],
                    true,
                    Some(Qwen3VlEncoder::<B>::load_filter()),
                    "text encoder",
                )?
                .no_grad();
                conditioner = Some(Qwen3VlConditioner::new(
                    encoder,
                    &tokenizer_path(&config.model, &base),
                    max_length,
                )?);
            }
            conditioner
                .as_ref()
                .expect("just initialized")
                .encode_captions(&[caption], &device)
        },
    )?;
    Ok(())
}

/// The M13 checkpointing split, mirroring `BurnTrainer`'s.
fn dispatch_checkpointing<B: burn::tensor::backend::Backend>(
    config: &TrainConfig,
    device: B::Device,
    sink: &mut dyn FnMut(TrainEvent),
) -> Result<PathBuf> {
    use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    if config.compute.grad_checkpointing {
        sink(TrainEvent::Warning {
            message: "activation checkpointing enabled (BalancedCheckpointing): \
                      lower memory, slower steps, identical numerics"
                .into(),
        });
        run_diffusion::<Autodiff<B, BalancedCheckpointing>>(config, device, sink)
    } else {
        run_diffusion::<Autodiff<B>>(config, device, sink)
    }
}

/// A burn-store adapter that casts every float snapshot to `target` before
/// it is applied.
///
/// burn-store's applier deliberately **preserves the file's dtype** on the
/// created param tensor (`Tensor::from_data(data, (device, snapshot.dtype))`).
/// On a backend whose working dtype differs — `Wgpu<f16>` loading a stock
/// Krea-2-Raw checkpoint (174 F32 + 256 BF16 tensors) — that leaves params
/// in dtypes the backend either doesn't support at all (burn-wgpu asserts
/// `!supports_dtype(BF16)`) or mis-executes in mixed-dtype kernels: observed
/// on the real model as `first.forward` saturating to f16-max from O(2.6)
/// inputs and O(0.6) file weights, NaN'ing the whole forward. burn-store's
/// own `HalfPrecisionAdapter` can't close this (it passes BF16 through and
/// only covers a fixed module-type list), so this adapter converts every
/// float tensor unconditionally.
#[derive(Clone)]
pub struct CastFloatsAdapter {
    /// The dtype every float snapshot is converted to (the loading module's
    /// working float dtype).
    pub target: DType,
}

impl ModuleAdapter for CastFloatsAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let is_float = matches!(
            snapshot.dtype,
            DType::F64 | DType::F32 | DType::Flex32 | DType::F16 | DType::BF16
        );
        if !is_float || snapshot.dtype == self.target {
            return snapshot.clone();
        }
        let original = snapshot.clone_data_fn();
        let target = self.target;
        TensorSnapshot::from_closure(
            Rc::new(move || Ok(original()?.convert_dtype(target))),
            target,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Load a component checkpoint into a freshly-initialized module.
fn load_module<B: burn::tensor::backend::Backend, M: ModuleSnapshot<B>>(
    mut module: M,
    path: &Path,
    remap: &[(&str, &str)],
    transpose_linears: bool,
    filter: Option<&str>,
    what: &str,
) -> Result<M> {
    let remapper = KeyRemapper::from_patterns(remap.to_vec())
        .unwrap_or_else(|e| panic!("invalid {what} remap patterns: {e}"));
    // Skip enum-variant path segments: `Mmdit`'s `BaseLinear` sites are an
    // enum in the module tree, and burn-store would otherwise inject the
    // active variant name (`Plain`/`Quant`) into every key path
    // (`blocks.0.attn.wq.Plain.weight`), so the checkpoint's
    // `blocks.0.attn.wq.weight` would not match. Inert for enum-free modules
    // (the VAE/text encoder) — they contain no `Enum:` container.
    let mut store = SafetensorsStore::from_file(path)
        .remap(remapper)
        .skip_enum_variants(true);
    if let Some(pattern) = filter {
        store = store.with_regex(pattern).allow_partial(true);
    }
    // Every float tensor is cast to the backend's working float dtype — see
    // [`CastFloatsAdapter`]: a stock checkpoint's F32/BF16 tensors must not
    // survive as-is into an f16 module's params.
    let cast = CastFloatsAdapter {
        target: <B::FloatElem as Element>::dtype(),
    };
    store = if transpose_linears {
        store.with_from_adapter(PyTorchToBurnAdapter.chain(cast))
    } else {
        store.with_from_adapter(cast)
    };
    let result = module
        .load_from(&mut store)
        .with_context(|| format!("loading {what} from {}", path.display()))?;
    if !result.errors.is_empty() {
        bail!(
            "{what} load errors from {}: {:?}",
            path.display(),
            result.errors
        );
    }
    if !result.missing.is_empty() {
        bail!(
            "{what} at {} is missing parameters: {:?}",
            path.display(),
            result.missing
        );
    }
    Ok(module)
}

/// Load a scaled-fp8 checkpoint into a freshly-initialized module — the fp8
/// twin of [`load_module`] (M15, #82). [`crate::fp8::load_fp8_snapshots`]
/// supplies the lazily-dequantizing f32 snapshots; the remap, the
/// `PyTorchToBurnAdapter` linear transpose, [`CastFloatsAdapter`], and
/// `module.apply` are the exact machinery `SafetensorsStore::apply_to`
/// drives on the bf16/f32 path, so both loads land byte-equivalent params.
///
/// Unlike [`load_module`], `unused` is a hard error: after fp8 weights and
/// their consumed `*_scale` sidecars are accounted for, leftover tensors
/// (e.g. an fp8mixed repack's baked-in LoRA) mean the file is not a clean
/// repack of the model this config names. The guard is sound only while
/// [`Mmdit`] contains no burn normalization modules: the Applier's
/// alternative-param-name lookup (serving burn LayerNorm/RmsNorm's
/// `weight`→`gamma` rename) deliberately does not mark the file-side key
/// visited, so adopting those modules would land their keys in `unused` and
/// false-positive this error. Mmdit's norms are custom `ZRmsNorm` with
/// param `scale`, which loads by name.
pub fn load_fp8_module<B: burn::tensor::backend::Backend, M: ModuleSnapshot<B>>(
    mut module: M,
    path: &Path,
    remap: &[(&str, &str)],
    what: &str,
) -> Result<M> {
    let snapshots = crate::fp8::load_fp8_snapshots(path)
        .with_context(|| format!("loading {what} (scaled fp8) from {}", path.display()))?;
    let remapper = KeyRemapper::from_patterns(remap.to_vec())
        .unwrap_or_else(|e| panic!("invalid {what} remap patterns: {e}"));
    let (snapshots, _) = remapper.remap(snapshots);
    // Same adapter chain as [`load_module`]'s transpose_linears=true arm.
    let cast = CastFloatsAdapter {
        target: <B::FloatElem as Element>::dtype(),
    };
    let adapter: Box<dyn ModuleAdapter> = Box::new(PyTorchToBurnAdapter.chain(cast));
    // `skip_enum_variants = true`: `Mmdit`'s `BaseLinear` enum sites must not
    // inject a variant name into key paths — see [`load_module`].
    let result = module.apply(snapshots, None, Some(adapter), true);
    if !result.errors.is_empty() {
        bail!(
            "{what} load errors from {}: {:?}",
            path.display(),
            result.errors
        );
    }
    if !result.missing.is_empty() {
        bail!(
            "{what} at {} is missing parameters: {:?}",
            path.display(),
            result.missing
        );
    }
    if !result.unused.is_empty() {
        bail!(
            "{what} at {} contains tensors the model does not consume \
             (not a clean scaled-fp8 repack of this architecture): {:?}",
            path.display(),
            result.unused
        );
    }
    Ok(module)
}

/// Load a checkpoint into an **already-[`into_quantized`](Mmdit::into_quantized)**
/// MMDiT, quantizing each frozen base weight from the file **one tensor at a
/// time** — the int8 twin of [`load_module`] / [`load_fp8_module`] (PR-B3, #96).
///
/// ## Memory discipline (the whole point)
///
/// The full f32 MMDiT is ~49 GB; this loader must **never** materialize it. It
/// receives a module whose `Quant` sites already hold int8 placeholders (the
/// ~14 GB skeleton) and overwrites every tensor through two lazy, streaming
/// seams — peak ≈ the int8 skeleton + ONE transient f32 layer weight:
///
/// - **Base-linear WEIGHT keys** go through the per-tensor quant pass: force
///   the file snapshot to a single transient f32 `[d_out, d_in]` tensor,
///   quantize it to int8, replace the placeholder, drop the f32. The full-model
///   f32 tensors are never collected together.
/// - **Everything else** (norms, modulations, first/last/projector, the base
///   linears' biases, and any site `into_quantized` left `Plain` on an
///   unaligned `d_in`) flows through the store applier over the SAME lazy
///   snapshots — one tensor materialized at a time, exactly as
///   [`load_fp8_module`].
///
/// Snapshots are auto-detected scaled-fp8 (dequantized to f32 lazily) or plain
/// safetensors, then remapped with [`Mmdit::key_remap`].
///
/// ## Weight orientation
///
/// A PyTorch `Linear` checkpoint weight is `[d_out, d_in]` (file layout) —
/// exactly what [`quantize_linear_weight`] consumes and what
/// [`Mmdit::into_quantized`] produces (it transposes burn's stored
/// `[d_in, d_out]` back to `[d_out, d_in]` before quantizing). So a raw base
/// weight is quantized **without a transpose** here; only the applier's `Plain`
/// path applies `PyTorchToBurnAdapter`'s transpose (to reach burn's storage).
///
/// ## Completeness (no silent partial load)
///
/// Every `Quant` site MUST find its weight in the checkpoint (else bail), the
/// applier must have no `unused` file tensors and no *genuine* `missing` param.
/// The applier legitimately reports each `Quant` weight as `missing` — it
/// visits the QFloat `Param<Tensor>` via `map_float` but its snapshot was
/// partitioned into the quant pass — so "missing" is filtered against the set
/// of weights the quant pass actually overwrote; anything left is a real gap.
// `pub` so the on-box memory/quality probe (`examples/quant_probe.rs`) loads the
// real base through the EXACT path the trainer uses — the VRAM and dequant-error
// numbers it reports then describe production behavior, not a replica that could
// drift. Not part of the crate's stable surface; it moves with the trainer.
pub fn load_quant_module<B: burn::tensor::backend::Backend>(
    mut module: Mmdit<B>,
    path: &Path,
    remap: &[(&str, &str)],
    what: &str,
    device: &B::Device,
    sink: &mut dyn FnMut(TrainEvent),
) -> Result<Mmdit<B>> {
    // 1. Lazy per-tensor snapshots (fp8 auto-detected from the header), remapped
    //    to module paths. Each materializes its tensor only when forced.
    let snapshots = if crate::fp8::is_fp8_checkpoint(path)? {
        crate::fp8::load_fp8_snapshots(path)
            .with_context(|| format!("loading {what} (scaled fp8) from {}", path.display()))?
    } else {
        crate::fp8::load_plain_snapshots(path)
            .with_context(|| format!("loading {what} from {}", path.display()))?
    };
    let remapper = KeyRemapper::from_patterns(remap.to_vec())
        .unwrap_or_else(|e| panic!("invalid {what} remap patterns: {e}"));
    let (snapshots, _) = remapper.remap(snapshots);
    let mut by_key: HashMap<String, TensorSnapshot> =
        snapshots.into_iter().map(|s| (s.full_path(), s)).collect();

    // 2. Quant pass, ONE tensor at a time: overwrite each Quant site's int8
    //    weight from the checkpoint. A site left Plain by into_quantized keeps
    //    its weight key in `by_key` for the applier (step 4).
    let mut quantized = 0usize;
    let mut left_plain = 0usize;
    let mut covered: HashSet<String> = HashSet::new();
    for (path_key, base) in module.all_base_linears_mut() {
        let weight_key = format!("{path_key}.weight");
        match base {
            BaseLinear::Quant(q) => {
                let snapshot = by_key.remove(&weight_key).ok_or_else(|| {
                    anyhow!(
                        "{what}: quantized base site {path_key} has no {weight_key} in the \
                         checkpoint {}",
                        path.display()
                    )
                })?;
                // Force ONE transient f32 [d_out, d_in]; quantize; drop it.
                let data = snapshot
                    .to_data()
                    .map_err(|e| anyhow!("forcing {weight_key} from {}: {e:?}", path.display()))?
                    .convert_dtype(DType::F32);
                let w = Tensor::<B, 2>::from_data(data, device);
                q.weight = Param::from_tensor(quantize_linear_weight(w));
                quantized += 1;
                covered.insert(weight_key);
            }
            // Unaligned d_in (tiny fixtures only): load the f32 weight via the
            // applier — leave its key in `by_key`.
            BaseLinear::Plain(_) => left_plain += 1,
        }
    }

    // 3. Loud accounting (review): a surprise misalignment on the real model —
    //    where every base linear is block-aligned and should quantize — must be
    //    visible here, not a silent OOM later.
    sink(TrainEvent::Warning {
        message: format!(
            "{what}: int8-quantized {quantized} frozen-base linear sites, {left_plain} left \
             full-precision (unaligned d_in)"
        ),
    });

    // 4. Applier over the REST — norms, modulations, first/last/projector, base
    //    biases, and any Plain base weight. Same adapter chain as load_module's
    //    transpose_linears=true / load_fp8_module path; `skip_enum_variants` so
    //    the BaseLinear enum sites don't inject `Plain`/`Quant` into key paths.
    let cast = CastFloatsAdapter {
        target: <B::FloatElem as Element>::dtype(),
    };
    let adapter: Box<dyn ModuleAdapter> = Box::new(PyTorchToBurnAdapter.chain(cast));
    let rest: Vec<TensorSnapshot> = by_key.into_values().collect();
    let result = module.apply(rest, None, Some(adapter), true);

    if !result.errors.is_empty() {
        bail!(
            "{what} load errors from {}: {:?}",
            path.display(),
            result.errors
        );
    }
    if !result.unused.is_empty() {
        bail!(
            "{what} at {} contains tensors the model does not consume: {:?}",
            path.display(),
            result.unused
        );
    }
    // Every quantized Quant weight is expected in `missing` (its QFloat param is
    // visited but its snapshot went to the quant pass). A GENUINE missing param
    // is any `missing` entry not covered by the quant pass — no silent gap.
    let real_missing: Vec<&(String, String)> = result
        .missing
        .iter()
        .filter(|(p, _)| !covered.contains(p))
        .collect();
    if !real_missing.is_empty() {
        bail!(
            "{what} at {} is missing parameters: {:?}",
            path.display(),
            real_missing
        );
    }
    Ok(module)
}

/// One training job on backend `AB`: re-read the cache [`encode_phase`]
/// populated, load the MMDiT, and train the injected LoRA.
///
/// `AB: QuantBackend` so the MMDiT forward can route through
/// [`BaseLinear`](crate::mmdit::BaseLinear) sites; every backend
/// [`dispatch_checkpointing`] instantiates satisfies it via the blanket
/// `Autodiff<B, C>: QuantBackend` impl.
fn run_diffusion<AB: AutodiffBackend + QuantBackend>(
    config: &TrainConfig,
    device: AB::Device,
    sink: &mut dyn FnMut(TrainEvent),
) -> Result<PathBuf> {
    AB::seed(&device, config.seed);

    let total = config.steps.max(1);
    let base = PathBuf::from(&config.model.base);
    let variant = config.model.variant;
    let (mmdit_cfg, _, _, max_length) = variant_configs(variant);
    let patch = mmdit_cfg.patch;

    // ---- Phase 1: read the cache the f32 encode phase just wrote. The
    // closures only fire on a cache miss, which after `encode_phase` means
    // the dataset changed mid-run — bail rather than re-encoding at the
    // training precision (f16 encoders are exactly the bug this split
    // exists to prevent).
    let fingerprint = encoder_fingerprint(variant, max_length);
    let prepared = prepare_dataset::<AB>(
        &config.dataset,
        &fingerprint,
        &device,
        |_| bail!("latent cache miss after the encode phase — did the dataset change mid-run?"),
        |_| {
            bail!(
                "conditioning cache miss after the encode phase — did the dataset change mid-run?"
            )
        },
    )?;
    let batches = prepared.batches(config.dataset.batch_size.max(1) as usize);
    if batches.is_empty() {
        bail!("the dataset produced no batches");
    }

    // ---- Phase 2: the denoiser + adapters. ----
    let denoiser = denoiser_path(&config.model, &base);
    let mmdit = if config.compute.quant == Quant::Int8 {
        // int8 frozen base (#96). Build + load + quantize on the NON-autodiff
        // INNER backend, at the TARGET device directly — NOT default+to_device:
        // a lifted QFloat module cannot `.to_device()` (`Autodiff::q_to_device`
        // is unimplemented), and `into_quantized` requires a non-autodiff
        // backend (burn 0.21's `Autodiff::quantize_dynamic` is `todo!()`).
        // `load_quant_module` streams the checkpoint into the int8 skeleton one
        // tensor at a time (never the ~49 GB f32 model); `from_inner` then lifts
        // it into the autodiff backend the training loop steps.
        let inner = Mmdit::<AB::InnerBackend>::init(mmdit_cfg, &device).into_quantized(&device);
        let inner = load_quant_module(
            inner,
            &denoiser,
            &Mmdit::<AB::InnerBackend>::key_remap(),
            "MMDiT",
            &device,
            sink,
        )?;
        <Mmdit<AB> as AutodiffModule<AB>>::from_inner(inner).no_grad()
    } else {
        // Non-quant path (unchanged): load on the backend's DEFAULT device, then
        // move to the target device. For ndarray/wgpu the default IS the target
        // (a no-op move); for candle the default is the CPU, which sidesteps a
        // load-time double allocation on Metal (the freshly-initialized params
        // plus the store's replacement tensors peaking at ~2× the ~25 GB model —
        // observed as "Failed to create metal resource: Buffer"). `to_device`
        // then migrates tensor by tensor inside unified memory, so peak stays
        // ~one model.
        let init = Mmdit::<AB>::init(mmdit_cfg, &AB::Device::default());
        // Scaled-fp8 repacks are auto-detected from the file header, never from
        // the variant or the filename — a bf16 official checkpoint and a local
        // fp8 repack both route to the right loader under any name.
        if crate::fp8::is_fp8_checkpoint(&denoiser)? {
            load_fp8_module(init, &denoiser, &Mmdit::<AB>::key_remap(), "MMDiT")?
        } else {
            load_module(
                init,
                &denoiser,
                &Mmdit::<AB>::key_remap(),
                true,
                None,
                "MMDiT",
            )?
        }
        .to_device(&device)
        .no_grad()
    };

    let sites = mmdit.injectable_sites();
    let mut set = build_adapters::<AB>(&sites, &config.lora, &device);
    if set.deltas.is_empty() {
        bail!(
            "no injectable site matched lora.targets — the MMDiT advertises \
             paths like blocks.0.attn.wq; a pattern such as \"blocks\\\\.\" \
             matches all {} sites",
            sites.len()
        );
    }

    let adapter_path = config
        .output
        .dir
        .join(format!("{}.safetensors", config.output.name));

    // Resume: an existing final artifact is loaded back into the fresh
    // adapters and training continues from it — running the same config
    // again extends the adapter rather than restarting it. (The export
    // carries no optimizer state, so AdamW re-warms its moments.)
    if adapter_path.exists() {
        let loaded = import_adapters(&mut set, ExportFormat::Krea2Diffusers, &adapter_path)
            .with_context(|| format!("resuming from {}", adapter_path.display()))?;
        sink(TrainEvent::Warning {
            message: format!(
                "resuming from existing adapter {} ({loaded} deltas loaded)",
                adapter_path.display()
            ),
        });
    }

    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.optim.weight_decay as f32)
        .init::<AB, LoraAdapters<AB>>();
    let mut loss_scale = INITIAL_LOSS_SCALE;
    let mut clean_streak = 0u32;
    let checkpoint_every = config.output.checkpoint_every.max(1);

    for step in 1..=total {
        let batch = &batches[((step - 1) as usize) % batches.len()];
        let [b, z, h, w] = batch.latents.dims();
        let flat = z * h * w;

        // The M8 objective over the cached latents.
        let t = sample_timesteps::<AB>(b, config.flow, &device);
        let eps = Tensor::<AB, 4>::random([b, z, h, w], Distribution::Normal(0.0, 1.0), &device);
        let x0 = batch.latents.clone();
        let xt = interpolate(
            x0.clone().reshape([b, flat]),
            eps.clone().reshape([b, flat]),
            t.clone(),
        )
        .reshape([b, z, h, w]);
        let target = patchify(
            velocity_target(x0.reshape([b, flat]), eps.reshape([b, flat])).reshape([b, z, h, w]),
            patch,
        );

        // Token layout: text first, image tokens on the patch grid.
        let img_tokens = patchify(xt, patch);
        let (gh, gw) = (h / patch, w / patch);
        let txt_len = batch.conditioning.dims()[1];
        let pos = krea2_positions::<AB>(txt_len, gh, gw, b, &device);
        let mask = Tensor::cat(
            vec![
                batch.mask.clone().float(),
                Tensor::ones([b, gh * gw], &device),
            ],
            1,
        );

        let pred =
            mmdit.forward_with_adapters(img_tokens, batch.conditioning.clone(), t, pos, mask, &set);

        let diff = pred - target;
        let loss = diff.clone().mul(diff).mean();
        let loss_value: f32 = loss.clone().into_scalar().elem();
        // Fail fast on numeric divergence: a non-finite loss poisons the
        // adapters within one optimizer step, and silently "training" NaNs
        // for the remaining steps only wastes hours and exports garbage
        // (observed with f16 on the real 12B before the f32 encode split).
        if !loss_value.is_finite() {
            bail!(
                "non-finite loss ({loss_value}) at step {step} — numeric overflow. \
                 With compute.precision: f16 this means an activation exceeded \
                 f16's range; try f32, or report the model/config combination"
            );
        }
        sink(TrainEvent::Step {
            step,
            loss: loss_value,
            lr: config.optim.lr,
        });

        // Loss scaling: backprop S·loss instead of loss. In f16 the
        // per-element loss gradient starts at 2·diff/N (~1e-4 at 256²) and
        // shrinks through 28 blocks, underflowing f16's normal range (6e-5)
        // to EXACTLY zero — observed on the real model as `lora_up` never
        // moving off its zero init while the loss stayed healthy. AdamW's
        // update is scale-invariant (m̂ and √v̂ both carry S, which cancels),
        // so scaling needs no un-scaling step and is a numeric no-op on f32
        // backends — it purely keeps f16 gradients representable.
        let scaled = loss * loss_scale;
        let grads = GradientsParams::from_grads(scaled.backward(), &set);

        // Dynamic-scale guard: one reduced scalar over every adapter
        // gradient (Inf/NaN propagate through the abs-sum). Non-finite ⇒
        // the scaled backward overflowed f16 — skip this update, halve the
        // scale, and continue; the loss itself was finite, so the run is
        // healthy. Identity-cost on f32 backends (always finite).
        let grads_finite = if std::env::var_os("LORACTL_SKIP_GRAD_CHECK").is_some() {
            true
        } else {
            let mut acc: Option<Tensor<AB::InnerBackend, 1>> = None;
            for delta in &set.deltas {
                for id in [delta.lora_a.weight.id, delta.lora_b.weight.id] {
                    if let Some(g) = grads.get::<AB::InnerBackend, 2>(id) {
                        let s = g.abs().sum();
                        acc = Some(match acc {
                            Some(a) => a + s,
                            None => s,
                        });
                    }
                }
            }
            acc.map(|a| a.into_scalar().elem::<f32>().is_finite())
                .unwrap_or(true)
        };

        if grads_finite {
            set = optim.step(config.optim.lr, set, grads);
            clean_streak += 1;
            if clean_streak >= SCALE_GROWTH_INTERVAL && loss_scale < MAX_LOSS_SCALE {
                loss_scale = (loss_scale * 2.0).min(MAX_LOSS_SCALE);
                clean_streak = 0;
            }
        } else {
            clean_streak = 0;
            loss_scale = (loss_scale / 2.0).max(MIN_LOSS_SCALE);
            sink(TrainEvent::Warning {
                message: format!(
                    "step {step}: non-finite f16 gradients — update skipped, \
                     loss scale halved to {loss_scale}"
                ),
            });
        }

        if step % checkpoint_every == 0 && step != total {
            let path = config
                .output
                .dir
                .join(format!("checkpoint-{step}.safetensors"));
            export_adapters(&set, ExportFormat::Krea2Diffusers, &path)
                .with_context(|| format!("writing checkpoint at step {step}"))?;
            sink(TrainEvent::Checkpoint { step, path });
        }
    }

    // The final artifact IS the interop artifact: a kohya-ss export.
    export_adapters(&set, ExportFormat::Krea2Diffusers, &adapter_path)
        .context("writing the final adapter export")?;
    sink(TrainEvent::Finished {
        adapter_path: adapter_path.clone(),
    });
    Ok(adapter_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelVariant};

    fn model(
        base: &str,
        denoiser: Option<&str>,
        text_encoder: Option<&str>,
        vae: Option<&str>,
        tokenizer: Option<&str>,
    ) -> ModelConfig {
        ModelConfig {
            base: base.into(),
            variant: ModelVariant::Krea2,
            checkpoint: None,
            denoiser: denoiser.map(PathBuf::from),
            text_encoder: text_encoder.map(PathBuf::from),
            vae: vae.map(PathBuf::from),
            tokenizer: tokenizer.map(PathBuf::from),
        }
    }

    /// No override → the historical `base/<default>` layout, byte-identical to
    /// before this feature (the back-compat guarantee).
    #[test]
    fn no_override_uses_the_base_dir_layout() {
        let base = Path::new("/models/krea2-raw");
        let m = model("/models/krea2-raw", None, None, None, None);
        assert_eq!(
            vae_path(&m, base),
            base.join("vae/diffusion_pytorch_model.safetensors")
        );
        assert_eq!(
            text_encoder_path(&m, base),
            base.join("text_encoder/model.safetensors")
        );
        assert_eq!(
            tokenizer_path(&m, base),
            base.join("tokenizer/tokenizer.json")
        );
        assert_eq!(denoiser_path(&m, base), base.join("raw.safetensors"));
    }

    /// An **absolute** override is used verbatim — the scattered-ComfyUI case
    /// (each component in its own `models/<kind>/…` tree).
    #[test]
    fn absolute_overrides_are_verbatim() {
        let base = Path::new("/anything");
        let m = model(
            "/anything",
            Some("/comfy/models/diffusion_models/krea2/raw_fp8.safetensors"),
            Some("/comfy/models/text_encoders/qwen/enc_fp8.safetensors"),
            Some("/comfy/models/vae/qwen/vae.safetensors"),
            Some("/comfy/models/tokenizers/qwen/tokenizer.json"),
        );
        assert_eq!(
            denoiser_path(&m, base),
            Path::new("/comfy/models/diffusion_models/krea2/raw_fp8.safetensors")
        );
        assert_eq!(
            vae_path(&m, base),
            Path::new("/comfy/models/vae/qwen/vae.safetensors")
        );
        assert_eq!(
            text_encoder_path(&m, base),
            Path::new("/comfy/models/text_encoders/qwen/enc_fp8.safetensors")
        );
        assert_eq!(
            tokenizer_path(&m, base),
            Path::new("/comfy/models/tokenizers/qwen/tokenizer.json")
        );
    }

    /// A **relative** override joins onto `base` — so a `base` of the ComfyUI
    /// `models/` root plus relative subpaths reads cleanly.
    #[test]
    fn relative_overrides_join_onto_base() {
        let base = Path::new("/comfy/models");
        let m = model(
            "/comfy/models",
            Some("diffusion_models/krea2/raw_fp8.safetensors"),
            Some("text_encoders/qwen/enc_fp8.safetensors"),
            Some("vae/qwen/vae.safetensors"),
            None,
        );
        assert_eq!(
            denoiser_path(&m, base),
            base.join("diffusion_models/krea2/raw_fp8.safetensors")
        );
        assert_eq!(vae_path(&m, base), base.join("vae/qwen/vae.safetensors"));
        assert_eq!(
            text_encoder_path(&m, base),
            base.join("text_encoders/qwen/enc_fp8.safetensors")
        );
        // The un-overridden tokenizer still falls back to the base-dir layout.
        assert_eq!(
            tokenizer_path(&m, base),
            base.join("tokenizer/tokenizer.json")
        );
    }

    /// `denoiser` (full path) and `checkpoint` (filename-in-base) coexist:
    /// `denoiser` wins when set, else `checkpoint` still steers the filename.
    #[test]
    fn denoiser_override_supersedes_checkpoint() {
        let base = Path::new("/models");
        let mut m = model("/models", None, None, None, None);
        m.checkpoint = Some("turbo_fp8_scaled.safetensors".into());
        assert_eq!(
            denoiser_path(&m, base),
            base.join("turbo_fp8_scaled.safetensors")
        );
        m.denoiser = Some(PathBuf::from("/elsewhere/my.safetensors"));
        assert_eq!(
            denoiser_path(&m, base),
            Path::new("/elsewhere/my.safetensors")
        );
    }
}
