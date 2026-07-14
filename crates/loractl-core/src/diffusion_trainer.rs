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
//! <base>/raw.safetensors                          the MMDiT
//! <base>/text_encoder/model.safetensors           Qwen3-VL (vision tower dropped at load)
//! <base>/tokenizer/tokenizer.json                 the Qwen tokenizer
//! <base>/vae/diffusion_pytorch_model.safetensors  the Qwen-Image VAE
//! ```
//!
//! [`ModelVariant`] names the architecture explicitly (`krea2` |
//! `tiny-krea2`) — a config mistake is a clear error, never a creative
//! shape inference.
//!
//! ## Memory sequencing
//!
//! The VAE and text encoder run only during dataset preparation (everything
//! they produce is cached by M12), so they are loaded, used, and **dropped
//! before the MMDiT loads** — peak memory holds either the encoders or the
//! denoiser, never both.
//!
//! Like every trainer, this emits [`TrainEvent`]s through the sink and never
//! renders; the front-ends changed only at their constructor seam (a
//! two-armed factory on `model.base`).

use crate::adapters::{LoraAdapters, build_adapters};
use crate::config::{BackendKind, ModelVariant, Precision, TaskKind, TrainConfig};
use crate::dataset::prepare_dataset;
use crate::event::TrainEvent;
use crate::export::{ExportFormat, export_adapters};
use crate::flow::{interpolate, sample_timesteps, velocity_target};
use crate::mmdit::{Mmdit, MmditConfig, krea2_positions, patchify};
use crate::qwen_vae::{QwenVae, QwenVaeConfig};
use crate::qwen3vl::{Qwen3VlConditioner, Qwen3VlConfig, Qwen3VlEncoder};
use crate::train::Trainer;
use anyhow::{Context, Result, bail};
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Distribution, ElementConversion, Tensor};
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use std::path::{Path, PathBuf};

#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;

/// The per-variant architecture bundle: (MMDiT, encoder, VAE, caption budget).
fn variant_configs(variant: ModelVariant) -> (MmditConfig, Qwen3VlConfig, QwenVaeConfig, usize) {
    match variant {
        ModelVariant::Krea2 => (
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

        match config.compute.backend {
            BackendKind::Ndarray => {
                if config.compute.precision != Precision::F32 {
                    bail!(
                        "compute.precision = f16 is only supported on the wgpu backend \
                         (selected backend: ndarray)"
                    );
                }
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
            },
            #[cfg(not(feature = "wgpu"))]
            BackendKind::Wgpu => bail!(
                "config selected the 'wgpu' backend but this binary was built without it; \
                 rebuild with `--features wgpu`"
            ),
            BackendKind::Cuda | BackendKind::Tch => bail!(
                "the diffusion trainer currently wires ndarray and wgpu; \
                 cuda/tch land once they can be verified on real hardware"
            ),
        }
    }
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
    let mut store = SafetensorsStore::from_file(path).remap(remapper);
    if let Some(pattern) = filter {
        store = store.with_regex(pattern).allow_partial(true);
    }
    if transpose_linears {
        store = store.with_from_adapter(PyTorchToBurnAdapter);
    }
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

/// One training job on backend `AB`: prepare (and cache) the dataset with the
/// frozen encoders, drop them, load the MMDiT, and train the injected LoRA.
fn run_diffusion<AB: AutodiffBackend>(
    config: &TrainConfig,
    device: AB::Device,
    sink: &mut dyn FnMut(TrainEvent),
) -> Result<PathBuf> {
    AB::seed(&device, config.seed);

    let total = config.steps.max(1);
    sink(TrainEvent::Started { total_steps: total });
    std::fs::create_dir_all(&config.output.dir)
        .with_context(|| format!("creating output dir {}", config.output.dir.display()))?;

    let base = PathBuf::from(&config.model.base);
    let variant = config.model.variant;
    let (mmdit_cfg, enc_cfg, vae_cfg, max_length) = variant_configs(variant);
    let patch = mmdit_cfg.patch;

    // ---- Phase 1: encode + cache the dataset; encoders dropped after. ----
    let fingerprint = format!("{variant:?}-ml{max_length}").to_lowercase();
    let prepared = {
        let vae = load_module(
            QwenVae::<AB>::init(vae_cfg, &device),
            &base.join("vae/diffusion_pytorch_model.safetensors"),
            &QwenVae::<AB>::key_remap(),
            false,
            None,
            "VAE",
        )?
        .no_grad();
        let encoder = load_module(
            Qwen3VlEncoder::<AB>::init(enc_cfg, &device),
            &base.join("text_encoder/model.safetensors"),
            &[],
            true,
            Some(Qwen3VlEncoder::<AB>::load_filter()),
            "text encoder",
        )?
        .no_grad();
        let conditioner =
            Qwen3VlConditioner::new(encoder, &base.join("tokenizer/tokenizer.json"), max_length)?;
        prepare_dataset::<AB>(
            &config.dataset,
            &fingerprint,
            &device,
            |image| Ok(vae.encode(image)),
            |caption| conditioner.encode_captions(&[caption], &device),
        )?
    };
    let batches = prepared.batches(config.dataset.batch_size.max(1) as usize);
    if batches.is_empty() {
        bail!("the dataset produced no batches");
    }

    // ---- Phase 2: the denoiser + adapters. ----
    let mmdit = load_module(
        Mmdit::<AB>::init(mmdit_cfg, &device),
        &base.join("raw.safetensors"),
        &Mmdit::<AB>::key_remap(),
        true,
        None,
        "MMDiT",
    )?
    .no_grad();

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

    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.optim.weight_decay as f32)
        .init::<AB, LoraAdapters<AB>>();
    let checkpoint_every = config.output.checkpoint_every.max(1);
    let adapter_path = config
        .output
        .dir
        .join(format!("{}.safetensors", config.output.name));

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
        sink(TrainEvent::Step {
            step,
            loss: loss_value,
            lr: config.optim.lr,
        });

        let grads = GradientsParams::from_grads(loss.backward(), &set);
        set = optim.step(config.optim.lr, set, grads);

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
