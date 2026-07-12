//! The real, burn-backed trainer.
//!
//! [`BurnTrainer`] is the milestone-2 (#1) replacement for
//! [`MockTrainer`](crate::MockTrainer): it trains the LoRA factors of a
//! [`LoraMlp`] classifier with real autodiff, a real optimizer, and a real
//! cross-entropy loss, then writes an honest burn-native record to disk. It
//! satisfies the same [`Trainer`] contract, so the CLI swaps it in by changing a
//! single constructor line — the whole point of the event abstraction.
//!
//! **Compute backend (M7, #18).** The training loop is generic over
//! `B: AutodiffBackend`; [`BurnTrainer::train`](Trainer::train) is a thin
//! runtime dispatcher that reads [`config.compute.backend`](crate::ComputeConfig)
//! and calls the monomorphized [`run_training`] for the selected backend. The
//! `ndarray` (CPU) arm is always compiled and is the default, so `cargo test`
//! and CI stay offline; the `wgpu`/`cuda`/`tch` arms are `#[cfg]`-gated on their
//! cargo features and a `#[cfg(not(...))]` arm fails loudly when a config selects
//! a backend this binary was not built with (never a silent CPU fallback).
//! Because selection flows through the config *into* the trainer, the front-end
//! trainer-construction seams do not change at all — the event abstraction is
//! over-satisfied.
//!
//! **Default run — synthetic demo.** With no `mnist` feature (or any
//! `model.base` other than `"mnist"`), the trainer fabricates a seeded,
//! in-memory Gaussian-blob classification set and trains on it. This keeps the
//! default `cargo test` / `loractl train` fully offline, fast, and dependency-
//! light while still exercising the *real* training loop (loss genuinely
//! decreases). It emits one honest [`Warning`](TrainEvent::Warning) saying so.
//!
//! **`--features mnist` + `model.base = "mnist"`.** The trainer instead loads
//! the real MNIST dataset (flattened, normalized) and trains the same LoRA-MLP
//! on it. That path pulls a networked dataset downloader, so it is strictly
//! opt-in and never part of the default build.
//!
//! **Task dispatch (M8, #19).** `config.task` selects the training objective
//! *inside* the backend-generic [`run_training`] — never around the
//! feature-gated backend ladder, which would double its `#[cfg]` arms.
//! `classification` (the default) is the M2 demo above; `flow-matching` trains
//! the same `LoraMlp` shape as a rectified-flow *velocity net* on a synthetic
//! latent toy — input `concat[x_t, t]`, output = predicted `v`, plain MSE
//! against `ε − x_0` — with all flow math routed through the golden-pinned
//! [`crate::flow`] helpers (see [`flow_batches`]). Unsupported combinations
//! (flow + `sample_every > 0`) are rejected up front in
//! [`BurnTrainer::train`](Trainer::train), before the backend match.
//!
//! **Honest I/O.** Checkpoints and the final adapter are written as
//! adapter-only `.safetensors` files via [`crate::adapter::save_adapter`] —
//! only the trainable LoRA factors are persisted; the frozen base is
//! regenerated deterministically at load time from the run's seed. See the
//! [`adapter`](crate::adapter) module docs for the tensor-naming scheme and
//! why a JSON sidecar carries the reconstruction metadata. This is milestone
//! 4 (#3): the interoperable-format adapter I/O milestone 2's `.mpk` stopgap
//! deferred.
//!
//! **Validation samples.** When `config.output.sample_every > 0`, every N
//! steps the trainer runs one deterministic forward pass (see
//! [`crate::sample`]) on a FIXED probe input and writes the result to
//! `sample-{step}.json`, emitting [`TrainEvent::Sample`]. Using the same
//! fixed probe across every periodic sample within a run is deliberate: it
//! lets you watch one input's prediction/logits evolve as training
//! progresses across the successive `sample-{step}.json` files.
//!
//! **Invariant.** This module imports only `burn`/`anyhow`/`serde_json`/`std`,
//! never `clap`, and never writes to stdout/stderr — all progress flows
//! through the `&mut dyn FnMut(TrainEvent)` sink.

use crate::adapter;
use crate::config::{BackendKind, FlowConfig, TaskKind, TrainConfig};
use crate::event::TrainEvent;
use crate::flow;
use crate::model::LoraMlp;
use crate::sample;
use crate::train::Trainer;
use anyhow::{Context, Result};
use burn::backend::Autodiff;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Distribution, ElementConversion, Int, Tensor, TensorData};
use std::path::PathBuf;

#[cfg(feature = "cuda")]
use burn::backend::{Cuda, cuda::CudaDevice};
#[cfg(feature = "tch")]
use burn::backend::{LibTorch, libtorch::LibTorchDevice};
#[cfg(feature = "wgpu")]
use burn::backend::{Wgpu, wgpu::WgpuDevice};

/// One training batch: features `[batch, 784]` and integer class labels
/// `[batch]`, on the run's selected backend `B`.
type Batch<B> = (Tensor<B, 2>, Tensor<B, 1, Int>);

/// One flow-matching training batch: velocity-net input
/// `[batch, FLOW_LATENT_DIM + 1]` (`concat[x_t, t]`) and its v-prediction
/// target `[batch, FLOW_LATENT_DIM]`.
pub type FlowBatch<B> = (Tensor<B, 2>, Tensor<B, 2>);

/// Flattened MNIST-shaped input width (28×28).
const INPUT_DIM: usize = 784;
/// Hidden width of the frozen random-feature projection.
const HIDDEN_DIM: usize = 256;
/// Number of classes (MNIST digits, and the synthetic demo mirrors it).
const NUM_CLASSES: usize = 10;

/// Latent width of the synthetic flow-matching toy (M8, #19). The velocity
/// net's input is one column wider (`concat[x_t, t]`) and its output is this
/// wide (the predicted velocity). Duplicated in `tests/flow_convergence.rs`
/// with a MUST-match comment.
const FLOW_LATENT_DIM: usize = 16;
/// Hidden width of the flow velocity net's frozen random-feature projection.
const FLOW_HIDDEN: usize = 64;
/// Synthetic flow sample count and batch size — 2000/64 = 31 pre-generated
/// batches, cycled by the training loop (mirrors the classification demo's
/// `synthetic_batches` sizing).
const FLOW_SAMPLES: usize = 2_000;
/// Batch size of the synthetic flow toy.
const FLOW_BATCH_SIZE: usize = 64;

/// Fixed seed for every periodic validation sample within a run.
///
/// Deliberately NOT derived from `step`: using the SAME probe input for every
/// sample lets you watch the model's prediction/logits on one fixed input
/// evolve across the successive `sample-{step}.json` files as training
/// progresses — that comparison is the actual value of a "validation
/// sample," and it would be lost if each sample used a different input.
const VALIDATION_SAMPLE_SEED: u64 = 0;

/// A real LoRA trainer built on burn's autodiff backend.
///
/// Unit struct, like [`MockTrainer`](crate::MockTrainer) — constructed as
/// `BurnTrainer` and driven through the [`Trainer`] trait. It stays a concrete
/// (non-generic) type so `Trainer` remains object-safe (loractl-api boxes it as
/// `Box<dyn Trainer>`); the compute backend is selected at run time inside
/// [`train`](Trainer::train), not baked into the type.
pub struct BurnTrainer;

impl Trainer for BurnTrainer {
    fn train(&mut self, config: &TrainConfig, sink: &mut dyn FnMut(TrainEvent)) -> Result<PathBuf> {
        // Config validation FIRST — deliberately BEFORE the backend match, so
        // it is compiled once (not per backend arm) and an invalid combination
        // fails identically on every backend: before `B::seed`, before any
        // TrainEvent reaches the sink, before any filesystem I/O.
        if config.task == TaskKind::FlowMatching && config.output.sample_every > 0 {
            anyhow::bail!(
                "validation sampling (output.sample_every = {}) is classification-specific — \
                 the flow-matching task trains a velocity net with no classifier sample path; \
                 set output.sample_every to 0",
                config.output.sample_every
            );
        }

        // Runtime dispatch over the config-selected backend. The ndarray arm is
        // always compiled; each GPU arm is `#[cfg]`-gated and paired with a
        // `#[cfg(not(...))]` arm that bails loudly (never a silent CPU
        // fallback) when the feature is absent.
        match config.compute.backend {
            BackendKind::Ndarray => {
                if config.compute.device != 0 {
                    sink(TrainEvent::Warning {
                        message: format!(
                            "ndarray (CPU) backend ignores device index {}; running on CPU",
                            config.compute.device
                        ),
                    });
                }
                let device = NdArrayDevice::default();
                run_training::<Autodiff<NdArray>>(config, device, sink)
            }
            #[cfg(feature = "wgpu")]
            BackendKind::Wgpu => {
                let device = wgpu_device(config.compute.device);
                run_training::<Autodiff<Wgpu>>(config, device, sink)
            }
            #[cfg(not(feature = "wgpu"))]
            BackendKind::Wgpu => anyhow::bail!(
                "config selected the 'wgpu' backend but this binary was built without it; \
                 rebuild with `--features wgpu` (Metal on macOS, Vulkan/DX12 elsewhere)"
            ),
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => {
                let device = CudaDevice::new(config.compute.device);
                run_training::<Autodiff<Cuda>>(config, device, sink)
            }
            #[cfg(not(feature = "cuda"))]
            BackendKind::Cuda => anyhow::bail!(
                "config selected the 'cuda' backend but this binary was built without it; \
                 rebuild with `--features cuda` on a Linux+NVIDIA host (CUDA toolkit \
                 required). cuda is not runnable on macOS"
            ),
            #[cfg(feature = "tch")]
            BackendKind::Tch => {
                let device = tch_device(config.compute.device);
                run_training::<Autodiff<LibTorch>>(config, device, sink)
            }
            #[cfg(not(feature = "tch"))]
            BackendKind::Tch => anyhow::bail!(
                "config selected the 'tch' backend but this binary was built without it; \
                 rebuild with `--features tch` (a linked libtorch binary is required)"
            ),
        }
    }
}

/// Build the wgpu device for `index`.
///
/// Index `0` maps to `WgpuDevice::default()` — the auto-selected best GPU, which
/// on Apple Silicon is the single Metal GPU. This is the ONLY path verified on
/// the dev machine.
#[cfg(feature = "wgpu")]
fn wgpu_device(index: usize) -> WgpuDevice {
    if index == 0 {
        WgpuDevice::default()
    } else {
        // NOTE (unverified on this Mac): the Apple GPU is integrated (unified
        // memory), so `DiscreteGpu(index)` is correct only on an x86 host with
        // discrete GPUs. A non-zero index on Apple Silicon is host-dependent and
        // untested — index 0 (the default GPU) is the supported path.
        WgpuDevice::DiscreteGpu(index)
    }
}

/// Build the libtorch device for `index`.
///
/// UNVERIFIED on this Mac: libtorch is not linked here. `Cuda(index)` is the
/// NVIDIA mapping; `LibTorchDevice::Mps` would be the Apple path if a
/// libtorch-with-MPS build were linked. Validate the exact mapping on target
/// hardware before relying on the `tch` backend.
#[cfg(feature = "tch")]
fn tch_device(index: usize) -> LibTorchDevice {
    LibTorchDevice::Cuda(index)
}

/// Run one training job on the given backend `B` and device, driving the
/// whole event → I/O pipeline. Generic over `B: AutodiffBackend` so the same
/// pipeline runs on ndarray, wgpu, cuda, or tch.
///
/// Does the task-independent setup (seed, `Started`, output dir), then
/// dispatches on `config.task` — the task branch lives INSIDE this generic
/// fn, never around the feature-gated backend ladder in
/// [`BurnTrainer::train`](Trainer::train), which would double its `#[cfg]`
/// arms.
///
/// The seed → construct → data-generation ordering is preserved intact and is
/// backend-independent — see `.claude/rules/burn-lazy-param-init.md`:
/// `B::seed` runs first, then [`LoraMlp::new`] force-materializes the frozen
/// base, then `select_batches`/[`flow_batches`] draw the synthetic data.
fn run_training<B: AutodiffBackend>(
    config: &TrainConfig,
    device: B::Device,
    sink: &mut dyn FnMut(TrainEvent),
) -> Result<PathBuf> {
    // Seed FIRST — before the model's Kaiming init of `lora_a` and before any
    // synthetic data is drawn — so a run is fully reproducible.
    B::seed(&device, config.seed);

    let total = config.steps.max(1);
    sink(TrainEvent::Started { total_steps: total });

    // Ensure the output dir exists so checkpoint/finish records can be
    // written — the trainer owns its own honest I/O.
    std::fs::create_dir_all(&config.output.dir)
        .with_context(|| format!("creating output dir {}", config.output.dir.display()))?;

    match config.task {
        TaskKind::Classification => run_classification::<B>(config, device, sink),
        TaskKind::FlowMatching => run_flow_matching::<B>(config, device, sink),
    }
}

/// Train the LoRA-MLP classifier (the M2 demo / opt-in MNIST path) — the
/// former tail of [`run_training`], unchanged apart from the rename.
fn run_classification<B: AutodiffBackend>(
    config: &TrainConfig,
    device: B::Device,
    sink: &mut dyn FnMut(TrainEvent),
) -> Result<PathBuf> {
    let total = config.steps.max(1);
    let rank = config.lora.rank.max(1) as usize;
    let mut model = LoraMlp::<B>::new(
        INPUT_DIM,
        HIDDEN_DIM,
        NUM_CLASSES,
        rank,
        config.lora.alpha as f64,
        &device,
    );

    let batches = select_batches::<B>(config, &device, sink);

    // AdamW (decoupled weight decay) so `optim.weight_decay` is honored; at the
    // default `0.0` this is numerically identical to plain Adam, so the numerics
    // goldens are unaffected. `AdamWConfig`'s own default decay is 1e-4, so we
    // always set it explicitly from config.
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.optim.weight_decay as f32)
        .init::<B, LoraMlp<B>>();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let checkpoint_every = config.output.checkpoint_every.max(1);
    let sample_every = config.output.sample_every;

    for step in 1..=total {
        let (x, y) = &batches[(step as usize - 1) % batches.len()];
        let logits = model.forward(x.clone());
        let loss = loss_fn.forward(logits, y.clone());
        // Read the loss BEFORE `backward()` consumes the graph — this order
        // must match the PyTorch reference's record-before-step ordering.
        // `.elem()` converts `B::FloatElem` to `f32` so this compiles for any
        // backend (a concrete `AB` would let `into_scalar()` yield `f32`
        // directly, but the generic `B::FloatElem` is not provably `f32`).
        let loss_value: f32 = loss.clone().into_scalar().elem();
        sink(TrainEvent::Step {
            step,
            loss: loss_value,
            lr: config.optim.lr,
        });

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        // `step` consumes the module and returns a new one — must reassign.
        model = optim.step(config.optim.lr, model, grads);

        let want_checkpoint = step % checkpoint_every == 0;
        let want_sample = sample_every > 0 && step % sample_every == 0;
        if want_checkpoint || want_sample {
            // Compute the eval-mode snapshot once and reuse it for both
            // writes below — `.valid()` clones the whole model.
            let valid_model = model.valid();

            if want_checkpoint {
                let path = config
                    .output
                    .dir
                    .join(format!("checkpoint-{step}.safetensors"));
                adapter::save_adapter(&valid_model, &path, config.seed, config.task)
                    .with_context(|| format!("writing checkpoint at step {step}"))?;
                sink(TrainEvent::Checkpoint { step, path });
            }

            if want_sample {
                let sample_out = sample::run_sample(&valid_model, VALIDATION_SAMPLE_SEED, &device)
                    .with_context(|| format!("running validation sample at step {step}"))?;
                let sample_path = config.output.dir.join(format!("sample-{step}.json"));
                let report = serde_json::json!({
                    "step": step,
                    "predicted_class": sample_out.predicted_class,
                    "logits": sample_out.logits,
                });
                let report_json = serde_json::to_string_pretty(&report)
                    .context("serializing validation sample")?;
                std::fs::write(&sample_path, report_json)
                    .with_context(|| format!("writing sample at step {step}"))?;
                sink(TrainEvent::Sample {
                    step,
                    path: sample_path,
                });
            }
        }
    }

    // Write the final adapter honestly, then report the path that exists.
    let adapter_path = config
        .output
        .dir
        .join(&config.output.name)
        .with_extension("safetensors");
    adapter::save_adapter(&model.valid(), &adapter_path, config.seed, config.task)
        .with_context(|| format!("writing final adapter to {}", adapter_path.display()))?;
    sink(TrainEvent::Finished {
        adapter_path: adapter_path.clone(),
    });
    Ok(adapter_path)
}

/// Train the LoRA readout of a [`LoraMlp`] *velocity net* on the synthetic
/// rectified-flow toy (M8, #19).
///
/// The velocity net IS the [`LoraMlp`] shape at flow dims — input
/// `concat[x_t, t]` (`FLOW_LATENT_DIM + 1` wide), frozen random-feature
/// projection, LoRA readout predicting `v ∈ ℝ^FLOW_LATENT_DIM`. ReLU is
/// hidden-only, so the linear readout handles negative velocities, and
/// checkpointing/adapter save-load (sidecar carries the dims and the task)
/// plus the eager frozen-param materialization are all inherited unchanged.
///
/// The loss is plain MSE against [`flow::velocity_target`] with weighting
/// ≡ 1.0: the logit-normal scheme's `t/(1−t)` emphasis is delivered by the
/// *sampling density* ([`flow::sample_timesteps`]), and a multiplicative loss
/// weight on top would double-count it.
fn run_flow_matching<B: AutodiffBackend>(
    config: &TrainConfig,
    device: B::Device,
    sink: &mut dyn FnMut(TrainEvent),
) -> Result<PathBuf> {
    sink(TrainEvent::Warning {
        message: "M8 flow-matching trains a synthetic latent-velocity toy (rectified-flow \
                  v-prediction); model.base and dataset.path are unused — real image-latent \
                  ingestion arrives with the Krea 2 stack (M9–M12)."
            .into(),
    });

    let total = config.steps.max(1);
    let rank = config.lora.rank.max(1) as usize;
    // Constructed BEFORE `flow_batches` so the frozen params' eager
    // materialization stays pinned right after `B::seed` — the same RNG
    // ordering contract as the classification path (see
    // `.claude/rules/burn-lazy-param-init.md`).
    let mut model = LoraMlp::<B>::new(
        FLOW_LATENT_DIM + 1,
        FLOW_HIDDEN,
        FLOW_LATENT_DIM,
        rank,
        config.lora.alpha as f64,
        &device,
    );

    let batches = flow_batches::<B>(
        FLOW_SAMPLES / FLOW_BATCH_SIZE,
        FLOW_BATCH_SIZE,
        config.flow,
        &device,
    );

    // AdamW (decoupled weight decay); see the note in `run_classification`.
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.optim.weight_decay as f32)
        .init::<B, LoraMlp<B>>();
    let checkpoint_every = config.output.checkpoint_every.max(1);

    for step in 1..=total {
        let (input, target) = &batches[(step as usize - 1) % batches.len()];
        let pred = model.forward(input.clone());
        let diff = pred - target.clone();
        // Plain MSE, identical to the golden's ((pred - v)**2).mean().
        let loss = diff.clone().mul(diff).mean();
        // Read the loss BEFORE `backward()` consumes the graph — this order
        // must match the PyTorch reference's record-before-step ordering.
        // `.elem()` converts `B::FloatElem` to `f32` for any backend.
        let loss_value: f32 = loss.clone().into_scalar().elem();
        sink(TrainEvent::Step {
            step,
            loss: loss_value,
            lr: config.optim.lr,
        });

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        // `step` consumes the module and returns a new one — must reassign.
        model = optim.step(config.optim.lr, model, grads);

        if step % checkpoint_every == 0 {
            let path = config
                .output
                .dir
                .join(format!("checkpoint-{step}.safetensors"));
            adapter::save_adapter(&model.valid(), &path, config.seed, config.task)
                .with_context(|| format!("writing checkpoint at step {step}"))?;
            sink(TrainEvent::Checkpoint { step, path });
        }
    }

    // Write the final adapter honestly, then report the path that exists.
    let adapter_path = config
        .output
        .dir
        .join(&config.output.name)
        .with_extension("safetensors");
    adapter::save_adapter(&model.valid(), &adapter_path, config.seed, config.task)
        .with_context(|| format!("writing final adapter to {}", adapter_path.display()))?;
    sink(TrainEvent::Finished {
        adapter_path: adapter_path.clone(),
    });
    Ok(adapter_path)
}

/// Build the synthetic rectified-flow training set (M8, #19).
///
/// The data distribution is a point mass: `x_0 ≡ c`, a FIXED constant vector
/// (`c[i] = ±1.5` alternating; no RNG). With `x_0` deterministic, `ε` is a
/// function of `(x_t, t)`, so `E[v | x_t, t] = (x_t − c)/t` exactly — the
/// irreducible conditional-variance floor is zero and any loss-ratio drop is
/// attributable purely to learning. (The model's own representational floor —
/// frozen random features + low-rank readout — remains nonzero, which is why
/// the convergence gate is a loss *ratio*, never an absolute near-zero loss.)
///
/// Per batch: `ε ~ N(0, I)` from the seeded device RNG, `t` from
/// [`flow::sample_timesteps`] (logit-normal + shift), then — MANDATORILY —
/// `x_t` from [`flow::interpolate`] and the target from
/// [`flow::velocity_target`]: routing through the golden-pinned helpers,
/// never inline `ε − c` arithmetic, is what keeps the sign conventions from
/// silently flipping here. The `t` column appended to the input is the SAME
/// shifted `t` tensor fed to `interpolate`.
///
/// Public (not `pub(crate)`) so the sign-pinning identity test in
/// `tests/flow_convergence.rs` can assert those conventions on real output —
/// the toy is exactly sign-symmetric, so the convergence gate alone cannot
/// see a flip.
pub fn flow_batches<B: Backend>(
    n_batches: usize,
    batch_size: usize,
    flow_cfg: FlowConfig,
    device: &B::Device,
) -> Vec<FlowBatch<B>> {
    // x_0 ≡ c, tiled to [batch, FLOW_LATENT_DIM] — fixed and RNG-free.
    let c: Vec<f32> = (0..FLOW_LATENT_DIM)
        .map(|i| if i % 2 == 0 { 1.5 } else { -1.5 })
        .collect();
    let tiled: Vec<f32> = c
        .iter()
        .copied()
        .cycle()
        .take(batch_size * FLOW_LATENT_DIM)
        .collect();
    let x0 = Tensor::<B, 2>::from_data(
        TensorData::new(tiled, [batch_size, FLOW_LATENT_DIM]),
        device,
    );

    let n_batches = n_batches.max(1);
    let mut batches = Vec::with_capacity(n_batches);
    for _ in 0..n_batches {
        let eps = Tensor::<B, 2>::random(
            [batch_size, FLOW_LATENT_DIM],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let t = flow::sample_timesteps::<B>(batch_size, flow_cfg, device);
        let x_t = flow::interpolate(x0.clone(), eps.clone(), t.clone());
        let target = flow::velocity_target(x0.clone(), eps);
        let t_col: Tensor<B, 2> = t.unsqueeze_dim(1);
        let input = Tensor::cat(vec![x_t, t_col], 1);
        batches.push((input, target));
    }
    batches
}

/// Pick the training data for this run and emit the honest [`Warning`] that
/// explains which path was taken.
///
/// Default: a seeded synthetic classification set (offline, fast). With the
/// `mnist` feature *and* `model.base == "mnist"`: the real MNIST dataset.
/// Generic over `B: Backend` (the weaker bound — data building needs no
/// autodiff), instantiated with the run's autodiff backend.
///
/// [`Warning`]: TrainEvent::Warning
fn select_batches<B: Backend>(
    config: &TrainConfig,
    device: &B::Device,
    sink: &mut dyn FnMut(TrainEvent),
) -> Vec<Batch<B>> {
    #[cfg(feature = "mnist")]
    if config.model.base == "mnist" {
        // Cap the sample count so an opt-in run stays reasonably short.
        return mnist_batches::<B>(device, 64, 6_000);
    }

    #[cfg(not(feature = "mnist"))]
    if config.model.base == "mnist" {
        sink(TrainEvent::Warning {
            message: "model.base=\"mnist\" requested but the crate was built without \
                      --features mnist; falling back to the synthetic demo."
                .into(),
        });
    }

    sink(TrainEvent::Warning {
        message: "M2 BurnTrainer trains a synthetic LoRA-MLP classifier demo; real \
                  base-model + image-dataset ingestion arrives in a later milestone. \
                  Build with --features mnist and set model.base=\"mnist\" to train on MNIST."
            .into(),
    });
    synthetic_batches::<B>(device, NUM_CLASSES, 2_000, 64)
}

/// Build a seeded synthetic classification set of Gaussian blobs.
///
/// Each class gets a random centroid (scaled out so classes are well separated);
/// samples are centroid + unit Gaussian noise. Labels cycle through the classes
/// so every batch is class-balanced. Uses burn's now-seeded RNG, so the whole
/// set is reproducible for a given seed.
fn synthetic_batches<B: Backend>(
    device: &B::Device,
    n_classes: usize,
    samples: usize,
    batch_size: usize,
) -> Vec<Batch<B>> {
    // Centroids `[n_classes, INPUT_DIM]`, pushed apart by the ×3 scale so the
    // task is separable and the LoRA readout converges decisively.
    let centroids = Tensor::<B, 2>::random(
        [n_classes, INPUT_DIM],
        Distribution::Normal(0.0, 1.0),
        device,
    )
    .mul_scalar(3.0);

    let n_batches = (samples / batch_size).max(1);
    let mut batches = Vec::with_capacity(n_batches);
    for b in 0..n_batches {
        let labels: Vec<i64> = (0..batch_size)
            .map(|j| ((b * batch_size + j) % n_classes) as i64)
            .collect();
        let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(labels, [batch_size]), device);
        let centers = centroids.clone().select(0, idx.clone());
        let noise = Tensor::<B, 2>::random(
            [batch_size, INPUT_DIM],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        batches.push((centers + noise, idx));
    }
    batches
}

/// Build training batches from the real MNIST training split (opt-in).
///
/// Flattens each 28×28 image to a 784-vector, normalizes pixels to `[0, 1]`, and
/// packs into fixed-size batches with `i64` labels. `cap` bounds how many samples
/// are used so an opt-in run stays short. Requires the `mnist` feature (which
/// pulls a networked dataset downloader).
#[cfg(feature = "mnist")]
fn mnist_batches<B: Backend>(device: &B::Device, batch_size: usize, cap: usize) -> Vec<Batch<B>> {
    use burn::data::dataset::Dataset;

    let dataset = burn::data::dataset::vision::MnistDataset::train();
    let n = dataset.len().min(cap);
    let mut batches = Vec::new();
    let mut i = 0;
    while i < n {
        let end = (i + batch_size).min(n);
        let rows = end - i;
        let mut features = Vec::with_capacity(rows * INPUT_DIM);
        let mut labels = Vec::with_capacity(rows);
        for idx in i..end {
            let item = dataset.get(idx).expect("index within dataset length");
            for row in item.image.iter() {
                for &px in row.iter() {
                    features.push(px / 255.0);
                }
            }
            labels.push(item.label as i64);
        }
        let x = Tensor::<B, 2>::from_data(TensorData::new(features, [rows, INPUT_DIM]), device);
        let y = Tensor::<B, 1, Int>::from_data(TensorData::new(labels, [rows]), device);
        batches.push((x, y));
        i = end;
    }
    batches
}
