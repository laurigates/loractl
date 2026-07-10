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
use crate::config::{BackendKind, TrainConfig};
use crate::event::TrainEvent;
use crate::model::LoraMlp;
use crate::sample;
use crate::train::Trainer;
use anyhow::{Context, Result};
use burn::backend::Autodiff;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
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

/// Flattened MNIST-shaped input width (28×28).
const INPUT_DIM: usize = 784;
/// Hidden width of the frozen random-feature projection.
const HIDDEN_DIM: usize = 256;
/// Number of classes (MNIST digits, and the synthetic demo mirrors it).
const NUM_CLASSES: usize = 10;

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

/// Train a LoRA-MLP on the given backend `B` and device, driving the whole
/// event → I/O pipeline. This is the former body of [`BurnTrainer::train`],
/// made generic over `B: AutodiffBackend` so the same loop runs on ndarray,
/// wgpu, cuda, or tch.
///
/// The seed → construct → data-generation ordering is preserved intact and is
/// backend-independent — see `.claude/rules/burn-lazy-param-init.md`:
/// `B::seed` runs first, then [`LoraMlp::new`] force-materializes the frozen
/// base, then `select_batches` draws the synthetic data.
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

    let mut optim = AdamConfig::new().init::<B, LoraMlp<B>>();
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
                adapter::save_adapter(&valid_model, &path, config.seed)
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
    adapter::save_adapter(&model.valid(), &adapter_path, config.seed)
        .with_context(|| format!("writing final adapter to {}", adapter_path.display()))?;
    sink(TrainEvent::Finished {
        adapter_path: adapter_path.clone(),
    });
    Ok(adapter_path)
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
