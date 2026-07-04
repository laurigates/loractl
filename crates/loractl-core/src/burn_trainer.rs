//! The real, burn-backed trainer.
//!
//! [`BurnTrainer`] is the milestone-2 (#1) replacement for
//! [`MockTrainer`](crate::MockTrainer): it trains the LoRA factors of a
//! [`LoraMlp`] classifier with real autodiff, a real optimizer, and a real
//! cross-entropy loss, then writes an honest burn-native record to disk. It
//! satisfies the same [`Trainer`] contract, so the CLI swaps it in by changing a
//! single constructor line — the whole point of the event abstraction.
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
//! **Honest I/O.** Checkpoints and the final adapter are written as real
//! MessagePack records (`.mpk`) via [`NamedMpkFileRecorder`], available under
//! the already-enabled `std` feature — every emitted path actually exists on
//! disk. This is deliberately *not* the interoperable safetensors format, which
//! is milestone 4 (#3); a distinct extension avoids over-claiming.
//!
//! **Invariant.** This module imports only `burn`/`anyhow`/`std`, never `clap`,
//! and never writes to stdout/stderr — all progress flows through the
//! `&mut dyn FnMut(TrainEvent)` sink.

use crate::config::TrainConfig;
use crate::event::TrainEvent;
use crate::model::LoraMlp;
use crate::train::Trainer;
use anyhow::{Context, Result};
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Distribution, Int, Tensor, TensorData};
use std::path::PathBuf;

/// The autodiff-wrapped CPU backend the trainer runs on.
type AB = Autodiff<NdArray>;

/// One training batch: features `[batch, 784]` and integer class labels
/// `[batch]`.
type Batch = (Tensor<AB, 2>, Tensor<AB, 1, Int>);

/// Flattened MNIST-shaped input width (28×28).
const INPUT_DIM: usize = 784;
/// Hidden width of the frozen random-feature projection.
const HIDDEN_DIM: usize = 256;
/// Number of classes (MNIST digits, and the synthetic demo mirrors it).
const NUM_CLASSES: usize = 10;

/// A real LoRA trainer built on burn's ndarray + autodiff backend.
///
/// Unit struct, like [`MockTrainer`](crate::MockTrainer) — constructed as
/// `BurnTrainer` and driven through the [`Trainer`] trait.
pub struct BurnTrainer;

impl Trainer for BurnTrainer {
    fn train(&mut self, config: &TrainConfig, sink: &mut dyn FnMut(TrainEvent)) -> Result<PathBuf> {
        let device: Device<AB> = Default::default();
        // Seed FIRST — before the model's Kaiming init of `lora_a` and before any
        // synthetic data is drawn — so a run is fully reproducible.
        AB::seed(&device, config.seed);

        let total = config.steps.max(1);
        sink(TrainEvent::Started { total_steps: total });

        // Ensure the output dir exists so checkpoint/finish records can be
        // written — the trainer owns its own honest I/O.
        std::fs::create_dir_all(&config.output.dir)
            .with_context(|| format!("creating output dir {}", config.output.dir.display()))?;

        let rank = config.lora.rank.max(1) as usize;
        let mut model = LoraMlp::<AB>::new(
            INPUT_DIM,
            HIDDEN_DIM,
            NUM_CLASSES,
            rank,
            config.lora.alpha as f64,
            &device,
        );

        let batches = select_batches(config, &device, sink);

        let mut optim = AdamConfig::new().init::<AB, LoraMlp<AB>>();
        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let checkpoint_every = config.output.checkpoint_every.max(1);

        for step in 1..=total {
            let (x, y) = &batches[(step as usize - 1) % batches.len()];
            let logits = model.forward(x.clone());
            let loss = loss_fn.forward(logits, y.clone());
            // Read the loss BEFORE `backward()` consumes the graph — this order
            // must match the PyTorch reference's record-before-step ordering.
            let loss_value: f32 = loss.clone().into_scalar();
            sink(TrainEvent::Step {
                step,
                loss: loss_value,
                lr: config.optim.lr,
            });

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            // `step` consumes the module and returns a new one — must reassign.
            model = optim.step(config.optim.lr, model, grads);

            if step % checkpoint_every == 0 {
                let stem = config.output.dir.join(format!("checkpoint-{step}"));
                model
                    .valid()
                    .save_file(stem.clone(), &recorder)
                    .with_context(|| format!("writing checkpoint at step {step}"))?;
                sink(TrainEvent::Checkpoint {
                    step,
                    path: stem.with_extension("mpk"),
                });
            }
        }

        // Write the final adapter honestly, then report the path that exists.
        let stem = config.output.dir.join(&config.output.name);
        model
            .valid()
            .save_file(stem.clone(), &recorder)
            .with_context(|| format!("writing final adapter to {}", stem.display()))?;
        let adapter_path = stem.with_extension("mpk");
        sink(TrainEvent::Finished {
            adapter_path: adapter_path.clone(),
        });
        Ok(adapter_path)
    }
}

/// Pick the training data for this run and emit the honest [`Warning`] that
/// explains which path was taken.
///
/// Default: a seeded synthetic classification set (offline, fast). With the
/// `mnist` feature *and* `model.base == "mnist"`: the real MNIST dataset.
///
/// [`Warning`]: TrainEvent::Warning
fn select_batches(
    config: &TrainConfig,
    device: &Device<AB>,
    sink: &mut dyn FnMut(TrainEvent),
) -> Vec<Batch> {
    #[cfg(feature = "mnist")]
    if config.model.base == "mnist" {
        // Cap the sample count so an opt-in run stays reasonably short.
        return mnist_batches(device, 64, 6_000);
    }

    #[cfg(not(feature = "mnist"))]
    if config.model.base == "mnist" {
        sink(TrainEvent::Warning(
            "model.base=\"mnist\" requested but the crate was built without \
             --features mnist; falling back to the synthetic demo."
                .into(),
        ));
    }

    sink(TrainEvent::Warning(
        "M2 BurnTrainer trains a synthetic LoRA-MLP classifier demo; real \
         base-model + image-dataset ingestion arrives in a later milestone. \
         Build with --features mnist and set model.base=\"mnist\" to train on MNIST."
            .into(),
    ));
    synthetic_batches(device, NUM_CLASSES, 2_000, 64)
}

/// Build a seeded synthetic classification set of Gaussian blobs.
///
/// Each class gets a random centroid (scaled out so classes are well separated);
/// samples are centroid + unit Gaussian noise. Labels cycle through the classes
/// so every batch is class-balanced. Uses burn's now-seeded RNG, so the whole
/// set is reproducible for a given seed.
fn synthetic_batches(
    device: &Device<AB>,
    n_classes: usize,
    samples: usize,
    batch_size: usize,
) -> Vec<Batch> {
    // Centroids `[n_classes, INPUT_DIM]`, pushed apart by the ×3 scale so the
    // task is separable and the LoRA readout converges decisively.
    let centroids = Tensor::<AB, 2>::random(
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
        let idx = Tensor::<AB, 1, Int>::from_data(TensorData::new(labels, [batch_size]), device);
        let centers = centroids.clone().select(0, idx.clone());
        let noise = Tensor::<AB, 2>::random(
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
fn mnist_batches(device: &Device<AB>, batch_size: usize, cap: usize) -> Vec<Batch> {
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
        let x = Tensor::<AB, 2>::from_data(TensorData::new(features, [rows, INPUT_DIM]), device);
        let y = Tensor::<AB, 1, Int>::from_data(TensorData::new(labels, [rows]), device);
        batches.push((x, y));
        i = end;
    }
    batches
}
