//! Real-MNIST LoRA convergence proof (issue #1, criterion 2 — the on-real-data
//! half). Feature-gated behind `mnist` and `#[ignore]`d because it downloads the
//! MNIST dataset over the network; run it explicitly with `just test-mnist`.
//!
//! It drives the *production* [`BurnTrainer`] over real MNIST (proving the same
//! trainer the CLI uses converges on real data), asserts the loss stream trends
//! down, then reloads the written adapter record and scores classification
//! accuracy on `MnistDataset::test()`, asserting it beats chance by a wide
//! margin. The base-frozen guarantee is proven deterministically elsewhere
//! (`lora_reference.rs` bit-exact + the `lora.rs` autodiff freeze test), so this
//! test's must-haves are just: loss decreases and accuracy is real.

#![cfg(feature = "mnist")]

use burn::backend::{Autodiff, NdArray};
use burn::data::dataset::Dataset;
use burn::data::dataset::vision::MnistDataset;
use burn::tensor::{Device, Tensor, TensorData};
use loractl_core::config::{
    ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig, OutputConfig,
    TaskKind,
};
use loractl_core::{BurnTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::PathBuf;

type AB = Autodiff<NdArray>;

const INPUT_DIM: usize = 784;
const RANK: usize = 8;
const ALPHA: f64 = 16.0;

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len() as f32
}

/// Build a single evaluation batch from the first `cap` MNIST test items with the
/// same preprocessing the trainer uses (flatten 28×28 → 784, normalize `/255`).
fn eval_batch(
    dataset: &MnistDataset,
    device: &Device<AB>,
    cap: usize,
) -> (Tensor<AB, 2>, Vec<i64>) {
    let n = dataset.len().min(cap);
    let mut features = Vec::with_capacity(n * INPUT_DIM);
    let mut labels = Vec::with_capacity(n);
    for idx in 0..n {
        let item = dataset.get(idx).expect("index within dataset");
        for row in item.image.iter() {
            for &px in row.iter() {
                features.push(px / 255.0);
            }
        }
        labels.push(item.label as i64);
    }
    let x = Tensor::<AB, 2>::from_data(TensorData::new(features, [n, INPUT_DIM]), device);
    (x, labels)
}

#[test]
#[ignore = "downloads MNIST over the network; run via `just test-mnist`"]
fn mnist_lora_converges() {
    let device: Device<AB> = Default::default();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let out_dir =
        std::env::temp_dir().join(format!("loractl-mnist-{}-{nanos}", std::process::id()));

    let steps = 400u64;
    let config = TrainConfig {
        steps,
        seed: 7,
        task: TaskKind::Classification,
        model: ModelConfig {
            base: "mnist".into(),
        },
        lora: LoraConfig {
            rank: RANK as u32,
            alpha: ALPHA as f32,
            dropout: 0.0,
            targets: vec![],
        },
        dataset: DatasetConfig {
            path: PathBuf::from("unused"),
            resolution: 28,
        },
        optim: OptimConfig {
            lr: 0.005,
            weight_decay: 0.0,
        },
        output: OutputConfig {
            dir: out_dir.clone(),
            name: "mnist-adapter".into(),
            checkpoint_every: 10_000,
            sample_every: 0,
        },
        // Default (ndarray) backend — the opt-in MNIST proof stays on CPU.
        compute: ComputeConfig::default(),
        // Unused by the classification task.
        flow: FlowConfig::default(),
    };

    let mut losses = Vec::new();
    let mut trainer = BurnTrainer;
    let adapter = trainer
        .train(&config, &mut |event| {
            if let TrainEvent::Step { loss, .. } = event {
                losses.push(loss);
            }
        })
        .expect("mnist training run succeeds");

    assert_eq!(losses.len(), steps as usize);
    let third = losses.len() / 3;
    let first = mean(&losses[..third]);
    let last = mean(&losses[losses.len() - third..]);
    assert!(
        last < 0.8 * first,
        "loss should trend down on MNIST: first-third {first:.4}, last-third {last:.4}"
    );

    // Reload the written adapter (proves the checkpoint is a real, loadable
    // safetensors + sidecar record, not just a path that happens to exist)
    // and score accuracy on the MNIST test split. `load_adapter` derives the
    // frozen base deterministically from the sidecar's seed/shape, so no
    // hardcoded hidden/class-count constants are needed here.
    let model = loractl_core::adapter::load_adapter::<AB>(&adapter, &device)
        .expect("reload trained adapter record");

    let (x, labels) = eval_batch(&MnistDataset::test(), &device, 2_000);
    let preds: Vec<i64> = model
        .forward(x)
        .argmax(1)
        .into_data()
        .iter::<i64>()
        .collect();
    let correct = preds
        .iter()
        .zip(labels.iter())
        .filter(|(p, t)| p == t)
        .count();
    let accuracy = correct as f64 / labels.len() as f64;
    eprintln!(
        "mnist_lora observed: first-third loss {first:.4} -> last-third {last:.4}; \
         test accuracy {accuracy:.4} ({correct}/{})",
        labels.len()
    );

    // OBSERVED on this machine (Apple Silicon, ndarray backend, seed 7, 400
    // steps, Adam lr 5e-3): loss 1.2552 -> 0.4112 and test accuracy 0.844
    // (1688/2000). Chance is 0.1; a frozen-random-feature MLP + rank-8 LoRA
    // readout clears it decisively. Threshold pinned ~10% below the observed
    // value; raise it if a future change reliably improves accuracy.
    assert!(
        accuracy > 0.75,
        "MNIST test accuracy {accuracy:.3} should beat chance (0.1) by a wide margin"
    );

    let _ = std::fs::remove_dir_all(&out_dir);
}
