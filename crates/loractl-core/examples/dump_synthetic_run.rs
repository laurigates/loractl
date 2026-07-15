//! Dump a synthetic `BurnTrainer` run's exact initial weights and training
//! batches, so PyTorch can replay the same loop (#49 H9).
//!
//! `cargo run -p loractl-core --example dump_synthetic_run -- <out-dir>`
//! (driven by `just burn-trainer-reference`; you should not need to run it by
//! hand).
//!
//! ## Why a dump at all
//!
//! Every other golden in this repo is derived *purely* in Python: the toy's
//! weights and inputs are fixed constants, so `reference/lora_reference.py` can
//! generate them and burn can reproduce them. `BurnTrainer`'s synthetic
//! classification run cannot work that way — its frozen base, its LoRA `A`
//! factor, and its Gaussian-blob dataset all come out of burn's seeded
//! `StdRng`, and PyTorch cannot reproduce that stream. So the reference is fed
//! burn's *actual* tensors and independently recomputes the *losses* from them:
//! the arithmetic (forward, cross-entropy, AdamW) stays an independent PyTorch
//! implementation, which is the part the golden is meant to pin.
//!
//! ## Format
//!
//! `manifest.json` (hyperparams + tensor shapes) plus one raw little-endian
//! binary per tensor — `f32` for weights/features, `i64` for labels. Raw bins
//! keep the ~4 MB of batch data out of JSON; nothing here is committed (the
//! dump is a throwaway under a temp dir, only the tiny loss golden lands in
//! git).

use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Int, Tensor};
use loractl_core::burn_trainer::synthetic_run_inputs;
use loractl_core::config::{
    ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig, OutputConfig,
    TaskKind,
};
use loractl_core::{LoraMlp, TrainConfig};
use std::path::{Path, PathBuf};

/// The run under test — MUST match `crates/loractl-core/tests/burn_trainer_reference.rs`
/// (the test re-asserts these against the golden's recorded hyperparams, so a
/// drift here fails loudly rather than silently regenerating a golden for a
/// different run).
const SEED: u64 = 7;
const STEPS: u64 = 12;
const RANK: u32 = 4;
const ALPHA: f32 = 8.0;
const LR: f64 = 0.01;
/// The two weight-decay settings the golden pins. `0.0` is the plain trajectory;
/// `1.0` exercises AdamW's *decoupled* decay against torch's own AdamW — the
/// kill-test value `.claude/rules/burn-optimizer-and-dropout.md` prescribes,
/// chosen so the two trajectories separate by ~5e-2 (far above the golden's
/// 1e-3 tolerance) instead of vanishing into the noise floor.
const WEIGHT_DECAYS: [f64; 2] = [0.0, 1.0];

type B = Autodiff<NdArray>;

fn config(out_dir: &Path) -> TrainConfig {
    TrainConfig {
        steps: STEPS,
        seed: SEED,
        task: TaskKind::Classification,
        model: ModelConfig {
            base: "synthetic".into(),
            variant: Default::default(),
            checkpoint: None,
        },
        lora: LoraConfig {
            rank: RANK,
            alpha: ALPHA,
            dropout: 0.0,
            targets: vec![],
        },
        dataset: DatasetConfig {
            path: PathBuf::from("unused"),
            resolution: 28,
            batch_size: 1,
        },
        optim: OptimConfig {
            lr: LR,
            // Irrelevant to the dump: weight decay touches only the optimizer,
            // never the init or the data, so ONE dump serves both trajectories.
            weight_decay: WEIGHT_DECAYS[0],
        },
        output: OutputConfig {
            dir: out_dir.to_path_buf(),
            name: "adapter".into(),
            checkpoint_every: u64::MAX,
            sample_every: 0,
        },
        compute: ComputeConfig::default(),
        flow: FlowConfig::default(),
    }
}

fn write_f32<const D: usize>(dir: &Path, name: &str, tensor: Tensor<B, D>) -> Vec<usize> {
    let dims = tensor.dims().to_vec();
    let values: Vec<f32> = tensor
        .into_data()
        .convert::<f32>()
        .into_vec()
        .expect("tensor converts to f32");
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.f32.bin")), bytes).expect("write tensor");
    dims
}

fn write_i64(dir: &Path, name: &str, tensor: Tensor<B, 1, Int>) -> Vec<usize> {
    let dims = tensor.dims().to_vec();
    let values: Vec<i64> = tensor
        .into_data()
        .convert::<i64>()
        .into_vec()
        .expect("tensor converts to i64");
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.i64.bin")), bytes).expect("write tensor");
    dims
}

fn main() {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: dump_synthetic_run <out-dir>"),
    );
    std::fs::create_dir_all(&out).expect("create the dump dir");

    let device = Default::default();
    let config = config(&out);
    let (model, batches): (LoraMlp<B>, _) = synthetic_run_inputs::<B>(&config, &device);

    let mut shapes = serde_json::Map::new();
    let mut record = |name: &str, dims: Vec<usize>| {
        shapes.insert(name.to_string(), serde_json::json!(dims));
    };

    // Weights, in burn's `[d_in, d_out]` layout — the Python side owns every
    // transpose (torch `Linear.weight` is `[out, in]`), exactly as in
    // `reference/lora_reference.py`.
    record(
        "fc1_weight",
        write_f32(&out, "fc1_weight", model.fc1.weight.val()),
    );
    let fc1_bias = model.fc1.bias.as_ref().expect("fc1 has a bias").val();
    record("fc1_bias", write_f32(&out, "fc1_bias", fc1_bias));
    record(
        "fc2_base_weight",
        write_f32(&out, "fc2_base_weight", model.fc2.base.weight.val()),
    );
    let fc2_bias = model
        .fc2
        .base
        .bias
        .as_ref()
        .expect("fc2 base has a bias")
        .val();
    record("fc2_base_bias", write_f32(&out, "fc2_base_bias", fc2_bias));
    record(
        "lora_a_weight",
        write_f32(&out, "lora_a_weight", model.fc2.lora_a.weight.val()),
    );
    record(
        "lora_b_weight",
        write_f32(&out, "lora_b_weight", model.fc2.lora_b.weight.val()),
    );

    // Only the batches the run actually consumes (`steps <= batches.len()`, so
    // no cycling) — dumping all 31 would triple the size for nothing.
    let used = (STEPS as usize).min(batches.len());
    for (i, (x, y)) in batches.iter().take(used).enumerate() {
        record(
            &format!("batch_{i}_x"),
            write_f32(&out, &format!("batch_{i}_x"), x.clone()),
        );
        record(
            &format!("batch_{i}_y"),
            write_i64(&out, &format!("batch_{i}_y"), y.clone()),
        );
    }

    let manifest = serde_json::json!({
        "hyperparams": {
            "seed": SEED,
            "steps": STEPS,
            "rank": RANK,
            "alpha": ALPHA,
            "lr": LR,
            "weight_decays": WEIGHT_DECAYS,
            "scaling": ALPHA as f64 / RANK as f64,
        },
        "batches_dumped": used,
        "shapes": shapes,
    });
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).expect("manifest serializes"),
    )
    .expect("write the manifest");
}
