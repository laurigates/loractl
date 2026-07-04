//! End-to-end round-trip proof for the adapter-only safetensors format
//! (milestone 4, #3 — acceptance criterion 1).
//!
//! Trains a tiny [`LoraMlp`] for one real optimizer step first — `lora_b` is
//! zero-initialized, so saving/loading a freshly constructed adapter would
//! trivially "round-trip" even through a broken load path (zero stays zero
//! either way). After the step `lora_b` has genuinely moved, so a bit-exact
//! forward match after save+reload is real evidence the load path is
//! correct, not a coincidence of the initialization.
//!
//! Also asserts (offline, always-run):
//! - the safetensors file holds *exactly* the two trainable LoRA tensors —
//!   proving the adapter is lean, not a full checkpoint (see `adapter.rs`);
//! - the `<path>.json` sidecar's [`AdapterMeta`] round-trips the run's shape
//!   and seed;
//! - `sample::run_sample` is deterministic end-to-end on the reloaded model.

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Int, Tensor, TensorData, Tolerance};
use burn_store::{ModuleStore, SafetensorsStore};
use loractl_core::LoraMlp;
use loractl_core::adapter::{self, AdapterMeta};
use loractl_core::sample;
use std::collections::BTreeSet;
use std::path::PathBuf;

/// Autodiff-wrapped CPU backend for the one real training step.
type AB = Autodiff<NdArray>;
/// The plain (non-autodiff) backend the reloaded model lives on.
type TB = NdArray;

const SEED: u64 = 123;
const D_IN: usize = 8;
const HIDDEN: usize = 6;
const OUT: usize = 4;
const RANK: usize = 2;
const ALPHA: f64 = 8.0;

/// A unique temp output dir so concurrent test runs don't collide or litter
/// the repo. Removed on drop. (Mirrors `tests/convergence.rs`'s helper.)
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("loractl-{tag}-{}-{nanos}", std::process::id()));
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixed_input<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(
        TensorData::new(
            vec![0.1f32, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
            [1, D_IN],
        ),
        device,
    )
}

#[test]
fn adapter_round_trips_after_a_real_training_step() {
    let device: Device<AB> = Default::default();
    // Seed FIRST, then construct immediately — the same ordering
    // `adapter::load_adapter` depends on to reconstruct the frozen base from
    // the persisted seed alone.
    AB::seed(&device, SEED);
    let mut model = LoraMlp::<AB>::new(D_IN, HIDDEN, OUT, RANK, ALPHA, &device);

    // One real optimizer step: forward, cross-entropy loss against an
    // arbitrary fixed target, backward, one Adam step on the LoRA params.
    let x = fixed_input::<AB>(&device);
    let target = Tensor::<AB, 1, Int>::from_data(TensorData::new(vec![2i64], [1]), &device);
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let loss = loss_fn.forward(model.forward(x), target);
    let loss_value: f32 = loss.clone().into_scalar();
    assert!(loss_value.is_finite(), "loss must be finite");

    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let mut optim = AdamConfig::new().init::<AB, LoraMlp<AB>>();
    model = optim.step(0.5, model, grads);

    let pre_save = model.valid();
    let b_sum: f32 = pre_save.fc2.lora_b.weight.val().abs().sum().into_scalar();
    assert!(
        b_sum > 0.0,
        "lora_b must have moved off zero after the training step"
    );

    let out = TempDir::new("adapter-roundtrip");
    let adapter_path = out.0.join("adapter.safetensors");
    adapter::save_adapter(&pre_save, &adapter_path, SEED).expect("save adapter");

    // --- The file holds ONLY the two trainable LoRA tensors (criterion 4: a
    // lean adapter, not a full-model checkpoint). ---
    let mut store = SafetensorsStore::from_file(adapter_path.as_path());
    let keys: BTreeSet<String> = store
        .keys()
        .expect("read tensor keys")
        .into_iter()
        .collect();
    assert_eq!(
        keys,
        BTreeSet::from([
            "fc2.lora_a.weight".to_string(),
            "fc2.lora_b.weight".to_string(),
        ]),
        "the adapter file must hold exactly the two trainable LoRA tensors"
    );

    // --- The sidecar metadata round-trips the run's shape + seed. ---
    let sidecar_path = PathBuf::from(format!("{}.json", adapter_path.display()));
    let sidecar = std::fs::read_to_string(&sidecar_path).expect("read metadata sidecar");
    let meta: AdapterMeta = serde_json::from_str(&sidecar).expect("parse metadata sidecar");
    assert_eq!(meta.seed, SEED);
    assert_eq!(meta.rank, RANK as u32);
    assert_eq!(meta.alpha, ALPHA as f32);
    assert_eq!(meta.d_in, D_IN);
    assert_eq!(meta.hidden, HIDDEN);
    assert_eq!(meta.out, OUT);

    // --- Reload on a plain (non-autodiff) backend and compare forwards. ---
    let reloaded = adapter::load_adapter::<TB>(&adapter_path, &device).expect("load adapter");

    let expected = pre_save.forward(fixed_input::<TB>(&device));
    let actual = reloaded.forward(fixed_input::<TB>(&device));
    expected
        .into_data()
        .assert_approx_eq::<f32>(&actual.into_data(), Tolerance::default());

    // --- Determinism end-to-end: two `run_sample` calls on the reloaded
    // model with the same seed must be bit-identical. ---
    let s1 = sample::run_sample(&reloaded, 42, &device);
    let s2 = sample::run_sample(&reloaded, 42, &device);
    assert_eq!(s1.predicted_class, s2.predicted_class);
    assert_eq!(s1.logits, s2.logits);
}
