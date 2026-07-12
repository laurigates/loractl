//! Safetensors adapter round-trip proof (milestone 4, issue #3 — acceptance
//! a & d).
//!
//! Trains a tiny [`LoraMlp`] for one real optimizer step (so `lora_b` moves
//! off zero — see [`train_one_step`] for why that matters), saves it via
//! [`save_adapter`], reloads it via [`load_adapter`] with a FRESH backend
//! instance, and asserts the reloaded model's forward pass matches the
//! pre-save model's forward pass on the same input. Also asserts the
//! adapter-only claim (only the two LoRA tensors are on disk) and that the
//! JSON sidecar round-trips.

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor, TensorData, Tolerance};
use loractl_core::LoraMlp;
use loractl_core::adapter::{AdapterMeta, load_adapter, save_adapter};
use loractl_core::config::TaskKind;
use loractl_core::sample::run_sample;
use std::path::PathBuf;
use std::sync::Mutex;

/// burn's ndarray backend keeps its RNG in a single process-global static
/// (`burn_ndarray::backend::SEED`), so any test that seeds it and relies on
/// what gets drawn afterward (as [`train_one_step`] + [`load_adapter`] do)
/// is not safe to run concurrently with another such test in the same
/// process — `cargo test` runs `#[test]` fns in parallel threads by default,
/// and an interleaved reseed/draw from a sibling test would silently corrupt
/// the sequence this file's determinism depends on. This lock serializes the
/// tests in this file against each other.
static RNG_LOCK: Mutex<()> = Mutex::new(());

/// Autodiff-wrapped CPU backend — a real training step needs gradients.
type AB = Autodiff<NdArray>;
/// Plain CPU backend — the reconstructed/reloaded models are inference-only.
type TB = NdArray;

const D_IN: usize = 8;
const HIDDEN: usize = 6;
const OUT: usize = 4;
const RANK: usize = 2;
const ALPHA: f64 = 8.0;
const SEED: u64 = 123;

/// A unique temp output dir so concurrent test runs don't collide or litter
/// the repo. Removed on drop — same convention as `convergence.rs`'s
/// `TempDir`. `save_adapter` creates the directory itself, so this struct
/// only ever needs to hold (and clean up) the path.
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

/// Build a fresh, seeded [`LoraMlp`] and run ONE real optimizer step so
/// `lora_b` moves off zero before it's saved.
///
/// This matters: `lora_b` is zero-initialized (see `lora.rs`), so a
/// round-trip test against a still-zero adapter would trivially "pass" even
/// with a completely broken load path — a zero adapter is a no-op regardless
/// of whether loading actually worked.
fn train_one_step(device: &burn::tensor::Device<AB>) -> LoraMlp<AB> {
    AB::seed(device, SEED);
    let mut model = LoraMlp::<AB>::new(D_IN, HIDDEN, OUT, RANK, ALPHA, 0.0, device);

    let x = Tensor::<AB, 2>::random([5, D_IN], Distribution::Default, device);
    let targets =
        Tensor::<AB, 1, Int>::from_data(TensorData::new(vec![0i64, 1, 2, 3, 0], [5]), device);
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let logits = model.forward(x);
    let loss = loss_fn.forward(logits, targets);
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let mut optim = AdamConfig::new().init::<AB, LoraMlp<AB>>();
    model = optim.step(1e-2, model, grads);

    let b_sum: f32 = model.fc2.lora_b.weight.val().abs().sum().into_scalar();
    assert!(
        b_sum > 0.0,
        "precondition: lora_b must move off zero before the round trip is meaningful"
    );
    model
}

#[test]
fn round_trip_forward_matches_pre_save() {
    let _guard = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let device: burn::tensor::Device<AB> = Default::default();
    let model = train_one_step(&device);
    let valid_model = model.valid();

    let out = TempDir::new("adapter-roundtrip");
    let path = out.0.join("adapter.safetensors");
    save_adapter(&valid_model, &path, SEED, TaskKind::Classification)
        .expect("save_adapter succeeds");

    // Reload with a FRESH backend instance/device — proves the reconstruction
    // doesn't rely on any state shared with `valid_model`.
    let fresh_device: burn::tensor::Device<TB> = Default::default();
    let reloaded = load_adapter::<TB>(&path, &fresh_device).expect("load_adapter succeeds");

    let probe = Tensor::<TB, 2>::from_data(
        TensorData::new(vec![0.1f32; D_IN], [1, D_IN]),
        &fresh_device,
    );
    let pre_save_logits = valid_model.forward(probe.clone());
    let reloaded_logits = reloaded.forward(probe);

    // `Tolerance::default()` resolves to `Tolerance::balanced()`: relative 0.5%
    // and absolute 1e-5 (see `burn_tensor::data::compare::Tolerance`). The
    // reconstruction is meant to be bit-identical (the same seed regenerates
    // `fc1`/`fc2.base` deterministically, and `lora_a`/`lora_b` are the exact
    // tensors read back off disk), but the reload deliberately runs through a
    // FRESH backend instance/device (see above) to prove the reconstruction
    // doesn't depend on shared state — so `default()`'s wider-than-`strict()`
    // tolerance also absorbs any legitimate floating-point summation-order
    // difference between the two backend instances, not just the intended
    // zero-divergence case. Contrast with `lora_reference.rs`'s
    // `Tolerance::absolute(1e-5)`, tuned for roundoff accumulated over 20
    // training steps rather than a single deterministic reconstruction.
    pre_save_logits
        .into_data()
        .assert_approx_eq::<f32>(&reloaded_logits.into_data(), Tolerance::default());

    // Also exercise the exact code path `loractl sample` runs
    // (load_adapter -> run_sample, see crates/loractl-cli/src/cli.rs's
    // `sample()`) so a regression in that glue — e.g. a shape mismatch from a
    // future model.rs change — fails here instead of only being caught by
    // manual testing (issue #3, acceptance criterion 2).
    let sample_out =
        run_sample(&reloaded, 0, &fresh_device).expect("run_sample succeeds on a reloaded adapter");
    assert!(
        sample_out.logits.iter().all(|l| l.is_finite()),
        "sample logits from a freshly loaded adapter must be finite"
    );
}

#[test]
fn saved_file_contains_only_the_lora_tensors() {
    use burn_store::{ModuleStore, SafetensorsStore};

    let _guard = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let device: burn::tensor::Device<AB> = Default::default();
    let model = train_one_step(&device).valid();

    let out = TempDir::new("adapter-keys");
    let path = out.0.join("adapter.safetensors");
    save_adapter(&model, &path, SEED, TaskKind::Classification).expect("save_adapter succeeds");

    let mut store = SafetensorsStore::from_file(&path);
    let mut keys = store.keys().expect("read adapter keys");
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "fc2.lora_a.weight".to_string(),
            "fc2.lora_b.weight".to_string()
        ],
        "only the trainable LoRA factors may be persisted — no frozen-base leakage"
    );
}

#[test]
fn sidecar_round_trips_meta() {
    let _guard = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let device: burn::tensor::Device<AB> = Default::default();
    let model = train_one_step(&device).valid();

    let out = TempDir::new("adapter-sidecar");
    let path = out.0.join("adapter.safetensors");
    save_adapter(&model, &path, SEED, TaskKind::Classification).expect("save_adapter succeeds");

    let mut sidecar = path.clone().into_os_string();
    sidecar.push(".json");
    let json = std::fs::read_to_string(&sidecar).expect("sidecar exists");
    let meta: AdapterMeta = serde_json::from_str(&json).expect("sidecar parses");

    assert_eq!(meta.seed, SEED);
    assert_eq!(meta.rank, RANK as u32);
    assert_eq!(meta.alpha, ALPHA as f32);
    assert_eq!(meta.d_in, D_IN);
    assert_eq!(meta.hidden, HIDDEN);
    assert_eq!(meta.out, OUT);
    assert_eq!(meta.task, TaskKind::Classification);
}
