//! Offline forward parity: the hand-built burn Qwen3-VL text encoder vs. the
//! transformers reference, on identical weights (M10, #21 — acceptance a + b).
//!
//! A real `Qwen3VLModel` at a tiny fixed config (seed 10) — WITH a tiny
//! vision tower, so the fixture carries `visual.*` keys — was dumped to
//! `tests/fixtures/tiny-qwen3vl/model.safetensors` with staged goldens in
//! `tests/fixtures/qwen3vl_tiny_golden.json` (regenerate with
//! `just qwen3vl-reference`). This test loads the text trunk into
//! [`loractl_core::Qwen3VlEncoder`] — proving the `^language_model\.`
//! drop-filter and the PyTorch `nn.Linear` transpose adapter — and asserts
//! the staged forward reproduces the golden, fully offline.
//!
//! The golden's batch has a RIGHT-PADDED row (interior mask zeros with a live
//! tail — the suffix-after-padding shape of the real conditioner), so parity
//! here pins key-padding masking and arange position ids, not just the
//! causal mask. A tolerance-free backstop (conditioning argmax + cosine)
//! guards against a tolerance masking a real error.

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use loractl_core::qwen3vl::{Qwen3VlConfig, Qwen3VlEncoder};
use serde::Deserialize;

/// Plain (no-autodiff) CPU backend — this is a pure forward-parity check.
type B = NdArray;

const GOLDEN: &str = include_str!("fixtures/qwen3vl_tiny_golden.json");
const SAFETENSORS: &str = "tests/fixtures/tiny-qwen3vl/model.safetensors";

#[derive(Deserialize)]
struct Golden {
    input_ids: Vec<i64>,
    input_shape: Vec<usize>,
    attention_mask: Vec<i64>,
    select_layers: Vec<usize>,
    prefix_idx: usize,
    num_hidden_states: usize,
    safetensors_keys: Vec<String>,
    after_embed: Vec<f32>,
    hidden_first_select: Vec<f32>,
    hidden_last_select: Vec<f32>,
    conditioning: Vec<f32>,
    conditioning_shape: Vec<usize>,
}

/// Max absolute difference between two equal-length flat slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Assert a burn activation matches the golden within `tol`, reporting the
/// observed max-abs diff so a widened tolerance is always visible.
fn assert_stage(name: &str, got: &[f32], want: &[f32], tol: f32) {
    let diff = max_abs_diff(got, want);
    assert!(diff <= tol, "{name}: max|Δ| = {diff:e} exceeds tol {tol:e}",);
    eprintln!("{name}: max|Δ| = {diff:e} (tol {tol:e})");
}

/// Flatten a burn tensor to a row-major `Vec<f32>`.
fn flatten<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().into_vec::<f32>().unwrap()
}

/// Cosine similarity of two equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

/// Index of the maximum element (the tolerance-free backstop's "argmax").
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

#[test]
fn tiny_qwen3vl_conditioning_matches_transformers_golden() {
    use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden json");
    let config = Qwen3VlConfig::tiny();

    // Config-drift guards: the fixture's aggregation setup is the tiny
    // config's, and hidden_states has embedding + every decoder layer (the
    // trunk builds only up to the last selected one).
    assert_eq!(golden.select_layers, config.select_layers);
    assert_eq!(golden.num_hidden_states, 5, "embedding + 4 tiny layers");
    assert!(
        golden
            .safetensors_keys
            .iter()
            .any(|k| k.starts_with("visual.")),
        "fixture must carry visual.* keys for the drop-filter proof"
    );

    let device = Default::default();
    let mut model = Qwen3VlEncoder::<B>::init(config.clone(), &device);

    // The production load recipe: keep only `language_model.*` (drops the
    // vision tower), transpose nn.Linear weights (PyTorchToBurnAdapter) —
    // unlike GPT-2 (Conv1D, no transpose) and the M9 VAE (convs, verbatim).
    let mut store = SafetensorsStore::from_file(SAFETENSORS)
        .with_regex(Qwen3VlEncoder::<B>::load_filter())
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true); // visual.* + post-select layers are unloaded by design
    let result = model.load_from(&mut store).expect("safetensors load");

    assert!(result.errors.is_empty(), "load errors: {:?}", result.errors);
    assert!(
        result.missing.is_empty(),
        "unexpected missing params: {:?}",
        result.missing
    );
    // The trunk wants embed_tokens + 3 layers × 11 tensors (layer 3 of the
    // fixture and its final norm are dead for select_layers [1, 3]).
    let expected = 1 + config.num_layers() * 11;
    assert_eq!(
        result.applied.len(),
        expected,
        "expected {expected} text-trunk tensors applied, got {}",
        result.applied.len()
    );

    let [b, s] = [golden.input_shape[0], golden.input_shape[1]];
    let ids =
        Tensor::<B, 2, Int>::from_data(TensorData::new(golden.input_ids.clone(), [b, s]), &device);
    let mask = Tensor::<B, 2, Int>::from_data(
        TensorData::new(golden.attention_mask.clone(), [b, s]),
        &device,
    );
    // The padded row is the point: key-padding masking must be live.
    assert!(
        golden.attention_mask.contains(&0),
        "golden must include a padded row"
    );

    let trace = model.forward_trace(ids, mask, golden.prefix_idx);

    // Pinned tolerance; observed max|Δ| reported per stage below.
    let tol = 1e-4f32;

    // ---- Stage by stage, so a failure localizes. ----
    assert_stage(
        "after_embed",
        &flatten(trace.after_embed),
        &golden.after_embed,
        tol,
    );
    assert_stage(
        "first_select",
        &flatten(trace.first_select),
        &golden.hidden_first_select,
        tol,
    );
    assert_stage(
        "last_select",
        &flatten(trace.last_select),
        &golden.hidden_last_select,
        tol,
    );
    assert_eq!(
        trace.conditioning.dims().to_vec(),
        golden.conditioning_shape,
        "conditioning stack shape must match the golden"
    );
    let conditioning = flatten(trace.conditioning);
    assert_stage("conditioning", &conditioning, &golden.conditioning, tol);

    // ---- Tolerance-free backstops. ----
    assert_eq!(
        argmax(&conditioning),
        argmax(&golden.conditioning),
        "conditioning argmax must match the golden"
    );
    let cos = cosine(&conditioning, &golden.conditioning);
    assert!(
        cos > 0.99999,
        "conditioning cosine similarity {cos} must exceed 0.99999"
    );
    eprintln!("conditioning cosine = {cos:.8}");
}
