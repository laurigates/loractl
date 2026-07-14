//! Offline encode/decode parity: the hand-built burn Qwen-Image VAE vs. the
//! diffusers reference, on identical weights (M9, #20 — acceptance a + b).
//!
//! A real `AutoencoderKLQwenImage` at a tiny fixed config (seed 9) was dumped
//! to `tests/fixtures/tiny-qwen-vae/diffusion_pytorch_model.safetensors` with
//! staged golden activations in `tests/fixtures/qwen_vae_tiny_golden.json`
//! (regenerate with `just vae-reference`). This test loads those *same*
//! weights into [`loractl_core::QwenVae`] and asserts the burn encode and
//! decode reproduce the golden — so both frameworks run identical parameters
//! and differ only by f32 rounding, fully offline on every `cargo test`.
//!
//! Parity is brought up **stage by stage** on each path (encode: conv_in →
//! down trunk → mid block → moments → latent; decode: conv_in → mid block →
//! image), so a mismatch pinpoints the faulty stage. The decode path is fed
//! the *golden* normalized latent, isolating it from any encode drift.
//! Tolerance-free backstops (latent argmax + cosine similarity) guard against
//! a widened tolerance masking a real error.

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use loractl_core::qwen_vae::{QwenVae, QwenVaeConfig};
use serde::Deserialize;

/// Plain (no-autodiff) CPU backend — this is a pure forward-parity check.
type B = NdArray;

const GOLDEN: &str = include_str!("fixtures/qwen_vae_tiny_golden.json");
const SAFETENSORS: &str = "tests/fixtures/tiny-qwen-vae/diffusion_pytorch_model.safetensors";

#[derive(Deserialize)]
struct Golden {
    input: Vec<f32>,
    input_shape: Vec<usize>,
    latents_mean: Vec<f64>,
    latents_std: Vec<f64>,
    safetensors_keys: Vec<String>,
    enc_conv_in: Vec<f32>,
    enc_down: Vec<f32>,
    enc_mid: Vec<f32>,
    moments: Vec<f32>,
    latent_mode: Vec<f32>,
    latent_norm: Vec<f32>,
    latent_norm_shape: Vec<usize>,
    dec_conv_in: Vec<f32>,
    dec_mid: Vec<f32>,
    decoded: Vec<f32>,
}

/// Load the tiny VAE fixture into a burn [`QwenVae`], asserting the state-dict
/// mapping is clean: no load errors, no missing parameters, and every fixture
/// tensor applied.
fn load_tiny(expected_tensors: usize) -> QwenVae<B> {
    use burn_store::{KeyRemapper, ModuleSnapshot, SafetensorsStore};

    let device = Default::default();
    let mut model = QwenVae::<B>::init(QwenVaeConfig::tiny(), &device);

    // The ONLY remapping the VAE needs: flattening the reference's
    // `nn.Sequential` index on the resample convs (`resample.1` → `resample`).
    // Everything else — convs in PyTorch [out, in, k…] layout, RMS-norm
    // `gamma`s — loads by name with NO transpose.
    let remapper = KeyRemapper::from_patterns(QwenVae::<B>::key_remap().to_vec())
        .expect("valid resample remap patterns");
    let mut store = SafetensorsStore::from_file(SAFETENSORS).remap(remapper);

    let result = model.load_from(&mut store).expect("safetensors load");

    assert!(result.errors.is_empty(), "load errors: {:?}", result.errors);
    assert!(
        result.missing.is_empty(),
        "unexpected missing params: {:?}",
        result.missing
    );
    assert_eq!(
        result.applied.len(),
        expected_tensors,
        "expected all {} fixture tensors applied, got {}",
        expected_tensors,
        result.applied.len()
    );

    model
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

/// Flatten a burn tensor to a row-major `Vec<f32>`. The goldens' 5-D video
/// shapes (`T = 1`) flatten identically to the port's squeezed 4-D shapes.
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
fn tiny_qwen_vae_encode_decode_matches_diffusers_golden() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden json");
    let config = QwenVaeConfig::tiny();

    // Sanity: the fixture's latent stats are the tiny config's — the Rust
    // config and the Python TINY_CFG must not drift apart.
    assert_eq!(golden.latents_mean, config.latents_mean);
    assert_eq!(golden.latents_std, config.latents_std);

    let device = Default::default();
    let model = load_tiny(golden.safetensors_keys.len());

    let input = Tensor::<B, 1>::from_data(
        TensorData::new(golden.input.clone(), [golden.input.len()]),
        &device,
    )
    .reshape([
        golden.input_shape[0],
        golden.input_shape[1],
        golden.input_shape[2],
        golden.input_shape[3],
    ]);

    // Pinned tolerance. Observed max|Δ| across all stages on this fixture is
    // reported per stage below; pinned with margin over f32 rounding drift.
    let tol = 1e-4f32;

    // ---- Encode, stage by stage. ----
    let enc = model.encode_trace(input);
    assert_stage(
        "enc_conv_in",
        &flatten(enc.after_conv_in),
        &golden.enc_conv_in,
        tol,
    );
    assert_stage("enc_down", &flatten(enc.after_down), &golden.enc_down, tol);
    assert_stage("enc_mid", &flatten(enc.after_mid), &golden.enc_mid, tol);
    assert_stage("moments", &flatten(enc.moments), &golden.moments, tol);
    assert_stage(
        "latent_mode",
        &flatten(enc.latent_mode),
        &golden.latent_mode,
        tol,
    );
    let latent = flatten(enc.latent);
    assert_stage("latent_norm", &latent, &golden.latent_norm, tol);

    // ---- Decode, stage by stage, from the GOLDEN normalized latent. ----
    let golden_latent = Tensor::<B, 1>::from_data(
        TensorData::new(golden.latent_norm.clone(), [golden.latent_norm.len()]),
        &device,
    )
    .reshape([
        golden.latent_norm_shape[0],
        golden.latent_norm_shape[1],
        // The golden is the 5-D video shape [b, z, 1, h', w']; T = 1 squeezes.
        golden.latent_norm_shape[3],
        golden.latent_norm_shape[4],
    ]);
    let dec = model.decode_trace(golden_latent);
    assert_stage(
        "dec_conv_in",
        &flatten(dec.after_conv_in),
        &golden.dec_conv_in,
        tol,
    );
    assert_stage("dec_mid", &flatten(dec.after_mid), &golden.dec_mid, tol);
    let image = flatten(dec.image);
    assert_stage("decoded", &image, &golden.decoded, tol);

    // ---- Tolerance-free backstops. ----
    assert_eq!(
        argmax(&latent),
        argmax(&golden.latent_norm),
        "latent argmax must match the golden"
    );
    let latent_cos = cosine(&latent, &golden.latent_norm);
    let image_cos = cosine(&image, &golden.decoded);
    assert!(
        latent_cos > 0.99999,
        "latent cosine similarity {latent_cos} must exceed 0.99999"
    );
    assert!(
        image_cos > 0.99999,
        "decoded-image cosine similarity {image_cos} must exceed 0.99999"
    );
    eprintln!("latent cosine = {latent_cos:.8}; image cosine = {image_cos:.8}");
}
