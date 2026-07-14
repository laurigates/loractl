//! Opt-in encode/decode parity against the **real** `Qwen/Qwen-Image` VAE —
//! the exact checkpoint Krea 2's `autoencoder.py` wraps (M9, #20 acceptance:
//! real weights load with `errors`/`missing` empty + encode parity).
//!
//! Gated behind the `qwen-vae-real` feature AND `#[ignore]`, so the default
//! `cargo test` never compiles or runs it. The real VAE's safetensors (~500 MB
//! as f32) and its golden are **not** checked in; both are produced locally
//! by:
//!
//! ```sh
//! just vae-real-reference   # downloads Qwen/Qwen-Image's vae via diffusers,
//!                           # re-saves it as f32 safetensors + staged golden
//! just test-vae-real        # runs THIS test
//! ```
//!
//! It loads those real weights into the *same* [`QwenVae`] used by the tiny
//! test (only the config differs — [`QwenVaeConfig::qwen_image`]) and asserts
//! encode and decode parity, so it exercises the exact production loading and
//! forward path at full scale. If the fixtures are absent it fails with a
//! message pointing at `just vae-real-reference` rather than fabricating
//! numbers.
#![cfg(feature = "qwen-vae-real")]

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use loractl_core::qwen_vae::{QwenVae, QwenVaeConfig};
use serde::Deserialize;
use std::path::Path;

type B = NdArray;

const SAFETENSORS: &str = "tests/fixtures/qwen-vae-real/diffusion_pytorch_model.safetensors";
const GOLDEN: &str = "tests/fixtures/qwen_vae_real_golden.json";

#[derive(Deserialize)]
struct Golden {
    input: Vec<f32>,
    input_shape: Vec<usize>,
    safetensors_keys: Vec<String>,
    latent_norm: Vec<f32>,
    latent_norm_shape: Vec<usize>,
    decoded: Vec<f32>,
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

#[test]
#[ignore = "opt-in: needs the real Qwen-Image VAE fixtures produced by `just vae-real-reference` (network + torch)"]
fn real_qwen_vae_encode_decode_matches_diffusers_golden() {
    use burn_store::{KeyRemapper, ModuleSnapshot, SafetensorsStore};

    if !Path::new(SAFETENSORS).exists() || !Path::new(GOLDEN).exists() {
        panic!(
            "real Qwen-Image VAE fixtures missing ({SAFETENSORS} / {GOLDEN}).\n\
             Generate them first with:  just vae-real-reference"
        );
    }

    let golden: Golden =
        serde_json::from_str(&std::fs::read_to_string(GOLDEN).unwrap()).expect("parse real golden");
    let device = Default::default();

    let mut model = QwenVae::<B>::init(QwenVaeConfig::qwen_image(), &device);
    let remapper = KeyRemapper::from_patterns(QwenVae::<B>::key_remap().to_vec())
        .expect("valid remap patterns");
    let mut store = SafetensorsStore::from_file(SAFETENSORS).remap(remapper);
    let result = model.load_from(&mut store).expect("safetensors load");

    // Acceptance (#20): the real checkpoint maps cleanly — nothing errored,
    // nothing the module wants is missing, every checkpoint tensor applied.
    assert!(result.errors.is_empty(), "load errors: {:?}", result.errors);
    assert!(
        result.missing.is_empty(),
        "unexpected missing params: {:?}",
        result.missing
    );
    assert_eq!(
        result.applied.len(),
        golden.safetensors_keys.len(),
        "expected all {} checkpoint tensors applied, got {}",
        golden.safetensors_keys.len(),
        result.applied.len()
    );

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

    // Encode parity on the normalized latent (what training consumes).
    let latent = model
        .encode(input)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let enc_diff = max_abs_diff(&latent, &golden.latent_norm);
    let enc_cos = cosine(&latent, &golden.latent_norm);
    eprintln!("real encode: max|Δ| = {enc_diff:e}, cosine = {enc_cos:.8}");

    // Decode parity from the GOLDEN latent (isolated from encode drift).
    let golden_latent = Tensor::<B, 1>::from_data(
        TensorData::new(golden.latent_norm.clone(), [golden.latent_norm.len()]),
        &device,
    )
    .reshape([
        golden.latent_norm_shape[0],
        golden.latent_norm_shape[1],
        golden.latent_norm_shape[3],
        golden.latent_norm_shape[4],
    ]);
    let image = model
        .decode(golden_latent)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let dec_diff = max_abs_diff(&image, &golden.decoded);
    let dec_cos = cosine(&image, &golden.decoded);
    eprintln!("real decode: max|Δ| = {dec_diff:e}, cosine = {dec_cos:.8}");

    // Real-scale accumulation drift is larger than the tiny fixture's; the
    // cosine backstops are the primary claim, the max|Δ| bound the alarm.
    assert!(
        enc_diff <= 5e-3,
        "encode max|Δ| = {enc_diff:e} exceeds 5e-3"
    );
    assert!(
        dec_diff <= 5e-3,
        "decode max|Δ| = {dec_diff:e} exceeds 5e-3"
    );
    assert!(
        enc_cos > 0.9999,
        "encode cosine {enc_cos} must exceed 0.9999"
    );
    assert!(
        dec_cos > 0.9999,
        "decode cosine {dec_cos} must exceed 0.9999"
    );
}
