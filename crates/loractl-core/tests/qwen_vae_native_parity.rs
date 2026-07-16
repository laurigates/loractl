//! ComfyUI-native (Qwen/WAN) keyed VAE parity: the same tiny Qwen-Image VAE
//! weights, loaded under BOTH the diffusers `AutoencoderKLQwenImage` scheme and
//! the ComfyUI-native Wan-VAE scheme, must produce **identical** latents
//! (Phase 4 of the ComfyUI-scattered-file arc, issue #25).
//!
//! A ComfyUI Qwen-Image VAE file names the same weights under a different
//! state-dict *scheme* than the diffusers file [`QwenVae`] mirrors (top-level
//! `conv1`/`conv2`, `{…}.middle.*.residual.*`, flat `downsamples`/`upsamples`,
//! …). [`QwenVae::native_key_remap`] maps that scheme onto the exact burn paths
//! the diffusers [`QwenVae::key_remap`] reaches. `tests/fixtures/
//! tiny-qwen-vae-native/qwen_image_vae.safetensors` is the committed diffusers
//! tiny fixture **re-keyed** into native naming (regenerate with
//! `just vae-native-reference`), so it carries byte-identical tensor data — only
//! the keys differ.
//!
//! A subtly-wrong rename table would either bail (missing/unused) or load
//! silently-*wrong* weights. So the guard is encode-parity, not merely "it
//! loaded": both models encode the SAME input and their latents must match
//! bit-for-bit (same weights, same backend, same math — only the load keys
//! differed). The native load additionally must be complete: zero errors, zero
//! missing params, zero unused file tensors, all 154 fixture tensors applied.

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use burn_store::{ApplyResult, KeyRemapper, ModuleSnapshot, SafetensorsStore};
use loractl_core::qwen_vae::{QwenVae, QwenVaeConfig};
use serde::Deserialize;

/// Plain (no-autodiff) CPU backend — a pure forward-parity check.
type B = NdArray;

const GOLDEN: &str = include_str!("fixtures/qwen_vae_tiny_golden.json");
const DIFFUSERS: &str = "tests/fixtures/tiny-qwen-vae/diffusion_pytorch_model.safetensors";
const NATIVE: &str = "tests/fixtures/tiny-qwen-vae-native/qwen_image_vae.safetensors";

/// The `input` + `input_shape` fields of the shared VAE golden — a fixed seeded
/// image; the two loads are compared on it, not against the golden itself.
#[derive(Deserialize)]
struct Golden {
    input: Vec<f32>,
    input_shape: Vec<usize>,
}

/// Load the tiny VAE fixture at `path` through `remap`, returning the model and
/// the load result so callers can assert completeness.
fn load(path: &str, remap: Vec<(&str, &str)>) -> (QwenVae<B>, ApplyResult) {
    let device = Default::default();
    let mut model = QwenVae::<B>::init(QwenVaeConfig::tiny(), &device);
    let remapper = KeyRemapper::from_patterns(remap).expect("valid remap patterns");
    let mut store = SafetensorsStore::from_file(path).remap(remapper);
    let result = model.load_from(&mut store).expect("safetensors load");
    (model, result)
}

/// Flatten a burn tensor to a row-major `Vec<f32>`.
fn flatten<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().into_vec::<f32>().unwrap()
}

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

#[test]
fn native_keyed_vae_matches_diffusers_keyed_vae() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden json");
    let device = Default::default();

    // The diffusers-keyed load (the existing, byte-identical path).
    let (diffusers_vae, diffusers_res) = load(DIFFUSERS, QwenVae::<B>::key_remap().to_vec());
    assert!(
        diffusers_res.errors.is_empty(),
        "diffusers load errors: {:?}",
        diffusers_res.errors
    );

    // The native-keyed load (the feature under test) — must be COMPLETE: no
    // errors, no missing params, no unused file tensors, every tensor applied.
    let (native_vae, native_res) = load(NATIVE, QwenVae::<B>::native_key_remap());
    assert!(
        native_res.errors.is_empty(),
        "native load errors: {:?}",
        native_res.errors
    );
    assert!(
        native_res.missing.is_empty(),
        "native load missing params (rename table is incomplete): {:?}",
        native_res.missing
    );
    assert!(
        native_res.unused.is_empty(),
        "native load unused file tensors (rename table maps to a nonexistent \
         param, or a key was dropped): {:?}",
        native_res.unused
    );
    assert_eq!(
        native_res.applied.len(),
        diffusers_res.applied.len(),
        "native load applied {} tensors; diffusers applied {} — the two fixtures \
         hold the same weights and must apply the same count",
        native_res.applied.len(),
        diffusers_res.applied.len()
    );

    // The shared seeded input image.
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

    // Encode-parity: same weights → BIT-IDENTICAL latents. Any nonzero diff
    // means the native table landed a weight in the wrong place.
    let diffusers_latent = flatten(diffusers_vae.encode(input.clone()));
    let native_latent = flatten(native_vae.encode(input));
    let diff = max_abs_diff(&diffusers_latent, &native_latent);
    assert_eq!(
        diff, 0.0,
        "native vs diffusers latent max|Δ| = {diff:e} — the native rename table \
         is subtly wrong (a weight loaded into the wrong slot)"
    );
    eprintln!(
        "native-keyed VAE parity: {} latents, max|Δ| = {diff:e} (bit-identical), \
         {} tensors applied cleanly",
        native_latent.len(),
        native_res.applied.len()
    );
}
