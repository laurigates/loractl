//! Opt-in conditioning parity against the **real** Krea-2-Raw text encoder —
//! the exact shipped Qwen3-VL weights + tokenizer Krea 2 conditions on
//! (M10, #21 acceptance: real weights load with `errors`/`missing` empty +
//! caption → embedding parity).
//!
//! Gated behind the `qwen3vl-real` feature AND `#[ignore]`. The fixtures
//! (~18 GB f32 re-save + goldens) are **not** checked in; produce them with:
//!
//! ```sh
//! just qwen3vl-real-reference   # downloads krea/Krea-2-Raw's text_encoder
//!                               # + tokenizer, dumps goldens
//! just test-qwen3vl-real        # runs THIS test (release build)
//! ```
//!
//! This exercises the full production path: [`Qwen3VlConditioner`] tokenizes
//! the golden's caption through the krea-2 chat template (asserting the token
//! ids match the reference tokenizer exactly — the Rust-side tokenizer
//! parity), the encoder loads the real checkpoint text-only (dropping
//! `visual.*`, transposing `nn.Linear`s), and the resulting conditioning
//! stack is compared against the reference stack.
#![cfg(feature = "qwen3vl-real")]

use burn::backend::NdArray;
use loractl_core::qwen3vl::{Qwen3VlConditioner, Qwen3VlConfig, Qwen3VlEncoder};
use serde::Deserialize;
use std::path::Path;

type B = NdArray;

const SAFETENSORS: &str = "tests/fixtures/qwen3vl-real/model.safetensors";
const TOKENIZER: &str = "tests/fixtures/qwen3vl-real/tokenizer/tokenizer.json";
const GOLDEN_JSON: &str = "tests/fixtures/qwen3vl_real_golden.json";
const GOLDEN_ST: &str = "tests/fixtures/qwen3vl_real_golden.safetensors";

#[derive(Deserialize)]
struct Golden {
    caption: String,
    input_ids: Vec<i64>,
    input_shape: Vec<usize>,
    attention_mask: Vec<i64>,
    select_layers: Vec<usize>,
    max_length: usize,
    conditioning_shape: Vec<usize>,
    safetensors_keys: Vec<String>,
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Read a named f32 tensor out of the golden safetensors file.
fn read_f32(st: &safetensors::SafeTensors, name: &str) -> Vec<f32> {
    let view = st.tensor(name).expect("golden tensor present");
    assert_eq!(view.dtype(), safetensors::Dtype::F32);
    view.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Read a named i64 tensor out of the golden safetensors file.
fn read_i64(st: &safetensors::SafeTensors, name: &str) -> Vec<i64> {
    let view = st.tensor(name).expect("golden tensor present");
    assert_eq!(view.dtype(), safetensors::Dtype::I64);
    view.data()
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "opt-in: needs the real Krea-2-Raw text-encoder fixtures produced by `just qwen3vl-real-reference` (network + torch)"]
fn real_qwen3vl_conditioning_matches_transformers_golden() {
    use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

    for f in [SAFETENSORS, TOKENIZER, GOLDEN_JSON, GOLDEN_ST] {
        if !Path::new(f).exists() {
            panic!(
                "real qwen3vl fixture missing ({f}).\n\
                 Generate them first with:  just qwen3vl-real-reference"
            );
        }
    }

    let golden: Golden = serde_json::from_str(&std::fs::read_to_string(GOLDEN_JSON).unwrap())
        .expect("parse real golden");
    let config = Qwen3VlConfig::krea2_4b();
    assert_eq!(golden.select_layers, config.select_layers);

    let device = Default::default();
    let mut model = Qwen3VlEncoder::<B>::init(config.clone(), &device);
    let mut store = SafetensorsStore::from_file(SAFETENSORS)
        .with_regex(Qwen3VlEncoder::<B>::load_filter())
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true); // visual.*, layer 35, and the final norm are unloaded by design
    let result = model.load_from(&mut store).expect("safetensors load");

    // Acceptance (#21): the real checkpoint maps cleanly text-only.
    assert!(result.errors.is_empty(), "load errors: {:?}", result.errors);
    assert!(
        result.missing.is_empty(),
        "unexpected missing params: {:?}",
        result.missing
    );
    let expected = 1 + config.num_layers() * 11; // embed + 35 layers x 11 tensors
    assert_eq!(
        result.applied.len(),
        expected,
        "expected {expected} text-trunk tensors applied, got {}",
        result.applied.len()
    );
    assert!(
        golden
            .safetensors_keys
            .iter()
            .any(|k| k.starts_with("visual.")),
        "real checkpoint should carry the (dropped) vision tower"
    );

    // Tokenizer parity: the Rust conditioner's template + tokenizer must
    // reproduce the reference's exact ids and mask for the same caption.
    let conditioner = Qwen3VlConditioner::new(model, Path::new(TOKENIZER), golden.max_length)
        .expect("load tokenizer");
    let (ids, mask, [b, s]) = conditioner
        .tokenize(&[golden.caption.as_str()])
        .expect("tokenize caption");
    assert_eq!([b, s].to_vec(), golden.input_shape, "token grid shape");
    assert_eq!(
        ids, golden.input_ids,
        "token ids must match the reference tokenizer"
    );
    assert_eq!(mask, golden.attention_mask, "attention mask must match");

    // Full caption → conditioning parity.
    let (conditioning, mask_sliced) = conditioner
        .encode_captions(&[golden.caption.as_str()], &device)
        .expect("encode caption");
    assert_eq!(conditioning.dims().to_vec(), golden.conditioning_shape);
    let got = conditioning
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();

    let bytes = std::fs::read(GOLDEN_ST).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let want = read_f32(&st, "conditioning");

    // The conditioner's sliced mask (what the MMDiT consumes) must equal the
    // reference's, element for element.
    let got_mask: Vec<i64> = mask_sliced
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap();
    assert_eq!(
        got_mask,
        read_i64(&st, "mask_sliced"),
        "sliced conditioning mask must match the reference"
    );

    let diff = max_abs_diff(&got, &want);
    let cos = cosine(&got, &want);
    eprintln!("real conditioning: max|Δ| = {diff:e}, cosine = {cos:.8}");

    // Real-scale accumulation drift across a 4B trunk is larger than the
    // tiny fixture's; the cosine backstop is the primary claim.
    assert!(diff <= 2e-2, "conditioning max|Δ| = {diff:e} exceeds 2e-2");
    assert!(cos > 0.9999, "conditioning cosine {cos} must exceed 0.9999");
}
