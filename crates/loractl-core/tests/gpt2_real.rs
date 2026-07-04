//! Opt-in forward-pass parity against **real pretrained** GPT-2 weights
//! (`openai-community/gpt2`) — the M3 (#2) pretrained-weights bonus on top of
//! the always-run tiny-fixture parity proof.
//!
//! This is gated behind the `gpt2-real` feature AND `#[ignore]`, so the default
//! `cargo test` never compiles or runs it. Unlike the tiny fixture, real GPT-2's
//! ~500 MB safetensors and its golden are **not** checked in; both are produced
//! locally by:
//!
//! ```sh
//! just gpt2-reference   # downloads openai-community/gpt2 via transformers,
//!                       # writes tests/fixtures/gpt2-real/model.safetensors
//!                       # and tests/fixtures/gpt2_real_golden.json
//! just test-gpt2-real   # runs THIS test
//! ```
//!
//! It loads those real weights into the *same* [`Gpt2`] used by the tiny test
//! (only the config differs — [`Gpt2Config::gpt2_small`]) and asserts logit
//! parity, so it exercises the exact production loading + forward path at full
//! scale. If the fixtures are absent it fails with a message pointing at
//! `just gpt2-reference` rather than fabricating numbers.
#![cfg(feature = "gpt2-real")]

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use loractl_core::gpt2::{Gpt2, Gpt2Config};
use serde::Deserialize;
use std::path::Path;

type B = NdArray;

const SAFETENSORS: &str = "tests/fixtures/gpt2-real/model.safetensors";
const GOLDEN: &str = "tests/fixtures/gpt2_real_golden.json";

#[derive(Deserialize)]
struct Golden {
    input_ids: Vec<i64>,
    logits: Vec<f32>,
    logits_shape: Vec<usize>,
}

#[test]
#[ignore = "opt-in: needs the real gpt2 fixtures produced by `just gpt2-reference` (network + torch)"]
fn real_gpt2_forward_matches_pytorch_golden() {
    use burn_store::{KeyRemapper, ModuleSnapshot, SafetensorsStore};

    if !Path::new(SAFETENSORS).exists() || !Path::new(GOLDEN).exists() {
        panic!(
            "real gpt2 fixtures missing ({SAFETENSORS} / {GOLDEN}).\n\
             Generate them first with:  just gpt2-reference"
        );
    }

    let golden: Golden =
        serde_json::from_str(&std::fs::read_to_string(GOLDEN).unwrap()).expect("parse real golden");
    let device = Default::default();

    let mut model = Gpt2::<B>::init(Gpt2Config::gpt2_small(), &device);
    let remapper = KeyRemapper::from_patterns(Gpt2::<B>::layernorm_key_remap().to_vec())
        .expect("valid remap patterns");
    let mut store = SafetensorsStore::from_file(SAFETENSORS)
        .allow_partial(true)
        .remap(remapper);
    let result = model
        .load_from(&mut store)
        .expect("load real gpt2 safetensors");
    assert!(result.errors.is_empty(), "load errors: {:?}", result.errors);
    assert!(
        result.missing.is_empty(),
        "unexpected missing params: {:?}",
        result.missing
    );

    let seq = golden.input_ids.len();
    let ids =
        Tensor::<B, 1, Int>::from_data(TensorData::new(golden.input_ids.clone(), [seq]), &device)
            .reshape([1, seq]);
    let logits = model.forward(ids);

    let vocab = model.config.vocab_size;
    assert_eq!(golden.logits_shape, vec![seq, vocab]);
    let got: Vec<f32> = logits.into_data().convert::<f32>().into_vec().unwrap();

    let max_diff = got
        .iter()
        .zip(&golden.logits)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    // fp32 accumulation over 12 layers drifts more than the tiny model; pin
    // generously and rely on the top-1 + cosine backstops. Report the observed
    // number so a widened tolerance is always visible.
    let tol = 1e-2f32;
    eprintln!("real gpt2 logits max|Δ| = {max_diff:e} (tol {tol:e})");
    assert!(
        max_diff <= tol,
        "real gpt2 logits max|Δ| = {max_diff:e} exceeds {tol:e}"
    );

    // Tolerance-free backstops on the last token.
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    };
    let last = seq - 1;
    assert_eq!(
        argmax(&got[last * vocab..(last + 1) * vocab]),
        argmax(&golden.logits[last * vocab..(last + 1) * vocab]),
        "real gpt2 last-token top-1 must match the golden"
    );
    let dot: f64 = got
        .iter()
        .zip(&golden.logits)
        .map(|(x, y)| *x as f64 * *y as f64)
        .sum();
    let na: f64 = got.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = golden
        .logits
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cos = dot / (na * nb);
    assert!(
        cos > 0.9999,
        "real gpt2 logits cosine {cos} must exceed 0.9999"
    );
    eprintln!("real gpt2 last-token top-1 matches; logits cosine = {cos:.8}");
}
