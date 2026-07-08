//! Zero-init no-op at every injectable site (milestone 6, #17).
//!
//! Attaches a [`LoraAdapters`] set at *all four* projection sites across *all*
//! blocks of the loaded tiny GPT-2 and asserts the adapted forward reproduces
//! the base golden logits bit-for-bit. Because every delta's `B` is
//! zero-initialized, injection at any (and every) site is an exact no-op until
//! training moves the deltas — this is the whole-model generalization of the
//! single-site attach-integrity check in `gpt2_lora_step.rs`.
//!
//! Pure forward parity: plain (no-autodiff) backend, no freezing needed, reuses
//! the checked-in parity fixture. Also exercises `injectable_sites` +
//! `build_adapters` end to end.

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use loractl_core::adapters::build_adapters;
use loractl_core::config::{LoraConfig, TargetSpec};
use loractl_core::gpt2::{Gpt2, Gpt2Config};
use serde::Deserialize;

/// Plain (no-autodiff) CPU backend — a pure forward-parity check.
type B = NdArray;

const GOLDEN: &str = include_str!("fixtures/gpt2_tiny_golden.json");
const SAFETENSORS: &str = "tests/fixtures/tiny-gpt2/model.safetensors";

#[derive(Deserialize)]
struct Golden {
    input_ids: Vec<i64>,
    logits: Vec<f32>,
    logits_shape: Vec<usize>,
}

fn load_tiny() -> Gpt2<B> {
    use burn_store::{KeyRemapper, ModuleSnapshot, SafetensorsStore};

    let device = Default::default();
    let mut model = Gpt2::<B>::init(Gpt2Config::tiny(), &device);
    let remapper = KeyRemapper::from_patterns(Gpt2::<B>::layernorm_key_remap().to_vec())
        .expect("valid remap patterns");
    let mut store = SafetensorsStore::from_file(SAFETENSORS)
        .allow_partial(true)
        .remap(remapper);
    let result = model.load_from(&mut store).expect("safetensors load");
    assert!(result.errors.is_empty(), "load errors: {:?}", result.errors);
    model
}

#[test]
fn zero_init_adapters_at_all_sites_are_a_noop() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden");
    let device = Default::default();
    let model = load_tiny();

    // Attach zero-init deltas at every injectable site.
    let sites = model.injectable_sites();
    let cfg = LoraConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
        targets: vec![TargetSpec {
            pattern: r".*".to_string(),
            rank: None,
            alpha: None,
        }],
    };
    let set = build_adapters(&sites, &cfg, &device);
    assert_eq!(
        set.deltas.len(),
        Gpt2Config::tiny().n_layer * 4,
        "every site should get a delta"
    );

    let seq = golden.input_ids.len();
    let ids = Tensor::<B, 1, Int>::from_data(TensorData::new(golden.input_ids, [seq]), &device)
        .reshape([1, seq]);

    let adapted = model.forward_with_adapters(ids, &set);
    let [b, s, v] = adapted.dims();
    assert_eq!(vec![s, v], golden.logits_shape, "logits shape");
    assert_eq!(b, 1);

    let flat: Vec<f32> = adapted.into_data().convert::<f32>().into_vec().unwrap();
    let max_diff = flat
        .iter()
        .zip(&golden.logits)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff <= 1e-4,
        "zero-init adapters at every site must reproduce the base golden: max|Δ| = {max_diff:e}"
    );
}
