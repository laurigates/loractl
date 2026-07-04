//! Deterministic sampling — milestone 4 (#3), acceptance b.
//!
//! Shared logic behind both the periodic in-training validation sample (see
//! [`BurnTrainer`](crate::burn_trainer::BurnTrainer)) and the standalone
//! `loractl sample` CLI command: run one forward pass of a [`LoraMlp`] on a
//! deterministic, seed-derived synthetic input.
//!
//! **Why this never touches burn's global device RNG.** [`crate::adapter::load_adapter`]
//! reseeds the device (`B::seed`) to regenerate the frozen base
//! deterministically, and a live training loop has its own RNG state riding
//! on that same global seed. Sample-input generation must never disturb
//! either of those, so this module hand-rolls a small, dependency-free,
//! self-contained deterministic generator instead of `Tensor::random`/
//! `Distribution` — the seed here is purely local data, never fed through
//! `B::seed`.
//!
//! `LoraMlp` is a synthetic classifier with no tokenizer (see
//! `docs/adrs/0002-adapter-format-and-sample-semantics.md`), so "sampling"
//! here means exactly one deterministic forward pass, honestly labeled as
//! such — not text generation.

use crate::model::LoraMlp;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// FNV-1a, a well-known, stable-forever 64-bit hash.
///
/// Deliberately NOT `std::collections::hash_map::DefaultHasher`: its docs
/// explicitly say the algorithm is unspecified and may change across Rust
/// releases, which would silently break "the same prompt always reproduces
/// the same sample" the next time the toolchain changes.
fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    bytes
        .iter()
        .fold(OFFSET, |hash, &b| (hash ^ b as u64).wrapping_mul(PRIME))
}

/// Derive a deterministic sample seed from an optional CLI `--prompt`.
///
/// `None` (no prompt given) is seed `0`; `Some(s)` hashes `s` with FNV-1a so
/// the same prompt text always reproduces the same sample input and output.
/// This is an honest, reproducible effect — not a claim that the prompt's
/// *content* influences the output the way a language model would.
pub fn seed_from_prompt(prompt: Option<&str>) -> u64 {
    match prompt {
        None => 0,
        Some(s) => fnv1a64(s.as_bytes()),
    }
}

/// A tiny, dependency-free deterministic float generator (splitmix64), used
/// to build the sample input vector without touching burn's `Tensor::random`
/// or its global RNG.
///
/// Intentionally simple and non-Gaussian — fine for a demo/probe input, not a
/// statistical claim about the distribution.
fn splitmix64_vec(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            ((z as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32 * 3.0
        })
        .collect()
}

/// The result of one deterministic sample forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleOutput {
    /// `argmax` of [`logits`](Self::logits).
    pub predicted_class: usize,
    /// Raw logits, in class order.
    pub logits: Vec<f32>,
}

/// Run one forward pass of `model` on a deterministic, `seed`-derived input.
///
/// The input width is read off `model.fc1` (`d_in`), so this works for any
/// `LoraMlp` shape with no hardcoded constants.
pub fn run_sample<B: Backend>(model: &LoraMlp<B>, seed: u64, device: &B::Device) -> SampleOutput {
    let d_in = model.fc1.weight.dims()[0];
    let data = splitmix64_vec(seed, d_in);
    let input = Tensor::<B, 2>::from_data(TensorData::new(data, [1, d_in]), device);
    let logits = model.forward(input);
    let logits: Vec<f32> = logits.into_data().convert::<f32>().into_vec().unwrap();
    let predicted_class = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    SampleOutput {
        predicted_class,
        logits,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TB = NdArray;

    #[test]
    fn seed_from_prompt_is_deterministic() {
        assert_eq!(seed_from_prompt(None), 0);
        assert_eq!(seed_from_prompt(Some("x")), seed_from_prompt(Some("x")));
        assert_eq!(
            seed_from_prompt(Some("hello")),
            seed_from_prompt(Some("hello"))
        );
    }

    #[test]
    fn run_sample_is_deterministic() {
        let device = Default::default();
        let model = LoraMlp::<TB>::new(8, 6, 4, 2, 8.0, &device);

        let a = run_sample(&model, 42, &device);
        let b = run_sample(&model, 42, &device);

        assert_eq!(a, b, "same model + seed must produce byte-identical output");
    }
}
