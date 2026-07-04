//! Deterministic sampling from a [`LoraMlp`] (milestone 4, #3).
//!
//! `LoraMlp` is a synthetic/MNIST-shaped **classifier**, not a language
//! model — there is no tokenizer, and real text generation is deliberately
//! out of scope until a real base language model lands (see
//! `docs/adrs/0001-first-real-target-model.md` and
//! `docs/adrs/0002-adapter-format-and-sample-semantics.md`). So "sampling"
//! here means: turn a seed into a deterministic synthetic input vector, run
//! it through the model, and report the resulting logits/predicted class.
//! The CLI's `--prompt` flag deterministically seeds that input (via
//! [`seed_from_prompt`]) rather than pretending to generate text — an
//! honest, reproducible effect instead of a faked one.
//!
//! This module deliberately never touches burn's global device RNG (no
//! `Backend::seed` call anywhere here): that keeps it safe to call from *two*
//! very different call sites without any ordering hazard —
//! [`crate::adapter::load_adapter`]'s cold CLI path (which reseeds the
//! device for the frozen-base reconstruction) and [`crate::burn_trainer`]'s
//! in-training validation-sample path (which has a live model already built
//! from a seed advanced far past construction). A [`splitmix64`]-derived
//! generator keyed purely by an explicit `u64` seed sidesteps that entirely.
//!
//! [`splitmix64`]: https://prng.di.unimi.it/splitmix64.c

use crate::model::LoraMlp;
use burn::backend::NdArray;
use burn::tensor::{Device, Tensor, TensorData};
use serde::Serialize;

/// The result of one deterministic sample run.
///
/// `Serialize` is derived (not hand-implemented at each call site) so that
/// `burn_trainer.rs`'s in-training `sample-{step}.json` report can reuse this
/// struct's own field names directly — a future field added here (e.g. a
/// confidence score) can't silently be missed by a hand-copied field list.
#[derive(Debug, Clone, Serialize)]
pub struct SampleOutput {
    /// The argmax class over `logits`.
    pub predicted_class: usize,
    /// Raw logits, in class order.
    pub logits: Vec<f32>,
}

/// One step of the splitmix64 PRNG. Intentionally simple and NOT
/// statistically rigorous — this generates a toy/demo input, in the same
/// honestly-documented spirit as `burn_trainer.rs`'s own synthetic Gaussian
/// blobs (`synthetic_batches`): a convenient, deterministic stand-in, not a
/// real dataset or a claim of high-quality randomness.
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Generate a deterministic `Vec<f32>` of length `len` from `seed`, scaled to
/// roughly `[-3, 3]` — the same rough spread as `burn_trainer.rs`'s
/// synthetic training centroids (`.mul_scalar(3.0)`), so a sample input looks
/// vaguely like a training-distribution point without being one.
fn synthetic_input(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            let bits = splitmix64_next(&mut state);
            // Top 24 bits -> a uniform value in [0, 1) (24 bits is exactly
            // representable in an f32 mantissa), remapped to [-3, 3).
            let unit = (bits >> 40) as f32 / (1u32 << 24) as f32;
            unit * 6.0 - 3.0
        })
        .collect()
}

/// Run one deterministic sample through `model`.
///
/// `d_in` is derived from `model.fc1.weight`'s own shape — never a
/// cross-module constant — so this works unmodified whether called cold from
/// the CLI (a freshly [`load_adapter`](crate::adapter::load_adapter)ed model)
/// or from inside [`crate::burn_trainer`]'s in-training validation-sample
/// path (a live model already in scope). Both call sites use the same
/// concrete `NdArray` backend today (this crate is single-model,
/// single-backend), so — unlike [`crate::adapter::load_adapter`], which
/// genuinely is exercised with a second (autodiff) backend — this isn't
/// backend-generic.
pub fn run_sample(model: &LoraMlp<NdArray>, seed: u64, device: &Device<NdArray>) -> SampleOutput {
    let d_in = model.fc1.weight.dims()[0];
    let input = synthetic_input(seed, d_in);
    let x = Tensor::<NdArray, 2>::from_data(TensorData::new(input, [1, d_in]), device);
    let logits = model.forward(x);

    let logits: Vec<f32> = logits
        .into_data()
        .convert::<f32>()
        .into_vec()
        .expect("logits tensor data converts to Vec<f32>");
    let predicted_class = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);

    SampleOutput {
        predicted_class,
        logits,
    }
}

/// Derive a deterministic sample seed from an optional CLI `--prompt`.
///
/// `None` (no `--prompt` given) always maps to a fixed, documented default
/// seed of `0`. `Some(prompt)` hashes the prompt's UTF-8 bytes with FNV-1a,
/// implemented by hand rather than reaching for
/// `std::collections::hash_map::DefaultHasher` — the standard library
/// explicitly does **not** guarantee `DefaultHasher`'s algorithm or output is
/// stable across Rust versions, which would silently break "the same prompt
/// always reproduces the same sample" the next time the toolchain is
/// upgraded. FNV-1a's algorithm is a fixed public specification, so this
/// stays stable forever.
pub fn seed_from_prompt(prompt: Option<&str>) -> u64 {
    match prompt {
        None => 0,
        Some(s) => fnv1a(s.as_bytes()),
    }
}

/// FNV-1a: `hash = offset_basis; for byte in data { hash = (hash ^ byte) * prime }`.
fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes.iter().fold(OFFSET_BASIS, |hash, &byte| {
        (hash ^ byte as u64).wrapping_mul(PRIME)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_prompt_is_the_fixed_default_seed() {
        assert_eq!(seed_from_prompt(None), 0);
    }

    #[test]
    fn same_prompt_is_deterministic() {
        let a = seed_from_prompt(Some("the quick brown fox"));
        let b = seed_from_prompt(Some("the quick brown fox"));
        assert_eq!(a, b);
    }

    #[test]
    fn different_prompts_give_different_seeds() {
        assert_ne!(
            seed_from_prompt(Some("hello")),
            seed_from_prompt(Some("world"))
        );
    }
}
