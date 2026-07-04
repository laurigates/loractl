//! Offline forward-pass parity: a hand-built burn GPT-2 vs. the PyTorch
//! reference, on identical weights (M3, #2 — acceptance a + b).
//!
//! A real HF `GPT2LMHeadModel` at a tiny fixed config (seed 1234) was dumped to
//! `tests/fixtures/tiny-gpt2/model.safetensors` with golden logits and
//! intermediate activations in `tests/fixtures/gpt2_tiny_golden.json`
//! (regenerate with `reference/gpt2_tiny_reference.py`). This test loads those
//! *same* weights into [`loractl_core::Gpt2`] and asserts the burn forward
//! reproduces the golden — so both frameworks run identical parameters and
//! differ only by f32 rounding. The weights + goldens are small enough to check
//! in, so this proves load + forward parity fully offline on every `cargo test`.
//!
//! Parity is brought up **stage by stage** (embeddings → block 0 → final
//! LayerNorm → logits): a mismatch pinpoints the faulty stage instead of only
//! showing a wrong final logit. A tolerance-free backstop (last-token argmax +
//! logits cosine similarity) guards against a tolerance masking a real error.

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use loractl_core::gpt2::{Gpt2, Gpt2Config};
use serde::Deserialize;

/// Plain (no-autodiff) CPU backend — this is a pure forward-parity check.
type B = NdArray;

const GOLDEN: &str = include_str!("fixtures/gpt2_tiny_golden.json");
const SAFETENSORS: &str = "tests/fixtures/tiny-gpt2/model.safetensors";

#[derive(Deserialize)]
struct Golden {
    input_ids: Vec<i64>,
    logits: Vec<f32>,
    logits_shape: Vec<usize>,
    hidden_after_embed: Vec<f32>,
    hidden_after_block0: Vec<f32>,
    hidden_after_lnf: Vec<f32>,
    hidden_shape: Vec<usize>,
    safetensors_keys: Vec<String>,
}

/// Load the tiny GPT-2 fixture into a burn [`Gpt2`], asserting the state-dict
/// mapping is clean: no load errors, and no missing parameters (the tied head
/// has no separate param, so nothing should be missing).
fn load_tiny() -> Gpt2<B> {
    use burn_store::{KeyRemapper, ModuleSnapshot, SafetensorsStore};

    let device = Default::default();
    let mut model = Gpt2::<B>::init(Gpt2Config::tiny(), &device);

    // The ONLY remapping GPT-2 needs: LayerNorm weight/bias -> burn gamma/beta.
    // Everything else loads by name with NO transpose (Conv1D weights are
    // already burn's [in, out] Linear layout).
    let remapper = KeyRemapper::from_patterns(Gpt2::<B>::layernorm_key_remap().to_vec())
        .expect("valid LayerNorm remap patterns");
    let mut store = SafetensorsStore::from_file(SAFETENSORS)
        .allow_partial(true) // tolerate the absent tied-head key (see module docs)
        .remap(remapper);

    let result = model.load_from(&mut store).expect("safetensors load");

    // Adversarial-review guard: assert on `errors` and `missing`, NOT on
    // `unused`. `unused` = source tensors with no matching param (HF causal-mask
    // buffers would land here and are expected); asserting it empty is spurious.
    assert!(result.errors.is_empty(), "load errors: {:?}", result.errors);
    // The tied head is implemented in the forward (logits = h · wteᵀ) with no
    // separate lm_head param, so nothing the module wants should be missing.
    assert!(
        result.missing.is_empty(),
        "unexpected missing params: {:?}",
        result.missing
    );
    // Every one of the 28 fixture tensors should have been applied.
    assert_eq!(
        result.applied.len(),
        28,
        "expected all 28 fixture tensors applied, got {}",
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

/// Flatten a burn tensor to a row-major `Vec<f32>` (matching the golden's flat
/// layout).
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

#[test]
fn tiny_gpt2_forward_matches_pytorch_golden() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden json");

    // Sanity: the fixture's key set is the 28 HF GPT-2 keys we expect (no
    // lm_head — the head is tied to wte).
    assert_eq!(golden.safetensors_keys.len(), 28);
    assert!(
        golden
            .safetensors_keys
            .iter()
            .all(|k| k.starts_with("transformer.")),
        "all fixture keys should be under transformer.*"
    );
    assert!(
        !golden
            .safetensors_keys
            .iter()
            .any(|k| k.contains("lm_head")),
        "the tied head must NOT have a safetensors key"
    );

    let device = Default::default();
    let model = load_tiny();

    let seq = golden.input_ids.len();
    let ids =
        Tensor::<B, 1, Int>::from_data(TensorData::new(golden.input_ids.clone(), [seq]), &device)
            .reshape([1, seq]);

    let trace = model.forward_trace(ids);

    // Pinned tolerance. Observed max|Δ| across all stages on this fixture is
    // ~1e-6 (well under 1e-4); we pin at rel/abs 1e-4 with margin. See the
    // per-stage eprintln output for the live numbers.
    let tol = 1e-4f32;

    // ---- Bring parity up incrementally so a failure localizes the bug. ----
    // 1. Embeddings (wte + wpe) — catches embedding load / position bugs.
    assert_eq!(golden.hidden_shape, vec![seq, model.config.n_embd]);
    assert_stage(
        "after_embed",
        &flatten(trace.after_embed),
        &golden.hidden_after_embed,
        tol,
    );
    // 2. After block 0 — one full block: attention, causal mask, gelu, LN.
    assert_stage(
        "after_block0",
        &flatten(trace.after_block0),
        &golden.hidden_after_block0,
        tol,
    );
    // 3. After final LayerNorm. `trace.after_lnf` is the model's pre-head normed
    // features — HF's last `hidden_states[-1]`, which is already `ln_f`-applied
    // (`hidden_states[-1] @ wteᵀ` reproduces the logits exactly), so the golden's
    // `hidden_after_lnf` is that same single-`ln_f` state and we compare directly.
    let after_lnf = flatten(trace.after_lnf);
    assert_stage("after_lnf", &after_lnf, &golden.hidden_after_lnf, tol);
    // Companion invariant: the pre-head features are ln_f-normalized (row mean ≈ 0),
    // confirming `trace.after_lnf` is the post-ln_f state, not the raw block output.
    let n_embd = model.config.n_embd;
    for row in after_lnf.chunks(n_embd) {
        let mean: f32 = row.iter().sum::<f32>() / n_embd as f32;
        assert!(
            mean.abs() < 1e-4,
            "pre-head features should be ln_f-normalized (row mean {mean:e} ≈ 0)"
        );
    }
    // 4. Logits (the tied head).
    assert_eq!(golden.logits_shape, vec![seq, model.config.vocab_size]);
    let logits = flatten(trace.logits);
    assert_stage("logits", &logits, &golden.logits, tol);

    // ---- Tolerance-free backstops. ----
    let vocab = model.config.vocab_size;
    // Last-token top-1 argmax must match the golden's exactly.
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    };
    let last = seq - 1;
    let got_top1 = argmax(&logits[last * vocab..(last + 1) * vocab]);
    let want_top1 = argmax(&golden.logits[last * vocab..(last + 1) * vocab]);
    assert_eq!(
        got_top1, want_top1,
        "last-token top-1 argmax must match the golden"
    );
    // Logits cosine similarity must be essentially 1.
    let cos = cosine(&logits, &golden.logits);
    assert!(
        cos > 0.99999,
        "logits cosine similarity {cos} must exceed 0.99999"
    );
    eprintln!("last-token top1 = {got_top1}; logits cosine = {cos:.8}");
}
