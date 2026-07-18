//! Frozen-base quantization of the Krea 2 MMDiT (PR-B2, #96) — int8 (`Q8S`)
//! and int4 (`Q4S`).
//!
//! Offline (ndarray) proofs that `Mmdit::into_quantized` swaps each frozen-base
//! `Linear` for its quantized [`BaseLinear::Quant`] twin **without** disturbing
//! behavior or the M6 LoRA attach seam. The forward-equivalence and
//! single-linear orientation proofs run for both schemes (int4 with looser
//! tolerances — coarser quant); the attach-seam, one-step-train, and coverage
//! proofs are scheme-independent and run on int8:
//!
//! 1. **Forward equivalence** — a quantized forward tracks the plain one
//!    within a loose int8 bound. This is the test that proves the weight
//!    orientation: burn's `Linear` stores its weight `[d_in, d_out]` and
//!    computes `x·W`, so `into_quantized` transposes to file layout
//!    `[d_out, d_in]` before quantizing and the quant matmul
//!    (`x · dequant(wq)ᵀ`) restores `x·W`. A wrong transpose would give
//!    O(1) drift (or a shape mismatch), not int8 noise.
//! 2. **Zero-delta attach integrity** — zero-init LoRA deltas are a
//!    bit-identical no-op on the quantized base (the seam is untouched).
//! 3. **One-step LoRA train** — gradients flow through the custom quant op to
//!    the adapters, which move; the frozen QFloat base is a constant.
//! 4. **`base_linears_mut` coverage** — its paths equal `injectable_sites`.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::Backend;
use burn::tensor::quantization::QuantValue;
use burn::tensor::{Tensor, TensorData};
use loractl_core::adapters::LoraAdapters;
use loractl_core::lora::LoraDelta;
use loractl_core::mmdit::{Mmdit, MmditConfig, krea2_positions};

/// Plain (no-autodiff) CPU backend for the forward/quantize proofs.
type B = NdArray;
/// Autodiff-wrapped backend for the training-step proof.
type AB = Autodiff<NdArray>;

/// Max abs deviation of the quantized forward from the plain one, normalized
/// by the plain output's peak magnitude. A correct int8 forward drifts a few
/// percent; a wrong weight orientation gives O(1) (or panics on a shape
/// mismatch), so this catches a gross bug while tolerating int8 noise.
const REL_TOL: f32 = 5.0e-2;

/// The int4 (`Q4S`) counterpart of [`REL_TOL`]. int4 has 15 levels vs int8's
/// 255, so per-weight error is ~17× coarser and the whole-model forward drifts
/// more — the bound is looser but still catches a gross bug (a wrong weight
/// orientation is O(1), far past this).
const REL_TOL_INT4: f32 = 5.0e-1;

/// Cosine-similarity floor for a correct quantized forward: int8 preserves the
/// output direction to ~1e-3; int4's coarser quant loosens the floor a touch.
const COS_MIN_INT8: f64 = 0.999;
const COS_MIN_INT4: f64 = 0.99;

/// Deterministic bounded input data: a strided `sin` ramp.
fn ramp(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.137 + phase).sin() * 0.5)
        .collect()
}

/// A rank-`D` tensor from a flat `sin` ramp of the right length.
fn seq<Bk: Backend, const D: usize>(
    shape: [usize; D],
    phase: f32,
    device: &Bk::Device,
) -> Tensor<Bk, D> {
    let n: usize = shape.iter().product();
    Tensor::<Bk, 1>::from_data(TensorData::new(ramp(n, phase), [n]), device).reshape(shape)
}

fn flatten<Bk: Backend, const D: usize>(t: Tensor<Bk, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().into_vec::<f32>().unwrap()
}

/// A tiny, dimension-consistent forward input set: b = 1, 3 text tokens, a
/// 2×2 patch grid (4 image tokens), all-valid mask.
#[allow(clippy::type_complexity)]
fn inputs<Bk: Backend>(
    cfg: &MmditConfig,
    device: &Bk::Device,
) -> (
    Tensor<Bk, 3>,
    Tensor<Bk, 4>,
    Tensor<Bk, 1>,
    Tensor<Bk, 3>,
    Tensor<Bk, 2>,
) {
    let b = 1;
    let txtlen = 3;
    let (gh, gw) = (2, 2);
    let img_tokens = gh * gw;
    let img_dim = cfg.channels * cfg.patch * cfg.patch;

    let img = seq::<Bk, 3>([b, img_tokens, img_dim], 0.1, device);
    let context = seq::<Bk, 4>([b, txtlen, cfg.txtlayers, cfg.txtdim], 0.7, device);
    let t = seq::<Bk, 1>([b], 0.3, device);
    let pos = krea2_positions::<Bk>(txtlen, gh, gw, b, device);
    let mask = Tensor::<Bk, 2>::ones([b, txtlen + img_tokens], device);
    (img, context, t, pos, mask)
}

/// A tiny adapter set (rank 4, α 8, zero-init B) over every injectable site.
fn zero_init_adapters<Bk: Backend>(model: &Mmdit<Bk>, device: &Bk::Device) -> LoraAdapters<Bk> {
    let sites = model.injectable_sites();
    LoraAdapters::<Bk> {
        deltas: sites
            .iter()
            .map(|s| LoraDelta::new(s.d_in, s.d_out, 4, 8.0, 0.0, device))
            .collect(),
        targets: sites.iter().map(|s| s.path.clone()).collect(),
    }
}

/// The cleanest orientation proof, free of whole-model amplification: quantize
/// one `Linear` and compare `BaseLinear::Quant::forward` to `Linear::forward`
/// on the same input. A correct transpose gives ~scheme precision (int8 ~0.3%,
/// int4 coarser); a wrong one gives O(1) (or a shape mismatch on non-square
/// projections). `tol` is the per-scheme relative bound.
fn single_linear_quant_matches(value: QuantValue, tol: f32) {
    use burn::module::Param;
    use burn::nn::LinearConfig;
    use loractl_core::mmdit::{BaseLinear, QuantLinear};
    use loractl_core::quant::quantize_linear_weight;

    let device = Default::default();
    // Non-square (64 → 96) so a missing transpose would mismatch shapes.
    let lin = LinearConfig::new(64, 96).init::<B>(&device);
    let x = seq::<B, 3>([2, 5, 64], 0.2, &device);
    let plain = flatten(lin.forward(x.clone()));
    // burn `Linear` weight is `[d_in, d_out]`; the quant path wants file layout
    // `[d_out, d_in]`, so transpose before quantizing (mirrors into_quantized).
    let wq = quantize_linear_weight(lin.weight.val().transpose(), value);
    let q = BaseLinear::Quant(QuantLinear {
        weight: Param::from_tensor(wq),
        bias: lin.bias.clone(),
    });
    let quant = flatten(q.forward(x));
    let peak = plain.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let max_abs = plain
        .iter()
        .zip(&quant)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = max_abs / peak;
    eprintln!("single Linear quant vs plain ({value:?}): max|Δ|/peak = {rel:e}");
    assert!(
        rel <= tol,
        "single-Linear quant drifts {rel:e} ({value:?}) — orientation/precision bug"
    );
}

#[test]
fn single_linear_int8_quant_matches_plain() {
    single_linear_quant_matches(QuantValue::Q8S, 2.0e-2);
}

#[test]
fn single_linear_int4_quant_matches_plain() {
    single_linear_quant_matches(QuantValue::Q4S, 2.0e-1);
}

fn quantized_forward_matches_plain(value: QuantValue, rel_tol: f32, cos_min: f64) {
    let device = Default::default();
    let cfg = MmditConfig::tiny_krea2();
    let plain = Mmdit::<B>::init(cfg.clone(), &device);

    let (img, context, t, pos, mask) = inputs::<B>(&cfg, &device);
    // Materialize plain's (lazily-initialized) weights via the forward BEFORE
    // cloning — burn `Param`s draw their RNG on first `.val()`, so cloning an
    // un-materialized model and letting `into_quantized` materialize the clone
    // would quantize a DIFFERENT random draw (see burn-lazy-param-init.md).
    let out_plain = flatten(plain.forward(
        img.clone(),
        context.clone(),
        t.clone(),
        pos.clone(),
        mask.clone(),
    ));
    // The clone now copies materialized weights; quantize the clone in place.
    let quant = plain.clone().into_quantized(value, &device);
    let out_quant = flatten(quant.forward(img, context, t, pos, mask));

    assert_eq!(out_plain.len(), out_quant.len(), "output length mismatch");
    let peak = out_plain.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(peak > 0.0, "degenerate plain output");
    let max_abs = out_plain
        .iter()
        .zip(&out_quant)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = max_abs / peak;
    let cos = cosine(&out_plain, &out_quant);
    eprintln!("quant vs plain ({value:?}): max|Δ|/peak = {rel:e}, cos = {cos:.6}");

    // Quantization actually happened (not a silent Plain passthrough)...
    assert!(
        max_abs > 0.0,
        "quantized forward is bit-identical to plain — did quantization run?"
    );
    // ...and the whole quantized forward tracks the plain one: a wrong weight
    // orientation would give O(1) drift (or a shape-mismatch panic on the
    // non-square projections), not quant noise. Both bounds are loose relative
    // to the observed drift so they fail only on a gross bug.
    assert!(
        rel <= rel_tol,
        "quantized forward drifts rel = {rel:e} from plain ({value:?}, tol {rel_tol:e}) — \
         suspect the weight orientation in into_quantized/BaseLinear::forward"
    );
    assert!(
        cos >= cos_min,
        "quantized forward direction diverged ({value:?}, cos = {cos:.6}) — suspect orientation"
    );
}

#[test]
fn quantized_forward_matches_plain_within_int8_tolerance() {
    quantized_forward_matches_plain(QuantValue::Q8S, REL_TOL, COS_MIN_INT8);
}

#[test]
fn quantized_forward_matches_plain_within_int4_tolerance() {
    quantized_forward_matches_plain(QuantValue::Q4S, REL_TOL_INT4, COS_MIN_INT4);
}

/// Cosine similarity of two flattened outputs.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

#[test]
fn zero_init_adapters_are_a_noop_on_quantized_base() {
    let device = Default::default();
    let cfg = MmditConfig::tiny_krea2();
    let base = Mmdit::<B>::init(cfg.clone(), &device).into_quantized(QuantValue::Q8S, &device);
    let set = zero_init_adapters(&base, &device);

    let (img, context, t, pos, mask) = inputs::<B>(&cfg, &device);
    let plain = flatten(base.forward(
        img.clone(),
        context.clone(),
        t.clone(),
        pos.clone(),
        mask.clone(),
    ));
    let adapted = flatten(base.forward_with_adapters(img, context, t, pos, mask, &set));

    let max_abs = plain
        .iter()
        .zip(&adapted)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max_abs, 0.0,
        "zero-init deltas must be bit-identical on the quantized base — \
         quantization disturbed the LoRA attach seam (max|Δ| = {max_abs:e})"
    );
}

#[test]
fn one_step_lora_train_on_quantized_base() {
    use burn::module::AutodiffModule;
    use burn::nn::loss::{MseLoss, Reduction};
    use burn::optim::{AdamConfig, GradientsParams, Optimizer};

    let device = Default::default();
    let cfg = MmditConfig::tiny_krea2();
    // burn 0.21's `Autodiff` has no `quantize` op (todo!()), so quantize on
    // the inner backend and lift the module with `from_inner` — the exact
    // path the trainer uses (see tests/quant.rs).
    let base_inner =
        Mmdit::<B>::init(cfg.clone(), &device).into_quantized(QuantValue::Q8S, &device);
    let base = <Mmdit<AB> as AutodiffModule<AB>>::from_inner(base_inner);

    let mut set = zero_init_adapters(&base, &device);
    let (img, context, t, pos, mask) = inputs::<AB>(&cfg, &device);

    let out = base.forward_with_adapters(img, context, t, pos, mask, &set);
    let loss = MseLoss::new().forward(out.clone(), out.zeros_like(), Reduction::Mean);
    let loss_value: f32 = loss.clone().into_scalar();
    assert!(
        loss_value.is_finite() && loss_value > 0.0,
        "loss must be finite and nonzero: {loss_value}"
    );

    let grads = loss.backward();
    // The custom quant op keeps the graph differentiable through `x`, so every
    // adapter's A and B receive finite gradients.
    for (delta, target) in set.deltas.iter().zip(&set.targets) {
        for (name, lin) in [("A", &delta.lora_a), ("B", &delta.lora_b)] {
            let g = lin
                .weight
                .val()
                .grad(&grads)
                .unwrap_or_else(|| panic!("LoRA {name} at {target} received no gradient"));
            let s: f32 = g.abs().sum().into_scalar();
            assert!(
                s.is_finite(),
                "LoRA {name} at {target} gradient not finite: {s}"
            );
        }
    }

    // Step the adapters only (the frozen QFloat base is a constant in the
    // quant op — it can't and mustn't get grads). B moves off its zero init.
    let mut optim = AdamConfig::new().init::<AB, LoraAdapters<AB>>();
    let grad_params = GradientsParams::from_grads(grads, &set);
    set = optim.step(1e-3, set, grad_params);
    for (delta, target) in set.deltas.iter().zip(&set.targets) {
        let b_sum: f32 = delta.lora_b.weight.val().abs().sum().into_scalar();
        assert!(
            b_sum > 0.0,
            "after one step the LoRA B at {target} must have moved off zero"
        );
    }
}

#[test]
fn base_linears_mut_paths_match_injectable_sites() {
    let device = Default::default();
    let mut model = Mmdit::<B>::init(MmditConfig::tiny_krea2(), &device);
    let site_paths: Vec<String> = model
        .injectable_sites()
        .into_iter()
        .map(|s| s.path)
        .collect();
    let base_paths: Vec<String> = model
        .base_linears_mut()
        .into_iter()
        .map(|(path, _)| path)
        .collect();
    assert!(!base_paths.is_empty(), "expected injectable base linears");
    assert_eq!(
        base_paths, site_paths,
        "base_linears_mut paths must equal injectable_sites paths (same set and order)"
    );
}

/// `all_base_linears_mut` (PR-B3's comprehensive accessor) must cover **every**
/// site `into_quantized` touches — a strict superset of the injectable subset.
/// If it missed a `Quant` site the streaming loader would leave that site's
/// placeholder int8 weight unfilled (and the fp8/plain applier would then apply
/// an f32 tensor to a QFloat param, panicking the forward), so this pins the
/// accessor and `into_quantized` to the same set.
#[test]
fn all_base_linears_mut_covers_every_into_quantized_site() {
    use loractl_core::mmdit::BaseLinear;
    let device = Default::default();

    let mut model = Mmdit::<B>::init(MmditConfig::tiny_krea2(), &device);
    let injectable: std::collections::HashSet<String> = model
        .base_linears_mut()
        .into_iter()
        .map(|(p, _)| p)
        .collect();
    let all: Vec<String> = model
        .all_base_linears_mut()
        .into_iter()
        .map(|(p, _)| p)
        .collect();
    let all_set: std::collections::HashSet<String> = all.iter().cloned().collect();

    assert_eq!(
        all.len(),
        all_set.len(),
        "all_base_linears_mut keys must be unique: {all:?}"
    );
    // tiny_krea2: (2 trunk + 2 layerwise + 2 refiner) blocks × (5 attn + 3 mlp)
    // = 48, plus tmlp.fc1/fc2 + tproj.fc + txtmlp.fc1/fc2 = 53.
    assert_eq!(all.len(), 53, "unexpected base-linear count: {all:?}");
    assert!(
        injectable.is_subset(&all_set),
        "injectable sites must be a subset of all base linears"
    );
    for expect in [
        "blocks.0.attn.gate", // quantized but NOT injectable (no LoRA on gates)
        "txtfusion.layerwise_blocks.0.attn.wq",
        "txtfusion.refiner_blocks.1.mlp.down",
        "tmlp.fc1",
        "tproj.fc",
        "txtmlp.fc2",
    ] {
        assert!(
            all_set.contains(expect),
            "all_base_linears_mut must include {expect}"
        );
    }

    // After into_quantized, every comprehensive site is `Quant` except those
    // whose d_in is not block-aligned — on tiny_krea2 that is exactly `tmlp.fc1`
    // (tdim = 16). This is the agreement between the accessor and into_quantized.
    // Block alignment is scheme-independent, so the Plain/Quant split is the
    // same for int8 and int4; int8 stands in for both here.
    let mut quantized = Mmdit::<B>::init(MmditConfig::tiny_krea2(), &device)
        .into_quantized(QuantValue::Q8S, &device);
    let mut plain_sites: Vec<String> = quantized
        .all_base_linears_mut()
        .into_iter()
        .filter(|(_, base)| matches!(base, BaseLinear::Plain(_)))
        .map(|(p, _)| p)
        .collect();
    plain_sites.sort();
    assert_eq!(
        plain_sites,
        vec!["tmlp.fc1".to_string()],
        "only the unaligned tmlp.fc1 (tdim=16) may stay Plain after into_quantized"
    );
}
