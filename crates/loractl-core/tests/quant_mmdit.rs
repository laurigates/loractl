//! Int8 frozen-base quantization of the Krea 2 MMDiT (PR-B2, #96).
//!
//! Four offline (ndarray) proofs that `Mmdit::into_quantized` swaps each
//! frozen-base `Linear` for its int8 [`BaseLinear::Quant`] twin **without**
//! disturbing behavior or the M6 LoRA attach seam:
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
/// on the same input. A correct transpose gives ~int8 precision (~0.3%); a
/// wrong one gives O(1) (or a shape mismatch on non-square projections).
#[test]
fn single_linear_quant_matches_plain() {
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
    let wq = quantize_linear_weight(lin.weight.val().transpose());
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
    eprintln!("single Linear quant vs plain: max|Δ|/peak = {rel:e}");
    assert!(
        rel <= 2.0e-2,
        "single-Linear quant drifts {rel:e} — orientation/precision bug"
    );
}

#[test]
fn quantized_forward_matches_plain_within_int8_tolerance() {
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
    let quant = plain.clone().into_quantized(&device);
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
    eprintln!("quant vs plain: max|Δ|/peak = {rel:e}, cos = {cos:.6}");

    // Quantization actually happened (not a silent Plain passthrough)...
    assert!(
        max_abs > 0.0,
        "quantized forward is bit-identical to plain — did quantization run?"
    );
    // ...and the whole quantized forward tracks the plain one: a wrong weight
    // orientation would give O(1) drift (or a shape-mismatch panic on the
    // non-square projections), not int8 noise. Both bounds are loose relative
    // to the observed ~0.2% / ~1.0 so they fail only on a gross bug.
    assert!(
        rel <= REL_TOL,
        "quantized forward drifts rel = {rel:e} from plain (tol {REL_TOL:e}) — \
         suspect the weight orientation in into_quantized/BaseLinear::forward"
    );
    assert!(
        cos >= 0.999,
        "quantized forward direction diverged (cos = {cos:.6}) — suspect orientation"
    );
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
    let base = Mmdit::<B>::init(cfg.clone(), &device).into_quantized(&device);
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
    let base_inner = Mmdit::<B>::init(cfg.clone(), &device).into_quantized(&device);
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
