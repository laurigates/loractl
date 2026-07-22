//! Relative-accuracy gate for the reduced-precision paths (#112), the
//! second tier alongside the fixed-truth torch goldens in `tests/quant.rs`
//! and `tests/fp8.rs`.
//!
//! Those goldens pin that quantization is done *correctly and reproducibly* —
//! the dequantized weights and the `x · dequant(wq)ᵀ` forward match torch to
//! `1e-6`/`1e-5`, because torch computes the *same quantized* operation. What
//! they deliberately do NOT measure is how far quantization moves the output
//! from the true full-precision answer — the QLoRA-quality question (int4's
//! coarser base is exactly the #25 A/B concern). That is an accuracy claim, and
//! an absolute `1e-5` is the wrong shape for it: int4 carries ~7% worst-case
//! weight error by construction.
//!
//! So this file measures each reduced-precision forward against a
//! **full-precision f32 oracle** run over the *same activations* (pure Rust on
//! ndarray — no torch), and gates the deviation with the calibrated
//! [`RelGate`] band + hard ceiling (the CAEF ADR-0006 protocol; see
//! `docs/adrs/0006-reduced-precision-accuracy-gate.md`). The gate has teeth
//! beyond a magic constant: the deviation must be non-zero (quantization
//! actually moved the output) and int4's must strictly exceed int8's (a coarser
//! base is measurably worse — a relationship no mis-set threshold can fake).

use burn::backend::NdArray;
use burn::tensor::quantization::QuantValue;
use burn::tensor::{Tensor, TensorData};
use loractl_core::accuracy::{RelGate, rel_deviation};
use loractl_core::fp8::e4m3fn_lut;
use loractl_core::quant::{QuantBackend, quantize_linear_weight};

type Cpu = NdArray;

const INT8_GOLDEN: &str = "tests/fixtures/quant_int8_golden.json";
const INT4_GOLDEN: &str = "tests/fixtures/quant_int4_golden.json";

// Calibrated bands. `d_bar` ≈ the deviation measured once from the current
// implementation over the pinned inputs (the `println!`s below re-emit it under
// `--nocapture` for recalibration); `2·d_bar` is the regression catch, `ceil` a
// backstop above it for gross corruption. The forward-output deviation is far
// below the worst-case per-element weight error (int4 ~7%) because the matmul
// averages independent per-block rounding errors — but on these small fixtures
// the averaging is limited, so the numbers sit above what a 12.8B site sees.
//
// int8 (Q8S): observed 5.79e-3 → band ≈ 1.16e-2, ceil ~4× the observed.
const INT8_GATE: RelGate = RelGate {
    d_bar: 5.8e-3,
    floor: 1e-3,
    ceil: 2.5e-2,
};
// int4 (Q4S): observed 9.39e-2 → band ≈ 1.88e-1, ceil above it.
const INT4_GATE: RelGate = RelGate {
    d_bar: 9.4e-2,
    floor: 1e-3,
    ceil: 2.0e-1,
};
// fp8 (e4m3fn, scale 1): observed 1.52e-2 → band ≈ 3.1e-2, ceil above it.
const FP8_GATE: RelGate = RelGate {
    d_bar: 1.55e-2,
    floor: 1e-3,
    ceil: 4.0e-2,
};

/// The `w` (original f32 weight) and `x` (activations) matrices from a quant
/// golden — the pinned inputs the fixed-truth tier already uses.
struct Inputs {
    w: (Vec<f32>, [usize; 2]),
    x: (Vec<f32>, [usize; 2]),
}

fn inputs(path: &str) -> Inputs {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("golden {path} present — regenerate with the `just` recipe"));
    let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let matrix = |key: &str| -> (Vec<f32>, [usize; 2]) {
        let rows = json[key].as_array().unwrap();
        let cols = rows[0].as_array().unwrap().len();
        let flat: Vec<f32> = rows
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter())
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        (flat, [rows.len(), cols])
    };
    Inputs {
        w: matrix("w"),
        x: matrix("x"),
    }
}

fn tensor<B: burn::tensor::backend::Backend>(
    (flat, shape): &(Vec<f32>, [usize; 2]),
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(flat.clone(), *shape), device)
}

fn to_vec(t: Tensor<Cpu, 2>) -> Vec<f32> {
    t.into_data().into_vec::<f32>().unwrap()
}

/// The relative deviation of the quantized forward `x · dequant(quantize(w))ᵀ`
/// from the full-precision oracle `x · wᵀ` (same activations, unquantized
/// weight). This isolates the *weight-quantization* error propagated to the
/// output — nothing else differs between the two forwards.
fn quant_deviation(inp: &Inputs, value: QuantValue) -> f32 {
    let device = Default::default();
    let w = tensor::<Cpu>(&inp.w, &device);
    let x = tensor::<Cpu>(&inp.x, &device);

    let oracle = to_vec(x.clone().matmul(w.clone().transpose()));
    let wq = quantize_linear_weight(w, value);
    let ours = to_vec(Cpu::quant_matmul_t(x, &wq));

    rel_deviation(&ours, &oracle)
}

#[test]
fn int8_forward_stays_within_the_calibrated_band() {
    let d = quant_deviation(&inputs(INT8_GOLDEN), QuantValue::Q8S);
    println!("int8 (Q8S) forward deviation d_ours = {d:e}");
    // Non-vacuity: quantization must actually perturb the output.
    assert!(
        d > 0.0,
        "int8 deviation is exactly zero — quantization was a no-op"
    );
    INT8_GATE.apply(d).expect_pass("int8 (Q8S) forward");
}

#[test]
fn int4_forward_stays_within_the_calibrated_band() {
    let d = quant_deviation(&inputs(INT4_GOLDEN), QuantValue::Q4S);
    println!("int4 (Q4S) forward deviation d_ours = {d:e}");
    assert!(
        d > 0.0,
        "int4 deviation is exactly zero — quantization was a no-op"
    );
    INT4_GATE.apply(d).expect_pass("int4 (Q4S) forward");
}

/// Teeth the calibrated constants alone cannot provide: int4's coarser base
/// (3-bit magnitude vs 7-bit) MUST move the output measurably more than int8's.
/// A regression that silently widened int8's error or narrowed int4's — or a
/// scheme mixup — breaks this ordering even if both still sat under their
/// individual bands.
#[test]
fn int4_is_measurably_coarser_than_int8() {
    // Same weight/activations for both, so the only variable is the scheme.
    let inp = inputs(INT8_GOLDEN);
    let d8 = quant_deviation(&inp, QuantValue::Q8S);
    let d4 = quant_deviation(&inp, QuantValue::Q4S);
    assert!(
        d4 > d8 * 4.0,
        "int4 must be markedly coarser than int8: int4 {d4:e} vs int8 {d8:e}"
    );
}

/// The nearest e4m3fn byte for a bounded f32 value, at scale 1 — the in-test
/// f32→fp8 encoder (loractl's `fp8.rs` only *loads* fp8; it has no encoder).
/// Skips the two NaN bytes.
fn nearest_fp8_byte(lut: &[f32; 256], v: f32) -> u8 {
    (0u16..=255)
        .filter(|&b| lut[b as usize].is_finite())
        .min_by(|&a, &b| {
            (lut[a as usize] - v)
                .abs()
                .partial_cmp(&(lut[b as usize] - v).abs())
                .unwrap()
        })
        .unwrap() as u8
}

/// fp8 representation error propagated through the matmul: round an f32 weight
/// to e4m3fn (loractl dequantizes exactly as `LUT[byte] · scale`), then compare
/// `x · dequant(w_fp8)ᵀ` against the f32 oracle `x · w_f32ᵀ`.
#[test]
fn fp8_forward_stays_within_the_calibrated_band() {
    let device = Default::default();
    let lut = e4m3fn_lut();
    let [d_out, d_in, n] = [24usize, 32usize, 20usize];

    // Deterministic bounded weights/activations — a strided sine ramp, well
    // inside e4m3fn's ±448 range and away from the exactly-representable grid.
    let w_f32: Vec<f32> = (0..d_out * d_in)
        .map(|i| (i as f32 * 0.137).sin() * 2.0 + (i as f32 * 0.041).cos() * 0.5)
        .collect();
    let x_f32: Vec<f32> = (0..n * d_in)
        .map(|i| (i as f32 * 0.211).sin() * 0.5)
        .collect();

    // Round each weight to its nearest fp8 value (the loractl dequant target).
    let w_fp8: Vec<f32> = w_f32
        .iter()
        .map(|&v| lut[nearest_fp8_byte(&lut, v) as usize])
        .collect();

    let x = Tensor::<Cpu, 2>::from_data(TensorData::new(x_f32, [n, d_in]), &device);
    let w_ref = Tensor::<Cpu, 2>::from_data(TensorData::new(w_f32, [d_out, d_in]), &device);
    let w_q = Tensor::<Cpu, 2>::from_data(TensorData::new(w_fp8, [d_out, d_in]), &device);

    let oracle = to_vec(x.clone().matmul(w_ref.transpose()));
    let ours = to_vec(x.matmul(w_q.transpose()));

    let d = rel_deviation(&ours, &oracle);
    println!("fp8 (e4m3fn) forward deviation d_ours = {d:e}");
    assert!(
        d > 0.0,
        "fp8 deviation is exactly zero — rounding was a no-op"
    );
    FP8_GATE.apply(d).expect_pass("fp8 (e4m3fn) forward");
}
