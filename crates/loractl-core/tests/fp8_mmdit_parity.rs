//! Offline forward parity for the scaled-fp8 load path (M15, #82) —
//! `tests/mmdit_parity.rs`'s twin over the quantized fixture.
//!
//! `just fp8-reference` quantized the seed-11 official tiny MMDiT weights to
//! `tests/fixtures/tiny-mmdit/model_fp8.safetensors` (fp8 weights +
//! `.weight_scale` sidecars) and dumped `tests/fixtures/fp8_mmdit_golden.json`
//! by running the OFFICIAL `krea-ai/krea-2` `mmdit.py` over the torch-side
//! dequantization of those exact tensors. So the golden's numerics are the
//! official forward over weights mathematically identical to what
//! [`load_fp8_module`] reconstructs — a staged parity proof of the whole fp8
//! pipeline (LUT, scale broadcast, remap, linear transpose, apply), fully
//! offline, with the same masked-position contract and thresholds as the
//! bf16/f32 parity test.
//!
//! **Masked-position contract:** positions blocked by the mask (the padded
//! tail, the masked text token) carry attention garbage that never leaks
//! into valid outputs and whose exact value is backend-dependent. The
//! reference zeroes them in the dumped stages; this test zeroes its own
//! stages with the same masks before comparing.

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use loractl_core::diffusion_trainer::load_fp8_module;
use loractl_core::mmdit::{Mmdit, MmditConfig};
use serde::Deserialize;
use std::path::Path;

/// Plain (no-autodiff) CPU backend for the parity check.
type B = NdArray;

const GOLDEN: &str = include_str!("fixtures/fp8_mmdit_golden.json");
const SAFETENSORS: &str = "tests/fixtures/tiny-mmdit/model_fp8.safetensors";

#[derive(Deserialize)]
struct GoldenConfig {
    features: usize,
    tdim: usize,
    txtdim: usize,
    heads: usize,
    kvheads: usize,
    multiplier: usize,
    layers: usize,
    patch: usize,
    channels: usize,
    txtheads: usize,
    txtkvheads: usize,
    txtlayers: usize,
}

#[derive(Deserialize)]
struct Golden {
    config: GoldenConfig,
    txtmask: Vec<i64>,
    t: Vec<f32>,
    img_tokens: Vec<f32>,
    img_tokens_shape: Vec<usize>,
    context: Vec<f32>,
    context_shape: Vec<usize>,
    pos: Vec<f32>,
    pos_shape: Vec<usize>,
    safetensors_keys: Vec<String>,
    after_first: Vec<f32>,
    tvec: Vec<f32>,
    after_txtfusion: Vec<f32>,
    after_txtmlp: Vec<f32>,
    after_block0: Vec<f32>,
    after_block0_shape: Vec<usize>,
    output: Vec<f32>,
    output_shape: Vec<usize>,
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

fn assert_stage(name: &str, got: &[f32], want: &[f32], tol: f32) {
    let diff = max_abs_diff(got, want);
    assert!(diff <= tol, "{name}: max|Δ| = {diff:e} exceeds tol {tol:e}",);
    eprintln!("{name}: max|Δ| = {diff:e} (tol {tol:e})");
}

fn flatten<Bk: burn::tensor::backend::Backend, const D: usize>(t: Tensor<Bk, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().into_vec::<f32>().unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

/// Zero a flat `[1, l, d]` stage at positions whose mask entry is 0.
fn zero_masked(stage: &mut [f32], mask: &[f32], d: usize) {
    for (i, &m) in mask.iter().enumerate() {
        if m == 0.0 {
            stage[i * d..(i + 1) * d].fill(0.0);
        }
    }
}

/// The golden's inputs as tensors: (img_tokens, context, t, pos, mask).
#[allow(clippy::type_complexity)]
fn golden_inputs<Bk: burn::tensor::backend::Backend>(
    golden: &Golden,
    device: &Bk::Device,
) -> (
    Tensor<Bk, 3>,
    Tensor<Bk, 4>,
    Tensor<Bk, 1>,
    Tensor<Bk, 3>,
    Tensor<Bk, 2>,
) {
    let s = &golden.img_tokens_shape;
    let img = Tensor::<Bk, 1>::from_data(
        TensorData::new(golden.img_tokens.clone(), [golden.img_tokens.len()]),
        device,
    )
    .reshape([s[0], s[1], s[2]]);
    let c = &golden.context_shape;
    let context = Tensor::<Bk, 1>::from_data(
        TensorData::new(golden.context.clone(), [golden.context.len()]),
        device,
    )
    .reshape([c[0], c[1], c[2], c[3]]);
    let t = Tensor::<Bk, 1>::from_data(TensorData::new(golden.t.clone(), [golden.t.len()]), device);
    let p = &golden.pos_shape;
    let pos = Tensor::<Bk, 1>::from_data(
        TensorData::new(golden.pos.clone(), [golden.pos.len()]),
        device,
    )
    .reshape([p[0], p[1], p[2]]);
    // Combined 0/1 key mask: the golden's text mask + all-ones image tokens.
    let mut mask: Vec<f32> = golden.txtmask.iter().map(|&m| m as f32).collect();
    mask.extend(std::iter::repeat_n(1.0f32, s[1]));
    let l = mask.len();
    let mask = Tensor::<Bk, 1>::from_data(TensorData::new(mask, [l]), device).reshape([1, l]);
    (img, context, t, pos, mask)
}

#[test]
fn fp8_tiny_mmdit_forward_matches_official_golden() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden json");
    let config = MmditConfig::tiny();

    // Config-drift guard: the fixture was generated at exactly the Rust tiny
    // config.
    let g = &golden.config;
    assert_eq!(
        (
            g.features,
            g.tdim,
            g.txtdim,
            g.heads,
            g.kvheads,
            g.multiplier
        ),
        (
            config.features,
            config.tdim,
            config.txtdim,
            config.heads,
            config.kvheads,
            config.multiplier
        )
    );
    assert_eq!(
        (
            g.layers,
            g.patch,
            g.channels,
            g.txtheads,
            g.txtkvheads,
            g.txtlayers
        ),
        (
            config.layers,
            config.patch,
            config.channels,
            config.txtheads,
            config.txtkvheads,
            config.txtlayers
        )
    );
    // Regeneration guard: the fixture must actually be quantized — a plain
    // re-dump would pass parity vacuously without touching the fp8 path.
    assert!(
        golden
            .safetensors_keys
            .iter()
            .any(|k| k.ends_with(".weight_scale")),
        "model_fp8.safetensors must carry weight_scale sidecars"
    );

    let device = Default::default();
    // `load_fp8_module` hard-errors on `errors`/`missing`/`unused`, so a
    // clean return is the applied-cleanly assertion of the bf16/f32 twin.
    let model = load_fp8_module(
        Mmdit::<B>::init(MmditConfig::tiny(), &device),
        Path::new(SAFETENSORS),
        &Mmdit::<B>::key_remap(),
        "tiny fp8 MMDiT",
    )
    .expect("scaled-fp8 tiny MMDiT loads");
    let (img, context, t, pos, mask) = golden_inputs::<B>(&golden, &device);

    let trace = model.forward_trace(img, context, t, pos, mask);

    let tol = 1e-4f32;
    let txtmask: Vec<f32> = golden.txtmask.iter().map(|&m| m as f32).collect();

    assert_stage(
        "after_first",
        &flatten(trace.after_first),
        &golden.after_first,
        tol,
    );
    assert_stage("tvec", &flatten(trace.tvec), &golden.tvec, tol);

    // Text stages: zero the masked text token on the Rust side (the golden
    // is dumped pre-zeroed — the masked-position contract).
    let mut after_txtfusion = flatten(trace.after_txtfusion);
    zero_masked(&mut after_txtfusion, &txtmask, golden.config.txtdim);
    assert_stage(
        "after_txtfusion",
        &after_txtfusion,
        &golden.after_txtfusion,
        tol,
    );
    let mut after_txtmlp = flatten(trace.after_txtmlp);
    zero_masked(&mut after_txtmlp, &txtmask, golden.config.features);
    assert_stage("after_txtmlp", &after_txtmlp, &golden.after_txtmlp, tol);

    // Trunk stage: zero everything the padded combined mask blocks.
    assert_eq!(
        trace.after_block0.dims().to_vec(),
        golden.after_block0_shape,
        "padded combined length must match (pad-to-256)"
    );
    let padded_len = golden.after_block0_shape[1];
    let mut padmask = txtmask.clone();
    padmask.extend(std::iter::repeat_n(1.0f32, golden.img_tokens_shape[1]));
    padmask.extend(std::iter::repeat_n(0.0f32, padded_len - padmask.len()));
    let mut after_block0 = flatten(trace.after_block0);
    zero_masked(&mut after_block0, &padmask, golden.config.features);
    assert_stage("after_block0", &after_block0, &golden.after_block0, tol);

    // The final velocity prediction over the image tokens — all valid.
    assert_eq!(trace.output.dims().to_vec(), golden.output_shape);
    let output = flatten(trace.output);
    assert_stage("output", &output, &golden.output, tol);

    // Tolerance-free backstops.
    assert_eq!(
        argmax(&output),
        argmax(&golden.output),
        "output argmax must match the golden"
    );
    let cos = cosine(&output, &golden.output);
    assert!(cos > 0.99999, "output cosine {cos} must exceed 0.99999");
    eprintln!("output cosine = {cos:.8}");
}
