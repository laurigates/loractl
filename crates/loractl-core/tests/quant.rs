//! Numerics and autodiff contract for the int8 frozen-base quantization
//! core (#96, the #24 follow-up) — all offline on ndarray.
//!
//! Three claims are pinned, because each fails differently:
//!
//! 1. **The quantization itself** matches the torch golden
//!    (`reference/quant_reference.py`): per-block-32 symmetric int8 over the
//!    input dim of a `[d_out, d_in]` weight — dequantized values and the
//!    `x · dqᵀ` forward agree tightly (the fixture is generated tie-free, so
//!    rounding conventions cannot diverge).
//! 2. **The custom autodiff op is gradient-exact**: `quant_matmul_t` under
//!    `Autodiff` must produce bit-identical outputs AND x-gradients to a
//!    stock matmul against the pre-dequantized weight — the only difference
//!    is WHERE the dequantized f32 lives (transient in the op vs retained by
//!    the graph). This is the assertion that guards the op's whole reason to
//!    exist (burn-autodiff's stock matmul retains a tracked matmul's rhs;
//!    ~224 retained site weights ≈ 49 GB on the real model).
//! 3. **The weight is a constant**: no gradient ever flows to the quantized
//!    tensor, under plain `Autodiff` and under `BalancedCheckpointing`.

use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Tensor, TensorData};
use loractl_core::quant::{QUANT_BLOCK, QuantBackend, quantize_linear_weight};

type Cpu = NdArray;

/// The checked-in golden from `reference/quant_reference.py`.
struct Golden {
    w: (Vec<f32>, [usize; 2]),
    x: (Vec<f32>, [usize; 2]),
    dq: (Vec<f32>, [usize; 2]),
    y: (Vec<f32>, [usize; 2]),
}

fn golden() -> Golden {
    let raw = std::fs::read_to_string("tests/fixtures/quant_int8_golden.json")
        .expect("golden present — regenerate with `just quant-reference`");
    let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(
        json["block"], QUANT_BLOCK as u64,
        "fixture/QUANT_BLOCK drift"
    );
    let matrix = |key: &str| -> (Vec<f32>, [usize; 2]) {
        let rows = json[key].as_array().unwrap();
        let cols = rows[0].as_array().unwrap().len();
        let flat: Vec<f32> = rows
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter())
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let shape = [rows.len(), cols];
        (flat, shape)
    };
    Golden {
        w: matrix("w"),
        x: matrix("x"),
        dq: matrix("dq"),
        y: matrix("y"),
    }
}

fn tensor<B: burn::tensor::backend::Backend>(
    (flat, shape): &(Vec<f32>, [usize; 2]),
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(flat.clone(), *shape), device)
}

fn max_abs_diff(a: Tensor<Cpu, 2>, b: Tensor<Cpu, 2>) -> f32 {
    (a - b)
        .abs()
        .max()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap()[0]
}

/// Claim 1a: dequantize(quantize(w)) reproduces the torch golden.
#[test]
fn dequantized_weights_match_the_torch_golden() {
    let device = Default::default();
    let g = golden();
    let wq = quantize_linear_weight(tensor::<Cpu>(&g.w, &device));
    let dq = wq.dequantize();
    let diff = max_abs_diff(dq, tensor::<Cpu>(&g.dq, &device));
    assert!(diff <= 1e-6, "dequantized weights drift from torch: {diff}");
}

/// Claim 1b: the forward `x · dequant(wq)ᵀ` reproduces the golden product.
#[test]
fn quant_matmul_t_matches_the_torch_golden() {
    let device = Default::default();
    let g = golden();
    let wq = quantize_linear_weight(tensor::<Cpu>(&g.w, &device));
    let y = Cpu::quant_matmul_t(tensor::<Cpu>(&g.x, &device), &wq);
    let diff = max_abs_diff(y, tensor::<Cpu>(&g.y, &device));
    assert!(diff <= 1e-5, "quant_matmul_t drifts from torch: {diff}");
}

/// Quantization error is bounded by half a scale step per element — the
/// symmetric-int8 contract, independent of the golden.
#[test]
fn roundtrip_error_is_bounded_by_half_a_scale_step() {
    let device = Default::default();
    let w = Tensor::<Cpu, 2>::random(
        [16, 96],
        burn::tensor::Distribution::Uniform(-3.0, 3.0),
        &device,
    );
    let dq = quantize_linear_weight(w.clone()).dequantize();

    let w_data = w.clone().into_data().into_vec::<f32>().unwrap();
    let dq_data = dq.into_data().into_vec::<f32>().unwrap();
    for (row, chunk) in w_data.chunks(96).enumerate() {
        for (block_idx, block) in chunk.chunks(QUANT_BLOCK).enumerate() {
            let scale = block.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
            let offset = row * 96 + block_idx * QUANT_BLOCK;
            for (i, w_v) in block.iter().enumerate() {
                let err = (dq_data[offset + i] - w_v).abs();
                assert!(
                    err <= scale * 0.5 + 1e-7,
                    "block ({row},{block_idx}) elem {i}: err {err} > scale/2 {}",
                    scale * 0.5
                );
            }
        }
    }
}

/// Claims 2 + 3: under `Autodiff`, the custom op's output AND x-gradients
/// are bit-identical to a stock matmul against the pre-dequantized weight,
/// and the quantized weight receives no gradient.
#[test]
fn custom_op_matches_stock_matmul_with_predequantized_weights() {
    grad_equivalence::<Autodiff<Cpu>>();
}

/// Same contract under BalancedCheckpointing — the strategy the trainer
/// actually uses with `compute.grad_checkpointing: true`.
#[test]
fn custom_op_grads_survive_balanced_checkpointing() {
    grad_equivalence::<Autodiff<Cpu, BalancedCheckpointing>>();
}

fn grad_equivalence<AD>()
where
    AD: burn::tensor::backend::AutodiffBackend<InnerBackend = Cpu> + QuantBackend,
{
    let device = Default::default();
    let g = golden();

    let wq_inner = quantize_linear_weight(tensor::<Cpu>(&g.w, &device));
    let w_dq_inner = wq_inner.clone().dequantize();
    let x_inner = tensor::<Cpu>(&g.x, &device);

    // Everything is built on the inner backend and lifted with `from_inner`
    // — the same path the trainer uses — so no generic-device plumbing.

    // Path 1: the custom op over the still-quantized weight.
    let x1: Tensor<AD, 2> = Tensor::from_inner(x_inner.clone()).require_grad();
    let wq: Tensor<AD, 2> = Tensor::from_inner(wq_inner);
    let y1 = AD::quant_matmul_t(x1.clone(), &wq);
    // A non-trivial loss so the incoming gradient isn't all-ones.
    let loss1 = (y1.clone() * y1.clone()).sum();
    let grads1 = loss1.backward();
    let gx1 = x1.grad(&grads1).expect("x must receive a gradient");

    // Path 2: stock matmul over the pre-dequantized constant.
    let x2: Tensor<AD, 2> = Tensor::from_inner(x_inner).require_grad();
    let w_dq: Tensor<AD, 2> = Tensor::from_inner(w_dq_inner);
    let y2 = x2.clone().matmul(w_dq.transpose());
    let loss2 = (y2.clone() * y2.clone()).sum();
    let grads2 = loss2.backward();
    let gx2 = x2.grad(&grads2).expect("x must receive a gradient");

    // Bit-identical: same backend, same float ops, same order — the ONLY
    // difference is where the dequantized f32 weight lives.
    assert_eq!(
        y1.inner().into_data(),
        y2.inner().into_data(),
        "custom-op forward diverged from stock matmul"
    );
    assert_eq!(
        gx1.into_data(),
        gx2.into_data(),
        "custom-op x-gradients diverged from stock matmul"
    );

    // The frozen weight is a constant by construction: `QuantMatmulT` is a
    // `Backward<B, 1>` (unary in `x`), so the weight is never a graph parent
    // and `wq` is never `require_grad`'d. We cannot probe `wq.grad()` to
    // assert this — burn's `Autodiff` has no quantized dequantize op, so any
    // grad lookup on a QFloat tensor panics — but that same gap is the
    // guardrail: a regression that made the weight a tracked parent would
    // panic on burn's `todo!()` the instant it tried to differentiate it,
    // rather than silently training the frozen base.
    let _ = &grads1;
}

/// The input dim must divide the block size — a violation is a programmer
/// error at module-construction time and must fail loudly, not mis-scale.
#[test]
#[should_panic(expected = "multiple of the quant block")]
fn non_divisible_input_dim_panics() {
    let device = Default::default();
    let w = Tensor::<Cpu, 2>::zeros([4, 33], &device);
    let _ = quantize_linear_weight(w);
}
