//! Int8 frozen-base quantization core (#96 — the #24 follow-up).
//!
//! Weight-only, symmetric, per-block int8 for the frozen base's `Linear`
//! sites: values live as burn-native `QFloat` tensors (packed u32 storage +
//! real quantize/dequantize kernels on cubecl backends; plain `Q8S` on
//! ndarray for the offline/CI path), the LoRA adapters stay f32, and the
//! forward dequantizes transiently per matmul — the QLoRA pattern on burn
//! 0.21's own quantization substrate.
//!
//! ## The autodiff boundary (why the custom op exists)
//!
//! burn 0.21's `Autodiff` implements no quantized op (its `QTensorOps` are
//! `todo!()`), so a `QFloat` tensor can carry through an autodiff graph only
//! as a **constant**. The naive route — dequantize per forward and feed a
//! stock `matmul` — is a memory trap: burn-autodiff eagerly checkpoints a
//! tracked matmul's rhs (`burn-autodiff-0.21.0/src/ops/tensor.rs:612`,
//! `rhs_state = lhs_tracked.then(|| prep.checkpoint(&rhs))`), and a
//! dequantized weight is a graph *leaf* that cannot be recomputed — so every
//! quantized site's f32 weight would be **retained until backward** (~224
//! sites ≈ 49 GB on the real 12.8B model). [`QuantMatmulT`] instead bakes
//! the (already-resident, Arc-backed) quantized primitive into its backward
//! state and dequantizes transiently in BOTH passes: forward
//! `x · dequant(wq)ᵀ`, backward `grad_x = grad_out · dequant(wq)`; peak
//! extra memory is one layer's f32 weight, in either pass. Gradients flow to
//! `x` (and through it to the LoRA adapters) — never to the weight.
//!
//! ## Delete-path
//!
//! burn 0.22 ships native LoRA/QLoRA (tracel-ai/burn#5139), which makes a
//! quantized frozen base a first-class differentiable-through-activations
//! citizen. When the 0.22 migration lands (issue #79), this module and the
//! `BaseLinear::Quant` arm are the surface to re-evaluate and likely delete.

use crate::config::Quant;
use burn::backend::Autodiff;
use burn::backend::NdArray;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind, unary};
use burn::tensor::backend::Backend;
use burn::tensor::quantization::{
    QTensorPrimitive, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantValue,
};
use burn::tensor::{Tensor, TensorPrimitive};

/// Contiguous elements per quantization block, along the **input** dimension
/// of a weight held in file layout `[d_out, d_in]` — the GGML-Q8_0 /
/// bitsandbytes-style grouping. 32 keeps the f32 scale overhead at 1/8 of
/// the int8 payload (~1.6 GB on the 12.8B base); per-output-channel is not
/// expressible (burn's `BlockSize` dims are u8-bounded, d_in reaches 16384).
pub const QUANT_BLOCK: usize = 32;

/// The scheme every frozen-base weight is quantized with: per-block symmetric
/// (zero-point-free), one f32 scale per `[1, QUANT_BLOCK]` block, parametrized
/// by the [`QuantValue`] the caller picks (`Q8S` for int8, `Q4S` for int4 —
/// both symmetric; `QUANT_BLOCK = 32` tiles either since `32 % num_quants` is
/// 0 for both). Built on the backend's own `default_scheme` so the *storage*
/// layout stays backend-appropriate (packed u32 on cubecl/cuda — the generic
/// `PackedU32` packs 8 int4 or 4 int8 per word — native i8 on ndarray).
pub fn quant_scheme<B: Backend>(value: QuantValue) -> QuantScheme {
    <B::QuantizedTensorPrimitive as QTensorPrimitive>::default_scheme()
        .with_value(value)
        .with_level(QuantLevel::block([1, QUANT_BLOCK as u8]))
        .with_param(QuantParam::F32)
        .with_mode(QuantMode::Symmetric)
}

/// Quantizes one `[d_out, d_in]` linear weight with [`quant_scheme`] at the
/// given [`QuantValue`] (min-max calibration — exact for symmetric int8/int4).
/// Panics when `d_in` does not divide into blocks: that is a
/// module-construction programmer error, and mis-aligned blocks would silently
/// mis-scale every row.
pub fn quantize_linear_weight<B: Backend>(weight: Tensor<B, 2>, value: QuantValue) -> Tensor<B, 2> {
    let [d_out, d_in] = weight.dims();
    assert!(
        d_in % QUANT_BLOCK == 0,
        "linear weight [{d_out}, {d_in}]: the input dim must be a multiple of the quant block \
         ({QUANT_BLOCK})"
    );
    weight.quantize_dynamic(&quant_scheme::<B>(value))
}

/// Row-chunk layout for a `[d_out, d_in]` weight under a dequant-transient
/// byte threshold (#128): the number of `d_out` rows in each chunk, in order.
/// Chunks concatenate along `d_out`; the counts always sum to `d_out`.
///
/// A **pure function of `(shape, threshold)`** — both quantize paths
/// ([`Mmdit::into_quantized`](crate::mmdit::Mmdit::into_quantized)'s skeleton
/// and the streaming loader
/// [`load_quant_module`](crate::diffusion_trainer::load_quant_module)) call
/// this, so they produce the same chunk layout by construction.
///
/// Semantics: `chunk_bytes == 0` disables chunking (one chunk); a weight
/// whose full f32 size (`d_out · d_in · 4`) is at or below the threshold
/// stays one chunk (byte-identical behavior to the unchunked code); larger
/// weights split into balanced row chunks, each of whose f32 size
/// (`rows · d_in · 4`) stays at or below the threshold — except that a chunk
/// is never smaller than one row (a single row wider than the threshold
/// cannot be split further; blocks live *within* rows, so a row is the
/// smallest bit-identical unit).
pub fn dequant_chunk_rows(d_out: usize, d_in: usize, chunk_bytes: u64) -> Vec<usize> {
    let full_bytes = (d_out as u64) * (d_in as u64) * 4;
    if chunk_bytes == 0 || full_bytes <= chunk_bytes || d_out <= 1 {
        return vec![d_out];
    }
    let row_bytes = (d_in as u64) * 4;
    let max_rows = (chunk_bytes / row_bytes).max(1) as usize;
    // Balanced ceil-split: n_chunks is the minimum count that keeps every
    // chunk at or under max_rows; the first `rem` chunks take one extra row.
    let n_chunks = d_out.div_ceil(max_rows);
    let base = d_out / n_chunks;
    let rem = d_out % n_chunks;
    (0..n_chunks)
        .map(|i| if i < rem { base + 1 } else { base })
        .collect()
}

/// The chunk-aware sibling of [`quantize_linear_weight`] (#128): quantizes a
/// `[d_out, d_in]` weight as the row chunks [`dequant_chunk_rows`] prescribes,
/// returning one `QFloat` tensor per chunk (in `d_out` order).
///
/// The quant scheme's blocks are `[1, QUANT_BLOCK]` — strictly **within** a
/// row along `d_in` — so quantizing a row chunk `[rows, d_in]` separately is
/// **bit-identical** to quantizing the whole `[d_out, d_in]` tensor and
/// taking those rows (per-block scales see exactly the same 32 values either
/// way; pinned in `tests/quant.rs`). A single-chunk layout takes the exact
/// [`quantize_linear_weight`] path (no `narrow`), so the default threshold is
/// byte-identical to the unchunked code.
pub fn quantize_linear_weight_chunked<B: Backend>(
    weight: Tensor<B, 2>,
    value: QuantValue,
    chunk_bytes: u64,
) -> Vec<Tensor<B, 2>> {
    let [d_out, d_in] = weight.dims();
    let rows = dequant_chunk_rows(d_out, d_in, chunk_bytes);
    if rows.len() == 1 {
        return vec![quantize_linear_weight(weight, value)];
    }
    let mut out = Vec::with_capacity(rows.len());
    let mut start = 0usize;
    for r in rows {
        out.push(quantize_linear_weight(
            weight.clone().narrow(0, start, r),
            value,
        ));
        start += r;
    }
    out
}

/// The [`Quant`] config knob → the burn [`QuantValue`] the frozen base is
/// quantized with. `None` → no quantization; `Int8` → `Q8S`; `Int4` → `Q4S`.
/// Lives here (not in `config.rs`) so the config module stays free of burn
/// imports — the same core-owns-the-vocabulary split as [`Quant`]'s `FromStr`.
pub fn quant_value(q: Quant) -> Option<QuantValue> {
    match q {
        Quant::None => None,
        Quant::Int8 => Some(QuantValue::Q8S),
        Quant::Int4 => Some(QuantValue::Q4S),
    }
}

/// The one quantized compute primitive the trainer needs: `x · dequant(wq)ᵀ`
/// with the weight treated as a constant.
///
/// The default body serves every real (non-autodiff) backend: dequantize to
/// f32 (a real kernel on cubecl backends), transpose (a free view), stock
/// matmul. `Autodiff` overrides it with the custom-op route documented in
/// the module docs — do NOT "simplify" that override back to this body; the
/// module docs explain the ~49 GB retention trap that would reintroduce.
pub trait QuantBackend: Backend {
    /// `out[n, d_out] = x[n, d_in] · dequant(wq[d_out, d_in])ᵀ`; `wq` is a
    /// frozen constant — no gradient ever flows to it.
    fn quant_matmul_t(x: Tensor<Self, 2>, wq: &Tensor<Self, 2>) -> Tensor<Self, 2> {
        x.matmul(wq.clone().dequantize().transpose())
    }
}

impl QuantBackend for NdArray {}

#[cfg(feature = "wgpu")]
impl QuantBackend for burn::backend::Wgpu {}

// `Wgpu<f16>` is a distinct concrete backend from the default `Wgpu` (f32) —
// the M13 half-precision path (`compute.precision: f16`) and the
// `trace_f16_forward` diagnostic run the *non-autodiff* forward directly on it,
// so it needs its own impl (the `Autodiff<Wgpu<f16>>` training path is already
// covered by the blanket `Autodiff` impl below). The default `quant_matmul_t`
// (dequantize + matmul) is dtype-generic over cubecl float elements.
#[cfg(feature = "wgpu")]
impl QuantBackend for burn::backend::Wgpu<burn::tensor::f16> {}

#[cfg(feature = "cuda")]
impl QuantBackend for burn::backend::Cuda {}

// candle has no quantized ops in burn 0.21; the config guard matrix keeps
// this arm unreachable (quant is ndarray/cuda-only), and the default body's
// missing q-ops fail loudly if it is ever reached anyway.
#[cfg(feature = "candle")]
#[allow(deprecated)]
impl QuantBackend for burn::backend::Candle {}

impl<B: Backend, C: CheckpointStrategy> QuantBackend for Autodiff<B, C> {
    fn quant_matmul_t(x: Tensor<Self, 2>, wq: &Tensor<Self, 2>) -> Tensor<Self, 2> {
        /// See the module docs: dequant-in-both-passes, weight-as-constant.
        #[derive(Debug)]
        struct QuantMatmulT;

        impl<B: Backend> Backward<B, 1> for QuantMatmulT {
            /// The quantized weight primitive — Arc-backed handle onto the
            /// already-resident int8 buffer, so retaining it costs nothing.
            type State = B::QuantizedTensorPrimitive;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let wq = ops.state;
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    // grad_x[n, d_in] = grad_out[n, d_out] · dequant(wq)[d_out, d_in]
                    // — the dequantized f32 weight is transient here exactly
                    // as it is in the forward.
                    let w = B::dequantize(wq, burn::tensor::FloatDType::F32);
                    B::float_matmul(grad, w)
                });
            }
        }

        let TensorPrimitive::Float(x) = x.into_primitive() else {
            unreachable!("activations are float tensors")
        };
        let wq = match wq.clone().into_primitive() {
            TensorPrimitive::QFloat(q) => q,
            TensorPrimitive::Float(_) => {
                panic!("quant_matmul_t requires a quantized weight — see quantize_linear_weight")
            }
        };

        // Forward on the inner backend: the f32 weight is dropped as soon as
        // the matmul completes.
        let w = B::dequantize(wq.clone(), burn::tensor::FloatDType::F32);
        let out = B::float_matmul(x.primitive.clone(), B::float_transpose(w));

        let out = match QuantMatmulT
            .prepare::<C>([x.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(wq, out),
            OpsKind::UnTracked(prep) => prep.finish(out),
        };
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }
}
