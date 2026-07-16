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

/// The scheme every frozen-base weight is quantized with: symmetric int8
/// (`Q8S`, zero-point-free), one f32 scale per `[1, QUANT_BLOCK]` block.
/// Built on the backend's own `default_scheme` so the *storage* layout stays
/// backend-appropriate (packed u32 on cubecl/cuda, native i8 on ndarray).
pub fn int8_scheme<B: Backend>() -> QuantScheme {
    <B::QuantizedTensorPrimitive as QTensorPrimitive>::default_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::block([1, QUANT_BLOCK as u8]))
        .with_param(QuantParam::F32)
        .with_mode(QuantMode::Symmetric)
}

/// Quantizes one `[d_out, d_in]` linear weight with [`int8_scheme`]
/// (min-max calibration — exact for symmetric int8). Panics when `d_in`
/// does not divide into blocks: that is a module-construction programmer
/// error, and mis-aligned blocks would silently mis-scale every row.
pub fn quantize_linear_weight<B: Backend>(weight: Tensor<B, 2>) -> Tensor<B, 2> {
    let [d_out, d_in] = weight.dims();
    assert!(
        d_in % QUANT_BLOCK == 0,
        "linear weight [{d_out}, {d_in}]: the input dim must be a multiple of the quant block \
         ({QUANT_BLOCK})"
    );
    weight.quantize_dynamic(&int8_scheme::<B>())
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
