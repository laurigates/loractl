//! The Krea 2 MMDiT denoiser (`SingleStreamDiT`) — the model the LoRA adapts
//! (M11, #22).
//!
//! Ported from `krea-ai/krea-2`'s `mmdit.py` (the authoritative source; there
//! is no pip package — the reference generator downloads it at a pinned
//! commit). It is a **single-stream** rectified-flow DiT: the text context and
//! the patchified image latents are concatenated into one token sequence and
//! run through `layers` identical blocks — not FLUX's double-stream layout.
//!
//! ## The modern-arch footguns, pinned against source
//!
//! - **Zero-centered RMSNorm** ([`ZRmsNorm`]): the parameter (`scale`) is
//!   zero-initialized and the effective weight is `scale + 1`, eps `1e-5`
//!   (not the encoder's `1e-6`), always computed in f32.
//! - **Gated-sigmoid attention** ([`MmditAttention`]): a fifth projection
//!   (`gate`) of the block input multiplies the *merged* attention output
//!   elementwise through a sigmoid, before the output projection:
//!   `wo(attn(q,k,v) * σ(gate(x)))`.
//! - **QK-Norm**: [`ZRmsNorm`] over `head_dim`, applied after the head split
//!   and before RoPE. **GQA**: 48 query heads over 12 KV heads (HF
//!   `repeat_kv` grouping order).
//! - **Rotation-matrix RoPE** ([`rope_tables`]/[`apply_rope`]): FLUX-style
//!   *consecutive-pair* rotation (`x₀' = cos·x₀ − sin·x₁`,
//!   `x₁' = sin·x₀ + cos·x₁`) at `theta = 1e3` — neither the half-split HF
//!   convention (M10) nor burn's built-in layout. The head dim splits across
//!   **3 position axes** `[hd − 12·(hd/16), 6·(hd/16), 6·(hd/16)]`
//!   (= `[32, 48, 48]` at `hd = 128`); text tokens sit at position
//!   `(0, 0, 0)`, image tokens at `(0, row, col)` on the patch grid.
//! - **Shared 6-way modulation** ([`DoubleSharedModulation`]): one `tvec =
//!   tproj(tmlp(temb(t)))` for the whole trunk; each block adds its own
//!   learned bias (`mod.lin`, a bare parameter) and chunks into
//!   pre/post scale-shift-gate. The timestep embedding ([`temb`]) is
//!   cos-first with the timestep scaled ×1000, period 1e4, and its
//!   `[b, 1, dim]` shape is what makes every modulation broadcast per-sample.
//! - **Text fusion** ([`TextFusionTransformer`]): 2 blocks attend across the
//!   M10 conditioner's **12-layer axis** per token, a `Linear(12 → 1)`
//!   projector collapses it, then 2 blocks refine along the sequence. No
//!   RoPE, no modulation, no GQA (20/20 heads) in fusion.
//! - **Masking is symmetric** (query ⊗ key outer product), bidirectional —
//!   no causal mask. The combined sequence is **zero-padded to a multiple of
//!   256** (mask false, positions zero) and the output sliced back to the
//!   image tokens. Masked positions carry attention garbage that never leaks
//!   into valid outputs (they are blocked as keys); the parity harness zeroes
//!   them on both sides before comparing.
//!
//! ## Weight loading
//!
//! The module tree mirrors `raw.safetensors` (430 tensors, single file).
//! Trunk/fusion projections are `nn.Linear`s → `PyTorchToBurnAdapter`
//! transpose (the bare modulation parameters live in custom containers the
//! adapter leaves untouched). Two naming gaps are remapped
//! ([`Mmdit::key_remap`]): the reference's `nn.Sequential` indices
//! (`tmlp.0/2`, `tproj.1`, `txtmlp.0/1/3`) and its `mod` field (a Rust
//! keyword, held here as `modulation`).
//!
//! ## LoRA attach (the point of the milestone)
//!
//! [`Mmdit::injectable_sites`] advertises every per-block projection
//! (`attn.{wq,wk,wv,wo}` + `mlp.{gate,up,down}`) to the M6
//! [`build_adapters`](crate::adapters::build_adapters) machinery, and
//! [`Mmdit::forward_with_adapters`] threads the name-keyed delta set through
//! the forward exactly like GPT-2's attach — zero-initialized deltas are a
//! bit-identical no-op.
//!
//! Like the rest of `loractl-core`, this module emits no output and imports
//! no CLI. Parity: `tests/mmdit_parity.rs` vs `reference/mmdit_reference.py`.

use crate::adapters::{LoraAdapters, LoraSite};
use crate::quant::{QUANT_BLOCK, QuantBackend, quantize_linear_weight};
use burn::module::{Module, Param};
use burn::nn::{Gelu, Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor, TensorData};

/// Additive-mask sentinel — finite **in f16 too** (f16 max ≈ 65504, so a
/// larger magnitude saturates to `-inf`, and a fully-masked row's
/// max-subtracted softmax computes `-inf − (-inf) = NaN`, which then spreads
/// to valid positions through `0 × NaN` in the value matmul). At −3e4 the
/// intended semantics hold in every float dtype: masked contributions
/// underflow to exactly 0 after the max shift, and fully-masked rows softmax
/// to uniform garbage that valid positions never read.
const MASK_NEG: f32 = -3.0e4;

/// Static architecture of a `SingleStreamDiT` variant. Field names mirror
/// `SingleMMDiTConfig`.
#[derive(Debug, Clone, PartialEq)]
pub struct MmditConfig {
    /// Trunk width.
    pub features: usize,
    /// Timestep-embedding width (before `tmlp`).
    pub tdim: usize,
    /// Text-conditioning width (the M10 encoder's hidden size).
    pub txtdim: usize,
    /// Trunk query heads.
    pub heads: usize,
    /// Trunk KV heads (GQA).
    pub kvheads: usize,
    /// SwiGLU width multiplier.
    pub multiplier: usize,
    /// Trunk depth.
    pub layers: usize,
    /// Patch size (latent pixels per token side).
    pub patch: usize,
    /// Latent channels (the M9 VAE's `z_dim`).
    pub channels: usize,
    /// Text-fusion heads.
    pub txtheads: usize,
    /// Text-fusion KV heads.
    pub txtkvheads: usize,
    /// Number of conditioner hidden-state layers fed in (the projector's
    /// input width) — NOT the fusion depth, which is fixed at 2 + 2.
    pub txtlayers: usize,
    /// RoPE base.
    pub theta: f64,
}

impl MmditConfig {
    /// The tiny fixture config. Must match `reference/mmdit_reference.py`'s
    /// `TINY` exactly. Deliberately non-degenerate: `features/2`, `kv_out`,
    /// `txtdim`, `head_dim`, `layers` and `txtlayers` are pairwise distinct,
    /// so deriving one width from the wrong config field (or conflating trunk
    /// depth with the fusion projector's layer axis) fails the always-run
    /// parity test rather than only the opt-in real one.
    pub fn tiny() -> Self {
        Self {
            features: 96,
            tdim: 16,
            txtdim: 40,
            heads: 6,
            kvheads: 2,
            multiplier: 4,
            layers: 2,
            patch: 2,
            channels: 4,
            txtheads: 2,
            txtkvheads: 2,
            txtlayers: 3,
            theta: 1e3,
        }
    }

    /// The composed tiny-Krea-2 bundle's denoiser (M14): dimension-matched
    /// to the tiny VAE (`channels = 4` = its `z_dim`) and the tiny Qwen3-VL
    /// (`txtdim = 32` = its hidden size, `txtlayers = 2` = its select-layer
    /// count), so the whole stack composes offline. Must match
    /// `reference/krea2_reference.py`'s `MMDIT` exactly.
    pub fn tiny_krea2() -> Self {
        Self {
            features: 64,
            tdim: 16,
            txtdim: 32,
            heads: 4,
            kvheads: 2,
            multiplier: 4,
            layers: 2,
            patch: 2,
            channels: 4,
            txtheads: 2,
            txtkvheads: 2,
            txtlayers: 2,
            theta: 1e3,
        }
    }

    /// The real Krea 2 denoiser (`single_mmdit_large_wide` in the reference's
    /// `inference.py`), ~12B parameters at full depth.
    pub fn krea2() -> Self {
        Self {
            features: 6144,
            tdim: 256,
            txtdim: 2560,
            heads: 48,
            kvheads: 12,
            multiplier: 4,
            layers: 28,
            patch: 2,
            channels: 16,
            txtheads: 20,
            txtkvheads: 20,
            txtlayers: 12,
            theta: 1e3,
        }
    }

    /// [`krea2`](Self::krea2) truncated to `layers` blocks — the real-weights
    /// staged-parity configuration (the full 28-block model in f32 exceeds a
    /// 48 GiB host; full-depth runs arrive with M13's quantization).
    pub fn krea2_truncated(layers: usize) -> Self {
        Self {
            layers,
            ..Self::krea2()
        }
    }

    /// Trunk head width.
    pub fn head_dim(&self) -> usize {
        self.features / self.heads
    }

    /// The 3-axis RoPE split of a head dim (`mmdit.py`'s `axes`).
    pub fn rope_axes(head_dim: usize) -> [usize; 3] {
        let unit = head_dim / 16;
        [head_dim - 12 * unit, 6 * unit, 6 * unit]
    }

    /// SwiGLU inner width: `int(2·features/3) · multiplier`, rounded up to a
    /// multiple of 128 (the reference's `multiple`).
    pub fn swiglu_dim(features: usize, multiplier: usize) -> usize {
        let raw = (2 * features / 3) * multiplier;
        raw.div_ceil(128) * 128
    }
}

/// Zero-centered RMSNorm: `x / sqrt(mean(x²) + 1e-5) · (scale + 1)`, with a
/// zero-initialized `scale` parameter (matching the checkpoint's `…scale`
/// keys).
#[derive(Module, Debug)]
pub struct ZRmsNorm<B: Backend> {
    /// Zero-centered per-channel gain over the last dimension.
    pub scale: Param<Tensor<B, 1>>,
}

impl<B: Backend> ZRmsNorm<B> {
    fn init(dim: usize, device: &B::Device) -> Self {
        Self {
            scale: Param::from_tensor(Tensor::zeros([dim], device)),
        }
    }

    /// Normalize the last dimension of a rank-`D` tensor.
    ///
    /// On an f16 backend this uses **range-safe pre-scaled algebra** rather
    /// than the literal formula: `x²` overflows f16 for any |x| > ~256 and
    /// Qwen-family hidden states carry outlier channels in the hundreds
    /// (observed ~600 post-projection on real Krea-2-Raw), so the input is
    /// scaled down by a constant first — `x/√(mean(x²)+ε) ≡
    /// (x/c)/√(mean((x/c)²)+ε/c²)` — keeping every intermediate
    /// representable. Deliberately NOT computed via f32 casts: burn 0.21's
    /// wgpu **f32** kernels are the numerically broken ones on this
    /// platform (NaN/corruption; see `examples/grad_compare.rs`), while the
    /// f16 kernels verify against CPU ground truth. The f32/f64 path is the
    /// literal formula, byte-identical to the parity goldens.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let gain = (self.scale.val() + 1.0).unsqueeze::<D>();
        if x.dtype() == DType::F16 {
            // Constant-inner-scale variance — exact algebra
            // `mean(x²) = mean((x/16)²)·256`: the literal x² overflows f16
            // for |x| > ~256 and Qwen-family activations carry outlier
            // channels to ~600 (observed on real Krea-2-Raw), while
            // (600/16)² ≈ 1400 is comfortably representable. A CONSTANT
            // scale keeps the backward free of data-dependent amplifiers
            // (a row-max prescale was tried and its max/clamp backward
            // NaN'd under loss scaling). ε is raised to the smallest
            // f16 NORMAL value: the true 1e-5 is subnormal, and GPU
            // flush-to-zero would turn an all-zero row's `0/√(0+ε)` into
            // 0/0 = NaN. Deliberately not an f32-cast island: burn 0.21's
            // wgpu f32 kernels are the broken ones on this platform (see
            // `examples/grad_compare.rs`).
            const C: f32 = 16.0;
            let variance = (x.clone() / C).powi_scalar(2).mean_dim(D - 1) * (C * C);
            x / (variance + 6.1e-5).sqrt() * gain
        } else {
            let variance = x.clone().powi_scalar(2).mean_dim(D - 1);
            x / (variance + 1e-5).sqrt() * gain
        }
    }
}

/// The q/k pre-RoPE norms (`QKNorm`), each over `head_dim`.
#[derive(Module, Debug)]
pub struct QkNorm<B: Backend> {
    /// Query norm.
    pub qnorm: ZRmsNorm<B>,
    /// Key norm.
    pub knorm: ZRmsNorm<B>,
}

/// Per-token RoPE tables: `(cos, sin)`, each `[b, l, head_dim / 2]`, built
/// from 3-axis integer positions with per-axis frequency bands (computed in
/// f64, like the reference).
pub struct RopeTables<B: Backend> {
    /// Cosine table.
    pub cos: Tensor<B, 3>,
    /// Sine table.
    pub sin: Tensor<B, 3>,
}

/// Build [`RopeTables`] from positions `[b, l, 3]` for a head dim split
/// across [`MmditConfig::rope_axes`].
pub fn rope_tables<B: Backend>(
    pos: Tensor<B, 3>,
    head_dim: usize,
    theta: f64,
    device: &B::Device,
) -> RopeTables<B> {
    let [b, l, three] = pos.dims();
    assert_eq!(three, 3, "positions carry 3 axes");
    let axes = MmditConfig::rope_axes(head_dim);
    let pos: Vec<f32> = pos.into_data().convert::<f32>().into_vec::<f32>().unwrap();

    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(b * l * half);
    let mut sin = Vec::with_capacity(b * l * half);
    for token in 0..b * l {
        for (axis, &dim) in axes.iter().enumerate() {
            let p = pos[token * 3 + axis] as f64;
            for j in 0..dim / 2 {
                let omega = theta.powf(-2.0 * j as f64 / dim as f64);
                let angle = p * omega;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    }
    RopeTables {
        cos: Tensor::from_data(TensorData::new(cos, [b, l, half]), device),
        sin: Tensor::from_data(TensorData::new(sin, [b, l, half]), device),
    }
}

/// Apply consecutive-pair rotation RoPE to `[b, heads, l, head_dim]`:
/// for each pair `(x₀, x₁)`, `x₀' = cos·x₀ − sin·x₁`, `x₁' = sin·x₀ + cos·x₁`.
fn apply_rope<B: Backend>(x: Tensor<B, 4>, tables: &RopeTables<B>) -> Tensor<B, 4> {
    let [b, h, l, hd] = x.dims();
    let half = hd / 2;
    let pairs = x.reshape([b, h, l, half, 2]);
    let x0 = pairs.clone().narrow(4, 0, 1);
    let x1 = pairs.narrow(4, 1, 1);
    // [b, l, half] -> [b, 1, l, half, 1], broadcast over heads and the pair.
    let cos = tables.cos.clone().reshape([b, 1, l, half, 1]);
    let sin = tables.sin.clone().reshape([b, 1, l, half, 1]);
    let y0 = cos.clone() * x0.clone() - sin.clone() * x1.clone();
    let y1 = sin * x0 + cos * x1;
    Tensor::cat(vec![y0, y1], 4).reshape([b, h, l, hd])
}

/// A frozen int8-quantized replacement for a base [`Linear`] at a trunk site.
///
/// The weight is a burn-native `QFloat` tensor in **file layout `[d_out, d_in]`**
/// (never transposed) — the layout
/// [`quant_matmul_t`](crate::quant::QuantBackend::quant_matmul_t) consumes.
/// burn's `Linear` stores its weight `[d_in, d_out]` and computes `x·W`;
/// [`Mmdit::into_quantized`] transposes to `[d_out, d_in]` before quantizing,
/// and [`BaseLinear::forward`] restores `x·W` as `x · dequant(wq)ᵀ`, so a
/// quantized site equals its plain twin up to int8 error.
#[derive(Module, Debug)]
pub struct QuantLinear<B: Backend> {
    /// Frozen int8 weight in FILE layout `[d_out, d_in]` (a `QFloat`
    /// primitive; never transposed, never receives a gradient — the quant op
    /// treats it as a constant).
    pub weight: Param<Tensor<B, 2>>,
    /// Optional bias `[d_out]`, kept in full precision (never quantized).
    pub bias: Option<Param<Tensor<B, 1>>>,
}

/// The frozen base linear at an injectable site: either the plain loaded
/// [`Linear`] (the default — byte-identical to pre-quant behavior) or its int8
/// [`QuantLinear`] twin. The M6 LoRA adapter attaches on top of *this* output
/// at [`site`], unaffected by which arm is active — quantizing the base does
/// not disturb the attach seam.
#[derive(Module, Debug)]
pub enum BaseLinear<B: Backend> {
    /// The plain, full-precision loaded linear.
    Plain(Linear<B>),
    /// An int8-quantized frozen linear (the memory knob for the ~12B base).
    Quant(QuantLinear<B>),
}

impl<B: Backend> BaseLinear<B> {
    /// The inner plain [`Linear`], for tests/diagnostics that read the base
    /// weight directly. Panics on the `Quant` arm — a low-traffic accessor,
    /// not a hot path.
    pub fn as_plain(&self) -> &Linear<B> {
        match self {
            BaseLinear::Plain(lin) => lin,
            BaseLinear::Quant(_) => {
                panic!("BaseLinear::as_plain called on a quantized (Quant) site")
            }
        }
    }
}

impl<B: QuantBackend> BaseLinear<B> {
    /// Forward through whichever arm is active — a numerical drop-in for
    /// [`Linear::forward`] up to int8 error. `Plain` delegates directly;
    /// `Quant` flattens the leading dims to a `[n, d_in]` matrix (mirroring
    /// burn `Linear`'s own batch-flatten), runs the weight-as-constant
    /// [`quant_matmul_t`](crate::quant::QuantBackend::quant_matmul_t)
    /// (`x · dequant(wq)ᵀ`), adds the broadcast bias, and reshapes back with
    /// `d_out` as the last dimension.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            BaseLinear::Plain(lin) => lin.forward(x),
            BaseLinear::Quant(q) => {
                let mut out_dims = x.dims();
                let d_in = out_dims[D - 1];
                let n: usize = out_dims[..D - 1].iter().product();
                let wq = q.weight.val();
                let d_out = wq.dims()[0]; // file layout [d_out, d_in]
                let x2 = x.reshape([n, d_in]);
                let mut y = B::quant_matmul_t(x2, &wq); // [n, d_out]
                if let Some(bias) = &q.bias {
                    y = y + bias.val().reshape([1, d_out]);
                }
                out_dims[D - 1] = d_out;
                y.reshape(out_dims)
            }
        }
    }
}

/// Run a base linear at an injectable site, adding any adapter registered for
/// `path` (the M6 attach seam; `None` is the plain loaded forward). `base`
/// carries whichever precision the frozen weight has — the seam is identical
/// for [`BaseLinear::Plain`] and [`BaseLinear::Quant`].
fn site<B: QuantBackend, const D: usize>(
    adapters: Option<&LoraAdapters<B>>,
    path: &str,
    base: &BaseLinear<B>,
    x: Tensor<B, D>,
) -> Tensor<B, D> {
    match adapters {
        Some(a) => a.apply(path, x.clone(), base.forward(x)),
        None => base.forward(x),
    }
}

/// The gated-sigmoid GQA attention (`mmdit.py`'s `Attention`).
#[derive(Module, Debug)]
pub struct MmditAttention<B: Backend> {
    /// Query projection.
    pub wq: BaseLinear<B>,
    /// Key projection (KV heads wide).
    pub wk: BaseLinear<B>,
    /// Value projection (KV heads wide).
    pub wv: BaseLinear<B>,
    /// The sigmoid gate projection (`dim → dim`), from the block input.
    pub gate: BaseLinear<B>,
    /// Output projection.
    pub wo: BaseLinear<B>,
    /// Pre-RoPE q/k norms.
    pub qknorm: QkNorm<B>,
    /// Query-head count.
    #[module(skip)]
    pub heads: usize,
    /// KV-head count.
    #[module(skip)]
    pub kvheads: usize,
}

impl<B: Backend> MmditAttention<B> {
    fn init(dim: usize, heads: usize, kvheads: usize, device: &B::Device) -> Self {
        let head_dim = dim / heads;
        let lin = |d_in: usize, d_out: usize| {
            BaseLinear::Plain(LinearConfig::new(d_in, d_out).with_bias(false).init(device))
        };
        Self {
            wq: lin(dim, head_dim * heads),
            wk: lin(dim, head_dim * kvheads),
            wv: lin(dim, head_dim * kvheads),
            gate: lin(dim, dim),
            wo: lin(dim, dim),
            qknorm: QkNorm {
                qnorm: ZRmsNorm::init(head_dim, device),
                knorm: ZRmsNorm::init(head_dim, device),
            },
            heads,
            kvheads,
        }
    }
}

impl<B: QuantBackend> MmditAttention<B> {
    /// `x` is `[b, l, dim]`; `rope` is `None` in the text-fusion blocks;
    /// `mask` the additive `[b, 1, l, l]` attention mask. `adapters`/`prefix`
    /// route the projections through injected LoRA deltas.
    fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: Option<&RopeTables<B>>,
        mask: Option<Tensor<B, 4>>,
        adapters: Option<&LoraAdapters<B>>,
        prefix: &str,
    ) -> Tensor<B, 3> {
        let [b, l, dim] = x.dims();
        let (heads, kv) = (self.heads, self.kvheads);
        let hd = dim / heads;

        let split = |t: Tensor<B, 3>, n: usize| t.reshape([b, l, n, hd]).swap_dims(1, 2);
        let q = split(
            site(adapters, &format!("{prefix}.wq"), &self.wq, x.clone()),
            heads,
        );
        let k = split(
            site(adapters, &format!("{prefix}.wk"), &self.wk, x.clone()),
            kv,
        );
        let v = split(
            site(adapters, &format!("{prefix}.wv"), &self.wv, x.clone()),
            kv,
        );
        let gate = self.gate.forward(x);

        let q = self.qknorm.qnorm.forward(q);
        let k = self.qknorm.knorm.forward(k);
        let (q, k) = match rope {
            Some(tables) => (apply_rope(q, tables), apply_rope(k, tables)),
            None => (q, k),
        };

        // GQA: repeat KV heads over their query groups (HF repeat_kv order).
        let groups = heads / kv;
        let expand_kv = |t: Tensor<B, 4>| {
            t.reshape([b, kv, 1, l, hd])
                .expand([b, kv, groups, l, hd])
                .reshape([b, heads, l, hd])
        };
        let k = expand_kv(k);
        let v = expand_kv(v);

        // Attention scores. On an f16 backend the raw q·k products overflow
        // (post-norm q/k carry ~600-magnitude Qwen outlier channels →
        // products ~3.6e5 > f16 max), so both operands are pre-scaled down
        // and the score rescaled after the reduction — the TRUE scores
        // (observed ≤ ~800 on the real model) fit f16 comfortably; only the
        // intermediates don't. Deliberately NOT an f32-cast island: burn
        // 0.21's wgpu f32 kernels are the broken ones on this platform (see
        // `examples/grad_compare.rs`), while all-f16 gradients verify
        // against CPU ground truth. The softmaxed weights are ≤ 1 and the
        // value mix is convex (bounded by max |v|), so the ctx matmul needs
        // no treatment. f32/f64 backends take the literal formula,
        // byte-identical to the parity goldens.
        let scale = (hd as f64).sqrt();
        let mut scores = if q.dtype() == DType::F16 {
            const QC: f64 = 32.0;
            (q / QC)
                .matmul(k.swap_dims(2, 3) / QC)
                .mul_scalar(QC * QC / scale)
        } else {
            q.matmul(k.swap_dims(2, 3)).div_scalar(scale)
        };
        if let Some(m) = mask {
            scores = scores + m;
        }
        let ctx = softmax(scores, 3).matmul(v); // [b, heads, l, hd]
        let merged = ctx.swap_dims(1, 2).reshape([b, l, heads * hd]);

        // The gated-sigmoid: gate the merged attention output, then project.
        site(
            adapters,
            &format!("{prefix}.wo"),
            &self.wo,
            merged * sigmoid(gate),
        )
    }
}

/// The SwiGLU feed-forward (`down(silu(gate(x)) · up(x))`), inner width per
/// [`MmditConfig::swiglu_dim`].
#[derive(Module, Debug)]
pub struct SwiGlu<B: Backend> {
    /// Gate projection.
    pub gate: BaseLinear<B>,
    /// Up projection.
    pub up: BaseLinear<B>,
    /// Down projection.
    pub down: BaseLinear<B>,
}

impl<B: Backend> SwiGlu<B> {
    fn init(features: usize, multiplier: usize, device: &B::Device) -> Self {
        let inner = MmditConfig::swiglu_dim(features, multiplier);
        let lin = |d_in: usize, d_out: usize| {
            BaseLinear::Plain(LinearConfig::new(d_in, d_out).with_bias(false).init(device))
        };
        Self {
            gate: lin(features, inner),
            up: lin(features, inner),
            down: lin(inner, features),
        }
    }
}

impl<B: QuantBackend> SwiGlu<B> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        adapters: Option<&LoraAdapters<B>>,
        prefix: &str,
    ) -> Tensor<B, 3> {
        let gated = silu(site(
            adapters,
            &format!("{prefix}.gate"),
            &self.gate,
            x.clone(),
        )) * site(adapters, &format!("{prefix}.up"), &self.up, x);
        site(adapters, &format!("{prefix}.down"), &self.down, gated)
    }
}

/// The per-block learned modulation bias: `tvec + lin`, chunked into
/// pre/post scale, shift, gate (each `[b, 1, features]`).
#[derive(Module, Debug)]
pub struct DoubleSharedModulation<B: Backend> {
    /// The learned bias, `[6 · features]`, zero-initialized (the checkpoint's
    /// `blocks.N.mod.lin` — `mod` is a Rust keyword, remapped at load).
    pub lin: Param<Tensor<B, 1>>,
}

impl<B: Backend> DoubleSharedModulation<B> {
    /// `tvec` is `[b, 1, 6 · features]` → six `[b, 1, features]` chunks.
    fn forward(&self, tvec: Tensor<B, 3>) -> [Tensor<B, 3>; 6] {
        let f = self.lin.dims()[0] / 6;
        let out = tvec + self.lin.val().unsqueeze::<3>();
        core::array::from_fn(|i| out.clone().narrow(2, i * f, f))
    }
}

/// The output layer's 2-way modulation (`SimpleModulation`).
#[derive(Module, Debug)]
pub struct SimpleModulation<B: Backend> {
    /// The learned bias, `[2, features]`, zero-initialized.
    pub lin: Param<Tensor<B, 2>>,
}

impl<B: Backend> SimpleModulation<B> {
    /// `t` is `[b, 1, features]` → `(scale, shift)`, each `[b, 1, features]`.
    fn forward(&self, t: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let out = t + self.lin.val().unsqueeze::<3>(); // [b, 2, features]
        (out.clone().narrow(1, 0, 1), out.narrow(1, 1, 1))
    }
}

/// One text-fusion block: plain pre-norm attention + SwiGLU residuals (no
/// RoPE, no modulation).
#[derive(Module, Debug)]
pub struct TextFusionBlock<B: Backend> {
    /// Pre-attention norm.
    pub prenorm: ZRmsNorm<B>,
    /// Pre-MLP norm.
    pub postnorm: ZRmsNorm<B>,
    /// The attention (no GQA in the real config: 20/20 heads).
    pub attn: MmditAttention<B>,
    /// The feed-forward.
    pub mlp: SwiGlu<B>,
}

impl<B: Backend> TextFusionBlock<B> {
    fn init(
        dim: usize,
        heads: usize,
        kvheads: usize,
        multiplier: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            prenorm: ZRmsNorm::init(dim, device),
            postnorm: ZRmsNorm::init(dim, device),
            attn: MmditAttention::init(dim, heads, kvheads, device),
            mlp: SwiGlu::init(dim, multiplier, device),
        }
    }
}

impl<B: QuantBackend> TextFusionBlock<B> {
    fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        let x = x.clone()
            + self
                .attn
                .forward(self.prenorm.forward(x), None, mask, None, "");
        x.clone() + self.mlp.forward(self.postnorm.forward(x), None, "")
    }
}

/// The text-fusion transformer: 2 blocks across the conditioner's layer axis
/// (per token), a `Linear(txtlayers → 1)` projector, then 2 blocks along the
/// sequence.
#[derive(Module, Debug)]
pub struct TextFusionTransformer<B: Backend> {
    /// The per-token, across-layers blocks.
    pub layerwise_blocks: Vec<TextFusionBlock<B>>,
    /// Collapses the layer axis (`txtlayers → 1`, no bias).
    pub projector: Linear<B>,
    /// The along-sequence refiner blocks.
    pub refiner_blocks: Vec<TextFusionBlock<B>>,
}

impl<B: QuantBackend> TextFusionTransformer<B> {
    /// `x` is the conditioner stack `[b, l, txtlayers, txtdim]`; `mask` the
    /// additive text mask `[b, 1, l, l]` (layerwise blocks run unmasked, like
    /// the reference).
    fn forward(&self, x: Tensor<B, 4>, mask: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, l, n, d] = x.dims();
        let mut y = x.reshape([b * l, n, d]);
        for block in &self.layerwise_blocks {
            y = block.forward(y, None);
        }
        // [b·l, n, d] -> [b, l, d, n] -> project the layer axis away.
        let y = y.reshape([b, l, n, d]).swap_dims(2, 3);
        let mut y: Tensor<B, 3> = self.projector.forward(y).squeeze_dim(3);
        for block in &self.refiner_blocks {
            y = block.forward(y, Some(mask.clone()));
        }
        y
    }
}

/// One trunk block: modulated pre-norm attention + SwiGLU residuals.
#[derive(Module, Debug)]
pub struct SingleStreamBlock<B: Backend> {
    /// The per-block modulation bias (`mod` in the checkpoint).
    pub modulation: DoubleSharedModulation<B>,
    /// Pre-attention norm.
    pub prenorm: ZRmsNorm<B>,
    /// Pre-MLP norm.
    pub postnorm: ZRmsNorm<B>,
    /// The gated GQA attention.
    pub attn: MmditAttention<B>,
    /// The feed-forward.
    pub mlp: SwiGlu<B>,
}

impl<B: Backend> SingleStreamBlock<B> {
    fn init(config: &MmditConfig, device: &B::Device) -> Self {
        Self {
            modulation: DoubleSharedModulation {
                lin: Param::from_tensor(Tensor::zeros([6 * config.features], device)),
            },
            prenorm: ZRmsNorm::init(config.features, device),
            postnorm: ZRmsNorm::init(config.features, device),
            attn: MmditAttention::init(config.features, config.heads, config.kvheads, device),
            mlp: SwiGlu::init(config.features, config.multiplier, device),
        }
    }
}

impl<B: QuantBackend> SingleStreamBlock<B> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        tvec: Tensor<B, 3>,
        rope: &RopeTables<B>,
        mask: Tensor<B, 4>,
        adapters: Option<&LoraAdapters<B>>,
        prefix: &str,
    ) -> Tensor<B, 3> {
        let [prescale, preshift, pregate, postscale, postshift, postgate] =
            self.modulation.forward(tvec);
        let attn_in = (prescale + 1.0) * self.prenorm.forward(x.clone()) + preshift;
        let x = x + pregate
            * self.attn.forward(
                attn_in,
                Some(rope),
                Some(mask),
                adapters,
                &format!("{prefix}.attn"),
            );
        let mlp_in = (postscale + 1.0) * self.postnorm.forward(x.clone()) + postshift;
        x + postgate * self.mlp.forward(mlp_in, adapters, &format!("{prefix}.mlp"))
    }
}

/// The output head: modulated norm + linear back to patch pixels.
#[derive(Module, Debug)]
pub struct LastLayer<B: Backend> {
    /// Pre-head norm.
    pub norm: ZRmsNorm<B>,
    /// `features → patch² · channels` (with bias).
    pub linear: Linear<B>,
    /// The 2-way modulation.
    pub modulation: SimpleModulation<B>,
}

impl<B: Backend> LastLayer<B> {
    fn forward(&self, x: Tensor<B, 3>, t: Tensor<B, 3>) -> Tensor<B, 3> {
        let (scale, shift) = self.modulation.forward(t);
        let x = (scale + 1.0) * self.norm.forward(x) + shift;
        self.linear.forward(x)
    }
}

/// The timestep MLP (`tmlp`, a `Sequential` in the reference — indices
/// remapped to fields at load).
#[derive(Module, Debug)]
pub struct TimestepMlp<B: Backend> {
    /// `tdim → features` (the reference's `tmlp.0`).
    pub fc1: BaseLinear<B>,
    /// `features → features` (`tmlp.2`).
    pub fc2: BaseLinear<B>,
}

/// The modulation projector (`tproj`): GELU then `features → 6·features`.
#[derive(Module, Debug)]
pub struct TimestepProj<B: Backend> {
    /// The projection (`tproj.1`).
    pub fc: BaseLinear<B>,
}

/// The fused-text projector (`txtmlp`): norm, up, GELU, out.
#[derive(Module, Debug)]
pub struct TextMlp<B: Backend> {
    /// `txtmlp.0` — the zero-centered norm.
    pub norm: ZRmsNorm<B>,
    /// `txtmlp.1` — `txtdim → features`.
    pub fc1: BaseLinear<B>,
    /// `txtmlp.3` — `features → features`.
    pub fc2: BaseLinear<B>,
}

/// Intermediate activations captured by [`Mmdit::forward_trace`], for parity
/// localization. Stages at masked/padded positions are attention garbage by
/// contract — zero them (via the masks) before comparing to a golden.
pub struct MmditTrace<B: Backend> {
    /// After the patchified-input projection (`first`).
    pub after_first: Tensor<B, 3>,
    /// The 6-way modulation vector (`tproj` output), `[b, 1, 6·features]`.
    pub tvec: Tensor<B, 3>,
    /// The fused text after [`TextFusionTransformer`], `[b, txtlen, txtdim]`.
    pub after_txtfusion: Tensor<B, 3>,
    /// The fused text projected to trunk width, `[b, txtlen, features]`.
    pub after_txtmlp: Tensor<B, 3>,
    /// The combined (padded-to-256) sequence after trunk block 0.
    pub after_block0: Tensor<B, 3>,
    /// The final velocity prediction over the image tokens,
    /// `[b, img_tokens, patch² · channels]`.
    pub output: Tensor<B, 3>,
}

/// The Krea 2 single-stream MMDiT denoiser.
///
/// Build with [`Mmdit::init`], populate with `burn_store` using
/// [`key_remap`](Self::key_remap) + the `PyTorchToBurnAdapter`, then run
/// [`forward`](Self::forward) (or [`forward_with_adapters`](Self::forward_with_adapters)
/// with an M6 LoRA set injected).
#[derive(Module, Debug)]
pub struct Mmdit<B: Backend> {
    /// Patchified-latent input projection (`channels·patch² → features`).
    pub first: Linear<B>,
    /// Timestep MLP.
    pub tmlp: TimestepMlp<B>,
    /// Modulation projector.
    pub tproj: TimestepProj<B>,
    /// The text-fusion transformer.
    pub txtfusion: TextFusionTransformer<B>,
    /// The fused-text projector to trunk width.
    pub txtmlp: TextMlp<B>,
    /// The trunk.
    pub blocks: Vec<SingleStreamBlock<B>>,
    /// The output head.
    pub last: LastLayer<B>,
    /// The architecture — drives the forward, not a loadable parameter.
    #[module(skip)]
    pub config: MmditConfig,
}

impl<B: Backend> Mmdit<B> {
    /// Build an MMDiT with placeholder weights of the right shapes, ready to
    /// be overwritten by `load_from`.
    pub fn init(config: MmditConfig, device: &B::Device) -> Self {
        let f = config.features;
        let lin = |d_in: usize, d_out: usize| LinearConfig::new(d_in, d_out).init(device);
        let lin_nb = |d_in: usize, d_out: usize| {
            LinearConfig::new(d_in, d_out).with_bias(false).init(device)
        };
        // Quantizable base sites start `Plain` (with bias, like `lin`);
        // `into_quantized` swaps in the int8 twin.
        let base = |d_in: usize, d_out: usize| BaseLinear::Plain(lin(d_in, d_out));
        let fusion_block = || {
            TextFusionBlock::init(
                config.txtdim,
                config.txtheads,
                config.txtkvheads,
                config.multiplier,
                device,
            )
        };
        Self {
            first: lin(config.channels * config.patch * config.patch, f),
            tmlp: TimestepMlp {
                fc1: base(config.tdim, f),
                fc2: base(f, f),
            },
            tproj: TimestepProj { fc: base(f, 6 * f) },
            txtfusion: TextFusionTransformer {
                layerwise_blocks: vec![fusion_block(), fusion_block()],
                projector: lin_nb(config.txtlayers, 1),
                refiner_blocks: vec![fusion_block(), fusion_block()],
            },
            txtmlp: TextMlp {
                norm: ZRmsNorm::init(config.txtdim, device),
                fc1: base(config.txtdim, f),
                fc2: base(f, f),
            },
            blocks: (0..config.layers)
                .map(|_| SingleStreamBlock::init(&config, device))
                .collect(),
            last: LastLayer {
                norm: ZRmsNorm::init(f, device),
                linear: lin(f, config.patch * config.patch * config.channels),
                modulation: SimpleModulation {
                    lin: Param::from_tensor(Tensor::zeros([2, f], device)),
                },
            },
            config,
        }
    }

    /// The regex → replacement rename pairs mapping checkpoint keys to burn
    /// module paths: the reference's `nn.Sequential` indices and its `mod`
    /// field (a Rust keyword). Everything else loads by name (with the
    /// `PyTorchToBurnAdapter` Linear transpose).
    pub fn key_remap() -> [(&'static str, &'static str); 7] {
        [
            (r"^tmlp\.0\.(weight|bias)$", r"tmlp.fc1.${1}"),
            (r"^tmlp\.2\.(weight|bias)$", r"tmlp.fc2.${1}"),
            (r"^tproj\.1\.(weight|bias)$", r"tproj.fc.${1}"),
            (r"^txtmlp\.0\.scale$", r"txtmlp.norm.scale"),
            (r"^txtmlp\.1\.(weight|bias)$", r"txtmlp.fc1.${1}"),
            (r"^txtmlp\.3\.(weight|bias)$", r"txtmlp.fc2.${1}"),
            (r"\.mod\.lin$", r".modulation.lin"),
        ]
    }

    /// Every injectable LoRA site: the trunk blocks' attention and MLP
    /// projections, advertised to
    /// [`build_adapters`](crate::adapters::build_adapters). Paths mirror the
    /// checkpoint (`blocks.{i}.attn.wq`, …), so a config `targets` pattern
    /// written against the reference naming matches.
    pub fn injectable_sites(&self) -> Vec<LoraSite> {
        let f = self.config.features;
        let hd = self.config.head_dim();
        let kv_out = hd * self.config.kvheads;
        let inner = MmditConfig::swiglu_dim(f, self.config.multiplier);
        let mut sites = Vec::with_capacity(self.config.layers * 7);
        for i in 0..self.config.layers {
            let p = format!("blocks.{i}");
            for (name, d_in, d_out) in [
                ("attn.wq", f, f),
                ("attn.wk", f, kv_out),
                ("attn.wv", f, kv_out),
                ("attn.wo", f, f),
                ("mlp.gate", f, inner),
                ("mlp.up", f, inner),
                ("mlp.down", inner, f),
            ] {
                sites.push(LoraSite {
                    path: format!("{p}.{name}"),
                    d_in,
                    d_out,
                });
            }
        }
        sites
    }

    /// Replace every frozen-base [`BaseLinear::Plain`] site with its int8
    /// [`BaseLinear::Quant`] twin (weight-only, per-block symmetric int8 via
    /// [`quantize_linear_weight`]) — the memory knob for the ~12B base
    /// (#24 → #96). burn's `Linear` stores its weight `[d_in, d_out]` and
    /// computes `x·W`; the quant path wants file layout `[d_out, d_in]` and
    /// computes `x · dequant(wq)ᵀ`, so each weight is **transposed** before
    /// quantizing and the two are numerically equal up to int8 error (proven
    /// in `tests/quant_mmdit.rs`). The M6 LoRA seam is untouched — adapters
    /// attach on the quantized site's output exactly as on the plain one.
    ///
    /// A site whose `d_in` is not a multiple of [`QUANT_BLOCK`] is left
    /// `Plain` (the block scheme can't tile it). On the real Krea 2 config
    /// every quantizable site is block-aligned; only tiny fixtures (e.g.
    /// `tmlp.fc1` at `tdim = 16`) keep a stray `Plain` site.
    ///
    /// Must run on a **non-autodiff** backend: burn 0.21's `Autodiff` has no
    /// `quantize` op (`todo!()`), so the trainer quantizes on the inner
    /// backend and lifts the module with
    /// [`AutodiffModule::from_inner`](burn::module::AutodiffModule::from_inner)
    /// — the pattern `tests/quant.rs` and `tests/quant_mmdit.rs` follow. The
    /// `device` argument is reserved for a future scheme that needs it
    /// (`quantize_dynamic` quantizes on each tensor's own device today).
    pub fn into_quantized(mut self, _device: &B::Device) -> Self {
        for block in &mut self.blocks {
            quantize_attention(&mut block.attn);
            quantize_swiglu(&mut block.mlp);
        }
        for block in &mut self.txtfusion.layerwise_blocks {
            quantize_attention(&mut block.attn);
            quantize_swiglu(&mut block.mlp);
        }
        for block in &mut self.txtfusion.refiner_blocks {
            quantize_attention(&mut block.attn);
            quantize_swiglu(&mut block.mlp);
        }
        quantize_field(&mut self.tmlp.fc1);
        quantize_field(&mut self.tmlp.fc2);
        quantize_field(&mut self.tproj.fc);
        quantize_field(&mut self.txtmlp.fc1);
        quantize_field(&mut self.txtmlp.fc2);
        self
    }

    /// Every injectable trunk site paired with its checkpoint-key base path
    /// (`blocks.{i}.attn.wq`, …) as a mutable handle — the same paths and
    /// order [`injectable_sites`](Self::injectable_sites) advertises (pinned
    /// by `tests/quant_mmdit.rs`). PR-B3's streaming loader fills each site
    /// from a checkpoint through this handle.
    pub fn base_linears_mut(&mut self) -> Vec<(String, &mut BaseLinear<B>)> {
        let mut out: Vec<(String, &mut BaseLinear<B>)> = Vec::with_capacity(self.blocks.len() * 7);
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let p = format!("blocks.{i}");
            let SingleStreamBlock { attn, mlp, .. } = block;
            out.push((format!("{p}.attn.wq"), &mut attn.wq));
            out.push((format!("{p}.attn.wk"), &mut attn.wk));
            out.push((format!("{p}.attn.wv"), &mut attn.wv));
            out.push((format!("{p}.attn.wo"), &mut attn.wo));
            out.push((format!("{p}.mlp.gate"), &mut mlp.gate));
            out.push((format!("{p}.mlp.up"), &mut mlp.up));
            out.push((format!("{p}.mlp.down"), &mut mlp.down));
        }
        out
    }

    /// **Every** quantizable frozen-base [`BaseLinear`] site paired with its
    /// remapped-checkpoint-key path — the superset of
    /// [`base_linears_mut`](Self::base_linears_mut) that PR-B3's streaming
    /// quantized loader must overwrite.
    ///
    /// [`base_linears_mut`](Self::base_linears_mut) advertises only the
    /// LoRA-*injectable* trunk subset
    /// ([`injectable_sites`](Self::injectable_sites): 7 sites per trunk block,
    /// **no** `attn.gate`), because that is the surface adapters attach to. But
    /// [`into_quantized`](Self::into_quantized) quantizes every base linear —
    /// the trunk blocks' `attn.gate` too, plus the four text-fusion blocks and
    /// the `tmlp`/`tproj`/`txtmlp` projections — so a loader that fills the
    /// int8 weights from a checkpoint must enumerate **all** of them, or those
    /// `Quant` sites keep their placeholder random weights (a silent,
    /// catastrophic load bug on the real model, where every base linear is
    /// block-aligned and therefore quantized). This is that enumeration; its
    /// order mirrors `into_quantized`'s traversal and its coverage is pinned
    /// against `into_quantized` in `tests/quant_mmdit.rs`.
    ///
    /// Keys are **remapped module paths** (post-[`key_remap`](Self::key_remap)):
    /// `blocks.{i}.attn.gate`, `txtfusion.layerwise_blocks.{i}.mlp.up`,
    /// `tmlp.fc1`, `tproj.fc`, `txtmlp.fc2`, … — the keys a checkpoint snapshot
    /// carries once [`key_remap`](Self::key_remap) has been applied, so the
    /// loader looks up `{path}.weight` directly.
    pub fn all_base_linears_mut(&mut self) -> Vec<(String, &mut BaseLinear<B>)> {
        // Push an attention block's five projections (wq, wk, wv, gate, wo) —
        // the order `quantize_attention` uses. Disjoint field reborrows of the
        // `&mut attn`, exactly as `base_linears_mut` does inline.
        fn push_attn<'a, B: Backend>(
            out: &mut Vec<(String, &'a mut BaseLinear<B>)>,
            prefix: &str,
            attn: &'a mut MmditAttention<B>,
        ) {
            out.push((format!("{prefix}.attn.wq"), &mut attn.wq));
            out.push((format!("{prefix}.attn.wk"), &mut attn.wk));
            out.push((format!("{prefix}.attn.wv"), &mut attn.wv));
            out.push((format!("{prefix}.attn.gate"), &mut attn.gate));
            out.push((format!("{prefix}.attn.wo"), &mut attn.wo));
        }
        // Push a SwiGLU's three projections (gate, up, down) —
        // `quantize_swiglu`'s order.
        fn push_swiglu<'a, B: Backend>(
            out: &mut Vec<(String, &'a mut BaseLinear<B>)>,
            prefix: &str,
            mlp: &'a mut SwiGlu<B>,
        ) {
            out.push((format!("{prefix}.mlp.gate"), &mut mlp.gate));
            out.push((format!("{prefix}.mlp.up"), &mut mlp.up));
            out.push((format!("{prefix}.mlp.down"), &mut mlp.down));
        }

        let mut out: Vec<(String, &mut BaseLinear<B>)> = Vec::new();
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let p = format!("blocks.{i}");
            let SingleStreamBlock { attn, mlp, .. } = block;
            push_attn(&mut out, &p, attn);
            push_swiglu(&mut out, &p, mlp);
        }
        for (i, block) in self.txtfusion.layerwise_blocks.iter_mut().enumerate() {
            let p = format!("txtfusion.layerwise_blocks.{i}");
            let TextFusionBlock { attn, mlp, .. } = block;
            push_attn(&mut out, &p, attn);
            push_swiglu(&mut out, &p, mlp);
        }
        for (i, block) in self.txtfusion.refiner_blocks.iter_mut().enumerate() {
            let p = format!("txtfusion.refiner_blocks.{i}");
            let TextFusionBlock { attn, mlp, .. } = block;
            push_attn(&mut out, &p, attn);
            push_swiglu(&mut out, &p, mlp);
        }
        out.push(("tmlp.fc1".to_string(), &mut self.tmlp.fc1));
        out.push(("tmlp.fc2".to_string(), &mut self.tmlp.fc2));
        out.push(("tproj.fc".to_string(), &mut self.tproj.fc));
        out.push(("txtmlp.fc1".to_string(), &mut self.txtmlp.fc1));
        out.push(("txtmlp.fc2".to_string(), &mut self.txtmlp.fc2));
        out
    }
}

/// Quantize one base site in place: a block-aligned [`BaseLinear::Plain`]
/// becomes its int8 [`BaseLinear::Quant`] twin (the `[d_in, d_out]` Linear
/// weight transposed to file layout `[d_out, d_in]`, then
/// [`quantize_linear_weight`]); a `Quant` arm, or a `Plain` whose `d_in` is
/// not a multiple of [`QUANT_BLOCK`], is left untouched.
fn quantize_field<B: Backend>(base: &mut BaseLinear<B>) {
    let BaseLinear::Plain(lin) = base else {
        return;
    };
    let [d_in, _d_out] = lin.weight.dims();
    if !d_in.is_multiple_of(QUANT_BLOCK) {
        return;
    }
    let wq = quantize_linear_weight(lin.weight.val().transpose());
    let bias = lin.bias.clone();
    *base = BaseLinear::Quant(QuantLinear {
        weight: Param::from_tensor(wq),
        bias,
    });
}

/// Quantize an attention block's five projections in place.
fn quantize_attention<B: Backend>(attn: &mut MmditAttention<B>) {
    quantize_field(&mut attn.wq);
    quantize_field(&mut attn.wk);
    quantize_field(&mut attn.wv);
    quantize_field(&mut attn.gate);
    quantize_field(&mut attn.wo);
}

/// Quantize a SwiGLU's three projections in place.
fn quantize_swiglu<B: Backend>(mlp: &mut SwiGlu<B>) {
    quantize_field(&mut mlp.gate);
    quantize_field(&mut mlp.up);
    quantize_field(&mut mlp.down);
}

impl<B: QuantBackend> Mmdit<B> {
    /// Denoise: `img` is the pre-patchified latent tokens
    /// `[b, img_tokens, channels·patch²]`, `context` the M10 conditioner
    /// stack `[b, txtlen, txtlayers, txtdim]`, `t` the per-sample timesteps
    /// `[b]`, `pos` the 3-axis positions `[b, txtlen + img_tokens, 3]`
    /// (text at the origin, image on the patch grid), and `mask` the 0/1
    /// key mask over the combined sequence. Returns the velocity prediction
    /// `[b, img_tokens, channels·patch²]`.
    pub fn forward(
        &self,
        img: Tensor<B, 3>,
        context: Tensor<B, 4>,
        t: Tensor<B, 1>,
        pos: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        self.forward_inner(img, context, t, pos, mask, None).output
    }

    /// [`forward`](Self::forward) with the name-keyed M6 LoRA set injected at
    /// every matching trunk site. Zero-initialized deltas make this
    /// bit-identical to the plain forward — the free attach-integrity check.
    pub fn forward_with_adapters(
        &self,
        img: Tensor<B, 3>,
        context: Tensor<B, 4>,
        t: Tensor<B, 1>,
        pos: Tensor<B, 3>,
        mask: Tensor<B, 2>,
        adapters: &LoraAdapters<B>,
    ) -> Tensor<B, 3> {
        self.forward_inner(img, context, t, pos, mask, Some(adapters))
            .output
    }

    /// [`forward`](Self::forward) capturing localizing intermediates.
    pub fn forward_trace(
        &self,
        img: Tensor<B, 3>,
        context: Tensor<B, 4>,
        t: Tensor<B, 1>,
        pos: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> MmditTrace<B> {
        self.forward_inner(img, context, t, pos, mask, None)
    }

    fn forward_inner(
        &self,
        img: Tensor<B, 3>,
        context: Tensor<B, 4>,
        t: Tensor<B, 1>,
        pos: Tensor<B, 3>,
        mask: Tensor<B, 2>,
        adapters: Option<&LoraAdapters<B>>,
    ) -> MmditTrace<B> {
        let device = img.device();
        let [b, imglen, _] = img.dims();
        let txtlen = context.dims()[1];
        let gelu = Gelu::new_approximate();

        // 1. Patch tokens into the trunk width.
        let img = self.first.forward(img);
        let after_first = img.clone();

        // 2. Timestep embedding -> per-sample [b, 1, features] -> 6-way vec.
        let t = self
            .tmlp
            .fc2
            .forward(
                gelu.forward(
                    self.tmlp
                        .fc1
                        .forward(temb::<B>(t, self.config.tdim, &device)),
                ),
            );
        let tvec = self.tproj.fc.forward(gelu.forward(t.clone()));

        // 3. Fuse the 12-layer conditioner stack, then project to trunk width.
        let txtmask = mask.clone().narrow(1, 0, txtlen);
        let after_txtfusion = self
            .txtfusion
            .forward(context, additive_mask(txtmask, &device));
        let after_txtmlp = self.txtmlp.fc2.forward(
            gelu.forward(
                self.txtmlp
                    .fc1
                    .forward(self.txtmlp.norm.forward(after_txtfusion.clone())),
            ),
        );

        // 4. Concatenate text + image and zero-pad to a multiple of 256
        // (mask false, positions zero), exactly like the reference.
        let mut combined = Tensor::cat(vec![after_txtmlp.clone(), img], 1);
        let mut mask = mask;
        let mut pos = pos;
        let fulllen = combined.dims()[1];
        let padlen = fulllen.div_ceil(256) * 256 - fulllen;
        if padlen > 0 {
            let f = self.config.features;
            combined = Tensor::cat(vec![combined, Tensor::zeros([b, padlen, f], &device)], 1);
            mask = Tensor::cat(vec![mask, Tensor::zeros([b, padlen], &device)], 1);
            pos = Tensor::cat(vec![pos, Tensor::zeros([b, padlen, 3], &device)], 1);
        }

        let mask4 = additive_mask(mask, &device);
        let rope = rope_tables(pos, self.config.head_dim(), self.config.theta, &device);

        // 5. The trunk.
        let mut after_block0 = combined.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            combined = block.forward(
                combined,
                tvec.clone(),
                &rope,
                mask4.clone(),
                adapters,
                &format!("blocks.{i}"),
            );
            if i == 0 {
                after_block0 = combined.clone();
            }
        }

        // 6. Head, then slice the image tokens back out.
        let final_ = self.last.forward(combined, t);
        let output = final_.narrow(1, txtlen, imglen);

        MmditTrace {
            after_first,
            tvec,
            after_txtfusion,
            after_txtmlp,
            after_block0,
            output,
        }
    }
}

/// The symmetric additive attention mask: `MASK_NEG` wherever the query⊗key
/// outer product of the 0/1 key mask is 0 (the reference's `_mask` with bool
/// semantics folded into an additive form).
fn additive_mask<B: Backend>(mask: Tensor<B, 2>, _device: &B::Device) -> Tensor<B, 4> {
    let [b, l] = mask.dims();
    let outer = mask.clone().reshape([b, 1, l, 1]) * mask.reshape([b, 1, 1, l]);
    (-outer + 1.0) * MASK_NEG
}

/// The sinusoidal timestep embedding (`temb`): cos-first, period 1e4, the
/// timestep scaled ×1000, shaped `[b, 1, dim]` so downstream modulation
/// broadcasts per sample.
fn temb<B: Backend>(t: Tensor<B, 1>, dim: usize, device: &B::Device) -> Tensor<B, 3> {
    let b = t.dims()[0];
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|j| (-(1e4f64.ln()) * j as f64 / half as f64).exp() as f32)
        .collect();
    let freqs =
        Tensor::<B, 1>::from_data(TensorData::new(freqs, [half]), device).reshape([1, 1, half]);
    let args = t.mul_scalar(1e3).reshape([b, 1, 1]) * freqs;
    Tensor::cat(vec![args.clone().cos(), args.sin()], 2)
}

/// Patchify a latent `[b, c, h, w]` into MMDiT image tokens
/// `[b, (h/p)·(w/p), c·p²]` — `sampling.py`'s
/// `rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)")`, channel-major
/// within each patch. `h`/`w` must divide by `patch`.
pub fn patchify<B: Backend>(latent: Tensor<B, 4>, patch: usize) -> Tensor<B, 3> {
    let [b, c, h, w] = latent.dims();
    assert!(
        h.is_multiple_of(patch) && w.is_multiple_of(patch),
        "latent {h}x{w} not divisible by patch {patch}"
    );
    let (gh, gw) = (h / patch, w / patch);
    latent
        .reshape([b, c, gh, patch, gw, patch])
        .permute([0, 2, 4, 1, 3, 5]) // [b, gh, gw, c, p, p]
        .reshape([b, gh * gw, c * patch * patch])
}

/// The Krea 2 position grid for a combined text+image sequence
/// (`sampling.py`'s `prepare()`): text tokens all at the origin `(0, 0, 0)`,
/// image tokens at `(0, row, col)` on the `gh × gw` patch grid. Returns
/// `[batch, txt_len + gh·gw, 3]`.
pub fn krea2_positions<B: Backend>(
    txt_len: usize,
    gh: usize,
    gw: usize,
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let len = txt_len + gh * gw;
    let mut pos = vec![0.0f32; len * 3];
    for row in 0..gh {
        for col in 0..gw {
            let token = txt_len + row * gw + col;
            pos[token * 3 + 1] = row as f32;
            pos[token * 3 + 2] = col as f32;
        }
    }
    let one = Tensor::<B, 2>::from_data(TensorData::new(pos, [len, 3]), device);
    Tensor::cat(vec![one.unsqueeze_dim::<3>(0); batch], 0)
}
