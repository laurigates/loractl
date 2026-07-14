//! The Qwen-Image latent VAE (`AutoencoderKLQwenImage`) — Krea 2's autoencoder
//! (M9, #20).
//!
//! Krea 2's `autoencoder.py` is a 22-line wrapper around diffusers'
//! `AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae")`
//! plus per-channel `latents_mean`/`latents_std` (de)normalization, so *that*
//! diffusers module (`autoencoder_kl_qwenimage.py`, itself derived from the Wan
//! video VAE) is the authoritative architecture source this port mirrors. It is
//! an **f8, 16-latent-channel** VAE: images compress 8× spatially into 16
//! channels (`z_dim = 16`), the latent space the Krea 2 MMDiT (M11) denoises
//! and the dataset pipeline (M12) caches.
//!
//! ## What the source actually says (report-vs-code findings)
//!
//! - The published `Qwen/Qwen-Image` VAE config is `base_dim = 96`,
//!   `dim_mult = [1, 2, 4, 4]`, `num_res_blocks = 2`, `z_dim = 16`,
//!   `temperal_downsample = [false, true, true]`, `attn_scales = []`.
//! - `attn_scales = []` removes attention from the *down/up trunks only* — the
//!   **mid block always carries one single-head spatial self-attention**
//!   (`QwenImageMidBlock` hardcodes `num_layers = 1`). The "attention-free"
//!   shorthand in the issue notes was wrong; [`SpatialAttention`] ports it.
//! - `latents_mean`/`latents_std` are **config values, not checkpoint
//!   tensors** (16 measured per-channel latent statistics registered as
//!   buffers by Krea's wrapper). They live on [`QwenVaeConfig`].
//!
//! ## Image-only (`T = 1`) semantics
//!
//! The VAE is a *video* model: 5-D `[batch, channel, time, height, width]`
//! tensors, causal 3-D convolutions, and a per-4-frame feature-cache protocol
//! for chunked streaming. loractl trains **image** LoRAs, so this port fixes
//! `T = 1` — exactly what Krea 2's wrapper does (`rearrange(x, "b c h w -> b c
//! 1 h w")`). At `T = 1` the reference's cache path is equivalent to its
//! cache-free path (every cache entry is `None` on a first chunk), and the
//! resample blocks' `time_conv`s are **skipped entirely** (both `upsample3d`'s
//! frame-doubling and `downsample3d`'s temporal stride only fire from the
//! second chunk on). The `time_conv` *parameters* are still declared so the
//! checkpoint loads with `missing` empty; the forward documents where the
//! reference skips them.
//!
//! ## Weight loading — PyTorch layout verbatim, one Sequential rename
//!
//! burn's `Conv2d`/`Conv3d` weights share PyTorch's `[out, in, k…]` layout, and
//! diffusers already names the custom RMS-norm parameter `gamma`, so *every*
//! tensor loads by name without transposing. The single remap
//! ([`QwenVae::key_remap`]) flattens the reference's `nn.Sequential` index in
//! the spatial resample convs: `….resample.1.weight` → `….resample.weight`
//! (index 0 is a parameter-less pad/upsample). The heterogeneous
//! `encoder.down_blocks.{i}` list (res blocks and resamples sharing one flat
//! index space) is mirrored by [`DownLayer`], a union module whose field names
//! reproduce the checkpoint paths exactly.
//!
//! Like the rest of `loractl-core`, this module emits no output and imports no
//! CLI: it is pure model code, proven by the staged parity harness
//! (`tests/qwen_vae_parity.rs`) against `reference/qwen_vae_reference.py`.

use burn::module::{Module, Param};
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::module::{conv3d, interpolate};
use burn::tensor::ops::{ConvOptions, InterpolateMode, InterpolateOptions, PadMode};
use burn::tensor::{Tensor, TensorData};

/// Static architecture of an `AutoencoderKLQwenImage` variant.
///
/// Field names mirror the diffusers config JSON. Held on [`QwenVae`] as a
/// non-parameter (`#[module(skip)]`) field — it drives construction and the
/// latent (de)normalization but is not a tensor to load. `attn_scales` (always
/// `[]` in the published config) and `dropout` (0.0, and the VAE is frozen for
/// LoRA training) are deliberately not modeled; `input_channels` is fixed at 3.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenVaeConfig {
    /// Base channel width; trunk widths are `base_dim * dim_mult[i]`.
    pub base_dim: usize,
    /// Latent channel count (the encoder's moments have `2 * z_dim` channels).
    pub z_dim: usize,
    /// Per-stage channel multipliers (the number of stages).
    pub dim_mult: Vec<usize>,
    /// Residual blocks per encoder stage (decoder stages use one more).
    pub num_res_blocks: usize,
    /// Per-downsample temporal flags: `true` = `downsample3d` (adds a
    /// `time_conv`), `false` = `downsample2d`. Reversed for the decoder's
    /// upsamplers. Length `dim_mult.len() - 1`.
    pub temperal_downsample: Vec<bool>,
    /// Measured per-channel latent means (length `z_dim`) — config values from
    /// the model card, not checkpoint tensors.
    pub latents_mean: Vec<f64>,
    /// Measured per-channel latent standard deviations (length `z_dim`).
    pub latents_std: Vec<f64>,
}

impl QwenVaeConfig {
    /// The tiny fixture config used by the always-run offline parity test.
    /// Must match `reference/qwen_vae_reference.py`'s `TINY_CFG` exactly. It
    /// exercises every layer kind the real config uses: `downsample2d` *and*
    /// `downsample3d` (+`time_conv`), `upsample3d` *and* `upsample2d`, res
    /// blocks with and without `conv_shortcut`, the mid-block attention, and
    /// (via `num_res_blocks = 2`, like the real config) the second
    /// same-stage residual block — so the constructors' `in_dim → out_dim`
    /// advance is covered without the opt-in real-weights test.
    pub fn tiny() -> Self {
        Self {
            base_dim: 8,
            z_dim: 4,
            dim_mult: vec![1, 2, 2],
            num_res_blocks: 2,
            temperal_downsample: vec![false, true],
            latents_mean: vec![0.1, -0.2, 0.3, -0.4],
            latents_std: vec![1.5, 0.8, 1.2, 2.0],
        }
    }

    /// The real `Qwen/Qwen-Image` VAE (the checkpoint Krea 2 uses), per its
    /// `vae/config.json`: f8 spatial compression into 16 latent channels.
    pub fn qwen_image() -> Self {
        Self {
            base_dim: 96,
            z_dim: 16,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temperal_downsample: vec![false, true, true],
            latents_mean: vec![
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134,
                -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
            ],
            latents_std: vec![
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526,
                2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
            ],
        }
    }

    /// The spatial compression factor: one 2× downsample per stage transition
    /// (`dim_mult.len() - 1` of them). 8 for the real config, 4 for the tiny.
    pub fn spatial_compression(&self) -> usize {
        1 << (self.dim_mult.len() - 1)
    }
}

/// A causal 3-D convolution (`QwenImageCausalConv3d`): time is padded
/// asymmetrically (all `2 * padding` zeros on the "past" side), height/width
/// symmetrically, then a valid [`conv3d`] runs. Owns its parameters directly so
/// its burn paths are exactly the checkpoint's `….weight` / `….bias`.
///
/// Weights are placeholder zeros at construction — a [`QwenVae`] is only
/// meaningful after `load_from` populates it.
#[derive(Module, Debug)]
pub struct CausalConv3d<B: Backend> {
    /// Kernel `[out, in, k_t, k_h, k_w]` — PyTorch layout, loaded verbatim.
    pub weight: Param<Tensor<B, 5>>,
    /// Bias `[out]`.
    pub bias: Param<Tensor<B, 1>>,
    /// Stride `[t, h, w]`.
    #[module(skip)]
    pub stride: [usize; 3],
    /// The reference's *symmetric* padding parameter `[t, h, w]`; the forward
    /// converts the `t` component into causal (left-only, doubled) padding.
    #[module(skip)]
    pub padding: [usize; 3],
}

impl<B: Backend> CausalConv3d<B> {
    fn init(
        in_ch: usize,
        out_ch: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
        padding: [usize; 3],
        device: &B::Device,
    ) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros(
                [out_ch, in_ch, kernel[0], kernel[1], kernel[2]],
                device,
            )),
            bias: Param::from_tensor(Tensor::zeros([out_ch], device)),
            stride,
            padding,
        }
    }

    /// Causal forward: `[b, c, t, h, w]` → `[b, c', t', h', w']`.
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let [pt, ph, pw] = self.padding;
        let x = if pt + ph + pw > 0 {
            // Causality: the whole temporal budget (2·pt) goes on the left
            // (the "past"); nothing on the right (the "future"). Spatial
            // padding stays symmetric. Mirrors the reference's `F.pad`.
            x.pad([(2 * pt, 0), (ph, ph), (pw, pw)], PadMode::Constant(0.0))
        } else {
            x
        };
        conv3d(
            x,
            self.weight.val(),
            Some(self.bias.val()),
            ConvOptions::new(self.stride, [0, 0, 0], [1, 1, 1], 1),
        )
    }
}

/// `QwenImageRMS_norm` over a 5-D video tensor (`images = false`): L2-normalize
/// along the channel dim, rescale by `sqrt(C)`, multiply by a per-channel
/// `gamma` of shape `[C, 1, 1, 1]`. No bias (the reference's `bias = False`).
#[derive(Module, Debug)]
pub struct RmsNormVideo<B: Backend> {
    /// Per-channel gain, `[C, 1, 1, 1]` (matches the checkpoint shape).
    pub gamma: Param<Tensor<B, 4>>,
}

impl<B: Backend> RmsNormVideo<B> {
    fn init(dim: usize, device: &B::Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([dim, 1, 1, 1], device)),
        }
    }

    /// `x / max(‖x‖₂, 1e-12) · sqrt(C) · gamma`, channel-wise.
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let c = x.dims()[1];
        let norm = x.clone().powi_scalar(2).sum_dim(1).sqrt().clamp_min(1e-12);
        x.div(norm)
            .mul_scalar((c as f64).sqrt())
            .mul(self.gamma.val().unsqueeze::<5>())
    }
}

/// `QwenImageRMS_norm` over a 4-D image tensor (`images = true`, used inside
/// the mid-block attention): same math, `gamma` of shape `[C, 1, 1]`.
#[derive(Module, Debug)]
pub struct RmsNormImage<B: Backend> {
    /// Per-channel gain, `[C, 1, 1]`.
    pub gamma: Param<Tensor<B, 3>>,
}

impl<B: Backend> RmsNormImage<B> {
    fn init(dim: usize, device: &B::Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([dim, 1, 1], device)),
        }
    }

    /// `x / max(‖x‖₂, 1e-12) · sqrt(C) · gamma`, channel-wise.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let c = x.dims()[1];
        let norm = x.clone().powi_scalar(2).sum_dim(1).sqrt().clamp_min(1e-12);
        x.div(norm)
            .mul_scalar((c as f64).sqrt())
            .mul(self.gamma.val().unsqueeze::<4>())
    }
}

/// A residual block (`QwenImageResidualBlock`): two norm→SiLU→causal-conv
/// stages plus a (possibly 1×1×1-projected) shortcut.
#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    /// Pre-conv1 RMS norm.
    pub norm1: RmsNormVideo<B>,
    /// First 3×3×3 causal conv (`in → out`).
    pub conv1: CausalConv3d<B>,
    /// Pre-conv2 RMS norm.
    pub norm2: RmsNormVideo<B>,
    /// Second 3×3×3 causal conv (`out → out`).
    pub conv2: CausalConv3d<B>,
    /// 1×1×1 shortcut projection, present only when `in != out`.
    pub conv_shortcut: Option<CausalConv3d<B>>,
}

impl<B: Backend> ResBlock<B> {
    fn init(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        Self {
            norm1: RmsNormVideo::init(in_dim, device),
            conv1: CausalConv3d::init(in_dim, out_dim, [3, 3, 3], [1, 1, 1], [1, 1, 1], device),
            norm2: RmsNormVideo::init(out_dim, device),
            conv2: CausalConv3d::init(out_dim, out_dim, [3, 3, 3], [1, 1, 1], [1, 1, 1], device),
            conv_shortcut: (in_dim != out_dim).then(|| {
                CausalConv3d::init(in_dim, out_dim, [1, 1, 1], [1, 1, 1], [0, 0, 0], device)
            }),
        }
    }

    /// `conv2(silu(norm2(conv1(silu(norm1(x)))))) + shortcut(x)`. The
    /// reference's dropout sits between `norm2` and `conv2` with `p = 0.0` — a
    /// no-op omitted here (the VAE is frozen during LoRA training).
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let h = match &self.conv_shortcut {
            Some(conv) => conv.forward(x.clone()),
            None => x.clone(),
        };
        let y = self.conv1.forward(silu(self.norm1.forward(x)));
        let y = self.conv2.forward(silu(self.norm2.forward(y)));
        y + h
    }
}

/// Single-head spatial self-attention (`QwenImageAttentionBlock`), applied
/// per-frame inside the mid block: 1×1-conv QKV over `[H·W]` positions, plain
/// scaled dot-product (no mask), 1×1-conv output projection, residual add.
#[derive(Module, Debug)]
pub struct SpatialAttention<B: Backend> {
    /// Pre-attention RMS norm (the `images = true` variant).
    pub norm: RmsNormImage<B>,
    /// Fused QKV projection, `C → 3C`, 1×1 conv.
    pub to_qkv: Conv2d<B>,
    /// Output projection, `C → C`, 1×1 conv.
    pub proj: Conv2d<B>,
}

impl<B: Backend> SpatialAttention<B> {
    fn init(dim: usize, device: &B::Device) -> Self {
        Self {
            norm: RmsNormImage::init(dim, device),
            to_qkv: Conv2dConfig::new([dim, dim * 3], [1, 1]).init(device),
            proj: Conv2dConfig::new([dim, dim], [1, 1]).init(device),
        }
    }

    /// `[b, c, t, h, w]` → same shape; attention mixes the `h·w` positions of
    /// each frame independently.
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let identity = x.clone();
        let [b, c, t, h, w] = x.dims();
        let x = x.swap_dims(1, 2).reshape([b * t, c, h, w]);
        let x = self.norm.forward(x);

        // QKV: [bt, 3c, h, w] -> [bt, h·w, 3c] -> q, k, v each [bt, h·w, c].
        let qkv = self
            .to_qkv
            .forward(x)
            .reshape([b * t, 3 * c, h * w])
            .swap_dims(1, 2);
        let q = qkv.clone().narrow(2, 0, c);
        let k = qkv.clone().narrow(2, c, c);
        let v = qkv.narrow(2, 2 * c, c);

        // Scaled dot-product over the position axis (single head, no mask).
        let scores = q.matmul(k.swap_dims(1, 2)).div_scalar((c as f64).sqrt());
        let ctx = softmax(scores, 2).matmul(v); // [bt, h·w, c]

        let x = ctx.swap_dims(1, 2).reshape([b * t, c, h, w]);
        let x = self.proj.forward(x);
        x.reshape([b, t, c, h, w]).swap_dims(1, 2) + identity
    }
}

/// Which spatial resampling a [`Resample`] / resampling [`DownLayer`] performs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResampleMode {
    /// Zero-pad right/bottom by 1, stride-2 valid 3×3 conv (`C → C`).
    Down2d,
    /// [`Down2d`](ResampleMode::Down2d) spatially; owns a temporal-stride
    /// `time_conv` that only fires from a second chunk on (never at `T = 1`).
    Down3d,
    /// 2× nearest-neighbor upsample, 3×3 conv halving channels (`C → C/2`).
    Up2d,
    /// [`Up2d`](ResampleMode::Up2d) spatially; owns a frame-doubling
    /// `time_conv` that only fires from a second chunk on (never at `T = 1`).
    Up3d,
}

/// Shared spatial resample math (`QwenImageResample.resample`): applied
/// per-frame via a `[b·t, c, h, w]` reshape, exactly like the reference.
fn spatial_resample<B: Backend>(
    conv: &Conv2d<B>,
    mode: ResampleMode,
    x: Tensor<B, 5>,
) -> Tensor<B, 5> {
    let [b, c, t, h, w] = x.dims();
    let x = x.swap_dims(1, 2).reshape([b * t, c, h, w]);
    let x = match mode {
        ResampleMode::Up2d | ResampleMode::Up3d => {
            // `nearest-exact` at an exact 2× factor selects the same source
            // pixel as plain nearest (floor((i+0.5)/2) == floor(i/2) for
            // integer 2×), so burn's Nearest reproduces the reference.
            let up = interpolate(
                x,
                [2 * h, 2 * w],
                InterpolateOptions::new(InterpolateMode::Nearest),
            );
            conv.forward(up)
        }
        // The reference's nn.ZeroPad2d((0, 1, 0, 1)) — right/bottom only — is
        // folded into the conv's asymmetric Explicit padding (see
        // `resample_conv`), so the stride-2 conv runs directly.
        ResampleMode::Down2d | ResampleMode::Down3d => conv.forward(x),
    };
    let [_, c2, h2, w2] = x.dims();
    x.reshape([b, t, c2, h2, w2]).swap_dims(1, 2)
}

/// Build the spatial conv of a resample block for `mode`.
fn resample_conv<B: Backend>(dim: usize, mode: ResampleMode, device: &B::Device) -> Conv2d<B> {
    match mode {
        ResampleMode::Up2d | ResampleMode::Up3d => Conv2dConfig::new([dim, dim / 2], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device),
        // nn.ZeroPad2d((0, 1, 0, 1)) = right/bottom-only zero padding, as
        // burn's asymmetric (top, left, bottom, right) explicit config.
        ResampleMode::Down2d | ResampleMode::Down3d => Conv2dConfig::new([dim, dim], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
            .init(device),
    }
}

/// Build the `time_conv` of a 3-D resample block for `mode` (loaded for key
/// parity; unused in the `T = 1` forward — see the module docs).
fn resample_time_conv<B: Backend>(
    dim: usize,
    mode: ResampleMode,
    device: &B::Device,
) -> Option<CausalConv3d<B>> {
    match mode {
        ResampleMode::Up3d => Some(CausalConv3d::init(
            dim,
            dim * 2,
            [3, 1, 1],
            [1, 1, 1],
            [1, 0, 0],
            device,
        )),
        ResampleMode::Down3d => Some(CausalConv3d::init(
            dim,
            dim,
            [3, 1, 1],
            [2, 1, 1],
            [0, 0, 0],
            device,
        )),
        _ => None,
    }
}

/// A decoder upsampler (`QwenImageResample`), keyed `upsamplers.0.…` inside an
/// [`UpBlock`]. The `resample` conv is the reference's `Sequential` index 1
/// (see [`QwenVae::key_remap`]).
#[derive(Module, Debug)]
pub struct Resample<B: Backend> {
    /// The spatial conv (after the pad/upsample, which has no parameters).
    pub resample: Conv2d<B>,
    /// Temporal conv of the 3-D modes — loaded, never run at `T = 1`.
    pub time_conv: Option<CausalConv3d<B>>,
    /// Which resampling this block performs.
    #[module(skip)]
    pub mode: ResampleMode,
}

impl<B: Backend> Resample<B> {
    fn init(dim: usize, mode: ResampleMode, device: &B::Device) -> Self {
        Self {
            resample: resample_conv(dim, mode, device),
            time_conv: resample_time_conv(dim, mode, device),
            mode,
        }
    }

    /// Spatial resample; the temporal path is a first-chunk no-op (`T = 1`).
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        spatial_resample(&self.resample, self.mode, x)
    }
}

/// One entry of the encoder's flat, heterogeneous `down_blocks` list: each
/// index is *either* a residual block *or* a resample, sharing one index space
/// (`down_blocks.0` and `down_blocks.2` differ in kind). Mirrored as a union
/// module — res fields populated for a res entry, `resample`/`time_conv` for a
/// resample entry — so every parameter path matches the checkpoint verbatim.
/// Built only through [`DownLayer::res`] / [`DownLayer::downsample`].
#[derive(Module, Debug)]
pub struct DownLayer<B: Backend> {
    /// Res entry: pre-conv1 norm.
    pub norm1: Option<RmsNormVideo<B>>,
    /// Res entry: first conv.
    pub conv1: Option<CausalConv3d<B>>,
    /// Res entry: pre-conv2 norm.
    pub norm2: Option<RmsNormVideo<B>>,
    /// Res entry: second conv.
    pub conv2: Option<CausalConv3d<B>>,
    /// Res entry: shortcut projection when widening.
    pub conv_shortcut: Option<CausalConv3d<B>>,
    /// Resample entry: the spatial conv.
    pub resample: Option<Conv2d<B>>,
    /// Resample entry (3-D): temporal conv — loaded, never run at `T = 1`.
    pub time_conv: Option<CausalConv3d<B>>,
    /// `Some(mode)` for a resample entry, `None` for a res entry.
    #[module(skip)]
    pub mode: Option<ResampleMode>,
}

impl<B: Backend> DownLayer<B> {
    fn res(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        let block = ResBlock::init(in_dim, out_dim, device);
        Self {
            norm1: Some(block.norm1),
            conv1: Some(block.conv1),
            norm2: Some(block.norm2),
            conv2: Some(block.conv2),
            conv_shortcut: block.conv_shortcut,
            resample: None,
            time_conv: None,
            mode: None,
        }
    }

    fn downsample(dim: usize, mode: ResampleMode, device: &B::Device) -> Self {
        Self {
            norm1: None,
            conv1: None,
            norm2: None,
            conv2: None,
            conv_shortcut: None,
            resample: Some(resample_conv(dim, mode, device)),
            time_conv: resample_time_conv(dim, mode, device),
            mode: Some(mode),
        }
    }

    /// Dispatch on the entry kind (see the type docs).
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        match self.mode {
            Some(mode) => spatial_resample(
                self.resample.as_ref().expect("resample entry has its conv"),
                mode,
                x,
            ),
            None => {
                let h = match &self.conv_shortcut {
                    Some(conv) => conv.forward(x.clone()),
                    None => x.clone(),
                };
                let norm1 = self.norm1.as_ref().expect("res entry has norm1");
                let conv1 = self.conv1.as_ref().expect("res entry has conv1");
                let norm2 = self.norm2.as_ref().expect("res entry has norm2");
                let conv2 = self.conv2.as_ref().expect("res entry has conv2");
                let y = conv1.forward(silu(norm1.forward(x)));
                let y = conv2.forward(silu(norm2.forward(y)));
                y + h
            }
        }
    }
}

/// The encoder/decoder mid block (`QwenImageMidBlock`): res → attention → res.
/// The reference hardcodes `num_layers = 1`, i.e. exactly one attention
/// sandwiched by two res blocks — *regardless* of `attn_scales`.
#[derive(Module, Debug)]
pub struct MidBlock<B: Backend> {
    /// The two sandwiching res blocks (`resnets.0`, `resnets.1`).
    pub resnets: Vec<ResBlock<B>>,
    /// The single spatial attention (`attentions.0`).
    pub attentions: Vec<SpatialAttention<B>>,
}

impl<B: Backend> MidBlock<B> {
    fn init(dim: usize, device: &B::Device) -> Self {
        Self {
            resnets: vec![
                ResBlock::init(dim, dim, device),
                ResBlock::init(dim, dim, device),
            ],
            attentions: vec![SpatialAttention::init(dim, device)],
        }
    }

    /// `resnets[0] → attentions[0] → resnets[1]`.
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let x = self.resnets[0].forward(x);
        let x = self.attentions[0].forward(x);
        self.resnets[1].forward(x)
    }
}

/// The encoder trunk (`QwenImageEncoder3d`): `conv_in` → flat `down_blocks`
/// (see [`DownLayer`]) → mid → `norm_out`/SiLU/`conv_out` (emitting `2·z_dim`
/// distribution moments).
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    /// Input conv, `3 → base_dim`.
    pub conv_in: CausalConv3d<B>,
    /// The flat res/resample chain.
    pub down_blocks: Vec<DownLayer<B>>,
    /// Mid block at the deepest width.
    pub mid_block: MidBlock<B>,
    /// Output head norm.
    pub norm_out: RmsNormVideo<B>,
    /// Output head conv, `deepest → 2·z_dim`.
    pub conv_out: CausalConv3d<B>,
}

impl<B: Backend> Encoder<B> {
    fn init(config: &QwenVaeConfig, device: &B::Device) -> Self {
        // dims = [dim * u for u in [1] + dim_mult]
        let dims: Vec<usize> = std::iter::once(1)
            .chain(config.dim_mult.iter().copied())
            .map(|u| config.base_dim * u)
            .collect();

        let mut down_blocks = Vec::new();
        for i in 0..config.dim_mult.len() {
            let (mut in_dim, out_dim) = (dims[i], dims[i + 1]);
            for _ in 0..config.num_res_blocks {
                down_blocks.push(DownLayer::res(in_dim, out_dim, device));
                in_dim = out_dim;
            }
            if i != config.dim_mult.len() - 1 {
                let mode = if config.temperal_downsample[i] {
                    ResampleMode::Down3d
                } else {
                    ResampleMode::Down2d
                };
                down_blocks.push(DownLayer::downsample(out_dim, mode, device));
            }
        }

        let deepest = *dims.last().expect("dim_mult is non-empty");
        Self {
            conv_in: CausalConv3d::init(3, dims[0], [3, 3, 3], [1, 1, 1], [1, 1, 1], device),
            down_blocks,
            mid_block: MidBlock::init(deepest, device),
            norm_out: RmsNormVideo::init(deepest, device),
            conv_out: CausalConv3d::init(
                deepest,
                2 * config.z_dim,
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
                device,
            ),
        }
    }
}

/// One decoder stage (`QwenImageUpBlock`): `num_res_blocks + 1` res blocks and
/// an optional upsampler (a `Vec` so its path is the checkpoint's
/// `upsamplers.0`).
#[derive(Module, Debug)]
pub struct UpBlock<B: Backend> {
    /// The stage's res blocks.
    pub resnets: Vec<ResBlock<B>>,
    /// Zero or one [`Resample`].
    pub upsamplers: Vec<Resample<B>>,
}

impl<B: Backend> UpBlock<B> {
    /// Res chain then (maybe) upsample.
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let mut x = x;
        for resnet in &self.resnets {
            x = resnet.forward(x);
        }
        match self.upsamplers.first() {
            Some(up) => up.forward(x),
            None => x,
        }
    }
}

/// The decoder trunk (`QwenImageDecoder3d`): `conv_in` → mid → up stages →
/// `norm_out`/SiLU/`conv_out` (back to 3 image channels).
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    /// Input conv, `z_dim → deepest`.
    pub conv_in: CausalConv3d<B>,
    /// Mid block at the deepest width.
    pub mid_block: MidBlock<B>,
    /// The upsampling stages.
    pub up_blocks: Vec<UpBlock<B>>,
    /// Output head norm.
    pub norm_out: RmsNormVideo<B>,
    /// Output head conv, `base_dim → 3`.
    pub conv_out: CausalConv3d<B>,
}

impl<B: Backend> Decoder<B> {
    fn init(config: &QwenVaeConfig, device: &B::Device) -> Self {
        // dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        let last = *config.dim_mult.last().expect("dim_mult is non-empty");
        let dims: Vec<usize> = std::iter::once(last)
            .chain(config.dim_mult.iter().rev().copied())
            .map(|u| config.base_dim * u)
            .collect();
        // temperal_upsample = temperal_downsample[::-1]
        let temperal_upsample: Vec<bool> =
            config.temperal_downsample.iter().rev().copied().collect();

        let mut up_blocks = Vec::new();
        let mut out_dim = dims[0];
        for i in 0..config.dim_mult.len() {
            // The previous stage's upsampler halved the channel count.
            let mut in_dim = if i > 0 { dims[i] / 2 } else { dims[i] };
            out_dim = dims[i + 1];

            let mut resnets = Vec::new();
            for _ in 0..config.num_res_blocks + 1 {
                resnets.push(ResBlock::init(in_dim, out_dim, device));
                in_dim = out_dim;
            }
            let upsamplers = if i != config.dim_mult.len() - 1 {
                let mode = if temperal_upsample[i] {
                    ResampleMode::Up3d
                } else {
                    ResampleMode::Up2d
                };
                vec![Resample::init(out_dim, mode, device)]
            } else {
                vec![]
            };
            up_blocks.push(UpBlock {
                resnets,
                upsamplers,
            });
        }

        Self {
            conv_in: CausalConv3d::init(
                config.z_dim,
                dims[0],
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
                device,
            ),
            mid_block: MidBlock::init(dims[0], device),
            up_blocks,
            norm_out: RmsNormVideo::init(out_dim, device),
            conv_out: CausalConv3d::init(out_dim, 3, [3, 3, 3], [1, 1, 1], [1, 1, 1], device),
        }
    }
}

/// Intermediate activations of [`QwenVae::encode_trace`], for parity
/// localization: each stage is asserted against the PyTorch golden so a
/// mismatch pinpoints the faulty stage rather than only the final latent.
pub struct QwenVaeEncodeTrace<B: Backend> {
    /// After `encoder.conv_in`.
    pub after_conv_in: Tensor<B, 5>,
    /// After the last down block (the mid block's input).
    pub after_down: Tensor<B, 5>,
    /// After the mid block.
    pub after_mid: Tensor<B, 5>,
    /// The `2·z_dim`-channel distribution moments (after `quant_conv`).
    pub moments: Tensor<B, 5>,
    /// The deterministic latent (distribution mode = mean), un-normalized.
    pub latent_mode: Tensor<B, 4>,
    /// The normalized latent `(mode - latents_mean) / latents_std` — what
    /// training consumes and M12 caches.
    pub latent: Tensor<B, 4>,
}

/// Intermediate activations of [`QwenVae::decode_trace`].
pub struct QwenVaeDecodeTrace<B: Backend> {
    /// After `decoder.conv_in`.
    pub after_conv_in: Tensor<B, 5>,
    /// After the decoder mid block.
    pub after_mid: Tensor<B, 5>,
    /// The decoded image batch `[b, 3, h, w]`, clamped to `[-1, 1]`.
    pub image: Tensor<B, 4>,
}

/// The full autoencoder (`AutoencoderKLQwenImage`) with Krea 2's latent
/// (de)normalization folded in.
///
/// Build with [`QwenVae::init`], populate with `burn_store`'s
/// `ModuleSnapshot::load_from` (remapping via [`QwenVae::key_remap`]), then
/// [`encode`](QwenVae::encode) images to normalized latents and
/// [`decode`](QwenVae::decode) normalized latents back to images.
#[derive(Module, Debug)]
pub struct QwenVae<B: Backend> {
    /// The encoder trunk.
    pub encoder: Encoder<B>,
    /// 1×1×1 conv over the distribution moments (`2·z_dim → 2·z_dim`).
    pub quant_conv: CausalConv3d<B>,
    /// 1×1×1 conv over the latent before decoding (`z_dim → z_dim`).
    pub post_quant_conv: CausalConv3d<B>,
    /// The decoder trunk.
    pub decoder: Decoder<B>,
    /// The architecture — drives the forward, not a loadable parameter.
    #[module(skip)]
    pub config: QwenVaeConfig,
}

impl<B: Backend> QwenVae<B> {
    /// Build a VAE with placeholder (zero) weights of the right shapes, ready
    /// to be overwritten by `load_from`.
    pub fn init(config: QwenVaeConfig, device: &B::Device) -> Self {
        assert_eq!(
            config.temperal_downsample.len(),
            config.dim_mult.len() - 1,
            "temperal_downsample must have one flag per stage transition"
        );
        assert_eq!(config.latents_mean.len(), config.z_dim);
        assert_eq!(config.latents_std.len(), config.z_dim);
        Self {
            encoder: Encoder::init(&config, device),
            quant_conv: CausalConv3d::init(
                2 * config.z_dim,
                2 * config.z_dim,
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],
                device,
            ),
            post_quant_conv: CausalConv3d::init(
                config.z_dim,
                config.z_dim,
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],
                device,
            ),
            decoder: Decoder::init(&config, device),
            config,
        }
    }

    /// The regex → replacement rename pairs mapping checkpoint keys to burn
    /// module paths. The single rename flattens the reference's
    /// `nn.Sequential` index on the resample convs (`resample.1` — index 0 is
    /// the parameter-less pad/upsample). Everything else loads by name with no
    /// transpose. Exposed so the loader and its documentation share one source
    /// of truth.
    pub fn key_remap() -> [(&'static str, &'static str); 1] {
        [(r"resample\.1\.(weight|bias)$", r"resample.${1}")]
    }

    /// Per-channel latent stats as a broadcastable `[1, z_dim, 1, 1]` pair.
    fn latent_stats(&self, device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let z = self.config.z_dim;
        let to_tensor = |v: &[f64]| {
            let data: Vec<f32> = v.iter().map(|&x| x as f32).collect();
            Tensor::<B, 1>::from_data(TensorData::new(data, [z]), device).reshape([1, z, 1, 1])
        };
        (
            to_tensor(&self.config.latents_mean),
            to_tensor(&self.config.latents_std),
        )
    }

    /// Encode an image batch `[b, 3, h, w]` (values in `[-1, 1]`, `h`/`w`
    /// divisible by [`spatial_compression`](QwenVaeConfig::spatial_compression))
    /// into the **normalized** deterministic latent `[b, z_dim, h/f, w/f]`.
    pub fn encode(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        self.encode_trace(images).latent
    }

    /// [`encode`](QwenVae::encode), capturing localizing intermediates.
    pub fn encode_trace(&self, images: Tensor<B, 4>) -> QwenVaeEncodeTrace<B> {
        let [_b, c, h, w] = images.dims();
        let f = self.config.spatial_compression();
        assert_eq!(c, 3, "expected RGB input, got {c} channels");
        assert!(
            h % f == 0 && w % f == 0,
            "input {h}x{w} not divisible by the spatial compression {f}"
        );
        let device = images.device();

        // The video model sees a single frame: [b, 3, 1, h, w].
        let x = images.unsqueeze_dim::<5>(2);
        let after_conv_in = self.encoder.conv_in.forward(x);

        let mut y = after_conv_in.clone();
        for layer in &self.encoder.down_blocks {
            y = layer.forward(y);
        }
        let after_down = y.clone();

        let after_mid = self.encoder.mid_block.forward(y);
        let head = self
            .encoder
            .conv_out
            .forward(silu(self.encoder.norm_out.forward(after_mid.clone())));
        let moments = self.quant_conv.forward(head);

        // DiagonalGaussianDistribution.mode(): the first z_dim channels (the
        // mean); the trailing z_dim are the log-variance, unused when
        // encoding deterministically.
        let z = self.config.z_dim;
        let latent_mode: Tensor<B, 4> = moments.clone().narrow(1, 0, z).squeeze_dim(2);

        let (mean, std) = self.latent_stats(&device);
        let latent = (latent_mode.clone() - mean) / std;

        QwenVaeEncodeTrace {
            after_conv_in,
            after_down,
            after_mid,
            moments,
            latent_mode,
            latent,
        }
    }

    /// Decode a **normalized** latent batch `[b, z_dim, h', w']` back to an
    /// image batch `[b, 3, 8h', 8w']` clamped to `[-1, 1]`.
    pub fn decode(&self, latents: Tensor<B, 4>) -> Tensor<B, 4> {
        self.decode_trace(latents).image
    }

    /// [`decode`](QwenVae::decode), capturing localizing intermediates.
    pub fn decode_trace(&self, latents: Tensor<B, 4>) -> QwenVaeDecodeTrace<B> {
        let device = latents.device();
        let (mean, std) = self.latent_stats(&device);
        let z = latents * std + mean;

        let x = z.unsqueeze_dim::<5>(2);
        let x = self.post_quant_conv.forward(x);
        let after_conv_in = self.decoder.conv_in.forward(x);
        let after_mid = self.decoder.mid_block.forward(after_conv_in.clone());

        let mut y = after_mid.clone();
        for up in &self.decoder.up_blocks {
            y = up.forward(y);
        }
        let y = self
            .decoder
            .conv_out
            .forward(silu(self.decoder.norm_out.forward(y)));

        // The reference clamps inside `_decode`.
        let image: Tensor<B, 4> = y.clamp(-1.0, 1.0).squeeze_dim(2);

        QwenVaeDecodeTrace {
            after_conv_in,
            after_mid,
            image,
        }
    }
}
