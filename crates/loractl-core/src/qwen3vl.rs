//! The Qwen3-VL text-conditioning encoder — Krea 2's caption encoder (M10, #21).
//!
//! Krea 2 does not condition on FLUX's T5+CLIP stack; its conditioner
//! (`krea-ai/krea-2`'s `encoder.py`) is a **frozen Qwen3-VL** multimodal LLM
//! run **text-only**: tokenize a templated caption, forward the decoder trunk,
//! and hand the MMDiT a **stack of 12 intermediate hidden states**
//! (`select_layers = 2, 5, 8, …, 35`) sliced past the 34-token template
//! prefix. There is *no* fusion stage in the encoder — the MMDiT itself
//! consumes the 12-layer stack (its config has `txtlayers = 12`,
//! `txtdim = 2560`).
//!
//! ## What the source pins down (all verified against
//! `transformers/models/qwen3_vl/modeling_qwen3_vl.py` and the checkpoint)
//!
//! - **Text-only M-RoPE collapses to plain RoPE.** With 2-D (text-only)
//!   position ids, all three M-RoPE streams share identical positions, so the
//!   interleaved section overwrites are no-ops: the rotation is standard
//!   **half-split** RoPE (`emb = cat(freqs, freqs)` + `rotate_half`) at
//!   `rope_theta = 5e6`. This is HF's convention, *not* burn's interleaved
//!   `RotaryEncoding` — hence the hand-rolled [`rope_tables`]/[`rotate_half`].
//! - **QK-Norm**: per-head RMSNorm over `head_dim`, applied to the projected
//!   q/k **before** the head transpose and **before** RoPE.
//! - **GQA**: 32 query heads over 8 KV heads (4B config); KV heads repeat in
//!   HF's `repeat_kv` grouping order (query head `h` reads KV head
//!   `h / groups`).
//! - **Pre-norm residuals**, SwiGLU MLP (`down(silu(gate(x)) * up(x))`),
//!   RMSNorm with `weight`-named parameters (T5-style), no biases anywhere.
//! - **`hidden_states[i]` is the output of decoder layer `i` (1-based;
//!   `[0]` is the embedding)**, so the largest selected layer (35) needs only
//!   decoder layers `0..35` — the 36th layer and the final `norm` are dead
//!   for conditioning and are neither built nor loaded.
//!
//! ## Weight loading
//!
//! The module tree mirrors Krea-2-Raw's shipped `text_encoder/` checkpoint (a
//! bare `Qwen3VLModel`: keys `language_model.*` + `visual.*`). The vision
//! tower is never used for conditioning, so loading applies a
//! `^language_model\.` filter that **drops every `visual.*` tensor**, and the
//! projections are genuine PyTorch `nn.Linear`s (`[out, in]`), so — unlike
//! GPT-2's `Conv1D`s and the M9 VAE's convs — this load *does* attach
//! burn-store's `PyTorchToBurnAdapter` transpose. The custom [`RmsNorm`]
//! holds its parameter as `weight`, matching the checkpoint with no rename.
//!
//! Like the rest of `loractl-core`, this module emits no output and imports
//! no CLI. Parity: `tests/qwen3vl_parity.rs` (tiny fixture, offline,
//! including a right-padded row that pins key-padding masking) and
//! `tests/qwen3vl_real.rs` (opt-in, the real Krea-2-Raw text encoder).

use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

/// Additive-mask sentinel: large-negative but far from `f32::MIN`, so the
/// causal and padding masks can sum without overflowing to `-inf`.
const MASK_NEG: f32 = -1.0e30;

/// Static architecture of a Qwen3-VL *text* trunk used as a conditioner.
///
/// Field names follow the HF `text_config`. Held on [`Qwen3VlEncoder`] as a
/// non-parameter (`#[module(skip)]`) field.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3VlConfig {
    /// Hidden width (`hidden_size`): 2560 for the 4B model.
    pub hidden_size: usize,
    /// Query-head count (`num_attention_heads`).
    pub num_heads: usize,
    /// KV-head count (`num_key_value_heads`); GQA groups = heads / kv_heads.
    pub num_kv_heads: usize,
    /// Per-head width (`head_dim` — explicit in the config, NOT
    /// `hidden_size / num_heads`, which differs for the 4B model).
    pub head_dim: usize,
    /// SwiGLU inner width (`intermediate_size`).
    pub intermediate_size: usize,
    /// Vocabulary size (rows of `embed_tokens`).
    pub vocab_size: usize,
    /// RMSNorm epsilon (`rms_norm_eps`).
    pub rms_norm_eps: f64,
    /// RoPE base (`rope_theta`).
    pub rope_theta: f64,
    /// Which `hidden_states` indices feed the conditioning stack, 1-based
    /// (index `i` = output of decoder layer `i`; `0` would be the embedding).
    /// Sorted ascending. The trunk builds `max(select_layers)` layers — later
    /// checkpoint layers are dead for conditioning and never loaded.
    pub select_layers: Vec<usize>,
}

impl Qwen3VlConfig {
    /// The tiny fixture config. Must match `reference/qwen3vl_reference.py`'s
    /// `TINY_TEXT` + `TINY_SELECT_LAYERS` exactly. Deliberately
    /// non-degenerate, like the real 4B config: `head_dim ≠ hidden / heads`,
    /// `heads ≠ kv_heads ≠ GQA groups`, and `heads · head_dim ≠ hidden`
    /// (non-square projections), so conflating any of them fails the
    /// always-run parity test rather than only the opt-in real one.
    pub fn tiny() -> Self {
        Self {
            hidden_size: 32,
            num_heads: 6,
            num_kv_heads: 2,
            head_dim: 6,
            intermediate_size: 64,
            vocab_size: 93,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            select_layers: vec![1, 3],
        }
    }

    /// The real conditioner: Qwen3-VL-4B's text trunk with Krea 2's 12
    /// aggregation layers (`krea-ai/krea-2` `encoder.py`).
    pub fn krea2_4b() -> Self {
        Self {
            hidden_size: 2560,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 9728,
            vocab_size: 151_936,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            select_layers: vec![2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
        }
    }

    /// Decoder layers the trunk must build/load: `max(select_layers)`.
    pub fn num_layers(&self) -> usize {
        *self
            .select_layers
            .last()
            .expect("select_layers is non-empty")
    }
}

/// T5-style RMSNorm (`Qwen3VLTextRMSNorm`): `x / sqrt(mean(x²) + eps) * weight`.
/// The parameter is named `weight`, matching the checkpoint key with no
/// rename (and, being a custom container, the PyTorch adapter leaves it
/// untouched).
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Per-channel gain over the last dimension.
    pub weight: Param<Tensor<B, 1>>,
    /// Variance epsilon.
    #[module(skip)]
    pub eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    fn init(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
            eps,
        }
    }

    /// Normalize the last dimension of a rank-`D` tensor.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let variance = x.clone().powi_scalar(2).mean_dim(D - 1);
        let x = x / (variance + self.eps).sqrt();
        x * self.weight.val().unsqueeze::<D>()
    }
}

/// One attention block (`Qwen3VLTextAttention`): GQA projections with
/// per-head q/k RMSNorm and half-split RoPE.
#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    /// Query projection `hidden → heads · head_dim` (no bias).
    pub q_proj: Linear<B>,
    /// Key projection `hidden → kv_heads · head_dim`.
    pub k_proj: Linear<B>,
    /// Value projection `hidden → kv_heads · head_dim`.
    pub v_proj: Linear<B>,
    /// Output projection `heads · head_dim → hidden`.
    pub o_proj: Linear<B>,
    /// Per-head query norm (over `head_dim`), applied before RoPE.
    pub q_norm: RmsNorm<B>,
    /// Per-head key norm (over `head_dim`), applied before RoPE.
    pub k_norm: RmsNorm<B>,
}

impl<B: Backend> SelfAttention<B> {
    fn init(config: &Qwen3VlConfig, device: &B::Device) -> Self {
        let lin = |d_in: usize, d_out: usize| {
            LinearConfig::new(d_in, d_out).with_bias(false).init(device)
        };
        let (h, hd) = (config.hidden_size, config.head_dim);
        Self {
            q_proj: lin(h, config.num_heads * hd),
            k_proj: lin(h, config.num_kv_heads * hd),
            v_proj: lin(h, config.num_kv_heads * hd),
            o_proj: lin(config.num_heads * hd, h),
            q_norm: RmsNorm::init(hd, config.rms_norm_eps, device),
            k_norm: RmsNorm::init(hd, config.rms_norm_eps, device),
        }
    }

    /// `x` is `[b, s, hidden]`; `rope` the `(cos, sin)` tables `[s, head_dim]`;
    /// `mask` the additive attention mask `[b, 1, s, s]`.
    fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &(Tensor<B, 2>, Tensor<B, 2>),
        mask: Tensor<B, 4>,
        config: &Qwen3VlConfig,
    ) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let (heads, kv, hd) = (config.num_heads, config.num_kv_heads, config.head_dim);

        // Project, norm per head (before the transpose and before RoPE —
        // exactly the reference's order), then split heads.
        let split = |t: Tensor<B, 3>, n: usize| t.reshape([b, s, n, hd]).swap_dims(1, 2);
        let q = split(
            self.q_norm
                .forward(self.q_proj.forward(x.clone()).reshape([b, s, heads, hd]))
                .reshape([b, s, heads * hd]),
            heads,
        );
        let k = split(
            self.k_norm
                .forward(self.k_proj.forward(x.clone()).reshape([b, s, kv, hd]))
                .reshape([b, s, kv * hd]),
            kv,
        );
        let v = split(self.v_proj.forward(x), kv);

        // Half-split RoPE on q/k.
        let q = apply_rope(q, rope);
        let k = apply_rope(k, rope);

        // GQA: repeat each KV head over its query group (HF `repeat_kv`
        // order: query head h reads KV head h / groups).
        let groups = heads / kv;
        let expand_kv = |t: Tensor<B, 4>| {
            t.reshape([b, kv, 1, s, hd])
                .expand([b, kv, groups, s, hd])
                .reshape([b, heads, s, hd])
        };
        let k = expand_kv(k);
        let v = expand_kv(v);

        // Scaled dot-product with the combined (causal + padding) mask.
        let scale = (hd as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale) + mask;
        let ctx = softmax(scores, 3).matmul(v); // [b, heads, s, hd]

        let merged = ctx.swap_dims(1, 2).reshape([b, s, heads * hd]);
        self.o_proj.forward(merged)
    }
}

/// The SwiGLU feed-forward (`Qwen3VLTextMLP`): `down(silu(gate(x)) * up(x))`.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    /// Gate projection `hidden → intermediate`.
    pub gate_proj: Linear<B>,
    /// Up projection `hidden → intermediate`.
    pub up_proj: Linear<B>,
    /// Down projection `intermediate → hidden`.
    pub down_proj: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn init(config: &Qwen3VlConfig, device: &B::Device) -> Self {
        let lin = |d_in: usize, d_out: usize| {
            LinearConfig::new(d_in, d_out).with_bias(false).init(device)
        };
        Self {
            gate_proj: lin(config.hidden_size, config.intermediate_size),
            up_proj: lin(config.hidden_size, config.intermediate_size),
            down_proj: lin(config.intermediate_size, config.hidden_size),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gated = silu(self.gate_proj.forward(x.clone())) * self.up_proj.forward(x);
        self.down_proj.forward(gated)
    }
}

/// One pre-norm decoder layer (`Qwen3VLTextDecoderLayer`):
/// `x + attn(ln1(x))` then `x + mlp(ln2(x))`.
#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    /// Pre-attention RMSNorm.
    pub input_layernorm: RmsNorm<B>,
    /// The GQA attention block.
    pub self_attn: SelfAttention<B>,
    /// Pre-MLP RMSNorm.
    pub post_attention_layernorm: RmsNorm<B>,
    /// The SwiGLU feed-forward.
    pub mlp: Mlp<B>,
}

impl<B: Backend> DecoderLayer<B> {
    fn init(config: &Qwen3VlConfig, device: &B::Device) -> Self {
        Self {
            input_layernorm: RmsNorm::init(config.hidden_size, config.rms_norm_eps, device),
            self_attn: SelfAttention::init(config, device),
            post_attention_layernorm: RmsNorm::init(
                config.hidden_size,
                config.rms_norm_eps,
                device,
            ),
            mlp: Mlp::init(config, device),
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &(Tensor<B, 2>, Tensor<B, 2>),
        mask: Tensor<B, 4>,
        config: &Qwen3VlConfig,
    ) -> Tensor<B, 3> {
        let attn =
            self.self_attn
                .forward(self.input_layernorm.forward(x.clone()), rope, mask, config);
        let x = x + attn;
        let mlp = self
            .mlp
            .forward(self.post_attention_layernorm.forward(x.clone()));
        x + mlp
    }
}

/// The text trunk, keyed `language_model.*` like the checkpoint. Only the
/// first `max(select_layers)` decoder layers exist — later layers and the
/// final norm are dead for conditioning (see the module docs).
#[derive(Module, Debug)]
pub struct LanguageModel<B: Backend> {
    /// Token embedding `[vocab, hidden]` (loads verbatim — no transpose).
    pub embed_tokens: Embedding<B>,
    /// Decoder layers `0 .. max(select_layers)`.
    pub layers: Vec<DecoderLayer<B>>,
}

/// Intermediate activations captured by
/// [`forward_trace`](Qwen3VlEncoder::forward_trace), for parity localization.
pub struct Qwen3VlTrace<B: Backend> {
    /// The embedding output (`hidden_states[0]`).
    pub after_embed: Tensor<B, 3>,
    /// `hidden_states[select_layers[0]]` — the first selected state.
    pub first_select: Tensor<B, 3>,
    /// `hidden_states[select_layers[last]]` — the last selected state.
    pub last_select: Tensor<B, 3>,
    /// The conditioning stack `[b, s - prefix, n_select, hidden]`.
    pub conditioning: Tensor<B, 4>,
}

/// A hand-built Qwen3-VL **text-only** encoder that loads the real
/// `text_encoder` checkpoint and produces Krea 2's conditioning stack.
///
/// Build with [`Qwen3VlEncoder::init`], populate with `burn_store`'s
/// `ModuleSnapshot::load_from` using [`load_filter`](Self::load_filter) and
/// the `PyTorchToBurnAdapter`, then call
/// [`forward_conditioning`](Self::forward_conditioning).
#[derive(Module, Debug)]
pub struct Qwen3VlEncoder<B: Backend> {
    /// The text trunk (`language_model.*`).
    pub language_model: LanguageModel<B>,
    /// The architecture — drives the forward, not a loadable parameter.
    #[module(skip)]
    pub config: Qwen3VlConfig,
}

impl<B: Backend> Qwen3VlEncoder<B> {
    /// Build an encoder with placeholder weights of the right shapes, ready
    /// to be overwritten by `load_from`.
    pub fn init(config: Qwen3VlConfig, device: &B::Device) -> Self {
        assert!(
            config.select_layers.windows(2).all(|w| w[0] < w[1]),
            "select_layers must be strictly ascending"
        );
        assert!(
            *config.select_layers.first().expect("non-empty") >= 1,
            "select_layers are 1-based (0 is the embedding)"
        );
        let language_model = LanguageModel {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers: (0..config.num_layers())
                .map(|_| DecoderLayer::init(&config, device))
                .collect(),
        };
        Self {
            language_model,
            config,
        }
    }

    /// The load-time key filter: keep only the text trunk. Drops every
    /// `visual.*` tensor (the vision tower is never used text-only) — and the
    /// checkpoint's post-`max(select_layers)` decoder layers plus its final
    /// `norm` simply have no matching parameters (they land in `unused`).
    pub fn load_filter() -> &'static str {
        r"^language_model\."
    }

    /// Forward the trunk on token ids `[b, s]` with a 0/1 attention mask
    /// `[b, s]`, collecting the selected hidden states into the conditioning
    /// stack `[b, s - prefix_idx, n_select, hidden]` (the reference slices
    /// the templated prompt's first `prefix_idx` positions off).
    pub fn forward_conditioning(
        &self,
        ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2, Int>,
        prefix_idx: usize,
    ) -> Tensor<B, 4> {
        self.forward_trace(ids, attention_mask, prefix_idx)
            .conditioning
    }

    /// [`forward_conditioning`](Self::forward_conditioning) with localizing
    /// intermediates for the parity harness.
    pub fn forward_trace(
        &self,
        ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2, Int>,
        prefix_idx: usize,
    ) -> Qwen3VlTrace<B> {
        let [b, s] = ids.dims();
        let device = ids.device();
        let config = &self.config;

        // Combined additive mask: causal + key-padding, [b, 1, s, s].
        let mask = attention_masks::<B>(attention_mask, &device);

        // Position ids are a plain arange (text-only M-RoPE collapse).
        let rope = rope_tables::<B>(s, config.head_dim, config.rope_theta, &device);

        let mut h = self.language_model.embed_tokens.forward(ids);
        let after_embed = h.clone();

        let mut selected: Vec<Tensor<B, 3>> = Vec::with_capacity(config.select_layers.len());
        for (i, layer) in self.language_model.layers.iter().enumerate() {
            h = layer.forward(h, &rope, mask.clone(), config);
            // hidden_states[i + 1] is this layer's output.
            if config.select_layers.contains(&(i + 1)) {
                selected.push(h.clone());
            }
        }

        let first_select = selected.first().expect("at least one select layer").clone();
        let last_select = selected.last().expect("at least one select layer").clone();

        // Stack on a new dim 2: [b, s, n_select, hidden], then slice off the
        // template prefix.
        let n = selected.len();
        let hidden = config.hidden_size;
        let stacked = Tensor::cat(
            selected
                .into_iter()
                .map(|t| t.reshape([b, s, 1, hidden]))
                .collect::<Vec<_>>(),
            2,
        );
        let conditioning = stacked.narrow(1, prefix_idx, s - prefix_idx);
        debug_assert_eq!(conditioning.dims()[2], n);

        Qwen3VlTrace {
            after_embed,
            first_select,
            last_select,
            conditioning,
        }
    }
}

/// Build the combined additive attention mask `[b, 1, s, s]`: large-negative
/// above the causal diagonal and at every key position whose `attention_mask`
/// is 0 (right-padding), `0.0` elsewhere. The two contributions use a finite
/// sentinel ([`MASK_NEG`]) so their sum cannot overflow to `-inf`.
fn attention_masks<B: Backend>(
    attention_mask: Tensor<B, 2, Int>,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [b, s] = attention_mask.dims();

    // Causal part [1, 1, s, s].
    let mut causal = vec![0.0f32; s * s];
    for i in 0..s {
        for j in (i + 1)..s {
            causal[i * s + j] = MASK_NEG;
        }
    }
    let causal =
        Tensor::<B, 2>::from_data(TensorData::new(causal, [s, s]), device).reshape([1, 1, s, s]);

    // Key-padding part [b, 1, 1, s]: (1 - mask) * MASK_NEG.
    let pad = attention_mask
        .float()
        .neg()
        .add_scalar(1.0)
        .mul_scalar(MASK_NEG)
        .reshape([b, 1, 1, s]);

    causal + pad
}

/// Half-split RoPE tables for positions `0..seq`: `(cos, sin)` each
/// `[seq, head_dim]`, with `emb = cat(freqs, freqs)` — HF's convention (the
/// reference's `apply_rotary_pos_emb`), *not* burn's interleaved layout.
fn rope_tables<B: Backend>(
    seq: usize,
    head_dim: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let half = head_dim / 2;
    let mut freqs = Vec::with_capacity(seq * half);
    for p in 0..seq {
        for i in 0..half {
            let inv = theta.powf(-2.0 * i as f64 / head_dim as f64);
            freqs.push((p as f64 * inv) as f32);
        }
    }
    let f = Tensor::<B, 2>::from_data(TensorData::new(freqs, [seq, half]), device);
    let emb = Tensor::cat(vec![f.clone(), f], 1); // [seq, head_dim]
    (emb.clone().cos(), emb.sin())
}

/// Apply half-split RoPE to `[b, heads, s, head_dim]`:
/// `x * cos + rotate_half(x) * sin`, `rotate_half(x) = cat(-x₂, x₁)`.
fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,
    (cos, sin): &(Tensor<B, 2>, Tensor<B, 2>),
) -> Tensor<B, 4> {
    let [_, _, s, hd] = x.dims();
    let half = hd / 2;
    let cos = cos.clone().reshape([1, 1, s, hd]);
    let sin = sin.clone().reshape([1, 1, s, hd]);
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.clone().narrow(3, half, half);
    let rotated = Tensor::cat(vec![x2.neg(), x1], 3);
    x * cos + rotated * sin
}

/// Krea 2's caption-conditioning front door: the chat template, the
/// tokenizer, and the [`Qwen3VlEncoder`] composed into
/// "captions in → conditioning stack out" (`encoder.py`'s
/// `Qwen3VLConditioner`).
///
/// Not a burn `Module`: the tokenizer is runtime state, not checkpoint
/// state. The encoder inside is the loadable part.
pub struct Qwen3VlConditioner<B: Backend> {
    /// The loaded text trunk.
    pub encoder: Qwen3VlEncoder<B>,
    tokenizer: tokenizers::Tokenizer,
    max_length: usize,
}

/// `encoder.py`'s prompt template, verbatim.
pub const PROMPT_PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
/// The assistant-turn suffix appended (as separately-tokenized ids) *after*
/// the right-padded body — pad tokens sit between caption and suffix.
pub const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
/// Token length of [`PROMPT_PREFIX`]; the conditioning slice starts here.
pub const PROMPT_PREFIX_LEN: usize = 34;
/// Token length of [`PROMPT_SUFFIX`] (folded into the body's pad budget).
pub const PROMPT_SUFFIX_LEN: usize = 5;
/// The pad token (Qwen vocabulary).
pub const PAD_TOKEN: &str = "<|endoftext|>";

impl<B: Backend> Qwen3VlConditioner<B> {
    /// Wrap a loaded encoder with the tokenizer at `tokenizer_json`
    /// (a HF `tokenizer.json`, e.g. Krea-2-Raw's `tokenizer/tokenizer.json`).
    /// `max_length` is the caption budget (512 in `encoder.py`).
    pub fn new(
        encoder: Qwen3VlEncoder<B>,
        tokenizer_json: &std::path::Path,
        max_length: usize,
    ) -> anyhow::Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_json)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
        Ok(Self {
            encoder,
            tokenizer,
            max_length,
        })
    }

    /// Tokenize captions through the template exactly like the reference:
    /// `prefix + caption` truncated/right-padded to
    /// `max_length + PREFIX_LEN - SUFFIX_LEN`, then the suffix ids
    /// concatenated *after* the padding. Returns `(ids, mask)` as row-major
    /// `[b, s]` vectors.
    pub fn tokenize(&self, captions: &[&str]) -> anyhow::Result<(Vec<i64>, Vec<i64>, [usize; 2])> {
        let body_len = self.max_length + PROMPT_PREFIX_LEN - PROMPT_SUFFIX_LEN;
        let pad_id = self
            .tokenizer
            .token_to_id(PAD_TOKEN)
            .ok_or_else(|| anyhow::anyhow!("tokenizer lacks the {PAD_TOKEN} pad token"))?
            as i64;

        let suffix = self
            .tokenizer
            .encode(PROMPT_SUFFIX, false)
            .map_err(|e| anyhow::anyhow!("tokenizing suffix: {e}"))?;
        let suffix_ids: Vec<i64> = suffix.get_ids().iter().map(|&i| i as i64).collect();

        let s = body_len + suffix_ids.len();
        let mut ids = Vec::with_capacity(captions.len() * s);
        let mut mask = Vec::with_capacity(captions.len() * s);
        for caption in captions {
            let text = format!("{PROMPT_PREFIX}{caption}");
            let enc = self
                .tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| anyhow::anyhow!("tokenizing caption: {e}"))?;
            let mut row: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
            row.truncate(body_len);
            let live = row.len();
            row.resize(body_len, pad_id);
            ids.extend_from_slice(&row);
            ids.extend_from_slice(&suffix_ids);
            mask.extend(std::iter::repeat_n(1i64, live));
            mask.extend(std::iter::repeat_n(0i64, body_len - live));
            mask.extend(std::iter::repeat_n(1i64, suffix_ids.len()));
        }
        Ok((ids, mask, [captions.len(), s]))
    }

    /// Captions → the conditioning stack
    /// `[b, s - PREFIX_LEN, n_select, hidden]` plus the matching sliced 0/1
    /// mask `[b, s - PREFIX_LEN]` the MMDiT consumes.
    pub fn encode_captions(
        &self,
        captions: &[&str],
        device: &B::Device,
    ) -> anyhow::Result<(Tensor<B, 4>, Tensor<B, 2, Int>)> {
        let (ids, mask, [b, s]) = self.tokenize(captions)?;
        let ids = Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [b, s]), device);
        let mask = Tensor::<B, 2, Int>::from_data(TensorData::new(mask, [b, s]), device);
        let conditioning = self
            .encoder
            .forward_conditioning(ids, mask.clone(), PROMPT_PREFIX_LEN);
        let mask_sliced = mask.narrow(1, PROMPT_PREFIX_LEN, s - PROMPT_PREFIX_LEN);
        Ok((conditioning, mask_sliced))
    }
}
