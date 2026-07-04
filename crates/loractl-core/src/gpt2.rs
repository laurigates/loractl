//! A hand-built GPT-2 transformer that loads real HF safetensors weights.
//!
//! This is milestone 3 (#2): the first *real* base model. [`Gpt2`] is a
//! pre-LayerNorm GPT-2 (the [openai-community/gpt2] architecture) written module
//! by module so its parameter paths mirror Hugging Face's state-dict keys, then
//! populated from a checked-in `model.safetensors` via [`burn_store`]. The point
//! is *forward-pass parity*: given identical weights, this burn forward and the
//! PyTorch reference must agree to f32 rounding. The parity harness
//! (`tests/gpt2_parity.rs`) proves that offline against a tiny real GPT-2 whose
//! weights and golden activations are checked in.
//!
//! ## Weight loading — no transpose, LayerNorm key rename only
//!
//! GPT-2's linear layers are HF `Conv1D`s, whose weight is stored as
//! `[in, out]` — **exactly** burn's [`Linear`] layout (`[d_input, d_output]`),
//! and the token/position embeddings are already `[n_vocab, d_model]` /
//! `[n_positions, d_model]` = burn's [`Embedding`] layout. So the default
//! identity adapter loads every projection and embedding **verbatim, without
//! transposing** (attaching burn-store's `PyTorchToBurnAdapter` would wrongly
//! transpose the `Conv1D`-derived weights). The *only* renames needed are for
//! LayerNorm, whose parameters HF calls `weight`/`bias` but burn calls
//! `gamma`/`beta` — handled by a [`KeyRemapper`]. This no-transpose story is
//! GPT-2-specific; a future `nn.Linear`-based target (e.g. SmolLM) *would* need
//! the transpose. See `docs/adrs/0001-first-real-target-model.md`.
//!
//! ## The weight tie is explicit in the forward
//!
//! GPT-2 ties its output head to the token embedding, so the safetensors has
//! **no `lm_head` key**. Rather than materialize a head parameter, the forward
//! computes `logits = h · wteᵀ` directly. `allow_partial(true)` at load time
//! merely tolerates the absence of a head param (and any HF causal-mask buffers)
//! — it does not fabricate one.
//!
//! ## Invariant
//!
//! Like the rest of `loractl-core`, this module emits no output and imports no
//! CLI: it is pure model code. It depends only on `burn` / `burn_store`.
//!
//! [openai-community/gpt2]: https://huggingface.co/openai-community/gpt2

use crate::lora::LoraLinear;
use burn::module::Module;
use burn::nn::{
    Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

/// Static architecture of a GPT-2 variant.
///
/// Parameterizes the whole module tree so the same code serves the tiny
/// checked-in fixture and a real pretrained GPT-2. Held on [`Gpt2`] as a
/// non-parameter (`#[module(skip)]`) field — it drives the forward but is not a
/// tensor to load.
#[derive(Debug, Clone, PartialEq)]
pub struct Gpt2Config {
    /// Vocabulary size (rows of the token embedding / columns of the logits).
    pub vocab_size: usize,
    /// Maximum sequence length (rows of the position embedding).
    pub n_positions: usize,
    /// Hidden width `d_model`.
    pub n_embd: usize,
    /// Number of transformer blocks.
    pub n_layer: usize,
    /// Number of attention heads. `head_dim = n_embd / n_head`.
    pub n_head: usize,
    /// Inner (feed-forward) width; GPT-2 uses `4 * n_embd`.
    pub n_inner: usize,
    /// LayerNorm epsilon.
    pub layer_norm_epsilon: f64,
}

impl Gpt2Config {
    /// The tiny fixture config: a real GPT-2 architecture at minimal dims, used
    /// by the always-run offline parity test. Must match the Python reference
    /// (`reference/gpt2_tiny_reference.py`) exactly.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 61,
            n_positions: 16,
            n_embd: 32,
            n_layer: 2,
            n_head: 2,
            n_inner: 64,
            layer_norm_epsilon: 1e-5,
        }
    }

    /// GPT-2 small (`openai-community/gpt2`), the M3 real target: 124M params,
    /// 12 layers, 12 heads, `d_model = 768`. Used by the opt-in real-weights
    /// test.
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_inner: 3072,
            layer_norm_epsilon: 1e-5,
        }
    }
}

/// GPT-2's per-block multi-head self-attention: a fused QKV projection
/// (`c_attn`) and an output projection (`c_proj`).
///
/// Both are HF `Conv1D`s loaded verbatim into burn [`Linear`]s. `c_attn` is
/// `[n_embd, 3·n_embd]` with bias; `c_proj` is `[n_embd, n_embd]` with bias.
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    /// Fused query/key/value projection `n_embd -> 3·n_embd`.
    pub c_attn: Linear<B>,
    /// Output projection `n_embd -> n_embd`.
    pub c_proj: Linear<B>,
}

impl<B: Backend> Attention<B> {
    fn init(config: &Gpt2Config, device: &B::Device) -> Self {
        Self {
            c_attn: LinearConfig::new(config.n_embd, 3 * config.n_embd).init(device),
            c_proj: LinearConfig::new(config.n_embd, config.n_embd).init(device),
        }
    }

    /// Finish attention from an already-computed QKV projection `[b, s, 3e]`:
    /// split heads, scaled dot-product with the additive causal `mask`, merge,
    /// and apply `c_proj`. Shared by [`Attention::forward`] and the LoRA-attach
    /// path so the adapted projection reuses identical math.
    fn attend(&self, qkv: Tensor<B, 3>, mask: Tensor<B, 4>, n_head: usize) -> Tensor<B, 3> {
        let [b, s, e3] = qkv.dims();
        let e = e3 / 3;
        let hd = e / n_head;
        let scale = (hd as f64).sqrt();

        // Split the fused projection into q, k, v each [b, s, e].
        let q = qkv.clone().narrow(2, 0, e);
        let k = qkv.clone().narrow(2, e, e);
        let v = qkv.narrow(2, 2 * e, e);

        // [b, s, e] -> [b, n_head, s, head_dim].
        let split = |t: Tensor<B, 3>| t.reshape([b, s, n_head, hd]).swap_dims(1, 2);
        let q = split(q);
        let k = split(k);
        let v = split(v);

        // Scaled dot-product scores [b, n_head, s, s], causal-masked, softmaxed.
        let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale) + mask;
        let probs = softmax(scores, 3);

        // Context [b, n_head, s, head_dim] -> merged [b, s, e] -> output proj.
        let ctx = probs.matmul(v).swap_dims(1, 2).reshape([b, s, e]);
        self.c_proj.forward(ctx)
    }

    /// Full self-attention on the (already LayerNorm-ed) input `x` `[b, s, e]`.
    fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 4>, n_head: usize) -> Tensor<B, 3> {
        self.attend(self.c_attn.forward(x), mask, n_head)
    }
}

/// GPT-2's per-block feed-forward network: `c_fc` up-projection, `gelu_new`
/// activation, `c_proj` down-projection.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    /// Up-projection `n_embd -> n_inner`.
    pub c_fc: Linear<B>,
    /// Down-projection `n_inner -> n_embd`.
    pub c_proj: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn init(config: &Gpt2Config, device: &B::Device) -> Self {
        Self {
            c_fc: LinearConfig::new(config.n_embd, config.n_inner).init(device),
            c_proj: LinearConfig::new(config.n_inner, config.n_embd).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // `gelu_new` = tanh-approximate GELU = burn's `Gelu::new_approximate`.
        let h = Gelu::new_approximate().forward(self.c_fc.forward(x));
        self.c_proj.forward(h)
    }
}

/// One pre-LayerNorm GPT-2 transformer block: `h + attn(ln_1(h))` then
/// `h + mlp(ln_2(h))`.
#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    /// Pre-attention LayerNorm.
    pub ln_1: LayerNorm<B>,
    /// Multi-head self-attention.
    pub attn: Attention<B>,
    /// Pre-MLP LayerNorm.
    pub ln_2: LayerNorm<B>,
    /// Feed-forward network.
    pub mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    fn init(config: &Gpt2Config, device: &B::Device) -> Self {
        let ln = || {
            LayerNormConfig::new(config.n_embd)
                .with_epsilon(config.layer_norm_epsilon)
                .init(device)
        };
        Self {
            ln_1: ln(),
            attn: Attention::init(config, device),
            ln_2: ln(),
            mlp: Mlp::init(config, device),
        }
    }

    /// Pre-LN residual block. `qkv_override`, when `Some`, replaces this block's
    /// `c_attn` projection with a LoRA-adapted one (the M3 LoRA-attach path);
    /// `None` uses the block's own loaded `c_attn`.
    fn forward(
        &self,
        h: Tensor<B, 3>,
        mask: Tensor<B, 4>,
        n_head: usize,
        qkv_override: Option<&LoraLinear<B>>,
    ) -> Tensor<B, 3> {
        let normed = self.ln_1.forward(h.clone());
        let attn_out = match qkv_override {
            Some(lora) => self.attn.attend(lora.forward(normed), mask, n_head),
            None => self.attn.forward(normed, mask, n_head),
        };
        let h = h + attn_out;
        let mlp_out = self.mlp.forward(self.ln_2.forward(h.clone()));
        h + mlp_out
    }
}

/// The transformer trunk: token + position embeddings, the block stack, and the
/// final LayerNorm. Field names mirror HF (`wte`, `wpe`, `h`, `ln_f`).
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    /// Token embedding `[vocab, n_embd]`.
    pub wte: Embedding<B>,
    /// Position embedding `[n_positions, n_embd]`.
    pub wpe: Embedding<B>,
    /// The transformer blocks. As a `Vec` field named `h`, burn names its
    /// children `h.0`, `h.1`, … — matching the HF `transformer.h.{i}` keys.
    pub h: Vec<Block<B>>,
    /// Final LayerNorm before the (tied) output head.
    pub ln_f: LayerNorm<B>,
}

/// Intermediate activations captured during a forward pass, for parity
/// localization: the parity test asserts each stage against the PyTorch golden
/// so a mismatch pinpoints the faulty stage rather than only the final logits.
pub struct Gpt2Trace<B: Backend> {
    /// Hidden state right after `wte + wpe` (before block 0).
    pub after_embed: Tensor<B, 3>,
    /// Hidden state after block 0 (HF `hidden_states[1]`).
    pub after_block0: Tensor<B, 3>,
    /// Hidden state after the final `ln_f` (pre-head normed features).
    pub after_lnf: Tensor<B, 3>,
    /// Output logits `[batch, seq, vocab]` from the tied head.
    pub logits: Tensor<B, 3>,
}

/// A hand-built GPT-2 that loads real HF safetensors weights.
///
/// Build with [`Gpt2::init`], populate with [`burn_store`]'s
/// [`ModuleSnapshot::load_from`], then run [`Gpt2::forward`] /
/// [`Gpt2::forward_trace`]. See the [module docs](self) for the loading and
/// weight-tie details.
///
/// [`ModuleSnapshot::load_from`]: burn_store::ModuleSnapshot::load_from
#[derive(Module, Debug)]
pub struct Gpt2<B: Backend> {
    /// The transformer trunk. The single top-level field named `transformer`
    /// makes every parameter path begin `transformer.…`, matching HF.
    pub transformer: Transformer<B>,
    /// The architecture — drives the forward, not a loadable parameter.
    #[module(skip)]
    pub config: Gpt2Config,
}

impl<B: Backend> Gpt2<B> {
    /// Build a GPT-2 with freshly initialized weights of the right shapes,
    /// ready to be overwritten by [`load_from`](burn_store::ModuleSnapshot::load_from).
    pub fn init(config: Gpt2Config, device: &B::Device) -> Self {
        let transformer = Transformer {
            wte: EmbeddingConfig::new(config.vocab_size, config.n_embd).init(device),
            wpe: EmbeddingConfig::new(config.n_positions, config.n_embd).init(device),
            h: (0..config.n_layer)
                .map(|_| Block::init(&config, device))
                .collect(),
            ln_f: LayerNormConfig::new(config.n_embd)
                .with_epsilon(config.layer_norm_epsilon)
                .init(device),
        };
        Self {
            transformer,
            config,
        }
    }

    /// The regex → replacement rename pairs that map HF LayerNorm parameter
    /// names (`…weight`/`…bias`) to burn's (`…gamma`/`…beta`). This is the
    /// *only* remapping GPT-2 needs; every other tensor loads by name with no
    /// transpose. Exposed so the loader and its documentation share one source
    /// of truth.
    pub fn layernorm_key_remap() -> [(&'static str, &'static str); 2] {
        // Match the `ln_1` / `ln_2` / `ln_f` LayerNorm params specifically, so a
        // `.weight` on a `Linear`/`Embedding` is never touched.
        [
            (r"(ln_1|ln_2|ln_f)\.weight$", r"${1}.gamma"),
            (r"(ln_1|ln_2|ln_f)\.bias$", r"${1}.beta"),
        ]
    }

    /// Forward pass, capturing localizing intermediates. `qkv_override`, when
    /// `Some`, replaces block 0's `c_attn` with a LoRA adapter (the M3
    /// LoRA-attach path); `None` is the plain loaded forward.
    fn forward_inner(
        &self,
        ids: Tensor<B, 2, Int>,
        qkv_override: Option<&LoraLinear<B>>,
    ) -> Gpt2Trace<B> {
        let [batch, seq] = ids.dims();
        let device = ids.device();
        let n_head = self.config.n_head;
        let t = &self.transformer;

        // Position ids [batch, seq] = 0..seq per row.
        let pos_data: Vec<i64> = (0..batch).flat_map(|_| 0..seq as i64).collect();
        let positions =
            Tensor::<B, 2, Int>::from_data(TensorData::new(pos_data, [batch, seq]), &device);

        // Embeddings: h = wte(ids) + wpe(positions), both [batch, seq, n_embd].
        let mut h = t.wte.forward(ids) + t.wpe.forward(positions);
        let after_embed = h.clone();

        // Additive causal mask [1, 1, seq, seq]: 0 on/below the diagonal,
        // large-negative above it, broadcast over batch and heads.
        let mask = causal_mask::<B>(seq, &device);

        let mut after_block0 = after_embed.clone();
        for (i, block) in t.h.iter().enumerate() {
            let override_here = if i == 0 { qkv_override } else { None };
            h = block.forward(h, mask.clone(), n_head, override_here);
            if i == 0 {
                after_block0 = h.clone();
            }
        }

        // Final LayerNorm, then the tied head: logits = h · wteᵀ.
        let after_lnf = t.ln_f.forward(h);
        let logits = self.tied_logits(after_lnf.clone());

        Gpt2Trace {
            after_embed,
            after_block0,
            after_lnf,
            logits,
        }
    }

    /// Project final hidden states through the tied output head:
    /// `logits = h · wteᵀ`. GPT-2 has no separate `lm_head` weight — the token
    /// embedding *is* the head — so the tie is implemented here explicitly.
    fn tied_logits(&self, h: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, n_embd] = h.dims();
        let wte = self.transformer.wte.weight.val(); // [vocab, n_embd]
        let vocab = wte.dims()[0];
        let logits = h.reshape([batch * seq, n_embd]).matmul(wte.transpose()); // [batch*seq, vocab]
        logits.reshape([batch, seq, vocab])
    }

    /// Forward pass returning only the logits `[batch, seq, vocab]`.
    pub fn forward(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_inner(ids, None).logits
    }

    /// Forward pass returning the logits plus localizing intermediate
    /// activations (see [`Gpt2Trace`]). Used by the parity harness.
    pub fn forward_trace(&self, ids: Tensor<B, 2, Int>) -> Gpt2Trace<B> {
        self.forward_inner(ids, None)
    }

    /// Forward pass with block 0's `c_attn` projection replaced by a LoRA
    /// adapter — the M3 LoRA-attach entry point. Because the adapter's `B`
    /// factor is zero-initialized, the result is bit-identical to
    /// [`forward`](Self::forward) until training moves `B` off zero, giving a
    /// free attach-integrity check.
    pub fn forward_with_lora_c_attn(
        &self,
        ids: Tensor<B, 2, Int>,
        lora: &LoraLinear<B>,
    ) -> Tensor<B, 3> {
        self.forward_inner(ids, Some(lora)).logits
    }
}

/// Build the additive causal attention mask `[1, 1, seq, seq]`: `0.0` on and
/// below the diagonal, a large negative above it so masked positions vanish
/// under softmax. A finite sentinel (`f32::MIN`) rather than `-inf` keeps
/// autodiff well-defined for the LoRA training step.
fn causal_mask<B: Backend>(seq: usize, device: &B::Device) -> Tensor<B, 4> {
    let mut data = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            data[i * seq + j] = f32::MIN;
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, [seq, seq]), device).reshape([1, 1, seq, seq])
}
