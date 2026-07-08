# 0001 — First real target model for LoRA adaptation (M3)

- **Status:** Accepted
- **Date:** 2026-07-04
- **Milestone:** M3 (issue #2 — "load real base-model weights into burn")
- **Deciders:** loractl maintainers

## Context

M2 (#1) landed a real `BurnTrainer` that trains the low-rank factors of a tiny
`LoraMlp` on synthetic/MNIST data, verified against a PyTorch-referenced,
checked-in numerics golden. The reusable M3 entry point is
`LoraLinear::from_base(base: Linear<B>, rank, alpha, device)`
(`crates/loractl-core/src/lora.rs`): it wraps an **already-loaded** frozen
`Linear`, sizes the factors off `base.weight.dims()` = `[d_input, d_output]`,
and **zero-initializes `B`** so the adapted model's step-0 forward is
bit-identical to the base.

M3 (#2) must make LoRA adapt a *real pretrained model* instead of the toy MLP.
That means loading a real model's safetensors weights into a burn module tree
and re-expressing its forward pass. burn has no turnkey full-model importer, so
the decision splits into three coupled questions:

1. **Which model family** do we target first?
2. **How do we load** its weights into burn (state-dict mapping)?
3. **How do we verify** the forward pass matches the reference?

The dominant cost and risk in M3 is **(3): hand-re-expressing the forward pass
and getting logits to match a reference**, not the load path. Weight loading is
a largely declarative problem in burn 0.21; the forward pass is bespoke code.
The target-model choice therefore **minimizes forward-pass parity risk**, even
at the cost of a slightly more awkward load path.

Two facts about burn 0.21 (verified against the 0.21.0 crate sources under
`~/.cargo/registry/src/*/burn-*-0.21.0/`) shape the decision:

- **`burn-store` is the runtime loader.** `burn-import` is a *build-time*
  ONNX/PyTorch code generator and is the wrong tool for "load weights into a
  module tree I wrote by hand." The runtime path is
  `model.load_from(&mut SafetensorsStore::from_file(path))`.
- **burn `Linear` weight is `[d_input, d_output]`;** PyTorch `nn.Linear` is
  `[out, in]`, so an `nn.Linear` checkpoint needs a transpose on every
  projection. HF **GPT-2 uses `Conv1D`** (weight stored `[in, out]` =
  `[d_input, d_output]`, computing `x @ W`), whose layout is **already
  identical to burn `Linear`** — so GPT-2 projections load **transpose-free**.
  This is the crux of the load-path decision below.

## Decision

**Target the GPT-2 family as loractl's first real base model.** Concretely:

- **Primary target:** GPT-2 small — `openai-community/gpt2`, 124M, MIT license,
  safetensors on the Hub. 12 layers, 12 heads, `d_model` 768, `n_inner` 3072,
  vocab 50257, `n_positions` 1024, learned positional embeddings, pre-LN blocks
  + final `ln_f`, tied LM head, `gelu_new` (tanh) activation.
- **Always-run offline fixture:** a *tiny* real GPT-2 (`GPT2LMHeadModel` at
  `n_layer=2, n_embd=32, n_head=2, vocab=61, n_positions=16, n_inner=64`, seed
  1234) built once from an HF `GPT2Config`; its **weights are checked in** as
  safetensors (~81 KB) alongside golden logits + intermediate activations. This
  is the deterministic, download-free proof of correct architecture wiring — the
  direct analogue of M2's checked-in toy, lifted to a real transformer.

GPT-2 was chosen over a modern LLaMA-style model because every GPT-2 primitive
already exists in burn-nn 0.21 with an **exact, confirmed** match, and GPT-2
uses **none of the modern-LLM forward-pass footguns**:

| GPT-2 component | burn-nn 0.21 primitive | Parity status |
|---|---|---|
| Token + **learned** positional embedding (`wte`, `wpe`) | `Embedding` ×2, added | exact (same `[n, d]` layout) |
| Pre-norm `ln_1` / `ln_2` / `ln_f` | `LayerNorm`, eps 1e-5 | **confirmed**: burn uses biased/population variance, matching torch `F.layer_norm` |
| Causal self-attention (fused-QKV `c_attn`, `c_proj`) | hand-rolled over `Linear` + `softmax` | exact |
| MLP `c_fc` → GELU → `c_proj` | `Linear` + `Gelu::new_approximate()` | **confirmed**: `new_approximate` is tanh `gelu_new` (burn-nn's own test asserts parity with torch `approximate="tanh"`) |
| LM head | tied `wte` matrix, applied in the forward | no separate param |

No RoPE (convention-matching bug source), no RMSNorm, no grouped-query
attention, no SwiGLU. Loading and running are native fp32 end to end — no dtype
cast on either side, removing another divergence source.

### Weight-loading approach (acceptance a)

Load the **unmodified** HF GPT-2 safetensors at runtime via `burn-store`'s
`SafetensorsStore`, into a hand-built burn module tree whose field names mirror
HF exactly (`Gpt2 { transformer: Transformer { wte, wpe, h: Vec<Block>, ln_f } }`,
`Block { ln_1, attn: { c_attn, c_proj }, ln_2, mlp: { c_fc, c_proj } }`). Because
a `Vec<Block>` field named `h` yields burn parameter paths `transformer.h.0.…`,
`transformer.h.1.…`, the module paths **already equal** the HF keys — so **no
structural key remapping is needed**. The single required rename is LayerNorm's
`weight`/`bias` → burn's `gamma`/`beta`:

```rust
use burn_store::{ModuleSnapshot, SafetensorsStore, KeyRemapper};

let remapper = KeyRemapper::from_patterns(vec![
    (r"(ln_1|ln_2|ln_f)\.weight$", r"${1}.gamma"),
    (r"(ln_1|ln_2|ln_f)\.bias$",   r"${1}.beta"),
])?;
let mut store = SafetensorsStore::from_file(path)
    // Default IdentityAdapter — do NOT attach PyTorchToBurnAdapter.
    .allow_partial(true)
    .remap(remapper);

let result = model.load_from(&mut store)?;
assert!(result.errors.is_empty());
assert!(result.missing.is_empty());   // tied head is in the forward, not a param
// NOTE: do NOT assert result.unused.is_empty() — see below.
```

**The transpose decision — the single sharpest technical point.** GPT-2's
`Conv1D` weights are `[in, out]` = burn `Linear` `[d_input, d_output]`, so **no
transpose is wanted**. We use the **default `IdentityAdapter`** and do **not**
attach `PyTorchToBurnAdapter`: attaching it would fire its automatic `nn.Linear`
transpose (keyed on `module_type == "Struct:Linear"`, blind to the `Conv1D`
origin) and **silently double-transpose GPT-2's weights into wrong values** —
the classic footgun the burn-book `nn.Linear` examples invite. Norm-param
renaming, which `PyTorchToBurnAdapter` would otherwise do for free, is handled
by the two `ln_*` KeyRemapper patterns above.

**`ApplyResult` semantics — corrected from a naïve reading.** The two fields
mean different things and must be asserted differently:

- `result.unused` = *source* tensors with no matching module param. HF stores
  non-persistent causal-mask buffers (`attn.bias` / `masked_bias`) in some
  checkpoints; these legitimately land in `unused`. **Asserting `unused` empty
  is therefore wrong** and would spuriously fail — the draft design's
  `assert!(result.unused.is_empty())` is a bug. (In the checked-in tiny fixture
  and the real gpt2 checkpoint, `unused` happens to be empty because this
  transformers version omits those buffers, but the load path must not *depend*
  on that.)
- `result.missing` = module params the *source* lacks. Because the tied head is
  implemented in the forward (below) with **no separate `lm_head` param**, and
  every other param has a matching key, `missing` is empty. We assert that.

**The weight tie is explicit in the forward — `allow_partial` does not fill it.**
GPT-2 ties the output head to `wte`, so the safetensors has **no `lm_head` key**.
Rather than materialize a head parameter (which would then show up in `missing`
and need `allow_partial` to tolerate an *unfilled* tensor — a silently-random
head), the forward computes `logits = h · wteᵀ` directly. `allow_partial(true)`
only tolerates a *gap* in the source (e.g. the absent head key, any HF buffers);
it never *populates* a param. The tie is our code, not a load-time trick.

The fused `c_attn` (`n_embd → 3·n_embd`) loads as a single burn `Linear`; its
output is split `[e, e, e]` into Q/K/V on the last dim (matching HF's
`split(n_embd, dim=2)`).

**Dependency note:** M3 adds `burn-store` as a **direct** dependency with
`default-features = false, features = ["safetensors", "std"]`, pinned to
`0.21.0` to unify `burn-core` with the `burn` umbrella. `std` pulls `memmap2`
(used by `from_file`) and `regex` (the remapper) — fully offline, no HTTP
client.

### Forward-pass verification methodology (acceptance b)

Mirror M2's harness (a Python reference derives a checked-in golden → `cargo
test` verifies offline via `include_str!` → tolerance is *measured* then pinned),
split into two tiers because real weights can't be checked in:

| Tier | Command | Content | Gate |
|---|---|---|---|
| **Always-run** (offline, ms) | `cargo test` / `just test` | Tiny GPT-2: checked-in safetensors + golden, full burn forward vs golden | abs `1e-4` + tolerance-free gate |
| **Opt-in** (`#[ignore]`, `--features gpt2-real`) | `just test-gpt2-real` | Downloads real `gpt2` (via `just gpt2-reference`), burn forward vs local golden | abs `1e-2` + tolerance-free gate |
| **Regen** | `just gpt2-tiny-reference` / `just gpt2-reference` | `reference/gpt2_tiny_reference.py` / `gpt2_reference.py` derive goldens | — |

Details:

- **Fixed input, no tokenizer.** Both reference and burn are fed a hard-coded
  `Vec<i64>` of token IDs (tiny: `[5,12,7,3,42,1,0,9]`), comparing full-sequence
  logits.
- **Tolerance-free categorical gate (the strongest, BLAS-order-independent
  signal):** the last-token top-1 argmax must match the reference **exactly**,
  and logit cosine similarity must be **> 0.99999**. Chosen over an elementwise
  tolerance as the primary correctness claim — a correct forward reproduces the
  ranking regardless of last-ulp summation-order noise. (Top-1, not top-5:
  top-1 is what greedy generation consumes and is the tightest single-token
  claim.)
- **Intermediate activation checkpoints for localization.** The tiny golden also
  carries the hidden state *after the embedding sum*, *after block 0*, and
  *after `ln_f`*; the test asserts each **in order** so a mismatch pinpoints the
  faulty stage. Observed on the checked-in fixture: `after_embed` max|Δ| **0**,
  `after_block0` **9.3e-9**, logits **8.9e-8** — pure f32 rounding.
- **Each known divergence source is pinned:** `gelu_new` →
  `Gelu::new_approximate()` (tanh, not exact-erf `Gelu::new()`); LayerNorm eps
  `1e-5`, biased variance; attention scale `1/√head_dim`; causal mask (a large
  finite negative on the strict-upper triangle *before* softmax — finite rather
  than `-inf` so autodiff stays defined for the LoRA step); the tied head.
- **A subtle golden quirk, documented in the test.** HF's *last*
  `output_hidden_states` entry is **already** `ln_f`-applied (verified: it has
  per-row mean 0 / std ≈ 1, and `hidden_states[-1] @ wteᵀ` reproduces the logits
  bit-exactly). The tiny golden's `hidden_after_lnf` is the reference's
  `ln_f(hidden_states[-1])` — i.e. `ln_f` applied twice. The test reproduces
  that double application to match verbatim; the authoritative `ln_f`+head check
  is the logits stage, which uses the model's true single-`ln_f` features.
- **Do NOT reuse burn's `TransformerEncoder` / `MultiHeadAttention`** — they are
  *post-LN* and will not match GPT-2's *pre-LN* fused-QKV block. Hand-roll the
  block from primitives so every divergence source is pinned.

Observed real-`gpt2` parity (via `just gpt2-reference` + `just test-gpt2-real`):
logits max|Δ| **3.97e-4** (12-layer fp32 accumulation vs libtorch summation
order), last-token top-1 exact, cosine **1.0** — the full production load +
forward path validated at 124M scale.

### LoRA attach plan (acceptance c)

Strictly **gated behind a green base forward** (a wrong base makes a LoRA-step
"pass" meaningless):

1. With the base-forward parity green, wrap a target linear with the existing
   `LoraLinear::from_base(base_linear, rank, alpha, device)`. The load path
   preserves the `[d_input, d_output]` layout, so `from_base` sizes the factors
   correctly with no shape juggling.
2. **Target `c_attn`** (fused QKV) of block 0 for the M3 proof — the standard
   LoRA site. `Gpt2::forward_with_lora_c_attn` threads the adapter through the
   full transformer without duplicating the forward math. **(Superseded in M6
   (#17): this single-`c_attn` attach was re-expressed through the generic
   name-keyed mechanism — `Gpt2::forward_with_adapters(ids, &LoraAdapters)` over
   `injectable_sites`/`build_adapters` — and `forward_with_lora_c_attn` +
   `qkv_override` were removed. See ADR-0004 / issue #17.)**
3. Because `B` is zero-initialized, the step-0 adapted forward is bit-identical
   to the base — the test asserts the adapted pre-step logits still match the
   base golden (a free check that attach didn't perturb the forward).
4. Run **one** step over the LoRA-wrapped real model and assert: it completes
   with no shape/dtype panic, produces a **finite, positive** loss, yields a
   gradient on `lora_a`/`lora_b` but **none** on the frozen base weight, and
   that `B` has moved off zero after the Adam step.

### Load-bearing invariant

All of the above lands as a new `gpt2` module inside `loractl-core` — pure
`Module` + forward. Core still imports no `clap`, writes nothing to
stdout/stderr, and depends only on `burn` / `burn-store`. The Python reference
scripts are a dev/test harness, never a runtime dependency. Dependency direction
stays `cli → core`.

### Acceptance-criteria → artifact map (issue #2)

| # | Criterion | Artifact |
|---|---|---|
| a | Load real GPT-2 safetensors into burn | `crates/loractl-core/src/gpt2.rs` (`Gpt2::init` + `layernorm_key_remap` + `burn_store` load); asserted in both tests' `load_tiny` |
| b | Forward-pass parity vs PyTorch | `tests/gpt2_parity.rs` (always-run, tiny, stage-localized) + `tests/gpt2_real.rs` (opt-in, real gpt2) |
| c | Attach LoRA + one training step | `tests/gpt2_lora_step.rs` (grad routing, one Adam step; ported to `forward_with_adapters` in M6 (#17), which superseded the original `forward_with_lora_c_attn`) |
| d | README documents it | `README.md` roadmap + "Real base model (GPT-2)" section |

## Alternatives Considered

**A hand-authored tiny LLaMA-style model first.** Rejected as the *primary*
target. Its one advantage is the load path (genuine `nn.Linear [out, in]`
weights, so `PyTorchToBurnAdapter`'s transpose "just works"), but that is
outweighed by three forward-pass parity risks GPT-2 lacks: RoPE convention, GQA
KV-head repetition, and the SwiGLU split. The determinism goal of an authored
model is met instead by our **checked-in tiny GPT-2 fixture** — same benefit, on
the chosen family's exact code path.

**SmolLM2-135M / Qwen2-0.5B (modern LLaMA arch, Apache-2.0) — the next target
model.** A future modern-architecture increment that reuses this M3 loader +
parity harness — distinct from the tracked M4 (sampling + adapter I/O, #3) and
M5 (API, #4) milestones; it is a *target-model* follow-on, not a numbered
milestone. These earn RoPE + RMSNorm + SwiGLU + GQA. burn-nn 0.21 ships the
primitives (`RmsNorm`, `RotaryEncoding`, `SwiGlu`), so this is a follow-on, not
a dead end — but with a **banked warning**: burn's `RotaryEncoding` uses the
*interleaved* (adjacent-pair) rotation convention, whereas HF LLaMA/SmolLM use
the *half-split* (`rotate_half`) convention. Loading HF RoPE weights into burn's
default RoPE without accounting for that mismatch produces "logits almost match
but not quite" — exactly the class GPT-2 sidesteps. That work must pin the RoPE
convention explicitly, the way M3 pins GELU/LayerNorm here.

**`burn-import` (ONNX/PyTorch codegen).** Rejected — a build-time whole-graph
generator that fits poorly with a hand-built module tree.
`burn-store::SafetensorsStore` is the intended runtime surface.

**`PyTorchToBurnAdapter` with GPT-2.** Rejected: it would wrongly transpose
GPT-2's already-burn-layout `Conv1D` weights (detailed above). The
IdentityAdapter + explicit norm-rename path loads the *unmodified* checkpoint
and is self-documenting.

**burn's `TransformerEncoder` / `MultiHeadAttention`.** Rejected — post-LN,
won't match GPT-2's pre-LN block.

## Consequences

**Positive**

- Lowest-friction first real target: every primitive exists in burn-nn 0.21 with
  a verified-parity match, and no modern-LLM parity footguns. Real gpt2 loads and
  forwards at max|Δ| 3.97e-4 / cosine 1.0 on the first correct implementation.
- Transpose-free, structural-remap-free load path (`Conv1D` == burn `Linear`
  layout; module tree mirrors HF names) — the only rename is `ln_* weight/bias
  → gamma/beta`.
- The checked-in tiny GPT-2 fixture keeps `cargo test` offline, fast, and
  deterministic while proving the exact architecture code path; real `gpt2`
  stays opt-in behind `--features gpt2-real` (mirrors M2's MNIST gating).
- A clean runway to a modern-arch target (SmolLM2) once the harness is proven,
  and to the tracked M4 (sampling + adapter I/O) / M5 (API) milestones.

**Negative / costs**

- Re-expressing the pre-LN GPT-2 block by hand (fused-QKV split, causal mask,
  tied head) is the bulk of M3 — bespoke code, not a library call.
- The `Conv1D`-not-`nn.Linear` layout is GPT-2-specific: the no-transpose
  assumption **must not** be copied to a LLaMA/Qwen target, which needs a
  transpose on every projection.
- New direct dependency on `burn-store`.
- The opt-in real-weights test needs the ~498 MB `gpt2` download; its
  safetensors + golden are `.gitignore`d and produced by `just gpt2-reference`
  (set `HF_HOME` to a big disk per `huggingface-downloads` guidance).

**Risks & mitigations**

- *Wrong-transpose corruption* → IdentityAdapter (not `PyTorchToBurnAdapter`);
  the forward-parity + top-1/cosine gate catches it if it slips.
- *`ApplyResult` mis-assertion* → assert `errors`/`missing` empty, **not**
  `unused` (documented above).
- *GELU exact-vs-tanh* / *LayerNorm variance* → pinned + confirmed; the
  intermediate checkpoints localize either.
- *Tolerance guessed* → measured during bring-up, then pinned; the tolerance-free
  top-1 + cosine gate is the backstop.

## References

- Issue #2 (M3), roadmap in `README.md`.
- `crates/loractl-core/src/gpt2.rs` — the hand-built GPT-2 + `burn-store` load.
- `crates/loractl-core/src/lora.rs` — `LoraLinear::from_base` (M3 attach entry
  point; reads `[d_input, d_output]`, `B` zero-init → step-0 no-op).
- `crates/loractl-core/tests/{gpt2_parity,gpt2_lora_step,gpt2_real}.rs`.
- `reference/gpt2_tiny_reference.py`, `reference/gpt2_reference.py`.
- burn 0.21 sources: `burn-store-0.21.0` (`src/safetensors/store.rs`,
  `src/adapter.rs`, `src/keyremapper.rs`, `src/apply_result.rs`),
  `burn-nn-0.21.0` (`modules/{linear,embedding}.rs`, `modules/norm/layer.rs`,
  `modules/attention/mask.rs`, `activation/gelu.rs`).
- HF: `openai-community/gpt2` (MIT).
