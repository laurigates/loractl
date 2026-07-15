---
id: ADR-0004
status: Accepted
date: 2026-07-08
---

# 0004 — Krea 2 as the image-diffusion LoRA target (M6+)

- **Status:** Accepted
- **Date:** 2026-07-08
- **Milestones:** M6–M14 (issues [#17](https://github.com/laurigates/loractl/issues/17)–[#25](https://github.com/laurigates/loractl/issues/25)); amended by M15 ([#82](https://github.com/laurigates/loractl/issues/82))
- **Deciders:** loractl maintainers

## Context

M1–M5 landed a complete but **text-domain** LoRA harness: a burn-backed
`BurnTrainer` that adapts a single `Linear` (`LoraLinear::from_base`), a
hand-built GPT-2 with forward-pass parity vs PyTorch (ADR-0001), portable
`.safetensors` adapter I/O + deterministic sampling (ADR-0002), and an HTTP/SSE
event API (ADR-0003). The objective throughout is supervised cross-entropy; the
backend is ndarray CPU; the largest model is 124M.

The next goal is a genuinely different domain: **train LoRA adapters for
"Krea 2", an image-generation model.** This ADR records the target-model
decision, confirms feasibility, and lays out the roadmap the M6–M14 issues
implement. It is the image-domain analogue of ADR-0001's target-model choice.

### Feasibility — Krea 2 is open-weights (the make-or-break finding)

Krea 2 shipped as **open weights** on 2026-06-23 (Krea AI). Two checkpoints on
Hugging Face:

- **`krea/Krea-2-Raw`** — undistilled base, explicitly "a good base for
  finetuning." **This is the LoRA-training target.**
- **`krea/Krea-2-Turbo`** — distilled (~8-step) fast-inference variant.

The official inference repo (`github.com/krea-ai/krea-2`) documents the intended
workflow — **"train a LoRA on Raw, apply it to Turbo"** — and names diffusers,
ostris/ai-toolkit, fal, and kohya as supported training tools. License is a
custom source-available "Krea 2 Community License"; weights are downloadable, so
local training is not blocked. Had Krea 2 been API-only, local LoRA training
would have been impossible — this finding is what makes the roadmap real.

Note the lineage without conflating it: **Krea 1 → FLUX.1 Krea [dev] (2025, a
Black Forest Labs × Krea collaboration) → Krea 2 (2026, built from scratch by
Krea).** Krea 2 is *not* a FLUX fine-tune; it is an independently-built ~12B
rectified-flow transformer that happens to share FLUX's model class, which is
why the FLUX LoRA tooling transfers.

### Krea 2 architecture (from the official technical report + model card)

| Component | Krea 2 | Contrast with what loractl has / with FLUX.1 |
|---|---|---|
| Denoiser | **MMDiT / DiT**, rectified-flow transformer, ~12B params, latent-space | loractl has GPT-2 (dense causal LM, 124M) |
| Objective | **rectified flow matching, v-parameterization** | loractl uses cross-entropy; not DDPM epsilon |
| Attention | **GQA + gated-sigmoid attention, QK-Norm** | GPT-2 is plain MHA |
| Norm | **zero-centered RMSNorm** | GPT-2 is LayerNorm |
| Positional | **3D axial RoPE** | GPT-2 is learned absolute (no RoPE footgun) |
| Text encoder | **Qwen 3 VL, text-only**, 12-layer feature aggregation | FLUX.1 uses T5-XXL + CLIP-L; loractl has neither |
| VAE | **`AutoencoderKLQwenImage`**, f8 **16-channel** latents, attention-free trunks (mid block keeps one attention — see the M9 correction below) | loractl has no VAE |
| LoRA format | safetensors, kohya / diffusers-PEFT naming | loractl uses a bespoke `fc2.lora_a/b` scheme (M4) |

The **Qwen 3 VL text encoder** and the **autoencoder** are the two
components with **zero Rust prior art** in either burn or candle — they dominate
the from-scratch cost. Both have concrete, researched burn design targets in
their issues ([#20](https://github.com/laurigates/loractl/issues/20),
[#21](https://github.com/laurigates/loractl/issues/21)), with several notable
de-riskers: the encoder runs **text-only** (drop the vision tower via a
`visual.*` regex filter at load; its M-RoPE collapses to plain 1D RoPE, and as a
frozen extractor it needs **no Autodiff** backend at all — a clean split from the
DiT+LoRA that do), and the VAE was researched as **attention-free** (pure
conv/norm/act with a custom left-padded causal `Conv3d`) — a claim the M9
correction below revises. Every specific (12 aggregation layers, 28/11 enc/dec
depth, 16-channel f8 latents) is a *target to confirm against `krea-ai/krea-2`
source*, per the report-vs-code risk below.

> **Correction (M9, 2026-07-14).** The source settled the AE's identity: it is
> the **stock Qwen-Image VAE** — `krea-ai/krea-2`'s `autoencoder.py` wraps
> diffusers' `AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image",
> subfolder="vae")` verbatim, adding only per-channel `latents_mean`/
> `latents_std` (de)normalization — not a Qwen-Image/FLUX-2 *hybrid* as this
> section's research suggested. Two further report-vs-code deltas landed with
> the M9 port (`crates/loractl-core/src/qwen_vae.rs`): the AE is **not** fully
> attention-free (`attn_scales: []` only strips trunk attention; the mid block
> always carries one single-head spatial self-attention), and its norms are the
> Qwen RMS-norm variant + no `GroupNorm` on the actual checkpoint path. Depths
> are per-config (`dim_mult` [1,2,4,4], `num_res_blocks` 2), not "28/11".

## Decision

**Target `krea/Krea-2-Raw` for image-diffusion LoRA training, and build the full
denoiser + VAE + text-encoder + diffusion-loop stack in burn.** Concretely:

1. **Stay on burn.** loractl remains a single burn codebase; the Krea 2 stack is
   built greenfield in burn, reusing the ADR-0001 methodology (module tree
   mirrors source keys → `burn-store` load → staged parity golden →
   tolerance-free top-k/cosine gate). See *Alternatives Considered* for the
   burn-vs-candle weighing.
2. **Pin the LoRA output format early (M6).** Adopt a real interop convention
   (kohya `lora_unet_*`/`.lora_down`/`.lora_up`/`.alpha` or diffusers-PEFT
   `transformer.*.lora_A`/`lora_B`) so a Rust-produced adapter loads in
   ComfyUI/diffusers/Krea. This contract is independent of the model internals
   and cheap to get right up front.
3. **Follow the M6–M14 roadmap** (below), each milestone a tracking issue that
   mirrors the M1–M5 "prove the piece in isolation against a reference before
   scaling" discipline.

### Roadmap (M6–M14)

| M | Issue | Scope | Depends on |
|---|---|---|---|
| M6 | [#17](https://github.com/laurigates/loractl/issues/17) | Generic LoRA injection (by name pattern) + kohya/PEFT safetensors export | — |
| M7 | [#18](https://github.com/laurigates/loractl/issues/18) | GPU compute backend (wgpu/cuda); ndarray stays the offline test backend | — |
| M8 | [#19](https://github.com/laurigates/loractl/issues/19) | Rectified-flow v-param objective + logit-normal timestep sampling; proven on a synthetic latent toy | — |
| M9 | [#20](https://github.com/laurigates/loractl/issues/20) | Krea 2 latent VAE in burn; image→latent parity vs `autoencoder.py` | M7 |
| M10 | [#21](https://github.com/laurigates/loractl/issues/21) | Qwen 3 VL text encoder in burn (largest gap, no Rust prior art) | M7 |
| M11 | [#22](https://github.com/laurigates/loractl/issues/22) | Krea 2 MMDiT denoiser in burn + forward parity + LoRA attach | M6, M7 |
| M12 | [#23](https://github.com/laurigates/loractl/issues/23) | Image dataset pipeline: bucketing + latent/embedding caching | M9, M10 |
| M13 | [#24](https://github.com/laurigates/loractl/issues/24) | Single-GPU 12B fit: bf16, grad checkpointing, 8-bit Adam, QLoRA/NF4 | M11, M7 |
| M14 | [#25](https://github.com/laurigates/loractl/issues/25) | End-to-end `DiffusionTrainer` + ComfyUI/Turbo interop proof | M6–M13 |

### Load-bearing invariant (unchanged)

The Krea 2 trainer lands as a **new `impl Trainer`** (`DiffusionTrainer`) in
`loractl-core`, emitting the same `TrainEvent`s through the callback sink. Core
still imports no `clap` and prints nothing; each front-end changes only its one
constructor line. If a diffusion trainer forces front-end changes beyond that
seam, the event abstraction has leaked — fix the abstraction, not the front-end.

## Alternatives Considered

**Pivot to candle.** candle has a pure-Rust FLUX.1 inference implementation
(`candle-transformers::models::flux`: DiT + rectified-flow sampling + FLUX AE),
plus T5, CLIP, `mmdit` (SD3), and `EricLBuehler/candle-lora`. That is materially
more diffusion prior art than burn (whose only image-diffusion port is
`Gadersd/stable-diffusion-burn` — SD-1.4 UNet, epsilon-prediction, inference
only: wrong architecture family and inference-only). Pivoting would give the DiT
forward and rectified-flow loop a Rust cross-reference to port from.

**Rejected as a repo pivot, retained as a reference.** loractl's thesis and its
entire M1–M5 investment (parity harness, `burn-store` loading discipline,
`LoraLinear`, event stream, CLI/config/API) are burn-native; a pivot discards
that for a partial head-start. Crucially, the **two hardest components — Qwen 3
VL and the Qwen-Image AE — are greenfield in candle too**, so candle's advantage is
concentrated in the DiT (M11), which the ADR-0001 methodology already knows how
to build and verify. Decision: **stay on burn**, but treat candle's `flux`
module and `candle-lora` as **cross-references** when implementing M8/M11 (a
second Rust implementation to diff parity against, alongside the Python golden).

**Target FLUX.1 Krea [dev] instead of Krea 2.** Its T5+CLIP stack has candle
prior art and would sidestep the Qwen 3 VL gap. Rejected as the *primary* target
because the user goal is Krea 2 specifically; FLUX.1 Krea may still serve as an
*intermediate* proving ground for M8/M11 (same rectified-flow class, simpler
encoders) if M10 proves a bottleneck — a fallback, not the plan.

## Consequences

**Positive**
- Feasible: open weights + a documented "train on Raw, apply to Turbo" workflow.
- Reuses the proven ADR-0001 parity methodology and the M1–M5 architecture
  (events, config, adapter I/O, API) wholesale — only the model code is new.
- The kohya/PEFT format contract (M6) makes outputs interoperable from the first
  produced adapter.

**Negative / costs**
- This is effectively a second, larger project: a 12B multimodal diffusion stack
  greenfield in burn. The Qwen 3 VL encoder (M10) and Qwen-Image AE (M9) have no
  Rust prior art anywhere.
- Every modern-arch footgun GPT-2 was chosen to avoid (3D axial RoPE half-split
  vs interleaved, RMSNorm, GQA, SwiGLU) now applies at once (M11), at 12B scale
  with a sharded checkpoint.
- Requires GPU + a quantization/offload stack (M7, M13) that M1–M5 never needed.

**Risks & mitigations**
- *RoPE / norm convention drift* → pin each against `krea-ai/krea-2` source and a
  golden, per ADR-0001; use candle's `flux` as a second cross-reference.
- *Report-vs-code divergence* → read block structure, AE scale/shift + channel
  count, and Qwen3-VL aggregation layers from the **source code**, not the
  technical report (which gives shape, not truth).
- *Scale intractability* → each milestone proves its piece on a small
  fixture/CPU before the full 12B run; QLoRA (M13) targets single-GPU VRAM.

## Amendment — M15 (2026-07-15): training directly on Krea-2-Turbo

The original decision took the official workflow at its word — **"train a LoRA
on Raw, apply it to Turbo"** — and targeted `krea/Krea-2-Raw` exclusively. M15
([#82](https://github.com/laurigates/loractl/issues/82)) amends that: training
**directly on Krea-2-Turbo is now supported**. Raw remains the default and the
recommended target; what changed is the recognition that the Turbo restriction
was a *load seam*, not an architecture gap.

**Why the amendment is cheap.** Turbo is architecturally identical to Raw —
the same 430 tensor keys, with per-tensor distillation deltas of 3–11% — so
the MMDiT port (M11), its key remap, and the flow-matching objective (M8)
apply unchanged. All that blocked turbo training was loading: the denoiser
filename was hardcoded to `raw.safetensors`, and the widely-distributed
scaled-fp8 turbo repacks (13.1 GB vs 26.3 GB bf16) could not load at all.

**What landed:**

- **Variant seam + filename override.** `ModelVariant::Krea2Turbo`
  (`variant: krea2-turbo`) reuses `MmditConfig::krea2()` and shares Raw's
  encoder-cache fingerprint (no cache invalidation), defaulting the denoiser
  filename to `turbo.safetensors`; an optional `model.checkpoint` overrides
  the filename for any variant (e.g. a local
  `krea2_turbo_fp8_scaled.safetensors`). Existing configs parse unchanged.
- **Scaled-fp8 loading, auto-detected from the safetensors header.** The
  ComfyUI-style repack stores `float8_e4m3fn` weights plus f32 0-d
  `*.weight_scale` sidecars (the verified local file: 686 keys = 256 F8_E4M3 +
  256 scalar scales + 174 BF16, quantization map in `__metadata__`).
  burn-store 0.21 has **no `F8_E4M3` dtype arm** — it errors while building
  snapshots, before any `ModuleAdapter` can intervene — so M15 adds a custom
  snapshot source (`src/fp8.rs`) that lazily dequantizes
  `LUT[byte] · weight_scale` to f32 per tensor (exact 256-entry e4m3fn LUT;
  per-tensor scalar and per-output-channel scales) and hands the snapshots to
  the same remap → transpose → cast → apply pipeline the bf16 path uses.
  Non-fp8 headers keep the existing burn-store path untouched, and the
  mmap-backed per-tensor-lazy streaming memory profile is preserved.
- **Deliberate hard-error scope.** Out-of-contract files fail loudly rather
  than half-load (the M7 no-silent-fallback rule): the legacy ComfyUI
  `scaled_fp8` convention (marker key / `.scale_weight` keys), scale shapes
  that are neither 0-d nor per-output-channel, and unexpected leftover keys —
  e.g. the `Krea2_Turbo_fp8mixed` repack's baked-in `last.up`/`last.down`
  LoRA — are all errors.

**Scope note.** M15 is the load seam only; no real turbo training run is
claimed — that remains future work alongside M14's open real-run checkbox.
Deferred follow-ups, tracked separately: an optional Turbo training adapter
(assistant-LoRA merge-at-load,
[#83](https://github.com/laurigates/loractl/issues/83)) and dynamic
resolution-based timestep-shift parity
([#84](https://github.com/laurigates/loractl/issues/84)).

## References

- Krea 2 Technical Report — <https://www.krea.ai/blog/krea-2-technical-report>
- Official inference code — <https://github.com/krea-ai/krea-2> (`mmdit.py`,
  `autoencoder.py`, `encoder.py`)
- Model cards — <https://huggingface.co/krea/Krea-2-Raw>,
  <https://huggingface.co/krea/Krea-2-Turbo>
- Predecessor (distinct model) — <https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev>
- LoRA recipe reference — diffusers `train_dreambooth_lora_flux`
  (<https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md>),
  FLUX QLoRA (<https://huggingface.co/blog/flux-qlora>)
- Rust prior art — `candle-transformers` (flux/t5/clip/mmdit),
  `EricLBuehler/candle-lora`, `Gadersd/stable-diffusion-burn`
- loractl ADR-0001 (target-model + parity methodology this ADR scales up)
