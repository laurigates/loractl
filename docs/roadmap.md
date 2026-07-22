# Roadmap & milestone history

The detailed, per-milestone record for `loractl`. The [README](../README.md)
carries a compact checklist; this document is the long-form history and the
current-direction detail. Milestones are tracked as GitHub issues #1–#4 and
#17–#25; keep the two in sync when a milestone lands.

## Where the project is

Milestones M1–M15 (#1–#4, #17–#24, plus #82) have landed. The remaining open
box is **M14's real-run interop proof** (#25): train a LoRA on
`krea/Krea-2-Raw` through the landed `DiffusionTrainer` and prove the exported
adapter loads and visibly conditions generation in ComfyUI / Krea-2-Turbo.

The strategy and gap analysis for the Krea 2 image-diffusion target is
[ADR-0004](adrs/0004-krea2-image-diffusion-target.md).

## Text-domain harness (M1–M5)

- **M1 — Skeleton.** Workspace, CLI (`train`/`sample`/`completions`), config
  layering, event → progress-bar rendering, `MockTrainer`.
- **M2 — Correctness harness** ([#1](https://github.com/laurigates/loractl/issues/1)).
  burn-backed `BurnTrainer` trains a LoRA `Module` (frozen base, trained A·B)
  on a tiny MLP; numerics pinned against a PyTorch reference (offline,
  always-run); real MNIST convergence + accuracy proven behind an opt-in
  `mnist` feature. The loop is verified in isolation before any large model.
- **M3 — Real base model** ([#2](https://github.com/laurigates/loractl/issues/2)).
  Hand-built GPT-2 loads real HF safetensors into burn (transpose-free
  state-dict mapping via burn-store), forward-pass parity proven against
  PyTorch on a checked-in tiny GPT-2 (offline, always-run) and real `gpt2`
  (opt-in); LoRA attached to the loaded model runs a training step. See
  [ADR-0001](adrs/0001-first-real-target-model.md).
- **M4 — Sampling & adapter I/O** ([#3](https://github.com/laurigates/loractl/issues/3)).
  Adapters save to and load from real `.safetensors` files (adapter-only + a
  JSON sidecar), `loractl sample` runs a deterministic, prompt-seeded forward
  pass, and periodic validation samples are written and reported during
  training. See [ADR-0002](adrs/0002-adapter-format-and-sample-semantics.md).
- **M5 — API crate** ([#4](https://github.com/laurigates/loractl/issues/4)).
  `loractl-api` exposes the event stream over HTTP so a GUI can be built
  independently: `POST /runs` starts a training run, `GET /runs/{id}/events`
  streams its events as SSE (full replay from event 0, then live tail), with
  the wire shapes pinned byte-for-byte by a golden test. See
  [ADR-0003](adrs/0003-http-api-event-streaming.md).

## Krea 2 image-diffusion LoRA (M6–M15)

M1–M5 built a complete but **text-domain** harness. The remaining goal is a
different domain entirely: training LoRA adapters for **Krea 2**, an
open-weights (`krea/Krea-2-Raw`) ~12B rectified-flow **image** model. This
reuses loractl's architecture (event stream, config, `burn-store` loading, the
parity-golden methodology) but almost none of its model code — the denoiser,
VAE, and text encoder were all greenfield in burn.

- **M6 — Generic LoRA injection + kohya-ss export** ([#17](https://github.com/laurigates/loractl/issues/17)).
  `LoraAdapters` injects a name-keyed set of low-rank deltas across a module
  tree (config `targets` patterns → `build_adapters` over a model's
  `injectable_sites`); GPT-2's attach is re-expressed through it.
  `export_adapters` writes a kohya-ss `.safetensors` (transposed
  `lora_down`/`lora_up` + `.alpha` scalar) so a LoRA loads in ComfyUI/Krea,
  proven offline against a golden. A `PeftDiffusers` format is reserved behind
  the `AdapterNameMapper` seam. The Krea 2 export's key names are additionally
  pinned against ComfyUI's *own* LoRA key map (`tests/krea2_lora_keys.rs`,
  golden from pinned upstream source) — a golden alone pins our convention, not
  the consumer's, which is the gap
  [#137](https://github.com/laurigates/loractl/issues/137) surfaced. ComfyUI
  accepts both the bare diffusers key we emit and the native
  `diffusion_model.blocks.N.*` form community LoRAs use.
- **M7 — GPU compute backend** ([#18](https://github.com/laurigates/loractl/issues/18)).
  The training loop is generic over `B: AutodiffBackend`; `BurnTrainer`
  dispatches a config-selected backend (`compute.backend`) at run time —
  `ndarray` (CPU, always compiled, the offline/CI default), `wgpu` (GPU: Metal
  on Apple Silicon), and compile-gated `cuda`/`tch`. Selecting a backend the
  binary wasn't built with fails loudly, never a silent CPU fallback. `just
  test` stays offline on ndarray; the GPU path is verified locally on Metal
  (`just test-wgpu`).
- **M8 — Rectified-flow objective** ([#19](https://github.com/laurigates/loractl/issues/19)).
  Flow-matching v-prediction (`v = ε − x₀`, SD3 time convention: t=0 data, t=1
  noise) with logit-normal + shifted timestep sampling
  (`crates/loractl-core/src/flow.rs`; kohya/SD3 `shift: 3.0` default). `task:
  flow-matching` trains a LoRA velocity net on a synthetic latent toy, pinned
  against a PyTorch golden (M2 methodology, `just flow-reference`); adapter
  sidecars record the task and `loractl sample` refuses velocity nets.
- **M9 — Krea 2 latent VAE** ([#20](https://github.com/laurigates/loractl/issues/20)).
  Krea 2's autoencoder is the **stock Qwen-Image VAE** (diffusers
  `AutoencoderKLQwenImage` + per-channel latent stats), so `QwenVae`
  (`src/qwen_vae.rs`) ports it: an f8, 16-latent-channel *video* VAE run
  image-only (`T = 1`), causal 3-D convs, Qwen RMS-norms, mid-block
  single-head attention. Weights load verbatim (one `resample.1` rename),
  proven by staged encode/decode parity vs diffusers on a checked-in tiny
  fixture (`just vae-reference`) and an opt-in real-weights proof. `encode`
  emits the **normalized** latents training consumes and M12 caches.
- **M10 — Qwen 3 VL text encoder** ([#21](https://github.com/laurigates/loractl/issues/21)).
  `Qwen3VlEncoder` (`src/qwen3vl.rs`) ports the Qwen3-VL *text* trunk (GQA
  32/8 heads, per-head QK-RMSNorm before half-split RoPE at θ=5e6, SwiGLU,
  pre-norm residuals) and loads Krea-2-Raw's own `text_encoder/` text-only (a
  `^language_model\.` filter drops the vision tower; first 35 decoder layers
  load). `Qwen3VlConditioner` adds the exact chat template + tokenizer and
  emits the conditioning stack `[b, s, 12, 2560]` + mask the MMDiT consumes.
  Proven by staged parity vs transformers on a checked-in tiny fixture
  (including a right-padded row) plus an opt-in real-weights + tokenizer-parity
  proof.
- **M11 — Krea 2 MMDiT denoiser** ([#22](https://github.com/laurigates/loractl/issues/22)).
  `Mmdit` (`src/mmdit.rs`) ports `krea-ai/krea-2`'s ~12B **single-stream**
  `SingleStreamDiT` (text + image tokens concatenated through 28 identical
  blocks): zero-centered RMSNorm, gated-sigmoid GQA attention (48/12), QK-norm,
  rotation-matrix RoPE over 3 position axes at θ=1e3, shared 6-way timestep
  modulation, the 2+2-block text-fusion transformer collapsing M10's 12-layer
  stack, and pad-to-256/masking/output-slice semantics. Proven by staged
  parity vs the official `mmdit.py` (pinned commit, `just mmdit-reference`) on
  a tiny fixture, plus an opt-in real-weights staged proof depth-truncated to
  fit a 48 GiB host. The M6 LoRA attaches across every trunk projection.
- **M12 — Image dataset pipeline** ([#23](https://github.com/laurigates/loractl/issues/23)).
  `dataset` (`src/dataset.rs`) implements the kohya/ai-toolkit convention: scan
  a folder of images + same-named `.txt` captions (missing caption =
  unconditional example), group into **aspect-ratio buckets** (every dimension
  a multiple of 16), resize cover-style + center-crop, and cache **VAE latents
  + conditioning stacks** as safetensors under `<dataset>/.loractl-cache/`,
  keyed by file name, bucket shape, and a hashed encoder fingerprint. Encoders
  are injected as closures — M14 wires the real frozen models; the offline
  tests wire mocks (and a cache-reuse test passes encoders that *panic*,
  proving warm epochs are pure tensor reads). Per-bucket batching never mixes
  shapes.
- **M13 — Single-GPU 12B fit** ([#24](https://github.com/laurigates/loractl/issues/24)).
  Two config-toggleable memory knobs, both overridable per layer:
  **`compute.precision: f16`** (wgpu only; any other backend fails loudly)
  halves resident weight memory, fitting the ~12B Krea 2 base (~49 GB f32 →
  ~24.6 GB f16) on a 48 GiB host; **`compute.grad_checkpointing: true`** swaps
  burn's `Autodiff` to `BalancedCheckpointing` — proven bit-identical to stored
  activations. Deliberately *not* built: 8-bit Adam (LoRA optimizer state is
  adapter-only, tens of MB) and — at the time — base quantization; int8/int4
  became the #24 follow-up for ≤16 GB GPUs (landed via #96/#119, below).
- **M14 — End-to-end + interop** ([#25](https://github.com/laurigates/loractl/issues/25)).
  *Code landed; the real-run interop proof is the remaining checkbox.*
  `DiffusionTrainer` (`src/diffusion_trainer.rs`) composes the whole stack as
  one `impl Trainer` behind core's two-armed `select_trainer` factory on
  `model.base`: the M12 pipeline caches M9 latents + M10 conditioning **then
  drops the encoders before the MMDiT loads** (peak memory never holds both),
  the M8 objective drives the M11 denoiser through the M6 adapter injection,
  and every checkpoint + the final artifact is a kohya-ss export. The offline
  proof composes the per-milestone tiny fixtures into a dimension-matched
  **tiny Krea 2** (`just krea2-reference`, `tests/diffusion_trainer.rs`) and
  trains it end to end through the real loading paths (events framed, `B` off
  zero, kohya key grammar pinned, reseeded warm-cache rerun bit-identical).
  Per-step loss is deliberately not asserted to decrease — fresh `(t, ε)` each
  step makes it noise-dominated by construction.
- **M15 — Train on Krea-2-Turbo** ([#82](https://github.com/laurigates/loractl/issues/82)).
  Turbo is architecturally identical to Raw — the same 430 tensor keys,
  per-tensor distillation deltas of 3–11% — so the M11 port, key remap, and M8
  objective apply unchanged (amending
  [ADR-0004](adrs/0004-krea2-image-diffusion-target.md)'s "train on Raw" 
  decision). `variant: krea2-turbo` defaults the denoiser filename to
  `turbo.safetensors`, and an optional `model.checkpoint` overrides it for any
  variant. The ComfyUI-style **scaled-fp8** repacks (`float8_e4m3fn` weights +
  f32 `weight_scale` sidecars) now load: burn-store 0.21 has no fp8 dtype, so
  `src/fp8.rs` lazily dequantizes `LUT[byte] · weight_scale` to f32 (exact
  256-entry e4m3fn LUT), auto-detected from the safetensors header so bf16/f32
  checkpoints keep the proven burn-store path. Out-of-contract files fail
  loudly. Follow-up tracked separately: a Turbo training adapter
  ([#83](https://github.com/laurigates/loractl/issues/83)). Dynamic
  timestep-shift parity ([#84](https://github.com/laurigates/loractl/issues/84))
  landed as `flow.shift_mode: resolution` — per-batch `exp(μ(gh·gw))` with Krea
  2's ai-toolkit-documented anchors (0.5@256 → 1.15@6400 image tokens),
  golden-pinned; the krea2 example configs train with it.

## Frozen-base quantization (int8/int4, #96/#119)

`compute.quant: int8` / `int4` (Q4S) load the frozen ~12.8B MMDiT base
per-block quantized (weight-only, symmetric) while adapters train in f32 — the
**QLoRA** pattern. A custom autodiff matmul dequantizes transiently per layer,
so gradients flow to the adapters, never the base. Restricted to
`(ndarray|cuda, f32)` by the trainer guard; the synthetic `BurnTrainer` rejects
the knob. Loading is streamed from an mmap'd file (bf16/f32 or auto-detected
scaled-fp8), so peak load memory is the quantized skeleton plus one transient
f32 tensor.

## Current direction — the real run (M14's remaining checkbox, #25)

Train a LoRA on `krea/Krea-2-Raw` through the landed `DiffusionTrainer` and
prove the exported adapter loads and visibly conditions generation in ComfyUI /
Krea-2-Turbo.

The cuda route was **VRAM-bound**. The #132 retention-ledger attribution
([ADR-0005](adrs/0005-int4-training-vram-bound.md) Addendum 2, PR #133)
measured the monolithic step's true logical demand at **67.9 GiB pinned per
forward** (~3× the RTX 4090) — burn-autodiff eagerly pins the whole tracked
trunk interior, topology-driven and independent of resolution/site-count. The
measured fix is **#134 — block-level gradient checkpointing**
(`src/block_ckpt.rs::checkpointed_step`): `compute.grad_checkpointing: true` on
the diffusion path runs the trunk forward graph-free storing only block inputs,
then replays each block on its own standalone graph in backward (grads
bit-identical to the monolithic path; incompatible with `lora.dropout > 0`).

**int4 (~10.1 GB reclaimed resident base) + block checkpointing is the 24 GB
training route** (estimate ≈ 16–18 GB). Verify fit with `just step-probe`
(#126) — the gate is a **zero-panic** run, never a survived OOM storm. The wgpu
f16 route (`config/examples/krea2-lora.yaml`, the 48 GiB Metal host) stays
blocked by burn's GPU autodiff bug (burn#5162, unchanged).

## A note on the text side

A smaller optional detour on the *text* side is **SmolLM2-135M** — a modern
LLaMA-style architecture (RoPE + RMSNorm + SwiGLU) that reuses M3's loader and
parity harness and would bank the RoPE-convention work
([ADR-0001](adrs/0001-first-real-target-model.md)) ahead of M11's 3D axial
RoPE — but it is not on the critical path to Krea 2.
