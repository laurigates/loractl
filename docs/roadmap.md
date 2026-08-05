# Roadmap & milestone history

The detailed, per-milestone record for `loractl`. The [README](../README.md)
carries a compact checklist; this document is the long-form history and the
current-direction detail. Milestones are tracked as GitHub issues #1–#4,
#17–#25, plus #82; keep the two in sync when a milestone lands.

## Where the project is

Milestones M1–M15 (#1–#4, #17–#25, plus #82) have landed, **M14 included** —
its real-run interop proof closed 2026-07-23: a LoRA trained on
`krea/Krea-2-Raw` through `DiffusionTrainer` visibly conditions Krea-2-Turbo
generation in ComfyUI, on a 24 GB card (details below).

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
  `diffusion_model.blocks.N.*` form community LoRAs use. Every export also
  carries a safetensors `__metadata__` header (`src/metadata.rs`): kohya
  `ss_*` (network topology, optimizer, buckets, `ss_tag_frequency`,
  `ss_trained_words`), `modelspec.*`, and the `sshs_*` sd-webui hashes — the
  provenance ComfyUI/Forge/Civitai read back. Author-supplied fields live in a
  `metadata:` config block; everything a run knows is derived. `loractl
  inspect` prints any file's header.
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
  *Landed in full, real-run interop proof included (see below).*
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
  loudly. The Turbo training adapter
  ([#83](https://github.com/laurigates/loractl/issues/83)) **landed** as
  optional `model.training_adapter`: a LoRA `.safetensors` (diffusers/PEFT
  `lora_A`/`lora_B` or kohya `lora_down`/`lora_up`, `diffusion_model.*`-prefixed)
  merged into the frozen base *before* LoRA injection — `W += (alpha/rank)·B·A`
  per targeted base-linear site, rank auto-detected (`src/training_adapter.rs`).
  This is ai-toolkit's distillation-aware turbo recipe minus the preview
  inversion (loractl never samples during training); the trained LoRA still
  deploys on plain turbo. Golden-pinned merge math plus a producer-contract read
  test that is loud on any unmatched key. Merge-at-load needs a full-precision
  base, so it is **rejected with `compute.quant`** — merging into
  `load_quant_module`'s f32 transient is the remaining follow-up. Dynamic
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

## M14 landed in full — the real run (#25, closed 2026-07-23)

A LoRA trained on `krea/Krea-2-Raw` through `DiffusionTrainer` now demonstrably
conditions Krea-2-Turbo generation in ComfyUI: a 300-step "sks dog" DreamBooth
run at 512px (`config/examples/krea2-dog.yaml`), whose kohya export applies
**with no key conversion** (the A/B is `docs/evidence/m14-krea2-dog-interop.jpg`).

Getting there meant solving a VRAM wall. The #132 retention-ledger attribution
([ADR-0005](adrs/0005-int4-training-vram-bound.md) Addendum 2, PR #133)
measured the monolithic step's true logical demand at **67.9 GiB pinned per
forward** (~3× the RTX 4090) — burn-autodiff eagerly pins the whole tracked
trunk interior, topology-driven: independent of trained-site count and LoRA
rank, and scaling with sequence (67.9 GiB at seq 1536 / 512px vs 51.7 GiB at
seq 1280 / 384px), so resolution was a *non-lever* for the monolithic step
rather than irrelevant — even 384px demand was >2x the card (ADR-0005
Addendum 2 §Corrections item 1). The fix was **#134 — block-level gradient
checkpointing** (`src/block_ckpt.rs::checkpointed_step`):
`compute.grad_checkpointing: true` on
the diffusion path runs the trunk forward graph-free storing only block inputs,
then replays each block on its own standalone graph in backward (grads
bit-identical to the monolithic path; incompatible with `lora.dropout > 0`; a
nested-backward custom op is impossible on burn 0.21 — verified deadlock, now
fixed upstream for 0.22 by burn#5194).

**int4 (~10.1 GiB reclaimed resident base) + block checkpointing is the 24 GB
training route**, measured: a zero-panic `just step-probe` (#126) at 512px int4
peaks at **19.4 GB** — 3/3 steps, 196/196 sites, ~4 GB headroom (ADR-0005
Addendum 3). The gate is always a **zero-panic** run, never a survived OOM
storm. The wgpu f16 route (`config/examples/krea2-lora.yaml`, the 48 GiB Metal
host) stays blocked by burn's GPU autodiff bug (burn#5162, unchanged).

Open from here: step **throughput** is unmeasured on real hardware — the extra
per-block forward costs something, and the #110 harness that can now price it
(`just bench`, `crates/loractl-bench` + `loractl-core::bench`) has only been
exercised on the offline fixture; the number needs a GPU dispatch. int4's
dequant error vs adapter quality is a separate question from fit (now tracked
as #159) — the fit-vs-quality separation and the adapter-parameterization menu
behind it are [ADR-0007](adrs/0007-adapter-algorithm-strategy.md). The
dataset-pipeline ergonomics issues (#147–#149) are the next user-facing gap. The next *memory* lever — offloading the #134 block-boundary
activations to host RAM (#158) — is now scoped and priced by
[ADR-0008](adrs/0008-host-offload-mechanism-and-scope.md): explicit scheduled
transfer rather than demand paging, worth ~1.06 GB of the 19.4 GB peak at 512px
(~3.17 GB at 1024px; batch-1, derived), and blocked on that dispatch because it
is the first lever that spends throughput to buy VRAM.

The *throughput* levers are triaged ahead of that dispatch by
[ADR-0010](adrs/0010-rtx4090-throughput-lever-triage.md), which records which of
them are already shipped (M12's latent cache; cubecl's own pinned H2D staging;
zune-jpeg via `image` 0.25), which are dead at loractl's layer, and the two that
are real: **`fusion` is compiled off on the CUDA path** (burn declares burn-cuda
`default-features = false`, so `burn::backend::Cuda` is the raw `CubeBackend`,
not `Fusion<CubeBackend>`), and the quantized load walks its sites strictly
serially with a single-threaded CPU dequant. Both are gated on the same
dispatch. It also records a *fit* finding the memory work has not accounted for:
`prepare_dataset` holds every example's conditioning on the device for the whole
run, at a fixed `[1, 512, 12, 2560]` f32 = 60 MiB per example, so VRAM scales
with dataset size against ADR-0005's ~4 GB of headroom.

Sharing ComfyUI's *resident* VRAM — importing its already-loaded Krea 2 weights
over CUDA IPC so training and generation alias one copy — is closed by
[ADR-0011](adrs/0011-comfyui-cross-process-vram-sharing.md). cubecl 0.10's
`ComputeStorage` admits memory only through `alloc(size)`, and `GpuStorage`'s
pointer map is private with no non-allocating insertion, so there is nothing to
hand a foreign pointer to; had there been, `perform_deallocations` would free it
out from under ComfyUI. The budget also fails independently: the only shareable
copy is ComfyUI's ~13.5 GB int8 (or ~13.1 GB scaled-fp8) base, *larger* than the
~10.1 GB int4 base loractl already uses. Co-tenancy stays a time-division
problem. Reading ComfyUI's model **directory** was never blocked and remains
shipped (`config/examples/krea2-comfyui.yaml`).

That ADR's Decision 5 records a separate, closer-to-home gap it turned up: the
ComfyUI **int8** repack (`krea2_turbo_int8_convrot.safetensors` — `I8` weights +
`weight_scale` + `comfy_quant`) carries no `F8_E4M3`, so `is_fp8_checkpoint` is
false and it routes to the plain loader, which never applies the scales — raw
`I8` values cast to f32, `weight_scale` keys unused, **no error**. Supported
denoiser inputs remain bf16/f32 or a scaled-**fp8** repack.

## First measured throughput on the 4090 (#110, 2026-08-05)

Every memory lever so far — int4, block-level gradient checkpointing — was
adopted on a memory argument with **no throughput price attached**, which is
what [ADR-0010](adrs/0010-rtx4090-throughput-lever-triage.md) means by calling
its own premise unmeasured. The first valid `just bench` dispatch
([run 30982204201](https://github.com/laurigates/loractl/actions/runs/30982204201))
closes that gap:

| | |
|---|---|
| median step | **4482.18 ms** (~4.5 s) |
| peak VRAM | **19,691 MiB** |
| derived | `tok_s=342.6901`, `tflops=26.4414` |
| validity | `sanity=ok`, `plausible=true`, `x2_ratio=2.002`, `steps_counted=6` |

Configuration: RTX 4090 (24 GB), cuda + `precision: f32` + `quant: int4` +
`grad_checkpointing: true`, batch 1, 512px, `--seq-len 1536` declared. The
denoiser was a ComfyUI **scaled-fp8** repack of `krea/Krea-2-Raw`, which
loractl quantizes to int4 on device (the supported input forms are recorded in
[ADR-0011](adrs/0011-comfyui-cross-process-vram-sharing.md) Decision 5).

**`vram_peak_mib=19691` independently corroborates
[ADR-0005](adrs/0005-int4-training-vram-bound.md) Addendum 3's ~19.4 GB** for
this configuration — a different tool, a different day, within ~1%. That is the
strongest thing about this measurement, and it is worth more than the step time
itself: two instruments agreeing on the peak is what makes the 24 GB route a
measured fact rather than a single observation.

### What these numbers do and do not license

- **`tok_s` and `tflops` are quotients of the run's own `MODEL` line and must
  never be quoted without it.** That line's
  `excludes=text_fusion,modulation,norms,rope,softmax,patch_embed,lora_delta`
  makes both an **under**-count, and `seq_len_source=declared` means 1536 was
  supplied rather than measured.
- **They are not comparable to other LoRA trainers.** The FLOP count is a
  loractl accounting choice, so a cross-tool comparison would be dividing by two
  different denominators. Only `ms=` and `vram_peak_mib=` are measured
  quantities, and those are comparable solely against a run matching on base
  model, quantization, checkpointing, batch, resolution, sequence length, and
  GPU. A step time is a property of model + config + hardware, not of the
  trainer.
- **The timed window excludes encoding.** The dataset's cache was warm, so text
  and VAE encode sit outside it. Trainers differ on whether their published
  `s/it` includes that, which is one more reason the figure does not travel.
- **This is one run over 6 counted steps on a 5-image dataset.** Adequate for
  step timing, which is driven by resolution and bucket shape rather than corpus
  size; not a repeated-trial benchmark, and not a statement about adapter
  quality ([#159](https://github.com/laurigates/loractl/issues/159) is that
  question).

The decision record for the timing mechanism itself is
[#162](https://github.com/laurigates/loractl/issues/162) (ADR-0009).

Two failure modes had to be fixed before a number existed at all, both of which
produced a *plausible-looking wrong answer* rather than an error: a contended
GPU (ComfyUI holding 17.6 GB of the card while idle) surfaced as
`non-finite loss (NaN) at step 1` on an f32 config, and a full runner disk
surfaced as `ld terminated with signal 7 [Bus error]`. Both are now preflighted
in `gpu.yml`.

## A note on the text side

A smaller optional detour on the *text* side is **SmolLM2-135M** — a modern
LLaMA-style architecture (RoPE + RMSNorm + SwiGLU) that reuses M3's loader and
parity harness and would bank the RoPE-convention work
([ADR-0001](adrs/0001-first-real-target-model.md)) ahead of M11's 3D axial
RoPE — but it is not on the critical path to Krea 2.
