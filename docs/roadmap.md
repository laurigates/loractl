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
  emits the conditioning stack `[b, max_length, 12, 2560]` (512 for Krea 2) +
  mask the MMDiT consumes — a length the conditioner now guarantees by
  deriving the template's token lengths from the loaded tokenizer
  ([#163](https://github.com/laurigates/loractl/issues/163)).
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
  shapes. Two opt-in knobs were added later:
  **`dataset.no_upscale`** ([#147](https://github.com/laurigates/loractl/issues/147)),
  which gives an image smaller than its bucket a smaller aligned box of the
  same aspect instead of Lanczos-inventing detail, and
  **`dataset.bucketing: grid`** ([#148](https://github.com/laurigates/loractl/issues/148)),
  a kohya-style symmetric 2-D grid that caps the worst-case crop loss near 4%
  across the whole 0.25–4.0 aspect range (against 55% for the fixed
  seven-ratio set) at the cost of many more buckets — and therefore many more
  partial batches, since batches never mix. Both default off; the fixed set
  remains slightly *better* at the seven ratios it was hand-picked for.
  The host-side decode/resize also became parallel
  ([#178](https://github.com/laurigates/loractl/issues/178)) — rayon across
  examples in a bounded window, with the GPU-bound encode still strictly
  serial and the decoded buffer bit-identical to the serial path it replaced
  (cache keys are name/bucket/fingerprint and never content, so a shifted
  value would invalidate nothing). The cold-cache encode-phase wall time that
  buys is **not** claimed here: it needs a real dataset on a real machine.
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
behind it are [ADR-0007](adrs/0007-adapter-algorithm-strategy.md). #147
(`dataset.no_upscale`) and #148 (`dataset.bucketing: grid`) have landed as
opt-in `DatasetConfig` knobs (see M12 above), leaving **#149** as the
remaining dataset-pipeline ergonomics gap. The next *memory* lever — offloading the #134 block-boundary
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
dispatch. It also recorded a *fit* finding the memory work had not accounted
for — `prepare_dataset` held every example's conditioning on the device for the
whole run, at a fixed `[1, 512, 12, 2560]` f32 = 60 MiB per example, so VRAM
scaled with **dataset size** against ADR-0005's ~4 GB of headroom. **#175 closed
it**: `PreparedDataset` is now a backend-parameter-free *plan* over the on-disk
cache and the step loop materializes one batch at a time, so residency is
O(batch). The structural claim is proven offline by `tests/dataset_residency.rs`
(a counting global allocator over the ndarray backend, where device memory is
heap memory) and, at the call site, by
`diffusion_trainer.rs::the_trainer_reads_the_cache_inside_the_step_loop` —
deleting the latents after one full epoch must fail the next step, which a
trainer that hoisted every batch out of the loop would survive. Two numbers are
**not** claimed: the peak-VRAM figure needs a `gpu.yml` dispatch of
`just step-probe` against a ≥50-example dataset (the 4-image `dataset-tiny`
fixture cannot stand in for it), and the per-step cost this trades for it — one
batch's read, f32 decode and upload, of which only the read is served by the
page cache — needs a `just bench` dispatch.

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

## Caption-template lengths are derived, not transcribed (#163, 2026-08-07)

M10's conditioner wrapped every caption in Krea 2's chat template and then
sliced the template back off using two **hardcoded** lengths — 34 prefix
tokens, 5 suffix tokens — transcribed from the reference encoder and checked by
nothing. `Qwen3VlConditioner::new` now encodes `PROMPT_PREFIX` and
`PROMPT_SUFFIX` through the tokenizer it has just loaded and keeps the result
(`crates/loractl-core/src/qwen3vl.rs`), so the body budget and the slice offset
are the *same* derived value rather than two literals that can disagree.

- **Both failure modes become unrepresentable, not merely unlikely.** The
  emitted length is `max_length` by construction (`s = max_length +
  prefix_len`), and the slice offset is the value that budget was computed
  from, so a right-shaped-but-shifted stack cannot be produced.
  `tests/qwen3vl_template_length.rs` pins both over two tokenizers ×
  `max_length ∈ {8, 16, 33}`. Five kill-tests were run and reverted; the one
  that matters is a **shape-preserving** off-by-one, which every `dims()` check
  passes and only the mask-content assertions catch.
- **The offline path was already wrong, not merely unprotected.** Measured, not
  inferred: the checked-in tiny-krea2 stub tokenizer makes **36/7** of the same
  template, so every offline tiny run had been feeding the MMDiT two leftover
  template tokens ahead of the caption. The tiny path's conditioning length
  moves 18 → 16 and its loss values move with it; nothing pinned the old
  numbers, and they should not be restored.
- **The 34/5 claim is still not verifiable offline, deliberately.** Checking the
  numbers needs the real Qwen BPE vocabulary; the always-run tests pin the
  *invariant* instead. The opt-in `tests/qwen3vl_real.rs` asserts the derived
  pair equals both the golden's and the documented `(34, 5)`, and
  `reference/qwen3vl_reference.py` derives them from the real tokenizer and
  hard-asserts them before writing the golden — a transcribed pair drifts
  silently, a derived one fails loud. **That golden has to be regenerated
  before this is proven** (`just qwen3vl-real-reference && just
  test-qwen3vl-real`, network + GPU box): it gained `prefix_len`/`suffix_len`
  fields the old file does not carry.
- **The encoder cache fingerprint was bumped** (`enc32` → `enc32-t2`,
  `diffusion_trainer.rs::encoder_fingerprint`) because the change provably
  moves the conditioning the tiny path produces while the cache reader
  validated only tensor *rank* — a warm pre-#163 `.loractl-cache/` would have
  been accepted and trained on the old alignment with no error and no shape
  mismatch. Belt-and-braces beside it: `src/cache_guard.rs` refuses any cached
  stack whose length is not the variant's `max_length`, which is also the only
  thing that catches a half-warm cache mixing two alignments inside one run.
  Both `prepare_dataset` call sites are wired to it; `dataset.rs` is untouched.

The residual assumption, stated rather than fixed: the slice lands on the first
caption token only if `encode(prefix ++ caption)` begins with `encode(prefix)`.
Nothing enforces that — `tokenize` encodes the joint string while `prefix_len`
is measured on the prefix alone. It holds for every tokenizer in the repo
(captions are trimmed and `PROMPT_PREFIX` ends in a newline that closes the
pre-tokenizer), and it is recorded at the two places a reader would ask.

## The measured fit envelope is advisory, not enforced (#179, 2026-08-07)

Everything this repo says about a Krea 2 step *fitting* says it about one
point: 512px, int4, block-level gradient checkpointing, 24 GB RTX 4090, 19.4 GB
peak (ADR-0005 Addendum 3, corroborated to ~1% by #110's `vram_peak_mib=19691`
above). Every other resolution is arithmetic. Nothing checked a config against
that, so a `resolution: 1024` edit — one line, no flag — bought a 3× trunk
sequence and found out at OOM time, minutes into an encode phase.

`crates/loractl-core/src/envelope.rs` is the check, and it is **advisory, never
fatal**: a large resolution is legal, merely unmeasured. It emits one
`TrainEvent::Warning` — no new variant, so no wire-contract change — naming
what changes and by how much, then handing over to `just step-probe`.

- **It says what moved, never a predicted peak.** For 1024px + int4 it names
  the trunk sequence at both resolutions (4608 vs 1536), the retained
  block-input set at both (~1.06 GB → ~3.17 GB at batch 1, reproducing
  ADR-0008's table), labels them **derived**, and quotes the 19.4 GB as an
  anchor with its two still-open ambiguities (GB-vs-GiB, total-vs-above-
  baseline) attached. Restating an inference as a measurement is the thing
  ADR-0005's provenance sweep exists to prevent.
- **The second half is dataset residency**, the fit finding ADR-0010 recorded
  and the memory work had not accounted for: `prepare_dataset` keeps every
  example's conditioning device-resident for the whole run at a fixed 60 MiB
  (`[1, 512, 12, 2560]` f32 — derived from `select_layers`' 12 entries ×
  2560 × 512 × 4 B, exactly ADR-0010 ledger #5), so 49 images pin 2.87 GiB
  against ~4 GB of headroom.
- **The token derivation is now shared, not copied.** `mmdit::token_geometry`
  plus `mmdit::SEQUENCE_PAD` are the single home of `resolution / compression /
  patch` and the pad-to-256; `bench::StepWork::for_config` was switched onto
  it, and the bench's own three pre-existing tests are the behaviour-preserving
  proof (they fail when `token_geometry` is sabotaged). `token_geometry`
  deliberately has **no `max(1)` clamp** — the bench relies on seeing a derived
  `0` to refuse a degenerate denominator rather than print `tok_s=0.0000` as if
  it were a measurement.
- **It is delivered before the phase it warns about.** The advisory is emitted
  ahead of the first `encode` phase event, and so ahead of the ~16 GB text
  encoder, the VAE, every image decode, the MMDiT load and step 1
  (`diffusion_trainer.rs::encode_phase`, pinned by
  `the_fit_advisory_reaches_the_sink_before_the_encode_phase`). It costs one
  extra dataset scan — image headers and caption files, no decode — against a
  phase that costs minutes per cache miss.
- **Not noise.** One message, never a list. It is gated on the real Krea 2
  variants (the tiny fixture never fires), on `quant != none`, on `> 512` and
  not `>=`, and the residency half additionally on a non-`ndarray` backend. A
  test enumerates `config/examples/*.yaml` from the directory — rather than a
  hand-picked quiet subset — and asserts every shipped 512px example stays
  silent. There is deliberately **no knob to silence it**: a switch on an
  advisory is an invitation to turn it off.

Consciously not done: the eight lines of emit glue are exercised by one
integration test on the real `encode_phase`, but the pure function under it
carries the rest of the coverage, and 12 kill-tests were run and reverted
against it. `RESIDENCY_ADVISORY_MIB = 2048` is a **judgement call, not a
measurement** — half the ~4 GB headroom read as GiB, tripping at ~35 examples
against ADR-0010's ~65-example exhaustion — and it is pinned in both directions
so it cannot drift unnoticed.

## Explicit, lossless resume (#180, 2026-08-07)

Resume on the diffusion path used to be implicit and lossy: the only trigger was
"the final artifact happens to exist in `output.dir`", the step counter restarted
at 1, and AdamW re-warmed its moments from zero. The decisions below — and the
alternatives they were chosen over — are
[ADR-0012](adrs/0012-diffusion-resume-semantics.md).

- **The trigger is named.** `resume: { from, auto, allow_unfinished }`
  (`crates/loractl-core/src/config.rs`), with `--resume` / `--no-resume` /
  `--resume-unfinished` on the CLI. `auto: true` is the documented default, so
  no existing workflow changes shape.
- **`steps` is a total, not a remainder.** An artifact recording 200 steps
  resumed with `steps: 500` runs 201..=500 — which matches what
  `ss_max_train_steps` already meant, and what `Started.total_steps` (emitted
  before the resume source is opened) already reported. Re-running an
  already-complete config therefore trains **zero** further steps.
- **Provenance is read, not assumed.** The `__metadata__` header this repo
  already writes has three states, not two: finished (`ss_training_finished_at`
  present), unfinished (every mid-run checkpoint, by construction), and
  unrecorded (`metadata.embed: false`, or a third-party file). The `auto` path
  refuses an unfinished artifact by name — the final export is only ever written
  on a completed run, so one there means a checkpoint was copied into place —
  and `resume.allow_unfinished` is the documented way through. `resume.from` is
  explicit and never subject to that check.
- **Optimizer state round-trips through a sidecar** (`{name}.optim.safetensors`,
  `crates/loractl-core/src/resume.rs`), keyed by **site path** rather than
  `ParamId` — burn's ids are random per construction, so a `ParamId`-keyed dump
  would silently restore nothing in the next process. The kohya artifact is
  byte-shape unchanged; `tests/krea2_lora_keys.rs` and `tests/adapter_export.rs`
  are untouched.
- **The sidecar is checked against the source's provenance**, not merely found
  beside it: `save_optimizer_state` records `loractl_steps_done` and the adapter
  records `ss_steps`, and a disagreement means the sidecar belongs to another
  run (the documented recovery workflow — copy `checkpoint-N.safetensors` over
  the final export — produces exactly that). Shapes match in that case, so
  nothing else can see it. It is skipped and named in the resume `Warning`
  rather than restored, since re-warming AdamW is survivable and erroring would
  stop a recovery dead.
- **Both halves of a resume are checked for finiteness, not just the weights.**
  `import_adapters` refuses a non-finite factor by tensor name; the sidecar read
  refuses a non-finite *moment* by key, and says "sidecar". The asymmetry
  mattered: a NaN moment restores cleanly and kills the run one step later
  inside `check_step_loss`, whose message blames f16 range unconditionally —
  so without the guard the operator is sent to `compute.precision` instead of to
  the file. `save_optimizer_state` writes in place with no temp-and-rename, so a
  truncated sidecar is reachable rather than hypothetical.
- **Not restored: the RNG stream.** burn 0.21 exposes no save/restore, so a
  resumed run is a continuation and not a bit-identical replay. The single
  resume `Warning` says so.
- **The synthetic/`BurnTrainer` path has no resume, deliberately** — see the
  comment in its training loop for the three reasons, pinned by
  `tests/resume.rs::the_synthetic_trainer_does_not_resume`.

## A note on the text side

A smaller optional detour on the *text* side is **SmolLM2-135M** — a modern
LLaMA-style architecture (RoPE + RMSNorm + SwiGLU) that reuses M3's loader and
parity harness and would bank the RoPE-convention work
([ADR-0001](adrs/0001-first-real-target-model.md)) ahead of M11's 3D axial
RoPE — but it is not on the critical path to Krea 2.
