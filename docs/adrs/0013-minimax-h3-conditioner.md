---
id: ADR-0013
status: Proposed
date: 2026-08-09
---

# 0013 — MiniMax H3's conditioner: loractl's f32 encode path rules out the 32B, and a projected 4B is the only route that fits the box

- **Status:** Proposed
- **Date:** 2026-08-09
- **Milestones:** H3 feasibility spike, stage 1 of
  [#204](https://github.com/laurigates/loractl/issues/204)
- **Deciders:** loractl maintainers
- **Builds on:** [ADR-0004](0004-krea2-image-diffusion-target.md) (the Krea 2
  target, whose Qwen3-VL conditioner `qwen3vl.rs` implements),
  [ADR-0005](0005-int4-training-vram-bound.md) (the 24 GB envelope), and
  [ADR-0006](0006-reduced-precision-accuracy-gate.md) (why the frozen encoders
  run in f32 and nothing else).

## Context

[#204](https://github.com/laurigates/loractl/issues/204) proposes MiniMax H3 as
a second, video-domain LoRA target, and stages the work behind one gating
measurement: the resident int4 footprint of the ~33 B denoiser, with and
without an AdaLN factorisation. The text encoder sits at **stage 4** of that
plan, treated as ordinary work — "new config constructor for the 32 B variant"
— on the reasoning that `qwen3vl.rs` already implements the family.

That sequencing is wrong, and this ADR is the correction. **The conditioner is
a second, independent gate, it binds harder than the denoiser, and it binds on
system RAM rather than VRAM** — an axis neither #204 nor any third-party report
of H3 training is measuring, because it is an artifact of a deliberate loractl
design decision that other trainers do not share.

The prompt for looking was a third-party ComfyUI node
([`nicolab28/ComfyUI-ClipProj`](https://github.com/nicolab28/ComfyUI-ClipProj),
v0.1.0, MIT) that replaces H3's 32 B conditioner with the **4 B Qwen3-VL plus a
learned linear projection** into H3's 5120-dim conditioning space, calibrated
by ridge regression. For an inference user that is a convenience. For loractl
it is closer to a precondition.

### What our encode path actually costs

Three facts, read from this repo, not inferred:

1. **The encode phase always runs on the CPU ndarray backend, in f32.**
   `DiffusionTrainer` dispatches `encode_phase::<NdArray>` unconditionally
   before the training backend is selected at all
   (`crates/loractl-core/src/diffusion_trainer.rs`). The module docs give the
   reason: f16 overflows the Qwen encoders' numeric range, and burn 0.21's wgpu
   f32 kernels corrupted *sequential* encoder outputs progressively (clean →
   ~1e32 → NaN across identical calls). This is the ADR-0006 gate applied to
   the frozen half of the pipeline.
2. **A reduced-precision checkpoint does not reduce residency.** `fp8.rs` emits
   *lazily dequantizing* f32 snapshots: a ComfyUI F8_E4M3 text-encoder repack
   loads one tensor at a time and materializes each as f32, so peak is "the
   resident module + one tensor". The resident module is f32 either way. The
   checkpoint dtype buys download size and load peak — never residency.
3. **The encoder is never co-resident with the denoiser.** Conditioning is
   encoded once and cached to disk (`{stem}.{fingerprint}.cond.safetensors`);
   the training phase re-reads the cache on its own backend and its cache-miss
   closures bail rather than loading an encoder. A warm cache never loads the
   encoders at all.

Fact 3 is the good news and it disposes of the headline number every
third-party H3 report leads with. Those reports place the ~20.5 GB peak in the
*caption pass* — the 32 B encoder resident in VRAM, dropped before the base
loads. **In loractl that peak does not exist**, because the caption pass never
touches VRAM. It is also why `just bench` treats encode as its own phase.

Facts 1 and 2 are the bad news, and they are the point:

| Conditioner | Params | Resident, f32 (our path) |
|---|---|---|
| Krea 2 — `Qwen3VlConfig::krea2_4b()` | ~4 B | **~16 GB** (the figure the code itself quotes) |
| H3 — `text_encoder/`, 66.73 GB bf16 | ~33.4 B | **~133 GB** |
| H3 — if only 50 of 64 layers are read (below) | ~25.8 B | **~103 GB** |

The 4090 host has **46 GB of system RAM, ~36 GB available**. Both H3 rows are
impossible by a factor of ~3, and neither is a knob-tuning problem: batch size,
`max_length`, and checkpoint dtype all leave residency untouched. #204 already
lists RAM as a risk — citing the reference trainer's reported ~64 GB when *its*
encoder falls back to CPU — and concludes loractl "may not bind the same way"
because we stream and quantise per block on the way in. That reasoning holds
for the denoiser and fails for the encoder: streaming controls the load peak,
and the encoder's problem is the module that remains afterwards.

### The substitution, and what is actually established about it

The node's claim is that H3's conditioning can be produced by Qwen3-VL-4B plus
one learned `2560 → 5120` matrix, because the 4 B and the 32 B **share a
tokenizer** (151936 entries) — same prompt, same tokens, same positions, so a
position-wise map between hidden states is well-posed with no alignment
problem. Calibration is ridge regression over `XᵀX` / `XᵀY` accumulated across
N prompts; reported as under an hour on one 3090.

Reported quality, **third-party, one machine, v0.1.0 — recorded to be
falsified, not relied on**:

| Corpus | Tokens | Cross-prompt CKA | Test cosine |
|---|---|---|---|
| 200 prompts | 37 k | 0.95 | 0.699 |
| 2000 prompts | 289 k | 0.92 | 0.712 |

One internal consistency check does pass: the shipped matrix is 50 MB, and
`2560 × 5120` in f32 is 52.4 MB. The shape story hangs together.

The methodology is also better than the usual for this genre, and in a way this
repo should recognise: it ships **control matrices** — `W = 0` (no prompt
information) and identity (raw copy of the low 2560 dims), both normed to
within ~4 % of the learned matrix so the comparison is structural rather than
scale — with the explicit rule that *if the identity control ever looks fine,
the matrix adds nothing*. That is a kill-test in the sense
`.claude/rules/testing.md` means it, and it is the reason this is worth an ADR
rather than a dismissal.

## Decision 1 — the 32 B conditioner is out, on our path, unconditionally

Not "expensive", not "needs offload": ~133 GB f32 against 46 GB, with no lever
in the current design that moves it. Any H3 work that assumes the 32 B
conditioner is assuming a different encode path than the one loractl has.

Three ways that could change, none of them free, all of them larger than the
H3 spike they would serve:

- **Reduced-precision inference for frozen encoders.** Directly contradicts
  ADR-0006 and the wgpu corruption that motivated the CPU/f32 choice. Would
  need its own accuracy gate before it could carry an encoder at all.
- **Blockwise streaming through the encoder** (materialize layer *i*, run it
  over the whole caption batch, free it). Tractable in principle — the encoder
  is frozen, needs no Autodiff, and the cache means it runs once — but it is a
  new execution mode in `qwen3vl.rs`, and 66.73 GB still has to land on a
  volume with 101 GB free.
- **Encode off-box** and ship the cache. Defensible as a one-off; not a
  workflow loractl can claim to support.

## Decision 2 — the projected 4 B is the candidate, and it is nearly free to build

If the conditioner is a 4 B trunk plus one `Linear`, then:

- The trunk is **already implemented and parity-proven** —
  `Qwen3VlConfig::krea2_4b()` is the same model, same file, same code path, at
  the ~16 GB residency that works today.
- The download drops from #204's ~143 GB to ~77 GB, under the 101 GB free on
  the working volume. The disk blocker in #204's hardware-constraints section
  dissolves as a side effect.
- The plumbing already fits. Our conditioning cache is rank-4
  `[1, s, n, d]` because Krea 2 aggregates **12** intermediate hidden states;
  H3's `[seq, 5120]` is the `n = 1` degenerate case, which the cache handles
  shape-wise as-is. `encoder_fingerprint` already keys the cache per variant,
  so a projected variant cannot silently reuse Krea 2's entries.

What is new is a projection module, a variant constructor, and a calibration
job. That is a fraction of stage 4 as #204 currently scopes it — which also
called for a vision tower.

## Decision 3 — adopting it needs a *training*-side gate that nobody has run

This is the load-bearing caveat, and it is why the status here is Proposed.

Every result above validates **inference** substitution: same DiT, same seed,
projected conditioning, does the clip look right. Training bakes the projection
into the adapter, which is a different claim in two ways:

1. **Train/inference conditioning mismatch.** At cosine ~0.71, an adapter fit
   against projected conditioning has been calibrated on a distribution that is
   measurably not H3's. Exported to a user who runs it against the real 32 B
   encoder — the only encoder ComfyUI's stock H3 workflow loads — it is being
   asked to generalize *off* its training distribution. This is the failure
   shape `.claude/rules/testing.md` warns about in the interop section: it
   produces no error, only a worse adapter.
2. **Degraded captions are degraded supervision.** The node's own "what doesn't
   hold" is that 4 B-scale knowledge loss makes proper nouns unreliable —
   "assume any proper noun is at risk". For an inference user that is a prompt
   they must rephrase. For training it is noise injected into the supervision
   signal for every caption in the set, silently.

**The gate, before the projected conditioner is used for anything but a
spike:** two adapters, identical seed / dataset / step count, one trained
against real 32 B conditioning and one against projected, compared on
deterministic samples (ADR-0002 semantics). Nothing in any published H3 result
answers this, because no one training H3 has had a reason to ask it.

Note the ordering trap: that gate needs the 32 B encoder *once*, which Decision
1 says we cannot run. It is therefore an out-of-repo, one-time PyTorch job on
rented hardware — the same job as the ridge calibration itself, which also
needs both models resident. Doing both in one sitting is the efficient shape,
and it is the only part of H3 that needs a box we do not have.

## Decision 4 — calibrate our own, do not adopt a downloaded matrix

Per `.claude/rules/testing.md`: a contract generated from pinned upstream
source fails loud when the contract moves; a hand-copy — or here, a
third-party `.pt` fit against an unrecorded checkpoint at an unrecorded tap —
drifts silently. If a projected conditioner ships, its matrix is generated by a
`reference/` script that records the source checkpoint, the tap layer, and the
corpus, and the zero/identity controls come with it as kill-tests.

One reported result makes this cheaper than it sounds: a matrix calibrated on
bf16 weights transferred to an abliterated fp8 variant with a 0.0023 cosine
gap, i.e. the *source-side* quantization barely matters. Calibration against
ComfyUI's 15.7 GB NVFP4 distribution of the 32 B is therefore plausibly as good
as against the 66.73 GB bf16 one, which changes what hardware the one-time job
needs.

## Consequences

- **#204's staging changes.** The encoder is promoted from stage 4 to a stage-1
  gate alongside the int4 denoiser measurement. Unlike the denoiser question it
  has a cheap answer already sitting in `qwen3vl.rs`, so it can be resolved on
  paper first and does not need the 4090.
- **The third-party VRAM numbers in #204 are even less transferable than that
  section says.** They rank tools whose peak lives in a caption pass loractl
  does not have. Our encode phase costs system RAM and wall time; our peak is
  the denoiser. The two pipelines do not have a comparable high-water mark.
- **Two H3 facts are now check-worthy rather than settled.** #204's table reads
  64 layers from `config.json`, and the 66.73 GB blob confirms all 64 ship; the
  node reports H3 truncating to **50**. Both can be true — Krea 2 reads only up
  to layer 35 of the 4 B's 36, and `Qwen3VlConfig::num_layers()` already builds
  exactly `max(select_layers)`, so a 50-layer read is ~22 % off the loader for
  free. Confirm against `MiniMaxAI/MiniMax-H3` upstream source; it also tells
  us H3's tap/aggregation convention, which any projection has to target.
- **A ComfyUI bug now constrains how H3 goldens may be captured.** The node
  reports `SDClipModel.generate()` dropping `embeds_info` and never calling
  `build_image_inputs`, so image tokens land at linear positions instead of
  Qwen3-VL's 3D mRoPE, with no DeepStack injection — a path that will happily
  describe an image it never saw. Irrelevant to us today (we run text-only),
  but it would silently poison any conditioning golden captured through ComfyUI
  once #204's image-conditioning work starts. Capture from pinned upstream H3
  source, per the rule that already governs `krea2_lora_keys`.
- **Licensing is untouched.** Substituting the conditioner does not change
  H3's territorial grant: the DiT and video VAE are still MiniMax weights under
  the Community License, and #204's position — apply, keep the written
  authorization, download nothing first — stands unchanged.

## Alternatives considered

- **Keep the 32 B and add blockwise encoder streaming.** The honest fix, and
  the only one that preserves H3's real conditioning distribution end to end —
  which would make Decision 3's gate unnecessary rather than deferred. Rejected
  for now on cost and on disk: a new execution mode in `qwen3vl.rs` plus 66.73 GB
  against 101 GB free, to serve a target whose denoiser has not yet been shown
  to fit. Revisit if the denoiser measurement comes back favourable.
- **An MLP instead of a linear projection.** The node reports the linear map at
  its ceiling — 8× more calibration data bought 1.8 % of cosine. An MLP is the
  obvious next lever and would plausibly close some of the gap Decision 3
  measures. Premature: it adds a trained component to the *frozen* half of the
  pipeline, which is a real architectural change, and there is no point paying
  for it before knowing whether the linear version fails the training gate.
- **Drop the video path and train H3 stills only**, where third-party reports
  put the encoder peak and the step cost lowest. Does not help — the encoder
  residency is per-caption, not per-frame, so it is identical for stills. Also
  abandons the reason #204 exists.
- **Defer the whole conditioner question to after the denoiser spike**, as
  #204 currently sequences it. Rejected, and `step_probe`'s own module docs say
  why: the probe trains real steps, so a cold cache "pays the (slow, CPU)
  encode phase first". The gating measurement cannot be taken without a
  conditioner that runs, which puts the encoder decision upstream of it in
  practice even though it reads as downstream on the plan.

## Open questions

- What does H3 actually tap — a single hidden state, or an aggregation like
  Krea 2's 12? A projection targeting the wrong construction is untestable
  against the real thing.
- Does the AdaLN factorisation question in #204 interact with conditioning at
  all, or is it purely a denoiser-side concern? If modulation is factorised at
  load, the sites that consume conditioning may move.
- The node refuses `ref2va` and reports it untested. #204 already scopes to
  FL2VA, so this does not bind v1 — but it is a second, independent reason to
  treat Ref2VA as out of scope rather than assumed-compatible.
