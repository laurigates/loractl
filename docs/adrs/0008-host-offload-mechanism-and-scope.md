---
id: ADR-0008
status: Accepted
date: 2026-07-26
---

# 0008 — Host-RAM offload: explicit scheduled transfer, not demand paging; and why the lever stays reserved

- **Status:** Accepted
- **Date:** 2026-07-26
- **Milestones:** post-M15 memory follow-ups
  ([#158](https://github.com/laurigates/loractl/issues/158) the offload lever,
  [#110](https://github.com/laurigates/loractl/issues/110) the bench harness that
  gates it)
- **Deciders:** loractl maintainers
- **Builds on:** [ADR-0005](0005-int4-training-vram-bound.md) (the VRAM
  reclassification, its Addendum 2 retention attribution and Addendum 3 measured
  fit) and [ADR-0007](0007-adapter-algorithm-strategy.md) (the lever taxonomy,
  whose Decision 3 names this lever and leaves its mechanism unspecified)

## Context

[ADR-0007](0007-adapter-algorithm-strategy.md) Decision 3 names "CPU activation
offload paging the #134 per-block boundary tensors to pinned host RAM" as the
top *unlanded* activation lever, then reserves it. What it does not do — and
what nothing else in the repo does either — is say **how** such an offload would
work, **how much** it would actually reclaim, or **what it would cost**. Before
this ADR the repo contained zero discussion of DeepSpeed, ZeRO, unified memory,
or optimizer-state paging; the only offload content was the *layering* rule from
ADR-0005 (cubecl provides buffers, burn owns activations, loractl owns the model
and loop).

That is a real hole, because an offload lever is the first memory lever loractl
would take that **spends throughput to buy VRAM**. Every lever landed so far —
int8/int4 base quantization (#96/#119), block-level gradient checkpointing
(#134) — was adopted on a memory argument alone. This one cannot be.

The prompt for writing it now was an external prior-art review: an
r/StableDiffusion thread in which the author of a 12 GB SDXL full-fine-tune
trainer describes replacing bitsandbytes' *paged* AdamW-8bit with a direct
PCIe-offload optimizer, because paging produced "stalls, freezes, and occasional
crashes". Most of that thread is ground loractl has already measured or already
declined — the dedupe ledger below records every claim and its disposition, so
the source does not need re-mining — but the paging-versus-explicit-transfer
distinction is genuinely new here, and it is the one decision worth fixing
before anyone writes offload code.

Two tracking gaps surfaced while writing this and are now closed:
ADR-0007 reserved the lever "under #96", but **#96 is closed**, so the
reservation tracked nothing (now [#158](https://github.com/laurigates/loractl/issues/158));
and the int4-quality question, called open in six documents, had no issue at
all (now [#159](https://github.com/laurigates/loractl/issues/159)).

### What the lever is actually worth (derived)

`checkpointed_step`'s capture phase retains exactly `layers × [b, seq,
features]` block-input residual streams
(`crates/loractl-core/src/block_ckpt.rs:13-16`). For `MmditConfig::krea2()`
(`crates/loractl-core/src/mmdit.rs:170-186` — `layers: 28`, `features: 6144`) at
batch 1, f32:

Sequence length is **not** proportional to pixel area: the trunk concatenates
text and image tokens and zero-pads the result to a multiple of 256
(`crates/loractl-core/src/mmdit.rs:1388-1399`), and the text side is a fixed
512-token block — `tokenize` truncates *and* right-pads every caption to the
same `body_len` (`crates/loractl-core/src/qwen3vl.rs:611-618`), and
`variant_configs` passes `512` for Krea 2
(`crates/loractl-core/src/diffusion_trainer.rs:121`), so the text contribution
is exactly 512 for any caption, not merely capped at it. That makes the
decomposition below exact rather than an upper bound, and the constant text
block must not be scaled with resolution. The 384px and 512px rows reproduce
ADR-0005 Addendum 2's two stated anchors exactly, which is what makes the
1024px row trustworthy:

| resolution | latent (f8) | image tokens (patch 2) | + 512 text | pad → 256 | retained block inputs |
|---|---|---|---|---|---|
| 384px | 48² | 576 | 1088 | **1280** (ADR-0005 anchor ✓) | 28 × 1280 × 6144 × 4 B = **0.881 GB** (0.820 GiB) |
| 512px | 64² | 1024 | 1536 | **1536** (ADR-0005 anchor ✓) | 28 × 1536 × 6144 × 4 B = **1.057 GB** (0.984 GiB) |
| 1024px | 128² | 4096 | 4608 | **4608** | 28 × 4608 × 6144 × 4 B = **3.171 GB** (2.953 GiB) |

The whole set is co-resident at the peak: the reverse sweep drains it last→first,
so all 28 are live when the sweep begins, and the peak is one block interior on
top of that.

Two bounds on reading these figures as "what the lever is worth":

- **They are an upper bound on the reclaim, not the reclaim.** Decision 1's
  mechanism prefetches the next block's input during the current block's replay,
  so a prefetch depth of `d` leaves `d` block inputs device-resident at all
  times. At the minimum useful `d = 2` the net reclaim is `26/28` of the set —
  ~0.98 GB at 512px. Prefetch depth is a real tuning knob, and a deeper pipeline
  buys overlap by giving back reclaim.
- **They are batch-1 figures and scale linearly with batch.** At batch 4 the
  512px set is 4× the batch-1 figure (~4.2 GB). The peak scales too, so the
  *ratio* that Decision 2 argues from is more stable than either number — but
  the ratio is what the decision rests on, so it should be re-derived, not
  assumed, at any other batch size.

**These are derived figures — computed from the config and the capture
contract, not measured.** The one measured anchor is ADR-0005 Addendum 3's
19.4 GB peak (zero-panic, 3/3 steps, 196/196 sites, 512px int4), and that
figure carries its own recorded unit-basis caveat (GB vs GiB unresolved,
total-vs-above-baseline not stated). Against it, ~1.06 GB is roughly **5%** of
the peak with ~4 GB of headroom already in hand.

This ADR is written this way deliberately: ADR-0005 carries an entire withdrawal
sweep because inferred figures were once stated as measured.

## Decision

1. **If and when the block-boundary activation offload is taken, the mechanism
   is explicit scheduled transfer over pinned host buffers — never CUDA unified
   memory / demand paging.**

   The #134 capture/sweep structure produces a **statically known schedule**:
   the capture phase knows each block input the moment it is produced, and the
   reverse sweep knows exactly which one it needs next, one block ahead. That is
   precisely the property that makes explicit async copy viable *and* makes
   demand paging unnecessary — there is nothing to discover at fault time that
   the schedule does not already know.

   The supporting facts are upstream's own, not inference. bitsandbytes
   documents that its paged optimizers are "built on top of the unified memory
   feature of CUDA", that paging "only becomes active if you run out of GPU
   memory", and — the load-bearing part — that "the unified memory feature is
   less efficient than regular asynchronous memory transfers, and you usually
   won't be able to get full PCIe memory bandwidth utilization … still only
   about half or worse than the full PCIe memory bandwidth (tested on 16x lanes
   PCIe 3.0)". Demand paging also cannot overlap transfer with compute, because
   it faults *on access*; an explicit schedule can prefetch the next block's
   input during the current block's replay.

   This is also consistent with loractl's existing layering: cubecl's pinned
   pool is already "a transfer-staging buffer, not a GPU spill target"
   (ADR-0005 §Decision 2). Explicit transfer uses that pool as designed;
   demand paging would want a spill target that does not exist.

2. **The lever stays reserved, now with a number rather than an assertion.**
   ~1.06 GB of a 19.4 GB peak at 512px — less after prefetch retention — does
   not justify the complexity or the throughput cost while ~4 GB of headroom
   exists. It becomes a genuine candidate at **≥1024px** (~3.17 GB) or on
   **≤16 GB cards**. Both figures are batch-1; re-derive the ratio rather than
   reusing it if batch size changes. Tracked as
   [#158](https://github.com/laurigates/loractl/issues/158).

3. **It cannot merge before [#110](https://github.com/laurigates/loractl/issues/110)
   is finished.** This lever trades VRAM for PCIe time, and loractl has never
   timed a training step. #110 is therefore a **prerequisite**, not an adjacent
   nice-to-have — the first lever for which that is true.

   Be precise about what #110 is missing, because it is half-landed and easy to
   misread in both directions. The reusable core **exists**:
   `crates/loractl-bench` (a workspace member) carries the `RESULT`/`SANITY`
   line schema, the device-resident wall-sync timer that works around cubecl's
   broken `profile` window (cubecl#1421), the 2×-iters dead-graph ratio, and the
   `plausible()` guard. What does not exist is everything that would make it
   produce a number: **nothing in the workspace depends on it** — no
   burn-`Tensor`/`Autodiff` training-step adapter in `loractl-core`, no VRAM
   read-out, no `just bench` recipe. The probes loractl does run
   (`just step-probe`, `just quant-probe`, `just ledger-probe`) sample **memory
   only**. (ADR-0005 Addendum 3's "loractl has no benchmark harness" was true
   when written and is now imprecise — the core landed, the driver did not.)

   Order of magnitude for the round trip (~2.11 GB per step at 512px, derived):
   ~85 ms on PCIe 4.0 x16 at ~25 GB/s effective, ~176 ms on PCIe 3.0 x16 at
   ~12 GB/s — against a step time nobody has measured. bitsandbytes' own worked
   example lands in the same range from the other direction: ~1 GB evicted per
   loop under UVM on PCIe 3.0 x16 costs "125ms of overhead per optimizer step".

4. **Optimizer-state offload stays declined, and the reason is now a
   discriminator rather than an assertion.** loractl's optimizer state is
   adapter-only — tens of MB — so moving it anywhere saves nothing
   (`docs/roadmap.md`, ADR-0007 Decision 3). The external prior art does not
   contradict this; it *explains* it. A full fine-tune of a ~2.6B UNet carries
   gradients and two fp32 Adam moments **per trained parameter**, which is the
   dominant term in that regime and the reason CPU-resident optimizer state is
   load-bearing there. loractl trains ~10⁻³ of that. Same technique, different
   regime, opposite verdict.

5. **Layer/site targeting is re-affirmed as a non-lever for loractl, and the
   external look-alike is recorded as a trap.** ADR-0005 Addendum 1 measured it:
   a single trained site peaks and fails identically to all 196, because one
   adapter early in the graph makes the whole downstream trunk tracked —
   retention is topology-driven. `lora.targets` is a scope/quality choice only.

   The trap is that a full-fine-tune trainer's "layer targeting" is the *same
   words for a different mechanism*: excluding a layer there removes that
   layer's gradients and optimizer moments, which is weight-class memory
   proportional to trained-parameter count. Excluding sites here removes adapter
   optimizer state, which is a rounding error. Anyone importing that result into
   loractl would be importing the arithmetic of a regime we are not in.

6. **int8 as a *throughput* axis is logged as unexplored, and is a different
   axis from int8/int4 as a *memory* axis (landed, #96/#119).** Reduced-precision
   *compute* — int8 tensor-core matmul — would live in burn/cubecl kernels, not
   in loractl, and like Decision 3 it cannot be evaluated here until #110
   exists. It is recorded so the two axes stop being conflated, not scheduled.

## Consequences

- **Rejected: CUDA unified memory / demand paging as loractl's offload
  mechanism**, on upstream's documented efficiency cost and on the
  overlap argument, with the honest counterpoint recorded under Alternatives.
- **Rejected: adopting the external trainer's headline lever (layer targeting)
  as a fit lever**, per Decision 5. It is already measured false here.
- **Rejected: framing any offload as a pure win.** Every previous loractl memory
  lever was free at the memory-argument level; this one is not, and the ADR
  refuses to present it as though it were.
- **#110 is promoted from "missing measurement" to a blocking prerequisite** for
  #158, and is now argued for by two independent lines (the unmeasured #134
  recompute cost, and this lever's transfer cost).
- **Two untracked questions now have issues** —
  [#158](https://github.com/laurigates/loractl/issues/158) (the lever, replacing
  the dead "#96" reservation) and
  [#159](https://github.com/laurigates/loractl/issues/159) (int4 dequant error
  vs adapter quality, previously called open in **six** documents — `CLAUDE.md`,
  the cubecl rule, `docs/roadmap.md`, ADR-0005, ADR-0007 and the LoKr PRD — and
  tracked in none; all six now carry the pointer).
- **No code lands with this ADR.** There is nothing to implement until #110
  exists (Decision 3); shipping an implementation anyway would contradict the
  document.
- **Open follow-up (unchanged):** chunked attention inside the recomputed block
  remains the reserved follow-on above this one (ADR-0005 Addendum 3), and
  encoder unload after latent/conditioning caching remains the cheap fixed win.
  Neither is re-decided here.

## The source and its dedupe ledger

The external source is a single r/StableDiffusion thread (post `1v6ej0k`,
2026-07-24) announcing **Aozora**, an Apache-2.0 GUI trainer for SDXL/Anima
*full fine-tuning* on 12 GB consumer GPUs. It is recorded in full so it does not
need re-reading; **claims are labelled VERIFIED (a primary source was read) or
UNVERIFIED (author's claim only)**.

| # | Claim | Disposition for loractl |
|---|---|---|
| 1 | Paged AdamW-8bit gave the VRAM savings but caused stalls/freezes/crashes; replaced by direct PCIe offload ("Raven"), keeping ~95% of the savings (UNVERIFIED — single host, no methodology). Raven "stores optimizer state on the CPU and uses shared GPU buffers for FP32 update math"; a second mode ("Titan") offloads gradients too (VERIFIED — repo README) | **The one transferable insight.** Grounds Decision 1. The mechanism claim is corroborated independently by bitsandbytes' own docs (VERIFIED), which is what the decision actually rests on |
| 2 | Offloading is ultimately bounded by motherboard PCIe bandwidth, "just as with DeepSpeed" (UNVERIFIED as stated; ZeRO-Offload's abstract does state it is "designed to minimize the data movement to/from GPU", VERIFIED) | **New — the cost model.** Grounds Decision 3 and the #110 prerequisite |
| 3 | Layer targeting — exclude a few layers to "barely fit while keeping max quality"; implemented as exclusion keywords over UNet layer paths (VERIFIED — README) | **Trap, do not import.** Measured false for LoRA here; Decision 5 records why the same words mean different arithmetic |
| 4 | Optimizer-state offload is the load-bearing idea | **Already declined, correctly.** Decision 4 upgrades the reason to a regime discriminator |
| 5 | Author rejects int8 training: quantization error is irrecoverable, converting back to BF16 does not restore quality | **Category confusion worth recording.** He argues against training *weights* in int8; loractl runs the other thing — quantized frozen base, f32 trainables (QLoRA), gated by [ADR-0006](0006-reduced-precision-accuracy-gate.md). The objection does not reach loractl's design. His underlying instinct — fit must not silently cost quality — is exactly #159 |
| 6 | Commenter: INT8 + `torch.compile` ≈ 2× on 30-series; OneTrainer/ai-toolkit/ComfyUI have kernels (UNVERIFIED) | **New axis, not actionable here.** Decision 6 |
| 7 | Saved VRAM buys higher base resolution | **Corroborates** ADR-0005 Addendum 2's correction that demand scales with sequence length |
| 8 | Quality-first stance: memory savings must not cost output quality | **Corroborates** the fit-vs-quality separation of ADR-0006/ADR-0007; motivated filing #159 |
| 9 | NVIDIA/Windows lock-in — an AMD port would need replacements for PyTorch, SMI and bitsandbytes | Weak, one line: the portability cost burn's backend abstraction exists to avoid. loractl's own GPU story is separately constrained (burn#5162 blocks wgpu) |

**Deliberately excluded as zero-signal**, recorded so the exclusion is a
decision and not an oversight: two praise comments that restate the author's own
reply back to him without adding content, an SDXL-nostalgia comment, the
AMD/ROCm availability back-and-forth, and a user-confusion subthread about the
tool not being a LoRA trainer (product/UX, not technique).

## Alternatives considered

- **Demand paging (CUDA unified memory), as bitsandbytes does.** Rejected per
  Decision 1 — but its genuine advantage is recorded rather than suppressed:
  paging is **adaptive**, so it costs *nothing* when everything fits and only
  pays for what actually spills, whereas an explicit offload schedule pays the
  transfer unconditionally every step. bitsandbytes says exactly this ("a paged
  optimizer has zero overhead if all the memory fits onto the device"). That
  property is worth less to loractl than it sounds: at 512px everything already
  fits, so the adaptive path would do nothing at all, and at the resolutions
  where the lever matters it would spill every step anyway — paying UVM's
  documented bandwidth penalty for a transfer we could have scheduled. It is
  also not available: cubecl exposes no managed-memory allocation path
  (ADR-0005 §Decision 1, verified).
- **Offloading the quantized base weights instead of activations.** Rejected as
  the wrong target: ADR-0005 Addendum 2 attributed the wall to activations, and
  the int4 base is already down to ~10.1 GB resident. Streaming it per block
  would add far more PCIe traffic than the ~1 GB block-input set for a saving
  that quantization already took.
- **Doing nothing and leaving ADR-0007's one-line reservation as the record.**
  Rejected because the reservation pointed at a **closed** issue and specified
  no mechanism, so the next person to pick it up would have re-derived the
  paging-vs-explicit question from scratch — and might have reached for UVM,
  which is the wrong answer for a statically scheduled boundary.
- **Writing this as a standalone research note rather than an ADR.** Rejected on
  house convention: `.claude/rules/document-management.md` recognizes only
  PRD/ADR/PRP/TRP, and ADR-0004 is the established precedent for a
  prior-art-driven decision document with a `## References` list.

## References

External (VERIFIED — read for this ADR):

- bitsandbytes, "8-bit optimizers → Paged optimizers" —
  <https://huggingface.co/docs/bitsandbytes/main/en/explanations/optimizers>
  (the UVM mechanism, the bandwidth penalty, the 1 GB / 125 ms worked example,
  and the adaptive-vs-unconditional counterpoint)
- Aozora Trainer (Apache-2.0) — <https://github.com/Hysocs/Aozora_Trainer>
  (Raven/Titan offload modes; layer targeting via exclusion keywords)
- DeepSpeed ZeRO-Offload tutorial — <https://www.deepspeed.ai/tutorials/zero-offload/>
  (optimizer step runs on CPU via `DeepSpeedCPUAdam`)
- ZeRO-Offload, Ren et al., 2021 — <https://arxiv.org/abs/2101.06840>
  ("designed to minimize the data movement to/from GPU")

External (UNVERIFIED — claim only, no primary source read):

- r/StableDiffusion post `1v6ej0k` and its comment thread — the "~95% of paged
  AdamW-8bit's savings", the 11.8 GB / 1.55 s-per-iteration figures, and the
  INT8-plus-`torch.compile` "~2× on 30-series" claim. Single-host, no
  methodology published. None of this ADR's decisions rest on these numbers.

Internal:

- [ADR-0005](0005-int4-training-vram-bound.md) — the retention attribution
  (Addendum 2), the measured 19.4 GB fit and its unit caveat (Addendum 3), and
  the cubecl/burn/loractl offload layering (Decision 2)
- [ADR-0006](0006-reduced-precision-accuracy-gate.md) — what the
  reduced-precision gate does and does not claim
- [ADR-0007](0007-adapter-algorithm-strategy.md) — Decision 3's lever list,
  which this ADR gives a mechanism and a number
- `crates/loractl-core/src/block_ckpt.rs` — the capture/sweep contract the
  offload would hook into
- `.claude/rules/cubecl-pool-reclaim-stale-page.md` — the rule-level summary of
  the levers and measured non-levers
