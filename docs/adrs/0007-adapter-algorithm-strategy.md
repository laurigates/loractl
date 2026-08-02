---
id: ADR-0007
status: Proposed
date: 2026-07-22
---

# 0007 — Adapter-algorithm strategy: which PEFT methods to support, and separating parameter-efficiency from the VRAM wall

- **Status:** Proposed
- **Date:** 2026-07-22
- **Milestones:** M14 ([#25](https://github.com/laurigates/loractl/issues/25) the real run), follows M6 ([#17](https://github.com/laurigates/loractl/issues/17) the LoraAdapters/export seam)
- **Deciders:** loractl maintainers
- **Builds on:** [ADR-0004](0004-krea2-image-diffusion-target.md) (the Krea 2 target) and [ADR-0005](0005-int4-training-vram-bound.md) (the VRAM reclassification and its Addendum 2 retention attribution)
- **Relied on by:** [ADR-0008](0008-host-offload-mechanism-and-scope.md) (2026-07-26), which builds on Decision 3's lever taxonomy and corrects its #96 → #158 reservation

## Context

Two questions have been conflated in the roadmap, and this ADR separates them.
The first is a **quality/param-efficiency** question — *which adapter
parameterization* (plain LoRA, LoKr, LoHa, DoRA, VeRA, OFT/BOFT, LoRA-XS, …)
should loractl express its trainable delta as. The second is the **fit**
question that actually gates the #25 real run — *can a LoRA train against the
~12.8B Krea-2-Raw base on the 24 GB RTX 4090 at all*.

[ADR-0005](0005-int4-training-vram-bound.md) settled the fit question's
mechanism. The binding constraint is **activation retention**, not resident
weights and not a cubecl allocator defect: Addendum 2's retention-ledger
attribution (#132, PR #133) measured the monolithic step's true logical demand
at **67.9 GiB pinned per forward** (seq 1536, Balanced) — ~3× the card,
dominated by the attention-score trio (`[1, 48, 1536, 1536]` f32 scores +
mask-add + softmax max-subtract = 432 MiB × 28 × 3 = **35.4 GiB**), SwiGLU
outputs (10.5 GiB), and quant-site outputs (~9.6 GiB). Crucially, the demand is
**topology-driven**: one trainable adapter early in the trunk makes the entire
downstream graph tracked, so retention is independent of trained-site count and
LoRA rank (both measured dead in Addendum 2 and its predecessor sweep). Demand
does scale with sequence length — 67.9 GiB at seq 1536 / 512px vs 51.7 GiB at
seq 1280 / 384px — so resolution was a *non-lever* for the monolithic step
rather than irrelevant: even 384px demand exceeded 2× the card (ADR-0005
Addendum 2 §Corrections item 1). The measured fix is block-level gradient checkpointing (#134,
`src/block_ckpt.rs::checkpointed_step`) layered on an int4/int8 QLoRA base
(#119/#96); that combination measures **19.4 GB** peak on a 24 GB card
(ADR-0005 Addendum 3, against a 16–18 GB projection).

The consequence for adapter algorithms is the load-bearing fact of this ADR:
**parameter count is a rounding error against a ~19 GB working set whose
dominant classes are base-trunk activations, not adapter params.** Every method
that only changes *how the trainable delta is expressed* — LoKr's Kronecker
factors, LoHa's Hadamard product, VeRA's shared bases, DoRA's magnitude
vector — leaves the 35.4 GiB attention trio and the tracked-trunk topology
exactly where they are. This is corroborated outside loractl: huggingface/peft
independently measures a plain LoRA retaining all-layer activations at ~2× full
finetune once an early adapter taints the trunk. So the adapter menu is a
**quality/interop track that does not move the VRAM needle**, and the fit is a
separate track with only three real lever classes (activation retention,
resident base weights, optimizer/gradient — the last irrelevant for LoRA, whose
optimizer state is adapter-only).

The fit is now **measured, not an estimate** (ADR-0005 Addendum 3): a
zero-panic `just step-probe` at 512px int4 peaked at **19.4 GB** on the 24 GB
card — 3/3 steps, 196/196 sites, ~4 GB headroom — and the M14 real run closed
on that route (#25, 300 steps, ComfyUI A/B). The 16–18 GB projection held to
within ~1.4 GB. This does not change the ADR's load-bearing conclusion; it
raises the confidence behind it, and it means the *next* adapter-algorithm
decision is a quality decision made against a working fit rather than a
hypothetical one. Two backend facts bound what is buildable today:
burn 0.21's GPU autodiff is numerically broken except on the cuda-f32 path
([burn#5162](https://github.com/tracel-ai/burn/issues/5162), unchanged), and a
custom autodiff op cannot run a nested `backward()` on 0.21
([burn#5193](https://github.com/tracel-ai/burn/issues/5193); #134 works around
it with a two-phase graph-free capture). fp8 activations have no burn-store
dtype on 0.21. So "burn 0.21 vs 0.22-gated" is the axis that decides which
levers are reachable now.

The adapter subsystem already has the seam a second algorithm would plug into
(M6, #17): base-free `LoraDelta` deltas held in a name-keyed `LoraAdapters`
container, injected additively at each advertised site via `apply` (`base_out +
delta.forward(x)`), and exported through the format-agnostic
`AdapterNameMapper` trait. That seam is where any new method belongs — and any
new *format* it emits inherits the interop rule below.

## Decision

1. **Treat adapter parameterization and VRAM fit as two disjoint tracks, and
   say so in the roadmap.** The adapter menu is a **quality / param-efficiency /
   interop** track; the #25 fit is a separate track solved by activation
   levers. No adapter algorithm is a path to the fit — retention is
   topology-driven (ADR-0005 Addendum 2), so re-expressing the delta cannot
   un-taint the trunk. Choosing an adapter algorithm is a decision made *after*
   the fit is confirmed, never a way to reach it.

2. **Generalize the adapter abstraction to support multiple algorithms, behind
   the existing `LoraAdapters`/export seam, as a quality feature — LoKr first.**
   LoKr is the one worth adding first: smallest on-disk file, `lokr_*` keys
   parsed by ComfyUI, and expressible in plain burn ops (no nested backward).
   The generalization is a real but bounded structural change confined to core:
   the `LoraAdapters.deltas` field must move off the concrete `LoraDelta` type
   (a `#[derive(Module)]`-compatible enum over delta variants — `Vec<dyn Trait>`
   is not a burn `Module`), `build_adapters` gains a `method` discriminant, the
   `AdapterNameMapper` three-key `down/up/alpha` contract generalizes to an
   arbitrary per-site key set, and `block_ckpt.rs::track_adapters` plus its
   grad-collection loop must enumerate each new variant's trainable factors
   (re-marking `require_grad` while preserving `ParamId` — the burn 0.21
   `Param::clone` trap, see the rule and #134). The `mmdit.rs` forward is
   untouched: `apply`'s additive `[.., d_out]` contract is method-agnostic. This
   changes nothing about the fit — LoKr's naive path even *materializes* ΔW, so
   the bypass/sequential forward is mandatory to stay memory-neutral.

3. **Name the VRAM levers for the #25 real run explicitly, and which burn
   version gates each:**
   - **burn 0.21, landed and measured:** block-level gradient checkpointing
     (#134) on an int4/int8 QLoRA base (#119/#96) — the core fit mechanism,
     confirmed by a zero-panic `just step-probe` at **19.4 GB** peak on the
     24 GB card and by the #25 real run.
     [cite ADR-0005 Addendum 3.]
   - **burn 0.21, unlanded, near-term:** CPU activation offload paging the #134
     per-block boundary tensors to pinned host RAM — the top *new* activation
     lever, stacking on #134; and encoder unload after latent/conditioning
     caching (M12) to free a fixed few GB. Chunked attention inside the
     recomputed block is the reserved follow-on if the fit rides too close.
     [ADR-0008](0008-host-offload-mechanism-and-scope.md) now fixes the offload
     lever's *mechanism* (explicit scheduled transfer, never CUDA unified
     memory / demand paging) and sizes it: the retained block-input set is
     ~1.06 GB at 512px (seq 1536) and ~3.17 GB at 1024px (seq 4608 — the
     512-token caption block is fixed, so seq does not scale with pixel area),
     both batch-1 and derived, so it is worth ~5% of the measured peak today
     and gates on #110.
   - **burn 0.22-gated ([#79](https://github.com/laurigates/loractl/issues/79)):**
     COAT-style fp8 activation training (blocked today — no fp8 store dtype),
     the mechanistically correct attack on the attention trio; and burn's native
     LoRA/QLoRA + fusion-fused dequant, which can re-express #134 as a native
     custom op now that the nested-backward fix
     ([burn#5194](https://github.com/tracel-ai/burn/pull/5194)) is **merged
     upstream** (2026-07-24, unreleased — 0.21.0 is still the latest tag).
     Re-run the burn#5162 numerics ladder and the retention ledger on
     migration.
   - **Declined as fit levers** (correct existing instinct): 8/4-bit Adam, fused
     backward, GaLore, gradient accumulation, and every param-reducer (VeRA,
     LoRA-XS, NOLA, IA³). Optimizer state is adapter-only; GaLore trains full
     weights and *maximizes* trunk tracking. DoRA and LoHa mildly *worsen* the
     footprint (weight-class overhead / materialized ΔW) and are quality
     experiments only, deferred until after the fit is green.

4. **Set the interop gate as a merge requirement for any new adapter format.**
   No adapter algorithm merges until its export is proven loadable by the real
   consumer (ComfyUI / kohya) **and** carries a consumer-contract test in the
   sense of the interop testing rule (`.claude/rules/testing.md`,
   `tests/krea2_lora_keys.rs`): the real export path over the real site
   enumeration, every on-disk key asserted against the consumer's key map
   generated from **pinned upstream source**, with teeth (the un-renamed form
   must *not* be accepted; sabotaging the mapper to identity must fail it). This
   is non-negotiable because the failure shape is silent — a LoRA in an unparsed
   schema loads **without error and does nothing** (the #137/#138 misdiagnosis).
   ComfyUI's currently-enabled adapter set is LoRA, LoHa, LoKr, and OFT (plus
   DoRA `.dora_scale` application); **BOFT is present-but-disabled**, so it is
   unshippable regardless of training merit, and OFT must emit kohya `oft_blocks`
   keys, not the LyCORIS `oft_diag` form, or it silently no-ops. A self-golden
   alone is insufficient by construction — it pins our convention and cannot
   disagree with it.

## Consequences

- **Rejected:** DoRA, LoHa, VeRA, LoRA-XS, NOLA, and IA³ for the current cycle.
  DoRA and LoHa cost VRAM rather than save it (DoRA's overhead is the merged
  weight `W0+BA`, weight-class and sequence-independent — the widely-cited
  "+75%/tracked-norm" figure is misattributed; the column norm is autodiff-
  detached by default, so the credible cost is ~+5–10% end-to-end plus a QDoRA
  per-block dequant that fights the int4 route). The rest are pure param
  reducers with no native ComfyUI loader (bake-to-standard-LoRA required) and
  zero relief against loractl's activation-bound working set. BOFT is rejected
  as shipped-but-disabled in ComfyUI.
- **Rejected:** framing any adapter algorithm as a contribution to the #25 fit.
  The retention attribution (ADR-0005 Addendum 2) makes this a category error;
  the fit is an activation problem, the adapter menu a quality problem.
- **Rejected:** LoRA-FA as a fit lever. It is a cheap composable add-on (freeze
  the down-projection — one `require_grad(false)`, output loads as a standard
  LoRA, halves adapter optimizer state) but its activation saving touches only
  the adapter's own input, never the 35.4 GiB base-trunk trio, so it is
  partial-at-best and must be pitched as optimizer/hygiene, not the fit —
  measure whether burn 0.21's eager autodiff prunes the frozen-A input at all
  before claiming any saving.
- **Resolved (fit):** the zero-panic `just step-probe` for int4 + #134 was
  reported — **19.4 GB** peak, ~4 GB headroom, and #25 closed on that route
  (ADR-0005 Addendum 3). CPU activation offload on the #134 block boundary and
  encoder unload after caching are therefore *unneeded at 512px* and become
  relevant only at longer sequences or smaller cards; they stay reserved under
  [#158](https://github.com/laurigates/loractl/issues/158) — **not** under #96,
  which is closed, so the earlier reservation tracked nothing (corrected by
  [ADR-0008](0008-host-offload-mechanism-and-scope.md), which also fixes the
  mechanism and sizes the lever). What is genuinely unmeasured is the
  **throughput** cost of the extra per-block forward (needs #110's harness) —
  and #110 is now a hard prerequisite for the offload lever, which trades VRAM
  for PCIe time.
- **Open follow-up (quality/interop):** land LoKr behind the generalized seam
  (Decision 2) with its ComfyUI consumer-contract test (Decision 4); optionally
  OFT (Cayley-Neumann, burn has no differentiable matrix inverse) emitting
  kohya `oft_blocks`. Both are post-#25 — the real run proves plain LoRA visibly
  conditions generation first.
- **Open follow-up (quality, orthogonal to this ADR):** int4's ~7% worst-case
  dequant error and its effect on adapter quality (the #25 ComfyUI A/B) — memory
  fit and output quality are different questions, and the adapter algorithm does
  not decide either. Now tracked as
  [#159](https://github.com/laurigates/loractl/issues/159); it had no issue
  while six documents called it open.
- **burn 0.22 dependency:** the strongest activation lever (COAT fp8) and the
  cleanest #134 re-expression are gated on the milestone-scale migration (#79);
  the seam generalization in Decision 2 is designed to survive it (kohya export
  and key-map stay loractl's regardless of burn's native format).
