---
id: ADR-0005
status: Accepted
date: 2026-07-18
---

# 0005 — int4 real-model training is VRAM-bound, not a cubecl reclaim bug; offloading is burn/loractl's job

- **Status:** Accepted
- **Date:** 2026-07-18
- **Milestones:** M15 ([#119](https://github.com/laurigates/loractl/issues/119) int4 real run; [#96](https://github.com/laurigates/loractl/issues/96) memory)
- **Deciders:** loractl maintainers
- **Supersedes framing in:** laurigates/cubecl#2, and the working assumption in
  [#96](https://github.com/laurigates/loractl/issues/96) that the blocker is a
  cubecl pool reclaim race (upstream tracel-ai/cubecl#1401 / #1384).

## Context

The int4 (Q4S) real-model training step for Krea-2-Raw (ADR-0004) panicked on
the RTX 4090 with `couldn't find resource for that handle: Memory page N
doesn't exist` at `cubecl-cuda/src/compute/server.rs:{124,701}`. This was
attributed — in laurigates/cubecl#2, in #96, and upstream in
tracel-ai/cubecl#1401/#1384 — to a **CUDA pool reclaim race**: `cleanup` under
pool pressure tombstoning a page whose kernel binding was still pending. Two
fixes were attempted against that theory on the `laurigates/cubecl` fork
(branch `fix/stable-page-indices-v0.10`) and **neither unblocked the run**.

This ADR records what the failure actually is, established by instrumented
on-hardware measurement, so the effort is not spent again chasing the wrong
layer.

## Investigation (on-hardware, RTX 4090, CUDA 13.0)

Pool + storage instrumentation of the fork under the real int4 step, plus
`cuMemGetInfo` at the exact allocation-failure point and per-process VRAM
attribution:

1. **It is genuine VRAM exhaustion.** At the first `malloc` failure the driver
   reports **~0.58 GB free of 25.2 GB** — the card is ~98% full. Both
   `malloc_async` and `malloc_sync` fail with real `CUDA_ERROR_OUT_OF_MEMORY`.
   Not fragmentation (free is truly near-zero), not a reclaim race.
2. **There is no reclaim race.** Zero exclusive-pool tombstone events fired.
   The "Memory page 0" in the panic is the **uninitialized-handle default**
   (`MemoryLocation::uninit() = {pool:0, page:0}`): a failed allocation leaves
   its handle unbound, and every downstream lookup of an unbound handle reports
   "page 0 doesn't exist." The count of distinct missing handles equalled the
   count of allocation failures exactly (1264 = 1264). The 6368 "page 0" errors
   are **OOM fallout**, not the cause.
3. **The command queue cannot pile up.** cubecl's device command channel is
   bounded (`CHANNEL_MAX_TASK = 32`, double-buffered ≤ 64) with built-in
   client backpressure. A "queued transients pile up" mechanism is structurally
   impossible there; the memory pressure is genuinely-live tensors held above
   cubecl (the burn autodiff graph), not a queue artifact.
4. **A stream sync before reclaim does nothing.** Draining GPU work does not
   drop the host-side references keeping the memory live — byte-identical
   failure with and without it.
5. **Peak attribution.** loractl alone climbs to **~23.4 GB**; the only other
   GPU consumer is an idle ComfyUI at a constant **386 MiB**. The working set
   is a slow activation ramp then a **~+7 GB spike** at the backward pass. Total
   working set ≈ 25.5 GB, ~1.5–2 GB over the 24 GB card.
6. **The pressure is resolution-INDEPENDENT (measured, not assumed).** Re-running
   at 384px (from 512px) produced a **byte-identical peak** (~23.67 GB in_use)
   and the same OOM. The dominant pool holds **~10.9 GB in 328 weight-tile-sized
   buffers (~33 MB each)** plus a second pool of ~3.5 GB in 161 buffers — these
   are **dequantized-weight / gradient** allocations that scale with the number
   of *trained sites*, not image size. Image activations are a minor fraction.
   **Lowering resolution does not help** — the earlier assumption that the +7 GB
   spike was resolution-driven attention activations is falsified.

## Decision

1. **The int4-real-run blocker is reclassified as VRAM capacity, not a cubecl
   reclaim/queue defect.** No cubecl-side allocator change can fix it —
   verified: cubecl has no offload/spill/unified-memory mechanism; it hands out
   GPU buffers and OOMs when the card is full.
2. **Offloading is out of cubecl's scope by design.** The layering:
   - **cubecl** — buffer *mechanism* only (GPU + CPU-pinned pools; the pinned
     pool is a transfer-staging buffer, not a GPU spill target). No policy.
   - **burn** — owns the autodiff tape, activations, and checkpoint strategy;
     activation offload/recompute lives here.
   - **loractl** — owns the model, training loop, and config; base-weight
     streaming and resolution/batch/target choices live here.
3. **The unblock is a loractl/burn memory-reduction on the resolution-INDEPENDENT
   weight/gradient/dequant footprint** (lowering resolution is measured NOT to
   help — see Investigation #6). Effective levers, weight-side: **fewer trained
   sites** (fewer LoRA targets → less optimizer state and fewer simultaneous
   dequant/gradient buffers), **base-weight streaming**, and **reducing
   simultaneous dequantized-weight retention in the backward pass** (a
   burn-autodiff memory concern — `QuantMatmulT` already dequantizes transiently,
   but ~10.9 GB of weight-tile buffers are co-resident at peak). Ineffective:
   lower resolution, LoRA rank (params are a small fraction).
4. **The one cubecl change worth keeping** is defensive only:
   `laurigates/cubecl` PR #3 makes the four `NotFound` panic sites recover as
   stream errors instead of aborting the device thread (the same direction as
   the already-merged PR #1 / upstream #1384). It is hardening, **not** a fix
   for this OOM, and is documented as such.

## Consequences

- **Rejected:** any further cubecl-side reclaim/queue engineering for this
  blocker — GPU-cursor-aware cleanup (the maintainer's #1401 direction),
  sync-before-reclaim (attempt below), command-queue drain/backpressure. All
  target a race that does not occur here.
- **The sync-before-reclaim attempt** (a `ComputeStorage::sync()` primitive +
  a stream sync before the OOM reclaim) is closed and its fork branch deleted.
  This ADR is its record: it had zero measured effect because the memory is
  genuinely live above cubecl, not GPU-work-in-flight.
- **Open follow-up (loractl):** reduce the resolution-independent weight/gradient
  footprint so int4/real fits 24 GB — start with fewer LoRA targets (measure the
  co-resident dequant/gradient buffers per site), then base-weight streaming or
  reducing simultaneous dequant retention in backward. A 384px probe confirmed
  resolution is not a lever. Tracked under #119 / #96.
- **Upstream note:** tracel-ai/cubecl#1401's "reclaim race" framing does not
  explain this particular failure; PR #3's graceful-abort is the only
  upstreamable slice.

## Addendum (2026-07-18, same day — the prescribed measurement ran)

The `step_probe` sweep this ADR's follow-up called for (landed in #126) ran on
the RTX 4090 the same day: 5 LoRA target sets (1 / 56 / 84 / 112 / 196 of 196
injectable sites) × {no reclaim, post-load reclaim}, uncontended card, int4,
512px, 3 steps. Full table: the 2026-07-18 sweep comment on
[#96](https://github.com/laurigates/loractl/issues/96). Two of this ADR's
directions are **falsified by the measurement**, one is narrowed:

1. **"Fewer trained sites" is NOT a memory lever.** A single trained site
   peaks (~24 GB) and fails identically to all 196. The co-resident
   dequant/gradient transients arise at **every** quantized site because the
   activations are tracked through the whole autodiff graph regardless of
   which sites carry adapters — trained-site count changes optimizer state
   only, a rounding error here. (Decision 3's "fewer trained sites" lever is
   withdrawn; the attention-only config default in #126 stands as a
   scope/quality choice only.)
2. **Post-load pool reclaim is safe but NOT sufficient.** The explicit
   `sync → memory_cleanup → sync` bracket works on stock cubecl 0.10 at
   quiescence (validated twice: 15.1 → 10.3 GB) but the step's ~14–15 GB
   transient working set refills the card regardless: reclaimed base
   ~10.3 GB + step state ≈ 24.7 GB vs ~23.6 GB usable. PR #125 closed on
   this data (branch kept — it composes with the retention fix below).
3. **The binding constraint is the backward dequant-transient working set,
   and the gap is only ~1–2 GB.** The recurring failed allocations are
   1,576,693,760 B (identity unresolved — it matches no single weight's
   dequant size; the largest weight dequant is `tproj.fc` at ≈ 864 MiB) and
   37,748,736 B (trunk tiles).
   The route is now **chunked dequant in `QuantMatmulT`** at the
   packed-int/byte level (burn 0.21's `q_slice` is `unimplemented!()` on
   cuda — no QFloat `Tensor::slice`), largest transient first; base-weight
   streaming remains the fallback. Tracked as
   [#128](https://github.com/laurigates/loractl/issues/128).

Two operational findings from the same sweep, recorded here because they
change how future measurements must be read:

- **A run that "survives" at the pool ceiling is not a working run.** The
  3/3-step runs rode 13k–92k OOM panics and died before export; one produced
  a *finite but garbage* loss (4.9e8) with no preceding panic — **silent
  forward corruption at the ceiling**. The fit gate is a run with **zero**
  panics, not a survived storm.
- Device-global VRAM telemetry is meaningless on the shared box without a
  contention guard — an idle ComfyUI can hold ~18 GB of cached models and
  re-grab the card mid-run.

## Addendum 2 (2026-07-19 — retention-ledger attribution; #132, PR #133)

The per-op attribution the addendum above called for ran on the RTX 4090:
three instrumented arms (Balanced@512px / NoCheckpointing@512px /
Balanced@384px, int4, `blocks\.` targets) through a burn-autodiff fork ledger
that records every checkpoint retention decision host-side —
measured **logical demand**, valid even while the card storms (logs:
`loractl-training/ledger-*.{tsv.gz,report.txt,stdout.log}`). Three
corrections to the record, then the attribution and the verdict.

### Corrections

1. **Fact 6 of the Investigation ("resolution-INDEPENDENT") is withdrawn as
   stated, and Addendum 1's step framing shrinks-with-resolution.** Two
   compounding measurement artifacts: `mmdit.rs` pads the trunk sequence to a
   multiple of 256, and aspect bucketing changes token counts — the "512px"
   probe actually trains at seq **1536**, the "384px" probe at seq **1280**
   (not 832). Both prior arms rode the 24 GB ceiling, so their "byte-identical
   peaks" measured the card, not the demand. Ledger-measured demand DOES scale
   with sequence (67.9 → 51.7 GiB), but resolution remains a **non-lever**:
   even 384px demand is >2x the card.
2. **The recurring failed-allocation sizes are pool page-ladder artifacts, not
   tensors.** cubecl-runtime `memory_manage.rs` builds its SlicedPages pools
   by repeatedly dividing `max_page_size` by 4 (with alignment rounding):
   1,576,693,760 / 394,173,440 / 98,543,616 B are page sizes in that ladder.
   A failed-alloc size carries no tensor identity — which is why #130's
   complete weight chunking could not remove the "1.58 GB alloc": nothing
   tensor-shaped was ever behind it.
3. **The "~14-15 GB step transient / ~1-2 GB fit gap" framing is withdrawn.**
   It was derived from watching a saturated card. The true unconstrained
   step demand (below) is ~2.5-3x the card; no marginal lever closes that.

### The measured attribution (per step, seq 1536 unless noted)

| arm | pinned during forward | retained into backward | dropped unused at build | backward live peak |
|---|---|---|---|---|
| Balanced@512 | **67.9 GiB** | 32.8 GiB | **19.5 GiB** | 32.9 GiB |
| NoCheckpointing@512 | 60.8 GiB | 60.8 GiB (all) | 0 | 60.8 GiB |
| Balanced@384 (seq 1280) | 51.7 GiB | — (died at NaN loss) | — | — |

Composition of the Balanced 67.9 GiB (top classes): the **attention-score
trio** — raw `q·kT` scores, the mask add, and softmax's max-subtract, each
`[1, 48, 1536, 1536]` f32 = 432 MiB x 28 blocks x 3 = **35.4 GiB**; SwiGLU
gate/up outputs `[1536, 16384]` x 56 = 10.5 GiB (QuantMatmulT + adapter-path
copies); quant-site outputs ~9.6 GiB; misc elementwise `[1536, 6144]` classes
~8 GiB. Mechanism (code-verified in burn-autodiff 0.21): `BalancedCheckpointing`
eagerly clones any compute-bound tensor's primitive at forward
checkpoint-registration, and flips ops with **untracked parents** (masks, RoPE
tables, modulation, every `.no_grad()` base param) to compute-bound
(1,561 flip events/step measured) — cascading eager pins across the whole
tracked graph. This is also the real reason **trained-site count was a
non-lever**: one adapter early in the graph makes the entire downstream trunk
tracked; retention is topology-driven.

Also settled: **`grad_checkpointing: false` is not a lever** (60.8 vs
67.9 GiB, and it retains MORE into backward); and the storm-mode "completed"
steps are confirmed garbage — the NoCheckpointing arm printed a **negative
MSE loss**, which is impossible for a mean of squares (fact 6's zero-panic
gate stands).

### Verdict

The route is **manual per-block gradient checkpointing** (#134): wrap
`SingleStreamBlock::forward` in a custom autodiff op that stores only the
block inputs (~36 MiB each; ~1 GiB for 28 blocks) and recomputes the block
interior during backward — QuantMatmulT already re-dequantizes transiently,
so recompute adds one forward per block, not weight traffic. Every dominant
class above (scores trio, SwiGLU, site outputs) becomes a within-block
transient. Post-fix demand estimate: ~10.3 GB quantized base + ~1 GiB block
pins + ~2-3 GiB single-block recompute transient + optimizer state ≈
**16-18 GB — fits the 24 GB card with real margin**. Chunked attention inside
the recomputed block is the follow-on if the transient needs shrinking.
