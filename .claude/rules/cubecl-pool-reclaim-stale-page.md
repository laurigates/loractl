# int4 Real-Model Training Is VRAM-Bound (ADR-0005) — Spend Effort on Footprint Levers, Not cubecl Reclaim

**This rule supersedes its previous version**, which blamed the #25 real-run
panic on a cubecl-cuda pool-reclaim race ([tracel-ai/cubecl#1401]) and asserted
"not an out-of-memory — the live set fits." Instrumented on-hardware
measurement (RTX 4090, 2026-07-18) falsified that:
[ADR-0005](../../docs/adrs/0005-int4-training-vram-bound.md) is the canonical
record. Sibling to [`burn-wgpu-metal-numerics.md`](burn-wgpu-metal-numerics.md)
(the wgpu/Metal autodiff bug, which is unchanged and still blocks the f16
route).

## The reclassification (measured, not argued)

- **Genuine OOM.** At the first failed `malloc` the driver reports **~0.58 GB
  free of 25.2 GB** — the card is ~98% full. Not fragmentation, not a race.
- **Zero reclaim-race events.** No exclusive-pool tombstone ever fired. The
  `Memory page 0 doesn't exist` panics are **OOM fallout**: a failed
  allocation leaves its handle at the uninitialized default (`{pool:0,
  page:0}`), and the count of distinct missing handles equalled the count of
  allocation failures exactly (**1264 = 1264**).
- **The "queued transients pile up" mechanism is structurally impossible**:
  cubecl's device command channel is bounded (`CHANNEL_MAX_TASK = 32`,
  double-buffered ≤ 64) with client backpressure.
- ~~**The pressure is resolution-INDEPENDENT.**~~ **WITHDRAWN by ADR-0005
  Addendum 2 (§Corrections item 1) — do not act on this bullet.** The
  observation stands (a 384px re-run produced a **byte-identical peak** and
  the same OOM; the dominant pool held **~10.9 GB in 328 weight-tile-sized
  buffers** plus **~3.5 GB in 161 buffers**; working set ≈ **25.5 GB vs the
  24 GB card**) but its *interpretation* was wrong twice: both arms rode the
  24 GB ceiling, so the identical peaks measured **the card, not the
  demand** — ledger-measured demand does scale with sequence length
  (67.9 GiB @ seq 1536 vs 51.7 GiB @ 1280) — and the pinned bytes are
  **activations, not dequantized weights**, retained by graph topology rather
  than by trained-site count. See lever 2 and the non-lever list below, which
  this bullet used to contradict.

## The layering decision (where offload work belongs)

Per ADR-0005, no cubecl-side allocator change can fix this — cubecl has no
offload/spill/unified-memory mechanism, by design:

- **cubecl** — buffer *mechanism* only. Hands out GPU buffers; OOMs when full.
- **burn** — owns the autodiff tape, activations, checkpoint strategy;
  activation offload/recompute lives here.
- **loractl** — owns the model, training loop, config; base-weight streaming
  and target-set choices live here.

**If you reach for host offload, read
[ADR-0008](../../docs/adrs/0008-host-offload-mechanism-and-scope.md) first.** It
fixes the mechanism — **explicit scheduled transfer over pinned buffers, never
CUDA unified memory / demand paging**, because the #134 block boundary is a
statically known schedule and UVM is documented by bitsandbytes to lose "half or
worse" of PCIe bandwidth and cannot overlap transfer with compute. It also sizes
the lever (the retained block-input set is ~1.06 GB at 512px, ~3.17 GB at
1024px — batch-1, derived, and an upper bound before prefetch retention; note
seq does *not* scale with pixel area, since the 512-token caption block is
fixed) and makes #110's bench harness a hard prerequisite: this is
the first loractl memory lever that spends throughput to buy VRAM. Tracked as
#158; the older "reserved under #96" pointer is dead (#96 is closed).

## The levers (and the measured non-levers)

Updated by the 2026-07-19 **retention-ledger attribution** (#132, PR #133)
and the 2026-07-23 landed fix (#134, PR #135); ADR-0005 **Addenda 2 and 3**
are the canonical record — read them before touching this problem:

1. **Per-block gradient checkpointing (#134) — the route, now LANDED and
   MEASURED.** The monolithic step's true logical demand was **67.9 GiB
   pinned per forward** (seq 1536, Balanced; 60.8 GiB under NoCheckpointing)
   — ~3× the card, dominated by the attention-score trio (scores + mask-add
   + softmax max-subtract, 432 MiB × 28 × 3 = 35.4 GiB), SwiGLU outputs
   (10.5 GiB), and quant-site outputs (~9.6 GiB), all eagerly pinned by
   burn-autodiff's compute-bound checkpoint cloning + the untracked-parent
   fallback. `src/block_ckpt.rs::checkpointed_step` removes every dominant
   class — **not** as a custom op (a nested `backward()` deadlocks on burn
   0.21) but as a two-phase step: graph-free capture forward storing only
   block inputs, then a reverse per-block sweep of standalone graphs.
   Measured result: **19.4 GB** peak, zero panics, 3/3 steps, 196/196 sites
   at 512px int4 — the #25 real run rode it.
2. **Chunked weight dequant (#128, landed via #130)** — correct but
   measured peak-neutral: the pins are activations, not weight dequants.
3. **Post-load pool reclaim** — safe but insufficient (PR #125, closed).

Non-levers, **measured** — all of these are verdicts about the **monolithic
step**, i.e. reasons it could not be rescued without restructuring; they do
**not** carry over to the block-checkpointed regime the configs now ship:
**resolution** (demand scales with seq — note the trunk pads to a multiple of
256, so "384px" trains at seq 1280, "512px" at 1536 — but even 384px demand
was 51.7 GiB, >2× the card, so lowering it could not close the gap. Under
block checkpointing the peak is one block interior, which *does* scale with
seq — so resolution is a live variable again: re-probe after changing it),
**trained-site count** (one adapter early in the graph makes the whole downstream trunk
tracked — retention is topology-driven; `lora.targets`
is a scope/quality choice only), **`grad_checkpointing: false`** (60.8 GiB,
ALL retained into backward), and **LoRA rank** (params are a small
fraction).

Separately open (tracked as **#159**): int4's ~7% worst-case dequant error and
what it does to adapter *quality*. The #25 ComfyUI A/B proved the trained
adapter visibly conditions generation, which is a conditioning proof, not a
quality benchmark — memory fit and output quality remain different questions.
Also unmeasured: step **throughput** under block checkpointing (one extra trunk
forward per step; needs #110's harness — whose reusable core landed as
`crates/loractl-bench`, but nothing depends on it yet, so no training step has
ever been timed).

Measure with `just step-probe` (the recipe landed in #126) —
don't re-derive peaks from `nvidia-smi` eyeballing.

## What survives from the cubecl work

- Fork PRs `laurigates/cubecl#1` (graceful cursor) and `#3` (recover
  `NotFound` as stream errors instead of aborting the device thread, merged
  2026-07-18) are **defensive hardening only** — they turn the OOM fallout
  panics into recoverable errors. They are **not** a fix for the OOM.
- Upstream [tracel-ai/cubecl#1401] remains open and **may still be real for
  other workloads** (the original reporter's ~16 GB-resident generation
  workload). Our contribution to that thread was made under the wrong theory
  and was corrected on the thread (2026-07-18), including the
  free-VRAM-at-first-failure discriminator for classifying such panics.
- The sync-before-reclaim experiment is closed (zero measured effect) and its
  fork branch deleted; ADR-0005 is its record.

## Rationale

A full engineering push (tombstone pool, graceful cursor, sync-before-reclaim)
went into the wrong layer because the panic *looked like* an allocator
correctness bug. The discriminating facts — free VRAM at the failing malloc,
tombstone-event count, missing-handles == failed-allocations — took one
instrumented run to collect and settled it. Read ADR-0005 before touching
cubecl for this workload; spend the effort on the footprint levers above.

[tracel-ai/cubecl#1401]: https://github.com/tracel-ai/cubecl/issues/1401
