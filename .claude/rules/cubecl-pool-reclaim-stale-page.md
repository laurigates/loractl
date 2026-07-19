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
- **The pressure is resolution-INDEPENDENT.** A 384px re-run (from 512px)
  produced a **byte-identical peak** and the same OOM. The dominant pool holds
  **~10.9 GB in 328 weight-tile-sized buffers (~33 MB each)** plus **~3.5 GB
  in 161 buffers** — dequantized-weight/gradient allocations that scale with
  the number of **trained sites**, not image size. Working set ≈ **25.5 GB vs
  the 24 GB card**.

## The layering decision (where offload work belongs)

Per ADR-0005, no cubecl-side allocator change can fix this — cubecl has no
offload/spill/unified-memory mechanism, by design:

- **cubecl** — buffer *mechanism* only. Hands out GPU buffers; OOMs when full.
- **burn** — owns the autodiff tape, activations, checkpoint strategy;
  activation offload/recompute lives here.
- **loractl** — owns the model, training loop, config; base-weight streaming
  and target-set choices live here.

## The levers (and the measured non-levers)

Updated by the 2026-07-19 **retention-ledger attribution** (#132, PR #133;
ADR-0005 **Addendum 2** is the canonical record — read it before touching
this problem):

1. **Per-block gradient checkpointing (#134) — the one measured route.** The
   step's true logical demand is **67.9 GiB pinned per forward** (seq 1536,
   Balanced; 60.8 GiB under NoCheckpointing) — ~3× the card, dominated by
   the attention-score trio (scores + mask-add + softmax max-subtract,
   432 MiB × 28 × 3 = 35.4 GiB), SwiGLU outputs (10.5 GiB), and quant-site
   outputs (~9.6 GiB), all eagerly pinned by burn-autodiff's compute-bound
   checkpoint cloning + the untracked-parent fallback. A custom op storing
   only block inputs and recomputing the interior in backward removes every
   dominant class; post-fix estimate ≈ 16–18 GB.
2. **Chunked weight dequant (#128, landed via #130)** — correct but
   measured peak-neutral: the pins are activations, not weight dequants.
3. **Post-load pool reclaim** — safe but insufficient (PR #125, closed).

Non-levers, **measured**: **resolution** (demand scales with seq — note the
trunk pads to a multiple of 256, so "384px" trains at seq 1280, "512px" at
1536 — but even 384px demand is 51.7 GiB, >2× the card), **trained-site
count** (one adapter early in the graph makes the whole downstream trunk
tracked — retention is topology-driven; `lora.targets`
is a scope/quality choice only), **`grad_checkpointing: false`** (60.8 GiB,
ALL retained into backward), and **LoRA rank** (params are a small
fraction).

Separately open: int4's ~7% worst-case dequant error and what it does to
adapter quality (the #25 ComfyUI A/B) — memory fit and output quality are
different questions.

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
