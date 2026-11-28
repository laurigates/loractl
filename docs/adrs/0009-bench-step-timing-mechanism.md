---
id: ADR-0009
status: Accepted
date: 2026-11-28
---

# 0009 — Step throughput is timed from the `TrainEvent` stream, fenced by the loss readback

- **Status:** Accepted
- **Date:** 2026-11-28 (decision implemented 2026-07-27 in PR #161; first
  valid dispatch 2026-08-05)
- **Issue:** [#162](https://github.com/laurigates/loractl/issues/162), for the
  harness from [#110](https://github.com/laurigates/loractl/issues/110)
- **Deciders:** loractl maintainers
- **Builds on:** [ADR-0005](0005-int4-training-vram-bound.md) (why this waited
  for a measurement), [ADR-0008](0008-host-offload-mechanism-and-scope.md) and
  [ADR-0010](0010-rtx4090-throughput-lever-triage.md) (both cite this harness as
  the instrument that prices throughput levers)
- **Numbering:** reserved since ADR-0010 and written only after the first real
  dispatch, so it records a validated mechanism rather than a predicted one.

## Context

`just bench` (`examples/bench_step.rs`, `loractl-core::bench`,
`crates/loractl-bench`) times real training steps of a live `Trainer`. CAEF's
harness, which `loractl-bench` ports, fences a *batch* of identical work between
two full device syncs (`time_wall_sync`), because cubecl's own profiling window
spans only the first pass (cubecl#1421). A training step cannot be batched that
way: steps are not interchangeable, since each one mutates the adapters.

The mechanism is documented at length in `crates/loractl-core/src/bench.rs`'s
module docs. This record adds the decision, the rejected alternatives, and what
the first dispatch did and did not validate.

## Decision

**Time the run by observing the `TrainEvent` stream, relying on the fence the
loss readback already provides.**

A `TrainEvent::Step` carries `loss: f32`, a host scalar. Producing it requires
reading a device tensor back to the host, which drains the device queue for
everything submitted before it. So the window between two consecutive `Step`
events is fenced at both ends and holds exactly one step of compute,
phase-shifted by where the readback sits (monolithic arm:
`backward(n) + optimizer(n) + forward(n+1)`; block-checkpointed arm:
`optimizer(n) + step(n+1)`).

The implementation handles three consequences, each an invariant a later change
could break silently:

1. The first `Step` has no predecessor, so the load/encode window is never a
   sample.
2. A window containing a checkpoint or sample export also contains disk I/O. It
   is marked `ckpt=1` and excluded from the aggregate.
3. The observer's own cost (the `nvidia-smi` subprocess behind
   `resident_vram_mib`) is kept out of both adjacent windows by re-stamping the
   window start after the sample is built. This was a real bug caught in review:
   it biased every window uniformly, invisibly to the sanity ratio.

**Derived-figure policy.** `ms=` and `vram_peak_mib=` are measured. `tok_s=` and
`tflops=` are quotients of an analytic work model (`StepWork`) that is emitted
with them as a `MODEL` line (`step_flops=`, `excludes=`, `assumes=`,
`seq_len_source=`), and must never be quoted without it. The FLOP model uses
split backward multipliers (frozen projections ×2, activation×activation
attention ×3); a uniform multiplier under-counts by a whole attention term
(~2% of the step at seq 1536, ~5% at 4096).

## Alternatives rejected

- **Explicit `B::sync()` in the trainer loop.** Invasive, and it adds fence cost
  to production training to serve a measurement path.
- **A new `TrainEvent` variant carrying timing.** Changes the wire contract
  (`docs/api/events.md`, golden-pinned by `tests/event_json.rs`) for consumers
  that do not need it.
- **Extracting a replayable step so `time_wall_sync` could batch it.** Steps
  are not interchangeable.
- **cubecl profile timers.** Ruled out upstream (cubecl#1421).

## Validation: the first dispatch

[Run 30982204201](https://github.com/laurigates/loractl/actions/runs/30982204201)
(2026-08-05; RTX 4090, cuda + f32 + int4 + `grad_checkpointing`, batch 1, 512px,
`--seq-len 1536` declared; figures recorded in `docs/roadmap.md`):

| | |
|---|---|
| median step | 4482.18 ms |
| peak VRAM | 19,691 MiB |
| derived | `tok_s=342.6901`, `tflops=26.4414` (quotients of the run's `MODEL` line) |
| validity | `sanity=ok`, `plausible=true`, `x2_ratio=2.002`, `steps_counted=6` |

What it settles, against the points #162 listed as able to move:

- **The fence argument** did not depend on the dispatch and is unchanged.
- **`SANITY_LOW`/`SANITY_HIGH` (1.7–2.3, ±15% around 2.0)** held with margin:
  2.002 on a real run. The band stays.
- **The peak** agrees with ADR-0005 Addendum 3's ~19.4 GB within ~1%, from a
  different instrument on a different day. This is the strongest result of the
  run.

What it does not settle, so it stays open rather than being asserted here:

- **Per-step `nvidia-smi` perturbation of clocks across a long run.** Six
  counted steps cannot show drift either way. If a longer dispatch shows step
  times drifting (clock or thermal behaviour rather than compute), sample VRAM
  every N steps or from a watcher thread via `StepBench::with_vram_source`.
- **The `warmup + 7` floor for a quotable run.** One run is not a variance
  study. Keep the floor until repeated dispatches give a spread.
- **Where the modelled `tflops=` sits relative to hardware peak.** The FLOP
  count is a loractl accounting choice and an under-count (`excludes=`), so the
  ratio to a nominal peak is not a utilisation figure. Not compared here.

## Consequences

- Any change to the trainer that defers or batches the loss readback (for
  example, accumulating losses and reporting every N steps) removes the fence
  and makes every timing meaningless, with no failure and no test able to
  catch it (on `ndarray`, `Backend::sync` is a no-op). That invariant is
  recorded for anyone touching the loop in
  `.claude/rules/bench-readback-fence.md`.
- Throughput levers that spend time to buy VRAM (#158 host offload, #174
  fusion, #176/#177 load path) are priced with this harness against the
  baseline above, on a matching configuration.
