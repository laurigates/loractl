# `just bench` Depends on the Per-Step Loss Readback — Don't Defer or Batch It

`StepBench` (`crates/loractl-core/src/bench.rs`) times a training step as the
window between two consecutive `TrainEvent::Step` events. That window is only a
measurement because each `Step` carries a **host** `f32` loss: producing it
reads a device tensor back, which drains the device queue. The readback is the
fence at both ends of every window. Decision and validation:
[ADR-0009](../../docs/adrs/0009-bench-step-timing-mechanism.md).

## The trap

Any trainer change that removes the per-step readback silently invalidates
every bench number:

- accumulating losses on device and reporting every N steps;
- emitting `Step` before the loss is read back (e.g. sending a lazily-resolved
  value, or the previous step's loss);
- moving the readback behind an async/background channel.

Nothing fails. The bench still prints `RESULT` lines, and the 2×-iters sanity
ratio cannot see it, because the windows stay internally consistent while no
longer bounding one step of compute. No offline test can catch it: on
`ndarray`, `Backend::sync` is a no-op and there is no queue to drain.

## What to do instead

If a change needs to reduce per-step readbacks for throughput, keep one
host-visible readback per emitted `Step` on the bench path, or give the bench an
explicit fence (`device_fence`) and update ADR-0009. Re-run `just bench` on the
4090 after any change to where `DiffusionTrainer` reads the loss back, and
compare against the ADR-0009 baseline on a matching configuration.
