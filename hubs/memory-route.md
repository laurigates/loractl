---
summary: The three burn-0.21 workarounds the 24 GB training route depends on, each of which failed silently once.
anchors:
  - claim: >
      `checkpointed_step` is deliberately two-phase and NOT a custom autodiff op: a capture
      forward runs the trunk graph-free on the plain inner backend storing only each block's
      input, then backward replays each block on its own standalone graph. It is structured
      this way because a nested `Tensor::backward()` inside a `Backward::backward` impl
      deadlocks on burn 0.21. It also requires `lora.dropout == 0`, since the capture forward
      runs without autodiff.
    at:
      - crates/loractl-core/src/block_ckpt.rs > checkpointed_step
    hash: 2:e7df0cf32803
    id: c_18c63b4af2b909e00003
    verified_at: 2026-07-27T19:11:35Z
    verified_commit: 069d768374f09c4a8bfe9a8bcbb75ee863132a88
  - claim: >
      `track_adapters` re-marks each lifted adapter param with `require_grad()` while
      preserving its id, because burn 0.21's `Param::clone` rebuilds an initialized param via
      `Param::initialized(id, val())` and recomputes `require_grad` from the tensor — which is
      unconditionally false on a plain backend. Without the re-mark every replayed block
      computes zero adapter gradients, silently.
    at:
      - crates/loractl-core/src/block_ckpt.rs > track_adapters
    hash: 2:ace66b79e114
    id: c_18c63b4af37a9c900004
    verified_at: 2026-07-27T19:11:35Z
    verified_commit: 069d768374f09c4a8bfe9a8bcbb75ee863132a88
  - claim: >
      Every Mmdit checkpoint load must set `skip_enum_variants(true)` on the SafetensorsStore
      builder, because the `BaseLinear` Plain/Quant module enum injects its active variant name
      as a path segment; without it the checkpoint's `blocks.0.attn.wq.weight` never matches
      `...wq.Plain.weight`, the tensors report as unused, and the base silently stays at random
      init. `load_module` is the canonical site.
    at:
      - crates/loractl-core/src/diffusion_trainer.rs > load_module
    hash: 2:bef0ff04de40
    id: c_18c63b4af45c1f080005
    verified_at: 2026-07-27T19:11:35Z
    verified_commit: 069d768374f09c4a8bfe9a8bcbb75ee863132a88
refs: []
---

# The 24 GB training route

int4 base + block-level gradient checkpointing is the measured route onto a 24 GB card.
The monolithic step's demand was ~67.9 GiB pinned per forward; block checkpointing brings
the measured peak to ~19.4 GB. Those numbers, and the levers that turned out to be dead
ends, live in ADR-0005 — not here, because they are measurements and this gate does not
guard measurements.

What this hub guards is the three pieces of **code structure** that route depends on.
All three share a failure shape worth naming: each one compiles, runs, and produces
plausible output when broken.

| If this regresses | You observe |
|---|---|
| The two-phase step becomes a custom op | A permanent hang, no error |
| `track_adapters` loses the re-mark | Training "works", adapter grads are all zero |
| A load path drops `skip_enum_variants` | Base silently at random init; "Unused Tensors" |

That is exactly the class a green compile cannot catch, which is why they are anchored
here and not left to review.

**Boundary.** This hub guards the structure, not the measurements. A change elsewhere can
falsify the VRAM numbers in ADR-0005 while every anchor here stays green.
