---
id: ADR-0006
status: Accepted
date: 2026-07-22
---

# 0006 — Relative-accuracy gate for reduced-precision numerics (int8/int4/fp8/f16)

- **Status:** Accepted
- **Date:** 2026-07-22
- **Milestones:** M13–M15 memory/precision work
  ([#112](https://github.com/laurigates/loractl/issues/112);
  [#96](https://github.com/laurigates/loractl/issues/96) int8/int4,
  [#82](https://github.com/laurigates/loractl/issues/82) scaled-fp8)
- **Deciders:** loractl maintainers
- **Models on:** CAEF `docs/adrs/0006-fr11-f16-accuracy-protocol.md` (the f16
  golden-oracle protocol), ported to loractl's reduced-precision paths.

## Context

loractl verifies numerics against a fixed PyTorch golden at **absolute `1e-5`**
(`tests/lora_reference.rs`, `tests/quant.rs`, `tests/fp8.rs`, the MMDiT parity
suite). That threshold is correct for the f32 paths, and the existing quant/fp8
goldens meet it — because torch computes the *same quantized* operation, so what
they pin is that quantization is done **correctly and reproducibly** (the
dequantized weights and the `x · dequant(wq)ᵀ` forward match torch).

What no test measured is a different claim: how far quantization moves the
output from the **true full-precision answer** — the QLoRA-quality question that
int4's coarser (~7% worst-case weight error) frozen base raises directly (the
#25 ComfyUI A/B). An absolute `1e-5` is the wrong shape for that claim: int4
cannot meet it by construction, and picking any single absolute number for a
reduced-precision path is either so loose it proves nothing or so tight it fails
spuriously. The reduced-precision paths — int8/int4 (`quant.rs`), scaled-fp8
(`fp8.rs`), the f16 MMDiT path — had **no principled accuracy gate**.

## Decision

Adopt a **relative-accuracy gate**, added *alongside* the fixed-truth goldens,
not replacing them. Measure each reduced-precision forward's deviation from a
**full-precision f32 oracle** run over the *same activations* (so activation
representation is not conflated into the measure — the analog of the CAEF
protocol's "same quantized inputs"), and gate the deviation with:

```text
gate = all-finite  ∧  d_ours ≤ max(2·d_bar, floor)  ∧  d_ours ≤ ceil
```

- `d_ours` — peak-normalized max relative deviation of the reduced-precision
  output from the oracle (`accuracy::rel_deviation`).
- `d_bar` — the "known-good" deviation for a path × scheme, **calibrated once**
  from the current implementation and pinned. `2·d_bar` is the regression catch;
  it tolerates the inherent quantization error a fixed point cannot, while still
  tripping on a path that silently degrades.
- `floor` — keeps the band from collapsing below f32 rounding noise when `d_bar`
  is tiny (a near-exact path).
- `ceil` — a fixed backstop above the band that catches gross corruption (a
  broken kernel, a dtype mixup) even if `d_bar` were mis-calibrated high.

The oracle is loractl's **own** f32 unquantized forward on ndarray — no torch,
no network, so the whole gate runs offline in CI.

### Two tiers, deliberately kept

The relative gate does **not** replace the fixed-truth goldens; both stay,
because they fail on different bugs. A fixed-truth golden pins a known point and
catches any regression there. A bar-relative gate can *mask* a regression when
the bar itself drifts — CAEF ADR-0006's own autotuned-matmul-drift lesson — so
it is never trusted alone. Concretely, the `ceil` term is the fixed-truth
backstop inside the relative gate, and the existing torch goldens remain the
first tier.

### Teeth beyond the constants

A calibrated band of magic numbers can pass vacuously, so the accuracy tests add
checks a mis-set constant cannot satisfy:

- the deviation must be **non-zero** — quantization actually moved the output;
- **int4's deviation must strictly exceed int8's** on the same inputs — a
  coarser base is measurably worse, a relationship no threshold can fake, broken
  by a scheme mixup or a silent widening of one path.

## Consequences

- New `loractl_core::accuracy` module: `RelGate`, `GateOutcome`, `rel_deviation`
  — small, dependency-free, unit-tested, reusable by a future accuracy example
  or the bench harness (#110).
- New `tests/reduced_precision_accuracy.rs` gates int8 (Q8S), int4 (Q4S), and
  fp8 (e4m3fn). Calibrated `d_bar` (measured on the pinned fixtures): int8
  ~5.8e-3, int4 ~9.4e-2, fp8 ~1.5e-2. These sit above what a 12.8B site sees —
  the small fixtures give the matmul little room to average independent
  per-block rounding errors — which makes them a conservative regression band.
  Recalibrate by reading the `--nocapture` `d_ours` lines the tests print.
- **f16 is deferred**: the f16 MMDiT path is wgpu/cuda-only (burn#5162 blocks
  the wgpu route entirely), so its accuracy test needs a GPU and belongs with
  the GPU smokes, not the offline suite. The `RelGate` helper already applies to
  it unchanged when that lands.

## Alternatives considered

- **A new torch golden for the reduced-precision oracle.** Rejected: the oracle
  is loractl's own f32 path, so a torch reference adds a network/`uv` dependency
  and a second source of truth for a value we already compute exactly in Rust.
  The existing torch goldens already anchor the fixed-truth tier.
- **Tighten the absolute `1e-5` per scheme.** Rejected: a per-scheme absolute
  threshold is exactly the brittle knob this gate replaces; it carries no
  calibration and no gross-corruption backstop.
