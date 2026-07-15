# burn 0.21 wgpu/Metal Numerics Are Broken — Suspect the Backend Before Your Math

On this Apple-Silicon host, burn 0.21's wgpu (Metal) backend produces **silent
numeric corruption** in exactly the configuration LoRA training uses: a
`.no_grad()` model with a few tracked adapter params. Filed upstream with a
deterministic repro as [tracel-ai/burn#5162]; the third burn rule in this
directory, and the one with a hard behavioral consequence:

> **On ANY wgpu numeric anomaly (NaN loss, zero/absurd grads, drifting
> outputs), run `examples/metal_bisect.rs` and `examples/grad_compare.rs`
> FIRST — before touching loractl-side math.** A full night was spent fixing
> real (but secondary) f16 range issues in loractl while the primary defect
> was the backend all along.

## The truth table (loaded weights, deterministic, verified 2026-07-15)

| arm | forward loss | LoRA B-grads |
|---|---|---|
| ndarray f32 | 0.802559 | ground truth |
| wgpu f32, params-only | **NaN — the forward itself** | — |
| wgpu f32 + input tracked | 0.802559 (bit-identical to CPU) | ratio 1.000, all sites |
| wgpu f16, params-only | 0.8018 (≈ correct) | **all NaN** |
| wgpu f16 + input tracked | **0.7771 (wrong)** | **exactly 0.0** |

The f16 zeros are loss-scale invariant (S=64 vs S=16384) → dropped values,
**not** f16 range overflow. The exactly-zero-B signature is what the real 12B
probes showed. Tracking one extra tensor flipping f32 between NaN and
bit-exact means the defect is lazy-execution/kernel-boundary dependent.

## Discriminators already established (don't re-derive)

- **Load path is clean**: 92/92 tensors byte-identical after a wgpu
  `SafetensorsStore` load (`metal_bisect verify-load`).
- **Base-model backward is clean**: per-stage backward to the inputs is
  finite everywhere (`stages`). Only the params-only pruned backward breaks.
- **Process-warm healing, same dtype only**: a prior full backward in the
  same process makes a later params-only run come back clean (random init).
- **Not autotune** (`CUBECL_AUTOTUNE_LEVEL` no effect), **not fusion**
  (burn-fusion is not in loractl's dependency graph — plain `CubeBackend`).
- **Not distillable to plain burn**: six standalone attempts (generic
  transformer block, expand-GQA, pad-row masks, broadcast-matmul micro)
  all pass — the trigger needs the MMDiT graph.
- candle-metal bf16 is numerically sound but its allocator cannot host the
  12B model ([huggingface/candle#3464]); wgpu f32 is correct with input
  tracking but f32 weights (~49 GB) don't fit this 48 GB host. **There is no
  practical workaround** — the pure-Rust real run (#25) waits on upstream.

## The validation ladder (in order, after any burn/cubecl version bump)

1. `cargo run --release -p loractl-core --features wgpu --example metal_bisect -- no-load`
   (fixture-free; expect 0 NaN grads when fixed)
2. `metal_bisect -- verify-load / stages / adapters / workaround <bundle>`
   (mode semantics + observed baselines: the example's doc header)
3. `cargo run --release -p loractl-core --features wgpu --example grad_compare -- <bundle>`
   (deterministic value-level 4-backend comparison)
4. `just test-wgpu`, then a 2-step real-model probe, then the #25 real run.

## Watch / status

- Upstream: [tracel-ai/burn#5162] (the bug), [tracel-ai/cubecl#1375]
  (open Metal if/else miscompilation, same cubecl 0.10.0 — possibly related).
- burn 0.22 (`main`) has graph-capture/fusion/memory rework in exactly this
  neighborhood — re-run the ladder there. The migration itself is
  milestone-scale (backend-erased `Tensor`); map and plan live in issue #79.

[tracel-ai/burn#5162]: https://github.com/tracel-ai/burn/issues/5162
[tracel-ai/cubecl#1375]: https://github.com/tracel-ai/cubecl/issues/1375
[huggingface/candle#3464]: https://github.com/huggingface/candle/issues/3464
