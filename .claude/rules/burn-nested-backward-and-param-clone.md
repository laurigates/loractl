# Two burn 0.21 Autodiff Traps: Nested `backward()` Deadlocks; `Param::clone` Drops `require_grad`

Both found implementing #134 (`src/block_ckpt.rs`, PR #135); both are the
silent/hanging kind a future session would re-derive expensively. Sibling to
[`burn-lazy-param-init.md`](burn-lazy-param-init.md) and
[`burn-optimizer-and-dropout.md`](burn-optimizer-and-dropout.md).

## 1. Never call `Tensor::backward()` inside a `Backward::backward` impl

A custom autodiff op whose backward runs an inner graph's `.backward()` —
the natural design for op-level gradient checkpointing — **deadlocks
unconditionally**, even when the inner graph is fully disjoint:

- The outer backward holds its graph's `state` mutex for the ENTIRE step
  execution (`burn-autodiff/src/runtime/graph.rs`, `GraphMutexClient::backward`
  wraps `server.backward` in `graph.state.lock()`).
- Every `backward()` call ends with `GraphCleaner::cleanup_orphaned_entries()`,
  which iterates the global graph registry and locks **every** graph's mutex —
  including the outer one the same thread already holds. `parking_lot` is
  non-reentrant → permanent hang. A worker thread just converts it into a
  cross-thread embrace.

Verified 2026-07-19 by a ~30-line watchdog probe (commit `b346a27`'s message;
the probe printed "entering nested inner backward…" and never returned).

**The fix is structural, not clever**: restructure so no backward runs inside
another — `block_ckpt.rs`'s two-phase step (graph-free capture forward, then
head + reverse per-block sweep of *sequential* standalone graphs). Seed an
arbitrary cotangent `g` without a seeded-backward API via
`(out ⊙ g).sum().backward()` — exact (`1.0·g == g`).

## 2. `Param::clone` on an initialized param silently drops `require_grad`

`Param<T>::clone` (burn-core `module/param/base.rs`) has two branches:
uninitialized params copy the `require_grad` field; **initialized params
rebuild via `Param::initialized(self.id, self.val())`, which recomputes the
field from the tensor** — and `Tensor::is_require_grad()` is unconditionally
`false` on a plain (non-autodiff) backend. So the flag that
`valid()`/`from_inner` carefully preserve across the Autodiff↔inner boundary
dies on ANY clone of an initialized inner-backend param — and everything
downstream is *silently untracked*: forward runs fine, backward produces no
grads, `GradientsParams` simply omits the entries, the optimizer skips them.

Bit the #134 per-block adapter filtering (clones between `valid()` and
`from_inner`): every replayed block computed **zero adapter gradients**,
caught only by the completeness assertions in `tests/block_ckpt.rs` (assert
`2 × deltas.len()` grads present and non-zero — a value-only comparison
passes vacuously).

**Fix**: re-mark after lifting, preserving the id (the key the optimizer
routes by) — `block_ckpt.rs::track_adapters`:
`Param::initialized(weight.id, weight.val().require_grad())`.

**Test rule**: any wrapped/replayed-gradient path needs completeness teeth,
not just value comparison — missing grads are the failure mode this family
produces.

## Status

Both are upstream-report candidates (tracked in taskwarrior, project:loractl
+upstream) — verify against burn `main` first; 0.22 reworked this
neighborhood. Both workarounds live in `src/block_ckpt.rs` and are deletable
with the 0.22 migration (#79).
