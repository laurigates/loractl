# Burn's `Param<T>` Is Lazily Initialized — "Reseed Then Reconstruct" Isn't Enough Alone

burn's `Param<T>` (`burn-core::module::param::base`) has a documented "Core
Lazy Initialization Architecture": a fresh `Linear`/`Embedding`/etc. built via
`SomeConfig::init(device)` does **not** draw from the device RNG at
construction time. `Initializer::init_with` returns
`Param::uninitialized(id, closure_that_draws_rng, device, ...)` — the actual
`uniform_draw`/`normal_draw` call only fires on the **first** `.val()` access
(or any other access that forces the `SyncOnceCell`), not when `init()`
returns.

## The trap

A design that assumes "seed the device RNG, then construct the module" alone
pins a frozen layer's random values is **wrong** unless the first access to
that layer happens at the same point in the RNG stream every time. Anything
that draws randomness *between* construction and first use — generating
synthetic training batches, drawing dropout masks, building another module —
shifts where in the RNG stream the lazy draw actually happens, so the
materialized values silently differ from what a naive "same seed, same code"
intuition predicts.

Concretely, this bit M4's adapter round-trip (issue #3, PR #11): `BurnTrainer`
seeds the device, constructs `LoraMlp` (frozen `fc1`/`fc2.base` are lazy at
this point), then calls `select_batches` — which draws `Tensor::random` for
synthetic batch data — **before** the training loop's first `forward()` call
triggers the lazy init. So the frozen base's real values depend on how much
RNG `select_batches` consumed, not just the seed. `adapter::load_adapter`'s
"reseed, then reconstruct, then forward immediately" does **not** replicate
that intervening draw — a real trained adapter's reloaded frozen base would
silently diverge from the base that was actually trained against. A narrow
round-trip test that forwards immediately after construction in both the
"train" and "reload" paths (no intervening RNG-consuming call in either) will
**pass** while masking this — it doesn't reproduce the real
`BurnTrainer`-then-`load_adapter` sequence, so it can't catch the divergence.

## The fix

Force eager materialization of every param whose *value* (not just shape)
must be pinned independent of caller ordering, immediately after
construction — before returning from the constructor:

```rust
let model = Self { fc1, fc2 };

// Force the frozen base to materialize NOW, right after `device`'s RNG was
// (presumably) seeded — independent of whatever the caller does afterward
// (e.g. drawing synthetic batch data before the first forward pass).
let _ = model.fc1.weight.val();
if let Some(bias) = &model.fc1.bias {
    let _ = bias.val();
}
let _ = model.fc2.base.weight.val();
if let Some(bias) = &model.fc2.base.bias {
    let _ = bias.val();
}

model
```

(`crates/loractl-core/src/model.rs`, `LoraMlp::new` — landed in PR #11's
review-fix commit.) Only the tensors whose *actual random values* must be
reproducible need this — trainable params that get overwritten by a load
(e.g. `lora_a`/`lora_b`) don't, since their lazy-random init is discarded
anyway.

## When it bites

- Any new model/module in this crate that relies on "reseed the device, then
  reconstruct" to reproduce a frozen/random component deterministically —
  the pattern this crate already uses for `LoraMlp`'s frozen base and would
  reuse for any future architecture with the same "frozen random layer +
  trained delta" shape.
- A round-trip/determinism test that constructs-then-immediately-forwards in
  both the "original" and "reloaded" paths will not catch this — it needs to
  either force eager materialization (as above) or exercise the *real*
  intervening code path (e.g. drive the actual `BurnTrainer.train()` loop,
  not a narrower hand-rolled repro) before asserting reproducibility.

## Rationale

burn's own module doc names this "lazy initialization" explicitly, but it's
easy to miss because `LinearConfig::init(device)` *looks* eager (it returns a
fully-typed `Linear<B>` synchronously) — the deferred RNG draw is an
implementation detail one level down, in `Param`'s `SyncOnceCell`. Pinning it
at construction time, once, in the one place a frozen layer's determinism
actually matters, is cheaper than re-discovering the divergence via a
silently-wrong sampled output from a real trained adapter.
