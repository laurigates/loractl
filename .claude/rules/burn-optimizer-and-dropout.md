---
paths:
  - "crates/**/*.rs"
---
# Burn `optim`/`nn` Semantics — Weight Decay Is Coupled by Default; Dropout Is Train-Only and Free

Two burn-framework facts that bit `loractl` this session (fixing the dead-config
bugs #43 `weight_decay` and #44 `dropout`). Both are the kind of quiet
correctness trap where the code compiles, runs, and passes a suite that only
ever exercised the zero/default value. Sibling to
[`burn-lazy-param-init.md`](burn-lazy-param-init.md) (the RNG-stream fact below
connects back to it).

## 1. `AdamConfig` weight decay is **coupled L2**, not decoupled AdamW

burn's `AdamConfig` + `WeightDecayConfig` applies **coupled** L2 regularization:
`decay.rs` folds the penalty straight into the gradient
(`tensor.mul_scalar(self.penalty).add(grad)`). That is classic Adam+L2, **not**
the decoupled "AdamW-style" decay most configs mean when they say "weight decay".

For **decoupled** decay use `AdamWConfig` (its step applies
`decay_rate = lr * weight_decay` as a separate shrink, independent of the
gradient). At `weight_decay = 0.0` the two optimizers are **numerically
identical**, so switching `AdamConfig → AdamWConfig` is safe for existing
numerics goldens (they all pin `weight_decay = 0.0`).

```rust
// Decoupled (matches a config documented as "AdamW-style"):
let mut adam = AdamWConfig::new()
    .with_weight_decay(config.optim.weight_decay as f32) // AdamWConfig default is 1e-4 — set explicitly
    .init::<B, Model<B>>();
```

- **Verify the field's *intended* semantics before wiring.** `loractl`'s
  `OptimConfig::weight_decay` doc said "Decoupled weight decay (AdamW-style)", so
  wiring it into `AdamConfig` (coupled) would have been a silent correctness bug
  that matched neither the doc nor user expectation. Coupled vs decoupled is not
  a cosmetic choice.
- **Kill-test the wiring, not just the presence.** A run with `weight_decay = 1.0`
  vs `0.0` (same seed) must produce **different** loss trajectories; identical
  streams mean the value is being dropped (the original bug). Don't assert the
  field exists — assert it *changes training*.

## 2. burn `Dropout` is identity at `prob == 0.0` **and** on a non-autodiff backend

`burn::nn::Dropout::forward` early-returns the input unchanged when
`!B::ad_enabled(&input.device()) || self.prob == 0.0`. Two consequences worth
banking:

- **Train/eval separation is free — no manual flag.** Training runs on
  `Autodiff<B>` (`ad_enabled` true → dropout active); inference/sampling runs on
  a plain backend or via `.valid()` (`ad_enabled` false → dropout is identity).
  You do **not** thread a `training: bool` through the model; the backend type
  already encodes it. Put the `Dropout` on the LoRA path (`B(A(dropout(x)))`),
  never on the base, and it just works in both modes.
- **At `prob == 0.0` it draws NO RNG** (the early return skips the Bernoulli
  mask). So adding a `Dropout` field is a *true* no-op at the default — it does
  not perturb the frozen-base seed stream (see `burn-lazy-param-init.md`) and the
  numerics goldens stay bit-identical. `Dropout` also carries **no `Param`**, so
  it doesn't affect module snapshots / adapter save-load filters.
- **Kill-test both directions.** On `Autodiff<B>`, two applications of a
  `prob > 0` dropout to the same tensor must **differ** (masks are random →
  dropout is live). On a plain backend it must be **identity** (eval safety). A
  test that only runs at `prob = 0` proves nothing.

## 3. Persisting optimizer state: the record is `ParamId`-keyed and `hashbrown`-typed

Found while making resume lossless (#180, `crates/loractl-core/src/resume.rs`).
`Optimizer::to_record`/`load_record` are the only way in or out of an
`OptimizerAdaptor`'s state, and two properties of that record decide whether a
persisted copy actually restores anything:

- **`ParamId` is random per construction, not a stable counter.**
  `ParamId::new()` → `IdGenerator::generate()` → `gen_random::<[u8; 8]>()`
  (`burn-std`'s `id.rs`), so the ids a fresh module build mints in the next
  process share nothing with this one's. `Optimizer::Record` is
  `HashMap<ParamId, AdaptorRecord<O, B>>`, so **a record serialized as-is
  restores nothing on the next run** — every lookup misses, and it misses
  *silently*: training proceeds with re-warmed moments and looks fine. Key any
  on-disk copy by something stable (a module/site path) and rebuild the
  `ParamId` map against the freshly-built module at load. Also note
  `Param::from_tensor` mints a **new** id, so anything that replaces a param
  (an import, a merge) must have its optimizer state read *after* that
  replacement, not before.
- **The record's map is `hashbrown::HashMap`, not `std`'s.** Naming
  `std::collections::HashMap` in your signature is an E0308 whose message reads
  as if the two were the same type ("similar names, actually distinct types").
  The fix that needs no new dependency: seed the map from a fresh optimizer's
  `to_record()` (empty, correctly typed) and `insert` into that. Do **not** add
  `hashbrown` directly — a version skew from burn's own pin reintroduces the
  same error.
- **`AdamWState` is `{ momentum: AdaptiveMomentumState { time, moment_1,
  moment_2, max_moment_2 } }`**, converted via `AdaptorRecord::into_state::<D>` /
  `from_state::<D>`. `time` is a scalar step count, and it is **not** implied by
  the moments: persisting only `moment_1`/`moment_2` silently loses AdamW's bias
  correction, so the first step after a "lossless" resume is wrong by a factor
  nobody can see. `max_moment_2` is `None` while `amsgrad` is off
  (`AdamWConfig::new()`'s default) — assert that rather than assume it, so
  turning amsgrad on later fails loudly instead of dropping half the state.
- **Kill-test the round-trip, not the file.** §1's doctrine applies verbatim: a
  continuation with restored moments must produce **different** weights from one
  with the sidecar deleted. Asserting the file exists cannot distinguish a
  working restore from a file that is written and then ignored — which is the
  failure this whole family produces.

## Rationale

Both facts share a shape: the default value (`weight_decay = 0`, `dropout = 0`)
makes the wrong wiring and the right wiring behave identically, so a suite that
only tests the default can't tell them apart. The bug hides until a user sets a
non-zero value and silently gets nothing. The roadmap (M9–M14: Krea 2 diffusion
LoRA on GPU + QLoRA) will lean on both optimizers and dropout, so pin the
*behavior* of these knobs, not their presence — and pick the coupled/decoupled
and train/eval semantics deliberately, against burn's actual source, not from
the API's surface resemblance.
