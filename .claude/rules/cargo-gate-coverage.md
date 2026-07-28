---
paths:
  - "crates/**/*.rs"
---
# A Green `cargo build` Proves Nothing About `examples/` or `#[cfg(feature)]` Code

`cargo build --workspace` compiles neither **examples** nor code behind an
inactive **feature gate**. So a change that breaks either compiles clean, the
build gate passes, and the breakage surfaces only when someone runs the example
by hand or enables the feature — often on the GPU box, days later. The gates
that actually cover them are `just lint` (`cargo clippy --all-targets`) and the
`just lint-<feature>` recipes, which CI runs as its `feature-lints` job.

## What each gate actually compiles

| Command | lib + bins | `examples/` | `#[cfg(feature = "x")]` |
|---|---|---|---|
| `cargo build --workspace` | ✅ | ❌ | ❌ |
| `cargo test --workspace` | ✅ | ❌ | ❌ |
| `just lint` (`clippy --all-targets`) | ✅ | ✅ | ❌ |
| `just lint-mnist` / `-wgpu` / `-mmdit-real` / … | ✅ | ✅ | ✅ (that one) |

**`--all-targets` is the examples lever; `--features` is the cfg lever. Neither
implies the other**, which is why `feature-lints` is a separate CI job and why
`just lint` alone is not sufficient before touching a feature-gated path.

## Evidence — it bit twice in one session (2026-07-27, #165)

Both times the same shape: green build, broken code.

1. **Adding a `TrainEvent` variant.** Every exhaustive `match` was updated and
   `cargo build --workspace` passed — while `crates/loractl-core/examples/step_probe.rs`
   still had a non-exhaustive match. Only `just lint` caught it.
2. **The `PhaseName` refactor.** `cargo build -p loractl-core` passed while
   `burn_trainer.rs`'s two mnist-gated `Phase` emits were still broken. Invisible
   until built with `--features mnist`.

`burn-store-skip-enum-variants.md` documents the *inverse* trap (code that
compiles under every gate and is still wrong at runtime); this rule is about
code that never got compiled at all.

## Corollary: a string vocabulary that crosses a module or feature boundary drifts silently

Incident 2 had a second cause worth naming. The phase-name vocabulary lived as
`const PHASE_*: &str` **private to `diffusion_trainer.rs`**, so `burn_trainer.rs`
spelled `"dataset"` as a bare string literal — twice, behind the `mnist` feature.
Nothing could catch it: not the type system (both are `&str`), not the build (the
feature was off), not a test (the path is networked).

**When a vocabulary is shared across modules *and* is part of a wire contract,
make it a compiler-enforced closed set.** A `#[serde(rename_all = "snake_case")]`
enum serializes to exactly the same tokens a `&str` constant did, so the wire
cost is **zero** — and the drift becomes a compile error the moment the type
changes. That is precisely how the `"dataset"` drift was finally found: it had
been in the tree, uncaught, until `name: String` became `name: PhaseName`.

## Verify

Before committing a change to a shared enum, a trait's method set, or anything
with exhaustive `match`es:

```
just fmt-check && just lint
```

and, when the change touches a feature-gated path, the matching
`just lint-<feature>`. A green `cargo build`/`cargo test` is not the gate.

## Rationale

The failure is silent and delayed by construction: nothing errors, because
nothing compiled. The cost of prevention is one `just lint` (which the repo's
documented pre-commit gate already prescribes — see
[`development.md`](development.md)); the cost of missing it is a broken example
discovered by hand on the GPU host, or a feature path that has been quietly
uncompilable for weeks.
