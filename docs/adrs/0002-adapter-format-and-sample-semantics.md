# 0002 — Adapter format and sample semantics (M4)

- **Status:** Accepted
- **Date:** 2026-07-04
- **Milestone:** M4 (issue #3 — "sampling & adapter I/O")
- **Deciders:** loractl maintainers

## Context

M2 (#1) shipped a real `BurnTrainer` that checkpoints and saves the final
`LoraMlp` adapter as a burn-native MessagePack record (`.mpk`, via
`NamedMpkFileRecorder`) — an honest but explicitly non-interoperable stopgap
(see the M2 module docs: "not the interoperable safetensors format, which is
milestone 4"). M4 (#3) has two coupled acceptance criteria that this ADR
covers:

1. Adapters save to and load from `.safetensors`, round-tripping to the
   original forward output (bit-for-bit or a documented tolerance).
2. `loractl sample` produces output from a saved adapter instead of bailing,
   and an in-training validation sample is written and reported via
   `TrainEvent::Sample`.

Two things shape the decisions below, both verified against source rather
than assumed:

- **`burn-store` 0.21's safetensors metadata is write-only.**
  `SafetensorsStore::metadata(key, value)` lets you *add* custom
  `__metadata__` entries when saving, but there is no public method to read
  them back out after opening a file for a load — the crate's own accessor
  (`SafetensorsStore::get_metadata`, `safetensors/store.rs`) is private. So
  embedding the reconstruction parameters (seed, rank, alpha, layer widths)
  in the safetensors header itself is a dead end for round-tripping them.
- **The trained model, `LoraMlp` (`crates/loractl-core/src/model.rs`), is a
  small synthetic classifier with no tokenizer.** It is not attached to any
  downloadable public base model (unlike M3's GPT-2). ADR-0001 already scopes
  a real language-model training loop (GPT-2/SmolLM2 generation) as future
  work beyond M4/M5. `sample` therefore cannot honestly mean "generate text."

A third fact surfaced during implementation, not anticipated going in: burn
0.21's `Param<Tensor>` is **lazily initialized** — a freshly constructed
`Linear`'s weight/bias don't actually draw from the RNG until first accessed
(deref / `.val()`), which by default is whenever the model's first `forward`
call happens to touch them. Left alone, this makes "reseed, then construct"
an unreliable way to reproduce a model's frozen weights: anything that
consumes the RNG between construction and first use (e.g. a trainer drawing
synthetic batch data before its first forward pass) shifts what the lazily
materialized weights turn out to be, and a reload path that doesn't replay
that same intervening RNG consumption would silently diverge. See
**Consequences** for the fix this required in `LoraMlp::new` itself.

## Decision

### A. Adapter-only safetensors + a JSON sidecar

Persist **only the two trainable LoRA tensors** —
`fc2.lora_a.weight` and `fc2.lora_b.weight` — to a real `.safetensors` file,
using `burn-store`'s `SafetensorsStore` with an inclusive `PathFilter`:

```rust
let filter = PathFilter::new().with_regex(r"\.lora_(a|b)\.weight$");
let mut store = SafetensorsStore::from_file(path).filter(filter).overwrite(true);
model.save_into(&mut store)?;
```

`PathFilter::new()` matches nothing by default; `with_regex` OR's in this one
inclusive rule, so the frozen base (`fc1`, `fc2.base`) never touches the file
— this is deliberate, not an oversight, and is asserted by
`tests/adapter_roundtrip.rs::saved_file_contains_only_the_lora_tensors`.

Alongside `path`, write a **JSON sidecar** at `<path>.json` (the whole
filename with a literal `.json` appended — NOT `Path::with_extension`, which
would replace the `.safetensors` suffix instead of appending to it):

```json
{ "seed": 42, "rank": 16, "alpha": 16.0, "d_in": 784, "hidden": 256, "out": 10 }
```

This is the *only* place a working metadata read/write round-trip exists
given burn-store's write-only header API (see Context). It is also close in
spirit to how Hugging Face's PEFT ships `adapter_config.json` alongside
`adapter_model.safetensors` — a familiar shape, not a coincidence.

**Reconstructing the frozen base.** `load_adapter` reads the sidecar, reseeds
the device (`B::seed(device, meta.seed)`), and constructs a fresh `LoraMlp`
from the sidecar's shape/rank/alpha — the *same* determinism trick the
trainer itself relies on for reproducible runs. Because burn's RNG is
deterministic given a seed (once the lazy-initialization timing issue below
is pinned down), this regenerates `fc1`/`fc2.base` bit-identically to the
original run, so the file only needs to carry the two tensors that actually
diverged from their initial values. `load_from` is called with
`allow_partial(true)` and the result is asserted on `result.errors.is_empty()`
and `result.applied.len() == 2` — **not** `result.missing.is_empty()`: the 4
frozen-base tensors are legitimately absent from this adapter-only file, the
same documented footgun ADR-0001 calls out for `unused` in the GPT-2 loader.

**Tensor-naming scheme** (documented in `crates/loractl-core/src/adapter.rs`'s
module docs, the artifact for this acceptance criterion): `fc2.lora_a.weight`
and `fc2.lora_b.weight`, mirroring the *pattern* of community LoRA
conventions (PEFT's `lora_A`/`lora_B`) without claiming literal interop —
`LoraMlp` isn't attached to a downloadable public base model, so there is no
PEFT checkpoint to actually be compatible *with*, only a recognizable naming
pattern and adapter-only shape.

### B. `loractl sample` runs one deterministic forward pass, not text generation

`sample --prompt <text>` loads the adapter and runs **one forward pass on a
deterministic, seed-derived synthetic input**:

- No `--prompt`: seed `0`.
- `--prompt <text>`: seed = FNV-1a hash of the prompt bytes (`crate::sample::seed_from_prompt`).
  FNV-1a is used instead of `std::collections::hash_map::DefaultHasher`
  because the latter's algorithm is explicitly unspecified and may change
  across Rust releases — that would silently break "the same prompt always
  reproduces the same sample" the next time the toolchain changes.
- The seed feeds a small, dependency-free, splitmix64-based generator
  (`crate::sample::run_sample`) that builds the input vector **without**
  touching burn's `Tensor::random`/global device RNG — sample-input
  generation must never interfere with `load_adapter`'s own use of `B::seed`
  for frozen-base reconstruction, nor with a live training loop's RNG state.

This is an honest, reproducible effect — the same prompt text always yields
the same output — but it is **not** language-model sampling: `LoraMlp` has no
tokenizer and no notion of the prompt's *content*. The CLI prints this
framing explicitly on every `sample` invocation rather than leaving it
implicit, so the honesty survives copy-pasted terminal output, not just this
document.

### C. In-training validation samples use one fixed seed per run

`config.output.sample_every` (default `0`, disabled) gates a periodic
validation sample during training, written to `sample-{step}.json` and
reported via `TrainEvent::Sample { step, path }`. Every periodic sample within
a single run uses the **same fixed seed** (`VALIDATION_SAMPLE_SEED = 0` in
`burn_trainer.rs`), deliberately **not** derived from `step`: using one fixed
probe input across every sample lets you watch that one input's
prediction/logits evolve as training progresses across the successive
`sample-{step}.json` files — that comparison is the actual value of a
"validation sample," and it would be lost if each sample used a different
random input.

## Alternatives Considered

**Keep the M2 `.mpk` full-model checkpoint.** Rejected outright — issue #3
explicitly asks for the interoperable safetensors format; `.mpk` is a burn-only
format no other tool can read.

**Embed reconstruction metadata in the safetensors header
(`SafetensorsStore::metadata`).** Rejected: write-only in burn-store 0.21 (see
Context) — there is no public API to read `__metadata__` back after opening a
file for a load. A JSON sidecar is the only reliable, inspectable alternative
without depending on a private crate internal.

**Save the full model (frozen base included) to safetensors.** Rejected:
defeats the point of an *adapter* file (small, portable, base-independent) and
duplicates data that's already fully determined by `(seed, shape)` — the
adapter-only approach is both smaller and the more honest "this is a LoRA
adapter" artifact.

**Wire GPT-2 as the actually-trained/sampled model now, so `sample --prompt`
could do real generation.** Rejected for M4: ADR-0001 already scopes GPT-2/
SmolLM2 *training-loop* integration (tokenizer, text dataset, a generation
loop) as work beyond the tracked M4/M5 milestones — a much larger, separate
undertaking. M4's job is adapter I/O and sampling *for the model that
actually trains today* (`LoraMlp`), not a premature commitment to a
generation UX the trained model can't back up.

**Derive the periodic validation-sample seed from `step`.** Considered and
rejected: a per-step seed would draw a *different* random input at every
checkpoint, making it impossible to compare "how has the model's prediction
on this one input changed" across a run — exactly the comparison a
validation sample exists to support.

## Consequences

**Positive**

- Adapters are real, portable `.safetensors` files any tool can open, at a
  fraction of the full model's size — only the 2 trained tensors, not the 4
  frozen ones.
- The sidecar is trivially inspectable (`cat *.safetensors.json`) and mirrors
  a shape practitioners already recognize from PEFT.
- `loractl sample` produces real, reproducible, honestly-labeled output
  instead of bailing — unblocking the CLI's third subcommand without
  overclaiming what a 2-layer classifier can do.
- Periodic validation samples give a concrete, comparable signal
  (`sample-{step}.json`) of how the model's behavior changes over a run,
  without adding any new dependency.

**Negative / costs**

- A two-file artifact (`.safetensors` + `.json`) instead of one; losing the
  sidecar makes the safetensors file unloadable by `load_adapter` (though
  still readable by any generic safetensors tool).
- Adapter reconstruction is **not** portable across burn/backend versions
  that change RNG algorithms or default-initializer behavior — the sidecar's
  `seed` is only meaningful against the exact `LoraMlp::new` + burn version
  that produced it. This is an accepted limitation of the adapter-only
  design, not a compatibility guarantee.

**A required fix uncovered during implementation.** burn 0.21's `Param` lazy
initialization (see Context) meant the naive "reseed, then construct"
reconstruction was *not* actually bit-identical in practice — a real
trainer's batch-generation code draws from the same RNG before ever calling
`forward()`, but `load_adapter` never replays that. The fix: `LoraMlp::new`
now eagerly forces `fc1` and `fc2.base` to materialize immediately upon
construction (`let _ = model.fc1.weight.val();` etc.), pinning their random
initialization to happen right after the device is seeded, independent of
whatever the caller does afterward. This makes the "reseed + reconstruct"
contract actually hold rather than being an accident of caller ordering; see
`crates/loractl-core/src/model.rs`. `tests/adapter_roundtrip.rs` proves the
round trip against a model that has taken one real optimizer step (so
`lora_b` has moved off zero — a round trip against a still-zero adapter would
trivially "pass" regardless of whether loading actually worked).

**Risks & mitigations**

- *Silent frozen-base divergence* if something upstream changes burn's RNG
  algorithm or `LoraMlp::new`'s initializer sequence → the round-trip test
  (`adapter_roundtrip.rs`) is the regression gate; a divergence fails loudly
  as a forward-output mismatch, not a silent wrong answer.
- *`sample --prompt` read as real generation* → the CLI prints the honest
  framing on every invocation (see Decision B); this ADR and the README's
  "Sampling & adapter I/O" section state it explicitly for anyone reading the
  design rather than just running the binary.
- *Test flakiness from burn's global RNG being a single process-wide static*
  → `cargo test` runs `#[test]` fns within one binary in parallel by default;
  `tests/adapter_roundtrip.rs` serializes its own tests with a
  `std::sync::Mutex` so sibling tests can't interleave reseed/materialize
  calls against the same shared state.

## References

- Issue #3 (M4), roadmap in `README.md`.
- `crates/loractl-core/src/adapter.rs` — `save_adapter`/`load_adapter`, the
  tensor-naming scheme, and the sidecar rationale.
- `crates/loractl-core/src/sample.rs` — `seed_from_prompt`, `run_sample`, the
  FNV-1a/splitmix64 generators.
- `crates/loractl-core/src/burn_trainer.rs` — checkpoint/final adapter writes,
  `VALIDATION_SAMPLE_SEED`.
- `crates/loractl-core/src/model.rs` — `LoraMlp::new`'s eager materialization
  of the frozen base.
- `crates/loractl-core/tests/adapter_roundtrip.rs`.
- `docs/adrs/0001-first-real-target-model.md` — the `ApplyResult`
  `missing`-vs-`unused` footgun this ADR's load path also avoids, and the
  scoping of GPT-2/SmolLM2 training-loop integration beyond M4/M5.
- burn 0.21 sources: `burn-store-0.21.0/src/safetensors/store.rs` (metadata
  write-only), `burn-store-0.21.0/src/filter.rs` (`PathFilter` inclusive OR
  semantics), `burn-core-0.21.0/src/module/param/{base,tensor}.rs` (lazy
  `Param` initialization), `burn-ndarray-0.21.0/src/backend.rs` (the
  process-global RNG static).
