# 0002 — Adapter file format and `sample` semantics (M4)

- **Status:** Accepted
- **Date:** 2026-07-04
- **Milestone:** M4 (issue #3 — "sampling & adapter I/O")
- **Deciders:** loractl maintainers
- **References:** Builds on [ADR-0001](0001-first-real-target-model.md), which
  scoped the current target model (`LoraMlp`, a synthetic/MNIST-shaped
  classifier trained by `BurnTrainer`) and explicitly deferred a real
  language-model target (SmolLM2/GPT-2 *training* integration) to a future
  milestone beyond M4/M5.

## Context

M2 (#1) and M3 (#2) left the trainer writing full-model checkpoints as
burn-native MessagePack (`.mpk`) via `NamedMpkFileRecorder`, and left
`loractl sample` as a stub that `anyhow::bail!`s unconditionally. Issue #3
requires all four of:

1. Adapters save to and load from `.safetensors`, round-tripping forward
   output bit-for-bit (or within documented tolerance).
2. `loractl sample` produces output from a saved adapter — no more bailing.
3. Optional in-training validation samples are written and reported via
   `TrainEvent::Sample`.
4. The adapter tensor-naming scheme is documented.

The complication is what ADR-0001 already flagged: the only model
`BurnTrainer` trains today, `LoraMlp`, is a synthetic/MNIST-shaped
**classifier** — there is no tokenizer, and no text in or out of it. A real
generative "sample a language model" milestone needs a real base LM
(SmolLM2/GPT-2-family) actually wired into the *training* loop, which ADR-0001
deliberately scoped out of M4 and M5. So "make `sample` work" cannot mean
"generate text" without either faking it or silently expanding M4's scope
into that future milestone. Both are worse than the alternative below.

## Decision

### 1. Adapter-only `.safetensors` + a `<path>.json` metadata sidecar

`crates/loractl-core/src/adapter.rs` persists **only the two trainable LoRA
factors** — `fc2.lora_a.weight` and `fc2.lora_b.weight` — via
`burn-store`'s `SafetensorsStore`, filtered with a `PathFilter::with_regex`
matching exactly those two paths. The frozen base (`fc1.weight`, `fc1.bias`,
`fc2.base.weight`, `fc2.base.bias`) is **never** written — persisting it would
just re-invent the old `.mpk` full-model checkpoint with extra steps, and
defeats the point of a LoRA "adapter" as a small, portable artifact. This
mirrors the shape of community LoRA conventions (HF PEFT names its trainable
factors `lora_A`/`lora_B`) without claiming literal PEFT interoperability —
`LoraMlp` is not a downloadable public base model, so there is no shared base
checkpoint to actually be interoperable *with*.

A LoRA-only file is not self-describing on its own: reloading it needs the
frozen base's *exact* weights back, plus the adapter's shape (rank, alpha,
dimensions). We considered writing that into the safetensors file's own
`__metadata__` header via `SafetensorsStore::metadata(key, value)`, but
**verified against the `burn-store` 0.21.0 source**
(`burn-store-0.21.0/src/safetensors/store.rs`) that this is **write-only**:
the only accessor, `get_metadata`, is a private inherent method used
internally, with no public API to read a file's metadata back after opening
it for loading. So instead, `save_adapter` writes a plain `<path>.json`
sidecar (`AdapterMeta`) next to the `.safetensors` file, holding:

- `seed` — the training run's RNG seed. `load_adapter` calls `B::seed(device,
  meta.seed)` and then immediately constructs a fresh `LoraMlp::new(...)` of
  the persisted shape — the *exact* same seed → construct ordering
  `BurnTrainer::train` uses. Because burn's RNG is deterministic per seed and
  no other draws happen between seeding and model construction in either code
  path, this reconstructs the frozen `fc1`/`fc2.base` **bit-identically**,
  without ever writing those weights to disk.
- `rank`, `alpha`, `d_in`, `hidden`, `out` — derived self-describingly from
  the model's own tensor shapes at save time (`model.fc2.lora_a.weight.dims()`
  etc.), never hardcoded — enough to reconstruct `LoraMlp::new`'s exact call.

This is the same two-file shape as HF PEFT's own
`adapter_model.safetensors` + `adapter_config.json` convention — arguably
*more* interoperable in spirit than a custom embedded-metadata scheme, since
it's a plain, framework-agnostic JSON file any tool can read without even
touching the safetensors library.

`crates/loractl-core/src/adapter.rs`'s module docs are the canonical
reference for the tensor-naming scheme (acceptance criterion 4).

### 2. `--prompt` deterministically seeds a synthetic input, not text generation

`loractl sample --prompt <text>` cannot generate text — there is no tokenizer
and no language model. The rejected alternatives (silently no-op'ing, or
fabricating plausible-looking fake text) both misrepresent what the tool did.
Instead, `crates/loractl-core/src/sample.rs` treats `--prompt` as a
**deterministic seed** for a synthetic classifier input:

- `seed_from_prompt(None) = 0` (a fixed, documented default); `Some(prompt)`
  hashes the prompt's UTF-8 bytes with a hand-implemented FNV-1a (not
  `std::collections::hash_map::DefaultHasher`, whose algorithm/output the
  standard library explicitly does **not** guarantee stable across Rust
  versions — using it would silently break "the same prompt always
  reproduces the same sample" the next time the toolchain is upgraded).
- The seed drives a small dependency-free splitmix64 generator that produces
  a `Vec<f32>` input vector, scaled to roughly the same spread as
  `BurnTrainer`'s own synthetic training data. This is explicitly **not** a
  statistically rigorous RNG — it's a toy/demo input generator, in the same
  honestly-documented spirit as `burn_trainer.rs`'s own synthetic Gaussian
  blobs.
- The model's real forward pass then runs on that input and reports real
  logits/predicted class — genuine, reproducible model output, just not text.

The tradeoff: `loractl sample` is honest about what it does (seeds a
synthetic input deterministically) rather than appearing to do something it
doesn't (generate text from a prompt). Real generative sampling is explicitly
deferred to whichever future milestone actually wires a language model
(SmolLM2/GPT-2-family) into the training loop — the same follow-on ADR-0001
already flagged, not a new commitment made here.

`sample::run_sample` never calls `Backend::seed` itself (only `seed_from_prompt`
→ an explicit `u64` does), so it's safe to call from two different contexts
without any RNG-ordering hazard: the CLI's cold `load_adapter` path (which
*does* reseed the device, to reconstruct the frozen base) and
`BurnTrainer`'s in-training validation-sample path (a live model already
built from a seed advanced far past construction).

### 3. In-training validation samples via a new `output.sample_every` config field

`OutputConfig.sample_every: u64` (default `0` = disabled) mirrors
`checkpoint_every`'s shape. When `> 0` and `step % sample_every == 0`,
`BurnTrainer` runs one `sample::run_sample` call against a **fixed** probe
seed (`VALIDATION_SAMPLE_SEED = 0`, a constant, not per-call-random) and
writes a small `sample-{step}.json` report (`step`, `predicted_class`,
`logits`), emitting `TrainEvent::Sample { step, path }` (an event variant that
already existed in `event.rs` prior to this milestone). Using the *same*
fixed probe every time — rather than a fresh random one per call — is the
actual point of a "validation sample": it lets someone watch one fixed
input's prediction/logits evolve across successive `sample-N.json` files as
training progresses, instead of comparing unrelated random probes.

The checkpoint-due and sample-due checks share one `model.valid()` snapshot
per step (computed once, only when either is due) rather than cloning the
model's weights twice.

## Alternatives Considered

**Keep `NamedMpkFileRecorder` (`.mpk`) as-is, add safetensors as a second,
optional export path.** Rejected: issue #3's whole point is interoperable
adapter I/O, and maintaining two parallel formats (one full-model, one
adapter-only) doubles the I/O surface for no benefit — nothing downstream
needs the `.mpk` full-model checkpoint once the adapter-only format can
reconstruct an equivalent model from a seed.

**Embed metadata in the safetensors file's own header via
`SafetensorsStore::metadata(key, value)`.** Rejected: write-only in
burn-store 0.21 (see above) — there is no way to read it back after opening a
file for loading, so it can't actually round-trip the seed/shape.

**Wire a real language model (GPT-2/SmolLM2) into the *training* loop now, so
`sample --prompt` could do real generation.** Rejected: ADR-0001 already
scoped this out of the tracked M4/M5 milestones — it needs a tokenizer, a text
dataset, and a training-loop redesign around sequence data, a substantially
larger undertaking than "sampling & adapter I/O" as scoped in issue #3. Doing
it here would silently balloon M4's scope well past what the issue asks for.

**Silently no-op or fabricate plausible-looking output for `--prompt` without
documenting the seeding behavior.** Rejected on the same honesty grounds as
`burn_trainer.rs`'s existing synthetic-data warnings: loractl says what it
actually did, always.

## Consequences

**Positive**

- `loractl sample` and the in-training validation-sample path are both real,
  working code paths — issue #3's four acceptance criteria are all satisfied.
- The adapter file is lean (two tensors) and the sidecar is a plain JSON file
  any tool can inspect — no custom binary metadata format to maintain.
- `sample::run_sample`'s RNG-independence (no `Backend::seed` call inside it)
  makes it safely reusable from both the cold CLI path and the live
  in-training path with no ordering hazard.
- The design is honest about `LoraMlp`'s nature: nothing here claims text
  generation that doesn't happen.

**Negative / costs**

- The adapter format depends on the frozen base being *exactly*
  reconstructible from a seed — if `LoraMlp::new`'s internal RNG draw order
  or the trainer's seed→construct ordering ever changes, old adapter files
  become silently unloadable-with-wrong-base (a shape mismatch would error
  loudly; a *different* Kaiming draw sequence with the *same* shape would not
  — it would silently give a different, wrong frozen base). This is
  documented in `adapter.rs`'s module docs as a load-bearing ordering
  invariant.
- `--prompt`'s seeding effect, while honestly documented, is a real UX
  compromise: it looks like a language-model CLI flag but isn't one yet.
- A new real (non-dev) dependency: `serde_json` moves from `[dev-dependencies]`
  to `[dependencies]` in `loractl-core`, since `adapter.rs`/`burn_trainer.rs`
  now use it at runtime, not just in tests.

**Risks & mitigations**

- *Frozen-base reconstruction silently drifting* → pinned by
  `tests/adapter_roundtrip.rs`'s bit-exact forward-output comparison after a
  real (non-zero-`lora_b`) training step, which would fail loudly if the
  reconstruction ever diverged.
- *A trivial "round-trip" that doesn't actually prove the load path* →
  the round-trip test runs one real optimizer step before saving, specifically
  so `lora_b` moves off its zero-init (a freshly constructed, all-zero adapter
  would trivially "round-trip" even through a broken load path).

## References

- Issue #3 (M4), roadmap in `README.md`.
- `crates/loractl-core/src/adapter.rs` — the adapter-only safetensors
  save/load + tensor-naming scheme (its module docs are the primary
  reference for acceptance criterion 4).
- `crates/loractl-core/src/sample.rs` — deterministic sampling +
  `seed_from_prompt`.
- `crates/loractl-core/src/burn_trainer.rs` — the in-training
  checkpoint/sample wiring and the seed→construct ordering invariant.
- `crates/loractl-core/tests/adapter_roundtrip.rs` — the round-trip proof.
- `docs/adrs/0001-first-real-target-model.md` — the target-model scoping this
  ADR builds on.
- burn 0.21 sources: `burn-store-0.21.0/src/{traits,filter,safetensors/store,
  apply_result}.rs`.
- HF PEFT's `adapter_model.safetensors` + `adapter_config.json` convention
  (the two-file shape this mirrors in spirit).
