# Testing — loractl

loractl-specific testing requirements. General test-tier guidance is in the
user-global rules; this file records the Rust/loractl specifics.

## Commands

| Task | Command |
|---|---|
| Full test suite | `just test` (`cargo test`) |
| One test by name | `cargo test -p loractl-core <test_name>` |
| Lint (warnings-as-errors) | `just lint` |
| Format check | `just fmt-check` |

## What to test

- **Config layering** — the YAML → env → flag precedence in `load_config` and
  the post-extraction flag overrides. Assert that a flag beats an env var beats
  the file for the same key, including nested (`__`) keys.
- **Event stream** — a `Trainer` run drives the expected sequence of
  `TrainEvent`s through the callback sink. This is how core is tested without a
  renderer.
- **Numerics (M2, #1)** — the LoRA `Module` forward = `base(x) + (alpha/rank) ·
  B(A(x))`, base frozen, only A·B trained. Verify against a Python reference on a
  tiny model (MNIST MLP) before touching a real base model.

## Interop exports: pin the *consumer's* contract, not only your own golden

An offline golden for an export (`tests/adapter_export.rs`) pins **our own
serialization convention** — the keys, transpose, and `.alpha` scalar loractl
*chooses* to write. It says nothing about whether the real consumer (ComfyUI,
kohya-ss, diffusers) actually **accepts** those keys. Those are two different
claims, and the gap between them is silent: a LoRA with unmatched keys loads
**without error** and does nothing — the worst failure shape.

That gap produced a real misdiagnosis (#137, PR #138): the Krea 2 export was
believed broken because its key names differ from community LoRAs, when in fact
ComfyUI accepts *both* forms. Nothing in the repo could settle it either way,
because the only export test compared against our own golden — which pins our
convention by construction and can't disagree with it.

The rule for any outward-facing interop export:

- **Add a consumer-contract test** alongside the self-golden one. Run the
  **real export path** over the **real site enumeration** and assert every
  on-disk key is one the consumer's key map actually contains. `build_adapters`
  is config-derived, so the full Krea 2 196-site set builds from
  `MmditConfig::krea2()` without instantiating the ~12.8B model — offline,
  seconds (`tests/krea2_lora_keys.rs`).
- **Generate the contract from pinned upstream source, never a hand-copy.**
  `reference/krea2_lora_keys_reference.py` downloads `comfy/lora.py` +
  `comfy/utils.py` at a pinned commit, extracts the real `krea2_to_diffusers`
  via `ast`, executes it, and **asserts the specific alias lines the export
  depends on are still present** before emitting a golden. A transcribed map
  drifts silently; a pinned-source generator fails loud when the contract moves.
  Regenerate with `just krea2-lora-keys-reference`; bump the commit deliberately.
- **Give the contract test teeth.** Pin that the *un-renamed* form is **not**
  accepted, so the contract assertion can't pass vacuously, and kill-test it:
  sabotaging the mapper to identity must fail it (588/588 for #137 — the exact
  count the issue claimed the shipped export had, which the test reports as 0).

Same instinct as `diagnose-at-the-failure-point` and
`verify-upstream-before-patching`: the authoritative source is the consumer's
actual code, not your own inference about what it wants.

## Coverage expectations

The first tests landed in M2 (#1): the always-run numerics proof
(`tests/lora_reference.rs`, vs a PyTorch golden), synthetic convergence
(`tests/convergence.rs`), and an opt-in real-MNIST proof
(`tests/mnist_lora.rs`, `#[ignore]`d behind `--features mnist`). The bar for new
code: every `impl Trainer` and every config-layering change ships with tests.
ML-correctness code is verified against a reference, not merely asserted to
"run".

## CI parity

Local `just fmt-check && just lint && just test` should match what CI runs. The
`justfile` is the source of truth for the gate; keep CI and the recipes in sync.
