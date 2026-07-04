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
