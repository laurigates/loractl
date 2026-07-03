# loractl

A terminal-native LoRA trainer, in Rust.

Most LoRA trainers bolt a half-baked web GUI onto a Python training core.
`loractl` inverts that: the **CLI is the primary surface** — config-driven,
completion-friendly, pipe-able — and a GUI, if anyone wants one, is just
another renderer layered on the same core over an API. The name says the
thesis: a `*ctl` tool, like `kubectl` or `systemctl`.

> **Status: early scaffold (milestone 1).** The architecture and CLI are
> real and working; the trainer is currently a dependency-free `MockTrainer`
> that exercises the full event → render pipeline without any ML. Milestone 2
> wires in [burn](https://burn.dev) and a real LoRA module. See
> [Roadmap](#roadmap).

## Why

- **The pipeline is the product.** No GUI plumbing to distract from
  dataloading, bucketing, the LoRA module, and the training loop.
- **CLI-first UX.** `clap`-generated shell completions, YAML configs with
  env/flag overrides, structured progress output.
- **GUI-optional by construction.** Core emits events; it never draws. A CLI
  renders them as a progress bar today; an HTTP API could stream the same
  events as JSON tomorrow.

## Architecture

Three layers, one direction of dependency:

| Crate | Role | Depends on |
|---|---|---|
| `loractl-core` | The pipeline: config schema, `TrainEvent` stream, `Trainer` trait. **No CLI, no stdout.** | (burn, later) |
| `loractl-cli` | The `loractl` binary. Parses args, layers config, renders events. | `loractl-core` |
| `loractl-api` *(future)* | HTTP server / language bindings for an optional GUI. | `loractl-core` |

The load-bearing rule: **`loractl-core` never imports `clap` and never
prints.** A trainer reports progress by emitting [`TrainEvent`]s through a
callback; the caller decides how to surface them. That single discipline is
what makes "someone can build a GUI later" true instead of aspirational.

## Quickstart

```sh
# Build
cargo build

# Run a (mock) training job from the example config
cargo run -p loractl-cli -- train config/examples/lora.yaml

# Override config fields from the CLI
cargo run -p loractl-cli -- train config/examples/lora.yaml --lr 5e-5 --steps 2000

# ...or from the environment
LORACTL_OPTIM__LR=5e-5 cargo run -p loractl-cli -- train config/examples/lora.yaml

# Generate shell completions
cargo run -p loractl-cli -- completions zsh > ~/.zfunc/_loractl
```

Recipes live in the `justfile` (`just` to list): `just build`, `just train`,
`just completions fish`, `just lint`, `just fmt`.

## Config

A run is fully described by a YAML config (see `config/examples/lora.yaml`).
Precedence, lowest to highest: **YAML file → `LORACTL_`-prefixed env vars →
CLI flags.** Nested keys use `__` in env vars (`LORACTL_OUTPUT__DIR=/tmp/out`).

## Roadmap

- [x] **M1 — Skeleton.** Workspace, CLI (`train`/`sample`/`completions`),
      config layering, event → progress-bar rendering, `MockTrainer`.
- [ ] **M2 — Correctness harness.** Add burn; implement a LoRA `Module`
      (freeze base, train A·B); train a LoRA on a *tiny* model (MNIST MLP)
      and verify numerics against a Python reference — **before** any large
      model. Prove the loop is right in isolation.
- [ ] **M3 — Real base model.** Load a real base model's weights into burn
      (the genuinely hard part: state-dict mapping), wire the forward pass.
- [ ] **M4 — Sampling & adapter I/O.** `loractl sample`, safetensors adapter
      read/write, validation samples during training.
- [ ] **M5 — API crate.** Expose the event stream over HTTP so a GUI can be
      built independently.

## License

MIT © Lauri Gates
