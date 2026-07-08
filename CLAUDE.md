# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`loractl` is a terminal-native LoRA trainer in Rust: a **CLI-first** tool where
a GUI, if ever built, is just another renderer over the same core (the name is
a deliberate `*ctl` reference, like `kubectl`). It is an early-stage learning
project — see the roadmap in `README.md` and the tracking issues (#1–#4).

**Current status:** all five roadmap milestones (M1–M5, #1–#4) have landed.
The default trainer is a real, burn-backed `BurnTrainer` that trains a
**synthetic** LoRA-MLP demo (offline, fast), pinned against a PyTorch numerics
golden; real MNIST is behind an opt-in `mnist` feature and the dependency-free
`MockTrainer` remains available for pipeline testing. M3 added a real GPT-2
safetensors loader with forward-pass parity vs PyTorch; M4 added portable
`.safetensors` adapter I/O and deterministic sampling; M5 added `loractl-api`,
which streams the same `TrainEvent`s over HTTP/SSE (wire contract in
`docs/api/events.md`). See the roadmap in `README.md`.

**Next direction (M6–M14, #17–#25):** training LoRAs for **Krea 2**, an
open-weights ~12B rectified-flow **image** model — a different domain that
reuses this architecture but needs a greenfield burn diffusion stack (MMDiT
denoiser, VAE, Qwen 3 VL text encoder, flow-matching objective, GPU + QLoRA).
Strategy and gap analysis: [ADR-0004](docs/adrs/0004-krea2-image-diffusion-target.md).

## Commands

Recipes live in the `justfile` (`just` to list). Cargo directly also works.

| Task | Command |
|---|---|
| Build the workspace | `just build` (`cargo build`) |
| Run the CLI | `just run <args>` (`cargo run -p loractl-cli -- <args>`) |
| Train from a config (synthetic demo) | `just train [config]` — defaults to `config/examples/lora.yaml` |
| Serve the HTTP/SSE API | `just serve` (`cargo run -p loractl-api`; bind addr via `LORACTL_API_ADDR`, default `127.0.0.1:3000`) |
| Generate shell completions | `just completions [shell]` (e.g. `just completions fish`) |
| Lint (warnings-as-errors) | `just lint` (`cargo clippy --all-targets -- -D warnings`, default/offline features) |
| Lint the opt-in mnist path | `just lint-mnist` (compiles the networked dataset deps) |
| Format / check format | `just fmt` / `just fmt-check` |
| Tests (offline) | `just test` (`cargo test`) — numerics vs PyTorch golden + synthetic convergence |
| Real-MNIST convergence proof | `just test-mnist` (opt-in, downloads MNIST) |
| Regenerate the numerics golden | `just reference` (needs `torch` via `uv`) |
| One test by name | `cargo test -p loractl-core <test_name>` |

Before committing, the meaningful gate is `just fmt-check && just lint` — CI
parity is intended (the `justfile` mirrors what CI should run). rustfmt is
default style; expect it to reflow multi-line signatures onto one line.

## Architecture — the one rule that matters

The workspace is three crates:

| Crate | Role |
|---|---|
| `loractl-core` | The pipeline: `TrainConfig` schema, `TrainEvent` stream, `Trainer` trait, `MockTrainer`, the LoRA/GPT-2 modules and `BurnTrainer`. |
| `loractl-cli` | The `loractl` binary — parses args, layers config, renders events. |
| `loractl-api` | The `loractl-api` binary — serializes the same `TrainEvent`s over HTTP/SSE for a GUI; renders nothing itself. Wire contract: `docs/api/events.md`. |

**Load-bearing invariant: `loractl-core` emits events; it never renders.**
Concretely, core must not import `clap` and must not `println!`/write to
stdout/stderr. A `Trainer` reports progress by calling a `&mut dyn
FnMut(TrainEvent)` sink; the *caller* decides how to surface it. The CLI
renders `TrainEvent`s as an `indicatif` progress bar (see the match arm in
`crates/loractl-cli/src/cli.rs`); `loractl-api` serializes the same events
as JSON/SSE. **This is what makes "a GUI can be built separately" real — do
not break it** by having core print or by having the CLI reach into training
internals.

Dependency direction is strictly `cli → core` and `api → core`. Core has no
upward dependencies and no front-end has training logic.

### Swapping the trainer

Swapping the trainer means writing a new `impl Trainer` in core and changing
**one constructor line per front-end**: the line in `cli.rs` that constructs
`BurnTrainer` (`crates/loractl-cli/src/cli.rs`), and the single `BurnTrainer`
line in `loractl-api`'s `main.rs` (its `TrainerFactory` seam). If a new
trainer forces front-end changes beyond those constructors, the event
abstraction has leaked — fix the abstraction, not the front-end. The LoRA
math: freeze the base weights, train the low-rank factors, forward =
`base(x) + (alpha/rank) · B(A(x))`.

### Config layering

A run is fully described by a YAML `TrainConfig` (`config/examples/lora.yaml`).
Precedence, lowest to highest: **YAML file → `LORACTL_`-prefixed env vars (with
`__` for nested keys, e.g. `LORACTL_OPTIM__LR`) → CLI flags.** The env/file
layering is done by `figment` in `load_config`; **CLI flag overrides are
applied by mutating the struct *after* extraction** (`cli.rs`), not via
figment — this is deliberate, since flags are partial and must win last. Match
this pattern when adding new overridable flags.

## Conventions

- Edition 2024, `resolver = "3"`, MSRV pinned at `rust-version = "1.92"` in the
  workspace `Cargo.toml` (bumped from 1.85 to satisfy burn 0.21's MSRV). Shared
  deps go in `[workspace.dependencies]`.
- `Cargo.lock` **is committed** (this workspace produces a binary).
- Roadmap milestones are tracked as issues #1–#4 and linked from the README;
  keep the two in sync when a milestone lands.
