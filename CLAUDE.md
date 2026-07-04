# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`loractl` is a terminal-native LoRA trainer in Rust: a **CLI-first** tool where
a GUI, if ever built, is just another renderer over the same core (the name is
a deliberate `*ctl` reference, like `kubectl`). It is an early-stage learning
project — see the roadmap in `README.md` and the tracking issues (#1–#4).

**Current status:** milestone 1 (skeleton). There is **no ML yet** — the
default trainer is a dependency-free `MockTrainer` that drives the full
pipeline with synthetic loss. burn + a real LoRA module arrive in M2 (#1).

## Commands

Recipes live in the `justfile` (`just` to list). Cargo directly also works.

| Task | Command |
|---|---|
| Build the workspace | `just build` (`cargo build`) |
| Run the CLI | `just run <args>` (`cargo run -p loractl-cli -- <args>`) |
| Train from a config (mock trainer) | `just train [config]` — defaults to `config/examples/lora.yaml` |
| Generate shell completions | `just completions [shell]` (e.g. `just completions fish`) |
| Lint (warnings-as-errors) | `just lint` (`cargo clippy --all-targets --all-features -- -D warnings`) |
| Format / check format | `just fmt` / `just fmt-check` |
| Tests | `just test` (`cargo test`) — no suite exists yet; M2 adds the first tests |
| One test by name | `cargo test -p loractl-core <test_name>` |

Before committing, the meaningful gate is `just fmt-check && just lint` — CI
parity is intended (the `justfile` mirrors what CI should run). rustfmt is
default style; expect it to reflow multi-line signatures onto one line.

## Architecture — the one rule that matters

The workspace is two crates today, with a third planned:

| Crate | Role |
|---|---|
| `loractl-core` | The pipeline: `TrainConfig` schema, `TrainEvent` stream, `Trainer` trait, `MockTrainer`. |
| `loractl-cli` | The `loractl` binary — parses args, layers config, renders events. |
| `loractl-api` *(M5, #4)* | Future HTTP/bindings surface for a GUI. |

**Load-bearing invariant: `loractl-core` emits events; it never renders.**
Concretely, core must not import `clap` and must not `println!`/write to
stdout/stderr. A `Trainer` reports progress by calling a `&mut dyn
FnMut(TrainEvent)` sink; the *caller* decides how to surface it. The CLI
renders `TrainEvent`s as an `indicatif` progress bar (see the match arm in
`crates/loractl-cli/src/cli.rs`); a future API would serialize the same events
as JSON/SSE. **This is what makes "a GUI can be built separately" real — do
not break it** by having core print or by having the CLI reach into training
internals.

Dependency direction is strictly `cli → core` (and later `api → core`). Core
has no upward dependencies and no front-end has training logic.

### Swapping the trainer (M2 and beyond)

Adding the burn backend means writing a new `impl Trainer` in core and
changing the **one line** in `cli.rs` that constructs `MockTrainer`. If a new
trainer forces CLI changes beyond that constructor, the event abstraction has
leaked — fix the abstraction, not the CLI. The intended LoRA math for that
module: freeze the base weights, train the low-rank factors, forward =
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
