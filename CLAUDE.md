# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`loractl` is a terminal-native LoRA trainer in Rust: a **CLI-first** tool where
a GUI, if ever built, is just another renderer over the same core (the name is a
deliberate `*ctl` reference, like `kubectl`).

Milestones M1–M15 have landed. **[`docs/roadmap.md`](docs/roadmap.md) is the
canonical record** of what each one delivered, where the project currently
stands, and the measured memory/throughput findings.

**Do not restate roadmap or ADR content here.** Status, VRAM numbers, benchmark
results, and upstream bug status all drift; this file is loaded on every turn,
so a stale copy here is worse than a lookup. Read the roadmap when you need the
current state.

Strategy and the open questions live in the ADRs — most load-bearing:
[ADR-0004](docs/adrs/0004-krea2-image-diffusion-target.md) (the Krea 2 target),
[ADR-0005](docs/adrs/0005-int4-training-vram-bound.md) (why training is
VRAM-bound and which levers are measured dead),
[ADR-0008](docs/adrs/0008-host-offload-mechanism-and-scope.md) (host offload).

## Commands

Recipes live in the `justfile` — run `just` to list them, or `just --list` for
the full set with descriptions. Cargo directly also works.

**The gate before committing:**

```
just fmt-check && just lint
```

CI additionally runs `feature-lints` (clippy over the opt-in
mnist/gpt2-real/qwen-vae-real/qwen3vl-real/mmdit-real/wgpu paths) and `deny`
(`cargo deny check`). Run the matching `just lint-<feature>` / `just deny`
locally when a change touches a feature-gated path or the dependency graph.

**The few commands with judgment attached** — the rest are self-explanatory
from `just --list`:

| Command | What to know |
|---|---|
| `just step-probe` | The VRAM answer. The gate is a **zero-panic** run — a run that survived an OOM storm silently corrupts the forward (a negative MSE was observed once). |
| `just bench` | The throughput answer (#110). Never quote `tok_s=`/`tflops=` without the `MODEL` line they are a quotient of, or any timing from a run reading `sanity=SUSPECT` / `plausible=false`. |
| `just bench-offline` | Keeps the harness runnable without a GPU. Its *numbers are meaningless* (a 2-block toy at 32px) — smoke only. |
| `just test-cuda` / `just test-wgpu` | Real-GPU smokes, opt-in. Hosted CI is GPU-free (ndarray default). |
| `just quant-probe` | On-box int8/int4 VRAM + dequant-error proof. |
| `just surf` | The docs↔code drift gate over `hubs/*.md` (CI parity: `docs-drift.yml`). A hit means the anchored *code* moved — never that the prose is wrong. Re-read the claim, fix `hubs/` if it is now stale, and only then `surf verify` to re-seal. Needs `surf` on PATH; CI installs its own, so the gate holds either way. |

Real GPU proofs run on the self-hosted RTX 4090 via a **dispatchable**
workflow: `gh workflow run gpu.yml` (`-f suite=all` adds the wgpu smokes,
`-f int8_probe=true` adds the quant probe). See the `gpu.yml` header for the
runner-registration prerequisite.

rustfmt is default style; expect it to reflow multi-line signatures onto one
line.

## Architecture — the one rule that matters

The workspace is four crates:

| Crate | Role |
|---|---|
| `loractl-core` | The pipeline: `TrainConfig` schema, `TrainEvent` stream, `Trainer` trait, `MockTrainer`, the LoRA/GPT-2 modules and `BurnTrainer`. |
| `loractl-cli` | The `loractl` binary — parses args, layers config, renders events. |
| `loractl-api` | The `loractl-api` binary — serializes the same `TrainEvent`s over HTTP/SSE for a GUI; renders nothing itself. Wire contract: `docs/api/events.md`. |
| `loractl-bench` | Dependency-free measurement primitives ported from CAEF (#110): the `RESULT`/`SANITY`/`MODEL` line schema, the wall-sync timer, the dead-graph guards. Backend-agnostic by construction — the burn-side adapter that drives it lives in `loractl-core::bench`. |

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

Swapping the trainer means writing a new `impl Trainer` in core and adding
an arm to **core's `select_trainer`** (`src/train.rs`) — the single factory
that maps `model.base` to a concrete trainer ("synthetic"/"mnist" →
`BurnTrainer`, anything else → `DiffusionTrainer`; pinned by
`tests/trainer_routing.rs`). Both front-ends call it at their one
construction site each: `cli.rs`'s `train()` and the `TrainerFactory`
closure in `loractl-api`'s `main.rs`. If a new trainer forces front-end
changes beyond that factory, the event abstraction has leaked — fix the
abstraction, not the front-end. The LoRA math: freeze the base weights,
train the low-rank factors, forward = `base(x) + (alpha/rank) · B(A(x))`.

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
- Roadmap milestones are tracked as issues #1–#4, #17–#25 and #82 — keep
  [`docs/roadmap.md`](docs/roadmap.md), the README checklist, and the issues in
  sync when a milestone lands.

## Where the detail lives

| You need | Read |
|---|---|
| Current status, milestone history, measured findings | [`docs/roadmap.md`](docs/roadmap.md) |
| Why a design decision was made | [`docs/adrs/`](docs/adrs/) |
| The HTTP/SSE wire contract | [`docs/api/events.md`](docs/api/events.md) |
| How to run the gate, test conventions, PR flow | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| burn/cubecl-specific traps (loaded when editing Rust) | [`.claude/rules/`](.claude/rules/) |
