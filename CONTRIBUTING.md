# Contributing to loractl

Thanks for your interest in `loractl` — a terminal-native LoRA trainer in Rust.
This is a short pointer file; the detail lives in the artifacts that already
carry it, so it can't drift.

## Prerequisites

- **Rust** — edition 2024, MSRV `1.92` (pinned in the workspace `Cargo.toml`).
- **[`just`](https://github.com/casey/just)** — the task runner; `just` lists
  every recipe. `Cargo.lock` is committed, so a plain `cargo build` also works.
- Optional, only for regenerating goldens or the opt-in feature paths:
  [`uv`](https://docs.astral.sh/uv/) (for the PyTorch reference scripts),
  `cargo-audit`, and `cargo-deny`.

## The gate before you commit

Run the same checks CI runs:

```
just fmt-check && just lint && just test
```

- `just lint` is `cargo clippy --all-targets -- -D warnings` (warnings are
  errors) over the default, offline feature set.
- `just test` runs the offline suite — numerics vs. the PyTorch golden plus
  synthetic convergence; no network, no GPU.
- Supply-chain gates: `just audit` (RustSec advisories) and `just deny`
  (licenses/bans/sources). CI additionally runs `feature-lints` over the
  opt-in `mnist` / `gpt2-real` / `wgpu` paths — mirror those locally with
  `just lint-mnist` / `lint-gpt2-real` / `lint-wgpu` when you touch a
  feature-gated path.

Features are **offline by default**: `mnist`, `gpt2-real`, and `wgpu` are
opt-in and never part of the default build or `just test`.

## Testing conventions

New ML code lands with tests, and ML correctness is verified against a
reference, not merely asserted to run. The numerics proofs assert against
checked-in **PyTorch goldens** (regenerate with `just reference` /
`just flow-reference` / `just gpt2-tiny-reference`, which need `torch` via
`uv`). Follow RED → GREEN → REFACTOR — see
[`.claude/rules/development.md`](.claude/rules/development.md) and
[`.claude/rules/testing.md`](.claude/rules/testing.md).

## Commits & PRs

- **Conventional commits**: `type(scope): summary`. Scopes track the crates
  and subsystems: `core`, `cli`, `api`, `config`, `trainer`, `ci`, `docs`.
  release-please drives versioning off these, so the format matters.
- Keep the README roadmap and the tracking issues (#1–#4, #17–#25) in sync
  when a milestone lands.

## Where the detail lives

- [`CLAUDE.md`](CLAUDE.md) — architecture, the load-bearing event/render
  invariant, config layering, and the full command table.
- [`justfile`](justfile) — the source of truth for every recipe.
- [`docs/adrs/`](docs/adrs/) — the design record (why GPT-2 first, the adapter
  format, the HTTP API, the Krea 2 direction).
- [`docs/api/events.md`](docs/api/events.md) — the HTTP/SSE wire contract.
