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

## The docs drift gate (`just surf`)

`hubs/*.md` anchor prose claims to code symbols.
[Surface](https://github.com/Connorrmcd6/surface) fingerprints each anchored
symbol's *logic* — ignoring comments, formatting, and consistent renames — and
CI (`.github/workflows/docs-drift.yml`) fails when one moves.

It is optional locally: install `surf` and run `just surf` to get the same
verdict CI will. Without it installed, nothing breaks; the Action is the gate.

**When it fires, do not rubber-stamp it.** The gate is a prompt to re-read, not
a chore:

1. Read the claim it printed. Decide whether the prose is *still true* — Surface
   only knows the code moved, never whether the sentence is now wrong.
2. If the prose is stale, fix it first.
3. Only then `surf verify` to re-seal the hash, and commit prose + code together.

`surf check --format json` emits a machine-readable verdict if you want a
reviewer to judge the "is it still true?" half.

**What it does not cover.** Anchors guard the spans they point at, so a change
*elsewhere* can falsify a claim while this gate stays green. Empirical claims —
VRAM numbers, throughput, upstream bug status — are not anchorable at all; those
live in `docs/roadmap.md` and the ADRs and are maintained by hand.

**`surf lint`'s warnings are advisory, and five are load-bearing-ly ignored.**
Lint flags every unanchored public callable in a file some hub already anchors
into, so `diffusion_trainer.rs` is in scope purely because `memory-route.md`
anchors one (private) function there. Its five `pub fn`s — `denoiser_filename`,
`encoder_fingerprint`, `load_fp8_encoder`, `load_fp8_module`,
`load_quant_module` — are **deliberately unanchored** (decided in #166): the
file runs ~26 commits/90d, far past the 6-8-doc / low-churn bar the anchored
symbols met, and each is already pinned by a literal-asserting integration test
in `crates/loractl-core/tests/`. Surface 0.8.0 has no way to *record* an
acceptance — `surf.toml` takes only `hubs` and `bundles`, and there is no
ignore subcommand — so the warnings reappear on every run and this paragraph is
the only place the decision lives. `surf lint` exits 0; they block nothing.
Any new `pub fn` in that file joins the list, and that is a judgment call each
time — never a reason to anchor mechanically.

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
