# Development Workflow — loractl

Project-specific workflow for the `loractl` Rust workspace. General TDD, commit,
and code-quality conventions come from the user-global rules; this file records
only what is specific to loractl.

## Test-Driven Development

Follow RED → GREEN → REFACTOR:

1. Write a failing test (`cargo test -p <crate> <name>`).
2. Implement the minimal code to pass.
3. Refactor while the suite stays green.

The first tests landed with the burn backend in M2 (#1): a deterministic
numerics proof against a PyTorch golden (`tests/lora_reference.rs`) and a
black-box synthetic convergence test (`tests/convergence.rs`). New ML code (the
LoRA `Module`, the training loop) is exactly where TDD earns its keep: verify
numerics against a reference before scaling up.

## The load-bearing invariant

`loractl-core` emits events; it never renders. Core must not import `clap` and
must not `println!` / write to stdout/stderr. A `Trainer` reports progress
through a `&mut dyn FnMut(TrainEvent)` sink; the caller decides how to surface
it. Adding a trainer (e.g. the burn backend) means a new `impl Trainer` in core
plus **one constructor line per front-end**: the line in
`crates/loractl-cli/src/cli.rs` that constructs `BurnTrainer`, and the single
`BurnTrainer` line in `loractl-api`'s `main.rs` (its `TrainerFactory` seam). If
a new trainer forces front-end changes beyond those constructors, the event
abstraction has leaked; fix the abstraction, not the front-end.

Dependency direction is strictly `cli → core` and `api → core`. Core has
no upward dependencies.

## Config layering

Precedence, lowest to highest: YAML file → `LORACTL_`-prefixed env vars (`__`
for nested keys) → CLI flags. `figment` does the file/env layering in
`load_config`; CLI flag overrides are applied by mutating the struct *after*
extraction (flags are partial and must win last). Match this pattern when adding
overridable flags.

## Commit conventions

Conventional commits: `type(scope): summary`. Scopes track the crates and
subsystems: `core`, `cli`, `api`, `config`, `trainer`, `ci`, `docs`.
`Cargo.lock` is committed (this workspace builds a binary). Roadmap milestones
are issues #1–#4 and #17–#25 — keep the README roadmap and the issues in sync
when a milestone lands.

## Pre-commit gate

The meaningful local gate mirrors CI:

```
just fmt-check && just lint && just audit
```

`just lint` is `cargo clippy --all-targets -- -D warnings` (default/offline
features; warnings are errors). The opt-in `mnist` feature pulls a networked
dataset downloader, so its path is linted separately via `just lint-mnist` to
keep the default gate offline and fast. `just audit` is the RustSec advisory
scan (CI's `security-audit` workflow); its sibling `just deny` is the
supply-chain gate (licenses/bans/sources, CI's `deny` job). rustfmt uses
default style and will reflow multi-line signatures onto one line — expect
that.

## Post-release Cargo.lock sync

`release-please` bumps the version in `Cargo.toml` **only** — it does not touch
`Cargo.lock`. After merging a release PR the committed lockfile is a version
behind the manifest, so the next `cargo` invocation rewrites it. Run
`cargo update --workspace` and commit the result as `chore: sync Cargo.lock`
before starting other work, so the drift doesn't ride into an unrelated PR.
