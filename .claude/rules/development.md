# Development Workflow — loractl

Project-specific workflow for the `loractl` Rust workspace. General TDD, commit,
and code-quality conventions come from the user-global rules; this file records
only what is specific to loractl.

## Test-Driven Development

Follow RED → GREEN → REFACTOR:

1. Write a failing test (`cargo test -p <crate> <name>`).
2. Implement the minimal code to pass.
3. Refactor while the suite stays green.

No test suite exists yet — the first tests land with the burn backend in M2
(#1). New ML code (the LoRA `Module`, the training loop) is exactly where TDD
earns its keep: verify numerics against a reference before scaling up.

## The load-bearing invariant

`loractl-core` emits events; it never renders. Core must not import `clap` and
must not `println!` / write to stdout/stderr. A `Trainer` reports progress
through a `&mut dyn FnMut(TrainEvent)` sink; the caller decides how to surface
it. Adding a trainer (e.g. the burn backend) means a new `impl Trainer` in core
plus the **one** constructor line in `crates/loractl-cli/src/cli.rs` — if a new
trainer forces CLI changes beyond that line, the event abstraction has leaked;
fix the abstraction, not the CLI.

Dependency direction is strictly `cli → core` (and later `api → core`). Core has
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
are issues #1–#4 — keep the README roadmap and the issues in sync when a
milestone lands.

## Pre-commit gate

The meaningful local gate mirrors CI:

```
just fmt-check && just lint
```

`just lint` is `cargo clippy --all-targets --all-features -- -D warnings`
(warnings are errors). rustfmt uses default style and will reflow multi-line
signatures onto one line — expect that.
