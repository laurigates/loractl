---
summary: The core↔front-end contract — one routing seam, one event stream, and the rule that core never renders.
anchors:
  - claim: >
      `select_trainer` is the single seam that maps `model.base` to a concrete trainer,
      and both front-ends (the CLI's `train()` and loractl-api's TrainerFactory) call it
      rather than choosing a trainer themselves. "synthetic" and "mnist" are the built-in
      demo bases routing to BurnTrainer; anything else is treated as a Krea-2-Raw-layout
      checkpoint directory and routes to DiffusionTrainer.
    at:
      - crates/loractl-core/src/train.rs > select_trainer
      - crates/loractl-core/src/train.rs > is_builtin_demo_base
    hash: 2:4c4e33dfdc85
    id: c_18c63b4af50a8b600006
    verified_at: 2026-07-27T19:11:35Z
    verified_commit: 069d768374f09c4a8bfe9a8bcbb75ee863132a88
  - claim: >
      `TrainEvent` is the whole progress vocabulary and is the serialized wire contract:
      it is an internally-tagged serde enum (`tag = "type"`, snake_case variant names), so
      adding, renaming, or re-shaping a variant changes the HTTP/SSE payload that
      docs/api/events.md pins. Its payload types are part of that contract too:
      `PhaseName` is a closed enum serialized as snake_case tokens a consumer may key on,
      and `PhaseCounters` is `#[serde(flatten)]`ed onto `Phase`, so its counters ride the
      wire as sibling fields rather than a nested object — the bundling exists to make a
      `total` without a `done` unrepresentable in Rust, not to change the JSON. Trainers
      report by calling a sink with these values and must not render or write to
      stdout/stderr themselves.
    at:
      - crates/loractl-core/src/event.rs > TrainEvent
      - crates/loractl-core/src/train.rs > Trainer
    hash: 2:28ca2950a289
    id: c_18c63b4af5b5f2480007
    verified_at: 2026-07-27T20:26:18Z
    verified_commit: 86711000dea743caf8ccb60bf669fc129ea4530a
refs: []
---

# The trainer contract

`loractl-core` **emits events; it never renders.** A `Trainer` reports progress by
calling a `&mut dyn FnMut(TrainEvent)` sink and the *caller* decides how to surface it —
the CLI as an `indicatif` progress bar, `loractl-api` as JSON/SSE. This is what makes
"a GUI can be built separately" real.

Adding a trainer means a new `impl Trainer` in core plus an arm in `select_trainer` —
nothing in the CLI or the API changes. If a new trainer forces front-end edits beyond
that factory, the event abstraction has leaked: fix the abstraction, not the front-end.

Routing is independently pinned by `tests/trainer_routing.rs`, which discriminates the
arms by observable behavior rather than by re-spelling the match.

**Boundary.** This hub covers *which* trainer runs and *what vocabulary* it reports in.
It says nothing about what any trainer computes, and it cannot tell you whether
`docs/api/events.md` is still accurate — only that the enum it describes moved.
