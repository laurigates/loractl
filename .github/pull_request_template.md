<!--
PR title must follow Conventional Commits — `type(scope): summary` — so
release-please can version correctly. Scopes: core, cli, api, config, trainer,
ci, docs.
-->

## What

Briefly, what this PR changes.

## Why

The motivation — the problem, the milestone, or the issue it addresses.

## Verification

- [ ] `just fmt-check && just lint && just test` passes
- [ ] Touched a feature-gated path? Ran the matching `just lint-mnist` /
      `lint-gpt2-real` / `lint-wgpu`
- [ ] Touched dependencies? Ran `just audit` / `just deny`
- [ ] New ML code is verified against a reference (PyTorch golden), not just
      asserted to run

## Linked issue

Closes #
