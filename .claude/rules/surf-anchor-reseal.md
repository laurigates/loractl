# Re-sealing a Surface Anchor — Bare `surf verify` Re-hashes Every Claim

`just surf` fails when an anchored span's hash diverges from `hubs/*.md`.
CLAUDE.md already states the reading: **a hit means the anchored *code* moved —
never that the prose is wrong.** This file covers what to do next, because the
obvious command is wider than the problem.

## The trap

`surf verify` with **no TARGET** stamps *every* anchor in the workspace, not
just the diverged one. Any unrelated claim that had also drifted is silently
re-sealed in the same breath — and `surf stats` reports exactly this as a
**rubber-stamp rate**, so the tool is already counting it against you.

```sh
# Wrong — re-hashes all claims in every hub
surf verify

# Right — the one anchor, by its exact `at:` value
surf verify "crates/loractl-core/src/export.rs > export_adapters"
```

The target string is the `at:` line in the hub, verbatim. A claim with several
`at:` entries is re-sealed as a unit by naming any one of them.

## The sequence

1. **Read the claim, then the diff.** Ask only: does the change touch what the
   claim *asserts*? Not "did the file change" — the hash already told you that.
2. **Prose stale → fix `hubs/` first.** Re-sealing is what you do *after* the
   prose is true again, never instead.
3. **Prose holds → targeted `surf verify`.** The diff should be one `hash:` and
   its two timestamps. Anything wider means the target was too broad.
4. **Put the reasoning in the commit message.** `surf verify`'s own help says to
   re-hash *"after a human confirms the prose still holds"* — that confirmation
   is a judgment, so it has to be reviewable rather than implied by the stamp.

## Observed

2026-08-27 (#229). The `chunks_exact` → `as_chunks` migration edited
`import_adapters`, which sits inside `hubs/lora-interop.md`'s export-layout
anchor, so `surf check` reported `2:40642bc99015 → 2:7569189f1def`. The claim
asserts burn's `A`/`B` shapes, the transposes to kohya form, and the `.alpha`
scalar reconstructed as `scaling * rank`; the change swapped a byte-decode
iterator in the f32 reader and touched none of it. Prose held, seal was stale,
targeted verify produced a two-line diff.

## The boundary this gate does not cover

From surf's own help: it checks that the code a claim points at is **unchanged
since last verified**. It does *not* check that the documented invariant still
holds across the system — a change **elsewhere** can falsify a claim while its
anchored span, and this gate, stay green. Green `surf check` is evidence about
one span, not about the claim's truth.

`surf lint` warnings (public symbols with no claim in any hub) are advisory and
do not fail the gate; `surf check` divergences do.

## Related

- [`document-management.md`](document-management.md) — where docs live
- `~/.claude/rules/git-hazards.md` — the parent law: a green command is a claim
  about mechanics, not content
