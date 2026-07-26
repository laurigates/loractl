#!/usr/bin/env bash
# Report `pub` items in loractl-core that nothing outside their own file names.
#
# Why this exists: rustc's `dead_code` lint never fires on a `pub` item in a
# `pub mod`, because from the compiler's point of view a downstream crate might
# use it. loractl-core has exactly one set of consumers — the two front-end
# crates plus this repo's `tests/` and `examples/` — so "nothing in the
# workspace names it" is a much stronger signal here than it would be for a
# published library, and nothing built into cargo reports it.
#
# This is a REPORTING tool, not a gate. Its output is a list of candidates to
# consider narrowing to `pub(crate)`, and it is deliberately not wired into
# `just lint` — see the accuracy note below for why it must not be.
#
#   ./scripts/pub-census.sh            # human-readable report
#   ./scripts/pub-census.sh --porcelain # file|symbol, one per line
#
# ---------------------------------------------------------------------------
# Accuracy, from the one time this was run in anger (11 of 49 candidates were
# genuinely internal):
#
#   * FALSE POSITIVES ARE THE NORM. The scan matches names textually, so it
#     cannot see an item reached through type inference or field access:
#     `model.transformer.h[0]` never writes `Transformer`, and
#     `let out = sample_adapter(..)` never writes `SampleOutput`. Both look
#     unused here and are not. Roughly two thirds of the first run's hits were
#     this.
#   * Public documentation counts as use. rustdoc resolves intra-doc links, so
#     an item a public doc comment links to is part of the public story even if
#     no code names it. Count with
#     `cargo doc -p loractl-core --no-deps 2>&1 | grep -c '^warning: public documentation for'`
#     and compare against the base branch — and note the message spells
#     associated items `Type::method`, so any filter you write must allow `:`.
#   * A TYPE ALIAS NAMED IN A PUBLIC SIGNATURE IS PUBLIC SURFACE THAT NOTHING
#     WILL FLAG. Aliases are transparent, so `pub fn f() -> Vec<Batch<B>>` with
#     a `pub(crate) type Batch` compiles silently — the effective signature is
#     the expansion, which is public, so `private_interfaces` cannot fire and
#     rustdoc renders a name callers cannot write. The compiler cannot
#     adjudicate this one; read the item's own doc comment instead, which in
#     this repo tends to say outright why something is public.
#
#   So the workflow is: demote a candidate, then let `cargo check --all-targets`
#   and `cargo doc` adjudicate, and keep whatever they reject as `pub`.
#   Expect the reverts to cascade — restoring a parent type re-exposes the
#   field types you just demoted.
#
# Scope limits, so "0 candidates" is not mistaken for "nothing left to narrow":
#   * Declarations matched are `pub fn|struct|enum|trait|const|type` only —
#     not `pub static`, `pub async fn`, `pub unsafe fn`, `pub mod`, `pub use`.
#   * An item used only by another file *inside* core is not reported, though
#     it is still a `pub(crate)` candidate. Under-reporting is the safe way to
#     be wrong here, but it is still wrong.
set -euo pipefail

cd "$(dirname "$0")/.."

porcelain=false
case "${1:-}" in
    --porcelain) porcelain=true ;;
    "") ;;
    *)
        echo "usage: $(basename "$0") [--porcelain]" >&2
        exit 2
        ;;
esac

$porcelain || echo "Scanning crates/loractl-core/src for pub items unused outside their own file..."

count=0
# `find`, not a `src/*.rs` glob: core is flat today, but a future
# `src/mmdit/attention.rs` would otherwise be skipped in silence.
while IFS= read -r f; do
    # `sort -u` because a name can be declared once but matched twice (e.g. an
    # inherent impl and a trait impl in the same file).
    #
    # `[[:space:]]` rather than `\s`: `\s` is a GNU extension to POSIX BRE, and
    # BSD grep (macOS — this project's Apple-Silicon GPU host) would match
    # nothing at all, making the script report a cheerful "0 candidates"
    # instead of failing. A reporting tool that silently reports nothing is
    # worse than one that crashes.
    while read -r sym; do
        [[ -z "$sym" ]] && continue
        # -w so PROMPT_PREFIX does not match PROMPT_PREFIX_LEN; -F so a symbol
        # is never reinterpreted as a pattern. Searching all of crates/
        # deliberately includes tests/ and examples/: those are separate
        # crates, so an item they name must stay `pub`.
        if [[ -z "$(grep -rwlF "$sym" crates --include='*.rs' | grep -v "^$f$")" ]]; then
            if $porcelain; then
                echo "$f|$sym"
            else
                printf '  %-40s %s\n' "${f#crates/loractl-core/src/}" "$sym"
            fi
            count=$((count + 1))
        fi
    done < <(grep -o '^[[:space:]]*pub \(fn\|struct\|enum\|trait\|const\|type\) [A-Za-z_][A-Za-z0-9_]*' "$f" |
        sed 's/.* //' | sort -u)
    # Both loops are fed by process substitution rather than a pipe, so their
    # bodies run in this shell and `count` survives.
done < <(find crates/loractl-core/src -name '*.rs' | sort)

$porcelain || echo "$count candidate(s). Verify each with the compiler before demoting — see this script's header."
