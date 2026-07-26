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
# Accuracy, from the one time this was run in anger (14 of 49 candidates were
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
#     no code names it.
#
#   So the workflow is: demote a candidate, then let `cargo check --all-targets`
#   and `cargo doc` adjudicate, and keep whatever they reject as `pub`.
#   Expect the reverts to cascade — restoring a parent type re-exposes the
#   field types you just demoted.
set -euo pipefail

cd "$(dirname "$0")/.."

porcelain=false
[[ "${1:-}" == "--porcelain" ]] && porcelain=true

$porcelain || echo "Scanning crates/loractl-core/src for pub items unused outside their own file..."

count=0
for f in crates/loractl-core/src/*.rs; do
    # `sort -u` because a name can be declared once but matched twice (e.g. an
    # inherent impl and a trait impl in the same file).
    while read -r sym; do
        [[ -z "$sym" ]] && continue
        # -w so PROMPT_PREFIX does not match PROMPT_PREFIX_LEN. Searching all
        # of crates/ deliberately includes tests/ and examples/: those are
        # separate crates, so an item they name must stay `pub`.
        if [[ -z "$(grep -rwl "$sym" crates --include='*.rs' | grep -v "^$f$")" ]]; then
            if $porcelain; then
                echo "$f|$sym"
            else
                printf '  %-40s %s\n' "${f#crates/loractl-core/src/}" "$sym"
            fi
            count=$((count + 1))
        fi
    done < <(grep -o '^\s*pub \(fn\|struct\|enum\|trait\|const\|type\) [A-Za-z_][A-Za-z0-9_]*' "$f" |
        sed 's/.* //' | sort -u)
done

$porcelain || echo "$count candidate(s). Verify each with the compiler before demoting — see this script's header."
