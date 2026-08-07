#!/usr/bin/env bash
# The on-box before/after matrix for the two acceptance criteria PR #197 could
# not claim offline: #175's peak-VRAM scaling and #178's cold-encode speedup.
#
# Both are comparisons against a PREVIOUS REVISION, so this drives two git
# worktrees rather than the working tree, and both need a real 24 GB card with
# the multi-GB Krea 2 checkpoints. It is written to be scheduled and left
# alone: it refuses loudly rather than produce a poisoned number, and every
# arm's raw stdout is kept next to the summary.
#
# WHAT IT MEASURES
#
#   #175  "Peak VRAM no longer scales with example count"
#         A scaling claim, so it needs two dataset sizes per revision, not one
#         before/after pair. Pre-#175 code holds every example's conditioning
#         device-resident at [1, 512, 12, 2560] f32 = 60 MiB, so the `before`
#         arms should show peak(large) - peak(small) ~= (large-small) * 60 MiB
#         and the `after` arms ~= 0.
#
#   #175  "`just bench` before/after, so the per-step read cost is priced"
#         The lazy read trades disk + H2D per step for the residency win. The
#         bench arm is what stops that trade being assumed rather than paid.
#
#   #178  "Cold-cache encode-phase wall time before/after (>=40 images)"
#         Bracketed by the `encode` phase events the trainer already emits --
#         stdout is line-timestamped here, so no Rust change is needed to time
#         it. Note the encode report is emitted BEFORE each entry is processed
#         (dataset.rs::DatasetProgress), so the phase ENDS at the first
#         non-encode line, not at the last encode line.
#
# WHAT IT DOES NOT MEASURE
#
#   #147 (no_upscale) and #148 (grid bucketing). The pinned dataset is ~90%
#   square and mostly well above `resolution`, so bucket assignment runs but is
#   never stressed, and the template deliberately sets neither knob (they do
#   not exist on the `before` revision). Those stay covered by the offline
#   tests; do not read this matrix as evidence about them.
#
# THE GPU MUST BE IDLE. The step probe reads whole-GPU `nvidia-smi`, so any
# other process holding VRAM is added to every peak this reports -- silently,
# and in the direction that makes a regression look like a pass. This is not
# hypothetical: .claude/rules/gpu-runner-failure-signatures.md records an idle
# ComfyUI holding 17.6 GiB of 24.5 GiB against a config needing 19.4 GiB, which
# surfaced as a NaN loss blaming f16 precision on an f32 run. The preflight
# refuses when the card is busy.
#
# Two ways to free it, least disruptive first:
#
#   --free-endpoint URL   POST ComfyUI's /free to drop IDLE CACHED WEIGHTS
#                         without stopping the service. It reloads them on the
#                         next generation, so this costs a warm cache, not a
#                         restart. Only safe with an empty queue -- checked.
#   --stop-service UNIT   Stop the systemd unit outright, restarting it on exit
#                         (including on failure). The bigger hammer; use when
#                         the holder is not ComfyUI or /free is not enough.
#
# Usage:
#   scripts/residency-matrix.sh --models-root /path/to/ComfyUI/models
#   scripts/residency-matrix.sh --models-root DIR --free-endpoint http://127.0.0.1:8188
#   scripts/residency-matrix.sh --models-root DIR --stop-service comfyui.service
#   scripts/residency-matrix.sh --models-root DIR --dry-run   # preflight only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BEFORE_REV="origin/main"
AFTER_REV="origin/claude/open-issues-ultracode-m2eu5a"
DATASET_KEY="tuxemon"
N_SMALL=8
N_LARGE=56
MODELS_ROOT="${LORACTL_MODELS_ROOT:-}"
OUT=""
STOP_SERVICE=""
FREE_ENDPOINT=""
ALLOW_BUSY_GPU=0
SKIP_BENCH=0
DRY_RUN=0
BENCH_STEPS=8
# Free-VRAM floor for a trustworthy baseline. The probe's own baseline read is
# taken before the run, so a few hundred MiB of desktop compositor is tolerable;
# a multi-GB inference server is not.
IDLE_MIB=1024
# Conditioning caches to disk as f32 at 60 MiB/example, so the large arm's
# cache alone is ~3.4 GB per revision. Refuse rather than die mid-matrix.
MIN_FREE_GIB=25

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//;$d'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --before) BEFORE_REV="$2"; shift 2 ;;
        --after) AFTER_REV="$2"; shift 2 ;;
        --dataset) DATASET_KEY="$2"; shift 2 ;;
        --small) N_SMALL="$2"; shift 2 ;;
        --large) N_LARGE="$2"; shift 2 ;;
        --models-root) MODELS_ROOT="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --stop-service) STOP_SERVICE="$2"; shift 2 ;;
        --free-endpoint) FREE_ENDPOINT="$2"; shift 2 ;;
        --bench-steps) BENCH_STEPS="$2"; shift 2 ;;
        --allow-busy-gpu) ALLOW_BUSY_GPU=1; shift ;;
        --skip-bench) SKIP_BENCH=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

die() { echo "residency-matrix: $*" >&2; exit 1; }
log() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------- preflight --

[ -n "$MODELS_ROOT" ] || die "--models-root (or LORACTL_MODELS_ROOT) is required"
[ -d "$MODELS_ROOT" ] || die "models root does not exist: $MODELS_ROOT"

TEMPLATE="config/probes/residency-matrix.yaml.in"
[ -f "$TEMPLATE" ] || die "missing $TEMPLATE"

# The three checkpoints the template names. Checked here so a two-hour matrix
# does not die on a missing file at the first model load.
missing=0
while read -r rel; do
    [ -n "$rel" ] || continue
    if [ ! -f "$MODELS_ROOT/$rel" ]; then
        echo "missing checkpoint: $MODELS_ROOT/$rel" >&2
        missing=1
    fi
done <<EOF
$(grep -oE '(diffusion_models|text_encoders|vae)/[^ ]+\.safetensors' "$TEMPLATE" | sort -u)
EOF
[ "$missing" -eq 0 ] || die "one or more checkpoints are missing under $MODELS_ROOT"

command -v cargo >/dev/null || die "cargo not on PATH"
command -v nvidia-smi >/dev/null || die "nvidia-smi not on PATH -- this matrix needs the GPU host"
command -v uv >/dev/null || die "uv not on PATH (scripts/fetch_dataset.py runs via uv)"

git rev-parse --verify --quiet "$BEFORE_REV^{commit}" >/dev/null || die "unknown revision: $BEFORE_REV"
git rev-parse --verify --quiet "$AFTER_REV^{commit}" >/dev/null || die "unknown revision: $AFTER_REV"
BEFORE_SHA="$(git rev-parse --short "$BEFORE_REV")"
AFTER_SHA="$(git rev-parse --short "$AFTER_REV")"
[ "$BEFORE_SHA" != "$AFTER_SHA" ] || die "before and after resolve to the same commit ($BEFORE_SHA)"

free_gib="$(df -BG --output=avail . | tail -1 | tr -dc '0-9')"
[ "$free_gib" -ge "$MIN_FREE_GIB" ] || die "only ${free_gib} GiB free; the encode caches need >= ${MIN_FREE_GIB} GiB"

# Ask ComfyUI to drop its idle weight cache. Refuses on a non-empty queue:
# `/free` mid-generation would evict models a running prompt is using, and the
# whole point of this path is that it is the NON-disruptive option.
if [ -n "$FREE_ENDPOINT" ]; then
    command -v curl >/dev/null || die "--free-endpoint needs curl"
    queue="$(curl -sf --max-time 10 "$FREE_ENDPOINT/queue")" \
        || die "could not read $FREE_ENDPOINT/queue -- is ComfyUI up?"
    # Here-string, not a pipe: under `set -o pipefail` a `grep -q` that closes
    # stdin early SIGPIPEs the writer and the pipeline reports 141 on a match.
    if grep -qE '"queue_(running|pending)": \[[^]]' <<<"$queue"; then
        die "$FREE_ENDPOINT has a non-empty queue; refusing to /free mid-generation"
    fi
    log "freeing ComfyUI's idle weight cache via $FREE_ENDPOINT/free"
    curl -sf --max-time 30 -X POST "$FREE_ENDPOINT/free" \
        -H 'Content-Type: application/json' \
        -d '{"unload_models":true,"free_memory":true}' >/dev/null \
        || die "POST $FREE_ENDPOINT/free failed"
    for _ in $(seq 1 15); do
        used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
        [ "$used" -gt "$IDLE_MIB" ] || break
        sleep 2
    done
fi

# Stop the named service BEFORE the idle check, and restore it on any exit
# path -- a matrix that leaves the box's service down is worse than one that
# did not run.
if [ -n "$STOP_SERVICE" ]; then
    if systemctl is-active --quiet "$STOP_SERVICE"; then
        log "stopping $STOP_SERVICE for the duration (restarted on exit)"
        sudo -n systemctl stop "$STOP_SERVICE" || die "could not stop $STOP_SERVICE"
        # shellcheck disable=SC2064  # expand $STOP_SERVICE now, not at trap time
        trap "echo 'restarting $STOP_SERVICE'; sudo -n systemctl start '$STOP_SERVICE' || true" EXIT
        # The VM teardown is not instantaneous; give VRAM time to actually free.
        for _ in $(seq 1 30); do
            used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
            [ "$used" -gt "$IDLE_MIB" ] || break
            sleep 2
        done
    else
        echo "note: $STOP_SERVICE is not active; nothing to stop"
    fi
fi

gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
gpu_total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
compute_apps="$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader || true)"
if [ "$gpu_used" -gt "$IDLE_MIB" ] && [ "$ALLOW_BUSY_GPU" -eq 0 ]; then
    echo "GPU is not idle: ${gpu_used} MiB of ${gpu_total} MiB already in use." >&2
    [ -n "$compute_apps" ] && echo "holding processes:" >&2 && echo "$compute_apps" >&2
    die "every peak would include this. Free the card, or pass --stop-service <unit>. \
--allow-busy-gpu overrides, and makes the peak numbers uncomparable."
fi

echo "preflight OK"
echo "  before:  $BEFORE_REV ($BEFORE_SHA)"
echo "  after:   $AFTER_REV ($AFTER_SHA)"
echo "  dataset: $DATASET_KEY, arms of $N_SMALL and $N_LARGE examples"
echo "  gpu:     ${gpu_used} MiB used of ${gpu_total} MiB"
echo "  disk:    ${free_gib} GiB free"

[ "$DRY_RUN" -eq 1 ] && { echo "STATUS=DRY_RUN_OK"; exit 0; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-tmp/residency-matrix/$STAMP}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"

# Shared target dir so the two worktrees do not each build burn/cubecl from
# scratch (~30 GB and most of the wall time). The two revisions differ by one
# crate, so the cost of sharing is rebuilding loractl-core when the revision
# flips -- far cheaper than duplicating the dependency graph.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"

# ------------------------------------------------------------------ helpers --

# Prefix each stdout line with a monotonic-enough wall timestamp. This is what
# makes the encode phase timeable without touching the trainer.
stamp() { while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done; }

# Wall seconds spanned by the `encode` phase in a stamped log: from the first
# encode report to the first line after the phase. Empty when the phase never
# appeared (a fully warm cache reports no encode work).
encode_wall() {
    awk '
        !seen && $0 ~ / encode: / { seen = 1; start = $1; next }
        seen && $0 !~ / encode: / && !done { printf "%.1f", $1 - start; done = 1; exit }
    ' "$1"
}

# `peak_mib=` from the greppable STEP_PROBE_SUMMARY line.
probe_peak() { sed -n 's/.*peak_mib=\([0-9]*\).*/\1/p' "$1" | tail -1; }

# The last `vram peak so far:` ratchet. Recorded for EVERY arm, because on a
# genuine OOM the process aborts before the summary prints and this is then the
# only measurement that exists (step_probe's header says so). Kept as its own
# key rather than folded into PEAK_MIB: a ratchet is a lower bound sampled at
# ~200 ms, not the same quantity as a completed run's peak, and conflating them
# would let an OOM arm read as a clean measurement.
probe_ratchet() { sed -n 's/.*vram peak so far: \([0-9]*\) MiB.*/\1/p' "$1" | tail -1; }

kv() { printf '%s=%s\n' "$1" "$2" | tee -a "$SUMMARY"; }

# Render the template for one arm.
render_config() {
    local dataset="$1" outdir="$2" dest="$3"
    sed -e "s|@MODELS_ROOT@|$MODELS_ROOT|g" \
        -e "s|@DATASET@|$dataset|g" \
        -e "s|@OUT@|$outdir|g" \
        "$TEMPLATE" > "$dest"
}

# Run one arm. Never aborts the matrix: a pre-#175 large arm is EXPECTED to be
# able to OOM, and that failure is the finding, not an error. The probe's own
# ratchet lines survive in the log when the summary line does not.
run_arm() {
    local worktree="$1" example="$2" label="$3" config="$4"
    shift 4
    local logfile="$OUT/$label.log"
    echo "-- arm $label"
    set +e
    ( cd "$worktree" && cargo run --release -p loractl-core --features cuda \
        --example "$example" -- "$config" "$@" 2>&1 ) | stamp > "$logfile"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "   exit=$rc log=$logfile"
    return 0
}

# ------------------------------------------------------------------ datasets --

log "materializing datasets"
DS_SMALL="$REPO_ROOT/tmp/datasets/${DATASET_KEY}-${N_SMALL}"
DS_LARGE="$REPO_ROOT/tmp/datasets/${DATASET_KEY}-${N_LARGE}"
./scripts/fetch_dataset.py --dataset "$DATASET_KEY" --out "$DS_SMALL" --limit "$N_SMALL"
./scripts/fetch_dataset.py --dataset "$DATASET_KEY" --out "$DS_LARGE" --limit "$N_LARGE"

: > "$SUMMARY"
{
    echo "# residency matrix $STAMP"
    echo "# before=$BEFORE_REV ($BEFORE_SHA)  after=$AFTER_REV ($AFTER_SHA)"
    echo "# dataset=$DATASET_KEY small=$N_SMALL large=$N_LARGE"
    echo "# host_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
} >> "$SUMMARY"

# ---------------------------------------------------------------------- arms --

for side in before after; do
    if [ "$side" = before ]; then rev="$BEFORE_REV"; sha="$BEFORE_SHA"; else rev="$AFTER_REV"; sha="$AFTER_SHA"; fi
    worktree="$REPO_ROOT/tmp/residency-matrix/worktrees/$side"

    log "$side: worktree at $sha"
    git worktree remove --force "$worktree" 2>/dev/null || true
    git worktree add --detach --force "$worktree" "$rev" >/dev/null
    ( cd "$worktree" && cargo build --release -p loractl-core --features cuda \
        --example step_probe --example bench_step ) || die "$side: build failed at $sha"

    for size in small large; do
        if [ "$size" = small ]; then n="$N_SMALL"; src="$DS_SMALL"; else n="$N_LARGE"; src="$DS_LARGE"; fi

        # A fresh copy is what makes the cold arm cold: the encode cache is
        # written INTO the dataset directory, so reusing one across revisions
        # would silently serve the other revision's latents.
        data="$OUT/data/$side/$size"
        rm -rf "$data"; mkdir -p "$data"
        cp "$src"/*.jpg "$src"/*.txt "$data"/

        cfg="$OUT/config-$side-$size.yaml"
        render_config "$data" "$OUT/run/$side-$size" "$cfg"

        run_arm "$worktree" step_probe "$side-$size-cold" "$cfg"
        run_arm "$worktree" step_probe "$side-$size-warm" "$cfg"

        kv "ENCODE_WALL_S_${side}_${size}" "$(encode_wall "$OUT/$side-$size-cold.log")"
        kv "PEAK_MIB_${side}_${size}" "$(probe_peak "$OUT/$side-$size-warm.log")"
        kv "RATCHET_MIB_${side}_${size}" "$(probe_ratchet "$OUT/$side-$size-warm.log")"
        kv "EXAMPLES_${side}_${size}" "$n"
    done

    if [ "$SKIP_BENCH" -eq 0 ]; then
        # Warm cache by now (the large arms just ran), so this times steps
        # rather than the encode phase.
        run_arm "$worktree" bench_step "$side-bench" "$OUT/config-$side-large.yaml" \
            --steps "$BENCH_STEPS"
        grep -E '^(RESULT|SANITY|MODEL)' "$OUT/$side-bench.log" \
            | sed "s/^/BENCH_${side} /" >> "$SUMMARY" || true
    fi
done

# ------------------------------------------------------------------- verdict --

log "summary"
before_small="$(sed -n 's/^PEAK_MIB_before_small=//p' "$SUMMARY" | tail -1)"
before_large="$(sed -n 's/^PEAK_MIB_before_large=//p' "$SUMMARY" | tail -1)"
after_small="$(sed -n 's/^PEAK_MIB_after_small=//p' "$SUMMARY" | tail -1)"
after_large="$(sed -n 's/^PEAK_MIB_after_large=//p' "$SUMMARY" | tail -1)"

if [ -n "$before_small$before_large$after_small$after_large" ] &&
   [ -n "$before_small" ] && [ -n "$before_large" ] &&
   [ -n "$after_small" ] && [ -n "$after_large" ]; then
    kv "PEAK_SLOPE_MIB_PER_EXAMPLE_before" \
       "$(awk -v a="$before_small" -v b="$before_large" -v n="$N_SMALL" -v m="$N_LARGE" \
            'BEGIN { printf "%.2f", (b - a) / (m - n) }')"
    kv "PEAK_SLOPE_MIB_PER_EXAMPLE_after" \
       "$(awk -v a="$after_small" -v b="$after_large" -v n="$N_SMALL" -v m="$N_LARGE" \
            'BEGIN { printf "%.2f", (b - a) / (m - n) }')"
    kv "STATUS" "COMPLETE"
else
    # A missing peak is usually an OOM abort on a `before` arm, which is the
    # #175 bug demonstrating itself. Say so rather than print a slope over a
    # hole -- the last `vram peak so far` ratchet in that arm's log is the
    # measurement in that case.
    kv "STATUS" "INCOMPLETE_SEE_LOGS"
fi

cat "$SUMMARY"
echo
echo "raw logs: $OUT"
