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
#         A scaling claim, so it needs SEVERAL dataset sizes per revision, not
#         one before/after pair. Pre-#175 code holds every example's
#         conditioning device-resident at [1, 512, 12, 2560] f32 = 62,914,560 B
#         = exactly 60 MiB, so `before` should fit ~60 MiB/example and `after`
#         ~0.
#
#         Three points by default, because 60 MiB/example is a POINT PREDICTION
#         and the claim is about linearity: an O(dataset) bug predicts a
#         straight line, and two points always fit a line exactly, so a
#         two-point run cannot fail the check it exists to perform. The summary
#         reports a least-squares slope AND every consecutive-pair slope; when
#         the segments disagree the fitted slope is summarizing something that
#         is not a line. (Worked example: peaks that saturate after the middle
#         point fit as 30 MiB/example -- readable as "half as bad as predicted"
#         -- while the segments read 60 then 0, which is a different mechanism
#         entirely.)
#
#   #175  "`just bench` before/after, so the per-step read cost is priced"
#         The lazy read trades disk + H2D per step for the residency win. The
#         bench arm is what stops that trade being assumed rather than paid.
#
#   #178  "Cold-cache encode-phase wall time before/after (>=40 images)"
#         RECORDED BUT NOT INFORMATIVE, and the criterion should be re-scoped
#         rather than believed. Measured on this box 2026-08-07: encode work
#         runs ~8.5 min/image (the 4B text encoder), while the decode + Lanczos
#         resize that #178 parallelized is ~36 ms/image -- 0.007% of the phase.
#         No decode speedup, however large, is separable from run-to-run noise
#         at that ratio. The decode fraction needs its own probe.
#
# COST, measured rather than guessed (RTX 4090, 512px, int4):
#
#   cold arm  ~8.5 min/image  + ~10 min of lazy model loads
#   warm arm  ~1.8 min        <- the #175 measurement itself is CHEAP
#
# So essentially all the wall time is cache fill. That is why there is exactly
# ONE cold encode per run -- at the largest size, on the `before` side -- and
# every other arm is warm:
#
#   * The cache is shareable ACROSS REVISIONS (identical key format; the
#     expensive half, conditioning, is keyed `{stem}.{fingerprint}.cond` with
#     NO bucket, so it hits even if bucket assignment differs).
#   * The cache is keyed PER FILE, so the cache for N images already contains
#     the cache for any prefix -- and fetch_dataset.py materializes in filename
#     order, so every smaller arm IS a prefix.
#
# Net: the default 8/24/40 run costs one 40-image encode (~6 h) plus six warm
# arms (~12 min), rather than 19 h. Neither property is taken on trust --
# ENCODED_COUNT is reported for every warm arm and must be 0, so a cache that
# failed to transfer surfaces as a number rather than as a slow run.
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
# Three points, not two. An O(dataset) bug predicts a STRAIGHT line, and two
# points always fit a line perfectly -- they cannot fail that check, so they
# cannot test the very property #175 claims. Three can. Because the encode cache
# is keyed per file, the extra point is free: one cold encode of the largest set
# already contains the cache for every prefix.
#
# 8/24/40 also keeps the `before` arm clear of the ceiling. Predicted peak is
# 20,008 + (N-8)x60 MiB, so N=40 lands ~21.9 GB with ~2.6 GB of margin on a
# 24.5 GB card, where N=56 lands ~22.9 GB and risks losing the very arm that
# demonstrates the bug to an OOM (the probe's gate is a ZERO-PANIC run).
SIZES="8,24,40"
MODELS_ROOT="${LORACTL_MODELS_ROOT:-}"
OUT=""
STOP_SERVICE=""
FREE_ENDPOINT=""
ALLOW_BUSY_GPU=0
SKIP_BENCH=0
DRY_RUN=0
ENCODE_THREADS=1
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
        --sizes) SIZES="$2"; shift 2 ;;
        --models-root) MODELS_ROOT="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --stop-service) STOP_SERVICE="$2"; shift 2 ;;
        --free-endpoint) FREE_ENDPOINT="$2"; shift 2 ;;
        --bench-steps) BENCH_STEPS="$2"; shift 2 ;;
        --allow-busy-gpu) ALLOW_BUSY_GPU=1; shift ;;
        --skip-bench) SKIP_BENCH=1; shift ;;
        --no-encode-threads) ENCODE_THREADS=0; shift ;;
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

# Sizes: ascending, distinct, >=2 points. Ascending matters because every
# smaller arm is derived as a PREFIX of the largest one's cache.
SIZE_LIST="$(tr ',' ' ' <<<"$SIZES")"
N_MAX=0; count=0; prev=0
for n in $SIZE_LIST; do
    case "$n" in ''|*[!0-9]*) die "--sizes: '$n' is not a positive integer" ;; esac
    [ "$n" -gt 0 ] || die "--sizes: sizes must be > 0"
    [ "$n" -gt "$prev" ] || die "--sizes must be strictly ascending; got '$SIZES'"
    prev="$n"; N_MAX="$n"; count=$((count + 1))
done
[ "$count" -ge 2 ] || die "--sizes needs at least 2 points to fit a slope; got '$SIZES'"

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

# Resolve the Qwen3-VL tokenizer up front, and refuse without one.
#
# A ComfyUI layout ships no tokenizer, so the trainer would fall back to
# fetching it from `krea/Krea-2-Raw` -- gated since at least 2026-08-07, so the
# fetch 401s. It does so AFTER loading the 4.9 GiB text encoder, ~9 minutes
# into each arm, which is how the 2026-08-07 run burned eight arms discovering
# the same thing eight times. Checking a hash here costs milliseconds.
#
# `hf.rs` pins the file's SHA-256, so ANY byte-identical copy is provably the
# right tokenizer -- there is no "is this the correct one?" judgement to make.
# That is why this verifies the hash rather than trusting a path.
TOKENIZER_SHA256="be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"
if [ -z "${TOKENIZER:-}" ]; then
    # Portable locations only -- `hf.rs::cache_dir` writes to $HF_HOME/loractl,
    # else $XDG_CACHE_HOME/loractl, else ~/.cache/loractl. A copy anywhere else
    # is what `--tokenizer` is for; a host-specific path does not belong in a
    # committed script.
    for cand in \
        "${HF_HOME:-}/loractl/qwen3vl-4b-tokenizer.json" \
        "${XDG_CACHE_HOME:-}/loractl/qwen3vl-4b-tokenizer.json" \
        "$HOME/.cache/loractl/qwen3vl-4b-tokenizer.json" \
        "$MODELS_ROOT/tokenizer/tokenizer.json"; do
        [ -f "$cand" ] || continue
        [ "$(sha256sum "$cand" | cut -d' ' -f1)" = "$TOKENIZER_SHA256" ] || continue
        TOKENIZER="$cand"
        break
    done
fi
[ -n "${TOKENIZER:-}" ] || die "no Qwen3-VL tokenizer found matching the pinned SHA-256.
  loractl fetches it from krea/Krea-2-Raw, which is now GATED -- accept the terms at
  https://huggingface.co/krea/Krea-2-Raw and re-run, or pass --tokenizer <path> to a
  copy whose sha256 is $TOKENIZER_SHA256"
[ "$(sha256sum "$TOKENIZER" | cut -d' ' -f1)" = "$TOKENIZER_SHA256" ] \
    || die "--tokenizer $TOKENIZER does not match the pinned SHA-256 $TOKENIZER_SHA256"
TOKENIZER="$(cd "$(dirname "$TOKENIZER")" && pwd)/$(basename "$TOKENIZER")"

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
echo "  dataset: $DATASET_KEY, arms at $SIZES examples (one cold encode at $N_MAX)"
echo "  gpu:     ${gpu_used} MiB used of ${gpu_total} MiB"
echo "  disk:    ${free_gib} GiB free"
echo "  tokenizer: $TOKENIZER (sha256 verified)"

[ "$DRY_RUN" -eq 1 ] && { echo "STATUS=DRY_RUN_OK"; exit 0; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-tmp/residency-matrix/$STAMP}"
mkdir -p "$OUT"
# ABSOLUTE from here on. Every arm runs with cwd set to its worktree, so a
# relative path in the rendered config resolves against the wrong tree. This is
# not hypothetical and it does not fail cleanly: figment's `Format::file()`
# searches PARENT directories, so a relative config path is still found by
# walking up from the worktree to the repo root -- while `dataset.path`, read by
# a plain `read_dir`, is not. The 2026-08-07 run lost all eight arms to exactly
# that split (config loaded, dataset ENOENT).
OUT="$(cd "$OUT" && pwd)"
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

# Wall seconds spanned by the `encode` phase, first encode report to the first
# `dataset:` report (the phase that strictly follows it).
#
# The obvious definition -- "ends at the first non-encode line" -- is WRONG, and
# wrong in a way that reads as a plausible number rather than an error. Phases
# INTERLEAVE: the VAE and text encoder are loaded lazily from inside the encode
# phase, so the real sequence is
#     encode: scanning ... | load: VAE | load: text encoder | encode: 1/N | ...
# and the naive rule terminated the phase at the `load: VAE` line 17 ms in,
# reporting 0.0 s for a phase that actually ran 78 minutes (observed
# 2026-08-07). `dataset:` is the correct terminator because reading the prepared
# cache back cannot begin until every entry is encoded.
# How many entries were actually ENCODED (as opposed to served from cache). The
# phase reports `encode: encoding <name>` on a miss and `encode: cached <name>`
# on a hit, so this is also the direct on-hardware evidence for #175's
# "warm-epoch no-encoder guarantee" box: a warm arm must report 0.
encoded_count() { grep -c ' encode: encoding ' "$1" || true; }

encode_wall() {
    # Empty when nothing was encoded. A warm arm still opens the phase to scan
    # the directory and check cache keys (~0.6 s for 8 entries), and reporting
    # that as "encode wall" would invite comparing it against a cold arm's
    # minutes-per-image as though they measured the same thing.
    [ "$(encoded_count "$1")" -gt 0 ] || return 0
    awk '
        !start && $0 ~ / encode: / { start = $1 }
        start && $0 ~ / dataset: / { printf "%.1f", $1 - start; exit }
    ' "$1"
}

# Wall seconds of encode WORK only -- from the last lazy `load:` inside the
# phase to its end. The difference from encode_wall is the model-load cost,
# which is a fixed ~10 min on this stack and identical on both sides, so it
# dilutes a before/after comparison without informing it.
#
# Neither of these can answer #178. Measured 2026-08-07: encode work runs
# ~8.5 min/image (4B text encoder), while the decode+resize this phase's
# parallelization targets is ~36 ms/image -- 0.007% of the span. The decode
# fraction needs its own probe; see the note on issue #178.
encode_work() {
    awk '
        $0 ~ / load: / && !ended { lastload = $1 }
        lastload && $0 ~ / encode: / && !first { first = $1 }
        first && $0 ~ / dataset: / { printf "%.1f", $1 - first; ended = 1; exit }
    ' "$1"
}

# `peak_mib=` from the greppable STEP_PROBE_SUMMARY line.
probe_peak() { sed -n 's/.*peak_mib=\([0-9]*\).*/\1/p' "$1" | tail -1; }

# Did the arm actually run its steps? `yes` only when STEP_PROBE_SUMMARY reports
# steps=N/N with N > 0.
#
# THIS IS THE GUARD THAT STOPS THE HARNESS LYING. step_probe prints its summary
# on FAILURE too (deliberately -- a partial run is the measurement when a config
# OOMs), so a run that died before step 1 still emits a well-formed
# `peak_mib=<baseline>`. Without this check the verdict logic saw four non-empty
# peaks, computed a slope over four baselines, and printed
# `STATUS=COMPLETE ... SLOPE=0.00` -- the exact shape of a clean pass, from a
# matrix in which nothing ran at all. Observed 2026-08-07 across all eight arms.
probe_steps_ok() {
    local s
    s="$(sed -n 's/.*STEP_PROBE_SUMMARY.* steps=\([0-9]*\)\/\([0-9]*\) .*/\1 \2/p' "$1" | tail -1)"
    [ -n "$s" ] || { echo no; return; }
    awk -v a="${s% *}" -v b="${s#* }" 'BEGIN { print (a > 0 && a == b) ? "yes" : "no" }'
}

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
        -e "s|@TOKENIZER@|$TOKENIZER|g" \
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

log "materializing the dataset ($N_MAX images, the largest arm)"
DS_ROOT="$REPO_ROOT/tmp/datasets/${DATASET_KEY}-${N_MAX}"
./scripts/fetch_dataset.py --dataset "$DATASET_KEY" --out "$DS_ROOT" --limit "$N_MAX"

: > "$SUMMARY"
{
    echo "# residency matrix $STAMP"
    echo "# before=$BEFORE_REV ($BEFORE_SHA)  after=$AFTER_REV ($AFTER_SHA)"
    echo "# dataset=$DATASET_KEY sizes=$SIZES"
    echo "# host_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
} >> "$SUMMARY"

# --------------------------------------------------------------- build sides --

# Re-enable burn-ndarray's DEFAULT `multi-threads` (rayon +
# `matrixmultiply/threading`) in the worktree being built.
#
# loractl-core declares the burn umbrella with `default-features = false`, which
# silently drops it, so the encode phase -- which `diffusion_trainer.rs` forces
# onto the ndarray CPU backend on purpose (f16 overflows the Qwen encoders and
# burn 0.21's wgpu f32 corrupted sequential encoder outputs) -- runs the 4B text
# encoder on ONE core of 24.
#
# MEASURED on this box, same two images, cold cache:
#     without: 598.7 s/image  (5 consecutive gaps: 628 637 576 576 577)
#     with:    120.8 s/image
#   -> ~5x, turning a 40-image encode from ~6.7 h into ~1.4 h.
#
# This is a BUILD-CONFIG change applied identically to both sides, not a change
# to the revisions under test, and it cannot move the measured quantity: peak
# VRAM is sampled during the cuda training loop, which happens after the encode
# and shares nothing with burn-ndarray's host-side threading. It is applied here
# rather than committed to the revisions so `--before`/`--after` keep naming the
# real commits.
#
# `simd` is deliberately NOT included. It breaks
# `grad_checkpointing::checkpointing_is_numerically_identical_to_stored_activations`
# (SIMD reduction order makes a checkpoint replay differ in the 7th significant
# digit) and it is the smaller lever anyway -- 78.8 vs 94.2 ms/matmul, against
# 32.0 for multi-threads. Isolated by feature: multi-threads alone passes that
# test, simd alone fails it, at any thread count.
#
# `--no-encode-threads` opts out. The knob exists because the threaded GEMM
# reassociates float sums, so latent bytes are not identical to a serial encode
# -- irrelevant to a VRAM measurement, but if a future arm ever compares latent
# VALUES it must be turned off.
enable_ndarray_threads() {
    local wt="$1" manifest="$1/crates/loractl-core/Cargo.toml"
    grep -q '^burn-ndarray = .*multi-threads' "$manifest" && return 0
    python3 - "$manifest" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
anchor = 'burn-store = { version = "0.21.0"'
if anchor not in s:
    sys.exit(f"residency-matrix: cannot find the [dependencies] anchor in {p}")
dep = ('burn-ndarray = { version = "0.21", default-features = false, features = '
       '["std", "multi-threads"] }\n')
p.write_text(s.replace(anchor, dep + anchor, 1))
PY
}

for side in before after; do
    if [ "$side" = before ]; then rev="$BEFORE_REV"; sha="$BEFORE_SHA"; else rev="$AFTER_REV"; sha="$AFTER_SHA"; fi
    worktree="$REPO_ROOT/tmp/residency-matrix/worktrees/$side"
    log "$side: worktree at $sha"
    git worktree remove --force "$worktree" 2>/dev/null || true
    git worktree add --detach --force "$worktree" "$rev" >/dev/null
    if [ "$ENCODE_THREADS" -eq 1 ]; then
        enable_ndarray_threads "$worktree" || die "$side: could not enable ndarray threads"
        echo "  build config: burn-ndarray multi-threads ON (~5x encode; see the header)"
    fi
    ( cd "$worktree" && cargo build --release -p loractl-core --features cuda \
        --example step_probe --example bench_step ) || die "$side: build failed at $sha"
done

# ------------------------------------------------------- one cold encode only --
#
# The encode is the entire cost of this matrix (~8.5 min/image against a ~1.8
# min warm arm), so it runs ONCE, at the largest size, on the `before` side --
# not once per side and not once per size.
#
# Two properties make that sound rather than a shortcut:
#
#   * The cache is shareable across the revisions. Its key format is identical
#     on both, and the expensive half -- conditioning -- is keyed
#     `{stem}.{fingerprint}.cond` with NO bucket component, so it hits even if
#     bucket assignment differs.
#   * The cache is keyed PER FILE, so the cache for N images already contains
#     the cache for any prefix of them. Since fetch_dataset.py materializes in
#     filename order, every smaller arm IS a prefix.
#
# This is not taken on trust: every warm arm reports ENCODED_COUNT, so a cache
# that failed to transfer shows up as a non-zero count on that arm rather than
# as a silently slower run. Sharing is what makes a three-point line cost the
# same as a two-point delta.

COLD_SIDE=before
COLD_WT="$REPO_ROOT/tmp/residency-matrix/worktrees/$COLD_SIDE"
SHARED="$OUT/data/shared"
rm -rf "$SHARED"; mkdir -p "$SHARED"
cp "$DS_ROOT"/*.jpg "$DS_ROOT"/*.txt "$SHARED"/

# Seed from a cache left in the dataset root, so an interrupted encode RESUMES
# instead of starting over. At ~2 min/image (~10 without threading) a run killed
# 6 images in used to throw that away entirely. The fingerprint is encoder
# identity and carries nothing about the build, so entries written by an earlier
# run -- including one built without `multi-threads` -- are still valid hits.
# Copy a cache here to reuse it:  cp -a <old-run>/data/shared/.loractl-cache tmp/datasets/<key>-<N>/
if [ -d "$DS_ROOT/.loractl-cache" ]; then
    cp -a "$DS_ROOT/.loractl-cache" "$SHARED"/
    echo "seeded $(ls "$SHARED"/.loractl-cache | wc -l) cache entries from $DS_ROOT"
fi

log "cold encode: $N_MAX images on $COLD_SIDE (the slow part -- ~8.5 min/image)"
render_config "$SHARED" "$OUT/run/cold" "$OUT/config-cold.yaml"
run_arm "$COLD_WT" step_probe "cold-$N_MAX" "$OUT/config-cold.yaml"
kv "ARM_OK_cold" "$(probe_steps_ok "$OUT/cold-$N_MAX.log")"
kv "ENCODED_COUNT_cold" "$(encoded_count "$OUT/cold-$N_MAX.log")"
kv "ENCODE_WALL_S_cold" "$(encode_wall "$OUT/cold-$N_MAX.log")"
kv "ENCODE_WORK_S_cold" "$(encode_work "$OUT/cold-$N_MAX.log")"
[ "$(probe_steps_ok "$OUT/cold-$N_MAX.log")" = yes ] \
    || die "the cold encode arm did not complete -- every warm arm below would
  measure an incomplete cache. See $OUT/cold-$N_MAX.log"

# ---------------------------------------------------------------------- arms --
#
# Warm arms only, ~1.8 min each. Each size gets its own directory holding the
# first N images plus THEIR cache entries, copied out of the shared encode. Both
# cache filename forms -- `{file_name}.{w}x{h}.{fp}.latent` and
# `{stem}.{fp}.cond` -- begin with the stem, so one glob per image takes both.

for side in before after; do
    worktree="$REPO_ROOT/tmp/residency-matrix/worktrees/$side"
    log "$side: warm arms at sizes $SIZES"

    for n in $SIZE_LIST; do
        data="$OUT/data/$side/$n"
        rm -rf "$data"; mkdir -p "$data/.loractl-cache"
        i=0
        for img in $(cd "$SHARED" && ls *.jpg | sort); do
            i=$((i + 1)); [ "$i" -le "$n" ] || break
            stem="${img%.jpg}"
            cp "$SHARED/$img" "$SHARED/$stem.txt" "$data"/
            cp "$SHARED"/.loractl-cache/"$stem".* "$data"/.loractl-cache/ 2>/dev/null || true
        done

        cfg="$OUT/config-$side-$n.yaml"
        render_config "$data" "$OUT/run/$side-$n" "$cfg"
        run_arm "$worktree" step_probe "$side-$n-warm" "$cfg"

        # Recorded BEFORE the numbers, so a reader sees whether the arm ran
        # before seeing anything derived from it.
        kv "ARM_OK_${side}_${n}" "$(probe_steps_ok "$OUT/$side-$n-warm.log")"
        # Must be 0: the warm-epoch no-encoder guarantee measured on hardware,
        # AND the check that the shared cache transferred to this side.
        kv "ENCODED_COUNT_${side}_${n}" "$(encoded_count "$OUT/$side-$n-warm.log")"
        kv "PEAK_MIB_${side}_${n}" "$(probe_peak "$OUT/$side-$n-warm.log")"
        kv "RATCHET_MIB_${side}_${n}" "$(probe_ratchet "$OUT/$side-$n-warm.log")"
    done

    if [ "$SKIP_BENCH" -eq 0 ]; then
        run_arm "$worktree" bench_step "$side-bench" "$OUT/config-$side-$N_MAX.yaml" \
            --steps "$BENCH_STEPS"
        grep -E '^(RESULT|SANITY|MODEL)' "$OUT/$side-bench.log" \
            | sed "s/^/BENCH_${side} /" >> "$SUMMARY" || true
    fi
done

# ------------------------------------------------------------------- verdict --

log "summary"

# A peak is only a measurement if its arm completed its steps. Requiring
# `steps=N/N` is what separates "the fix works" from "nothing ran" -- both of
# which otherwise present as well-formed peaks and a 0.00 slope.
arms_ok=1
for side in before after; do
    for n in $SIZE_LIST; do
        [ "$(sed -n "s/^ARM_OK_${side}_${n}=//p" "$SUMMARY" | tail -1)" = yes ] || arms_ok=0
        [ -n "$(sed -n "s/^PEAK_MIB_${side}_${n}=//p" "$SUMMARY" | tail -1)" ] || arms_ok=0
    done
done

# Least-squares slope over ALL points, plus every consecutive-pair slope.
#
# The pairwise slopes are the linearity check and the reason for a third point:
# an O(dataset) residency bug predicts a straight line, so the segments should
# agree. If they disagree materially the fitted slope is a summary of something
# that is not a line, and the model of the bug -- not just its magnitude -- is
# in question. A two-point run cannot surface that, because two points always
# fit a line exactly.
fit_slope() {  # args: "n1:peak1 n2:peak2 ..."
    awk -v pts="$1" 'BEGIN {
        k = split(pts, a, " ")
        for (i = 1; i <= k; i++) { split(a[i], p, ":"); x = p[1]; y = p[2]
            sx += x; sy += y; sxy += x * y; sxx += x * x }
        d = k * sxx - sx * sx
        if (d == 0) { print "undefined"; exit }
        printf "%.2f", (k * sxy - sx * sy) / d
    }'
}

if [ "$arms_ok" -eq 1 ]; then
    for side in before after; do
        pts=""
        for n in $SIZE_LIST; do
            pts="$pts $n:$(sed -n "s/^PEAK_MIB_${side}_${n}=//p" "$SUMMARY" | tail -1)"
        done
        kv "PEAK_SLOPE_MIB_PER_EXAMPLE_${side}" "$(fit_slope "${pts# }")"

        prev_n=""; prev_p=""
        for n in $SIZE_LIST; do
            p="$(sed -n "s/^PEAK_MIB_${side}_${n}=//p" "$SUMMARY" | tail -1)"
            [ -n "$prev_n" ] && kv "SEGMENT_SLOPE_${side}_${prev_n}_to_${n}" \
                "$(awk -v a="$prev_p" -v b="$p" -v n="$prev_n" -v m="$n" \
                     'BEGIN { printf "%.2f", (b - a) / (m - n) }')"
            prev_n="$n"; prev_p="$p"
        done
    done
    # 60 MiB/example is arithmetic, not a fitted expectation:
    # [1, 512, 12, 2560] f32 = 512*12*2560*4 = 62,914,560 B = exactly 60 MiB.
    kv "PREDICTED_SLOPE_MIB_PER_EXAMPLE_before" "60.00"
    kv "STATUS" "COMPLETE"
else
    # No slope is printed here ON PURPOSE. An incomplete arm most often means
    # either a harness fault (nothing ran) or an OOM abort on a `before` arm --
    # the latter being the #175 bug demonstrating itself, whose measurement is
    # the last `vram peak so far` ratchet in that arm's log, not a slope.
    # Emitting a number over a hole is how a failed matrix reads as a pass.
    echo "arms that did not complete their steps:"
    grep -E '^ARM_OK_.*=no$' "$SUMMARY" || echo "  (none -- a peak was missing instead)"
    kv "STATUS" "INCOMPLETE_SEE_LOGS"
fi

cat "$SUMMARY"
echo
echo "raw logs: $OUT"
