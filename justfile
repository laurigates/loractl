# Forward recipe arguments to shell bodies as real positionals ($1.., $@), so a
# recipe can pass `"$@"` through with quoting/backslashes byte-intact (the
# step-probe regexes below need this). Existing `{{...}}` interpolations are
# unaffected.
set positional-arguments

default:
    @just --list

# Build the whole workspace.
build:
    cargo build

# Install the `loractl` binary via `cargo install`, with the GPU backend
# feature detected from the host: macOS → wgpu (Metal); Linux with nvcc →
# cuda,wgpu; otherwise wgpu (Vulkan/DX12). Override with an explicit feature
# list (`just install cuda`) or `just install cpu` for the ndarray-only build.
# Runtime backend selection stays in config (`compute.backend`) — the feature
# only compiles the backend in. Feature matrix: README § Install.
install features="detect":
    #!/usr/bin/env bash
    set -euo pipefail
    features="{{features}}"
    if [ "$features" = "detect" ]; then
        case "$(uname -s)" in
            Darwin) features="wgpu" ;;
            *) if command -v nvcc >/dev/null 2>&1; then features="cuda,wgpu"; else features="wgpu"; fi ;;
        esac
        echo "install: detected features: $features (override: just install <features> | cpu)"
    fi
    if [ "$features" = "cpu" ]; then
        cargo install --path crates/loractl-cli
    else
        cargo install --path crates/loractl-cli --features "$features"
    fi

# Install the NVIDIA CUDA toolkit (nvcc — what the `cuda` build feature needs)
# from NVIDIA's official apt repo, on the GPU host itself. The version is
# detected from the installed driver's ceiling (`nvidia-smi`'s "CUDA Version"),
# overridable: `just install-cuda 12.9`. Installs `cuda-toolkit-X-Y` ONLY —
# never the `cuda`/`cuda-drivers` metapackages, which would replace a
# distro-managed driver (on Pop!_OS that breaks the system76 driver stack; its
# own repo is no alternative, system76-cuda-latest is stuck at 11.2 < sm_89).
# Ends by symlinking nvcc into /usr/local/bin so `command -v nvcc` (and
# `just install`'s detection) sees it without shell-profile surgery.
install-cuda version="detect":
    #!/usr/bin/env bash
    set -euo pipefail
    [ "$(uname -s)" = "Linux" ] || { echo "install-cuda: Linux-only — run it on the GPU host, not from a Mac" >&2; exit 1; }
    command -v nvidia-smi >/dev/null || { echo "install-cuda: nvidia-smi not found — install the NVIDIA driver first" >&2; exit 1; }
    command -v apt-get >/dev/null || { echo "install-cuda: apt-based distros only; see https://developer.nvidia.com/cuda-downloads" >&2; exit 1; }
    version="{{version}}"
    if [ "$version" = "detect" ]; then
        if command -v nvcc >/dev/null; then
            echo "install-cuda: nvcc already present ($(nvcc --version | grep -o 'release [0-9.]*')) — pass a version explicitly to install another"
            exit 0
        fi
        version=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*')
        [ -n "$version" ] || { echo "install-cuda: could not read the driver's CUDA ceiling from nvidia-smi" >&2; exit 1; }
        echo "install-cuda: driver supports up to CUDA $version"
    fi
    major="${version%%.*}"; minor="${version#*.}"; minor="${minor%%.*}"
    . /etc/os-release
    case " $ID ${ID_LIKE:-} " in
        *ubuntu*) repo="ubuntu$(echo "${VERSION_ID}" | tr -d .)" ;;
        *) echo "install-cuda: only Ubuntu-family distros are mapped to an NVIDIA repo (got ID=$ID); see https://developer.nvidia.com/cuda-downloads" >&2; exit 1 ;;
    esac
    if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
        wget -qO /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/${repo}/x86_64/cuda-keyring_1.1-1_all.deb"
        sudo dpkg -i /tmp/cuda-keyring.deb
    fi
    sudo apt-get update
    sudo apt-get install -y "cuda-toolkit-${major}-${minor}"
    sudo ln -sf "/usr/local/cuda-${major}.${minor}/bin/nvcc" /usr/local/bin/nvcc
    nvcc --version
    echo "install-cuda: done — 'just install' will now detect the cuda feature"

# Clean build artifacts.
clean:
    cargo clean

# Run the CLI with arbitrary args, e.g. `just run train config/examples/lora.yaml`.
run *ARGS:
    cargo run -p loractl-cli -- {{ARGS}}

# Scaffold a starter training config from a template to stdout — redirect to a
# file, e.g. `just init krea2 > config/my.yaml`. Presets: synthetic (default),
# wgpu, flow, krea2. To write directly with an overwrite guard, call the binary:
# `loractl init --preset krea2 -o config/my.yaml`.
init preset="synthetic":
    cargo run -q -p loractl-cli -- init --preset {{preset}}

# Train from a config with the real BurnTrainer (synthetic LoRA-MLP demo by default).
train config="config/examples/lora.yaml":
    cargo run -p loractl-cli -- train {{config}}

# Serve the HTTP/SSE API (bind addr via LORACTL_API_ADDR, default 127.0.0.1:3000).
# Try: curl -sX POST localhost:3000/runs -H 'content-type: application/json' -d @run.json
# then: curl -N localhost:3000/runs/1/events
serve:
    cargo run -p loractl-api

# Print shell completions, e.g. `just completions fish`.
completions shell="zsh":
    cargo run -p loractl-cli -- completions {{shell}}

# Lint with clippy, warnings-as-errors. DEFAULT features only — the `mnist`
# feature pulls the burn-dataset HTTP downloader (reqwest/tokio), so keeping it
# out of the default gate preserves an offline, fast `just lint`. The mnist code
# path is linted on demand via `just lint-mnist`.
lint:
    cargo clippy --all-targets -- -D warnings

# Lint the opt-in mnist feature path (compiles burn-vision/reqwest/tokio; no network).
lint-mnist:
    cargo clippy -p loractl-core --all-targets --features mnist -- -D warnings

# Lint the opt-in gpt2-real feature path (compiles the ignored real-gpt2 test).
lint-gpt2-real:
    cargo clippy -p loractl-core --all-targets --features gpt2-real -- -D warnings

# Lint the opt-in qwen-vae-real feature path (compiles the ignored real-VAE test).
lint-vae-real:
    cargo clippy -p loractl-core --all-targets --features qwen-vae-real -- -D warnings

# Lint the opt-in qwen3vl-real feature path (compiles the ignored real-encoder test).
lint-qwen3vl-real:
    cargo clippy -p loractl-core --all-targets --features qwen3vl-real -- -D warnings

# Lint the opt-in mmdit-real feature path (compiles the ignored real-MMDiT test).
lint-mmdit-real:
    cargo clippy -p loractl-core --all-targets --features mmdit-real -- -D warnings

# Lint the opt-in wgpu GPU-backend path (M7). Compiles the cubecl/wgpu/naga
# subtree + the gated wgpu smoke test; no GPU is needed to COMPILE and nothing
# runs. Kept out of the default `just lint` so that stays offline and fast.
lint-wgpu:
    cargo clippy -p loractl-core --all-targets --features wgpu -- -D warnings

# Lint the opt-in candle GPU-backend path (Metal via candle-core; the bf16
# arm). macOS-ONLY: candle-metal pulls objc2, which refuses to compile on
# non-Apple platforms — so this recipe is local-only, not mirrored in CI
# (the inverse of the cuda/tch situation).
lint-candle:
    cargo clippy -p loractl-core --all-targets --features candle -- -D warnings

# NOTE: cuda/tch are intentionally NOT local lint recipes — burn-cuda needs the
# CUDA toolkit/nvcc and burn-tch a linked libtorch, neither present on this Mac.
# They are build-verifiable only on a Linux+NVIDIA / libtorch host; on such a
# host, `just test-cuda` runs the cuda smoke.

# Format the workspace.
fmt:
    cargo fmt --all

# Check formatting without writing (CI parity).
fmt-check:
    cargo fmt --all -- --check

# Run tests (offline, fast — mnist not enabled).
#
# `--examples` is a SECOND invocation on purpose: cargo defaults examples to
# `test = false`, so a plain `cargo test` only *builds* them and silently skips
# any `#[cfg(test)]` they carry. The measurement examples (`bench_step`,
# `step_probe`, `quant_probe`) are load-bearing tools whose flag contracts CI
# dispatches depend on, so their tests have to actually run — the same
# examples-are-a-blind-spot problem `.claude/rules/cargo-gate-coverage.md`
# documents for `cargo build`. Keep in sync with ci.yml's test job.
test:
    cargo test
    cargo test --examples

# Dead-code report: `pub` items nothing outside their own file names, plus
# manifest dependencies nothing imports. REPORTING ONLY, deliberately not part
# of `just lint` — the pub-item scan matches names textually, so it cannot see
# an item reached through type inference (`model.transformer.h[0]` never writes
# `Transformer`). On its one run in anger, 10 of 49 candidates were genuinely
# internal — so expect to reject most of what it prints. Treat the output as
# candidates and let `cargo check --all-targets` + `cargo doc` adjudicate each
# one; scripts/pub-census.sh's header documents the workflow.
#
# Deliberately NOT wired in as clippy lints, each measured on this tree:
#   unused_crate_dependencies  797 hits — fires per-target for every dep a
#                              given target does not use. Unusable.
#   unreachable_pub             26 hits — 5 are the idiomatic `pub fn` in
#                              tests/common/mod.rs, which is not a defect.
#   clippy::redundant_clone     11 hits — nursery, i.e. explicitly unstable;
#                              double-reports the same expression.
#   clippy::needless_pass_by_value  10 hits — all in the trainer entry points
#                              a future decomposition will rewrite anyway.
# Needs cargo-machete (`cargo install cargo-machete`); skipped if absent.
dead-code:
    @./scripts/pub-census.sh
    @echo
    @if command -v cargo-machete >/dev/null 2>&1; then cargo machete; else \
        echo "cargo-machete not installed — skipping the unused-dependency scan."; \
        echo "  cargo install cargo-machete"; fi

# Test-coverage summary via cargo-llvm-cov (default/offline features). Prints a
# per-file table to stdout; local only, no thresholds or CI upload. For a
# browsable report add `--html --open`; for CI add `--lcov --output-path …`.
coverage:
    cargo llvm-cov

# RustSec advisory scan of Cargo.lock (CI parity with
# .github/workflows/security-audit.yml). Accepted advisories are documented in
# .cargo/audit.toml; needs cargo-audit (`cargo install cargo-audit`).
audit:
    cargo audit

# Supply-chain gate: licenses + banned/duplicate crates + crate sources, per
# deny.toml (CI parity with the `deny` job in .github/workflows/ci.yml). Default
# features only, matching the committed Cargo.lock; advisories live in
# .cargo/audit.toml (see `just audit`), not deny.toml. Needs cargo-deny.
deny:
    cargo deny check licenses bans sources

# Anchors in hubs/*.md pin prose claims to code symbols; this fails when an
# anchored symbol's logic fingerprint moves. Cosmetic edits do not fire.
# Needs `surf` on PATH: https://github.com/Connorrmcd6/surface (see CONTRIBUTING).
# On a divergence, re-read the printed claim and fix the prose, THEN `surf verify`.
# Docs↔code drift gate (CI parity with .github/workflows/docs-drift.yml).
surf:
    surf lint
    surf check

# Run the (network + heavy) MNIST LoRA convergence proof — not part of `just test`.
test-mnist:
    cargo test -p loractl-core --features mnist -- --ignored mnist_lora_converges

# Run the opt-in real-GPT-2 forward-parity test (needs `just gpt2-reference` first).
test-gpt2-real:
    cargo test -p loractl-core --features gpt2-real -- --ignored real_gpt2_forward_matches_pytorch_golden

# Run the opt-in real Qwen-Image VAE parity test (needs `just vae-real-reference`
# first). --release: the real decoder is tens of GMACs; debug ndarray crawls.
test-vae-real:
    cargo test --release -p loractl-core --features qwen-vae-real -- --ignored real_qwen_vae_encode_decode_matches_diffusers_golden

# Run the opt-in real Krea-2-Raw text-encoder parity test (needs
# `just qwen3vl-real-reference` first). --release: a 4B-parameter forward.
test-qwen3vl-real:
    cargo test --release -p loractl-core --features qwen3vl-real -- --ignored real_qwen3vl_conditioning_matches_transformers_golden

# Run the opt-in real Krea-2-Raw MMDiT staged-parity test (needs
# `just mmdit-real-reference` first). Depth-truncated real widths — see the test docs.
test-mmdit-real:
    cargo test --release -p loractl-core --features mmdit-real -- --ignored real_mmdit_truncated_forward_matches_krea2_golden

# Run the opt-in real scaled-fp8 turbo proof (M15; depth-truncated like
# test-mmdit-real). Pass the path to a ComfyUI-style scaled-fp8 checkpoint,
# e.g. the local krea2_turbo_fp8_scaled.safetensors (13.1 GB).
test-turbo-real fp8_path:
    LORACTL_TURBO_FP8={{fp8_path}} cargo test --release -p loractl-core --features mmdit-real -- --ignored turbo_fp8_real

# Run the wgpu GPU smokes (M7 portability + the M13 f16/grad-checkpointing
# variant) on a real GPU — Metal on Apple Silicon. The ONLY way the
# double-gated `#[ignore]`d smokes run; never fires in CI.
test-wgpu:
    cargo test -p loractl-core --features wgpu -- --ignored wgpu

# Run the cuda GPU smokes (the synthetic f32 + grad-checkpointing smoke and
# the tiny-krea2 diffusion e2e) on a real NVIDIA GPU — Linux + CUDA toolkit
# at build time; NOT runnable on this Mac (see the cuda/tch note above). The
# ONLY way the double-gated `#[ignore]`d cuda tests run.
test-cuda:
    cargo test -p loractl-core --features cuda -- --ignored cuda

# Train on an NVIDIA GPU through the real CLI, backend selected purely from
# config/flags. f32-only (burn#5162 breaks non-f32 autodiff on cuda).
run-cuda config="config/examples/lora.yaml":
    cargo run --release -p loractl-cli --features cuda -- train {{config}} --backend cuda

# On-box quantization validation (#96): load a real Krea-2 denoiser at full
# krea2() depth on cuda as int8 (default) or int4 and report coverage, resident
# VRAM (int8 ≈ 13–15 GB, int4 ≈ 8 GB), and real-weight dequant error.
# Linux+NVIDIA + a 24 GB GPU; needs the multi-GB checkpoint. int4: `just
# quant-probe <denoiser> int4`.
quant-probe denoiser quant="int8":
    cargo run --release -p loractl-core --features cuda --example quant_probe -- {{denoiser}} --quant {{quant}}

# ADR-0005 step-VRAM probe (#96, #25): run a few REAL DiffusionTrainer steps
# from a config and report peak resident VRAM per LoRA target set, so this is
# the sweep that confirms a config fits the 24 GB card. Note what Addendum 2
# corrected: retention is topology-driven, so trained-site count is NOT the
# lever it was first thought to be, and demand DOES scale with sequence
# length — i.e. with resolution (the earlier "resolution is not a lever"
# reading came from two arms that both rode the ceiling). Re-probe after
# changing resolution or targets. Linux+NVIDIA + the real
# multi-GB checkpoints; a warm encode cache skips the slow CPU encode phase.
# e.g. `just step-probe config/examples/krea2-comfyui.yaml --target 'attn\.(wq|wv)$' --steps 3`
# Args forward through `"$@"` (set positional-arguments, top of file) so a
# quoted regex like 'attn\.(wq|wv)$' or 'blocks\.\d+\.attn\.' reaches the probe
# byte-intact — a bare {{args}} interpolation would strip the quoting and
# corrupt the pattern (0 sites matched → bogus fit verdict).
step-probe *args:
    cargo run --release -p loractl-core --features cuda --example step_probe -- "$@"

# #110 step-throughput bench: time REAL training steps and report the
# RESULT/SANITY/MODEL lines. The sibling of step-probe — that one answers "does
# this config fit the card", this one "how fast does it run". Needed before any
# lever that spends throughput to buy VRAM (#158/ADR-0008) can be judged, and
# for the int8/int4 QLoRA throughput question (#96). Args forward through "$@"
# for the same quoting reason as step-probe.
# e.g. `just bench config/examples/krea2-comfyui.yaml --steps 8 --seq-len 1536`
# Pass --seq-len: without it the token count is image-only and understates
# tok_s/tflops (ms= is measured either way). See the example's header.
# cuda-only, and there is deliberately no `bench-wgpu`: the harness itself is
# backend-agnostic (it reads `compute.backend`), but a wgpu real-model run has
# nothing trustworthy to time until burn#5162 is fixed — timing a numerically
# corrupt run would produce a real number for work nobody wants. Build with
# --features wgpu by hand if that changes.
bench *args:
    cargo run --release -p loractl-core --features cuda --example bench_step -- "$@"

# The offline arm of the bench: the tiny-krea2 fixture on ndarray, no GPU. Its
# NUMBERS are meaningless (a 2-block toy at 32px) — it exists to keep the
# harness itself runnable and honest on any host. Stages a dataset copy under
# tmp/ so the checked-in fixture never grows a .loractl-cache, exactly as
# ledger-probe does.
bench-offline *args:
    rm -rf tmp/bench
    mkdir -p tmp/bench/dataset
    cp crates/loractl-core/tests/fixtures/dataset-tiny/* tmp/bench/dataset/
    cargo run --release -p loractl-core --example bench_step -- config/probes/tiny-bench.yaml "$@"

# Materialize a PINNED PUBLIC captioned dataset into loractl's image + .txt
# layout. `just dataset-fetch` with no arguments lists the registry; otherwise
# `just dataset-fetch tuxemon tmp/datasets/tuxemon-56 56`.
#
# Public and revision-pinned on purpose: #175/#178 close on before/after
# numbers, and a number measured against a folder only one person has is an
# assertion rather than a result. Entries are taken in filename order, so a
# smaller `n` is a strict PREFIX of a larger one — which is what lets the
# residency matrix vary example count and nothing else.
#
# Set HF_HOME to a roomy disk first; the default cache lands on the root disk,
# which is the one that fills (see .claude/rules/gpu-runner-failure-signatures.md).
dataset-fetch key="" out="" n="0":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{key}}" ]; then exec ./scripts/fetch_dataset.py --list; fi
    out="{{out}}"
    [ -n "$out" ] || out="tmp/datasets/{{key}}-{{n}}"
    ./scripts/fetch_dataset.py --dataset "{{key}}" --out "$out" --limit "{{n}}"

# The #175/#178 on-box acceptance matrix: peak VRAM across several dataset
# sizes on BOTH sides of a revision pair, plus a bench arm so the per-step read
# the lazy loader added is priced rather than assumed.
#
# This is the measurement PR #197 explicitly did not claim — it needs the real
# 24 GB card and the multi-GB checkpoints, and #175's criterion is a SCALING
# claim ("peak no longer scales with example count"), which a single
# before/after pair cannot answer. Default is three sizes (8,24,40): two points
# always fit a line exactly, so only a third can test the linearity an
# O(dataset) bug predicts. `--sizes` overrides.
#
# Cost is one cold encode at the largest size (~8.5 min/image, measured) plus
# ~2 min per warm arm — the encode cache is shared across revisions and reused
# for every prefix, so the third point is nearly free.
#
# The card must be idle: the probe reads whole-GPU nvidia-smi, so a resident
# ComfyUI is added to every peak. The script refuses rather than report a
# poisoned number; `--free-endpoint http://127.0.0.1:8188` drops ComfyUI's idle
# weight cache without stopping it, `--stop-service comfyui.service` is the
# hammer. `--dry-run` runs the preflight alone, which is the cheap way to
# confirm a scheduled window will actually work.
#
# e.g. just residency-matrix --models-root ~/ComfyUI/models --free-endpoint http://127.0.0.1:8188
residency-matrix *args:
    ./scripts/residency-matrix.sh "$@"

# #132 retention-ledger validation (offline, ndarray): run both checkpointing
# arms over the tiny-krea2 fixture with the ledger enabled, then report each.
# Stages a dataset copy under tmp/ so the checked-in fixture never grows a
# .loractl-cache.
#
# DEGRADED WITHOUT THE FORK PIN: burn's Computed/DROP/BUILD event lines came
# from a burn-autodiff fork pinned via the workspace [patch.crates-io] block,
# removed once #132 closed. As it stands this recipe writes only loractl's own
# PHASE markers, and scripts/ledger-report.py aggregates whatever is present.
# TO RE-PIN: restore PR #133's [patch.crates-io] block (all 26 burn crates ->
# laurigates/burn @ feat/retention-ledger-v0.21 = v0.21.0 + 12b0ba1e/07c099e8)
# plus deny.toml's matching allow-git entry. Patch the WHOLE burn workspace,
# not just burn-autodiff, or the Backend traits stop unifying. NOTE: that
# branch is v0.21-based — after the burn 0.22 migration (#79) the two ledger
# commits must be PORTED forward, not merely re-pinned.
ledger-probe:
    rm -rf tmp/ledger-probe
    mkdir -p tmp/ledger-probe/dataset
    cp crates/loractl-core/tests/fixtures/dataset-tiny/* tmp/ledger-probe/dataset/
    LORACTL_RETENTION_LEDGER=tmp/ledger-probe/balanced.ledger cargo run --release -p loractl-core --example step_probe -- config/probes/tiny-ledger-balanced.yaml
    LORACTL_RETENTION_LEDGER=tmp/ledger-probe/nockpt.ledger cargo run --release -p loractl-core --example step_probe -- config/probes/tiny-ledger-nockpt.yaml
    python3 scripts/ledger-report.py tmp/ledger-probe/balanced.ledger
    python3 scripts/ledger-report.py tmp/ledger-probe/nockpt.ledger

# End-to-end acceptance #1: train on the GPU through the real CLI, backend
# selected purely from config (`compute.backend: wgpu`). Metal on this Mac.
run-wgpu config="config/examples/lora-wgpu.yaml":
    cargo run -p loractl-cli --features wgpu -- train {{config}}

# Regenerate the PyTorch golden fixture for the numerics test (needs torch via uv).
reference:
    uv run reference/lora_reference.py > crates/loractl-core/tests/golden/lora_toy.json

# Regenerate the per-block int8 quantization golden (#96; torch via uv, no network).
quant-reference:
    uv run reference/quant_reference.py

# Regenerate the per-block int4 (Q4S) quantization golden (#96; torch via uv, no network).
quant-int4-reference:
    uv run reference/quant_int4_reference.py

# Regenerate the BurnTrainer step-loss golden (#49 H9; needs torch via uv).
#
# Two-stage, unlike the other references: burn's frozen base, LoRA `A` init, and
# synthetic dataset all come out of its seeded ChaCha RNG, which PyTorch cannot
# reproduce — so torch cannot DERIVE the run's inputs, it has to be GIVEN them.
# Stage 1 dumps burn's actual init + batches (~5 MB, throwaway); stage 2 replays
# the same training loop in torch over that dump and emits the small loss golden,
# which is the only artifact that lands in git.
burn-trainer-reference:
    #!/usr/bin/env bash
    set -euo pipefail
    dump="$(mktemp -d)"
    trap 'rm -rf "$dump"' EXIT
    cargo run -q -p loractl-core --example dump_synthetic_run -- "$dump"
    uv run reference/burn_trainer_reference.py --dump "$dump" > "$dump/golden.json"
    mv "$dump/golden.json" crates/loractl-core/tests/golden/burn_trainer_steps.json

# Regenerate the kohya-ss export golden fixture (numpy only, no torch/network).
export-reference:
    uv run reference/lora_export_reference.py > crates/loractl-core/tests/golden/lora_export.json

# Regenerate the ComfyUI Krea 2 LoRA-key contract golden (#137). Downloads
# comfy/utils.py + comfy/lora.py at a pinned commit; no torch. Fails loudly if
# upstream drops an alias the export depends on — bump COMFY_COMMIT in the
# script deliberately, and re-read the Krea2 branch when you do.
krea2-lora-keys-reference:
    uv run reference/krea2_lora_keys_reference.py --out crates/loractl-core/tests/golden

# Regenerate the training-adapter merge-at-load golden (#83; numpy only, no
# torch/network). Pins the `W += (alpha/rank)·B·A` merge math + layout convention.
training-adapter-merge-reference:
    uv run reference/training_adapter_merge_reference.py > crates/loractl-core/tests/golden/training_adapter_merge.json

# Regenerate the flow-matching golden fixture for the M8 numerics test (needs torch via uv).
flow-reference:
    uv run reference/flow_reference.py > crates/loractl-core/tests/golden/flow_toy.json

# Regenerate the checked-in tiny-GPT-2 parity fixture (weights + golden; torch via uv).
gpt2-tiny-reference:
    uv run reference/gpt2_tiny_reference.py --out crates/loractl-core/tests/fixtures

# Download real gpt2 + generate its (uncommitted) golden for the opt-in parity test.
gpt2-reference:
    uv run reference/gpt2_reference.py --out crates/loractl-core/tests/fixtures

# Regenerate the checked-in tiny Qwen-Image VAE parity fixture (weights + golden;
# torch + diffusers via uv, no network — the tiny VAE is constructed locally).
vae-reference:
    uv run reference/qwen_vae_reference.py --out crates/loractl-core/tests/fixtures

# Download the real Qwen/Qwen-Image VAE (the checkpoint Krea 2 wraps) + generate
# its (uncommitted) f32 safetensors + golden for the opt-in parity test.
vae-real-reference:
    uv run reference/qwen_vae_reference.py --real --out crates/loractl-core/tests/fixtures

# Regenerate the checked-in ComfyUI-native (Qwen/WAN) keyed tiny VAE fixture by
# re-keying the tiny diffusers fixture (numpy + safetensors only, no torch/
# network); self-verifies the rename against diffusers' own converter.
vae-native-reference:
    uv run reference/qwen_vae_native_reference.py --out crates/loractl-core/tests/fixtures

# Regenerate the checked-in tiny Qwen3-VL parity fixture (weights + golden;
# torch + transformers via uv, no network — the tiny model is constructed locally).
qwen3vl-reference:
    uv run reference/qwen3vl_reference.py --out crates/loractl-core/tests/fixtures

# Download krea/Krea-2-Raw's text_encoder + tokenizer (~8.9 GB bf16, re-saved
# f32 ~18 GB) + generate its (uncommitted) goldens for the opt-in parity test.
qwen3vl-real-reference:
    uv run reference/qwen3vl_reference.py --real --out crates/loractl-core/tests/fixtures

# Regenerate the checked-in tiny MMDiT parity fixture (weights + golden;
# downloads krea-ai/krea-2's mmdit.py at a pinned commit; torch via uv).
mmdit-reference:
    uv run reference/mmdit_reference.py --out crates/loractl-core/tests/fixtures

# Download krea/Krea-2-Raw's raw.safetensors (26.3 GB, kept for M14) +
# generate the (uncommitted) depth-truncated real-width fixture + golden.
mmdit-real-reference:
    uv run reference/mmdit_reference.py --real --out crates/loractl-core/tests/fixtures

# Regenerate the checked-in tiny-krea2 composed bundle + tiny dataset (M14;
# torch/diffusers/transformers via uv, no network).
krea2-reference:
    uv run reference/krea2_reference.py

# Regenerate the checked-in fp8 goldens + fp8 tiny fixtures (M15: LUT, dequant
# cases, fp8 tiny-mmdit + golden, tiny-krea2 turbo_fp8 twins). Downloads the
# pinned mmdit.py at regen time only; torch/transformers/diffusers via uv.
fp8-reference:
    uv run reference/fp8_reference.py --out crates/loractl-core/tests/fixtures

# Regenerate the LoRA `__metadata__` consumer-contract golden — the metadata
# keys AUTOMATIC1111's Lora extension reads, extracted from PINNED upstream
# source (tag v1.10.1) rather than transcribed. Needs network; the emitted
# golden keeps `tests/lora_metadata_keys.rs` offline.
lora-metadata-keys-reference:
    uv run reference/lora_metadata_keys_reference.py --out crates/loractl-core/tests/golden
