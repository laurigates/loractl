default:
    @just --list

# Build the whole workspace.
build:
    cargo build

# Run the CLI with arbitrary args, e.g. `just run train config/examples/lora.yaml`.
run *ARGS:
    cargo run -p loractl-cli -- {{ARGS}}

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

# Lint the opt-in wgpu GPU-backend path (M7). Compiles the cubecl/wgpu/naga
# subtree + the gated wgpu smoke test; no GPU is needed to COMPILE and nothing
# runs. Kept out of the default `just lint` so that stays offline and fast.
lint-wgpu:
    cargo clippy -p loractl-core --all-targets --features wgpu -- -D warnings

# NOTE: cuda/tch are intentionally NOT local recipes — burn-cuda needs the CUDA
# toolkit/nvcc and burn-tch a linked libtorch, neither present on this Mac. They
# are build-verifiable only on a Linux+NVIDIA / libtorch host.

# Format the workspace.
fmt:
    cargo fmt --all

# Check formatting without writing (CI parity).
fmt-check:
    cargo fmt --all -- --check

# Run tests (offline, fast — mnist not enabled).
test:
    cargo test

# RustSec advisory scan of Cargo.lock (CI parity with
# .github/workflows/security-audit.yml). Accepted advisories are documented in
# .cargo/audit.toml; needs cargo-audit (`cargo install cargo-audit`).
audit:
    cargo audit

# Run the (network + heavy) MNIST LoRA convergence proof — not part of `just test`.
test-mnist:
    cargo test -p loractl-core --features mnist -- --ignored mnist_lora_converges

# Run the opt-in real-GPT-2 forward-parity test (needs `just gpt2-reference` first).
test-gpt2-real:
    cargo test -p loractl-core --features gpt2-real -- --ignored real_gpt2_forward_matches_pytorch_golden

# Run the wgpu GPU portability smoke (M7) on a real GPU — Metal on Apple Silicon.
# The ONLY way the double-gated `#[ignore]`d smoke runs; never fires in CI.
test-wgpu:
    cargo test -p loractl-core --features wgpu -- --ignored wgpu_training_smoke

# End-to-end acceptance #1: train on the GPU through the real CLI, backend
# selected purely from config (`compute.backend: wgpu`). Metal on this Mac.
run-wgpu config="config/examples/lora-wgpu.yaml":
    cargo run -p loractl-cli --features wgpu -- train {{config}}

# Regenerate the PyTorch golden fixture for the numerics test (needs torch via uv).
reference:
    uv run reference/lora_reference.py > crates/loractl-core/tests/golden/lora_toy.json

# Regenerate the kohya-ss export golden fixture (numpy only, no torch/network).
export-reference:
    uv run reference/lora_export_reference.py > crates/loractl-core/tests/golden/lora_export.json

# Regenerate the flow-matching golden fixture for the M8 numerics test (needs torch via uv).
flow-reference:
    uv run reference/flow_reference.py > crates/loractl-core/tests/golden/flow_toy.json

# Regenerate the checked-in tiny-GPT-2 parity fixture (weights + golden; torch via uv).
gpt2-tiny-reference:
    uv run reference/gpt2_tiny_reference.py --out crates/loractl-core/tests/fixtures

# Download real gpt2 + generate its (uncommitted) golden for the opt-in parity test.
gpt2-reference:
    uv run reference/gpt2_reference.py --out crates/loractl-core/tests/fixtures
