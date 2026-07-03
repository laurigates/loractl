default:
    @just --list

# Build the whole workspace.
build:
    cargo build

# Run the CLI with arbitrary args, e.g. `just run train config/examples/lora.yaml`.
run *ARGS:
    cargo run -p loractl-cli -- {{ARGS}}

# Train from a config with the current (mock) trainer.
train config="config/examples/lora.yaml":
    cargo run -p loractl-cli -- train {{config}}

# Print shell completions, e.g. `just completions fish`.
completions shell="zsh":
    cargo run -p loractl-cli -- completions {{shell}}

# Lint with clippy, warnings-as-errors.
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Format the workspace.
fmt:
    cargo fmt --all

# Check formatting without writing (CI parity).
fmt-check:
    cargo fmt --all -- --check

# Run tests.
test:
    cargo test
