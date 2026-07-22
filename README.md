<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brand/header-dark.png">
  <img alt="loractl" src="docs/brand/header-light.png" width="440">
</picture>

**A terminal-native LoRA trainer, in Rust.**

[![CI](https://github.com/laurigates/loractl/actions/workflows/ci.yml/badge.svg)](https://github.com/laurigates/loractl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.92%2B-orange.svg)](Cargo.toml)

</div>

Most LoRA trainers bolt a half-baked web GUI onto a Python training core.
`loractl` inverts that: the **CLI is the primary surface** — config-driven,
completion-friendly, pipe-able — and a GUI, if anyone wants one, is just
another renderer layered on the same core over an API. The name says the
thesis: a `*ctl` tool, like `kubectl` or `systemctl`.

> **Status.** An early-stage learning project. The text-domain harness
> (M1–M5) and the Krea 2 image-diffusion stack (M6–M15) have landed; the last
> open box is the real-run interop proof — training a LoRA on `krea/Krea-2-Raw`
> and confirming it conditions generation in ComfyUI. Full milestone history:
> **[docs/roadmap.md](docs/roadmap.md)**.

## Why

- **The pipeline is the product.** No GUI plumbing to distract from
  dataloading, bucketing, the LoRA module, and the training loop.
- **CLI-first UX.** `clap`-generated shell completions, YAML configs with
  env/flag overrides, structured progress output.
- **GUI-optional by construction.** Core emits events; it never draws. The CLI
  renders them as a progress bar, and the HTTP API streams the same events as
  JSON — a GUI is just one more renderer.

## Architecture

Three crates, one direction of dependency (`cli → core`, `api → core`):

| Crate | Role |
|---|---|
| `loractl-core` | The pipeline: config schema, `TrainEvent` stream, `Trainer` trait, the LoRA modules + trainers, model loaders, generic adapter injection + kohya-ss export. **No CLI, no stdout.** |
| `loractl-cli` | The `loractl` binary. Parses args, layers config, renders events. |
| `loractl-api` | The HTTP/SSE server: streams the same events as JSON for an optional GUI. |

The load-bearing rule: **`loractl-core` never imports `clap` and never
prints.** A trainer reports progress by emitting `TrainEvent`s through a
callback; the caller decides how to surface them. That single discipline is
what makes "someone can build a GUI later" true instead of aspirational.

## Quickstart

```sh
# Build
cargo build

# Scaffold a starter config from a template (presets: synthetic, wgpu, flow, krea2)
cargo run -p loractl-cli -- init --preset krea2 -o config/my-lora.yaml

# Train the default synthetic LoRA-MLP demo from the example config
cargo run -p loractl-cli -- train config/examples/lora.yaml

# Override config fields from the CLI...
cargo run -p loractl-cli -- train config/examples/lora.yaml --lr 5e-5 --steps 2000

# ...or from the environment
LORACTL_OPTIM__LR=5e-5 cargo run -p loractl-cli -- train config/examples/lora.yaml

# Generate shell completions
cargo run -p loractl-cli -- completions zsh > ~/.zfunc/_loractl
```

Recipes live in the `justfile` (`just` to list): `just build`, `just init`,
`just train`, `just completions fish`, `just lint`, `just fmt`, `just test`.

### Install

The workspace root is a virtual manifest, so `cargo install` must point at the
CLI crate. Default features are **empty** (CPU/ndarray only — this keeps `just
test` and CI offline and GPU-free), so pick the backend feature for your
hardware:

| Host | Features | Command |
|---|---|---|
| Any (CPU only) | — | `cargo install --path crates/loractl-cli` |
| macOS / Apple Silicon | `wgpu` (Metal) | `cargo install --path crates/loractl-cli --features wgpu` |
| Linux + NVIDIA, CUDA toolkit (`nvcc`) | `cuda,wgpu` | `cargo install --path crates/loractl-cli --features cuda,wgpu` |
| Linux without the CUDA toolkit | `wgpu` (Vulkan) | `cargo install --path crates/loractl-cli --features wgpu` |

`just install` runs this detection for you and prints what it picked; override
with `just install <features>` or `just install cpu`. On a Linux/NVIDIA host
that lacks the CUDA toolkit, `just install-cuda` installs it from NVIDIA's
official apt repo (toolkit only, never the driver).

A compiled-in feature only makes a backend *available* — the backend a run
actually uses is selected at runtime by `compute.backend` (see [Compute
backend](#compute-backend)), and selecting one the binary wasn't built with
fails loudly rather than falling back to CPU. The HTTP/SSE server is a
separate, CPU-only binary: `cargo install --path crates/loractl-api`.

## Usage

### Training & the correctness harness

- **Default trainer.** `loractl train` runs the real `BurnTrainer` on a seeded
  synthetic classification set — no network, no dataset needed (it warns that
  this is the synthetic demo). Point `model.base` at a `krea/Krea-2-Raw`-layout
  directory instead and core's `select_trainer` routes to the
  `DiffusionTrainer`.
- **Checkpoints** and the final adapter are written as real, interoperable
  **`.safetensors`** files — only the trainable LoRA tensors, never the frozen
  base — with a JSON sidecar carrying the seed/shape to reconstruct the base.
- **Numerics proof.** `just test` runs an always-on, offline test that pins the
  LoRA toy's trained factors and per-step losses against a checked-in PyTorch
  golden (`1e-5` tolerance; frozen base bit-exact), plus a black-box
  convergence test. `just test-mnist` adds an opt-in real-MNIST accuracy proof.

### Sampling & adapter I/O

`loractl sample` runs a real, reproducible forward pass through a saved
adapter. Because `LoraMlp` is a synthetic classifier with no tokenizer, a
prompt is hashed into a seed that deterministically derives the input — an
honest, reproducible effect, distinct from text generation (the CLI prints this
framing). Setting `output.sample_every: N` writes periodic validation samples
during training. Design and trade-offs:
[ADR-0002](docs/adrs/0002-adapter-format-and-sample-semantics.md).

```sh
cargo run -p loractl-cli -- sample output/my-lora.safetensors --prompt "a test prompt"
```

### HTTP API

`just serve` runs `loractl-api` (bind via `LORACTL_API_ADDR`, default
`127.0.0.1:3000`) — the same event pipeline as the CLI, rendered as JSON over
SSE:

- `POST /runs` — start a run from a JSON `TrainConfig`; returns
  `201 {"id":1,"events_url":"/runs/1/events"}`.
- `GET /runs/{id}/events` — SSE stream: full replay from event 0, then live
  tail, ending with exactly one terminal event (`finished`/`failed`).

The API is **unauthenticated by default**; the localhost bind is what makes
that safe and it is enforced — a non-loopback bind refuses to start unless
`LORACTL_API_TOKEN` is set. Output paths are confined under
`LORACTL_OUTPUT_BASE`, with `LORACTL_MAX_CONCURRENT_RUNS` and
`LORACTL_RUN_RETENTION` bounding concurrency and memory. The full wire contract
lives in [docs/api/events.md](docs/api/events.md); the design decisions in
[ADR-0003](docs/adrs/0003-http-api-event-streaming.md).

### Real base models

loractl's first real base model was the **GPT-2 family**
(`openai-community/gpt2`): a hand-built, pre-LayerNorm GPT-2 loads unmodified HF
safetensors via `burn-store` and re-expresses the forward pass, checked against
PyTorch for parity stage by stage (always-run tiny fixture + opt-in real
`gpt2`). See [ADR-0001](docs/adrs/0001-first-real-target-model.md). The current
target is **Krea 2**, an open-weights ~12B rectified-flow image model — its
VAE, text encoder, MMDiT denoiser, dataset pipeline, and end-to-end
`DiffusionTrainer` are all in place. See
[ADR-0004](docs/adrs/0004-krea2-image-diffusion-target.md) and the
[roadmap](docs/roadmap.md).

## Config

A run is fully described by a YAML config (see `config/examples/lora.yaml`).
Precedence, lowest to highest: **YAML file → `LORACTL_`-prefixed env vars →
CLI flags.** Nested keys use `__` in env vars (`LORACTL_OUTPUT__DIR=/tmp/out`).

### Compute backend

An optional `compute:` block selects the backend and device at run time:

```yaml
compute:
  backend: ndarray          # ndarray (default, CPU) | wgpu (GPU) | cuda | tch
  device: 0                 # GPU ordinal; ignored by ndarray
  precision: f32            # f32 (default) | f16 (wgpu only — halves weight memory)
  grad_checkpointing: false # recompute activations during backward (numerically identical)
  quant: none               # none (default) | int8 | int4 (frozen-base quant; ndarray/cuda + f32 only)
```

- **`ndarray`** is the default and always available — no build feature, so
  `just test` and CI stay offline and GPU-free.
- **`wgpu`** is the GPU backend (Metal on macOS, Vulkan/DX12 elsewhere), opt-in
  behind a build feature and the one GPU path verified on the dev machine
  (`just test-wgpu`).
- **`cuda`** (needs the CUDA toolkit at build time) is wired into both
  trainers, **f32-only** — burn's non-f32 autodiff produces exactly-zero
  adapter gradients on cuda
  ([burn#5162](https://github.com/tracel-ai/burn/issues/5162)). cuda f32 is the
  one GPU configuration with clean validated numerics. `tch` (libtorch) remains
  compile-gated.

Selecting a GPU backend in a binary built **without** its feature fails loudly
(never a silent CPU fallback). The GPU backend is a **portability** target (the
loop runs, loss decreases), not a bit-exact numerics one — the numerics-golden
tests stay on ndarray, since GPU float-reduction order differs.

### Memory knobs — precision, checkpointing, quantization

Fitting the ~12.8B Krea 2 base on a single GPU is a memory problem, addressed
by three orthogonal knobs above: `precision: f16` (wgpu, ~24.6 GB on a 48 GiB
host), `grad_checkpointing`, and frozen-base `quant: int8`/`int4` (the QLoRA
pattern — quantized frozen base, f32 adapters; `ndarray`/`cuda` + f32 only).
The measured reality on a 24 GB card is that the training step is **VRAM-bound**
at the full LoRA target set; int4 + block-level gradient checkpointing is the
route in progress. The full analysis is
[ADR-0005](docs/adrs/0005-int4-training-vram-bound.md); the precision-accuracy
trade-off is [ADR-0006](docs/adrs/0006-reduced-precision-accuracy-gate.md).

## Observability (GlitchTip / Sentry)

`loractl` reports errors and panics to a [GlitchTip](https://glitchtip.com) /
Sentry-protocol instance. Telemetry is **opt-in via one env var** and a
complete no-op when unset:

```sh
export SENTRY_DSN='http://<key>@<host>/<project-id>'
loractl train config/examples/lora.yaml
```

Panics and fatal command errors become issues; `tracing::error!` events become
issues and `warn!`/`info!` become breadcrumbs. Delivery is independent of
`RUST_LOG` (which only controls console output).

## Roadmap

Milestones are tracked as GitHub issues; the detailed history lives in
**[docs/roadmap.md](docs/roadmap.md)**.

- [x] **M1–M5 — Text-domain harness.** Skeleton + config layering + events
      (M1); burn `BurnTrainer` with a PyTorch numerics golden (M2, [#1](https://github.com/laurigates/loractl/issues/1));
      real GPT-2 loader with forward-pass parity (M3, [#2](https://github.com/laurigates/loractl/issues/2)); safetensors
      adapter I/O + sampling (M4, [#3](https://github.com/laurigates/loractl/issues/3)); HTTP/SSE API crate (M5, [#4](https://github.com/laurigates/loractl/issues/4)).
- [x] **M6–M13 — Krea 2 building blocks.** Generic LoRA injection + kohya-ss
      export (M6, [#17](https://github.com/laurigates/loractl/issues/17)); GPU compute backend (M7, [#18](https://github.com/laurigates/loractl/issues/18)); rectified-flow
      objective (M8, [#19](https://github.com/laurigates/loractl/issues/19)); Qwen-Image VAE (M9, [#20](https://github.com/laurigates/loractl/issues/20)); Qwen3-VL text encoder
      (M10, [#21](https://github.com/laurigates/loractl/issues/21)); MMDiT denoiser (M11, [#22](https://github.com/laurigates/loractl/issues/22)); image dataset pipeline (M12,
      [#23](https://github.com/laurigates/loractl/issues/23)); single-GPU 12B memory knobs (M13, [#24](https://github.com/laurigates/loractl/issues/24)).
- [ ] **M14 — End-to-end + interop** ([#25](https://github.com/laurigates/loractl/issues/25)). *Code landed; the real-run
      interop proof is the remaining checkbox.* `DiffusionTrainer` composes the
      whole stack and trains end to end offline on a tiny Krea 2. Remaining:
      train on `krea/Krea-2-Raw` and prove the export conditions generation in
      ComfyUI. The 24 GB training route (cuda + int4 + block checkpointing) is
      VRAM-bound per [ADR-0005](docs/adrs/0005-int4-training-vram-bound.md).
- [x] **M15 — Train on Krea-2-Turbo** ([#82](https://github.com/laurigates/loractl/issues/82)). `variant: krea2-turbo` +
      scaled-fp8 checkpoint loading; resolution-based timestep shift
      ([#84](https://github.com/laurigates/loractl/issues/84)).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Conventional commits are required
(release-please drives versioning); the local gate mirrors CI — run
`just fmt-check && just lint && just test` before opening a PR.

## License

MIT © Lauri Gates
