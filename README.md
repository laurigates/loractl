# loractl

[![CI](https://github.com/laurigates/loractl/actions/workflows/ci.yml/badge.svg)](https://github.com/laurigates/loractl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.92%2B-orange.svg)](Cargo.toml)

A terminal-native LoRA trainer, in Rust.

Most LoRA trainers bolt a half-baked web GUI onto a Python training core.
`loractl` inverts that: the **CLI is the primary surface** — config-driven,
completion-friendly, pipe-able — and a GUI, if anyone wants one, is just
another renderer layered on the same core over an API. The name says the
thesis: a `*ctl` tool, like `kubectl` or `systemctl`.

> **Status: the image dataset pipeline (milestone 12).** The Krea 2
> image-diffusion stack
> ([ADR-0004](docs/adrs/0004-krea2-image-diffusion-target.md)) is under way.
> M12 lands the **kohya-style dataset pipeline**: folder-of-images +
> same-named `.txt` captions, **aspect-ratio bucketing** (16-px-aligned, the
> Krea 2 patch-grid constraint), and **one-time latent/conditioning caching**
> to disk — after the first pass, epochs never touch the image decoder, the
> VAE, or the text encoder again
> ([#23](https://github.com/laurigates/loractl/issues/23)). It feeds M11's
> **~12B single-stream MMDiT** (zero-centered RMSNorm, gated-sigmoid GQA
> attention, 3-axis rotation-matrix RoPE, shared 6-way modulation, and the
> text-fusion transformer that collapses the conditioner's 12-layer
> hidden-state stack) with forward parity vs the official `krea-ai/krea-2`
> implementation and the M6 **LoRA attach across its trunk projections**
> ([#22](https://github.com/laurigates/loractl/issues/22)), which consumes
> M10's **Qwen3-VL caption conditioner** — a frozen, text-only trunk
> emitting the 12-layer hidden-state stack, plus the chat-template tokenizer
> ([#21](https://github.com/laurigates/loractl/issues/21)) — and M9's
> **Qwen-Image latent VAE** with encode/decode parity vs diffusers
> ([#20](https://github.com/laurigates/loractl/issues/20)); all built on
> M8's **rectified-flow** objective
> ([#19](https://github.com/laurigates/loractl/issues/19)), M7's
> **config-selectable GPU compute backend** (`wgpu`/Metal, compile-gated
> `cuda`/`tch`) ([#18](https://github.com/laurigates/loractl/issues/18)), and
> M6's name-keyed LoRA injection with a **kohya-ss `.safetensors` export**
> that loads in ComfyUI/Krea
> ([#17](https://github.com/laurigates/loractl/issues/17)). Earlier
> milestones built the text-domain harness: an HTTP/SSE API (M5), portable
> **`.safetensors`** adapter I/O and reproducible sampling (M4), and a real
> GPT-2 loader with forward-pass parity vs PyTorch (M3), all on the M2
> `BurnTrainer` pinned against a numerics golden (the dependency-free
> `MockTrainer` remains for pipeline testing). M13 adds the **memory knobs**
> (`compute.precision: f16` on wgpu, `compute.grad_checkpointing`) that fit
> the ~12B base on a 48 GiB host, and M14 lands the **`DiffusionTrainer`**
> composing it all — proven end to end offline on a composed tiny Krea 2;
> the real-run ComfyUI interop proof is the roadmap's last open box.
> See [Roadmap](#roadmap).

## Why

- **The pipeline is the product.** No GUI plumbing to distract from
  dataloading, bucketing, the LoRA module, and the training loop.
- **CLI-first UX.** `clap`-generated shell completions, YAML configs with
  env/flag overrides, structured progress output.
- **GUI-optional by construction.** Core emits events; it never draws. A CLI
  renders them as a progress bar today; an HTTP API could stream the same
  events as JSON tomorrow.

## Architecture

Three layers, one direction of dependency:

| Crate | Role | Depends on |
|---|---|---|
| `loractl-core` | The pipeline: config schema, `TrainEvent` stream, `Trainer` trait, the LoRA module + `BurnTrainer`, the GPT-2 model + safetensors loader, generic name-keyed adapter injection + kohya-ss export. **No CLI, no stdout.** | burn, burn-store, regex, safetensors |
| `loractl-cli` | The `loractl` binary. Parses args, layers config, renders events. | `loractl-core` |
| `loractl-api` | The HTTP/SSE server: streams the same events as JSON for an optional GUI. | `loractl-core` |

The load-bearing rule: **`loractl-core` never imports `clap` and never
prints.** A trainer reports progress by emitting [`TrainEvent`]s through a
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

# Override config fields from the CLI
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
CLI crate. Default features are **empty** (CPU/ndarray only — this is what
keeps `just test` and CI offline and GPU-free), so pick the backend feature
for your hardware:

| Host | Features | Command |
|---|---|---|
| Any (CPU only) | — | `cargo install --path crates/loractl-cli` |
| macOS / Apple Silicon | `wgpu` (Metal) | `cargo install --path crates/loractl-cli --features wgpu` |
| Linux + NVIDIA, CUDA toolkit (`nvcc`) installed | `cuda,wgpu` | `cargo install --path crates/loractl-cli --features cuda,wgpu` |
| Linux without the CUDA toolkit | `wgpu` (Vulkan) | `cargo install --path crates/loractl-cli --features wgpu` |

`just install` runs this detection for you and prints what it picked;
override with `just install <features>` or `just install cpu`.

On a Linux/NVIDIA host that lacks the CUDA toolkit, `just install-cuda`
(run on that host) installs it from NVIDIA's official apt repo — toolkit
only, never the driver — with the version auto-matched to the installed
driver's ceiling (`just install-cuda 12.9` overrides).

A compiled-in feature only makes that backend *available* — the backend a run
actually uses is selected at runtime by `compute.backend` (see
[Compute backend (M7)](#compute-backend-m7)), and selecting a backend the
binary wasn't built with fails loudly rather than falling back to CPU.
`cuda` requires the CUDA toolkit at **build** time; `tch` (libtorch) also
exists but needs a linked libtorch. The HTTP/SSE server is a separate,
CPU-only binary: `cargo install --path crates/loractl-api`.

### Trainer, checkpoints, and the correctness harness

- **Default trainer.** `loractl train` runs the real `BurnTrainer` on a seeded
  synthetic classification set — no network, no dataset needed. It emits an
  honest warning that this is the M2 synthetic demo.
- **Checkpoint format.** Checkpoints and the final adapter are written as real,
  interoperable **`.safetensors`** files — only the two trainable LoRA tensors
  (`fc2.lora_a.weight`, `fc2.lora_b.weight`), never the frozen base. A JSON
  sidecar (`<path>.json`) carries the seed/shape needed to reconstruct the
  frozen base deterministically at load time. See
  [Sampling & adapter I/O (M4)](#sampling--adapter-io-m4) below.
- **Numerics proof.** `cargo test` (a.k.a. `just test`) runs an always-on,
  offline test that reproduces a deterministic LoRA toy and asserts its trained
  factors and per-step losses against a checked-in PyTorch golden fixture
  (absolute tolerance `1e-5`; frozen base bit-exact), plus a black-box synthetic
  convergence test.
- **Real MNIST (opt-in).** `just test-mnist` builds the `mnist` feature and runs
  an `#[ignore]`d test that trains the same LoRA-MLP on real MNIST and scores
  test accuracy (observed ≈0.84). This pulls a networked dataset downloader, so
  it is **never** part of the default build or `just test`.
- **Regenerate the golden fixture** with `just reference` (needs `torch` via
  `uv`).
- **Lint split.** `just lint` lints the default (offline) build; `just
  lint-mnist` lints the `mnist` feature path (which compiles reqwest/tokio).

### Real base model — GPT-2 (M3)

loractl's first real base model is the **GPT-2 family** (`openai-community/gpt2`).
A hand-built, pre-LayerNorm GPT-2 (`crates/loractl-core/src/gpt2.rs`) loads
**unmodified** HF safetensors via [`burn-store`](https://docs.rs/burn-store) and
re-expresses the forward pass, so it can be checked against PyTorch for parity.
See [ADR-0001](docs/adrs/0001-first-real-target-model.md) for why GPT-2 first,
the no-transpose loading story, and the verification methodology.

- **Loading is transpose-free.** GPT-2's `Conv1D` weights are already burn's
  `Linear` `[d_input, d_output]` layout (and the embeddings are already burn's
  `Embedding` layout), so every projection and embedding loads verbatim with the
  default identity adapter. The *only* rename is LayerNorm `weight`/`bias` →
  burn's `gamma`/`beta`. The output head is **weight-tied** to the token
  embedding (`logits = h · wteᵀ`, computed in the forward — there is no
  `lm_head` tensor to load). This no-transpose story is GPT-2-specific; a modern
  `nn.Linear`-based target would need a transpose.
- **Always-run offline parity.** `cargo test` (`just test`) loads a **checked-in
  tiny real GPT-2** (a genuine `GPT2LMHeadModel` at minimal dims — ~81 KB of
  safetensors) and asserts the burn forward reproduces a checked-in PyTorch
  golden, **stage by stage** (embeddings → block 0 → final LayerNorm → logits) so
  a mismatch localizes to a stage. Observed logits max|Δ| ≈ `9e-8` (pure f32
  rounding), with a tolerance-free backstop (last-token top-1 exact + logits
  cosine > 0.99999).
- **LoRA on the loaded model.** The same test harness wraps the loaded
  `c_attn` projection with `LoraLinear::from_base`, confirms the zero-init
  adapter is a no-op, then runs one real training step (finite loss, gradient on
  the adapter, none on the frozen base).
- **Opt-in real `gpt2`.** `just test-gpt2-real` loads the **real pretrained
  gpt2** (124M) into the same module and asserts logit parity — the pretrained-
  weights bonus on top of the tiny proof. Its ~498 MB safetensors and golden are
  **not** checked in; generate them first with `just gpt2-reference` (downloads
  `openai-community/gpt2` via `transformers`). Observed: logits max|Δ| ≈ `4e-4`,
  top-1 exact, cosine 1.0.
- **Regenerate goldens.** `just gpt2-tiny-reference` rebuilds the checked-in tiny
  fixture; `just gpt2-reference` produces the (uncommitted) real-gpt2 fixture.
  Both need `torch`/`transformers` via `uv`.

### Sampling & adapter I/O (M4)

Adapters save to and load from real `.safetensors` files
(`crates/loractl-core/src/adapter.rs`), and `loractl sample` runs a real,
reproducible forward pass through a saved adapter
(`crates/loractl-core/src/sample.rs`). See
[ADR-0002](docs/adrs/0002-adapter-format-and-sample-semantics.md) for the full
design and its trade-offs.

- **Tensor-naming scheme.** Only the trainable LoRA factors are persisted —
  `fc2.lora_a.weight` and `fc2.lora_b.weight` — mirroring the *pattern* of
  community LoRA conventions (Hugging Face PEFT's `lora_A`/`lora_B`) without
  claiming literal interop. The frozen base (`fc1`, `fc2.base`) is never
  written to disk.
- **A JSON sidecar carries the reconstruction metadata.** `burn-store` 0.21's
  safetensors metadata API is write-only (no public read-back after opening a
  file for a load), so `<path>.json` — `{ "seed", "rank", "alpha", "d_in",
  "hidden", "out" }` — sits alongside the `.safetensors` file. `load_adapter`
  reseeds the device with that seed and reconstructs the model, which
  regenerates the frozen base bit-identically to the original run (proven by
  `tests/adapter_roundtrip.rs`) — so the file only needs to carry the two
  tensors that actually changed during training.
- **`loractl sample --prompt <text>` is a deterministic forward pass, not text
  generation.** `LoraMlp` is a synthetic classifier with no tokenizer (see
  [ADR-0001](docs/adrs/0001-first-real-target-model.md)'s scoping of a real
  language-model training loop as future work). A given prompt is hashed
  (FNV-1a) into a seed that deterministically derives the sample's synthetic
  input, so the same prompt always reproduces the same output — an honest,
  reproducible effect, clearly distinct from generation, and the CLI prints
  this framing on every invocation.
- **Periodic validation samples.** Setting `output.sample_every: N` in the YAML
  config writes `sample-{step}.json` every N steps during training and emits
  `TrainEvent::Sample`, using one **fixed** seed across the whole run so the
  same probe input's prediction/logits can be compared across steps. There is
  no dedicated `--sample-every` CLI flag (`train` only has `--lr`/`--steps`);
  override it via the config file or the `LORACTL_OUTPUT__SAMPLE_EVERY` env var.

```sh
cargo run -p loractl-cli -- sample output/my-lora.safetensors --prompt "a test prompt"
```

### HTTP API (M5)

`just serve` runs `loractl-api` (bind address via `LORACTL_API_ADDR`, default
`127.0.0.1:3000`) — the same event pipeline as the CLI, rendered as JSON over
SSE instead of a progress bar:

- `POST /runs` — start a training run from a JSON `TrainConfig` (same schema
  as the YAML file); returns `201 {"id":1,"events_url":"/runs/1/events"}`.
- `GET /runs/{id}/events` — SSE stream: full replay from event 0, then live
  tail, ending with exactly one terminal event (`finished` or `failed`).

The API is **unauthenticated by default** — the default localhost bind is what
makes that safe, and it is enforced: a non-loopback bind refuses to start
unless a token is configured. The guards:

| Env var | Default | Guard |
|---|---|---|
| `LORACTL_API_TOKEN` | unset (no auth) | When set, every request must carry `Authorization: Bearer <token>` (constant-time compare); otherwise `401`. Required to bind beyond loopback. |
| `LORACTL_OUTPUT_BASE` | `./runs` | A request's `output.dir`/`output.name` are confined under this base; absolute paths, `..`, and symlink escapes are a `400`. |
| `LORACTL_MAX_CONCURRENT_RUNS` | `4` | Simultaneous runs; `POST /runs` returns `429` while saturated. |
| `LORACTL_RUN_RETENTION` | `32` | Completed runs kept in memory; older ones are evicted and their events become a `404`. In-flight runs are never evicted. |

The full wire contract — event shapes (pinned byte-for-byte by a golden
test), SSE framing, lifecycle rules, and a copy-paste curl transcript — lives
in [docs/api/events.md](docs/api/events.md); the design decisions in
[ADR-0003](docs/adrs/0003-http-api-event-streaming.md).

## Config

A run is fully described by a YAML config (see `config/examples/lora.yaml`).
Precedence, lowest to highest: **YAML file → `LORACTL_`-prefixed env vars →
CLI flags.** Nested keys use `__` in env vars (`LORACTL_OUTPUT__DIR=/tmp/out`).

### Compute backend (M7)

An optional `compute:` block selects the backend and device at run time:

```yaml
compute:
  backend: ndarray         # ndarray (default, CPU) | wgpu (GPU) | cuda | tch
  device: 0                # GPU ordinal; ignored by ndarray. wgpu: 0 = default/best GPU
  precision: f32           # f32 (default) | f16 (wgpu only — M13; halves weight memory)
  grad_checkpointing: false # recompute activations during backward (M13; numerically identical)
  quant: none              # none (default) | int8 | int4 (frozen-base quant; ndarray/cuda + f32 only — #96/#119)
```

- **`ndarray`** is the default and is **always** available — it needs no extra
  build feature, so `just test` and CI stay offline and GPU-free. Omitting the
  `compute:` block runs on ndarray, exactly as before.
- **`wgpu`** is the GPU backend: **Metal** on macOS/Apple Silicon, Vulkan/DX12
  elsewhere. It is opt-in behind a build feature —
  `cargo run -p loractl-cli --features wgpu -- train …` (or `just run-wgpu`) —
  and is the one GPU path runnable and verified on the dev machine
  (`just test-wgpu`).
- **`cuda`** (NVIDIA; needs the CUDA toolkit at **build** time) is wired into
  **both trainers, f32-only** — non-f32 fails loudly because burn's non-f32
  autodiff produces exactly-zero adapter gradients on cuda
  ([burn#5162](https://github.com/tracel-ai/burn/issues/5162)). cuda f32 is
  the one GPU configuration with **clean validated numerics** (grad ratio
  1.00 vs the ndarray ground truth at every adapter site, verified on an
  RTX 4090). Not runnable on macOS; on a Linux+NVIDIA host `just test-cuda`
  runs the cuda smokes and `just run-cuda` trains through the CLI. **`tch`**
  (libtorch) remains compile-gated and unwired in the diffusion trainer.
- Selecting a GPU backend in a binary built **without** its feature fails
  loudly (never a silent CPU fallback). Layer it like any other field:
  `LORACTL_COMPUTE__BACKEND=wgpu` / `LORACTL_COMPUTE__DEVICE=0`, or the
  `--backend wgpu --device 0` flags.

The GPU backend is a **portability** target (the loop runs, loss decreases),
not a bit-exact numerics one — per ADR-0001 the numerics-golden parity tests
stay on ndarray, since GPU float-reduction order differs.

### Frozen-base quantization (int8/int4, #96/#119)

A third memory knob, orthogonal to `precision`, quantizes the diffusion
trainer's **frozen MMDiT base** to weight-only per-block symmetric int8 or
int4 (Q4S) while the LoRA adapters stay f32 — the **QLoRA** pattern:

```yaml
compute:
  backend: cuda            # or ndarray (offline/CI)
  precision: f32           # quantized weights dequantize to f32 — f16/bf16 are rejected
  quant: int4              # none (default) | int8 | int4
```

- **The point:** `precision: f16` only fits the ~12.8B Krea 2 base on a large
  (48 GB) Metal host, and burn's GPU autodiff is broken in f16 (burn#5162).
  On the **24 GB** card the measured reality
  ([ADR-0005](docs/adrs/0005-int4-training-vram-bound.md)) is: `int8` (~1/4 of
  f32) reclaims to a **~17.1 GB** resident base — it loads, but the training
  step OOMs; `int4` (Q4S per-block, ~1/8 of f32) cuts the reclaimed resident
  base to **~10.1 GB** (~14.9 GB before reclaim). Whether the *step* then fits
  is a separate question the base size alone doesn't settle: the full
  `blocks\.` target set is measured **not** to fit (step working set
  ≈ 25.5 GB vs 24 GB, resolution-independent), and reduced target sets are
  as-yet unmeasured — the `just step-probe` sweep (#126) is the measurement.
  So int4 + the numerically-clean **cuda f32** path is the quant choice for the
  24 GB training route (int8 fits load/inference only), gated on a small-enough
  trained-target set.
- **Where it applies:** the diffusion trainer's base only — the base weights are
  frozen int8 constants (a custom autodiff matmul dequantizes transiently per
  layer, so gradients flow to the adapters, never the base), and every
  checkpoint is still the same ComfyUI-loadable kohya-ss export. The synthetic
  `BurnTrainer` rejects the knob (no frozen base worth quantizing).
- **Where it's allowed:** `ndarray` (the offline/CI path) and `cuda` only, and
  `precision: f32` only. `wgpu` is untested (use `precision: f16` there),
  candle/tch have no quantized matmul, and any non-f32 precision is rejected —
  all fail loudly, never a silent fallback.
- **Loading is streamed:** the ~49 GB f32 checkpoint is never materialized — the
  loader quantizes one layer's weight at a time from an mmap'd file (bf16/f32 or
  auto-detected scaled-fp8), so peak load memory is the int8 skeleton plus one
  transient f32 tensor.
- **Validated offline** on the tiny-krea2 bundle (`cargo test`): the loader
  produces a correct trainable quantized model (finite losses, the 42-key kohya
  export, resume). **On-box memory on the 24 GB RTX 4090 is now measured**
  ([ADR-0005](docs/adrs/0005-int4-training-vram-bound.md)): the quantized base
  loads and fits (int8 **~17.1 GB** / int4 **~10.1 GB** reclaimed resident),
  but the full-target-set training step is **VRAM-bound** — working set
  ≈ 25.5 GB vs 24 GB, resolution-independent — so the remaining
  [#96](https://github.com/laurigates/loractl/issues/96) work is footprint
  reduction (fewer LoRA targets first).

## Observability (GlitchTip / Sentry)

`loractl` reports errors and panics to a [GlitchTip](https://glitchtip.com)
instance (GlitchTip speaks the Sentry ingest protocol, so the standard Rust
`sentry` SDK is the client). Telemetry is **opt-in via one env var** and a
complete no-op when it's unset — nothing GlitchTip-specific is baked into the
repo.

```sh
# Point at the local kind-fvh-dev GlitchTip project's DSN
export SENTRY_DSN='http://<key>@glitchtip.orb.local/<project-id>'
loractl train config/examples/lora.yaml
```

What gets sent:

| Signal | Becomes | Source |
|---|---|---|
| A panic | An issue | Sentry panic integration |
| A fatal command error (non-zero exit) | An issue | `capture_anyhow` in `main` |
| `tracing::error!` events | Issues | `sentry-tracing` layer |
| `tracing::warn!` / `info!` events | Breadcrumbs on the next issue | `sentry-tracing` layer |

Breadcrumb/issue delivery is independent of `RUST_LOG` — that variable only
controls what the console `fmt` layer prints.

### Reaching the kind-fvh-dev GlitchTip from the host

The cluster exposes GlitchTip at `http://glitchtip.orb.local` (envoy gateway).
If that name doesn't resolve on your machine (OrbStack's `*.orb.local` DNS
doesn't always pick up newer HTTPRoutes), point it at the gateway so the SDK
can deliver events:

```sh
echo "$(kubectl -n envoy-gateway-system get svc -l gateway.envoyproxy.io/owning-gateway-name=local-gateway -o jsonpath='{.items[0].status.loadBalancer.ingress[0].ip}')  glitchtip.orb.local" | sudo tee -a /etc/hosts
```

(As of writing the gateway IP is `192.168.97.4`; the command above reads it
live in case it changes.) Verify: `curl -sS -o /dev/null -w '%{http_code}\n' http://glitchtip.orb.local/`
should print a redirect/`200`.

## Roadmap

- [x] **M1 — Skeleton.** Workspace, CLI (`train`/`sample`/`completions`),
      config layering, event → progress-bar rendering, `MockTrainer`.
- [x] **M2 — Correctness harness** ([#1](https://github.com/laurigates/loractl/issues/1))**.** burn-backed `BurnTrainer` trains a LoRA
      `Module` (frozen base, trained A·B) on a tiny MLP; numerics pinned against
      a PyTorch reference (offline, always-run); real MNIST convergence + accuracy
      proven behind an opt-in `mnist` feature. The loop is verified in isolation
      before any large model.
- [x] **M3 — Real base model** ([#2](https://github.com/laurigates/loractl/issues/2))**.** Hand-built GPT-2 loads real HF safetensors into
      burn (transpose-free state-dict mapping via burn-store), forward-pass parity
      proven against PyTorch on a checked-in tiny GPT-2 (offline, always-run) and
      real `gpt2` (opt-in); LoRA attached to the loaded model runs a training
      step. See [ADR-0001](docs/adrs/0001-first-real-target-model.md).
- [x] **M4 — Sampling & adapter I/O** ([#3](https://github.com/laurigates/loractl/issues/3))**.** Adapters save to and load from
      real `.safetensors` files (adapter-only + a JSON sidecar), `loractl sample`
      runs a deterministic, prompt-seeded forward pass, and periodic validation
      samples are written and reported during training. See
      [ADR-0002](docs/adrs/0002-adapter-format-and-sample-semantics.md).
- [x] **M5 — API crate** ([#4](https://github.com/laurigates/loractl/issues/4))**.** `loractl-api` exposes the event stream over HTTP so a
      GUI can be built independently: `POST /runs` starts a training run,
      `GET /runs/{id}/events` streams its events as SSE (full replay from
      event 0, then live tail), with the wire shapes pinned byte-for-byte by
      a golden test. See
      [ADR-0003](docs/adrs/0003-http-api-event-streaming.md).

### Next direction — Krea 2 image-diffusion LoRA (M9–M15)

M1–M5 built a complete but **text-domain** harness, and M6–M8 landed the first
pieces that turn toward the image domain: generic LoRA injection with a
kohya-ss export (M6), a config-selectable GPU compute backend (M7), and the
rectified-flow objective (M8). The remaining goal is a different domain
entirely: training LoRA adapters for **Krea 2**, an open-weights
(`krea/Krea-2-Raw`) ~12B rectified-flow **image** model. This reuses loractl's
architecture (event stream, config, `burn-store` loading, the parity-golden
methodology) but almost none of its model code — the denoiser, VAE, and text
encoder are still greenfield in burn. The strategy (why Krea 2, why stay on
burn, the full gap analysis) is [ADR-0004](docs/adrs/0004-krea2-image-diffusion-target.md).

- [x] **M6 — Generic LoRA injection + kohya-ss export** ([#17](https://github.com/laurigates/loractl/issues/17))**.** `LoraAdapters` injects a name-keyed set of low-rank deltas across a module tree (config `targets` patterns → `build_adapters` over a model's `injectable_sites`); GPT-2's attach is re-expressed through it. `export_adapters` writes a kohya-ss `.safetensors` (transposed `lora_down`/`lora_up` + `.alpha` scalar) so a LoRA loads in ComfyUI/Krea, proven offline against a golden. A `PeftDiffusers` format is reserved behind the `AdapterNameMapper` seam.
- [x] **M7 — GPU compute backend** ([#18](https://github.com/laurigates/loractl/issues/18))**.** The training loop is generic over `B: AutodiffBackend`; `BurnTrainer` dispatches a config-selected backend (`compute.backend`) at run time — `ndarray` (CPU, always compiled, the offline/CI default), `wgpu` (GPU: Metal on Apple Silicon), and compile-gated `cuda`/`tch`. `just test` stays offline on ndarray; the GPU path is verified locally on Metal (`just test-wgpu`). See the [Config → Compute backend](#compute-backend-m7) section.
- [x] **M8 — Rectified-flow objective** ([#19](https://github.com/laurigates/loractl/issues/19))**.** Flow-matching v-prediction (`v = ε − x₀`, SD3 time convention: t=0 data, t=1 noise) with logit-normal + shifted timestep sampling (`crates/loractl-core/src/flow.rs`; kohya/SD3 `shift: 3.0` default, plus the FLUX resolution-dependent `exp(μ)` helper for M11). `task: flow-matching` trains a LoRA velocity net on a synthetic latent toy, pinned against a PyTorch golden (M2 methodology, `just flow-reference`); adapter sidecars record the task and `loractl sample` refuses velocity nets.
- [x] **M9 — Krea 2 latent VAE** ([#20](https://github.com/laurigates/loractl/issues/20))**.** Krea 2's autoencoder turned out to be the **stock Qwen-Image VAE** (`krea-ai/krea-2`'s `autoencoder.py` wraps diffusers' `AutoencoderKLQwenImage` + per-channel latent stats), so `QwenVae` ports that: an f8, 16-latent-channel *video* VAE run image-only (`T = 1`), causal 3-D convs, Qwen RMS-norms, and the mid-block single-head attention (the "attention-free" report claim was wrong — `attn_scales: []` only strips trunk attention). Weights load verbatim (PyTorch conv layout, `gamma` norms; one `resample.1` Sequential-index rename), proven by staged encode/decode parity vs diffusers on a checked-in tiny fixture (`just vae-reference`) and an opt-in real-weights proof (`just vae-real-reference && just test-vae-real`). `encode` emits the **normalized** latents training consumes and M12 caches.
- [x] **M10 — Qwen 3 VL text encoder** ([#21](https://github.com/laurigates/loractl/issues/21))**.** `Qwen3VlEncoder` ports the Qwen3-VL *text* trunk (GQA 32/8 heads, per-head **QK-RMSNorm before RoPE**, **half-split** RoPE at θ=5e6 — text-only M-RoPE collapses to plain RoPE, verified against `modeling_qwen3_vl.py` — SwiGLU, pre-norm residuals) and loads Krea-2-Raw's own `text_encoder/` **text-only**: a `^language_model\.` filter drops the vision tower, `PyTorchToBurnAdapter` transposes the `nn.Linear`s, and only the first 35 decoder layers load (`select_layers` max; the 36th layer + final norm are dead for conditioning). `Qwen3VlConditioner` adds `encoder.py`'s exact chat template + tokenizer (HF `tokenizers`, right-pad-then-suffix-concat) and emits the conditioning stack `[b, s, 12, 2560]` + mask the MMDiT (M11) consumes. Proven by staged parity vs transformers on a checked-in tiny fixture — whose golden includes a **right-padded row**, pinning key-padding masking — plus an opt-in real-weights + tokenizer-parity proof (`just qwen3vl-real-reference && just test-qwen3vl-real`).
- [x] **M11 — Krea 2 MMDiT denoiser** ([#22](https://github.com/laurigates/loractl/issues/22))**.** `Mmdit` ports `krea-ai/krea-2`'s `SingleStreamDiT` — a **single-stream** DiT (text + image tokens concatenated through 28 identical blocks, not FLUX's double-stream): **zero-centered RMSNorm** (`weight = scale + 1`, eps 1e-5, f32), **gated-sigmoid attention** (`wo(attn · σ(gate(x)))`), QK-norm, GQA 48/12, **rotation-matrix RoPE** over 3 position axes `[32, 48, 48]` at θ=1e3 (text at the origin, image on the patch grid), shared 6-way timestep modulation with per-block learned bias, the 2+2-block **text-fusion transformer** that collapses the M10 conditioner's 12-layer stack, and the reference's pad-to-256/masking/output-slice semantics. Proven by staged parity vs the official `mmdit.py` (fetched at a pinned commit by `just mmdit-reference`) on a checked-in tiny fixture, plus an opt-in **real-weights staged proof** at real widths, depth-truncated to fit a 48 GiB host (`just mmdit-real-reference && just test-mmdit-real`; full-depth runs arrive via M13's `precision: f16` knob). The M6 LoRA attaches across every trunk projection (`blocks.N.attn.{wq,wk,wv,wo}` + `mlp.{gate,up,down}`): zero-init adapters are a bit-identical no-op and one real step routes gradients to the adapters only.
- [x] **M12 — Image dataset pipeline** ([#23](https://github.com/laurigates/loractl/issues/23))**.** `dataset` implements the kohya/ai-toolkit convention `DatasetConfig` was scaffolded for: scan a folder of images + same-named `.txt` captions (missing caption = unconditional example), group into **aspect-ratio buckets** (every dimension a multiple of 16 — Krea 2's `ae.compression · patch` grid), resize cover-style + center-crop, and cache **VAE latents + conditioning stacks** as safetensors under `<dataset>/.loractl-cache/`, keyed by image file name (latents) / stem (conditioning), bucket shape, and a hashed encoder fingerprint. Encoders are injected as closures — M14 wires the real frozen `QwenVae`/`Qwen3VlConditioner`, the offline tests wire mocks — and the cache-reuse test passes encoders that *panic*, proving warm epochs are pure tensor reads. Per-bucket batching never mixes shapes.
- [x] **M13 — Single-GPU 12B fit** ([#24](https://github.com/laurigates/loractl/issues/24))**.** Two config-toggleable memory knobs, both overridable per layer (YAML → env → flag): **`compute.precision: f16`** (wgpu only; any other backend fails loudly — the M7 no-silent-fallback rule) halves resident weight memory, the knob that fits the ~12B Krea 2 base (~49 GB f32 → **~24.6 GB f16**) on this 48 GiB host; **`compute.grad_checkpointing: true`** swaps burn's `Autodiff` to `BalancedCheckpointing` (recompute activations during backward) — proven **bit-identical** to stored activations on the synthetic task, since recomputation replays the same deterministic ops. The wgpu smoke gains an f16 + checkpointing variant (`just test-wgpu`, Metal). Deliberately *not* built: 8-bit Adam — LoRA optimizer state lives only on the adapters (tens of MB at rank 16), not the multi-GB full-finetune case it exists for; and NF4/int8 base quantization — f16 already fits this host, so packed-int8 is tracked on [#24](https://github.com/laurigates/loractl/issues/24) as the follow-up that would unlock ≤16 GB GPUs.
- [ ] **M14 — End-to-end + interop** ([#25](https://github.com/laurigates/loractl/issues/25))**.** *Code landed; the real-run interop proof is the remaining checkbox.* `DiffusionTrainer` composes the whole stack as one `impl Trainer` behind a two-armed factory on `model.base` (the constructor seam, unchanged otherwise): the M12 pipeline caches M9 latents + M10 conditioning **then drops the encoders before the MMDiT loads** (peak memory never holds both), the M8 objective (`x_t = (1−t)x₀ + tε`, target `v = ε − x₀`, logit-normal+shift timesteps) drives the M11 denoiser through the M6 adapter injection, and every checkpoint + the final artifact is a **kohya-ss export**. The offline proof composes the per-milestone tiny fixtures into a dimension-matched **tiny Krea 2** (`just krea2-reference`) and trains it end to end through the real loading paths: events framed, `B` off zero, kohya key grammar pinned, and a reseeded warm-cache rerun **bit-identical**. Per-step loss is deliberately not asserted to decrease — fresh `(t, ε)` each step makes it noise-dominated by construction. Remaining for the checkbox: train on `krea/Krea-2-Raw` and prove the export conditions Krea-2-Turbo in ComfyUI. Route status: wgpu f16 + checkpointing (`config/examples/krea2-lora.yaml`) is blocked by burn#5162; cuda + int4 (`quant: int4`, [#119](https://github.com/laurigates/loractl/issues/119)) is **VRAM-bound** per [ADR-0005](docs/adrs/0005-int4-training-vram-bound.md) — working set ≈ 25.5 GB vs 24 GB, resolution-independent — with the footprint levers (fewer LoRA targets first) in progress.
- [x] **M15 — Train on Krea-2-Turbo** ([#82](https://github.com/laurigates/loractl/issues/82))**.** Turbo is architecturally identical to Raw — the same 430 tensor keys, per-tensor distillation deltas of 3–11% — so the M11 port, key remap, and M8 objective apply unchanged; what blocked turbo training was purely the load seam, and M15 opens it (amending [ADR-0004](docs/adrs/0004-krea2-image-diffusion-target.md)'s "train on Raw, apply to Turbo" decision). `variant: krea2-turbo` reuses the Krea 2 config and encoder-cache fingerprint but defaults the denoiser filename to `turbo.safetensors`, and an optional `model.checkpoint` overrides the filename for any variant. The widely-distributed ComfyUI-style **scaled-fp8** repacks (13.1 GB vs 26.3 GB bf16: `float8_e4m3fn` weights + f32 0-d `weight_scale` sidecars) now load: burn-store 0.21 has no `F8_E4M3` dtype arm, so `src/fp8.rs` lazily dequantizes `LUT[byte] · weight_scale` to f32 (exact 256-entry e4m3fn LUT; per-tensor and per-output-channel scales) and feeds the same remap → transpose → cast → apply pipeline — auto-detected from the safetensors header, so bf16/f32 checkpoints keep the proven burn-store path and the mmap streaming memory profile survives. Out-of-contract files fail loudly rather than half-load: the legacy ComfyUI `scaled_fp8` convention, unknown scale shapes, and unexpected leftover keys (e.g. the `fp8mixed` repack's baked-in `last.up`/`last.down` LoRA) are all hard errors. Follow-up tracked separately: a Turbo training adapter ([#83](https://github.com/laurigates/loractl/issues/83)). Dynamic timestep-shift parity ([#84](https://github.com/laurigates/loractl/issues/84)) landed as `flow.shift_mode: resolution`: per-batch `exp(μ(gh·gw))` with Krea 2's ai-toolkit-documented anchors (μ linear 0.5@256 → 1.15@6400 image tokens) as the `FlowConfig` defaults, golden-pinned against the PyTorch reference; the krea2 example configs train with it, matching how ai-toolkit trains raw and turbo alike.

A smaller optional detour on the *text* side is **SmolLM2-135M** — a modern
LLaMA-style architecture (RoPE + RMSNorm + SwiGLU) that reuses M3's loader and
parity harness and would bank the RoPE-convention work (burn's RoPE is
*interleaved* vs HF's *half-split*, see
[ADR-0001](docs/adrs/0001-first-real-target-model.md)) ahead of M11's 3D axial
RoPE — but it is not on the critical path to Krea 2.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Conventional commits are required
(release-please drives versioning); the local gate mirrors CI — run
`just fmt-check && just lint && just test` before opening a PR.

## License

MIT © Lauri Gates
