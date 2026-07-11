# loractl

A terminal-native LoRA trainer, in Rust.

Most LoRA trainers bolt a half-baked web GUI onto a Python training core.
`loractl` inverts that: the **CLI is the primary surface** — config-driven,
completion-friendly, pipe-able — and a GUI, if anyone wants one, is just
another renderer layered on the same core over an API. The name says the
thesis: a `*ctl` tool, like `kubectl` or `systemctl`.

> **Status: rectified-flow objective (milestone 8).** The latest milestones
> turn toward image diffusion. M8 adds the **rectified-flow** (flow-matching)
> training objective — v-parameterization (`v = ε − x₀`) with logit-normal +
> shifted timestep sampling, pinned against a PyTorch golden
> ([#19](https://github.com/laurigates/loractl/issues/19)); M7 makes the
> training loop generic over a **config-selectable GPU compute backend**
> (`wgpu`/Metal, with compile-gated `cuda`/`tch`)
> ([#18](https://github.com/laurigates/loractl/issues/18)); M6 generalizes LoRA
> from wrapping one `Linear` to a name-keyed adapter set injected across a
> module tree, plus a **kohya-ss `.safetensors` export** that loads in
> ComfyUI/Krea ([#17](https://github.com/laurigates/loractl/issues/17)). Earlier
> milestones built the text-domain harness: an HTTP/SSE API (M5), portable
> **`.safetensors`** adapter I/O and reproducible sampling (M4), and a real
> GPT-2 loader with forward-pass parity vs PyTorch (M3), all on the M2
> `BurnTrainer` pinned against a numerics golden (the dependency-free
> `MockTrainer` remains for pipeline testing). Next up is **M9+** — the
> greenfield burn diffusion stack for **Krea 2**
> ([ADR-0004](docs/adrs/0004-krea2-image-diffusion-target.md)). See
> [Roadmap](#roadmap).

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

# Train the default synthetic LoRA-MLP demo from the example config
cargo run -p loractl-cli -- train config/examples/lora.yaml

# Override config fields from the CLI
cargo run -p loractl-cli -- train config/examples/lora.yaml --lr 5e-5 --steps 2000

# ...or from the environment
LORACTL_OPTIM__LR=5e-5 cargo run -p loractl-cli -- train config/examples/lora.yaml

# Generate shell completions
cargo run -p loractl-cli -- completions zsh > ~/.zfunc/_loractl
```

Recipes live in the `justfile` (`just` to list): `just build`, `just train`,
`just completions fish`, `just lint`, `just fmt`, `just test`.

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
  backend: ndarray # ndarray (default, CPU) | wgpu (GPU) | cuda | tch
  device: 0        # GPU ordinal; ignored by ndarray. wgpu: 0 = default/best GPU
```

- **`ndarray`** is the default and is **always** available — it needs no extra
  build feature, so `just test` and CI stay offline and GPU-free. Omitting the
  `compute:` block runs on ndarray, exactly as before.
- **`wgpu`** is the GPU backend: **Metal** on macOS/Apple Silicon, Vulkan/DX12
  elsewhere. It is opt-in behind a build feature —
  `cargo run -p loractl-cli --features wgpu -- train …` (or `just run-wgpu`) —
  and is the one GPU path runnable and verified on the dev machine
  (`just test-wgpu`).
- **`cuda`** (NVIDIA; needs the CUDA toolkit) and **`tch`** (libtorch) are
  compile-gated behind their own features and are **not runnable on macOS** —
  build-verifiable only on the appropriate host.
- Selecting a GPU backend in a binary built **without** its feature fails
  loudly (never a silent CPU fallback). Layer it like any other field:
  `LORACTL_COMPUTE__BACKEND=wgpu` / `LORACTL_COMPUTE__DEVICE=0`, or the
  `--backend wgpu --device 0` flags.

The GPU backend is a **portability** target (the loop runs, loss decreases),
not a bit-exact numerics one — per ADR-0001 the numerics-golden parity tests
stay on ndarray, since GPU float-reduction order differs.

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

### Next direction — Krea 2 image-diffusion LoRA (M9–M14)

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
- [ ] **M9 — Krea 2 latent VAE** ([#20](https://github.com/laurigates/loractl/issues/20))**.** Hybrid Qwen-Image/FLUX-2 AE in burn; image→latent parity (no Rust prior art).
- [ ] **M10 — Qwen 3 VL text encoder** ([#21](https://github.com/laurigates/loractl/issues/21))**.** Caption conditioning in burn — the largest single gap, no Rust prior art.
- [ ] **M11 — Krea 2 MMDiT denoiser** ([#22](https://github.com/laurigates/loractl/issues/22))**.** ~12B DiT (3D axial RoPE, GQA, gated-sigmoid attn, QK-Norm, RMSNorm, SwiGLU); forward parity + LoRA attach — ADR-0001 methodology at 100× scale.
- [ ] **M12 — Image dataset pipeline** ([#23](https://github.com/laurigates/loractl/issues/23))**.** Aspect-ratio bucketing + latent/embedding caching (the shape `DatasetConfig` already models).
- [ ] **M13 — Single-GPU 12B fit** ([#24](https://github.com/laurigates/loractl/issues/24))**.** bf16, gradient checkpointing, 8-bit Adam, QLoRA/NF4.
- [ ] **M14 — End-to-end + interop** ([#25](https://github.com/laurigates/loractl/issues/25))**.** A `DiffusionTrainer` trains a real Krea 2 LoRA; the output loads and applies in ComfyUI / Krea-2-Turbo.

A smaller optional detour on the *text* side is **SmolLM2-135M** — a modern
LLaMA-style architecture (RoPE + RMSNorm + SwiGLU) that reuses M3's loader and
parity harness and would bank the RoPE-convention work (burn's RoPE is
*interleaved* vs HF's *half-split*, see
[ADR-0001](docs/adrs/0001-first-real-target-model.md)) ahead of M11's 3D axial
RoPE — but it is not on the critical path to Krea 2.

## License

MIT © Lauri Gates
