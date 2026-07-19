# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`loractl` is a terminal-native LoRA trainer in Rust: a **CLI-first** tool where
a GUI, if ever built, is just another renderer over the same core (the name is
a deliberate `*ctl` reference, like `kubectl`). It is an early-stage learning
project — see the roadmap in `README.md` and the tracking issues (#1–#4, #17–#25).

**Current status:** milestones M1–M13 (#1–#4, #17–#24) have landed, plus
M14's (#25) `DiffusionTrainer` code (the real-run ComfyUI interop proof is
the milestone's remaining checkbox).
The default trainer is a real, burn-backed `BurnTrainer` that trains a
**synthetic** LoRA-MLP demo (offline, fast), pinned against a PyTorch numerics
golden; real MNIST is behind an opt-in `mnist` feature and the dependency-free
`MockTrainer` remains available for pipeline testing. M3 added a real GPT-2
safetensors loader with forward-pass parity vs PyTorch; M4 added portable
`.safetensors` adapter I/O and deterministic sampling; M5 added `loractl-api`,
which streams the same `TrainEvent`s over HTTP/SSE (wire contract in
`docs/api/events.md`). M6 (#17) generalized LoRA from wrapping one `Linear` to
injecting a name-keyed set of adapters (`LoraAdapters`) across a module tree
(config `targets` patterns → `build_adapters` over a model's `injectable_sites`;
GPT-2's attach re-expressed through it) and added a kohya-ss `.safetensors`
export (`export_adapters`, transposed `lora_down`/`lora_up` + `.alpha` scalar)
so a LoRA loads in ComfyUI/Krea — proven offline against a golden. M7 (#18)
made the training loop generic over `B: AutodiffBackend` with a runtime,
config-selected compute backend (`compute.backend`): ndarray (CPU, always
compiled, the offline/CI default), wgpu (Metal on Apple Silicon), and
compile-gated cuda/tch — selecting a backend the binary wasn't built with
fails loudly, never a silent CPU fallback. M8 (#19) added the rectified-flow
(flow-matching) objective: `task: flow-matching` trains a LoRA velocity net
(v-prediction `v = ε − x₀`, SD3 time convention) with logit-normal + shifted
timestep sampling (`src/flow.rs`) on a synthetic latent toy, pinned against a
PyTorch golden; adapter sidecars record the producing task and `loractl
sample` refuses flow adapters. M9 (#20) landed the first Krea 2 model
component: `QwenVae` (`src/qwen_vae.rs`) ports Krea 2's autoencoder — which is
the **stock Qwen-Image VAE** (diffusers `AutoencoderKLQwenImage`, f8,
16 latent channels, run image-only at `T = 1`) — with staged encode/decode
parity vs diffusers on a checked-in tiny fixture plus an opt-in real-weights
proof; `encode` emits the normalized latents diffusion training consumes.
M10 (#21) landed Krea 2's caption conditioner: `Qwen3VlEncoder`
(`src/qwen3vl.rs`), a frozen text-only Qwen3-VL trunk (GQA, per-head QK-norm
before half-split RoPE, SwiGLU) that loads Krea-2-Raw's own `text_encoder/`
with the vision tower filtered out and emits the 12-layer hidden-state stack
the MMDiT cross-attends to; `Qwen3VlConditioner` wraps the chat template +
tokenizer (captions → conditioning `[b, s, 12, 2560]` + mask). Staged parity
vs transformers (tiny fixture incl. a right-padded row) + opt-in real-weights
and tokenizer parity. M11 (#22) landed the core model: `Mmdit`
(`src/mmdit.rs`) ports the ~12B single-stream `SingleStreamDiT`
(zero-centered RMSNorm, gated-sigmoid GQA attention, rotation-matrix RoPE
over 3 position axes, shared 6-way modulation, the 2+2-block text-fusion
transformer, pad-to-256 semantics) with staged parity vs the official
`mmdit.py`, an opt-in depth-truncated real-weights proof (full depth in f32
exceeds this 48 GiB host; M13's f16 knob is the full-depth path), and the M6
LoRA attach across every trunk projection. M12 (#23) landed the dataset pipeline (`src/dataset.rs`):
kohya-style folder scanning, 16-px-aligned aspect-ratio buckets, cover-resize
+ center-crop image loading, and one-time latent/conditioning caching to
`<dataset>/.loractl-cache/` (encoders injected as closures; M14 wires the
real frozen models). M13 (#24) landed the memory knobs:
`compute.precision: f16` (wgpu only, fails loudly elsewhere — halves weight
memory, fitting the ~12B base in ~24.6 GB on this 48 GiB host) and
`compute.grad_checkpointing` (burn `BalancedCheckpointing`, proven
bit-identical to stored activations); 8-bit Adam is a documented skip (LoRA
optimizer state is adapter-only) and int8/NF4 is the tracked follow-up on
#24 for ≤16 GB GPUs. M14 (#25) landed `DiffusionTrainer`
(`src/diffusion_trainer.rs`): the whole stack as one `impl Trainer` behind
core's `select_trainer` factory on `model.base` ("synthetic"/"mnist" →
`BurnTrainer`, a Krea-2-Raw-layout dir → the diffusion trainer), proven
offline on the composed tiny-krea2 bundle (`just krea2-reference`,
`tests/diffusion_trainer.rs`); kohya-ss exports at every checkpoint.
M15 (#82) opened direct Krea-2-Turbo training (amending ADR-0004's
"train on Raw" decision — Turbo is architecturally identical, same 430 keys):
`variant: krea2-turbo` (default denoiser `turbo.safetensors`), an optional
`model.checkpoint` filename override, and auto-detected loading of
ComfyUI-style scaled-fp8 checkpoints (`float8_e4m3fn` + `weight_scale`) via
a lazy `LUT[byte] · scale` dequant source (`src/fp8.rs`; burn-store 0.21 has
no fp8 dtype) — legacy/malformed fp8 files fail loudly; follow-up: training
adapter (#83). Timestep-shift parity (#84) landed as `flow.shift_mode:
resolution` — per-batch `exp(μ(gh·gw))` with Krea 2's ai-toolkit-documented
anchors (0.5@256 → 1.15@6400 image tokens) as the `FlowConfig` defaults,
golden-pinned; the krea2 example configs use it. See
the roadmap in `README.md`.

**Next direction (M14's remaining checkbox, #25):** the real run — train a
LoRA on `krea/Krea-2-Raw` through the landed `DiffusionTrainer` and prove
the exported adapter loads and visibly conditions generation in ComfyUI /
Krea-2-Turbo. The cuda route was **VRAM-bound**: the #132 retention-ledger
attribution ([ADR-0005](docs/adrs/0005-int4-training-vram-bound.md)
Addendum 2, PR #133) measured the monolithic step's true logical demand at
**67.9 GiB pinned per forward** (~3× the RTX 4090) — burn-autodiff eagerly
pins the whole tracked trunk interior (attention-score trio 35.4 GiB,
SwiGLU outputs, quant-site outputs), topology-driven and independent of
resolution/site-count/reclaim/chunking (all measured dead). The measured
fix is **#134 — block-level gradient checkpointing**
(`src/block_ckpt.rs::checkpointed_step`): `compute.grad_checkpointing:
true` on the diffusion path now runs the trunk forward graph-free storing
only block inputs, then replays each block on its own standalone graph in
backward (grads bit-identical to the monolithic path on the tiny fixture;
incompatible with `lora.dropout > 0`; a nested-backward custom op is
impossible on burn 0.21 — verified deadlock). The quant knobs are
unchanged and correct: `compute.quant: int8` (#96) / `int4` (Q4S, #119)
load the frozen ~12.8B base per-block quantized while adapters train in
f32 (QLoRA); restricted to `(ndarray|cuda, f32)` by the trainer guard.
**int4 (~10.1 GB reclaimed resident base) + block checkpointing is the
24 GB training route** (estimate ≈ 16–18 GB). Verify fit with
`just step-probe` (#126) — the gate is a **zero-panic** run, never a
survived OOM storm (a ceiling-riding run silently corrupts the forward —
a negative MSE was observed). The wgpu f16 route
(`config/examples/krea2-lora.yaml`, the 48 GiB Metal host) stays blocked
by burn's GPU autodiff bug (burn#5162, unchanged). Strategy and gap
analysis: [ADR-0004](docs/adrs/0004-krea2-image-diffusion-target.md).

## Commands

Recipes live in the `justfile` (`just` to list). Cargo directly also works.

| Task | Command |
|---|---|
| Build the workspace | `just build` (`cargo build`) |
| Install the `loractl` binary | `just install` — GPU feature auto-detected per host (macOS → `wgpu`; Linux+nvcc → `cuda,wgpu`; else `wgpu`); override: `just install <features>` / `just install cpu` |
| Install the CUDA toolkit (on the GPU host) | `just install-cuda` — NVIDIA apt repo, toolkit-only (never the driver), version matched to the driver's ceiling; override: `just install-cuda <version>` |
| Run the CLI | `just run <args>` (`cargo run -p loractl-cli -- <args>`) |
| Scaffold a config from a template | `just init [preset]` → stdout (`synthetic`/`wgpu`/`flow`/`krea2`/`krea2-comfyui`); or `loractl init --preset <p> -o <path>` to write a file (overwrite-guarded). Templates are the `config/examples/*.yaml`, embedded via `include_str!` |
| Run on the GPU (M7, Metal) | `just run-wgpu [config]` — end-to-end train through the CLI, backend selected from `compute.backend: wgpu`; defaults to `config/examples/lora-wgpu.yaml` |
| Train from a config (synthetic demo) | `just train [config]` — defaults to `config/examples/lora.yaml` |
| Serve the HTTP/SSE API | `just serve` (`cargo run -p loractl-api`; bind addr via `LORACTL_API_ADDR`, default `127.0.0.1:3000`) |
| Generate shell completions | `just completions [shell]` (e.g. `just completions fish`) |
| Lint (warnings-as-errors) | `just lint` (`cargo clippy --all-targets -- -D warnings`, default/offline features) |
| Lint the opt-in mnist path | `just lint-mnist` (compiles the networked dataset deps) |
| Lint the opt-in gpt2-real path | `just lint-gpt2-real` (compiles the real-gpt2 parity test path) |
| Lint the opt-in qwen-vae-real path | `just lint-vae-real` (compiles the real-VAE parity test path) |
| Lint the opt-in qwen3vl-real path | `just lint-qwen3vl-real` (compiles the real-encoder parity test path) |
| Lint the opt-in mmdit-real path | `just lint-mmdit-real` (compiles the real-MMDiT parity test path) |
| Lint the opt-in wgpu path | `just lint-wgpu` (compiles the wgpu GPU backend; no GPU needed to build) |
| Format / check format | `just fmt` / `just fmt-check` |
| RustSec advisory scan | `just audit` (`cargo audit` over `Cargo.lock`; accepted advisories in `.cargo/audit.toml`) |
| Supply-chain gate (licenses/bans/sources) | `just deny` (`cargo deny check licenses bans sources`, per `deny.toml`) |
| Coverage summary | `just coverage` (`cargo llvm-cov` — per-file table; local, no thresholds) |
| Tests (offline) | `just test` (`cargo test`) — numerics vs PyTorch golden + synthetic convergence |
| Real-MNIST convergence proof | `just test-mnist` (opt-in, downloads MNIST) |
| Real-GPT-2 forward-parity proof | `just test-gpt2-real` (opt-in; run `just gpt2-reference` first) |
| Real Qwen-Image VAE parity proof (M9) | `just test-vae-real` (opt-in; run `just vae-real-reference` first) |
| Real Krea text-encoder parity proof (M10) | `just test-qwen3vl-real` (opt-in; run `just qwen3vl-real-reference` first) |
| Real Krea MMDiT staged-parity proof (M11) | `just test-mmdit-real` (opt-in; run `just mmdit-real-reference` first — 26 GB download) |
| GPU smokes (M7 + M13 f16/ckpt, Metal) | `just test-wgpu` (opt-in; runs both wgpu smokes on a real GPU) |
| GPU smokes (cuda, Linux+NVIDIA host) | `just test-cuda` (opt-in; the synthetic f32+ckpt smoke and the tiny-krea2 diffusion e2e — needs the CUDA toolkit at build time) |
| Train on cuda through the CLI | `just run-cuda [config]` — f32-only (burn#5162); defaults to `config/examples/lora.yaml` with `--backend cuda` |
| Regenerate the numerics golden | `just reference` (needs `torch` via `uv`) |
| Regenerate the BurnTrainer step-loss golden | `just burn-trainer-reference` (dumps burn's real init + batches, replays the loop in `torch` via `uv`; needs `torch`) |
| Regenerate the kohya-ss export golden | `just export-reference` (numpy only, no torch/network) |
| Regenerate the flow-matching golden | `just flow-reference` (needs `torch` via `uv`) |
| Regenerate the tiny-GPT-2 fixture | `just gpt2-tiny-reference` (weights + golden; `torch` via `uv`) |
| Regenerate the real-gpt2 golden | `just gpt2-reference` (downloads `openai-community/gpt2`; `torch`/`transformers` via `uv`) |
| Regenerate the tiny Qwen-VAE fixture (M9) | `just vae-reference` (weights + golden; `torch`/`diffusers` via `uv`, no network) |
| Regenerate the real Qwen-VAE golden (M9) | `just vae-real-reference` (downloads `Qwen/Qwen-Image`'s vae; `torch`/`diffusers` via `uv`) |
| Regenerate the tiny Qwen3-VL fixture (M10) | `just qwen3vl-reference` (weights + golden; `torch`/`transformers` via `uv`, no network) |
| Regenerate the real Krea text-encoder golden (M10) | `just qwen3vl-real-reference` (downloads `krea/Krea-2-Raw`'s text_encoder; `torch`/`transformers` via `uv`) |
| Regenerate the tiny-krea2 bundle + dataset (M14) | `just krea2-reference` (composed fixture; `torch`/`diffusers`/`transformers` via `uv`, no network) |
| Regenerate the tiny MMDiT fixture (M11) | `just mmdit-reference` (downloads `krea-ai/krea-2`'s `mmdit.py` at a pinned commit; `torch` via `uv`) |
| Regenerate the real MMDiT golden (M11) | `just mmdit-real-reference` (downloads `raw.safetensors`, 26.3 GB; kept for M14) |
| One test by name | `cargo test -p loractl-core <test_name>` |

Before committing, the meaningful gate is `just fmt-check && just lint` — CI
parity is intended (the `justfile` mirrors what CI should run). CI additionally
runs the blocking `feature-lints` job (clippy over the opt-in
mnist/gpt2-real/qwen-vae-real/qwen3vl-real/mmdit-real/wgpu paths, mirroring
`just lint-mnist` / `lint-gpt2-real` / `lint-vae-real` /
`lint-qwen3vl-real` / `lint-mmdit-real` / `lint-wgpu`) and the `deny` job
(`cargo deny check`, mirroring `just deny`) —
run those locally too when a change touches a feature-gated path or the
dependency graph. rustfmt is default style; expect it to reflow multi-line
signatures onto one line.

Hosted CI is GPU-free (the `ndarray` default). The real GPU proofs live in a
**dispatchable** `.github/workflows/gpu.yml` (#113) that runs on the
self-hosted RTX 4090 (`popos`): `gh workflow run gpu.yml` (cuda smokes,
`just test-cuda`), `-f suite=all` (adds the wgpu smokes, likely red on the
Vulkan path until burn#5162), and `-f int8_probe=true` (the #96 on-box int8
VRAM/dequant proof, `just quant-probe`). It mirrors CAEF's bench.yml/ci.yml
template — including the `Swatinem/rust-cache` `cache-bin: false` gotcha for
the persistent runner. Needs the popos runner registered to this repo (see the
gpu.yml header).

## Architecture — the one rule that matters

The workspace is three crates:

| Crate | Role |
|---|---|
| `loractl-core` | The pipeline: `TrainConfig` schema, `TrainEvent` stream, `Trainer` trait, `MockTrainer`, the LoRA/GPT-2 modules and `BurnTrainer`. |
| `loractl-cli` | The `loractl` binary — parses args, layers config, renders events. |
| `loractl-api` | The `loractl-api` binary — serializes the same `TrainEvent`s over HTTP/SSE for a GUI; renders nothing itself. Wire contract: `docs/api/events.md`. |

**Load-bearing invariant: `loractl-core` emits events; it never renders.**
Concretely, core must not import `clap` and must not `println!`/write to
stdout/stderr. A `Trainer` reports progress by calling a `&mut dyn
FnMut(TrainEvent)` sink; the *caller* decides how to surface it. The CLI
renders `TrainEvent`s as an `indicatif` progress bar (see the match arm in
`crates/loractl-cli/src/cli.rs`); `loractl-api` serializes the same events
as JSON/SSE. **This is what makes "a GUI can be built separately" real — do
not break it** by having core print or by having the CLI reach into training
internals.

Dependency direction is strictly `cli → core` and `api → core`. Core has no
upward dependencies and no front-end has training logic.

### Swapping the trainer

Swapping the trainer means writing a new `impl Trainer` in core and adding
an arm to **core's `select_trainer`** (`src/train.rs`) — the single factory
that maps `model.base` to a concrete trainer ("synthetic"/"mnist" →
`BurnTrainer`, anything else → `DiffusionTrainer`; pinned by
`tests/trainer_routing.rs`). Both front-ends call it at their one
construction site each: `cli.rs`'s `train()` and the `TrainerFactory`
closure in `loractl-api`'s `main.rs`. If a new trainer forces front-end
changes beyond that factory, the event abstraction has leaked — fix the
abstraction, not the front-end. The LoRA math: freeze the base weights,
train the low-rank factors, forward = `base(x) + (alpha/rank) · B(A(x))`.

### Config layering

A run is fully described by a YAML `TrainConfig` (`config/examples/lora.yaml`).
Precedence, lowest to highest: **YAML file → `LORACTL_`-prefixed env vars (with
`__` for nested keys, e.g. `LORACTL_OPTIM__LR`) → CLI flags.** The env/file
layering is done by `figment` in `load_config`; **CLI flag overrides are
applied by mutating the struct *after* extraction** (`cli.rs`), not via
figment — this is deliberate, since flags are partial and must win last. Match
this pattern when adding new overridable flags.

## Conventions

- Edition 2024, `resolver = "3"`, MSRV pinned at `rust-version = "1.92"` in the
  workspace `Cargo.toml` (bumped from 1.85 to satisfy burn 0.21's MSRV). Shared
  deps go in `[workspace.dependencies]`.
- `Cargo.lock` **is committed** (this workspace produces a binary).
- Roadmap milestones are tracked as issues #1–#4 and #17–#25 and linked from
  the README; keep the two in sync when a milestone lands.
