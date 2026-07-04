# loractl

A terminal-native LoRA trainer, in Rust.

Most LoRA trainers bolt a half-baked web GUI onto a Python training core.
`loractl` inverts that: the **CLI is the primary surface** — config-driven,
completion-friendly, pipe-able — and a GUI, if anyone wants one, is just
another renderer layered on the same core over an API. The name says the
thesis: a `*ctl` tool, like `kubectl` or `systemctl`.

> **Status: real base model (milestone 3).** loractl now loads a **real
> GPT-2's** safetensors weights into a hand-built [burn](https://burn.dev) module
> tree and proves **forward-pass parity** against the PyTorch reference, then
> attaches LoRA to the loaded model and runs a training step. The default trainer
> remains the M2 `BurnTrainer` (a LoRA-adapted MLP with genuine autodiff, a real
> optimizer, and cross-entropy loss, pinned against a PyTorch numerics golden);
> the GPT-2 loading + parity harness ships as always-run offline tests plus an
> opt-in real-`gpt2` test. The dependency-free `MockTrainer` is still available
> for pipeline testing. See [Roadmap](#roadmap).

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
| `loractl-core` | The pipeline: config schema, `TrainEvent` stream, `Trainer` trait, the LoRA module + `BurnTrainer`, the GPT-2 model + safetensors loader. **No CLI, no stdout.** | burn, burn-store |
| `loractl-cli` | The `loractl` binary. Parses args, layers config, renders events. | `loractl-core` |
| `loractl-api` *(future)* | HTTP server / language bindings for an optional GUI. | `loractl-core` |

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
- **Checkpoint format.** Checkpoints and the final adapter are written as
  burn-native MessagePack records (`.mpk`) — every emitted path really exists on
  disk. Interoperable **safetensors** adapter I/O is milestone 4
  ([#3](https://github.com/laurigates/loractl/issues/3)), not yet here.
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

## Config

A run is fully described by a YAML config (see `config/examples/lora.yaml`).
Precedence, lowest to highest: **YAML file → `LORACTL_`-prefixed env vars →
CLI flags.** Nested keys use `__` in env vars (`LORACTL_OUTPUT__DIR=/tmp/out`).

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
- [ ] **M4 — Sampling, adapter I/O & modern arch** ([#3](https://github.com/laurigates/loractl/issues/3))**.** `loractl sample`, safetensors
      adapter read/write, validation samples during training — and the next base
      family, **SmolLM2-135M** (modern LLaMA-style: RoPE + RMSNorm + SwiGLU; note
      burn's RoPE is interleaved vs HF's half-split — see ADR-0001).
- [ ] **M5 — API crate** ([#4](https://github.com/laurigates/loractl/issues/4))**.** Expose the event stream over HTTP so a GUI can be
      built independently.

## License

MIT © Lauri Gates
