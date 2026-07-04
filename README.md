# loractl

A terminal-native LoRA trainer, in Rust.

Most LoRA trainers bolt a half-baked web GUI onto a Python training core.
`loractl` inverts that: the **CLI is the primary surface** — config-driven,
completion-friendly, pipe-able — and a GUI, if anyone wants one, is just
another renderer layered on the same core over an API. The name says the
thesis: a `*ctl` tool, like `kubectl` or `systemctl`.

> **Status: sampling & adapter I/O (milestone 4).** `loractl sample` now
> produces real output from a saved adapter, adapters save to and load from
> interoperable **safetensors** (adapter-only — just the trainable LoRA
> factors, plus a JSON metadata sidecar), and training can emit optional
> validation samples. M3 remains available: loractl loads a **real GPT-2's**
> safetensors weights into a hand-built [burn](https://burn.dev) module tree
> and proves **forward-pass parity** against the PyTorch reference, then
> attaches LoRA to the loaded model and runs a training step. The default
> trainer remains the M2 `BurnTrainer` (a LoRA-adapted MLP with genuine
> autodiff, a real optimizer, and cross-entropy loss, pinned against a
> PyTorch numerics golden). The dependency-free `MockTrainer` is still
> available for pipeline testing. See [Roadmap](#roadmap).

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
  adapter-only **safetensors** files — only the trainable LoRA factors
  (`fc2.lora_a.weight`, `fc2.lora_b.weight`), never the frozen base — plus a
  `<path>.json` metadata sidecar (seed, rank, alpha, dimensions) that makes
  the file self-describing enough to reconstruct the rest. See [Sampling &
  adapter I/O (M4)](#sampling--adapter-io-m4) below and
  [ADR-0002](docs/adrs/0002-adapter-format-and-sample-semantics.md).
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

`crates/loractl-core/src/adapter.rs` and `src/sample.rs` implement milestone 4
([#3](https://github.com/laurigates/loractl/issues/3)). See
[ADR-0002](docs/adrs/0002-adapter-format-and-sample-semantics.md) for the full
decision record.

- **Tensor-naming scheme.** An adapter file holds *only* the two trainable
  LoRA factors, at their natural burn module path:

  | Path | Meaning |
  |---|---|
  | `fc2.lora_a.weight` | Down-projection `A`, `hidden -> rank` |
  | `fc2.lora_b.weight` | Up-projection `B`, `rank -> out` |

  The frozen base (`fc1.*`, `fc2.base.*`) is never written — that's the whole
  point of an "adapter" as distinct from a full model checkpoint. This
  mirrors the *pattern* of community LoRA conventions (HF PEFT names its
  factors `lora_A`/`lora_B`) without claiming literal PEFT interop —
  `LoraMlp` is a synthetic classifier, not a downloadable public base model.
- **The `<path>.json` metadata sidecar.** burn-store 0.21's safetensors
  metadata is write-only (no public read-back API — verified against the
  crate source), so `save_adapter` writes a plain JSON sidecar next to the
  `.safetensors` file holding the training seed plus rank/alpha/dimensions.
  `load_adapter` re-seeds the device with that seed and reconstructs the
  model immediately, which reproduces the frozen base **bit-identically**
  without ever persisting it — the same two-file shape as HF PEFT's
  `adapter_model.safetensors` + `adapter_config.json`.
- **`loractl sample`.** Loads an adapter, derives a deterministic seed from
  `--prompt` (or `0` with no prompt), and runs the model on a synthetic input
  built from that seed — printing the predicted class and logits.
  **Honesty note:** `LoraMlp` is a synthetic/MNIST-shaped classifier, not a
  language model — there is no tokenizer, so `--prompt` deterministically
  seeds the sample input rather than generating text. Real generative
  sampling awaits a future language-model milestone (see
  [ADR-0001](docs/adrs/0001-first-real-target-model.md)).
- **In-training validation samples.** Set `output.sample_every: N` (`0` =
  disabled, the default) in the training config; every `N`th step,
  `BurnTrainer` runs one sample against a fixed probe input and writes
  `sample-{step}.json`, emitting `TrainEvent::Sample`. Using the same fixed
  probe every time (not a fresh random one) is the point — it lets you watch
  one input's prediction evolve across `sample-N.json` files as training
  progresses.
- **Round-trip proof.** `cargo test` (`just test`) includes
  `tests/adapter_roundtrip.rs`: trains one real optimizer step (so the
  adapter's `lora_b` factor is genuinely non-zero — a fresh, all-zero adapter
  would trivially "round-trip" even through a broken load path), saves,
  reloads on a plain backend, and asserts the forward pass is bit-identical.
  It also asserts the safetensors file holds *exactly* the two LoRA tensors
  and that the sidecar metadata round-trips.

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
- [x] **M4 — Sampling & adapter I/O** ([#3](https://github.com/laurigates/loractl/issues/3))**.** Adapters save to and load
      from adapter-only safetensors (LoRA factors only, plus a JSON metadata
      sidecar for self-describing reconstruction of the frozen base);
      `loractl sample` produces real output from a saved adapter, with
      `--prompt` deterministically seeding a synthetic input (honest about
      `LoraMlp` being a classifier, not a language model); optional
      in-training validation samples are written and reported via
      `TrainEvent::Sample`. See
      [ADR-0002](docs/adrs/0002-adapter-format-and-sample-semantics.md).
- [ ] **M5 — API crate** ([#4](https://github.com/laurigates/loractl/issues/4))**.** Expose the event stream over HTTP so a GUI can be
      built independently.

A natural next *target model* (beyond the tracked milestones) is **SmolLM2-135M**
— a modern LLaMA-style architecture (RoPE + RMSNorm + SwiGLU) that reuses M3's
loader and parity harness; note burn's RoPE is *interleaved* vs HF's *half-split*
(see [ADR-0001](docs/adrs/0001-first-real-target-model.md)).

## License

MIT © Lauri Gates
