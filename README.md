# loractl

A terminal-native LoRA trainer, in Rust.

Most LoRA trainers bolt a half-baked web GUI onto a Python training core.
`loractl` inverts that: the **CLI is the primary surface** — config-driven,
completion-friendly, pipe-able — and a GUI, if anyone wants one, is just
another renderer layered on the same core over an API. The name says the
thesis: a `*ctl` tool, like `kubectl` or `systemctl`.

> **Status: correctness harness (milestone 2).** The default trainer is now a
> real, [burn](https://burn.dev)-backed `BurnTrainer` that trains a LoRA-adapted
> MLP with genuine autodiff, a real optimizer, and cross-entropy loss. Out of
> the box it trains a **synthetic** LoRA-MLP classification demo (fully offline,
> fast); real base-model + image ingestion is a later milestone. The LoRA
> numerics are pinned against a PyTorch reference, and an opt-in `mnist` feature
> trains/scores the same model on real MNIST. The dependency-free `MockTrainer`
> is still available for pipeline testing. See [Roadmap](#roadmap).

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
| `loractl-core` | The pipeline: config schema, `TrainEvent` stream, `Trainer` trait, the LoRA module + `BurnTrainer`. **No CLI, no stdout.** | burn |
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
- [ ] **M3 — Real base model** ([#2](https://github.com/laurigates/loractl/issues/2))**.** Load a real base model's weights into burn
      (the genuinely hard part: state-dict mapping), wire the forward pass.
- [ ] **M4 — Sampling & adapter I/O** ([#3](https://github.com/laurigates/loractl/issues/3))**.** `loractl sample`, safetensors adapter
      read/write, validation samples during training.
- [ ] **M5 — API crate** ([#4](https://github.com/laurigates/loractl/issues/4))**.** Expose the event stream over HTTP so a GUI can be
      built independently.

## License

MIT © Lauri Gates
