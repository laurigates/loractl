# loractl

A terminal-native LoRA trainer, in Rust.

Most LoRA trainers bolt a half-baked web GUI onto a Python training core.
`loractl` inverts that: the **CLI is the primary surface** — config-driven,
completion-friendly, pipe-able — and a GUI, if anyone wants one, is just
another renderer layered on the same core over an API. The name says the
thesis: a `*ctl` tool, like `kubectl` or `systemctl`.

> **Status: early scaffold (milestone 1).** The architecture and CLI are
> real and working; the trainer is currently a dependency-free `MockTrainer`
> that exercises the full event → render pipeline without any ML. Milestone 2
> wires in [burn](https://burn.dev) and a real LoRA module. See
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
| `loractl-core` | The pipeline: config schema, `TrainEvent` stream, `Trainer` trait. **No CLI, no stdout.** | (burn, later) |
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

# Run a (mock) training job from the example config
cargo run -p loractl-cli -- train config/examples/lora.yaml

# Override config fields from the CLI
cargo run -p loractl-cli -- train config/examples/lora.yaml --lr 5e-5 --steps 2000

# ...or from the environment
LORACTL_OPTIM__LR=5e-5 cargo run -p loractl-cli -- train config/examples/lora.yaml

# Generate shell completions
cargo run -p loractl-cli -- completions zsh > ~/.zfunc/_loractl
```

Recipes live in the `justfile` (`just` to list): `just build`, `just train`,
`just completions fish`, `just lint`, `just fmt`.

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
- [ ] **M2 — Correctness harness** ([#1](https://github.com/laurigates/loractl/issues/1))**.** Add burn; implement a LoRA `Module`
      (freeze base, train A·B); train a LoRA on a *tiny* model (MNIST MLP)
      and verify numerics against a Python reference — **before** any large
      model. Prove the loop is right in isolation.
- [ ] **M3 — Real base model** ([#2](https://github.com/laurigates/loractl/issues/2))**.** Load a real base model's weights into burn
      (the genuinely hard part: state-dict mapping), wire the forward pass.
- [ ] **M4 — Sampling & adapter I/O** ([#3](https://github.com/laurigates/loractl/issues/3))**.** `loractl sample`, safetensors adapter
      read/write, validation samples during training.
- [ ] **M5 — API crate** ([#4](https://github.com/laurigates/loractl/issues/4))**.** Expose the event stream over HTTP so a GUI can be
      built independently.

## License

MIT © Lauri Gates
