//! The `loractl` command-line surface.
//!
//! This module is a *renderer* over `loractl-core`: it parses arguments,
//! layers config sources, drives a [`Trainer`], and turns the
//! [`TrainEvent`]s it emits into terminal output. It contains no training
//! logic — swapping `MockTrainer` for a burn-backed trainer later touches
//! only the one line that constructs it.

use anyhow::{Context, Result};
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use figment::{
    Figment,
    providers::{Env, Format, Yaml},
};
use indicatif::{ProgressBar, ProgressStyle};
use loractl_core::{
    BackendKind, Device, NdArray, Precision, Quant, TaskKind, TrainConfig, TrainEvent,
    select_trainer,
};
use std::path::{Path, PathBuf};
use tracing_subscriber::prelude::*;
use tracing_subscriber::{EnvFilter, filter::LevelFilter, fmt};

#[derive(Parser)]
#[command(
    name = "loractl",
    version,
    about = "Terminal-native LoRA trainer — config-driven, completion-friendly, GUI-optional."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a LoRA adapter from a YAML config.
    Train(TrainCmd),

    /// Run one deterministic sample forward pass from a trained adapter.
    Sample(SampleCmd),

    /// Scaffold a starter training config from a template (to stdout, or a file
    /// with `-o`). Presets: `synthetic` (default), `wgpu`, `flow`, `krea2`,
    /// `krea2-comfyui` (scattered ComfyUI file paths).
    Init(InitCmd),

    /// Print shell completions to stdout (e.g. `loractl completions zsh`).
    Completions {
        /// Shell to generate completions for.
        shell: Shell,
    },
}

/// A starter config template selectable by `loractl init --preset`. Each maps
/// to one of the canonical `config/examples/*.yaml` files, embedded verbatim at
/// build time via `include_str!` — so `init` *serves* the same files the docs
/// and tests reference rather than carrying a second, driftable copy of them.
///
/// This is a CLI-side packaging concern (which example to emit), not a config
/// *value*, so — unlike `BackendKind`/`TaskKind`/`Precision` — it lives here and
/// does not belong in core.
#[derive(Clone, Copy, ValueEnum)]
enum Preset {
    /// Offline synthetic LoRA-MLP demo (CPU/ndarray). No dataset or GPU needed.
    Synthetic,
    /// The synthetic demo on the wgpu GPU backend (Metal on macOS). Build with
    /// `--features wgpu`.
    Wgpu,
    /// Rectified-flow (flow-matching) synthetic latent toy (M8).
    Flow,
    /// A real Krea 2 image-diffusion LoRA run through the DiffusionTrainer
    /// (M14). Edit the placeholder `model.base`/`dataset.path` before running.
    Krea2,
    /// A real Krea 2 run pointing at a ComfyUI install's scattered files
    /// (`model.{denoiser,text_encoder,vae}` overrides) — no restructuring,
    /// no duplicate files, no symlinks. Edit the placeholder paths first.
    Krea2Comfyui,
}

impl Preset {
    /// The embedded template body for this preset.
    fn template(self) -> &'static str {
        match self {
            Preset::Synthetic => include_str!("../../../config/examples/lora.yaml"),
            Preset::Wgpu => include_str!("../../../config/examples/lora-wgpu.yaml"),
            Preset::Flow => include_str!("../../../config/examples/flow.yaml"),
            Preset::Krea2 => include_str!("../../../config/examples/krea2-lora.yaml"),
            Preset::Krea2Comfyui => {
                include_str!("../../../config/examples/krea2-comfyui.yaml")
            }
        }
    }

    /// The name clap parses/prints for this preset (e.g. `krea2`), for status
    /// messages. Kept in step with the `ValueEnum` derive's default kebab-casing.
    fn name(self) -> &'static str {
        match self {
            Preset::Synthetic => "synthetic",
            Preset::Wgpu => "wgpu",
            Preset::Flow => "flow",
            Preset::Krea2 => "krea2",
            Preset::Krea2Comfyui => "krea2-comfyui",
        }
    }
}

#[derive(Args)]
struct InitCmd {
    /// Which starter template to emit.
    #[arg(long, value_enum, default_value_t = Preset::Synthetic)]
    preset: Preset,

    /// Write the config to this file instead of stdout. Refuses to overwrite an
    /// existing file unless `--force`.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Overwrite the `--output` file if it already exists.
    #[arg(long)]
    force: bool,
}

/// Parse a `--backend` value through core's [`BackendKind`] `FromStr`, keeping
/// the backend vocabulary defined once in `loractl-core` (a `clap::ValueEnum`
/// derive would have to live in core and pull `clap` in, breaking the
/// core-never-imports-clap invariant).
fn parse_backend(s: &str) -> Result<BackendKind, String> {
    s.parse()
}

/// Parse a `--task` value through core's [`TaskKind`] `FromStr` — the same
/// core-owns-the-vocabulary pattern as [`parse_backend`].
fn parse_task(s: &str) -> Result<TaskKind, String> {
    s.parse()
}

/// Parse a `--precision` value through core's [`Precision`] `FromStr` — the
/// same core-owns-the-vocabulary pattern as [`parse_backend`].
fn parse_precision(s: &str) -> Result<Precision, String> {
    s.parse()
}

/// Parse a `--quant` value through core's [`Quant`] `FromStr` — the same
/// core-owns-the-vocabulary pattern as [`parse_backend`].
fn parse_quant(s: &str) -> Result<Quant, String> {
    s.parse()
}

#[derive(Args)]
struct TrainCmd {
    /// Path to the training config (YAML).
    config: PathBuf,

    /// Override the learning rate from the config.
    #[arg(long)]
    lr: Option<f64>,

    /// Override the number of steps from the config.
    #[arg(long)]
    steps: Option<u64>,

    /// Override the compute backend from the config: `ndarray` (default, CPU),
    /// `wgpu` (GPU — Metal on macOS), `cuda`, or `tch`. GPU backends require the
    /// matching build feature (e.g. `--features wgpu`), else the run bails.
    #[arg(long, value_parser = parse_backend)]
    backend: Option<BackendKind>,

    /// Override the compute device index (GPU ordinal; ignored by ndarray).
    #[arg(long)]
    device: Option<usize>,

    /// Override the training task from the config: `classification` (default,
    /// the synthetic/MNIST demo) or `flow-matching` (the M8 rectified-flow
    /// synthetic toy).
    #[arg(long, value_parser = parse_task)]
    task: Option<TaskKind>,

    /// Override the float precision from the config: `f32` (default) or
    /// `f16` (wgpu only — halves resident weight memory; M13).
    #[arg(long, value_parser = parse_precision)]
    precision: Option<Precision>,

    /// Override frozen-base quantization from the config: `none` (default),
    /// `int8` (the diffusion trainer's MMDiT base as per-block int8, ~1/4 f32),
    /// or `int4` (per-block int4, ~1/8 f32 — halves int8's resident base to fit
    /// a 24 GB step); ndarray or cuda + f32 only — #96.
    #[arg(long, value_parser = parse_quant)]
    quant: Option<Quant>,

    /// Override activation checkpointing from the config (M13): recompute
    /// activations during backward instead of storing them — numerically
    /// identical, less memory, slower per step. Bare `--grad-checkpointing`
    /// means true; an explicit `false` overrides a config-file `true`.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    grad_checkpointing: Option<bool>,

    /// Override `model.denoiser`: path to the denoiser file (ComfyUI
    /// scattered layout, #101). Absolute paths are used verbatim; relative
    /// paths join onto `model.base`. fp8-vs-bf16 is auto-detected from the
    /// file header.
    #[arg(long)]
    denoiser: Option<PathBuf>,

    /// Override `model.text_encoder`: path to the Qwen3-VL text-encoder
    /// file (#101). Absolute verbatim; relative joins onto `model.base`.
    #[arg(long)]
    text_encoder: Option<PathBuf>,

    /// Override `model.vae`: path to the Qwen-Image VAE file (#101).
    /// Absolute verbatim; relative joins onto `model.base`.
    #[arg(long)]
    vae: Option<PathBuf>,

    /// Override `model.tokenizer`: path to a `tokenizer.json` (#101).
    /// Absolute verbatim; relative joins onto `model.base`. Without this (and
    /// with no `base/tokenizer/tokenizer.json`), the model-invariant Qwen3-VL
    /// tokenizer is fetched once and cached; naming a missing file here is an
    /// error, never a silent fetch.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
}

#[derive(Args)]
struct SampleCmd {
    /// Path to the trained adapter (`.safetensors`).
    adapter: PathBuf,

    /// Optional text that deterministically seeds the sample's synthetic
    /// input (the same prompt always reproduces the same output). `LoraMlp`
    /// has no tokenizer, so this is not text generation — see `sample --help`
    /// output / README for the honest framing.
    #[arg(short, long)]
    prompt: Option<String>,
}

/// Initialize GlitchTip telemetry (via the Sentry-compatible SDK) and tracing.
///
/// Returns a guard that must be held for the lifetime of the process —
/// dropping it flushes any buffered events on exit. Telemetry is a no-op when
/// `SENTRY_DSN` is unset, so this is always safe to call.
///
/// Two tracing layers are installed:
/// - a `fmt` layer renders human-readable logs, gated by the usual `RUST_LOG`
///   env filter (console behaviour is unchanged);
/// - a Sentry layer forwards `INFO`-and-above tracing events to GlitchTip —
///   `ERROR` events become issues, `WARN`/`INFO` attach as breadcrumbs for
///   context — independent of `RUST_LOG` so telemetry doesn't hinge on log
///   verbosity.
pub fn init_telemetry() -> sentry::ClientInitGuard {
    // GlitchTip speaks the Sentry ingest protocol; the DSN is read from the
    // `SENTRY_DSN` environment variable. `release` tags events with the crate
    // version so issues group by build.
    let guard = sentry::init(sentry::ClientOptions {
        release: sentry::release_name!(),
        ..Default::default()
    });

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_target(false)
                .with_filter(EnvFilter::from_default_env()),
        )
        .with(sentry::integrations::tracing::layer().with_filter(LevelFilter::INFO))
        .init();

    if guard.is_enabled() {
        tracing::debug!("GlitchTip telemetry enabled");
    } else {
        tracing::debug!("GlitchTip telemetry disabled (SENTRY_DSN unset)");
    }

    guard
}

/// Parse arguments and dispatch. Called by `main`.
pub fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Train(cmd) => train(cmd),
        Command::Sample(cmd) => sample(cmd),
        Command::Init(cmd) => init(cmd),
        Command::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            clap_complete::generate(shell, &mut cmd, name, &mut std::io::stdout());
            Ok(())
        }
    }
}

/// Load a [`TrainConfig`], layering sources lowest-to-highest precedence:
/// the YAML file, then `LORACTL_`-prefixed environment variables. CLI flag
/// overrides are applied by the caller after extraction (they're the last
/// word).
fn load_config(path: &Path) -> Result<TrainConfig> {
    Figment::new()
        .merge(Yaml::file(path))
        .merge(Env::prefixed("LORACTL_").split("__"))
        .extract()
        .with_context(|| format!("loading config from {}", path.display()))
}

/// Resolve the effective [`TrainConfig`] for a train command, layering every
/// source lowest-to-highest precedence: the YAML file, then `LORACTL_`
/// environment variables (both via [`load_config`]), then the CLI flag
/// overrides — which are applied here, *after* extraction, so they are the
/// last word. Extracted from [`train`] so the precedence contract is testable
/// without running a real training loop.
fn resolve_config(cmd: &TrainCmd) -> Result<TrainConfig> {
    let mut config = load_config(&cmd.config)?;
    if let Some(lr) = cmd.lr {
        config.optim.lr = lr;
    }
    if let Some(steps) = cmd.steps {
        config.steps = steps;
    }
    if let Some(backend) = cmd.backend {
        config.compute.backend = backend;
    }
    if let Some(device) = cmd.device {
        config.compute.device = device;
    }
    if let Some(task) = cmd.task {
        config.task = task;
    }
    if let Some(precision) = cmd.precision {
        config.compute.precision = precision;
    }
    if let Some(quant) = cmd.quant {
        config.compute.quant = quant;
    }
    if let Some(grad_checkpointing) = cmd.grad_checkpointing {
        config.compute.grad_checkpointing = grad_checkpointing;
    }
    // The #101 per-component path overrides: the flags mirror the
    // `model.denoiser`/`text_encoder`/`vae`/`tokenizer` keys (relative paths
    // join onto `model.base` at load, same as the YAML/env layers).
    if let Some(denoiser) = &cmd.denoiser {
        config.model.denoiser = Some(denoiser.clone());
    }
    if let Some(text_encoder) = &cmd.text_encoder {
        config.model.text_encoder = Some(text_encoder.clone());
    }
    if let Some(vae) = &cmd.vae {
        config.model.vae = Some(vae.clone());
    }
    if let Some(tokenizer) = &cmd.tokenizer {
        config.model.tokenizer = Some(tokenizer.clone());
    }
    Ok(config)
}

fn train(cmd: TrainCmd) -> Result<()> {
    let config = resolve_config(&cmd)?;

    std::fs::create_dir_all(&config.output.dir)
        .with_context(|| format!("creating output dir {}", config.output.dir.display()))?;

    let bar = ProgressBar::new(config.steps.max(1));
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
        )
        .expect("valid progress template")
        .progress_chars("=>-"),
    );

    // The trainer factory — the constructor seam the load-bearing invariant
    // protects. Routing on `model.base` lives in core (`select_trainer`) so
    // the CLI and the API cannot drift apart.
    let mut trainer = select_trainer(&config);
    let adapter = trainer.train(&config, &mut |event| match event {
        TrainEvent::Started { total_steps } => bar.set_length(total_steps),
        TrainEvent::Step { step, loss, lr } => {
            bar.set_position(step);
            bar.set_message(format!("loss {loss:.4}  lr {lr:.2e}"));
        }
        TrainEvent::Checkpoint { step, path } => {
            bar.suspend(|| tracing::info!(step, path = %path.display(), "checkpoint"));
        }
        TrainEvent::Sample { step, path } => {
            bar.suspend(|| tracing::info!(step, path = %path.display(), "sample"));
        }
        TrainEvent::Warning { message } => {
            bar.suspend(|| tracing::warn!("{message}"));
        }
        TrainEvent::Finished { adapter_path } => {
            bar.finish_with_message(format!("done → {}", adapter_path.display()));
        }
    })?;

    println!("adapter: {}", adapter.display());
    Ok(())
}

fn sample(cmd: SampleCmd) -> Result<()> {
    // Inference-only: no autodiff needed, so this is decoupled from
    // `BurnTrainer`'s internal Autodiff-wrapped backend type. `NdArray`/
    // `Device` are re-exported from `loractl-core` (rather than depending on
    // `burn` directly here) so this crate's `Cargo.toml` doesn't track
    // burn's version/features a second time in lockstep with core's.
    type B = NdArray;
    let device: Device<B> = Default::default();

    let seed = loractl_core::sample::seed_from_prompt(cmd.prompt.as_deref());
    // One core-side call loads AND samples — `sample_adapter` reads the
    // sidecar's task and refuses flow-matching adapters (a velocity net has
    // no classes), so this renderer inherits the fail-fast check instead of
    // having to remember it.
    let output = loractl_core::sample::sample_adapter::<B>(&cmd.adapter, seed, &device)
        .with_context(|| format!("sampling from adapter {}", cmd.adapter.display()))?;

    println!(
        "note: LoraMlp is a synthetic classifier with no tokenizer — `--prompt` \
         deterministically seeds this sample's synthetic input rather than generating \
         text; real language-model sampling is future work beyond M4/M5 \
         (see docs/adrs/0002-adapter-format-and-sample-semantics.md)."
    );
    println!("predicted class: {}", output.predicted_class);

    // `total_cmp` (never `partial_cmp(...).unwrap()`) so this can't panic even
    // if a future change to `run_sample`'s validation is loosened — see
    // `loractl_core::sample::run_sample` for the primary NaN/Inf guard.
    let mut ranked: Vec<(usize, f32)> = output.logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("top logits:");
    for (class, logit) in ranked.iter().take(2) {
        println!("  class {class}: {logit:.4}");
    }

    Ok(())
}

/// Emit a starter config from the selected [`Preset`] — to stdout by default,
/// or to `--output` (creating parent dirs, refusing to clobber without
/// `--force`). Non-destructive by default and pipeable
/// (`loractl init --preset krea2 > config/my.yaml`); the template is the
/// canonical example file, embedded at build time, so `init` cannot drift from
/// the documented examples.
fn init(cmd: InitCmd) -> Result<()> {
    let template = cmd.preset.template();
    match &cmd.output {
        None => {
            print!("{template}");
            Ok(())
        }
        Some(path) => {
            if path.exists() && !cmd.force {
                anyhow::bail!(
                    "{} already exists; pass --force to overwrite, or -o a different path",
                    path.display()
                );
            }
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating parent dir {}", parent.display()))?;
            }
            std::fs::write(path, template)
                .with_context(|| format!("writing config to {}", path.display()))?;
            // Status to stderr so a piped stdout stays clean even with -o.
            eprintln!("wrote {} config to {}", cmd.preset.name(), path.display());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    //! Config-layering precedence (issue #47): YAML file < `LORACTL_` env <
    //! CLI flags. Nothing tested this before, so a swapped merge order in
    //! `load_config` or applying flags before extraction would silently break
    //! user overrides. `figment::Jail` isolates env vars and cwd per test.

    // `figment::Jail::expect_with`'s closure returns `Result<(), figment::Error>`;
    // `figment::Error` is large, so `?` here trips `clippy::result_large_err`.
    // It's a fixed part of the Jail test API, not our code to shrink.
    #![allow(clippy::result_large_err)]

    use super::*;
    use figment::Jail;

    /// A minimal-but-complete config YAML: `model`, `lora`, and `dataset` are
    /// the required keys (no serde default on `TrainConfig`), so all three must
    /// be present for extraction to succeed; `lora: {}` takes LoraConfig's own
    /// defaults, mirroring the API tests' `"lora": {}`.
    const YAML: &str = "steps: 10\n\
         model:\n  base: synthetic\n\
         dataset:\n  path: unused\n\
         lora: {}\n\
         optim:\n  lr: 0.0001\n";

    fn cmd_for(config: &str) -> TrainCmd {
        TrainCmd {
            config: config.into(),
            lr: None,
            steps: None,
            backend: None,
            device: None,
            task: None,
            precision: None,
            quant: None,
            grad_checkpointing: None,
            denoiser: None,
            text_encoder: None,
            vae: None,
            tokenizer: None,
        }
    }

    #[test]
    fn file_value_is_used_when_no_env_or_flag() {
        Jail::expect_with(|jail| {
            jail.create_file("config.yaml", YAML)?;
            let config = resolve_config(&cmd_for("config.yaml")).expect("resolve");
            assert_eq!(config.optim.lr, 0.0001);
            assert_eq!(config.steps, 10);
            Ok(())
        });
    }

    #[test]
    fn env_beats_file_and_nested_keys_split_on_double_underscore() {
        Jail::expect_with(|jail| {
            jail.create_file("config.yaml", YAML)?;
            jail.set_env("LORACTL_OPTIM__LR", "0.0002");
            jail.set_env("LORACTL_OUTPUT__DIR", "/tmp/from-env");

            let config = resolve_config(&cmd_for("config.yaml")).expect("resolve");
            assert_eq!(config.optim.lr, 0.0002, "env must beat the file value");
            assert_eq!(config.steps, 10, "unset keys keep the file value");
            assert_eq!(
                config.output.dir,
                std::path::PathBuf::from("/tmp/from-env"),
                "`__` must split into the nested output.dir key"
            );
            Ok(())
        });
    }

    #[test]
    fn cli_flags_beat_env_and_file() {
        Jail::expect_with(|jail| {
            jail.create_file("config.yaml", YAML)?;
            jail.set_env("LORACTL_OPTIM__LR", "0.0002");
            jail.set_env("LORACTL_STEPS", "20");

            let mut cmd = cmd_for("config.yaml");
            cmd.lr = Some(0.0003); // beats env 0.0002, file 0.0001
            cmd.steps = Some(30); // beats env 20, file 10
            cmd.backend = Some(BackendKind::Wgpu); // flag-only override
            cmd.task = Some(TaskKind::FlowMatching); // flag-only override
            cmd.precision = Some(Precision::F16); // M13 flag-only override
            cmd.quant = Some(Quant::Int8); // #96 flag-only override
            cmd.grad_checkpointing = Some(true); // M13 flag-only override
            // The #101 path overrides: --denoiser beats the env layer below;
            // the other three are flag-only.
            cmd.denoiser = Some("flag/denoiser.safetensors".into());
            cmd.text_encoder = Some("flag/te.safetensors".into());
            cmd.vae = Some("flag/vae.safetensors".into());
            cmd.tokenizer = Some("flag/tokenizer.json".into());
            jail.set_env("LORACTL_MODEL__DENOISER", "env/denoiser.safetensors");

            let config = resolve_config(&cmd).expect("resolve");
            assert_eq!(config.optim.lr, 0.0003, "flag must win over env and file");
            assert_eq!(config.steps, 30, "flag must win over env and file");
            assert_eq!(config.compute.backend, BackendKind::Wgpu);
            assert_eq!(config.task, TaskKind::FlowMatching);
            // The M13 knobs reach the config (the trainer-side dispatch is
            // covered in core: the f16 guard errors from inside the match,
            // and checkpointing is bit-identical by design).
            assert_eq!(config.compute.precision, Precision::F16);
            assert!(config.compute.grad_checkpointing);
            // The #96 quant knob reaches the config the same way (the trainer
            // guard restricts the legal backend/precision combos in core).
            assert_eq!(config.compute.quant, Quant::Int8);
            // The #101 path flags reach the config, and --denoiser beats the
            // env layer (the component loaders resolve relative-vs-absolute
            // in core; here only the layering is under test).
            assert_eq!(
                config.model.denoiser.as_deref(),
                Some(std::path::Path::new("flag/denoiser.safetensors")),
                "flag must win over the env layer"
            );
            assert_eq!(
                config.model.text_encoder.as_deref(),
                Some(std::path::Path::new("flag/te.safetensors"))
            );
            assert_eq!(
                config.model.vae.as_deref(),
                Some(std::path::Path::new("flag/vae.safetensors"))
            );
            assert_eq!(
                config.model.tokenizer.as_deref(),
                Some(std::path::Path::new("flag/tokenizer.json"))
            );
            Ok(())
        });
    }

    /// Every `loractl init` preset's embedded template must parse into a
    /// `TrainConfig`. Only `lora.yaml` was parse-pinned before (by
    /// `tests/example_config.rs`); this covers `wgpu`/`flow`/`krea2` too, so a
    /// schema change that breaks one of those example files fails here instead
    /// of silently handing users an un-parseable starter config.
    #[test]
    fn every_init_preset_template_parses() {
        for preset in Preset::value_variants() {
            let name = preset.name();
            let config: TrainConfig = Figment::new()
                .merge(Yaml::string(preset.template()))
                .extract()
                .unwrap_or_else(|e| panic!("preset `{name}` template must parse: {e}"));
            // A sanity check that the embedded body is the real example, not
            // empty: every example ships a non-zero step count.
            assert!(config.steps > 0, "preset `{name}` should set steps");
        }
    }

    #[test]
    fn m13_env_layer_reaches_compute_knobs() {
        Jail::expect_with(|jail| {
            jail.create_file("config.yaml", YAML)?;
            jail.set_env("LORACTL_COMPUTE__PRECISION", "f16");
            jail.set_env("LORACTL_COMPUTE__GRAD_CHECKPOINTING", "true");

            let config = resolve_config(&cmd_for("config.yaml")).expect("resolve");
            assert_eq!(config.compute.precision, Precision::F16);
            assert!(config.compute.grad_checkpointing);
            Ok(())
        });
    }
}
