//! The `loractl` command-line surface.
//!
//! This module is a *renderer* over `loractl-core`: it parses arguments,
//! layers config sources, drives a [`Trainer`], and turns the
//! [`TrainEvent`]s it emits into terminal output. It contains no training
//! logic — swapping `MockTrainer` for a burn-backed trainer later touches
//! only the one line that constructs it.

use anyhow::{Context, Result};
use clap::{ArgAction, Args, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use figment::{
    Figment,
    providers::{Env, Format, Yaml},
};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use loractl_core::{
    BackendKind, Device, NdArray, Precision, Quant, TaskKind, TrainConfig, TrainEvent,
    select_trainer,
};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{EnvFilter, filter::LevelFilter, fmt};

#[derive(Parser)]
#[command(
    name = "loractl",
    version,
    about = "Terminal-native LoRA trainer — config-driven, completion-friendly, GUI-optional."
)]
pub struct Cli {
    /// Increase verbosity of loractl's own logs: `-v` info, `-vv` debug,
    /// `-vvv` trace. Without it only warnings and errors are printed.
    /// Third-party crates stay at warn — use `RUST_LOG` for those (e.g.
    /// `RUST_LOG=wgpu_core=debug,warn`); a non-empty `RUST_LOG` overrides
    /// this flag entirely.
    #[arg(short, long, action = ArgAction::Count, global = true, conflicts_with = "quiet")]
    verbose: u8,

    /// Print errors only — suppress the warnings shown by default. Mutually
    /// exclusive with `-v`; `RUST_LOG` still overrides it.
    #[arg(short, long, global = true)]
    quiet: bool,

    #[command(subcommand)]
    command: Command,
}

impl Cli {
    /// The log filter this invocation asks for, resolving `RUST_LOG` over the
    /// `-v`/`-q` flags. Reading the environment is confined here so the
    /// precedence itself stays a pure, tested function ([`resolve_verbosity`]).
    pub fn verbosity(&self) -> Verbosity {
        resolve_verbosity(
            self.verbose,
            self.quiet,
            std::env::var("RUST_LOG").ok().as_deref(),
        )
    }
}

/// The resolved log filter for a run.
///
/// Two cases rather than one level, because `RUST_LOG` carries per-target
/// directives (`loractl_core=debug,warn`) that cannot be flattened into a
/// single [`LevelFilter`] without losing information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verbosity {
    /// `RUST_LOG` was set and non-empty; its directives are used verbatim.
    Env(String),
    /// Derived from `-v`/`-q`, or the WARN default when neither was passed.
    Level(LevelFilter),
}

/// Map the verbosity flags to a level, ignoring the environment.
///
/// The default is **WARN**, not "nothing": a run with no flags must still
/// surface the honest warnings a trainer emits as `TrainEvent::Warning`.
fn level_for(verbose: u8, quiet: bool) -> LevelFilter {
    if quiet {
        return LevelFilter::ERROR;
    }
    match verbose {
        0 => LevelFilter::WARN,
        1 => LevelFilter::INFO,
        2 => LevelFilter::DEBUG,
        _ => LevelFilter::TRACE,
    }
}

/// The env-filter directives a flag-derived `level` expands to.
///
/// **Scoped to loractl's own target on purpose.** A bare level is a *global*
/// default directive, and `tracing-subscriber`'s default `tracing-log` feature
/// bridges every `log::debug!`/`log::trace!` from wgpu, naga, cubecl, hyper and
/// friends into it — measured on a `--features wgpu` build, `-vv` over a
/// two-step run is tens of thousands of lines of naga type tables with the
/// handful of loractl phase lines lost inside. Third-party crates keep a `warn`
/// floor so a genuine upstream warning still reaches the operator; `-q` drops
/// that floor too.
///
/// The target is `loractl` (the `[[bin]] name`, which is what `module_path!`
/// roots at) — `loractl_cli` would match nothing, and `loractl-core` has no
/// `tracing` dependency at all, by the render invariant.
fn flag_directives(level: LevelFilter) -> String {
    let level = level.to_string().to_lowercase();
    let baseline = if level == "error" || level == "off" {
        level.as_str()
    } else {
        "warn"
    };
    format!("{baseline},loractl={level}")
}

/// Resolve the effective filter: an explicit, non-empty `RUST_LOG` wins over
/// the flags; otherwise the flags (or the WARN default) decide.
fn resolve_verbosity(verbose: u8, quiet: bool, rust_log: Option<&str>) -> Verbosity {
    match rust_log.map(str::trim).filter(|s| !s.is_empty()) {
        Some(directives) => Verbosity::Env(directives.to_string()),
        None => Verbosity::Level(level_for(verbose, quiet)),
    }
}

#[derive(Subcommand)]
enum Command {
    /// Train a LoRA adapter from a YAML config.
    Train(TrainCmd),

    /// Run one deterministic sample forward pass from a trained adapter.
    Sample(SampleCmd),

    /// Print a `.safetensors` file's embedded metadata — trigger words, base
    /// model, and the training record. Reads only the header, so it is
    /// instant even on a multi-gigabyte checkpoint.
    Inspect(InspectCmd),

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

    /// Override `metadata.trigger_words`: the caption token(s) that activate
    /// the trained LoRA, embedded in the exported file's `__metadata__`
    /// header (`ss_trained_words` / `modelspec.trigger_phrase`). Repeatable;
    /// passing any replaces the config's whole list.
    #[arg(long = "trigger-word", value_name = "WORD")]
    trigger_words: Vec<String>,

    /// Override `metadata.embed`: write no `__metadata__` header at all.
    /// The export then carries only tensors (byte-reproducible; no
    /// timestamps, no run details).
    #[arg(long)]
    no_metadata: bool,
}

#[derive(Args)]
struct InspectCmd {
    /// Path to the `.safetensors` file to read metadata from.
    file: PathBuf,

    /// Print the raw metadata map as JSON instead of the grouped, key-sorted
    /// listing (nested values such as `ss_tag_frequency` stay JSON strings).
    #[arg(long)]
    json: bool,
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
/// - a `fmt` layer renders human-readable logs, gated by the [`Verbosity`] the
///   caller resolved from `-v`/`-q`/`RUST_LOG` (default: warnings and above);
/// - a Sentry layer forwards `INFO`-and-above tracing events to GlitchTip —
///   `ERROR` events become issues, `WARN`/`INFO` attach as breadcrumbs for
///   context — independent of the console filter so telemetry doesn't hinge on
///   log verbosity.
///
/// Called *after* `Cli::parse()` (the flags are an input), which is why it is
/// a separate step from [`run`] rather than something `main` can do first.
pub fn init_telemetry(verbosity: Verbosity) -> sentry::ClientInitGuard {
    // GlitchTip speaks the Sentry ingest protocol; the DSN is read from the
    // `SENTRY_DSN` environment variable. `release` tags events with the crate
    // version so issues group by build.
    let guard = sentry::init(sentry::ClientOptions {
        release: sentry::release_name!(),
        ..Default::default()
    });

    let filter = match verbosity {
        Verbosity::Env(directives) => EnvFilter::new(directives),
        Verbosity::Level(level) => EnvFilter::new(flag_directives(level)),
    };

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false).with_filter(filter))
        .with(sentry::integrations::tracing::layer().with_filter(LevelFilter::INFO))
        .init();

    if guard.is_enabled() {
        tracing::debug!("GlitchTip telemetry enabled");
    } else {
        tracing::debug!("GlitchTip telemetry disabled (SENTRY_DSN unset)");
    }

    guard
}

/// Parse arguments. Split from [`run`] so `main` can bring telemetry up with
/// the parsed verbosity flags *before* dispatching, while keeping the sentry
/// guard alive across the whole run.
pub fn parse() -> Cli {
    Cli::parse()
}

/// Dispatch a parsed command line. Called by `main`.
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Train(cmd) => train(cmd),
        Command::Sample(cmd) => sample(cmd),
        Command::Inspect(cmd) => inspect(cmd),
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
    // Metadata overrides (#154). An empty `--trigger-word` list means "not
    // passed", so the config's list stands — the same partial-override
    // semantics as every flag above.
    if !cmd.trigger_words.is_empty() {
        config.metadata.trigger_words = cmd.trigger_words.clone();
    }
    if cmd.no_metadata {
        config.metadata.embed = false;
    }
    Ok(config)
}

/// The metadata vocabularies `inspect` groups by, in reading order: what the
/// file is, how it was trained, then its identity hashes.
///
/// The prefixes are mutually exclusive (`sshs_model_hash` does not start with
/// `ss_` — the third byte is `h`, not `_`), so a key lands in at most one
/// group. The "other" bucket is `!any(starts_with)` over this same list, so a
/// fourth vocabulary is one edit here rather than two.
const GROUPS: [(&str, &str); 3] = [
    ("modelspec", "modelspec."),
    ("training (kohya ss_*)", "ss_"),
    ("hashes", "sshs_"),
];

/// Render a `.safetensors` file's `__metadata__` header.
///
/// Core reads it ([`loractl_core::read_metadata`] — header bytes only, no
/// tensors); this function only decides how it looks, which is the
/// core-emits/CLI-renders split the workspace is built around.
fn inspect(cmd: InspectCmd) -> Result<()> {
    let meta = loractl_core::read_metadata(&cmd.file)
        .with_context(|| format!("reading metadata from {}", cmd.file.display()))?;

    if cmd.json {
        println!(
            "{}",
            serde_json::to_string_pretty(meta.as_map()).context("rendering metadata as JSON")?
        );
        return Ok(());
    }

    if meta.is_empty() {
        println!(
            "{}: no __metadata__ header (diffusers scripts and minimal trainers write none)",
            cmd.file.display()
        );
        return Ok(());
    }

    println!("{}  ({} keys)", cmd.file.display(), meta.len());
    // Grouped by vocabulary, in the order a reader cares about: what it is,
    // then how it was trained, then the file's identity hashes. Anything with
    // an unknown prefix still prints, under "other" — never silently dropped.
    for (label, prefix) in GROUPS {
        let group = meta.with_prefix(prefix);
        if group.is_empty() {
            continue;
        }
        println!("\n{label}:");
        for (key, value) in group.iter() {
            println!("  {key} = {}", pretty_value(value));
        }
    }
    let others: Vec<(&str, &str)> = meta
        .iter()
        .filter(|(k, _)| !GROUPS.iter().any(|(_, prefix)| k.starts_with(prefix)))
        .collect();
    if !others.is_empty() {
        println!("\nother:");
        for (key, value) in others {
            println!("  {key} = {}", pretty_value(value));
        }
    }
    Ok(())
}

/// Values such as `ss_tag_frequency` are JSON *inside* a string; re-render
/// those compactly-parsed rather than as an escape-laden blob, and leave
/// everything else verbatim.
fn pretty_value(value: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(value) {
        Ok(v) if v.is_object() || v.is_array() => {
            serde_json::to_string(&v).unwrap_or_else(|_| value.to_string())
        }
        _ => value.to_string(),
    }
}

/// Render a phase's optional counters as a trailing ` 12/40 (30%)`, or the
/// empty string when the phase is uncountable.
fn phase_progress(done: Option<u64>, total: Option<u64>) -> String {
    match (done, total) {
        (Some(done), Some(total)) if total > 0 => {
            format!(" {done}/{total} ({}%)", done.saturating_mul(100) / total)
        }
        (Some(done), Some(total)) => format!(" {done}/{total}"),
        (Some(done), None) => format!(" {done}"),
        _ => String::new(),
    }
}

/// Decides which `Phase` updates earn a durable scrollback line at `-v`.
///
/// Core throttles a countable phase to ~100 reports, which is right for a
/// live bar but far too many log lines — quantizing 261 sites would bury the
/// terminal. So: a new phase always logs, a countable phase logs once per
/// completed decile (≤ 11 lines), and an uncountable phase logs whenever its
/// detail changes. At `-vv` (DEBUG) the throttle is bypassed by the caller's
/// `tracing::enabled!` check, so nothing is lost when it is actually wanted.
#[derive(Default)]
struct PhaseLog {
    /// Whether any phase has been logged yet (distinguishes "no phase" from
    /// a phase whose name happens to be empty).
    started: bool,
    /// Name of the last *logged* phase.
    name: String,
    /// Detail of the last *logged* update.
    detail: String,
    /// Decile (0..=10) of the last *logged* update, when countable.
    decile: Option<u64>,
}

impl PhaseLog {
    /// Returns `true` when this update deserves a log line, recording it as
    /// the new baseline. Skipped updates leave the baseline untouched, so the
    /// decile gate measures distance from the last line actually written.
    fn should_log(
        &mut self,
        name: &str,
        detail: &str,
        done: Option<u64>,
        total: Option<u64>,
    ) -> bool {
        // Everything is worth a line at DEBUG and above; the throttle exists
        // only to keep the default `-v` (INFO) ladder readable.
        let verbose = tracing::enabled!(tracing::Level::DEBUG);
        let decile = match (done, total) {
            (Some(done), Some(total)) if total > 0 => {
                Some((done.saturating_mul(10) / total).min(10))
            }
            _ => None,
        };
        let log = if verbose || !self.started || self.name != name {
            true
        } else if let Some(decile) = decile {
            Some(decile) != self.decile || done == total
        } else {
            self.detail != detail
        };
        if log {
            self.started = true;
            self.name = name.to_string();
            self.detail = detail.to_string();
            self.decile = decile;
        }
        log
    }
}

/// Where a throttled `Phase` line goes.
#[derive(Debug, PartialEq, Eq)]
enum PhaseSink {
    /// Through `tracing` at INFO, suspending the bar around the write.
    Log,
    /// Printed directly (the same stream the `fmt` layer renders to): there is
    /// no bar drawing this progress and the log level would otherwise swallow
    /// it.
    Print,
    /// The operator asked for silence (`-q`).
    Drop,
}

/// Route a phase line.
///
/// indicatif draws **nothing** when its draw target — **stderr**, where the
/// tracing fmt layer also writes — is not a terminal, which is exactly the
/// documented long-run route (a dispatched `gpu.yml`, `nohup … > train.log
/// 2>&1`). Note this keys on *stderr*: redirecting only stdout
/// (`… > train.log`) leaves stderr a TTY, so the bar still draws and carries
/// the progress itself. There the bar's animated spinner and
/// message carry no information at all, so a log that stays empty for tens of
/// minutes is the same "looks hung" failure the steady tick fixed for TTYs.
/// With no bar to carry it, the line is written regardless of `-v` — but `-q`
/// asked for silence and still gets it.
fn phase_sink(bar_hidden: bool, info_enabled: bool, warn_enabled: bool) -> PhaseSink {
    if info_enabled {
        PhaseSink::Log
    } else if bar_hidden && warn_enabled {
        PhaseSink::Print
    } else {
        PhaseSink::Drop
    }
}

/// The run's progress bar, drawn to `target`.
///
/// Split out of [`train`] so the two lines that carry the "it isn't hung"
/// signal — the steady tick and the initial message — are reachable from a
/// test against a capturing draw target. `train()` itself is never executed by
/// any test (the crate has no `[lib]`, so it cannot even be linked), which is
/// how a missing `enable_steady_tick` stayed invisible in the first place.
fn build_progress_bar(steps: u64, target: ProgressDrawTarget) -> ProgressBar {
    let bar = ProgressBar::with_draw_target(Some(steps.max(1)), target);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
        )
        .expect("valid progress template")
        .progress_chars("=>-"),
    );
    // Redraw on a timer, not only when an event arrives. Setup (a one-time
    // dataset encode, a 13 GB checkpoint load) can run for tens of minutes
    // before the first `Step`; without a steady tick the spinner and
    // `{elapsed_precise}` sit frozen at `00:00:00` and the run looks hung.
    // `suspend` pauses the ticker while a log line is written, so this does
    // not fight the `bar.suspend(...)` calls below.
    bar.set_message("starting…");
    bar.enable_steady_tick(Duration::from_millis(100));
    bar
}

/// Render one `TrainEvent` onto the bar. The whole of the CLI's half of the
/// event contract lives here so it can be driven directly by a test.
fn render_event(bar: &ProgressBar, phase_log: &mut PhaseLog, event: TrainEvent) {
    match event {
        TrainEvent::Started { total_steps } => bar.set_length(total_steps),
        TrainEvent::Phase {
            name,
            detail,
            counters,
        } => {
            let (done, total) = match counters {
                Some(c) => (Some(c.done), Some(c.total)),
                None => (None, None),
            };
            // Setup progress goes in the *message*, never the bar's
            // position/length — those belong to the step count (`Started`
            // sized the bar to `config.steps`) and re-purposing them would
            // corrupt step accounting for the rest of the run. The animated
            // spinner and elapsed timer carry liveness; the message carries
            // where we are.
            let message = format!("{name}: {detail}{}", phase_progress(done, total));
            bar.set_message(message.clone());
            if phase_log.should_log(name.as_str(), &detail, done, total) {
                match phase_sink(
                    bar.is_hidden(),
                    tracing::enabled!(tracing::Level::INFO),
                    tracing::enabled!(tracing::Level::WARN),
                ) {
                    PhaseSink::Log => bar.suspend(|| tracing::info!("{message}")),
                    // stderr, NOT stdout: the bar and the tracing fmt layer both
                    // write to stderr, so a `println!` here would make progress
                    // swap streams depending on `-v` — and would pollute the
                    // stdout that `adapter: <path>` reserves for the one
                    // machine-readable line this command emits.
                    PhaseSink::Print => eprintln!("{message}"),
                    PhaseSink::Drop => {}
                }
            }
        }
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
    }
}

fn train(cmd: TrainCmd) -> Result<()> {
    let config = resolve_config(&cmd)?;

    std::fs::create_dir_all(&config.output.dir)
        .with_context(|| format!("creating output dir {}", config.output.dir.display()))?;

    let bar = build_progress_bar(config.steps, ProgressDrawTarget::stderr());

    // Scrollback throttle for `Phase`, kept across the whole run — see
    // `PhaseLog`. Only the transient bar message is updated per event.
    let mut phase_log = PhaseLog::default();

    // The trainer factory — the constructor seam the load-bearing invariant
    // protects. Routing on `model.base` lives in core (`select_trainer`) so
    // the CLI and the API cannot drift apart.
    let mut trainer = select_trainer(&config);
    let adapter = trainer.train(&config, &mut |event| {
        render_event(&bar, &mut phase_log, event);
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
    use loractl_core::{BucketMode, PhaseCounters, PhaseName};

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
            trigger_words: Vec::new(),
            no_metadata: false,
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
    fn model_training_adapter_env_layer() {
        // #83: the optional `model.training_adapter` path is env-overridable via
        // figment's `__` nesting (no CLI flag), like `model.checkpoint`.
        Jail::expect_with(|jail| {
            jail.create_file("config.yaml", YAML)?;
            jail.set_env(
                "LORACTL_MODEL__TRAINING_ADAPTER",
                "/loras/assistant.safetensors",
            );
            let config = resolve_config(&cmd_for("config.yaml")).expect("resolve");
            assert_eq!(
                config.model.training_adapter,
                Some(std::path::PathBuf::from("/loras/assistant.safetensors")),
                "LORACTL_MODEL__TRAINING_ADAPTER must populate model.training_adapter"
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

    /// The #154 metadata flags. Both encode a deliberate asymmetry that is
    /// easy to "fix" into something else, so both are pinned:
    ///
    /// - `--trigger-word` **replaces** the config's list rather than
    ///   appending — a repeatable flag whose values merged with the file's
    ///   would make the file's entries unremovable from the command line.
    /// - `--no-metadata` is a bare `bool`, so it can only force `embed` OFF.
    ///   There is no `--metadata` to turn it back on, because `true` is
    ///   already the default; an `Option<bool>` here would be API surface
    ///   with no reachable second state.
    #[test]
    fn metadata_flags_override_the_file() {
        Jail::expect_with(|jail| {
            jail.create_file(
                "config.yaml",
                &format!("{YAML}metadata:\n  trigger_words: [from-file]\n  embed: true\n"),
            )?;

            // No flags: the file's list and `embed` stand.
            let config = resolve_config(&cmd_for("config.yaml")).expect("resolve");
            assert_eq!(config.metadata.trigger_words, vec!["from-file".to_string()]);
            assert!(config.metadata.embed);

            // Flags win, and the repeated values REPLACE rather than append.
            let mut cmd = cmd_for("config.yaml");
            cmd.trigger_words = vec!["sks dog".into(), "in the style of sks".into()];
            cmd.no_metadata = true;
            let config = resolve_config(&cmd).expect("resolve");
            assert_eq!(
                config.metadata.trigger_words,
                vec!["sks dog".to_string(), "in the style of sks".to_string()],
                "--trigger-word replaces the file's list"
            );
            assert!(!config.metadata.embed, "--no-metadata forces embed off");
            Ok(())
        });
    }

    /// `inspect`'s one piece of formatting logic: kohya's structured values
    /// are JSON *inside* a string, and must render as JSON rather than as an
    /// escape-laden blob — while a plain scalar is passed through untouched
    /// (a bare `16` parses as JSON but must not be "re-rendered").
    #[test]
    fn inspect_renders_nested_json_but_leaves_scalars_alone() {
        assert_eq!(
            pretty_value(r#"{"data": {"sks dog": 3}}"#),
            r#"{"data":{"sks dog":3}}"#
        );
        assert_eq!(pretty_value(r#"["sks dog"]"#), r#"["sks dog"]"#);
        assert_eq!(pretty_value("16"), "16", "a scalar is not JSON-normalized");
        assert_eq!(pretty_value("networks.lora"), "networks.lora");
        assert_eq!(pretty_value("(512, 512)"), "(512, 512)");
    }

    /// Every metadata key `inspect` can encounter must land in exactly one
    /// group, or in the explicit "other" bucket — never be dropped. The
    /// prefixes are mutually exclusive, which is what lets the "other" filter
    /// be the negation of the same list.
    #[test]
    fn inspect_groups_are_mutually_exclusive() {
        for key in [
            "modelspec.title",
            "ss_network_dim",
            "sshs_model_hash",
            "civitai_model_id",
        ] {
            let hits = GROUPS
                .iter()
                .filter(|(_, prefix)| key.starts_with(prefix))
                .count();
            assert!(hits <= 1, "{key} matched {hits} groups");
        }
        assert!(
            !"sshs_model_hash".starts_with("ss_"),
            "the hash prefix must not fall into the ss_ group"
        );
        assert!(
            !GROUPS
                .iter()
                .any(|(_, p)| "civitai_model_id".starts_with(p)),
            "an unknown vocabulary must reach the `other` bucket"
        );
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

    /// The verbosity ladder (#…: "the run looks hung and says nothing").
    ///
    /// The kill-test is the first assertion: the default MUST be WARN. It used
    /// to be `EnvFilter::from_default_env()`, which with `RUST_LOG` unset
    /// enables *nothing* — every honest `TrainEvent::Warning` was silently
    /// dropped. A regression to "off" would make this test fail rather than
    /// making the tool quietly lie.
    #[test]
    fn verbosity_flags_map_to_levels() {
        assert_eq!(
            level_for(0, false),
            LevelFilter::WARN,
            "no flags must still surface warnings"
        );
        assert_eq!(level_for(1, false), LevelFilter::INFO);
        assert_eq!(level_for(2, false), LevelFilter::DEBUG);
        assert_eq!(level_for(3, false), LevelFilter::TRACE);
        assert_eq!(
            level_for(9, false),
            LevelFilter::TRACE,
            "saturates at trace"
        );
        assert_eq!(level_for(0, true), LevelFilter::ERROR, "-q is errors only");
    }

    /// `RUST_LOG` is an explicit env override and wins over the flags — but
    /// only when it actually says something; an empty or whitespace value is
    /// treated as unset, so `RUST_LOG= loractl -v` still gets INFO instead of
    /// an empty (silent) filter.
    #[test]
    fn rust_log_overrides_the_flags_when_non_empty() {
        assert_eq!(
            resolve_verbosity(3, false, Some("loractl_core=debug,warn")),
            Verbosity::Env("loractl_core=debug,warn".to_string())
        );
        assert_eq!(
            resolve_verbosity(0, true, Some("info")),
            Verbosity::Env("info".to_string()),
            "RUST_LOG beats -q too"
        );
        assert_eq!(
            resolve_verbosity(1, false, Some("   ")),
            Verbosity::Level(LevelFilter::INFO),
            "a blank RUST_LOG is not an override"
        );
        assert_eq!(
            resolve_verbosity(0, false, None),
            Verbosity::Level(LevelFilter::WARN)
        );
    }

    /// `-v` and `-q` are contradictory, so clap must reject them together
    /// rather than letting one silently win. They are `global`, so this holds
    /// both before and after the subcommand.
    #[test]
    fn verbose_and_quiet_conflict() {
        Cli::command()
            .try_get_matches_from(["loractl", "-v", "train", "c.yaml"])
            .expect("-v alone is fine");
        Cli::command()
            .try_get_matches_from(["loractl", "-q", "train", "c.yaml"])
            .expect("-q alone is fine");
        Cli::command()
            .try_get_matches_from(["loractl", "-q", "-v", "train", "c.yaml"])
            .expect_err("-q and -v must conflict");
        Cli::command()
            .try_get_matches_from(["loractl", "train", "-v", "-q", "c.yaml"])
            .expect_err("the conflict must hold after the subcommand too");
    }

    /// Countable phases are throttled to one line per decile, so the 261-site
    /// quantize pass cannot bury the terminal at `-v`; a new phase name always
    /// gets its line, and an uncountable phase logs on each new detail.
    #[test]
    fn phase_logging_is_throttled_per_decile() {
        let mut log = PhaseLog::default();
        let lines = (0..=261)
            .filter(|site| log.should_log("quantize", "streaming sites", Some(*site), Some(261)))
            .count();
        assert!(
            (2..=12).contains(&lines),
            "261 site updates logged {lines} lines; expected ~11"
        );

        // A different phase always earns its line, even mid-count.
        assert!(log.should_log("inject", "14 LoRA adapters", None, None));
        assert!(
            !log.should_log("inject", "14 LoRA adapters", None, None),
            "an unchanged uncountable update must not repeat"
        );
        assert!(
            log.should_log("inject", "196 sites matched", None, None),
            "a new detail on an uncountable phase must log"
        );
    }

    /// The flag ladder must never become a *global* directive: with
    /// `tracing-log` bridging `log` records, a bare `debug` turns `-vv` on a
    /// GPU build into a wall of wgpu/naga output. The kill-test is the last
    /// assertion — a bare level must not satisfy this.
    #[test]
    fn flag_levels_are_scoped_to_loractls_own_target() {
        assert_eq!(flag_directives(LevelFilter::WARN), "warn,loractl=warn");
        assert_eq!(flag_directives(LevelFilter::INFO), "warn,loractl=info");
        assert_eq!(flag_directives(LevelFilter::DEBUG), "warn,loractl=debug");
        assert_eq!(flag_directives(LevelFilter::TRACE), "warn,loractl=trace");
        assert_eq!(
            flag_directives(LevelFilter::ERROR),
            "error,loractl=error",
            "-q silences third-party warnings too"
        );
        for level in [LevelFilter::INFO, LevelFilter::DEBUG, LevelFilter::TRACE] {
            let directives = flag_directives(level);
            assert!(
                directives.contains("loractl="),
                "{directives} is a global directive — third-party log records would flood it"
            );
            // …and it must parse, or every run would silently lose its filter.
            EnvFilter::try_new(&directives).expect("directives parse");
        }
    }

    /// Without a bar, the phase line is the *only* progress an operator sees —
    /// and every documented long run (ssh, `gpu.yml`, `nohup … > train.log`) is
    /// non-TTY, so gating it on `-v` there reproduces the "looks hung" symptom
    /// in a log file. `-q` still wins.
    #[test]
    fn phase_lines_survive_a_hidden_bar_at_the_default_level() {
        // Bar drawn (a TTY): the bar carries progress, so the default level
        // keeps the scrollback quiet.
        assert_eq!(phase_sink(false, false, true), PhaseSink::Drop);
        assert_eq!(phase_sink(false, true, true), PhaseSink::Log);
        // No bar: the line must get out even though INFO is off.
        assert_eq!(
            phase_sink(true, false, true),
            PhaseSink::Print,
            "a redirected run must still report setup progress"
        );
        assert_eq!(phase_sink(true, true, true), PhaseSink::Log);
        // `-q` asked for silence, bar or not.
        assert_eq!(phase_sink(true, false, false), PhaseSink::Drop);
        assert_eq!(phase_sink(false, false, false), PhaseSink::Drop);
    }

    /// A capturing [`indicatif::TermLike`] so the bar's actual draws are
    /// observable — the only way to test the two lines that carry "this run is
    /// not hung", since `train()` itself is unreachable from any test.
    #[derive(Clone, Debug, Default)]
    struct CaptureTerm(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

    impl CaptureTerm {
        fn writes(&self) -> usize {
            self.0.lock().unwrap().len()
        }
        fn text(&self) -> String {
            self.0.lock().unwrap().join("\n")
        }
        fn push(&self, s: &str) {
            self.0.lock().unwrap().push(s.to_string());
        }
    }

    impl indicatif::TermLike for CaptureTerm {
        fn width(&self) -> u16 {
            120
        }
        fn height(&self) -> u16 {
            40
        }
        fn move_cursor_up(&self, _: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn move_cursor_down(&self, _: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn move_cursor_right(&self, _: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn move_cursor_left(&self, _: usize) -> std::io::Result<()> {
            Ok(())
        }
        fn write_line(&self, s: &str) -> std::io::Result<()> {
            self.push(s);
            Ok(())
        }
        fn write_str(&self, s: &str) -> std::io::Result<()> {
            self.push(s);
            Ok(())
        }
        fn clear_line(&self) -> std::io::Result<()> {
            Ok(())
        }
        fn flush(&self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn capturing_bar(steps: u64) -> (ProgressBar, CaptureTerm) {
        let term = CaptureTerm::default();
        let bar = build_progress_bar(steps, ProgressDrawTarget::term_like(Box::new(term.clone())));
        (bar, term)
    }

    /// The frozen-bar fix. Setup can run for tens of minutes before the first
    /// `Step`; without `enable_steady_tick` the spinner and `{elapsed_precise}`
    /// never redraw and the run looks hung — the original symptom.
    #[test]
    fn the_bar_redraws_on_a_timer_with_no_events() {
        let (bar, term) = capturing_bar(10);
        // Baseline *after* construction, so the setup draws don't count: the
        // question is whether anything is drawn while nothing happens.
        let before = term.writes();
        std::thread::sleep(Duration::from_millis(400));
        let ticks = term.writes() - before;
        bar.finish_and_clear();
        assert!(
            ticks >= 2,
            "the bar must redraw on a timer with no events (got {ticks} draws in 400ms)"
        );
    }

    /// A `Phase` lands in the bar's *message*, never its position/length.
    #[test]
    fn phase_events_render_into_the_bar_message() {
        let (bar, term) = capturing_bar(10);
        let mut log = PhaseLog::default();
        render_event(
            &bar,
            &mut log,
            TrainEvent::Phase {
                name: PhaseName::Encode,
                detail: "encoding a.png".into(),
                counters: Some(PhaseCounters::new(3, 40)),
            },
        );
        assert_eq!(bar.position(), 0, "a phase must not move the step count");
        assert_eq!(bar.length(), Some(10), "a phase must not resize the bar");
        bar.finish_and_clear();
        assert!(
            term.text().contains("encode: encoding a.png 3/40 (7%)"),
            "phase message missing from the bar: {}",
            term.text()
        );
    }

    #[test]
    fn phase_progress_renders_counters_only_when_present() {
        assert_eq!(phase_progress(Some(3), Some(40)), " 3/40 (7%)");
        assert_eq!(phase_progress(Some(40), Some(40)), " 40/40 (100%)");
        assert_eq!(phase_progress(Some(7), None), " 7");
        assert_eq!(phase_progress(None, Some(40)), "");
        assert_eq!(phase_progress(None, None), "");
        assert_eq!(
            phase_progress(Some(0), Some(0)),
            " 0/0",
            "no divide by zero"
        );
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

    /// The #147/#148 bucketing knobs deliberately ship with **no CLI flag**
    /// (the `model.checkpoint` / `flow.shift_mode` precedent: a per-dataset
    /// choice, not a per-invocation one), so the env layer is half their
    /// entire override surface and nothing else would catch it silently
    /// breaking. Also pins the case-insensitive `FromStr` route the
    /// hand-written `Deserialize` goes through — a `#[derive(Deserialize)]`
    /// would reject `GRID` and only this asserts otherwise.
    #[test]
    fn env_layer_reaches_dataset_bucketing_knobs() {
        Jail::expect_with(|jail| {
            jail.create_file("config.yaml", YAML)?;
            jail.set_env("LORACTL_DATASET__NO_UPSCALE", "true");
            jail.set_env("LORACTL_DATASET__BUCKETING", "grid");
            jail.set_env("LORACTL_DATASET__MIN_BUCKET_RESOLUTION", "256");

            let config = resolve_config(&cmd_for("config.yaml")).expect("resolve");
            assert!(config.dataset.no_upscale);
            assert_eq!(config.dataset.bucketing, BucketMode::Grid);
            assert_eq!(config.dataset.min_bucket_resolution, Some(256));

            // The spelling is case-insensitive, exactly like `--backend`.
            jail.set_env("LORACTL_DATASET__BUCKETING", "GRID");
            let config = resolve_config(&cmd_for("config.yaml")).expect("resolve");
            assert_eq!(config.dataset.bucketing, BucketMode::Grid);

            // …and an unknown spelling is a clear error, not a silent default.
            jail.set_env("LORACTL_DATASET__BUCKETING", "gird");
            let err = resolve_config(&cmd_for("config.yaml")).expect_err("typo must fail");
            assert!(
                format!("{err:#}").contains("aspects|grid"),
                "error should name the vocabulary: {err:#}"
            );
            Ok(())
        });
    }

    /// The YAML file layer, for the same three knobs — and the default when
    /// the block omits them (every pre-#147 config on disk).
    #[test]
    fn dataset_bucketing_knobs_default_when_the_file_omits_them() {
        Jail::expect_with(|jail| {
            jail.create_file("config.yaml", YAML)?;
            let config = resolve_config(&cmd_for("config.yaml")).expect("resolve");
            assert!(!config.dataset.no_upscale);
            assert_eq!(config.dataset.bucketing, BucketMode::Aspects);
            assert_eq!(config.dataset.min_bucket_resolution, None);

            jail.create_file(
                "grid.yaml",
                "steps: 10\n\
                 model:\n  base: synthetic\n\
                 dataset:\n  path: unused\n  no_upscale: true\n  bucketing: grid\n\
                 \x20 min_bucket_resolution: 256\n\
                 lora: {}\n",
            )?;
            let config = resolve_config(&cmd_for("grid.yaml")).expect("resolve");
            assert!(config.dataset.no_upscale);
            assert_eq!(config.dataset.bucketing, BucketMode::Grid);
            assert_eq!(config.dataset.min_bucket_resolution, Some(256));
            Ok(())
        });
    }
}
