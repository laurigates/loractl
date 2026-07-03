//! The `loractl` command-line surface.
//!
//! This module is a *renderer* over `loractl-core`: it parses arguments,
//! layers config sources, drives a [`Trainer`], and turns the
//! [`TrainEvent`]s it emits into terminal output. It contains no training
//! logic — swapping `MockTrainer` for a burn-backed trainer later touches
//! only the one line that constructs it.

use anyhow::{Context, Result};
use clap::{Args, CommandFactory, Parser, Subcommand};
use clap_complete::Shell;
use figment::{
    Figment,
    providers::{Env, Format, Yaml},
};
use indicatif::{ProgressBar, ProgressStyle};
use loractl_core::{MockTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::{Path, PathBuf};

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

    /// Generate a sample from a trained adapter. (not yet implemented)
    Sample(SampleCmd),

    /// Print shell completions to stdout (e.g. `loractl completions zsh`).
    Completions {
        /// Shell to generate completions for.
        shell: Shell,
    },
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
}

#[derive(Args)]
struct SampleCmd {
    /// Path to the trained adapter (`.safetensors`).
    adapter: PathBuf,

    /// Prompt to render.
    #[arg(short, long)]
    prompt: Option<String>,
}

/// Parse arguments and dispatch. Called by `main`.
pub fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    match Cli::parse().command {
        Command::Train(cmd) => train(cmd),
        Command::Sample(cmd) => sample(cmd),
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

fn train(cmd: TrainCmd) -> Result<()> {
    let mut config = load_config(&cmd.config)?;
    if let Some(lr) = cmd.lr {
        config.optim.lr = lr;
    }
    if let Some(steps) = cmd.steps {
        config.steps = steps;
    }

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

    let mut trainer = MockTrainer;
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
        TrainEvent::Warning(msg) => {
            bar.suspend(|| tracing::warn!("{msg}"));
        }
        TrainEvent::Finished { adapter_path } => {
            bar.finish_with_message(format!("done → {}", adapter_path.display()));
        }
    })?;

    println!("adapter: {}", adapter.display());
    Ok(())
}

fn sample(cmd: SampleCmd) -> Result<()> {
    anyhow::bail!(
        "`sample` is not implemented yet (arrives in milestone 2). \
         adapter={}, prompt={:?}",
        cmd.adapter.display(),
        cmd.prompt,
    )
}
