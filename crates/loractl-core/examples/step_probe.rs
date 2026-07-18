//! On-box step-VRAM probe (ADR-0005) — measure what a few REAL
//! [`DiffusionTrainer`](loractl_core::DiffusionTrainer) training steps cost in
//! peak resident VRAM, per LoRA target set. ADR-0005 established that the
//! int4 real-model step OOMs on genuine VRAM exhaustion driven by
//! resolution-INDEPENDENT dequant/gradient buffers that scale with the number
//! of TRAINED SITES — so the prescribed measurement is exactly this: sweep
//! `lora.targets` and watch the step peak. It reports:
//!
//! 1. **Site accounting** — how many of the architecture's injectable sites
//!    the (final) target list matches, of the total, computed from
//!    [`MmditConfig::for_variant`] alone (no 12.8B model is built for the
//!    count). Matching semantics are identical to `build_adapters`:
//!    unanchored `Regex::is_match` on each site path.
//! 2. **Live VRAM telemetry** — a watcher thread polls `nvidia-smi` every
//!    ~200 ms into a running max and prints a `vram peak so far:` line every
//!    time the peak ratchets ≥ 256 MiB. This matters because a genuine OOM
//!    aborts the process (cubecl panics the device thread), so on the failing
//!    runs the **last printed ratchet line IS the measurement** — the final
//!    summary never gets a chance to print.
//! 3. **The run itself** — real `select_trainer` → `Trainer::train` events:
//!    per-step loss with the resident VRAM at that step, warnings indented.
//! 4. **A greppable summary** — one `STEP_PROBE_SUMMARY` line (targets,
//!    matched/total sites, completed/requested steps, baseline and peak MiB)
//!    plus a fit verdict; printed on failure too, with the steps completed so
//!    far.
//!
//! The config is read from the YAML file ONLY (no `LORACTL_` env layering —
//! a probe run should be fully described by the file plus the flags below).
//! `--target` (repeatable) REPLACES `lora.targets` wholesale; `--steps N`
//! overrides `steps`. Encode-cache note: the dataset cache key ignores
//! `lora.targets` and `steps` (see `dataset.rs`), so a target sweep re-uses a
//! warm cache and never re-encodes; only a cold cache pays the (slow, CPU)
//! encode phase first.
//!
//! Usage (on the Linux + NVIDIA host; `--features cuda` for the real runs —
//! the backend comes from the config's `compute.backend`, so the same binary
//! drives the offline ndarray fixture too):
//!   cargo run --release -p loractl-core --features cuda --example step_probe -- \
//!     config/examples/krea2-comfyui.yaml [--target <regex>]... [--steps N]
//!
//! Not a numerics-golden target and never run in CI — the real measurement
//! needs multi-GB weights and a 24 GB GPU. It is the ADR-0005 sweep tool
//! behind the #96/#25 target-set decision.

use anyhow::{Context, Result};
use figment::Figment;
use figment::providers::{Format, Yaml};
use loractl_core::config::TargetSpec;
use loractl_core::mmdit::MmditConfig;
use loractl_core::{TrainConfig, TrainEvent, select_trainer};
use regex::Regex;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Resident VRAM (MiB) on device 0, via `nvidia-smi`. Best-effort — a
/// missing tool degrades to `None`, not a probe failure.
fn resident_vram_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

fn main() -> Result<()> {
    // Args: the config path (first positional), repeatable `--target <regex>`
    // (collected in order; if any are given they REPLACE config.lora.targets
    // wholesale), and `--steps N` (overrides config.steps).
    let mut config_path: Option<PathBuf> = None;
    let mut targets: Vec<String> = Vec::new();
    let mut steps_override: Option<u64> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--target" => {
                targets.push(args.next().context("--target needs a regex value")?);
            }
            "--steps" => {
                let val = args.next().context("--steps needs a value")?;
                steps_override =
                    Some(val.parse().with_context(|| {
                        format!("--steps expects a positive integer, got {val:?}")
                    })?);
            }
            other if other.starts_with("--") => {
                anyhow::bail!("unknown flag {other} (expected --target <regex>, --steps <n>)");
            }
            _ => {
                if config_path.is_some() {
                    anyhow::bail!(
                        "unexpected second positional argument {arg:?} — only one config path \
                         is accepted (targets go through --target)"
                    );
                }
                config_path = Some(PathBuf::from(arg));
            }
        }
    }
    let config_path = config_path.context("arg 1: path to a TrainConfig YAML")?;

    // File-only config load (deliberately no Env provider — see the header).
    let mut config: TrainConfig = Figment::new()
        .merge(Yaml::file(&config_path))
        .extract()
        .with_context(|| format!("loading config {}", config_path.display()))?;
    if !targets.is_empty() {
        config.lora.targets = targets
            .iter()
            .map(|pattern| TargetSpec {
                pattern: pattern.clone(),
                rank: None,
                alpha: None,
            })
            .collect();
    }
    if let Some(steps) = steps_override {
        config.steps = steps;
    }

    // Site accounting from the config alone — never builds the model. Same
    // matching semantics as adapters::build_adapters: a site trains iff any
    // pattern is_match-es its path (unanchored).
    let sites = MmditConfig::for_variant(config.model.variant).injectable_sites();
    let compiled: Vec<Regex> = config
        .lora
        .targets
        .iter()
        .map(|spec| {
            Regex::new(&spec.pattern)
                .with_context(|| format!("invalid LoRA target pattern {:?}", spec.pattern))
        })
        .collect::<Result<_>>()?;
    let matched = sites
        .iter()
        .filter(|site| compiled.iter().any(|re| re.is_match(&site.path)))
        .count();
    let joined_targets = config
        .lora
        .targets
        .iter()
        .map(|spec| spec.pattern.as_str())
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "targets [{joined_targets}] match {matched}/{} injectable sites ({:?})",
        sites.len(),
        config.model.variant
    );

    // Baseline, then the ratcheting peak watcher (see the header for why the
    // watcher prints: on a genuine OOM the process aborts and the last
    // ratchet line is the measurement).
    let baseline = resident_vram_mib();
    match baseline {
        Some(mib) => println!("baseline resident VRAM: {mib} MiB"),
        None => println!("baseline resident VRAM: nvidia-smi unavailable (poller idles)"),
    }
    let peak = Arc::new(AtomicU64::new(baseline.unwrap_or(0)));
    let stop = Arc::new(AtomicBool::new(false));
    let poller = {
        let peak = Arc::clone(&peak);
        let stop = Arc::clone(&stop);
        let enabled = baseline.is_some();
        std::thread::spawn(move || {
            let mut last_printed = peak.load(Ordering::Relaxed);
            while !stop.load(Ordering::Relaxed) {
                if enabled && let Some(now) = resident_vram_mib() {
                    let prev = peak.fetch_max(now, Ordering::Relaxed).max(now);
                    if prev >= last_printed + 256 {
                        println!("vram peak so far: {prev} MiB");
                        last_printed = prev;
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        })
    };

    // Drive the real path: the same factory + Trainer contract the CLI uses.
    let requested_steps = config.steps;
    let mut completed_steps: u64 = 0;
    let result = {
        let mut sink = |event: TrainEvent| match event {
            TrainEvent::Started { total_steps } => {
                println!("started: {total_steps} steps planned");
            }
            TrainEvent::Step { step, loss, .. } => {
                completed_steps = step;
                match resident_vram_mib() {
                    Some(mib) => println!("step {step} loss {loss:.6} resident-now {mib} MiB"),
                    None => println!("step {step} loss {loss:.6}"),
                }
            }
            TrainEvent::Warning { message } => println!("  warning: {message}"),
            TrainEvent::Checkpoint { step, path } => {
                println!("checkpoint at step {step}: {}", path.display());
            }
            TrainEvent::Sample { step, path } => {
                println!("sample at step {step}: {}", path.display());
            }
            TrainEvent::Finished { adapter_path } => {
                println!("finished: adapter {}", adapter_path.display());
            }
        };
        select_trainer(&config).train(&config, &mut sink)
    };

    stop.store(true, Ordering::Relaxed);
    let _ = poller.join();

    // The summary prints on success AND on a survivable error (a hard OOM
    // abort never reaches here — the poller's ratchet lines cover that case).
    let peak = peak.load(Ordering::Relaxed);
    let (baseline_label, peak_label) = match baseline {
        Some(mib) => (mib.to_string(), peak.to_string()),
        None => ("unavailable".into(), "unavailable".into()),
    };
    match (baseline, &result) {
        (Some(base), Ok(_)) => println!(
            "fit verdict: completed {completed_steps}/{requested_steps} steps with a peak of \
             {peak} MiB ({} MiB above baseline) — this target set fits this card",
            peak.saturating_sub(base)
        ),
        (Some(_), Err(_)) => println!(
            "fit verdict: the run FAILED after {completed_steps}/{requested_steps} steps — \
             see the error below and the last `vram peak so far` ratchet"
        ),
        (None, _) => println!(
            "fit verdict: VRAM telemetry unavailable (no nvidia-smi) — step/loss telemetry only"
        ),
    }
    println!(
        "STEP_PROBE_SUMMARY targets={joined_targets} sites={matched}/{} \
         steps={completed_steps}/{requested_steps} baseline_mib={baseline_label} \
         peak_mib={peak_label}",
        sites.len()
    );

    result.map(|_| ())
}
