//! Step-throughput bench (#110) — how long does one REAL training step take?
//!
//! The sibling of `examples/step_probe.rs`: that one answers *does this config
//! fit the card*, this one answers *how fast does it run*. Both drive the same
//! `select_trainer` → `Trainer::train` path a real run takes, so neither
//! measures a reconstruction of training — they measure training.
//!
//! It exists because loractl adopted every memory lever so far (int8/int4 base
//! quantization, #134 block checkpointing) on a memory argument alone, with no
//! way to price what they cost in time. Two live questions need that price:
//! whether int8/int4 QLoRA on the numerically-clean `cuda f32` path reaches a
//! usable step time on the 24 GB 4090 (#96), and what block-boundary
//! activation offload would spend to buy its VRAM back (#158, ADR-0008 — which
//! cannot merge until this can answer it).
//!
//! ## Output
//!
//! Grep-parseable lines from the `loractl-bench` schema:
//!
//! - `MODEL label=… …` — the analytic work model behind the derived
//!   throughputs. `step_flops=` is the actual numerator `tflops=` was divided
//!   by, so it can be checked against `ms=` without reassembling the component
//!   terms; `excludes=` lists what the count leaves out (all of it in the
//!   under-count direction) and `assumes=` what it takes on faith. Absent when
//!   nothing is modelled.
//! - `RESULT label=<label> ms=… tok_s=… tflops=… step=… loss=… ckpt=…
//!   counted=… vram_mib=…` — one per timed step window. `counted=0` marks a
//!   window the aggregate dropped, and `ckpt=1` says it was dropped for a
//!   checkpoint export rather than as warm-up; averaging the per-step lines
//!   without filtering on `counted=1` will not match the `_median` line.
//! - `RESULT label=<label>_median ms=… tok_s=… tflops=… steps_counted=…
//!   steps_timed=… plausible=… vram_peak_mib=… sanity=… x2_ratio=…` — the
//!   aggregate, and the line to quote: median step time over the counted
//!   windows, peak VRAM, and the 2×-steps dead-graph verdict.
//! - `SANITY x2_iters_ratio=… verdict=…` — the dead-graph check.
//!
//! `ms=` and `vram_mib=` are measured. `tok_s=` and `tflops=` are quotients of
//! the `MODEL` line's work count — read that line before quoting either.
//!
//! A `sanity=SUSPECT` or `plausible=false` on the aggregate voids the timings:
//! a run whose later steps went free, or whose loss was zero or non-finite,
//! was not doing the work being divided by.
//!
//! ## Usage
//!
//! The config is read from the YAML file ONLY (no `LORACTL_` env layering) —
//! same rule as `step_probe`: a measurement run should be fully described by
//! the file plus the flags. The backend comes from `compute.backend`, so the
//! same binary times the offline ndarray fixture and the real cuda run:
//!
//! ```text
//! cargo run --release -p loractl-core --example bench_step -- <config.yaml>
//! cargo run --release -p loractl-core --features cuda --example bench_step -- \
//!   config/examples/krea2-comfyui.yaml --steps 8 --seq-len 1536
//! ```
//!
//! Flags: `--steps N` (overrides `steps`), `--warmup N` (leading windows
//! excluded from the aggregate, default 1), `--seq-len N` (the trunk's real
//! sequence length — see below), `--label NAME` (the `RESULT label=`; must be
//! whitespace-free, since the lines are space-delimited).
//!
//! ## Why `--seq-len` is worth passing
//!
//! `tok_s=`/`tflops=` need the trunk's token count. The image half is exactly
//! derivable from the config — `(resolution / 8 / patch)²`, the VAE's f8
//! downsample then the patch grid — but the text half is not: caption length
//! is data, and the trunk pads the *combined* sequence to a multiple of 256.
//! So without `--seq-len` this reports the image-only count and labels it
//! `seq_len_source=derived_image_only`, which **under**-counts a real run (at
//! 512px: 1024 image tokens against the ~1536 an actual Krea 2 step runs).
//! Pass the true sequence length to make the throughputs exact; the flag is
//! recorded as `seq_len_source=declared` so a reader can tell which they have.
//! `ms=` is unaffected either way — it is measured, not modelled.
//!
//! Not a numerics-golden target and never run in CI: the real measurement
//! needs multi-GB weights and a real GPU. The offline fixture path exists so
//! the harness itself stays exercised without one.

use anyhow::{Context, Result};
use figment::Figment;
use figment::providers::{Format, Yaml};
use loractl_core::bench::{StepBench, StepWork};
use loractl_core::{TrainConfig, TrainEvent, select_trainer};
use std::path::PathBuf;

/// The parsed command line.
struct Args {
    config: PathBuf,
    steps: Option<u64>,
    warmup: usize,
    seq_len: Option<usize>,
    label: String,
}

fn parse_args() -> Result<Args> {
    let mut config: Option<PathBuf> = None;
    let (mut steps, mut seq_len) = (None, None);
    let mut warmup = 1usize;
    let mut label = "train_step".to_string();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        let mut value = |flag: &str| -> Result<String> {
            args.next().with_context(|| format!("{flag} needs a value"))
        };
        match arg.as_str() {
            "--steps" => {
                let raw = value("--steps")?;
                steps =
                    Some(raw.parse().with_context(|| {
                        format!("--steps expects a positive integer, got {raw:?}")
                    })?);
            }
            "--warmup" => {
                let raw = value("--warmup")?;
                warmup = raw.parse().with_context(|| {
                    format!("--warmup expects a non-negative integer, got {raw:?}")
                })?;
            }
            "--seq-len" => {
                let raw = value("--seq-len")?;
                let n: usize = raw.parse().with_context(|| {
                    format!("--seq-len expects a positive integer, got {raw:?}")
                })?;
                // Rejected here rather than handled downstream: a declared 0
                // is a typo, and the work model would otherwise have to
                // explain a degenerate denominator the caller asked for.
                if n == 0 {
                    anyhow::bail!("--seq-len must be positive (0 tokens/step models nothing)");
                }
                seq_len = Some(n);
            }
            // Rejected rather than mangled: the label is the ONE
            // caller-controlled token in a schema whose entire purpose is
            // whitespace-delimited grep-parseability, and `label=int4 512px`
            // reads as `label=int4` plus a stray token to any parser — in the
            // very log that gets pasted into #96/#158. Silently substituting
            // underscores would mean the label in the output isn't the one
            // that was asked for, so say so instead.
            "--label" => {
                label = value("--label")?;
                if label.is_empty() || label.contains(char::is_whitespace) {
                    anyhow::bail!(
                        "--label must be non-empty and whitespace-free (the RESULT/MODEL \
                         lines are space-delimited), got {label:?}"
                    );
                }
            }
            other if other.starts_with("--") => anyhow::bail!(
                "unknown flag {other} (expected --steps, --warmup, --seq-len, --label)"
            ),
            _ => {
                if config.is_some() {
                    anyhow::bail!(
                        "unexpected second positional argument {arg:?} — one config path"
                    );
                }
                config = Some(PathBuf::from(arg));
            }
        }
    }

    Ok(Args {
        config: config.context("arg 1: path to a TrainConfig YAML")?,
        steps,
        warmup,
        seq_len,
        label,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;

    let mut config: TrainConfig = Figment::new()
        .merge(Yaml::file(&args.config))
        .extract()
        .with_context(|| format!("loading config {}", args.config.display()))?;
    if let Some(steps) = args.steps {
        config.steps = steps;
    }

    // The aggregate needs one warm-up window plus at least two counted ones,
    // and every timed window is bounded by a Step event on each side — so the
    // run needs `warmup + 3` steps before it can say anything. Say so up
    // front rather than printing an empty summary at the end.
    // `warmup + 3` is the structural floor, but it is the *weakest* run that
    // reports anything: at 2 counted windows `sanity()`'s baseline is a single
    // sample, so one scheduling hiccup swings a verdict that is documented as
    // voiding the timings. `warmup + 7` gives a 3-window baseline, which is
    // why the `just bench` example passes `--steps 8`.
    let needed = args.warmup as u64 + 3;
    let comfortable = args.warmup as u64 + 7;
    if config.steps < needed {
        println!(
            "note: steps={} is below the {needed} this bench needs \
             (1 window is consumed opening the first, {} are warm-up, \
              and the 2×-steps sanity check needs 2 counted) — \
             the aggregate may be absent or weak",
            config.steps, args.warmup,
        );
    } else if config.steps < comfortable {
        println!(
            "note: steps={} meets the {needed}-step floor but leaves the 2×-steps \
             sanity baseline at 1–2 windows, where one scheduling hiccup can read \
             SUSPECT — prefer --steps {comfortable} or more for a quotable run",
            config.steps,
        );
    }

    let (work, note) = StepWork::for_config(&config, args.seq_len);
    println!(
        "config {} — backend {:?}, precision {:?}, quant {:?}, grad_ckpt {}",
        args.config.display(),
        config.compute.backend,
        config.compute.precision,
        config.compute.quant,
        config.compute.grad_checkpointing,
    );
    println!("{note}");
    println!("steps {}, warmup windows {}", config.steps, args.warmup);

    let mut bench = StepBench::new(args.label.as_str(), work).with_warmup_steps(args.warmup);
    let result = {
        let mut sink = |event: TrainEvent| {
            bench.record(&event);
            match &event {
                TrainEvent::Started { total_steps } => println!("started: {total_steps} steps"),
                TrainEvent::Warning { message } => println!("  warning: {message}"),
                TrainEvent::Finished { adapter_path } => {
                    println!("finished: adapter {}", adapter_path.display());
                }
                // Steps, checkpoints and samples all surface as RESULT lines
                // below — printing them twice would just make the log harder
                // to grep.
                _ => {}
            }
        };
        select_trainer(&config).train(&config, &mut sink)
    };

    // Print what was measured even when the run then failed: on a config that
    // OOMs at step 40, the steps before it are the measurement.
    for line in bench.result_lines() {
        println!("{line}");
    }
    if bench.summary_line().is_none() {
        println!(
            "no aggregate: {} step windows were timed, none survived warm-up \
             and checkpoint filtering",
            bench.samples().len()
        );
    }

    result.map(|_| ())
}
