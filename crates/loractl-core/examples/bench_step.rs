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
use loractl_core::mmdit::MmditConfig;
use loractl_core::{TrainConfig, TrainEvent, is_builtin_demo_base, select_trainer};
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
                seq_len = Some(raw.parse().with_context(|| {
                    format!("--seq-len expects a positive integer, got {raw:?}")
                })?);
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

/// The work model for this config, plus a human note about where the token
/// count came from.
///
/// Only the diffusion path is modelled: `BurnTrainer`'s synthetic/mnist demo
/// is a LoRA-MLP whose "tokens" mean nothing comparable, so it reports time
/// and VRAM and claims no throughput — the honest output for a run whose
/// denominator does not exist.
///
/// The which-trainer test is `is_builtin_demo_base`, the same predicate
/// `select_trainer` routes on, rather than a second copy of its match arm — a
/// stale copy here would model MMDiT work for a non-MMDiT trainer and show up
/// only as a wrong `tok_s=`.
fn work_model(config: &TrainConfig, seq_len_flag: Option<usize>) -> (StepWork, String) {
    if is_builtin_demo_base(&config.model.base) {
        return (
            StepWork::unmodelled(),
            "work model: none — the synthetic/mnist demo has no comparable token \
             geometry, so only ms= and vram_mib= are reported"
                .to_string(),
        );
    }

    let cfg = MmditConfig::for_variant(config.model.variant);
    let batch = config.dataset.batch_size as usize;
    // The image half, exactly: the VAE's f8 downsample, then the patch grid.
    let latent = (config.dataset.resolution as usize) / 8;
    let image_tokens = (latent / cfg.patch).pow(2);
    let (seq_len, source) = match seq_len_flag {
        Some(n) => (n, "declared"),
        None => (image_tokens, "derived_image_only"),
    };

    // A zero denominator is worse than an absent one: `tok_s=0.0000
    // tflops=0.0000` reads as a measurement rather than as "no model". The
    // derivation floors twice (`resolution/8` then `/patch`), so any
    // resolution below `8·patch` lands here. Unreachable from the shipped
    // configs, but the whole point of the MODEL line is that a denominator is
    // auditable, and one that is silently zero is neither.
    if seq_len == 0 || batch == 0 {
        return (
            StepWork::unmodelled(),
            format!(
                "work model: none — {batch} × {seq_len} tokens/step is degenerate \
                 (resolution {} over an f8 VAE and patch {} gives {image_tokens} image \
                 tokens); reporting ms= and vram_mib= only. Pass --seq-len to model it.",
                config.dataset.resolution, cfg.patch,
            ),
        );
    }

    let note = format!(
        "work model: {batch} × {seq_len} tokens/step ({source}); \
         {image_tokens} image tokens derived from resolution {} \
         (latent {latent}, patch {}), assuming the square bucket and a full batch",
        config.dataset.resolution, cfg.patch,
    );
    let work = StepWork::mmdit_step(&cfg, batch, seq_len, config.compute.grad_checkpointing)
        .with_term("seq_len_source", source)
        .with_term("image_tokens", image_tokens)
        // Recorded, not assumed away: the image-token derivation takes Krea
        // 2's f8 VAE. A bundle with a different downsample makes the derived
        // count wrong, and `--seq-len` is then the only correct route.
        .with_term("vae_downsample", 8)
        // Two more assumptions worth naming rather than burying. `dataset.rs`
        // buckets by aspect ratio (area-preserving, but 16-px aligned), so the
        // square derivation is approximate; and `batches()` chunks per bucket,
        // so a bucket's last batch can be smaller than `batch_size` — those
        // steps process fewer tokens than modelled, and `tok_s=`/`tflops=`
        // over-report for them.
        .with_term("assumes", "square_bucket,full_batch");
    (work, note)
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
    let needed = args.warmup as u64 + 3;
    if config.steps < needed {
        println!(
            "note: steps={} is below the {needed} this bench needs \
             (1 window is consumed opening the first, {} are warm-up, \
              and the 2×-steps sanity check needs 2 counted) — \
             the aggregate may be absent or weak",
            config.steps, args.warmup,
        );
    }

    let (work, note) = work_model(&config, args.seq_len);
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
