//! The step-throughput harness against real `Trainer`s (#110).
//!
//! `bench.rs`'s unit tests pin the window accounting against scripted event
//! streams. This suite asks the question those cannot: does the observer work
//! when a *real* trainer drives it, through the same
//! `Trainer::train(&config, &mut sink)` contract the CLI and the API use, from
//! outside the crate?
//!
//! Two trainers, for two different claims:
//!
//! - [`MockTrainer`] — no ML at all, so it isolates the *seam*: an integration
//!   consumer can wrap a sink, and a checkpoint export really does land inside
//!   a timed window and get marked. Deterministic and instant.
//! - [`BurnTrainer`] on its synthetic path — real burn tensors, a real
//!   backward, a real optimizer step. This is the one that proves the timings
//!   describe compute: the windows are positive, they are the same order of
//!   magnitude as each other, and the 2×-steps ratio is non-degenerate.
//!
//! Both stay on the always-compiled `ndarray` backend, so this suite is
//! offline and GPU-free. What it deliberately does NOT assert is any
//! particular speed — a wall-clock threshold on shared CI hardware is a
//! flake, and the harness's job is to report a number honestly, not to hit one.

use loractl_core::bench::{StepBench, StepWork};
use loractl_core::config::{OutputConfig, TaskKind};
use loractl_core::{BurnTrainer, MockTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::PathBuf;
use std::time::Duration;

/// A unique temp output dir, removed on drop, so concurrent runs never collide.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Self(std::env::temp_dir().join(format!("loractl-{tag}-{}-{nanos}", std::process::id())))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run `trainer` under a `StepBench` and hand back the bench for inspection —
/// the exact shape `examples/bench_step.rs` uses.
fn benched(
    trainer: &mut dyn Trainer,
    config: &TrainConfig,
    work: StepWork,
    warmup: usize,
) -> StepBench {
    let mut bench = StepBench::new("train_step", work).with_warmup_steps(warmup);
    {
        let mut sink = |event: TrainEvent| bench.record(&event);
        trainer
            .train(config, &mut sink)
            .expect("the offline synthetic path trains");
    }
    bench
}

#[test]
fn mock_trainer_run_yields_one_window_per_step_after_the_first() {
    let out = TempDir::new("bench-mock");
    let config = TrainConfig {
        steps: 6,
        output: OutputConfig {
            dir: out.0.clone(),
            name: "mock".into(),
            // Past `steps`, so nothing contaminates a window here.
            checkpoint_every: 1000,
            sample_every: 0,
        },
        ..Default::default()
    };

    let bench = benched(&mut MockTrainer, &config, StepWork::unmodelled(), 1);

    // Six steps, five inter-step windows: the first Step opens a window
    // instead of closing one, because what precedes it is setup, not a step.
    assert_eq!(bench.samples().len(), 5);
    assert_eq!(bench.counted().len(), 4, "one window is warm-up");
    assert!(
        bench.samples().iter().all(|s| s.window > Duration::ZERO),
        "every window must have positive duration"
    );
    assert!(
        bench.losses_plausible(),
        "MockTrainer's decaying loss is finite and non-zero"
    );

    // No work model was declared, so nothing derived is claimed.
    let summary = bench
        .summary_line()
        .expect("four counted windows")
        .to_string();
    assert!(summary.starts_with("RESULT label=train_step_median ms="));
    assert!(!summary.contains("tok_s="), "{summary}");
    assert!(
        bench.model_line().is_none(),
        "unmodelled work has no MODEL line"
    );
}

#[test]
fn a_checkpoint_export_contaminates_exactly_its_own_window() {
    let out = TempDir::new("bench-ckpt");
    let config = TrainConfig {
        steps: 6,
        output: OutputConfig {
            dir: out.0.clone(),
            name: "mock".into(),
            // Fires after steps 2 and 4 — mid-run, which is the case that
            // matters: the export's disk I/O lands inside the window that
            // step 3 (and step 5) then closes.
            checkpoint_every: 2,
            sample_every: 0,
        },
        ..Default::default()
    };

    let bench = benched(&mut MockTrainer, &config, StepWork::unmodelled(), 0);

    let contaminated: Vec<u64> = bench
        .samples()
        .iter()
        .filter(|s| s.contaminated)
        .map(|s| s.step)
        .collect();
    assert_eq!(
        contaminated,
        vec![3, 5],
        "an export between steps N and N+1 belongs to N+1's window, and to no other"
    );
    assert_eq!(bench.counted().len(), 3, "windows 2, 4 and 6 stay clean");
    assert!(bench.counted().iter().all(|s| !s.contaminated));

    // The per-step lines carry the flag, so a reader of the raw log can see
    // which window was excluded and why.
    let lines = bench.result_lines();
    assert_eq!(
        lines.iter().filter(|l| l.contains(" ckpt=1")).count(),
        2,
        "{lines:#?}"
    );
}

#[test]
fn real_burn_training_steps_are_timed_and_pass_the_dead_graph_checks() {
    let out = TempDir::new("bench-burn");
    let config = TrainConfig {
        steps: 8,
        seed: 42,
        task: TaskKind::Classification,
        output: OutputConfig {
            dir: out.0.clone(),
            name: "burn".into(),
            checkpoint_every: 10_000,
            sample_every: 0,
        },
        ..Default::default()
    };

    // A declared token count, so the derived-throughput path is exercised too.
    // The number is arbitrary for a LoRA-MLP — what is under test is that
    // `tok_s` is a real quotient of it, not that it means anything.
    let work = StepWork::unmodelled().with_tokens(1024);
    let bench = benched(&mut BurnTrainer, &config, work, 1);

    assert_eq!(bench.samples().len(), 7);
    assert!(
        bench.losses_plausible(),
        "a real synthetic run produces finite, non-zero losses"
    );

    let median = bench.median_step().expect("counted windows exist");
    assert!(median > Duration::ZERO, "a real step takes real time");

    // Every counted window should be the same *kind* of thing. A window two
    // orders of magnitude off the median is not a slow step, it is a step that
    // did something else — the shape a silently-skipped or silently-doubled
    // step would make. Deliberately loose: shared CI hardware is noisy, and a
    // tight bound here would be a flake, not a check.
    for sample in bench.counted() {
        assert!(
            sample.window < median * 100 && sample.window * 100 > median,
            "window at step {} ({:?}) is not the same kind of work as the median ({median:?})",
            sample.step,
            sample.window,
        );
    }

    // A verdict must exist and be a real number. Deliberately NOT asserting
    // `sanity.ok`: that band is ±15%, and a single scheduler hiccup in either
    // half of a 6-window run moves the ratio out of it — the same wall-clock
    // flake this suite's header disclaims, just one level of indirection away.
    // The `ok` direction is pinned deterministically in `bench.rs`'s
    // `sanity_ok_on_stable_synthetic_windows`; what belongs here is that a real
    // run is nowhere near the elided-graph shape, which is orders of magnitude
    // off, not percent.
    let sanity = bench.sanity().expect("two or more counted windows");
    assert!(
        (0.5..5.0).contains(&sanity.ratio),
        "a real run's 2x-steps ratio should be near 2, not degenerate (got {})",
        sanity.ratio
    );

    let summary = bench
        .summary_line()
        .expect("counted windows exist")
        .to_string();
    assert!(summary.contains("plausible=true"), "{summary}");
    // Present, not `ok` — see the ratio assertion above for why the
    // verdict itself is not a safe thing to pin against a wall clock.
    assert!(summary.contains("sanity="), "{summary}");
    assert!(
        summary.contains("tok_s="),
        "a declared token count is reported"
    );

    // MODEL comes first: the denominator is on screen before the quotient.
    let lines = bench.result_lines();
    assert!(lines[0].starts_with("MODEL label=train_step"), "{lines:#?}");
    assert!(lines.last().unwrap().starts_with("SANITY "), "{lines:#?}");
}
