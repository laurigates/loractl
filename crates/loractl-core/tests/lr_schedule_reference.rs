//! `optim.schedule` against diffusers' own definitions (#247).
//!
//! `src/schedule.rs` pins each curve with hand-computed values, which can only
//! confirm loractl's own reading of it. This test compares against a golden
//! produced by executing diffusers' `get_*_schedule_with_warmup` at a pinned
//! tag (`reference/lr_schedule_reference.py`, `just lr-schedule-reference`),
//! the scheduler kohya-ss and most LoRA trainers use.
//!
//! # Mapping
//!
//! diffusers' `lr_lambda(k)` takes a 0-based scheduler index and its value is
//! applied during optimizer step `k + 1` (the scheduler is stepped after the
//! optimizer). loractl's [`ScheduleConfig::lr_at`] takes the 1-based step `s`
//! the rate is applied during. So:
//!
//! - **Decay (`s > W`)**: `lr_at(1, s, total) == lambda[s - 1]`, exactly the
//!   diffusers curve, with `min_lr = 0` (diffusers has no floor).
//! - **Warmup (`s <= W`)**: `lr_at(1, s, total) == lambda[s]`. This is a
//!   **deliberate one-step divergence**: diffusers runs step 1 at rate 0 (a
//!   wasted step) and first reaches the peak at step `W + 1`; loractl starts
//!   at `1/W` and reaches the peak at step `W`, holding it for two steps (see
//!   the `schedule` module docs). The divergence is pinned here rather than
//!   hidden, so switching to the diffusers convention is a visible decision.
//!
//! `exponential` and `step` have no diffusers equivalent and are not in the
//! golden.

use loractl_core::config::{ScheduleConfig, ScheduleKind};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    source: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    kind: String,
    warmup_steps: u64,
    total: u64,
    lambda: Vec<f64>,
}

fn golden() -> Golden {
    serde_json::from_str(include_str!("golden/lr_schedule.json")).expect("golden parses")
}

fn schedule(case: &Case) -> ScheduleConfig {
    ScheduleConfig {
        kind: case.kind.parse::<ScheduleKind>().expect("known kind"),
        warmup_steps: case.warmup_steps,
        ..ScheduleConfig::default()
    }
}

const TOL: f64 = 1e-12;

#[test]
fn decay_matches_the_diffusers_curve_at_every_step() {
    let golden = golden();
    assert!(golden.source.starts_with("diffusers "), "{}", golden.source);
    let mut compared = 0;
    for case in &golden.cases {
        let sched = schedule(case);
        assert_eq!(case.lambda.len() as u64, case.total + 1, "golden shape");
        for s in (case.warmup_steps + 1)..=case.total {
            let ours = sched.lr_at(1.0, s, case.total);
            let theirs = case.lambda[(s - 1) as usize];
            assert!(
                (ours - theirs).abs() <= TOL,
                "{} W={} total={} step {s}: loractl {ours} vs diffusers {theirs}",
                case.kind,
                case.warmup_steps,
                case.total
            );
            compared += 1;
        }
    }
    // Teeth: every kind contributed, and a vacuous loop can't pass.
    for kind in ["constant", "linear", "cosine"] {
        assert!(
            golden.cases.iter().any(|c| c.kind == kind),
            "{kind} missing"
        );
    }
    assert!(compared > 100, "compared only {compared} steps");
}

#[test]
fn warmup_runs_one_step_ahead_of_diffusers_by_design() {
    let mut compared = 0;
    for case in golden().cases.iter().filter(|c| c.warmup_steps > 0) {
        let sched = schedule(case);
        for s in 1..=case.warmup_steps {
            let ours = sched.lr_at(1.0, s, case.total);
            assert!(
                (ours - case.lambda[s as usize]).abs() <= TOL,
                "{} W={} step {s}: loractl {ours} vs diffusers lambda[{s}] {}",
                case.kind,
                case.warmup_steps,
                case.lambda[s as usize]
            );
            // The divergence is real, not a vacuous equality: diffusers' rate
            // for the same step is strictly lower.
            assert!(ours > case.lambda[(s - 1) as usize]);
            compared += 1;
        }
        // diffusers' first step is a zero-rate step; loractl's is not.
        assert_eq!(case.lambda[0], 0.0);
    }
    assert!(compared >= 15, "compared only {compared} warmup steps");
}
