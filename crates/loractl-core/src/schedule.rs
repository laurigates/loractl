//! Learning-rate schedules (#212): `optim.schedule`.
//!
//! `optim.lr` is the **peak** (base) learning rate. [`ScheduleConfig`] says how
//! the rate that is actually applied moves relative to it: an optional linear
//! warmup, then a decay (or none). The default — [`ScheduleKind::Constant`]
//! with no warmup — applies `optim.lr` unchanged on every step, bit-for-bit, so
//! every existing config and every pinned loss trajectory is untouched.
//!
//! # A pure function of the 1-based global step
//!
//! The rate at step `s` of a run of `total` steps is
//! [`ScheduleConfig::lr_at`]`(base, s, total)` — a stateless function of
//! nothing but those three numbers. That is deliberate and is what makes
//! resume correct: the diffusion loop resumes at `plan.start_step()` (the
//! global step recorded in the artifact), so a resumed run evaluates the same
//! function at the same global step and lands on the same point of the curve an
//! uninterrupted run would have — no scheduler state to persist, round-trip, or
//! get out of sync with the step counter. A stateful scheduler that was
//! `.step()`ped once per optimizer step would silently restart its warmup
//! after a resume unless its position were saved separately.
//!
//! `burn::lr_scheduler` was considered and not used for exactly that reason:
//! its schedulers are stateful iterators (`LrScheduler::step`), so skipping to
//! step *k* means replaying *k* calls or checkpointing a record, and its set
//! has no min-LR floor or plain `constant-after-warmup` form without
//! composition. The five curves here are a handful of lines each, covered by
//! hand-computed expectations in the tests below.
//!
//! `total` is `steps` from the config — a TOTAL, not a remainder, matching
//! [`ResumeConfig`](crate::config::ResumeConfig)'s contract. Resuming with a
//! *different* `steps` therefore stretches the decay over the new total; keep
//! `steps` fixed across a resume to continue the original curve.
//!
//! # Shape
//!
//! Steps are 1-based and the rate is the one **applied during** that step.
//!
//! - **Warmup** (`warmup_steps = W`): step `s <= W` runs at `base * s / W`, so
//!   step `W` reaches the peak exactly.
//! - **Decay** starts after warmup. With `d = s - 1 - W` (0 on the first
//!   post-warmup step) and `span = max(total - W, 1)`, the progress is
//!   `p = min(d / span, 1)`:
//!   - `constant`: `base`.
//!   - `linear`: `base` falling straight to `min_lr` as `p` goes 0 → 1.
//!   - `cosine`: half-cosine from `base` to `min_lr`.
//!   - `exponential`: `base * gamma^d`, floored at `min_lr`.
//!   - `step`: `base * gamma^floor(d / step_size)`, floored at `min_lr`.
//!
//! The rate is sampled at the *start* of each step (the torch convention), so
//! `linear`/`cosine` approach `min_lr` over the run and the final step sits one
//! increment above it; `min_lr` is the asymptote, reached only by a step past
//! `total`, where `p` clamps at 1. The first post-warmup step is at `p = 0`, so
//! it repeats the peak: the peak is held for two steps, never skipped.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::str::FromStr;

/// The decay curve applied after warmup. See the [module docs](self) for the
/// formulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScheduleKind {
    /// No decay: the peak rate for the whole run (after any warmup). The
    /// default, and bit-identical to the pre-#212 bare `optim.lr`.
    #[default]
    Constant,
    /// Straight line from the peak down to `min_lr`.
    Linear,
    /// Half-cosine from the peak down to `min_lr`.
    Cosine,
    /// Multiply by `gamma` every step (floored at `min_lr`). Needs `gamma`.
    Exponential,
    /// Multiply by `gamma` every `step_size` steps (floored at `min_lr`).
    /// Needs `gamma` and `step_size`.
    Step,
}

// No `clap::ValueEnum` derive on purpose (same reasoning as `BackendKind`):
// the vocabulary lives once in core, and there is no CLI flag — the YAML and
// `LORACTL_OPTIM__SCHEDULE__KIND` env layers are the surface.
impl FromStr for ScheduleKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "constant" => Ok(Self::Constant),
            "linear" => Ok(Self::Linear),
            "cosine" => Ok(Self::Cosine),
            "exponential" => Ok(Self::Exponential),
            "step" => Ok(Self::Step),
            other => Err(format!(
                "unknown lr schedule {other:?} (constant|linear|cosine|exponential|step)"
            )),
        }
    }
}

// Deserialize through `FromStr` so YAML and env accept the same spellings (see
// `BackendKind`'s `Deserialize`); `Serialize` stays derived (kebab-case), so a
// round-trip is stable.
impl<'de> Deserialize<'de> for ScheduleKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl ScheduleKind {
    /// The spelling of this curve in the YAML and in the `ss_lr_scheduler`
    /// metadata key (kohya's name where kohya has one; `exponential` and `step`
    /// are loractl's own).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Linear => "linear",
            Self::Cosine => "cosine",
            Self::Exponential => "exponential",
            Self::Step => "step",
        }
    }
}

/// `optim.schedule`: how the learning rate moves over a run, relative to the
/// peak `optim.lr`. Every field defaults, and the default is "constant, no
/// warmup" — see the [module docs](self).
///
/// Env overrides: `LORACTL_OPTIM__SCHEDULE__KIND`, `..__WARMUP_STEPS`,
/// `..__MIN_LR`, `..__GAMMA`, `..__STEP_SIZE`. No CLI flags (`--lr` keeps
/// meaning the peak rate).
///
/// Unlike the rest of the (permissive) config schema this block is
/// `deny_unknown_fields`: a typo such as `warmup:` for `warmup_steps:` would
/// otherwise be dropped and the run would quietly train with no warmup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScheduleConfig {
    /// The decay curve. Default: `constant`.
    pub kind: ScheduleKind,
    /// Linear warmup from ~0 to the peak over this many steps (0 = none).
    /// Valid with every `kind`, including `constant` (warmup then flat).
    /// Must be less than `steps`.
    pub warmup_steps: u64,
    /// Floor the decay approaches (`linear`/`cosine`) or never goes below
    /// (`exponential`/`step`). Default 0. Must be within `0..=optim.lr`;
    /// rejected for `constant`, where it would do nothing.
    pub min_lr: f64,
    /// Decay factor in `(0, 1]` — per step for `exponential`, per `step_size`
    /// steps for `step`. Required by those two kinds, rejected by the rest.
    pub gamma: Option<f64>,
    /// Steps between drops, for `step` (>= 1). Required by `step`, rejected by
    /// the rest.
    pub step_size: Option<u64>,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            kind: ScheduleKind::Constant,
            warmup_steps: 0,
            min_lr: 0.0,
            gamma: None,
            step_size: None,
        }
    }
}

/// `weight_on_base * base + (1 - weight_on_base) * min`. Written so weight 1.0
/// returns `base` and weight 0.0 returns `min` exactly — `min + (base - min)`
/// can be off by an ulp at the peak, which would make the first decay step
/// differ from the warmup's last for no reason.
fn blend(base: f64, min: f64, weight_on_base: f64) -> f64 {
    base * weight_on_base + min * (1.0 - weight_on_base)
}

impl ScheduleConfig {
    /// The learning rate applied during the 1-based global `step` of a
    /// `total`-step run whose peak rate is `base`. Pure — see the
    /// [module docs](self).
    pub fn lr_at(&self, base: f64, step: u64, total: u64) -> f64 {
        let idx = step.saturating_sub(1);
        if idx < self.warmup_steps {
            return base * (idx + 1) as f64 / self.warmup_steps as f64;
        }
        let d = idx - self.warmup_steps;
        let span = total.saturating_sub(self.warmup_steps).max(1);
        let progress = (d as f64 / span as f64).min(1.0);
        match self.kind {
            ScheduleKind::Constant => base,
            ScheduleKind::Linear => blend(base, self.min_lr, 1.0 - progress),
            ScheduleKind::Cosine => blend(base, self.min_lr, 0.5 * (1.0 + (PI * progress).cos())),
            ScheduleKind::Exponential => {
                let gamma = self.gamma.unwrap_or(1.0);
                (base * gamma.powf(d as f64)).max(self.min_lr)
            }
            ScheduleKind::Step => {
                let gamma = self.gamma.unwrap_or(1.0);
                let drops = d / self.step_size.unwrap_or(1).max(1);
                (base * gamma.powf(drops as f64)).max(self.min_lr)
            }
        }
    }

    /// Reject a schedule that is malformed or would silently do nothing. Every
    /// message names the config key, like the dataset validators. `base` is
    /// `optim.lr` and `total` is `steps`.
    pub fn validate(&self, base: f64, total: u64) -> Result<()> {
        let kind = self.kind.as_str();
        if self.warmup_steps >= total.max(1) {
            bail!(
                "optim.schedule.warmup_steps = {} must be less than steps = {}; the run \
                 would never leave warmup",
                self.warmup_steps,
                total.max(1)
            );
        }
        let decays = self.kind != ScheduleKind::Constant;
        if !self.min_lr.is_finite() || self.min_lr < 0.0 {
            bail!(
                "optim.schedule.min_lr = {} must be within 0..=optim.lr ({base})",
                self.min_lr
            );
        }
        // The upper bound only means something for a decaying kind; for
        // `constant` the check below rejects any non-zero `min_lr`, and a
        // default schedule must not newly reject an odd `optim.lr`.
        if decays && self.min_lr > base {
            bail!(
                "optim.schedule.min_lr = {} must be within 0..=optim.lr ({base})",
                self.min_lr
            );
        }
        if !decays && self.min_lr != 0.0 {
            bail!(
                "optim.schedule.min_lr = {} has no effect with kind: constant; pick a \
                 decaying kind or drop it",
                self.min_lr
            );
        }
        let wants_gamma = matches!(self.kind, ScheduleKind::Exponential | ScheduleKind::Step);
        match self.gamma {
            Some(g) if !wants_gamma => bail!(
                "optim.schedule.gamma = {g} is only used by kind: exponential|step (got \
                 kind: {kind})"
            ),
            Some(g) if !(g.is_finite() && g > 0.0 && g <= 1.0) => {
                bail!("optim.schedule.gamma = {g} must be in (0, 1]")
            }
            None if wants_gamma => {
                bail!("optim.schedule.kind: {kind} requires optim.schedule.gamma")
            }
            _ => {}
        }
        match self.step_size {
            Some(n) if self.kind != ScheduleKind::Step => bail!(
                "optim.schedule.step_size = {n} is only used by kind: step (got kind: {kind})"
            ),
            Some(0) => bail!("optim.schedule.step_size must be at least 1"),
            None if self.kind == ScheduleKind::Step => {
                bail!("optim.schedule.kind: step requires optim.schedule.step_size")
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: f64 = 1e-3;

    fn sched(kind: ScheduleKind) -> ScheduleConfig {
        ScheduleConfig {
            kind,
            ..ScheduleConfig::default()
        }
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12 * b.abs().max(1e-12), "{a} vs {b}");
    }

    #[test]
    fn default_is_bit_identical_to_the_bare_lr() {
        let s = ScheduleConfig::default();
        for lr in [1e-4, 0.01, 3.3e-5, 0.1 + 0.2] {
            for step in [0, 1, 2, 999, 1000, 1001, u64::MAX] {
                assert_eq!(s.lr_at(lr, step, 1000).to_bits(), lr.to_bits());
            }
        }
    }

    #[test]
    fn warmup_ramps_to_the_peak_then_holds_for_constant() {
        let s = ScheduleConfig {
            warmup_steps: 4,
            ..ScheduleConfig::default()
        };
        let got: Vec<f64> = (1..=6).map(|k| s.lr_at(BASE, k, 100)).collect();
        // Hand-computed: base * {1,2,3,4}/4, then flat.
        let want = [0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 1e-3, 1e-3];
        for (g, w) in got.iter().zip(want) {
            close(*g, w);
        }
        assert_eq!(
            got[3].to_bits(),
            BASE.to_bits(),
            "step W is exactly the peak"
        );
    }

    #[test]
    fn linear_values() {
        let s = ScheduleConfig {
            min_lr: 1e-4,
            ..sched(ScheduleKind::Linear)
        };
        // total 10, no warmup: p = (step-1)/10 -> lr = 1e-3 - 9e-4 * p.
        close(s.lr_at(BASE, 1, 10), 1e-3);
        close(s.lr_at(BASE, 6, 10), 1e-3 - 9e-4 * 0.5);
        close(s.lr_at(BASE, 10, 10), 1e-3 - 9e-4 * 0.9);
        // Past the end p clamps at 1: exactly the floor.
        assert_eq!(s.lr_at(BASE, 11, 10).to_bits(), 1e-4f64.to_bits());
        assert_eq!(s.lr_at(BASE, 500, 10).to_bits(), 1e-4f64.to_bits());
    }

    #[test]
    fn cosine_with_warmup_matches_independent_values() {
        let s = ScheduleConfig {
            warmup_steps: 10,
            ..sched(ScheduleKind::Cosine)
        };
        // total 100 -> span 90. Step 56 -> d = 56 - 1 - 10 = 45 -> p = 0.5 -> half.
        close(s.lr_at(BASE, 56, 100), 0.5e-3);
        // Step 11 is the first decay step: p = 0, the peak again.
        assert_eq!(s.lr_at(BASE, 11, 100).to_bits(), BASE.to_bits());
        // Step 10 (end of warmup) is the peak.
        assert_eq!(s.lr_at(BASE, 10, 100).to_bits(), BASE.to_bits());
        // Quarter point: step 33 -> d = 22 -> p = 22/90.
        let p = 22.0 / 90.0;
        close(s.lr_at(BASE, 33, 100), 1e-3 * 0.5 * (1.0 + (PI * p).cos()));
        // Mid-warmup.
        close(s.lr_at(BASE, 5, 100), 0.5e-3);
        // Beyond total: floor (0).
        assert_eq!(s.lr_at(BASE, 101, 100), 0.0);
        assert_eq!(s.lr_at(BASE, 10_000, 100), 0.0);
    }

    #[test]
    fn cosine_edge_cases() {
        let s = sched(ScheduleKind::Cosine);
        // total = 1, warmup = 0: step 1 is p = 0 -> the peak, not the floor.
        assert_eq!(s.lr_at(BASE, 1, 1).to_bits(), BASE.to_bits());
        // total = 0 is treated like 1 (trainers do `steps.max(1)`).
        assert_eq!(s.lr_at(BASE, 1, 0).to_bits(), BASE.to_bits());
        // step 0 never happens, but must not underflow.
        assert_eq!(s.lr_at(BASE, 0, 10).to_bits(), BASE.to_bits());
        // warmup >= total: span clamps to 1, no division by zero / NaN.
        let w = ScheduleConfig {
            warmup_steps: 5,
            ..sched(ScheduleKind::Cosine)
        };
        let v = w.lr_at(BASE, 7, 3);
        assert!(v.is_finite() && (0.0..=BASE).contains(&v), "{v}");
    }

    #[test]
    fn exponential_values_and_floor() {
        let s = ScheduleConfig {
            gamma: Some(0.5),
            min_lr: 1.5e-4,
            ..sched(ScheduleKind::Exponential)
        };
        close(s.lr_at(BASE, 1, 100), 1e-3);
        close(s.lr_at(BASE, 2, 100), 5e-4);
        close(s.lr_at(BASE, 3, 100), 2.5e-4);
        // 1e-3 * 0.5^3 = 1.25e-4 < floor.
        assert_eq!(s.lr_at(BASE, 4, 100).to_bits(), 1.5e-4f64.to_bits());
    }

    #[test]
    fn step_values_with_warmup() {
        let s = ScheduleConfig {
            warmup_steps: 2,
            gamma: Some(0.1),
            step_size: Some(3),
            ..sched(ScheduleKind::Step)
        };
        // d = step - 3 after warmup: drops = d / 3.
        let at = |k| s.lr_at(BASE, k, 100);
        close(at(1), 0.5e-3);
        close(at(2), 1e-3);
        close(at(3), 1e-3); // d = 0
        close(at(5), 1e-3); // d = 2
        close(at(6), 1e-4); // d = 3 -> one drop
        close(at(9), 1e-5); // d = 6 -> two drops
    }

    #[test]
    fn schedule_is_monotone_after_warmup_for_decaying_kinds() {
        for s in [
            sched(ScheduleKind::Linear),
            sched(ScheduleKind::Cosine),
            ScheduleConfig {
                gamma: Some(0.97),
                ..sched(ScheduleKind::Exponential)
            },
        ] {
            let mut prev = f64::INFINITY;
            for k in 1..=60 {
                let v = s.lr_at(BASE, k, 50);
                assert!(
                    v <= prev && v >= 0.0,
                    "{:?} step {k}: {v} after {prev}",
                    s.kind
                );
                prev = v;
            }
        }
    }

    #[test]
    fn kind_parses_case_insensitively_and_round_trips() {
        for k in ["constant", "linear", "cosine", "exponential", "step"] {
            let kind: ScheduleKind = k.parse().unwrap();
            assert_eq!(kind.as_str(), k);
            assert_eq!(
                serde_json::to_value(kind).unwrap(),
                serde_json::Value::String(k.into())
            );
        }
        assert_eq!("Cosine".parse::<ScheduleKind>(), Ok(ScheduleKind::Cosine));
        assert!("poly".parse::<ScheduleKind>().is_err());
    }

    #[test]
    fn validate_accepts_the_default_and_good_schedules() {
        ScheduleConfig::default().validate(BASE, 1).unwrap();
        ScheduleConfig {
            warmup_steps: 9,
            min_lr: 1e-5,
            ..sched(ScheduleKind::Cosine)
        }
        .validate(BASE, 10)
        .unwrap();
        ScheduleConfig {
            gamma: Some(0.5),
            step_size: Some(10),
            ..sched(ScheduleKind::Step)
        }
        .validate(BASE, 100)
        .unwrap();
    }

    #[test]
    fn the_default_schedule_validates_for_any_lr() {
        // Pre-#212 configs never had `lr` range-checked; the default schedule
        // must not start rejecting one (e.g. lr: 0 or a negative lr).
        for lr in [0.0, -1e-4, 1e-4, 10.0] {
            ScheduleConfig::default().validate(lr, 10).unwrap();
        }
    }

    #[test]
    fn unknown_schedule_keys_are_rejected_not_ignored() {
        let err = serde_json::from_str::<ScheduleConfig>(r#"{"kind":"cosine","warmup":50}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown field `warmup`"), "{err}");
        // The real spelling still parses.
        let ok: ScheduleConfig =
            serde_json::from_str(r#"{"kind":"cosine","warmup_steps":50}"#).unwrap();
        assert_eq!(ok.warmup_steps, 50);
    }

    #[test]
    fn validate_rejects_each_misconfiguration_by_key() {
        let cases: Vec<(ScheduleConfig, u64, &str)> = vec![
            (
                ScheduleConfig {
                    warmup_steps: 10,
                    ..ScheduleConfig::default()
                },
                10,
                "warmup_steps",
            ),
            (
                ScheduleConfig {
                    min_lr: 2e-3,
                    ..sched(ScheduleKind::Linear)
                },
                10,
                "min_lr",
            ),
            (
                ScheduleConfig {
                    min_lr: -1.0,
                    ..sched(ScheduleKind::Linear)
                },
                10,
                "min_lr",
            ),
            (
                ScheduleConfig {
                    min_lr: 1e-5,
                    ..ScheduleConfig::default()
                },
                10,
                "no effect",
            ),
            (
                sched(ScheduleKind::Exponential),
                10,
                "requires optim.schedule.gamma",
            ),
            (
                ScheduleConfig {
                    gamma: Some(1.5),
                    ..sched(ScheduleKind::Exponential)
                },
                10,
                "(0, 1]",
            ),
            (
                ScheduleConfig {
                    gamma: Some(0.5),
                    ..sched(ScheduleKind::Cosine)
                },
                10,
                "only used by kind: exponential|step",
            ),
            (
                ScheduleConfig {
                    gamma: Some(0.5),
                    ..sched(ScheduleKind::Step)
                },
                10,
                "requires optim.schedule.step_size",
            ),
            (
                ScheduleConfig {
                    gamma: Some(0.5),
                    step_size: Some(0),
                    ..sched(ScheduleKind::Step)
                },
                10,
                "at least 1",
            ),
            (
                ScheduleConfig {
                    gamma: Some(0.5),
                    step_size: Some(2),
                    ..sched(ScheduleKind::Exponential)
                },
                10,
                "only used by kind: step",
            ),
        ];
        for (cfg, total, needle) in cases {
            let err = cfg.validate(BASE, total).unwrap_err().to_string();
            assert!(
                err.contains(needle),
                "{needle:?} not in {err:?} for {cfg:?}"
            );
        }
    }
}
