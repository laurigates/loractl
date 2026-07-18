//! `loractl-bench` — backend-agnostic compute measurement harness.
//!
//! Ported from `custom-attention-engine-framework`'s bench harness (loractl
//! #110). Four hard-won pieces that make GPU timing on this stack trustworthy:
//!
//! - a grep-parseable [`BenchResult`] (`RESULT …`) / [`Sanity`] (`SANITY …`)
//!   line schema;
//! - **device-resident wall-sync timing** ([`time_wall_sync`]) — cubecl's own
//!   `ComputeClient::profile` window only spans the first pass and reports
//!   physically impossible throughputs (cubecl#1421), so the only trustworthy
//!   timer fences a batch between two full device syncs;
//! - a **2×-iters [`Sanity`] ratio** that catches an elided / dead compute
//!   graph (total time must scale ~2× when iters double);
//! - a [`plausible`] dead-graph guard rejecting all-zero / non-finite output.
//!
//! It is deliberately backend-agnostic: [`time_wall_sync`] takes a `sync` fence
//! closure, so burn/cubecl, raw CubeCL, or a plain CPU loop all drive the same
//! harness. The burn-`Tensor`/`Autodiff` training-step adapter — the net-new
//! piece that emits a `RESULT` line per training step and reads VRAM — lands in
//! `loractl-core` (#110); this crate is the reusable, dependency-free core, and
//! sits on the raw-buffer boundary burn 0.22 preserves (not the `Tensor`
//! boundary it rewrites, #79).

use std::fmt;
use std::time::{Duration, Instant};

/// Dead-graph guard: an output is *plausible* only if it is non-empty, entirely
/// finite, and not identically zero. Rejects the two silent GPU failure modes
/// the harness exists to catch — an elided/dead compute graph (all zeros) and a
/// device-thread panic leaving NaN/Inf behind — so neither is reported as a
/// real measurement.
pub fn plausible(out: &[f32]) -> bool {
    !out.is_empty() && out.iter().all(|x| x.is_finite()) && out.iter().any(|x| *x != 0.0)
}

/// Time `work` with a **device-resident wall-sync** fence: run `warmup`
/// unmeasured iterations (shader compile / autotune), one `sync` fence, then
/// `iters` iterations between two `sync` fences, and return the
/// **per-iteration** average.
///
/// `sync` must fully drain the device queue (block on the backend's device
/// sync); without it you time async submission, not compute. This is the only
/// trustworthy timer on the burn 0.21 / cubecl 0.10 stack — cubecl's profiled
/// window only spans the first pass (cubecl#1421).
pub fn time_wall_sync(
    iters: u32,
    warmup: u32,
    mut work: impl FnMut(),
    mut sync: impl FnMut(),
) -> Duration {
    assert!(iters > 0, "iters must be > 0");
    for _ in 0..warmup {
        work();
    }
    sync();
    let start = Instant::now();
    for _ in 0..iters {
        work();
    }
    sync();
    start.elapsed() / iters
}

/// Lower/upper bounds on the 2×-iters scaling ratio for an `ok` verdict. Total
/// time at `2·iters` should be ~2× the total at `iters`; a ratio outside this
/// band means work was elided (a pass optimized away, a cached result).
pub const SANITY_LOW: f64 = 1.7;
/// See [`SANITY_LOW`].
pub const SANITY_HIGH: f64 = 2.3;

/// A 2×-iters scaling sanity check (`SANITY …` line).
#[derive(Debug, Clone, Copy)]
pub struct Sanity {
    /// total(2·iters) / total(iters); ~2.0 when per-iteration cost is stable.
    pub ratio: f64,
    /// Whether `ratio` falls within `[SANITY_LOW, SANITY_HIGH]`.
    pub ok: bool,
}

impl Sanity {
    /// Build from the two **per-iteration averages** measured at `N` and `2N`
    /// iterations (what [`time_wall_sync`] returns). Because those are per-iter
    /// averages, the total-time ratio is `2 · avg_2n / avg_n`.
    pub fn from_avgs(avg_n: Duration, avg_2n: Duration) -> Self {
        let a = avg_n.as_secs_f64();
        let ratio = if a > 0.0 {
            2.0 * avg_2n.as_secs_f64() / a
        } else {
            f64::NAN
        };
        Sanity {
            ratio,
            ok: (SANITY_LOW..=SANITY_HIGH).contains(&ratio),
        }
    }

    /// `"ok"` or `"SUSPECT"` — the token used in the `SANITY`/`RESULT` line.
    pub fn verdict(&self) -> &'static str {
        if self.ok { "ok" } else { "SUSPECT" }
    }
}

impl fmt::Display for Sanity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SANITY x2_iters_ratio={:.3} verdict={}",
            self.ratio,
            self.verdict()
        )
    }
}

/// One grep-parseable measurement line: `RESULT label=… ms=… [<unit>=…] [k=v …]
/// [sanity=… x2_ratio=…]`.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// A stable identifier for the measured path (the `path=` analogue).
    pub label: String,
    /// Per-iteration wall-sync time, milliseconds.
    pub ms: f64,
    /// Optional throughput, `(value, unit)` — e.g. `(tflops, "tflops")` or
    /// `(tokens_per_s, "tok_s")`.
    pub throughput: Option<(f64, &'static str)>,
    /// Arbitrary `key=value` annotations (e.g. `vram_mb=…`, `step=…`).
    pub extra: Vec<(String, String)>,
    /// Optional 2×-iters sanity verdict.
    pub sanity: Option<Sanity>,
}

impl BenchResult {
    /// A result from a per-iteration wall-sync duration.
    pub fn new(label: impl Into<String>, per_iter: Duration) -> Self {
        Self {
            label: label.into(),
            ms: per_iter.as_secs_f64() * 1e3,
            throughput: None,
            extra: Vec::new(),
            sanity: None,
        }
    }

    /// Attach a throughput figure printed as `<unit>=<value>`.
    pub fn with_throughput(mut self, value: f64, unit: &'static str) -> Self {
        self.throughput = Some((value, unit));
        self
    }

    /// Attach an arbitrary `key=value` annotation.
    pub fn with(mut self, key: impl Into<String>, value: impl fmt::Display) -> Self {
        self.extra.push((key.into(), value.to_string()));
        self
    }

    /// Attach a 2×-iters sanity verdict.
    pub fn with_sanity(mut self, sanity: Sanity) -> Self {
        self.sanity = Some(sanity);
        self
    }
}

impl fmt::Display for BenchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RESULT label={} ms={:.4}", self.label, self.ms)?;
        if let Some((value, unit)) = self.throughput {
            write!(f, " {unit}={value:.4}")?;
        }
        for (key, value) in &self.extra {
            write!(f, " {key}={value}")?;
        }
        if let Some(sanity) = &self.sanity {
            write!(
                f,
                " sanity={} x2_ratio={:.3}",
                sanity.verdict(),
                sanity.ratio
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn plausible_accepts_finite_nonzero() {
        assert!(plausible(&[0.0, 1.0, 0.0]));
        assert!(plausible(&[-3.0, 2.0]));
    }

    #[test]
    fn plausible_rejects_dead_and_nonfinite() {
        assert!(!plausible(&[]), "empty is a dead graph");
        assert!(!plausible(&[0.0, 0.0, 0.0]), "all-zero is a dead graph");
        assert!(!plausible(&[1.0, f32::NAN]), "NaN is a device panic");
        assert!(!plausible(&[f32::INFINITY, 1.0]), "Inf is a device panic");
    }

    #[test]
    fn wall_sync_calls_work_warmup_plus_iters_times() {
        let calls = Cell::new(0u32);
        let syncs = Cell::new(0u32);
        let _ = time_wall_sync(
            5,
            3,
            || calls.set(calls.get() + 1),
            || syncs.set(syncs.get() + 1),
        );
        assert_eq!(calls.get(), 8, "warmup(3) + iters(5)");
        assert_eq!(
            syncs.get(),
            2,
            "one fence before the timed region, one after"
        );
    }

    #[test]
    #[should_panic(expected = "iters must be > 0")]
    fn wall_sync_rejects_zero_iters() {
        time_wall_sync(0, 1, || {}, || {});
    }

    #[test]
    fn sanity_ok_when_per_iter_cost_stable() {
        // Equal per-iter averages → total doubles when iters double → ratio ~2.
        let s = Sanity::from_avgs(Duration::from_micros(100), Duration::from_micros(100));
        assert!((s.ratio - 2.0).abs() < 1e-9);
        assert!(s.ok);
        assert_eq!(s.verdict(), "ok");
    }

    #[test]
    fn sanity_suspect_when_work_elided() {
        // 2N run costs the same total as the N run (per-iter halved) → ratio ~1.
        let s = Sanity::from_avgs(Duration::from_micros(100), Duration::from_micros(50));
        assert!((s.ratio - 1.0).abs() < 1e-9);
        assert!(!s.ok);
        assert_eq!(s.verdict(), "SUSPECT");
    }

    #[test]
    fn result_line_is_grep_parseable() {
        let line = BenchResult::new("train_step", Duration::from_micros(2500))
            .with_throughput(24.2, "tflops")
            .with("vram_mb", 17300)
            .with_sanity(Sanity::from_avgs(
                Duration::from_micros(100),
                Duration::from_micros(100),
            ))
            .to_string();
        assert!(
            line.starts_with("RESULT label=train_step ms=2.5000"),
            "{line}"
        );
        assert!(line.contains("tflops=24.2000"), "{line}");
        assert!(line.contains("vram_mb=17300"), "{line}");
        assert!(line.contains("sanity=ok x2_ratio=2.000"), "{line}");
    }
}
