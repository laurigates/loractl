//! Relative-accuracy gate for reduced-precision numerics (#112).
//!
//! loractl's fixed-truth parity pins the f32 paths against a PyTorch golden at
//! absolute `1e-5` (`tests/lora_reference.rs`, `tests/quant.rs`, `tests/fp8.rs`,
//! the MMDiT parity suite). That is the right shape for f32, but the
//! reduced-precision paths — int8/int4 QLoRA (`quant.rs`), scaled-fp8 dequant
//! (`fp8.rs`), the f16 MMDiT path — **cannot** meet an absolute `1e-5` by
//! construction: int4's per-block symmetric quant carries ~7% worst-case weight
//! error. An absolute threshold is either too loose (useless) or unachievable
//! (spurious) for them.
//!
//! This is the CAEF ADR-0006 relative-oracle protocol ported to loractl (see
//! `docs/adrs/0006-reduced-precision-accuracy-gate.md`): measure the
//! reduced-precision output's deviation from a **full-precision oracle** run
//! over the *same activations* (so activation representation is not conflated
//! into the measure — the analog of ADR-0006's "same quantized inputs"), then
//! gate that deviation against a **calibrated band** plus a **hard ceiling**:
//!
//! ```text
//! gate = all-finite  ∧  d_ours ≤ max(2·d_bar, floor)  ∧  d_ours ≤ ceil
//! ```
//!
//! It does **not** replace the fixed-truth goldens — the two catch different
//! bugs, and this gate is added alongside them. A fixed-truth golden pins a
//! known point and catches any regression there; a bar-relative gate tolerates
//! the inherent quantization error a fixed point cannot, but can *mask* a
//! regression if the bar itself drifts (ADR-0006's own autotuned-matmul-drift
//! lesson). Keeping both tiers is the point, not an accident: the `ceil` is the
//! fixed-truth backstop the calibrated band can never widen past.

/// The calibrated band a reduced-precision path's relative error must sit in.
///
/// `d_bar` is the "known-good" relative deviation for a specific
/// path × scheme, measured once from the current implementation and pinned. The
/// accepted band is `max(2·d_bar, floor)`: `2·d_bar` gives headroom for benign
/// numeric drift while staying tight enough to catch a path that silently
/// degrades (e.g. int4 quant losing effective range), and `floor` keeps the
/// band from collapsing toward zero when `d_bar` is tiny. `ceil` is a fixed
/// backstop that the calibrated band can never exceed — it catches gross
/// corruption (a broken kernel, a dtype mixup) even if `d_bar` were
/// mis-calibrated high.
#[derive(Debug, Clone, Copy)]
pub struct RelGate {
    /// Calibrated known-good relative deviation for this path/scheme.
    pub d_bar: f32,
    /// Absolute floor on the band, so a near-exact path does not fail on
    /// `2·d_bar` shrinking below f32 rounding noise.
    pub floor: f32,
    /// Hard ceiling: the fixed-truth backstop the band never widens past.
    pub ceil: f32,
}

/// The outcome of applying a [`RelGate`], carrying the numbers that decided it
/// so a failing assertion prints *why*, not merely that it failed.
#[derive(Debug, Clone, Copy)]
pub struct GateOutcome {
    /// The measured relative deviation from the full-precision oracle.
    pub d_ours: f32,
    /// The accepted band, `max(2·d_bar, floor)`.
    pub band: f32,
    /// The hard ceiling copied from the gate.
    pub ceil: f32,
    /// Whether `d_ours` was finite (non-finite is always a failure).
    pub finite: bool,
    /// The verdict: `finite ∧ d_ours ≤ band ∧ d_ours ≤ ceil`.
    pub passed: bool,
}

impl GateOutcome {
    /// Assert the gate passed, with a message naming every term. Intended for
    /// tests; `label` identifies the path/scheme under test.
    pub fn expect_pass(self, label: &str) {
        assert!(
            self.passed,
            "{label}: relative-accuracy gate FAILED — d_ours = {:e}, \
             band = max(2·d_bar, floor) = {:e}, ceil = {:e}, all-finite = {}",
            self.d_ours, self.band, self.ceil, self.finite
        );
    }
}

impl RelGate {
    /// Apply the gate to a measured relative deviation.
    pub fn apply(self, d_ours: f32) -> GateOutcome {
        let band = (2.0 * self.d_bar).max(self.floor);
        let finite = d_ours.is_finite();
        let passed = finite && d_ours <= band && d_ours <= self.ceil;
        GateOutcome {
            d_ours,
            band,
            ceil: self.ceil,
            finite,
            passed,
        }
    }
}

/// Peak-normalized max relative deviation of `ours` from the full-precision
/// `reference`: `maxᵢ |oursᵢ − referenceᵢ| / max(peak|reference|, tiny)`.
///
/// Peak-normalization (rather than per-element division) is the same relative
/// measure the existing numerics tests use (`tests/quant.rs`'s chunked-gradient
/// check) and avoids blowing up on reference elements near zero. Returns `+∞`
/// — a guaranteed gate failure — if the lengths differ, either side is empty,
/// or any element is non-finite, so a corrupted forward can never pass by
/// producing a small-looking number.
pub fn rel_deviation(ours: &[f32], reference: &[f32]) -> f32 {
    if ours.is_empty() || ours.len() != reference.len() {
        return f32::INFINITY;
    }
    let mut peak = 0f32;
    let mut max_abs = 0f32;
    let mut finite = true;
    for (&a, &b) in ours.iter().zip(reference) {
        finite &= a.is_finite() && b.is_finite();
        peak = peak.max(b.abs());
        max_abs = max_abs.max((a - b).abs());
    }
    if !finite {
        return f32::INFINITY;
    }
    max_abs / peak.max(1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATE: RelGate = RelGate {
        d_bar: 0.01,
        floor: 1e-4,
        ceil: 0.05,
    };
    // band = max(2·0.01, 1e-4) = 0.02.

    #[test]
    fn passes_inside_the_band() {
        assert!(GATE.apply(0.015).passed, "0.015 ≤ band 0.02");
        assert!(GATE.apply(0.02).passed, "band edge is inclusive");
    }

    #[test]
    fn fails_above_the_band_even_under_the_ceiling() {
        // 0.03 is under ceil 0.05 but over band 0.02 — the band is the tight
        // regression catch, so this must fail.
        let out = GATE.apply(0.03);
        assert!(!out.passed);
        assert_eq!(out.band, 0.02);
    }

    #[test]
    fn ceiling_catches_a_mis_calibrated_high_bar() {
        // A bar so loose the band (2·d_bar = 0.2) would admit gross error; the
        // hard ceiling overrides it.
        let loose = RelGate {
            d_bar: 0.1,
            floor: 1e-4,
            ceil: 0.05,
        };
        assert!(
            !loose.apply(0.08).passed,
            "0.08 > ceil 0.05 despite band 0.2"
        );
        assert!(loose.apply(0.04).passed, "0.04 ≤ ceil 0.05 and ≤ band 0.2");
    }

    #[test]
    fn floor_admits_a_near_exact_path() {
        // A path with a tiny d_bar: 2·d_bar would be below f32 rounding, so the
        // floor sets the band instead.
        let tight = RelGate {
            d_bar: 1e-8,
            floor: 1e-4,
            ceil: 0.05,
        };
        assert_eq!(tight.apply(0.0).band, 1e-4);
        assert!(tight.apply(5e-5).passed, "5e-5 ≤ floor band 1e-4");
        assert!(!tight.apply(2e-4).passed, "2e-4 > floor band 1e-4");
    }

    #[test]
    fn non_finite_always_fails() {
        assert!(!GATE.apply(f32::NAN).passed);
        assert!(!GATE.apply(f32::INFINITY).passed);
    }

    #[test]
    fn rel_deviation_is_peak_normalized_max() {
        // reference peak = 4.0; max abs diff = 0.4 → 0.1.
        let d = rel_deviation(&[1.1, 2.0, 4.0], &[1.0, 2.0, 4.0]);
        assert!((d - 0.1 / 4.0).abs() < 1e-6, "got {d}");
    }

    #[test]
    fn rel_deviation_rejects_shape_and_non_finite() {
        assert_eq!(rel_deviation(&[1.0], &[1.0, 2.0]), f32::INFINITY);
        assert_eq!(rel_deviation(&[], &[]), f32::INFINITY);
        assert_eq!(rel_deviation(&[f32::NAN], &[1.0]), f32::INFINITY);
    }
}
