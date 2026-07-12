//! The eager-materialization defense in `LoraMlp::new` is load-bearing
//! (#49 H12; see `.claude/rules/burn-lazy-param-init.md`).
//!
//! burn's `Param` is lazily initialized: a fresh layer's weights are not drawn
//! from the RNG until first *accessed*, not when `init()` returns. `LoraMlp::new`
//! defends against this by force-materializing the frozen base (`.val()`) at
//! construction, so "reseed, then construct" alone pins the base regardless of
//! what the caller draws afterward (e.g. generating synthetic batches before the
//! first forward). Nothing tested that the defense actually matters: a naive
//! "same seed, construct twice" check passes with OR without it.
//!
//! This test interleaves an RNG draw *between* construction and first access.
//! With eager materialization the base is already fixed, so it is unchanged; if
//! the `.val()` calls in `LoraMlp::new` were removed (base left lazy), the base
//! would be drawn AFTER the throwaway draw and this assertion would fail. Its own
//! integration-test process, so the process-global ndarray RNG is uncontended.

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};
use loractl_core::{Device, LoraMlp, NdArray};

type B = NdArray;

fn fc1_weights(model: &LoraMlp<B>) -> Vec<f32> {
    model
        .fc1
        .weight
        .val()
        .into_data()
        .convert::<f32>()
        .into_vec()
        .unwrap()
}

#[test]
fn eager_materialization_pins_frozen_base_against_intervening_rng() {
    let device: Device<B> = Default::default();

    // Reference: construct immediately after seeding and read fc1 before anything
    // else draws from the RNG.
    B::seed(&device, 42);
    let m_ref = LoraMlp::<B>::new(8, 6, 4, 2, 8.0, 0.0, &device);
    let reference = fc1_weights(&m_ref);

    // Same seed, but draw from the RNG *between* construction and reading fc1.
    // Eager materialization fixed fc1 at construction time (before the throwaway),
    // so fc1 must be identical to the reference. Remove the eager `.val()` in
    // `LoraMlp::new` and fc1 becomes lazy — drawn here, after the throwaway — and
    // this assertion fails.
    B::seed(&device, 42);
    let m2 = LoraMlp::<B>::new(8, 6, 4, 2, 8.0, 0.0, &device);
    let _throwaway = Tensor::<B, 2>::random([16, 16], Distribution::Default, &device);
    let with_intervening_draw = fc1_weights(&m2);

    assert_eq!(
        with_intervening_draw, reference,
        "eager materialization must pin the frozen base at construction time, \
         independent of RNG draws that follow it"
    );
}
