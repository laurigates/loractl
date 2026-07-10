//! Rectified-flow (flow-matching) math — milestone 8 (#19).
//!
//! Pure, backend-generic tensor helpers behind the flow-matching training
//! objective: the data↔noise interpolation, the v-prediction target, and the
//! logit-normal + shift timestep sampler. No I/O, no events, and no RNG except
//! [`sample_timesteps`] — which draws from the seeded device RNG and contains
//! *no other math*, so every deterministic transform here is pinned end to end
//! by the PyTorch golden in `tests/flow_reference.rs`.
//!
//! ## Pinned conventions (SD3 paper + diffusers + kohya-ss — all agree)
//!
//! - **Time coordinate: `t = 0` is DATA, `t = 1` is NOISE** — the
//!   SD3/diffusers/FLUX/kohya convention, and the *opposite* of the original
//!   Lipman/Liu flow-matching papers. Do not mix the two.
//! - Interpolation (SD3 Eq. 13): `x_t = (1 − t)·x_0 + t·ε`, `ε ~ N(0, I)`.
//! - Velocity / v-prediction target: `v = dx_t/dt = ε − x_0` (noise minus
//!   data).
//! - Loss weighting is **identically 1.0** for the logit-normal scheme: the
//!   SD3 `t/(1−t)` emphasis is delivered by the *sampling density* (this
//!   module), never a multiplicative loss weight — applying both would
//!   double-count it (diffusers' `compute_loss_weighting_for_sd3` returns
//!   ones for `logit_normal`).
//! - Timestep sampling (SD3 Eq. 19): `u ~ N(logit_mean, logit_std)`,
//!   `t = sigmoid(u)`, then the constant shift transform.
//!
//! Krea 2's exact constants are unpublished; the FLUX/SD3-class defaults in
//! [`FlowConfig`] are the reimplementation basis (per ADR-0004's "FLUX-style"
//! language).

use crate::config::FlowConfig;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

/// Interpolate between data and noise at timestep `t` (SD3 Eq. 13):
/// `x_t = (1 − t)·x_0 + t·ε`.
///
/// `x0` and `eps` are `[batch, dim]`; `t` is `[batch]` and broadcasts across
/// `dim`. At `t = 0` this returns the data, at `t = 1` the noise — the
/// SD3-convention direction (see the module docs).
pub fn interpolate<B: Backend>(
    x0: Tensor<B, 2>,
    eps: Tensor<B, 2>,
    t: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let t: Tensor<B, 2> = t.unsqueeze_dim(1); // [batch, 1], broadcast over dim
    let one_minus_t = t.clone().neg().add_scalar(1.0);
    x0.mul(one_minus_t) + eps.mul(t)
}

/// The v-prediction target `v = dx_t/dt = ε − x_0` — **noise minus data**.
///
/// The sign is load-bearing and easy to flip silently (a flipped target trains
/// just as well on a sign-symmetric toy); it is pinned by the golden fixture
/// and by the `flow_batches` identity test in `tests/flow_convergence.rs`.
pub fn velocity_target<B: Backend>(x0: Tensor<B, 2>, eps: Tensor<B, 2>) -> Tensor<B, 2> {
    eps - x0
}

/// The constant timestep shift `t' = shift·t / (1 + (shift − 1)·t)`
/// (SD3/kohya `discrete_flow_shift`).
///
/// `shift > 1` pushes timesteps toward 1 (noise): at `shift = 3`,
/// `t = 0.5 → 0.75`. `shift = 1` is the identity.
pub fn shift_timesteps<B: Backend>(t: Tensor<B, 1>, shift: f64) -> Tensor<B, 1> {
    let numer = t.clone().mul_scalar(shift);
    let denom = t.mul_scalar(shift - 1.0).add_scalar(1.0);
    numer.div(denom)
}

/// The full logit-normal + shift composition: map STANDARD normal draws `u`
/// to training timesteps `t' = shift_timesteps(sigmoid(u·logit_std +
/// logit_mean), shift)`.
///
/// This single pure unit is the production transform — [`sample_timesteps`]
/// is nothing but this function fed fresh RNG draws — so the golden fixture
/// pins the composition end to end (including burn's sigmoid numerics, which
/// differ from torch's by ~1e-7).
pub fn logit_to_t<B: Backend>(u: Tensor<B, 1>, cfg: FlowConfig) -> Tensor<B, 1> {
    let t = sigmoid(u.mul_scalar(cfg.logit_std).add_scalar(cfg.logit_mean));
    shift_timesteps(t, cfg.shift)
}

/// Draw `batch` training timesteps from the logit-normal + shift scheme.
///
/// Deliberately contains NO math of its own: it draws `u ~ N(0, 1)` from
/// burn's seeded device RNG and hands them to [`logit_to_t`] — inlining any
/// part of the composition here would let it escape the golden.
pub fn sample_timesteps<B: Backend>(
    batch: usize,
    cfg: FlowConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    let u = Tensor::<B, 1>::random([batch], Distribution::Normal(0.0, 1.0), device);
    logit_to_t(u, cfg)
}

/// The FLUX resolution-dependent shift for a given image sequence length.
///
/// FLUX defines `μ` linear in `image_seq_len` through the anchor points
/// `(256, 0.5)` and `(4096, 1.15)`, and applies the *dynamic* form
/// `exp(μ) / (exp(μ) + (1/t − 1))` — which equals [`shift_timesteps`]'
/// constant form with `shift = exp(μ)`. **`μ` is a LOG shift**; this function
/// returns the LINEAR shift `exp(μ)`, directly consumable as
/// [`shift_timesteps`]' / [`FlowConfig::shift`]'s value (the μ-vs-exp(μ)
/// confusion guard for M11). `exp(μ(256)) ≈ 1.6487`, `exp(μ(4096)) ≈ 3.1582`.
pub fn resolution_shift(image_seq_len: usize) -> f64 {
    let mu = 0.5 + (1.15 - 0.5) * (image_seq_len as f64 - 256.0) / (4096.0 - 256.0);
    mu.exp()
}
