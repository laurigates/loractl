//! The LoRA adapter module.
//!
//! [`LoraLinear`] is the low-rank adapter at the heart of the trainer: it wraps
//! a base [`Linear`] whose weights are **frozen**, and learns two small factors
//! `A` and `B` instead. The adapted forward pass is
//!
//! ```text
//! y = base(x) + (alpha / rank) · B(A(x))
//! ```
//!
//! `A` projects the input down to `rank` dimensions and `B` projects it back up
//! to the output width, so the update `B·A` is rank-limited — that is the whole
//! point of LoRA. `B` is **zero-initialized**, which makes the adapter a no-op
//! at step 0: the module's initial output is exactly the base model's, and the
//! adapter only departs from it as training moves `B` away from zero. `A` and
//! `B` are expressed as bias-less [`Linear`] layers so the forward pass handles
//! any input rank (`[.., d_input]`) the way burn's own `linear` does.
//!
//! This is the correctness-harness building block for milestone 2 (#1): a
//! `Trainer` that trains one of these on a tiny model proves the LoRA math, the
//! freeze, and autodiff in isolation before a real base model is involved.

use burn::module::{Initializer, Module};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::{Tensor, backend::Backend};

/// A [`Linear`] layer adapted with a low-rank (LoRA) update.
///
/// The `base` layer is frozen (its parameters do not require gradients and are
/// never touched by an optimizer); only `lora_a` and `lora_b` are trained. Use
/// [`LoraLinear::new`] to wrap a fresh base layer, or [`LoraLinear::from_base`]
/// to adapt an already-loaded (e.g. pretrained) [`Linear`].
#[derive(Module, Debug)]
pub struct LoraLinear<B: Backend> {
    /// The frozen base transform.
    pub base: Linear<B>,
    /// Down-projection `A`: `d_input -> rank`, no bias. Trainable.
    pub lora_a: Linear<B>,
    /// Up-projection `B`: `rank -> d_output`, no bias, zero-initialized.
    /// Trainable.
    pub lora_b: Linear<B>,
    /// The LoRA scaling factor `alpha / rank`, applied to the adapter output.
    pub scaling: f64,
    /// Dropout applied to the adapter's input before the down-projection `A`
    /// (only the low-rank path; the base is never dropped). burn's `Dropout` is
    /// identity at `prob = 0.0` and on a non-autodiff backend, so it is a no-op
    /// at inference/sampling and draws no RNG when disabled.
    pub dropout: Dropout,
}

impl<B: Backend> LoraLinear<B> {
    /// Create a LoRA-adapted linear layer with a freshly initialized base.
    ///
    /// `rank` is clamped to at least 1. `alpha` is the LoRA scaling numerator;
    /// the effective scale applied to the adapter is `alpha / rank`.
    /// `dropout_prob` is the adapter-input dropout applied during training.
    pub fn new(
        d_input: usize,
        d_output: usize,
        rank: usize,
        alpha: f64,
        bias: bool,
        dropout_prob: f64,
        device: &B::Device,
    ) -> Self {
        let base = LinearConfig::new(d_input, d_output)
            .with_bias(bias)
            .init(device);
        Self::from_base(base, rank, alpha, dropout_prob, device)
    }

    /// Adapt an existing base [`Linear`], freezing it and attaching a fresh
    /// LoRA adapter sized to its input/output widths.
    ///
    /// This is the entry point milestone 3 (#2) uses once a real base model's
    /// weights have been loaded into a [`Linear`].
    pub fn from_base(
        base: Linear<B>,
        rank: usize,
        alpha: f64,
        dropout_prob: f64,
        device: &B::Device,
    ) -> Self {
        let rank = rank.max(1);
        let [d_input, d_output] = base.weight.dims();
        let base = freeze(base);
        let (lora_a, lora_b) = init_factors(d_input, d_output, rank, device);

        Self {
            base,
            lora_a,
            lora_b,
            scaling: alpha / rank as f64,
            dropout: DropoutConfig::new(dropout_prob).init(),
        }
    }

    /// The adapted forward pass: `base(x) + (alpha / rank) · B(A(dropout(x)))`.
    ///
    /// Dropout is applied only to the low-rank path's input (never the base),
    /// and only during training — it is identity at inference and at
    /// `dropout_prob = 0.0`. Works for any input rank `[.., d_input]`; the
    /// output has the same rank with the last dimension replaced by `d_output`.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let base = self.base.forward(input.clone());
        let delta = self
            .lora_b
            .forward(self.lora_a.forward(self.dropout.forward(input)));
        base.add(delta.mul_scalar(self.scaling))
    }
}

/// Initialize a fresh, bias-less LoRA factor pair sized to a linear of width
/// `d_input -> d_output` at the given `rank`.
///
/// `A` (`d_input -> rank`) gets burn's default (Kaiming) init; `B`
/// (`rank -> d_output`) is zero-initialized, so any adapter built from this pair
/// starts as an exact no-op and only departs from the base as training moves `B`
/// off zero. This is the shared init used by both [`LoraLinear::from_base`] (the
/// base-owning adapter) and [`LoraDelta`] (the base-free delta), so the two
/// cannot drift in how they seed their factors. `rank` is expected pre-clamped
/// (`>= 1`) by the caller — the caller also derives `scaling = alpha / rank`.
pub(crate) fn init_factors<B: Backend>(
    d_input: usize,
    d_output: usize,
    rank: usize,
    device: &B::Device,
) -> (Linear<B>, Linear<B>) {
    let lora_a = LinearConfig::new(d_input, rank)
        .with_bias(false)
        .init(device);
    let lora_b = LinearConfig::new(rank, d_output)
        .with_bias(false)
        .with_initializer(Initializer::Zeros)
        .init(device);
    (lora_a, lora_b)
}

/// A base-free low-rank update: just the trainable factors `A`/`B` and the
/// scaling, whose forward returns **only** the scaled delta
/// `(alpha / rank) · B(A(x))` — not `base(x) + delta`.
///
/// This is the injection-friendly counterpart to [`LoraLinear`]. Where
/// `LoraLinear` owns and freezes its base, a `LoraDelta` owns no base at all:
/// the base is whatever `Linear` already lives in the target module tree, and
/// the delta is added to that layer's output at the injection site (see
/// [`crate::adapters::LoraAdapters`]). Keeping the delta base-free is what lets
/// one name-keyed set of adapters ride on top of an unmodified base model — the
/// mechanism M6 (#17) generalizes to a diffusion DiT with dozens of targets.
///
/// Like [`LoraLinear`], `B` is zero-initialized, so a freshly built delta
/// contributes exactly zero until training moves it — the no-op-at-attach
/// invariant every injection site relies on.
#[derive(Module, Debug)]
pub struct LoraDelta<B: Backend> {
    /// Down-projection `A`: `d_input -> rank`, no bias. Trainable.
    pub lora_a: Linear<B>,
    /// Up-projection `B`: `rank -> d_output`, no bias, zero-initialized.
    /// Trainable.
    pub lora_b: Linear<B>,
    /// The LoRA scaling factor `alpha / rank`, applied to the adapter output.
    pub scaling: f64,
    /// Dropout applied to the delta's input before the down-projection `A`,
    /// during training only (identity at inference and at `prob = 0.0`).
    pub dropout: Dropout,
}

impl<B: Backend> LoraDelta<B> {
    /// Build a fresh delta sized `d_input -> d_output` at `rank`.
    ///
    /// `rank` is clamped to at least 1; the effective scale is `alpha / rank`.
    /// `B` is zero-initialized, so the delta starts as an exact no-op.
    /// `dropout_prob` is the delta-input dropout applied during training.
    pub fn new(
        d_input: usize,
        d_output: usize,
        rank: usize,
        alpha: f64,
        dropout_prob: f64,
        device: &B::Device,
    ) -> Self {
        let rank = rank.max(1);
        let (lora_a, lora_b) = init_factors(d_input, d_output, rank, device);
        Self {
            lora_a,
            lora_b,
            scaling: alpha / rank as f64,
            dropout: DropoutConfig::new(dropout_prob).init(),
        }
    }

    /// The scaled low-rank delta `(alpha / rank) · B(A(dropout(x)))`.
    ///
    /// Dropout is applied only during training (identity at inference and at
    /// `prob = 0.0`). Works for any input rank `[.., d_input]`; the output has
    /// the same rank with the last dimension replaced by `d_output`. Unlike
    /// [`LoraLinear::forward`], this does **not** add a base term — the caller
    /// adds it to the target layer's own output at the injection site.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.lora_b
            .forward(self.lora_a.forward(self.dropout.forward(input)))
            .mul_scalar(self.scaling)
    }
}

/// Return `linear` with every parameter marked as *not* requiring gradients, so
/// autodiff produces no gradient for it and an optimizer leaves it untouched.
///
/// `pub(crate)` so [`crate::model::LoraMlp`] can freeze its dense feature layer
/// with the same proven helper. Kept internal — not part of the public API.
pub(crate) fn freeze<B: Backend>(linear: Linear<B>) -> Linear<B> {
    Linear {
        weight: linear.weight.set_require_grad(false),
        bias: linear.bias.map(|b| b.set_require_grad(false)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::{Distribution, Tolerance};

    /// Plain CPU backend for value/shape checks.
    type TB = NdArray;
    /// Autodiff-wrapped backend for gradient checks.
    type AB = Autodiff<NdArray>;

    #[test]
    fn constructs_with_expected_shapes() {
        let device = Default::default();
        let lora = LoraLinear::<TB>::new(4, 3, 2, 8.0, true, 0.0, &device);

        assert_eq!(lora.base.weight.dims(), [4, 3]);
        assert_eq!(lora.lora_a.weight.dims(), [4, 2]);
        assert_eq!(lora.lora_b.weight.dims(), [2, 3]);
        assert_eq!(lora.scaling, 8.0 / 2.0);
    }

    #[test]
    fn rank_is_clamped_to_at_least_one() {
        let device = Default::default();
        let lora = LoraLinear::<TB>::new(4, 3, 0, 8.0, false, 0.0, &device);

        assert_eq!(lora.lora_a.weight.dims(), [4, 1]);
        assert_eq!(lora.lora_b.weight.dims(), [1, 3]);
        // scaling = alpha / clamped_rank, so it must be finite (not alpha / 0).
        assert_eq!(lora.scaling, 8.0);
    }

    #[test]
    fn forward_preserves_output_shape() {
        let device = Default::default();
        let lora = LoraLinear::<TB>::new(4, 3, 2, 8.0, false, 0.0, &device);

        let x = Tensor::<TB, 2>::random([5, 4], Distribution::Default, &device);
        let y = lora.forward(x);

        assert_eq!(y.dims(), [5, 3]);
    }

    #[test]
    fn zero_initialized_adapter_is_identity() {
        // `B` is zero-initialized, so the adapter contributes nothing at step 0:
        // the LoRA forward must equal the frozen base's forward exactly.
        let device = Default::default();
        let lora = LoraLinear::<TB>::new(6, 4, 3, 16.0, true, 0.0, &device);

        let x = Tensor::<TB, 2>::random([8, 6], Distribution::Default, &device);
        let base_out = lora.base.forward(x.clone());
        let lora_out = lora.forward(x);

        base_out
            .into_data()
            .assert_approx_eq::<f32>(&lora_out.into_data(), Tolerance::default());
    }

    #[test]
    fn base_is_frozen_and_adapter_is_trainable() {
        // A backward pass must produce a gradient for the trainable adapter but
        // none for the frozen base — the autodiff-level proof of the freeze.
        let device = Default::default();
        let lora = LoraLinear::<AB>::new(4, 3, 2, 8.0, true, 0.0, &device);

        let x = Tensor::<AB, 2>::random([5, 4], Distribution::Default, &device);
        let loss = lora.forward(x).sum();
        let grads = loss.backward();

        assert!(
            lora.base.weight.val().grad(&grads).is_none(),
            "frozen base must receive no gradient"
        );
        assert!(
            lora.lora_b.weight.val().grad(&grads).is_some(),
            "trainable adapter must receive a gradient"
        );
    }

    #[test]
    fn dropout_is_identity_at_inference() {
        // burn's `Dropout` is identity on a non-autodiff backend, so a high
        // dropout prob must NOT perturb the forward at inference/sampling: two
        // forwards of the same module are bit-identical and no RNG is consumed.
        let device = Default::default();
        let lora = LoraLinear::<TB>::new(6, 4, 3, 16.0, true, 0.9, &device);

        let x = Tensor::<TB, 2>::random([8, 6], Distribution::Default, &device);
        let a = lora.forward(x.clone());
        let b = lora.forward(x);

        a.into_data()
            .assert_approx_eq::<f32>(&b.into_data(), Tolerance::default());
    }

    #[test]
    fn dropout_is_active_during_training() {
        // On an autodiff (training) backend a non-zero dropout prob must
        // randomize its input: two applications to the same tensor draw
        // independent Bernoulli masks and so differ. A prob of 0.0 (or an
        // unwired dropout) would make them identical — the model-level guard
        // that the configured dropout is actually applied during training.
        let device = Default::default();
        let lora = LoraLinear::<AB>::new(6, 4, 3, 16.0, true, 0.9, &device);

        let x = Tensor::<AB, 2>::random([32, 6], Distribution::Default, &device);
        let a = lora.dropout.forward(x.clone());
        let b = lora.dropout.forward(x);

        assert_ne!(
            a.into_data(),
            b.into_data(),
            "dropout must randomize its input during training (autodiff backend)"
        );
    }
}
