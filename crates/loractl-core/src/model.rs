//! A minimal LoRA-adapted MLP classifier.
//!
//! [`LoraMlp`] is the tiny model the [`BurnTrainer`](crate::BurnTrainer) trains
//! in milestone 2 (#1). It exists to prove the LoRA training loop end-to-end —
//! forward, loss, autodiff, optimizer step, and the freeze — on a real (if
//! small) classification task before any large base model is involved.
//!
//! The architecture is deliberately spartan:
//!
//! ```text
//! x  ──▶  fc1 (Linear, FROZEN)  ──▶  ReLU  ──▶  fc2 (LoraLinear)  ──▶  logits
//! ```
//!
//! Both `fc1` and `fc2.base` are **frozen**; the *only* trainable parameters in
//! the entire model are `fc2.lora_a` and `fc2.lora_b`. `fc1` acts as a fixed
//! random-feature projection (à la a random-features kernel), and the LoRA
//! factors on the output layer are the low-rank readout that actually learns.
//! This is the strongest possible base-unchanged claim: the frozen weights carry
//! no gradient, so an optimizer leaves them byte-identical.

use crate::lora::LoraLinear;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::{Tensor, backend::Backend};

/// A frozen dense projection followed by a LoRA-adapted readout.
///
/// See the [module docs](self) for the architecture. Construct with
/// [`LoraMlp::new`].
#[derive(Module, Debug)]
pub struct LoraMlp<B: Backend> {
    /// Frozen dense feature projection `d_in -> hidden` (random fixed features).
    pub fc1: Linear<B>,
    /// LoRA-adapted readout `hidden -> out`; its base is frozen and only the
    /// low-rank factors `lora_a`/`lora_b` are trained.
    pub fc2: LoraLinear<B>,
}

impl<B: Backend> LoraMlp<B> {
    /// Build a fresh `LoraMlp`.
    ///
    /// `fc1` is a freshly initialized dense layer, immediately frozen with the
    /// same helper [`LoraLinear`] uses. `fc2` is a [`LoraLinear`] whose base is
    /// frozen inside `from_base`; `rank`/`alpha` come from the run's LoRA config.
    pub fn new(
        d_in: usize,
        hidden: usize,
        out: usize,
        rank: usize,
        alpha: f64,
        device: &B::Device,
    ) -> Self {
        let fc1 = crate::lora::freeze(LinearConfig::new(d_in, hidden).with_bias(true).init(device));
        let fc2 = LoraLinear::new(hidden, out, rank, alpha, true, device);
        Self { fc1, fc2 }
    }

    /// Forward pass: `fc2(relu(fc1(x)))`. Input `[batch, d_in]`, output
    /// `[batch, out]` (logits).
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(self.fc1.forward(x));
        self.fc2.forward(h)
    }
}
