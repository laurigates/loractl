//! A name-keyed set of LoRA adapters injected across a module tree — the
//! generalization at the heart of milestone 6 (#17).
//!
//! Where [`LoraLinear`](crate::lora::LoraLinear) adapts exactly one
//! [`Linear`](burn::nn::Linear), [`LoraAdapters`] holds a *dynamic set* of
//! [`LoraDelta`]s keyed by the module path of the layer each one adapts (e.g.
//! `transformer.h.0.attn.c_attn`). A forward pass through the base model, at
//! every injectable site, calls [`LoraAdapters::apply`] with that site's path:
//! if a delta is registered for it the scaled low-rank update is added to the
//! layer's output, otherwise the output passes through untouched. This is what
//! lets a single adapter set ride on top of an *unmodified* base whose LoRA
//! targets number in the dozens — a diffusion DiT — instead of hard-coding one
//! wrapped layer.
//!
//! ## Why a `Vec`, not a `HashMap`
//!
//! The deltas must be **trainable**, which means burn's autodiff, `AdamConfig`,
//! and `GradientsParams::from_grads` all have to *see* them as module children.
//! burn's `#[derive(Module)]` treats `Vec<T: Module>` as a first-class param
//! container (it visits each element by `ParamId`), but a `HashMap` is **not** a
//! `Module` — so the deltas live in a `Vec` and the parallel `targets: Vec<String>`
//! (a `#[module(skip)]` non-parameter) carries the path each `deltas[i]` adapts.
//! [`get`](LoraAdapters::get) is a linear scan over `targets`; the target count
//! is small (tens, not thousands) and each is hit once per forward, so the scan
//! is not worth a non-`Module` index.
//!
//! ## Invariant
//!
//! Like the rest of `loractl-core`, this module emits no output and imports no
//! CLI — it is pure `Module` + functions.

use crate::config::LoraConfig;
use crate::lora::LoraDelta;
use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};
use regex::Regex;

/// A dynamic, name-keyed set of LoRA deltas injected across a base module tree.
///
/// `deltas[i]` is the low-rank update for the layer at `targets[i]`. The two
/// vectors are kept in lockstep: `deltas` is a burn param container (trainable,
/// visible to autodiff and the optimizer) and `targets` is skipped metadata
/// (module paths, not tensors). Build one with [`build_adapters`] from a base
/// model's [injectable sites](crate::gpt2::Gpt2::injectable_sites), or by struct
/// literal in tests.
#[derive(Module, Debug)]
pub struct LoraAdapters<B: Backend> {
    /// The low-rank updates, one per registered target, in registration order.
    /// A burn param container so every delta is trainable.
    pub deltas: Vec<LoraDelta<B>>,
    /// The module path each `deltas[i]` adapts (e.g.
    /// `transformer.h.0.attn.c_attn`). Skipped: paths are metadata, not tensors.
    #[module(skip)]
    pub targets: Vec<String>,
}

impl<B: Backend> LoraAdapters<B> {
    /// The delta registered for `path`, or `None` if no adapter targets it.
    ///
    /// A linear scan over `targets` (see the [module docs](self) for why a
    /// `Vec`, not a map).
    pub fn get(&self, path: &str) -> Option<&LoraDelta<B>> {
        self.targets
            .iter()
            .position(|t| t == path)
            .map(|i| &self.deltas[i])
    }

    /// Add this site's low-rank update to a base layer's output.
    ///
    /// `base_out` is the output the target `Linear` already produced from `x`;
    /// `x` is that same input. If a delta targets `path`, the result is
    /// `base_out + delta.forward(x)`; otherwise `base_out` is returned unchanged
    /// (and `x` is dropped). This is the single call every injection site makes.
    pub fn apply<const D: usize>(
        &self,
        path: &str,
        x: Tensor<B, D>,
        base_out: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match self.get(path) {
            Some(delta) => base_out + delta.forward(x),
            None => base_out,
        }
    }
}

/// One injectable `Linear` site in a base model: its module path and the
/// input/output widths a delta attached to it must match.
///
/// A base model advertises its sites (e.g. via
/// [`Gpt2::injectable_sites`](crate::gpt2::Gpt2::injectable_sites)) and
/// [`build_adapters`] sizes a [`LoraDelta`] to each matched site.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraSite {
    /// The layer's module path, matched against config target patterns.
    pub path: String,
    /// Input width of the target `Linear` (`A`'s input width).
    pub d_in: usize,
    /// Output width of the target `Linear` (`B`'s output width).
    pub d_out: usize,
}

/// Build the adapter set for a base model from its injectable `sites` and the
/// run's [`LoraConfig`].
///
/// Each site whose `path` matches at least one `cfg.targets` pattern gets a
/// [`LoraDelta`] sized to that site, using the matching
/// [`TargetSpec`](crate::config::TargetSpec)'s per-target `rank`/`alpha` override
/// where present and the global `cfg.rank`/`cfg.alpha` otherwise. Sites matching
/// no pattern get no adapter. Registration order follows `sites`, so the
/// resulting `deltas`/`targets` are aligned with the model's site enumeration.
///
/// Panics if a target pattern is not a valid regex — an invalid pattern is a
/// config error that should surface immediately, not silently drop a target.
pub fn build_adapters<B: Backend>(
    sites: &[LoraSite],
    cfg: &LoraConfig,
    device: &B::Device,
) -> LoraAdapters<B> {
    let compiled: Vec<(Regex, &crate::config::TargetSpec)> = cfg
        .targets
        .iter()
        .map(|spec| {
            let re = Regex::new(&spec.pattern)
                .unwrap_or_else(|e| panic!("invalid LoRA target pattern {:?}: {e}", spec.pattern));
            (re, spec)
        })
        .collect();

    let mut deltas = Vec::new();
    let mut targets = Vec::new();
    for site in sites {
        if let Some((_, spec)) = compiled.iter().find(|(re, _)| re.is_match(&site.path)) {
            let rank = spec.rank.unwrap_or(cfg.rank) as usize;
            let alpha = spec.alpha.unwrap_or(cfg.alpha) as f64;
            deltas.push(LoraDelta::new(site.d_in, site.d_out, rank, alpha, device));
            targets.push(site.path.clone());
        }
    }
    LoraAdapters { deltas, targets }
}
