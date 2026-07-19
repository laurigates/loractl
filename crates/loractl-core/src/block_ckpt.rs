//! Block-level gradient checkpointing for the MMDiT trunk (#134).
//!
//! The #132 retention-ledger attribution (ADR-0005 Addendum 2) measured the
//! monolithic autodiff training step at **67.9 GiB of logical demand pinned
//! per forward** (~3× the 24 GB card): burn-autodiff eagerly clones every
//! compute-bound tensor at checkpoint registration, and the untracked-parent
//! fallback (masks, RoPE, modulation, every `.no_grad()` base param) flips
//! most elementwise ops compute-bound, so essentially the whole trunk
//! interior — attention scores, SwiGLU intermediates, quant-site outputs —
//! is pinned from forward until backward. This module removes all of it by
//! restructuring the step, not the ops:
//!
//! 1. **Capture** ([`Mmdit::forward_capture`]): the pre-trunk stages and the
//!    trunk run on the **plain inner backend** — no autodiff graph exists,
//!    nothing is pinned — recording only each block's input residual stream
//!    (`blocks.len()` × `[b, l, features]`).
//! 2. **Head graph**: the head + flow-matching loss run on a small
//!    standalone `Autodiff` graph over the trunk output, yielding the trunk
//!    cotangent `∂loss/∂x_final` (the loss scale is folded in here, once).
//! 3. **Reverse sweep**: blocks replay last→first, each on its **own fresh
//!    `Autodiff` graph** (block lifted `.no_grad()`, the block's adapters
//!    lifted tracked with their `ParamId`s preserved), seeded by the
//!    incoming cotangent; adapter gradients go straight into the returned
//!    [`GradientsParams`], and the block-input gradient becomes the next
//!    block's seed. Each graph drops before the next lift — the peak is ONE
//!    block interior instead of 28.
//!
//! ## Why not a custom autodiff op?
//!
//! The natural design — a `QuantMatmulT`-style op whose `Backward::backward`
//! recomputes the block on an inner graph — **deadlocks on burn 0.21**,
//! verified both by source reading and an executed probe (2026-07-19): the
//! outer backward holds its graph's `state` mutex across the whole step
//! execution (`burn-autodiff/src/runtime/graph.rs`), and every `backward()`
//! call ends with `cleanup_orphaned_entries()`, which locks **every**
//! registered graph's mutex — including the outer one the same thread
//! already holds; `parking_lot` mutexes are non-reentrant. The two-phase
//! step sidesteps this by construction: no backward ever runs inside
//! another.
//!
//! ## Seeding without a seeded-backward API
//!
//! burn 0.21's `Tensor::backward()` always seeds the root with ones. The
//! VJP with an arbitrary cotangent `g` is obtained exactly via
//! `(out ⊙ g).sum().backward()` — `d/d(out) Σ out⊙g = g`, bit-exact
//! (`1.0 · g == g` in IEEE 754).
//!
//! ## Correctness boundaries
//!
//! - `ParamId`s survive `valid()`/`from_inner` (burn preserves both the id
//!   and the `require_grad` flag), so the returned [`GradientsParams`] keys
//!   line up with the trainer's `set` — the optimizer and the grad-finiteness
//!   check consume it unchanged.
//! - LoRA dropout must be 0: the capture forward runs on a non-autodiff
//!   backend where `Dropout` is identity, and a replay would redraw masks.
//!   The trainer refuses `grad_checkpointing` + `lora.dropout > 0`.
//! - Deleted with the burn 0.22 migration (#79), which makes QLoRA
//!   first-class.

use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module, Param};
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor};

use crate::adapters::LoraAdapters;
use crate::mmdit::{Mmdit, RopeTables};
use crate::quant::QuantBackend;

/// Re-mark every adapter param as a tracked leaf, **preserving its
/// `ParamId`** (the key `GradientsParams` and the optimizer route by).
///
/// Necessary because burn 0.21's `Param::clone` on an *initialized* param
/// rebuilds via `Param::initialized(id, val())`, which recomputes the
/// `require_grad` flag from the tensor — unconditionally `false` on a plain
/// backend — instead of copying the field. The per-block filtering below
/// clones inner-backend params, so the flag `valid()` carefully preserved is
/// silently gone by the time `from_inner` re-lifts them; without this, every
/// replayed block computes ZERO adapter gradients (caught by the
/// completeness assertions in `tests/block_ckpt.rs`).
fn track_adapters<G: AutodiffBackend>(mut set: LoraAdapters<G>) -> LoraAdapters<G> {
    for delta in &mut set.deltas {
        for weight in [&mut delta.lora_a.weight, &mut delta.lora_b.weight] {
            let tracked = weight.val().require_grad();
            *weight = Param::initialized(weight.id, tracked);
        }
    }
    set
}

/// One block-checkpointed training step: forward capture + head loss +
/// reverse per-block sweep. Returns the (unscaled) loss value and the
/// complete adapter [`GradientsParams`] (every `lora_a`/`lora_b` of `set`,
/// keyed by the same `ParamId`s the trainer's optimizer uses).
///
/// `mmdit_inner` is the frozen denoiser on the plain inner backend
/// (`mmdit.valid()`, hoisted once by the trainer — an Arc rewrap, not a
/// copy). All tensor inputs are inner-backend values (`.inner()` of the
/// trainer's batch tensors). `loss_scale` is folded into the head seed
/// exactly like the monolithic path's `loss * loss_scale` (AdamW is
/// scale-invariant; the scale exists to keep f16 gradients representable).
#[allow(clippy::too_many_arguments)]
pub fn checkpointed_step<AB>(
    mmdit_inner: &Mmdit<AB::InnerBackend>,
    set: &LoraAdapters<AB>,
    img: Tensor<AB::InnerBackend, 3>,
    context: Tensor<AB::InnerBackend, 4>,
    t: Tensor<AB::InnerBackend, 1>,
    pos: Tensor<AB::InnerBackend, 3>,
    mask: Tensor<AB::InnerBackend, 2>,
    target: Tensor<AB::InnerBackend, 3>,
    loss_scale: f32,
) -> (f32, GradientsParams)
where
    AB: AutodiffBackend + QuantBackend,
    AB::InnerBackend: QuantBackend,
{
    type G<IB> = Autodiff<IB>;

    // Phase 1 — capture: pre-trunk + trunk on the plain backend, adapter
    // values included (their contribution is part of the forward), storing
    // each block's input. No graph, no pinning.
    let set_inner = set.valid();
    let cap = mmdit_inner.forward_capture(img, context, t, pos, mask, Some(&set_inner));

    // Per-block adapter subsets, filtered by dot-terminated prefix so
    // `blocks.1.` cannot match `blocks.10.attn.wq`. Parallel vecs keep the
    // site paths adapter-lookup-compatible inside the replayed block.
    let per_block: Vec<LoraAdapters<AB::InnerBackend>> = (0..mmdit_inner.blocks.len())
        .map(|i| {
            let prefix = format!("blocks.{i}.");
            let (deltas, targets) = set_inner
                .targets
                .iter()
                .zip(&set_inner.deltas)
                .filter(|(path, _)| path.starts_with(&prefix))
                .map(|(path, delta)| (delta.clone(), path.clone()))
                .unzip();
            LoraAdapters { deltas, targets }
        })
        .collect();

    // Phase 2a — head graph: trunk output re-tracked, head lifted frozen,
    // the trainer's exact loss replicated, loss scale folded into the seed.
    let (loss_value, mut grad_x) = {
        let x2: Tensor<G<AB::InnerBackend>, 3> = Tensor::from_inner(cap.x_final).require_grad();
        let last2 = crate::mmdit::LastLayer::from_inner(mmdit_inner.last.clone()).no_grad();
        let t2 = Tensor::from_inner(cap.t_embed.clone());
        let pred = last2
            .forward(x2.clone(), t2)
            .narrow(1, cap.txt_len, cap.img_len);
        let diff = pred - Tensor::from_inner(target);
        let loss = diff.clone().mul(diff).mean();
        let loss_value: f32 = loss.clone().into_scalar().elem();
        let mut grads = (loss * loss_scale).backward();
        let grad_x = x2
            .grad_remove(&mut grads)
            .expect("the trunk output feeds the loss — its gradient must exist");
        (loss_value, grad_x)
    };

    // Phase 2b — reverse sweep. Each iteration builds ONE standalone graph,
    // extracts its gradients, and drops it before the next lift.
    let mut out = GradientsParams::new();
    for i in (0..mmdit_inner.blocks.len()).rev() {
        let block2 =
            crate::mmdit::SingleStreamBlock::from_inner(mmdit_inner.blocks[i].clone()).no_grad();
        let adapters2: LoraAdapters<G<AB::InnerBackend>> =
            track_adapters(LoraAdapters::from_inner(per_block[i].clone()));
        let x2: Tensor<G<AB::InnerBackend>, 3> =
            Tensor::from_inner(cap.block_inputs[i].clone()).require_grad();
        let tvec2 = Tensor::from_inner(cap.tvec.clone());
        let rope2 = RopeTables {
            cos: Tensor::from_inner(cap.rope.cos.clone()),
            sin: Tensor::from_inner(cap.rope.sin.clone()),
        };
        let mask2 = Tensor::from_inner(cap.mask4.clone());

        let out2 = block2.forward(
            x2.clone(),
            tvec2,
            &rope2,
            mask2,
            Some(&adapters2),
            &format!("blocks.{i}"),
        );

        // VJP seed: d/d(out2) Σ out2⊙g = g, bit-exact.
        let seed = Tensor::from_inner(grad_x);
        let mut grads = (out2 * seed).sum().backward();

        for delta in &adapters2.deltas {
            for weight in [&delta.lora_a.weight, &delta.lora_b.weight] {
                let grad = weight.val().grad_remove(&mut grads).expect(
                    "every filtered adapter site is applied inside its block — \
                     its gradient must exist",
                );
                out.register::<AB::InnerBackend, 2>(weight.id, grad);
            }
        }
        grad_x = x2
            .grad_remove(&mut grads)
            .expect("the block input feeds the block output — its gradient must exist");
    }

    (loss_value, out)
}
