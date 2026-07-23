//! Optional **merge-at-load** of an external LoRA *training adapter* into the
//! frozen base weights — the Krea-2-Turbo assistant-LoRA seam (#83).
//!
//! ## What this is
//!
//! Krea-2-Turbo is a distilled (few-step) model. ai-toolkit's only
//! distillation-aware way to *train* on turbo is an "assistant" LoRA
//! (`ostris/krea2_turbo_training_adapter`): it is merged into the transformer at
//! strength `+1.0` **before** training — nudging the distilled weights back
//! toward a raw-like state so the objective is well-behaved — and inverted at
//! `-1.0` only for preview sampling to restore true turbo weights
//! (`krea2.py::load_training_adapter`).
//!
//! loractl does **not** sample during training, so only the one-time
//! merge-at-load applies here — a much smaller port than ai-toolkit's live
//! adapter machinery. For each targeted site the frozen base weight `W` is
//! updated in place:
//!
//! ```text
//! W += (alpha / rank) · B · A
//! ```
//!
//! where `A` (`lora_down` / `lora_A`) is `[rank, d_in]` and `B` (`lora_up` /
//! `lora_B`) is `[d_out, rank]` on disk (PyTorch/diffusers `[out, in]`
//! convention). burn's `Linear.weight` is `[d_in, d_out]`, so the delta is
//! folded in that layout as `(alpha/rank) · Aᵀ · Bᵀ` (identically
//! `((B·A)ᵀ)` — see the transpose note on [`merge_delta`]).
//!
//! ## Interop semantics (document this, it is load-bearing)
//!
//! A LoRA trained here is trained against **turbo + assistant adapter**, but is
//! deployed on **plain turbo** (ComfyUI / Krea-2-Turbo) — exactly as in
//! ai-toolkit, which inverts the assistant merge before it ever exports. The
//! merge changes only the *training-time* frozen base; the trained low-rank
//! delta this crate writes is unaffected in its keys or layout, so the exported
//! adapter drops onto stock turbo unchanged.
//!
//! ## Scope
//!
//! The merge folds into full-precision (`BaseLinear::Plain`) sites only. It is
//! rejected up front in the trainer's validation when combined with
//! `compute.quant` (int8/int4), where the base is stored quantized and an
//! in-place additive merge is not available — a documented follow-up (the seam
//! is the f32 transient in `diffusion_trainer::load_quant_module`).

use crate::event::TrainEvent;
use crate::mmdit::{BaseLinear, Mmdit};
use anyhow::{Context, Result, bail};
use burn::module::Param;
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

/// The four on-disk factor-key suffixes we accept, mapping to `(is_up, suffix)`.
/// diffusers/PEFT names factors `lora_A`/`lora_B`; kohya-ss names them
/// `lora_down`/`lora_up`. Both store `down = A [rank, d_in]` and
/// `up = B [d_out, rank]`, so the two conventions differ only in spelling.
const DOWN_SUFFIXES: [&str; 2] = [".lora_A.weight", ".lora_down.weight"];
const UP_SUFFIXES: [&str; 2] = [".lora_B.weight", ".lora_up.weight"];

/// One targeted site's low-rank factors, already lifted into **burn** layout
/// (`a: [d_in, rank]`, `b: [rank, d_out]`) and its effective `scaling`
/// (`alpha / rank`).
struct MergeSite<B: Backend> {
    /// The base-linear module path this delta folds into (e.g.
    /// `blocks.0.attn.wq`), after stripping any `diffusion_model.` prefix.
    path: String,
    a: Tensor<B, 2>,
    b: Tensor<B, 2>,
    scaling: f64,
}

/// A parsed external LoRA training adapter: per-site low-rank factors ready to
/// fold into a frozen base with [`merge_training_adapter`].
pub struct TrainingAdapter<B: Backend> {
    sites: Vec<MergeSite<B>>,
}

/// The frozen-base weight delta for one site, in burn `Linear` `[d_in, d_out]`
/// layout: `scaling · (A · B)` with `A: [d_in, rank]`, `B: [rank, d_out]`.
///
/// This equals `scaling · (B_disk · A_disk)ᵀ` where `A_disk = Aᵀ` `[rank, d_in]`
/// and `B_disk = Bᵀ` `[d_out, rank]` are the on-disk factors — i.e. the standard
/// PyTorch weight delta `(alpha/rank)·B·A` (`[d_out, d_in]`) transposed into
/// burn's `[d_in, d_out]` weight layout. Public so the golden test can pin the
/// math directly against the Python reference.
pub fn merge_delta<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 2>, scaling: f64) -> Tensor<B, 2> {
    a.matmul(b).mul_scalar(scaling)
}

impl<B: Backend> TrainingAdapter<B> {
    /// Parse a LoRA `.safetensors` (diffusers/PEFT `lora_A`/`lora_B` or kohya
    /// `lora_down`/`lora_up`, optionally `diffusion_model.`-prefixed) into
    /// per-site burn-layout factors.
    ///
    /// Rank is auto-detected from each `down` factor's leading dim; when a
    /// per-site `.alpha` scalar is present its `alpha/rank` is used, otherwise
    /// `scaling = 1.0` (the ai-toolkit assistant-merge default: the factors are
    /// merged at unit strength, i.e. `alpha = rank`).
    pub fn from_file(path: &Path, device: &B::Device) -> Result<Self> {
        // fp8-aware, though a LoRA adapter is virtually always bf16/f16/f32;
        // `to_data().convert_dtype(F32)` normalizes whatever it is.
        let snapshots = if crate::fp8::is_fp8_checkpoint(path)? {
            crate::fp8::load_fp8_snapshots(path)
                .with_context(|| format!("loading training adapter (fp8) {}", path.display()))?
        } else {
            crate::fp8::load_plain_snapshots(path)
                .with_context(|| format!("loading training adapter {}", path.display()))?
        };

        // name -> f32 [r, c] factor matrix, materialized once; and the separate
        // `.alpha` scalars (any rank), read straight to f32.
        let mut tensors: BTreeMap<String, Tensor<B, 2>> = BTreeMap::new();
        let mut alphas: BTreeMap<String, f32> = BTreeMap::new();
        for snap in snapshots {
            let name = snap.full_path();
            let data = snap
                .to_data()
                .map_err(|e| {
                    anyhow::anyhow!("forcing tensor {name} from {}: {e:?}", path.display())
                })?
                .convert_dtype(DType::F32);
            if name.ends_with(".alpha") {
                if let Ok(v) = data.into_vec::<f32>()
                    && let Some(first) = v.first()
                {
                    alphas.insert(name, *first);
                }
                continue;
            }
            if data.shape.len() != 2 {
                // Non-matrix, non-alpha tensors (e.g. bias, dora scales) are not
                // part of the merge — ignore them.
                continue;
            }
            tensors.insert(name, Tensor::from_data(data, device));
        }

        // Collect the down/up pairs keyed by site prefix.
        let mut sites: Vec<MergeSite<B>> = Vec::new();
        // Deterministic order: BTreeMap iteration is sorted, and we key sites by
        // the down factor.
        for (name, down) in tensors
            .iter()
            .filter_map(|(n, t)| down_prefix(n).map(|p| (p, t)))
        {
            let prefix = name; // the site key WITH any diffusion_model. prefix
            let up = up_for_prefix(&tensors, &prefix).ok_or_else(|| {
                anyhow::anyhow!(
                    "training adapter {}: site {prefix} has a down factor but no matching up \
                     (lora_B/lora_up) factor",
                    path.display()
                )
            })?;

            let down_dims = down.dims();
            let up_dims = up.dims();
            let rank = down_dims[0];
            if up_dims[1] != rank {
                bail!(
                    "training adapter {}: site {prefix} rank mismatch — down is {down_dims:?} \
                     (rank {rank}) but up is {up_dims:?} (rank {})",
                    path.display(),
                    up_dims[1]
                );
            }

            // Disk `down [rank, d_in]` -> burn `A [d_in, rank]`;
            // disk `up [d_out, rank]` -> burn `B [rank, d_out]`.
            let a = down.clone().transpose();
            let b = up.clone().transpose();

            let scaling = match alphas.get(&format!("{prefix}.alpha")) {
                Some(alpha) => *alpha as f64 / rank as f64,
                None => 1.0,
            };

            sites.push(MergeSite {
                path: strip_prefix(&prefix).to_string(),
                a,
                b,
                scaling,
            });
        }

        if sites.is_empty() {
            bail!(
                "training adapter {} contains no LoRA factor pairs (looked for \
                 *.lora_A/.lora_down + *.lora_B/.lora_up)",
                path.display()
            );
        }

        Ok(Self { sites })
    }

    /// Number of parsed sites (targets present in the file).
    pub fn len(&self) -> usize {
        self.sites.len()
    }

    /// Whether the adapter has no sites (always `false` post-parse — kept for
    /// clippy's `len_without_is_empty`).
    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }
}

/// Merge an external training adapter at `path` into `mmdit`'s frozen base,
/// folding `(alpha/rank)·B·A` into each targeted full-precision site.
///
/// Every site the adapter names must resolve to one of the MMDiT's injectable
/// base linears with matching shapes; a stray or misshaped key is a hard error
/// (never a silent no-op — an unmatched LoRA key is the worst failure shape, see
/// `.claude/rules/testing.md`). A quantized (`BaseLinear::Quant`) target is also
/// a hard error: the merge needs full precision. Returns the number of sites
/// merged and emits a summary [`TrainEvent::Warning`].
pub fn merge_training_adapter<B: Backend>(
    mmdit: &mut Mmdit<B>,
    path: &Path,
    device: &B::Device,
    sink: &mut dyn FnMut(TrainEvent),
) -> Result<usize> {
    let adapter = TrainingAdapter::<B>::from_file(path, device)?;

    // Index the parsed sites by path; drain as we consume so leftovers = keys
    // that matched no injectable site.
    let mut by_path: BTreeMap<String, MergeSite<B>> = adapter
        .sites
        .into_iter()
        .map(|s| (s.path.clone(), s))
        .collect();

    let mut merged = 0usize;
    for (site_path, base) in mmdit.base_linears_mut() {
        let Some(site) = by_path.remove(&site_path) else {
            continue;
        };
        match base {
            BaseLinear::Plain(lin) => {
                let weight = lin.weight.val();
                let [d_in, d_out] = weight.dims();
                let a_dims = site.a.dims();
                let b_dims = site.b.dims();
                if a_dims[0] != d_in || b_dims[1] != d_out {
                    bail!(
                        "training adapter {}: site {site_path} shape mismatch — base weight is \
                         [{d_in}, {d_out}] but adapter factors give A {a_dims:?} · B {b_dims:?}",
                        path.display()
                    );
                }
                let delta = merge_delta(site.a, site.b, site.scaling);
                // The base stays frozen: a fresh untracked Param (`.no_grad()`
                // is already applied upstream, and a computed tensor carries no
                // grad tracking).
                lin.weight = Param::from_tensor(weight + delta);
                merged += 1;
            }
            BaseLinear::Quant(_) => bail!(
                "training adapter {}: site {site_path} is quantized — merge-at-load requires a \
                 full-precision base; drop compute.quant for training-adapter runs (quant-path \
                 merge is the tracked #83 follow-up)",
                path.display()
            ),
        }
    }

    if !by_path.is_empty() {
        let unmatched: Vec<&String> = by_path.keys().collect();
        bail!(
            "training adapter {} names sites the MMDiT has no injectable base linear for: {:?} — \
             the trunk advertises blocks.<i>.{{attn.wq,attn.wk,attn.wv,attn.wo,mlp.gate,mlp.up,\
             mlp.down}}",
            path.display(),
            unmatched
        );
    }

    sink(TrainEvent::Warning {
        message: format!(
            "training adapter: merged {merged} assistant-LoRA site(s) into the frozen base from {}",
            path.display()
        ),
    });
    Ok(merged)
}

/// If `name` is a down-factor key (`*.lora_A.weight` / `*.lora_down.weight`),
/// return its site prefix (the key with the suffix removed); else `None`.
fn down_prefix(name: &str) -> Option<String> {
    DOWN_SUFFIXES
        .iter()
        .find_map(|s| name.strip_suffix(s).map(str::to_string))
}

/// Find the up factor paired with a down-factor `prefix`, trying both spellings.
fn up_for_prefix<B: Backend>(
    tensors: &BTreeMap<String, Tensor<B, 2>>,
    prefix: &str,
) -> Option<Tensor<B, 2>> {
    UP_SUFFIXES
        .iter()
        .find_map(|s| tensors.get(&format!("{prefix}{s}")).cloned())
}

/// Strip a leading `diffusion_model.` (the ComfyUI/ostris convention) so the
/// remaining key is a bare MMDiT module path.
fn strip_prefix(prefix: &str) -> &str {
    prefix.strip_prefix("diffusion_model.").unwrap_or(prefix)
}
