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
use regex::Regex;
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
    /// Whether the scaling for **every** site was determined from the file
    /// (a per-site `.alpha` scalar, or the `__metadata__` `ss_network_alpha`/
    /// `ss_network_dim` a kohya/ai-toolkit adapter carries). `false` means at
    /// least one site fell back to unit scaling (`alpha = rank`) — surfaced as
    /// a warning at merge time, since a silent `alpha != rank` would merge at
    /// the wrong strength.
    scaling_known: bool,
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
        // Global fallback scaling from the safetensors `__metadata__`
        // (`ss_network_alpha`/`ss_network_dim`, the kohya/ai-toolkit convention
        // the real Krea-2-Turbo assistant adapter uses) — applied to any site
        // that lacks a per-site `.alpha` scalar.
        let meta_scaling = metadata_scaling(path)?;

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
            // DoRA (weight-decomposed LoRA) carries per-site magnitude vectors
            // alongside `lora_A`/`lora_B`; folding only the low-rank part is a
            // silent *partial* merge. Refuse rather than drop it (this design is
            // fail-loud everywhere else).
            if name.contains("dora") || name.contains("magnitude") {
                bail!(
                    "training adapter {}: key {name} looks like DoRA (weight-decomposed LoRA); \
                     only plain LoRA merge is supported — a partial merge would be silently wrong",
                    path.display()
                );
            }
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
                // Non-matrix, non-alpha tensors (e.g. a bias) are not part of the
                // merge — ignore them.
                continue;
            }
            tensors.insert(name, Tensor::from_data(data, device));
        }

        // Collect the down/up pairs keyed by site prefix.
        let mut sites: Vec<MergeSite<B>> = Vec::new();
        let mut scaling_known = true;
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

            // Per-site `.alpha` wins; else the file's global metadata alpha/dim;
            // else unit scaling (and flag it so merge warns).
            let scaling = match alphas.get(&format!("{prefix}.alpha")) {
                Some(alpha) => *alpha as f64 / rank as f64,
                None => match meta_scaling {
                    Some(s) => s,
                    None => {
                        scaling_known = false;
                        1.0
                    }
                },
            };

            // Strip `diffusion_model.`, then apply the MMDiT's own checkpoint
            // remap (the `nn.Sequential` index renames, `.mod.lin`) so the site
            // matches the module paths `all_base_linears_mut` advertises.
            sites.push(MergeSite {
                path: remap_site_path::<B>(strip_prefix(&prefix)),
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

        Ok(Self {
            sites,
            scaling_known,
        })
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
/// Matches over **every** base linear in the trunk, the text-fusion blocks, and
/// the time/text projections ([`Mmdit::all_base_linears_mut`]) — not only the
/// LoRA-injectable subset — so a broad assistant adapter (which may target
/// `attn.gate`, the fusion blocks, or the `t*`/`txt*` projections) folds in
/// fully rather than blocking the run. Every site the adapter names must still
/// resolve to one of those base linears with matching shapes; a stray or
/// misshaped key is a hard error (never a silent no-op — an unmatched LoRA key
/// is the worst failure shape, see `.claude/rules/testing.md`). A quantized
/// (`BaseLinear::Quant`) target is also a hard error: the merge needs full
/// precision. Returns the number of sites merged and emits a summary
/// [`TrainEvent::Warning`].
pub fn merge_training_adapter<B: Backend>(
    mmdit: &mut Mmdit<B>,
    path: &Path,
    device: &B::Device,
    sink: &mut dyn FnMut(TrainEvent),
) -> Result<usize> {
    let adapter = TrainingAdapter::<B>::from_file(path, device)?;
    let site_count = adapter.sites.len();

    // No scaling info at all (no per-site `.alpha`, no `__metadata__`
    // alpha/dim) → at least one site defaulted to unit scaling. Surface it: an
    // adapter trained with `alpha != rank` would then merge at the wrong
    // strength — a quiet quality regression, the hardest kind to notice.
    if !adapter.scaling_known {
        sink(TrainEvent::Warning {
            message: format!(
                "training adapter {}: no `.alpha` scalar and no `ss_network_alpha`/`ss_network_dim` \
                 metadata — some site(s) merged at unit scaling (alpha = rank). If this adapter was \
                 trained with alpha != rank, its strength is wrong.",
                path.display()
            ),
        });
    }

    // Index the parsed sites by path; drain as we consume so leftovers = keys
    // that matched no base linear.
    let mut by_path: BTreeMap<String, MergeSite<B>> = adapter
        .sites
        .into_iter()
        .map(|s| (s.path.clone(), s))
        .collect();

    let mut merged = 0usize;
    for (site_path, base) in mmdit.all_base_linears_mut() {
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
                // Keep the base frozen. `Param::from_tensor` unconditionally
                // rebuilds the param with `require_grad = true` (burn 0.21
                // `param/tensor.rs`) — it ignores the tensor's own flag — so an
                // explicit `set_require_grad(false)` at the Param level is
                // required to preserve the upstream `.no_grad()` and not turn
                // the merged base weight into a tracked leaf.
                lin.weight = Param::from_tensor(weight + delta).set_require_grad(false);
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
            "training adapter {} names sites the MMDiT has no base linear for: {:?} — the model's \
             base linears are blocks.<i>.{{attn.<wq|wk|wv|gate|wo>,mlp.<gate|up|down>}}, \
             txtfusion.<layerwise|refiner>_blocks.<i>.*, and tmlp/tproj/txtmlp",
            path.display(),
            unmatched
        );
    }

    sink(TrainEvent::Warning {
        message: format!(
            "training adapter: merged {merged}/{site_count} assistant-LoRA site(s) into the \
             frozen base from {}",
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

/// The global merge scaling from a kohya/ai-toolkit `.safetensors`'s
/// `__metadata__` — `ss_network_alpha / ss_network_dim` (or the bare
/// `network_alpha`/`network_dim`) → `alpha / dim`. `None` when the header
/// carries no metadata, the keys are absent, or a value is not a plain scalar
/// (kohya can store a per-block alpha list; those aren't a single global
/// scaling and fall through to the per-site/unit path). Header-only read.
fn metadata_scaling(path: &Path) -> Result<Option<f64>> {
    use std::io::Read;
    // Bounded read: the 8-byte little-endian header length, then only the
    // header JSON span — never the (potentially hundreds of MB) tensor payload
    // that `from_file`'s snapshot loader reads separately. Best-effort: any read
    // hiccup just means "no metadata" (the snapshot loader raises the clear
    // error if the file is genuinely malformed).
    let Ok(mut file) = std::fs::File::open(path) else {
        return Ok(None);
    };
    let mut len_buf = [0u8; 8];
    if file.read_exact(&mut len_buf).is_err() {
        return Ok(None);
    }
    let header_len = u64::from_le_bytes(len_buf);
    // Sanity cap: safetensors headers are small even for thousands of tensors;
    // a corrupt length must not trigger a huge allocation.
    if header_len == 0 || header_len > 100 * 1024 * 1024 {
        return Ok(None);
    }
    // Parse the header JSON directly rather than `SafeTensors::read_metadata`,
    // which validates every tensor offset against the buffer and so requires
    // the whole file — the header alone carries `__metadata__` (string→string).
    let mut hdr = vec![0u8; header_len as usize];
    if file.read_exact(&mut hdr).is_err() {
        return Ok(None);
    }
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&hdr) else {
        return Ok(None);
    };
    let Some(meta) = json.get("__metadata__") else {
        return Ok(None);
    };
    let get = |a: &str, b: &str| -> Option<f64> {
        meta.get(a)
            .or_else(|| meta.get(b))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
    };
    match (
        get("ss_network_alpha", "network_alpha"),
        get("ss_network_dim", "network_dim"),
    ) {
        (Some(alpha), Some(dim)) if dim > 0.0 => Ok(Some(alpha / dim)),
        _ => Ok(None),
    }
}

/// Map a checkpoint-form site path (e.g. `tmlp.0`, `blocks.0.attn.wq`) to the
/// burn module path [`Mmdit::all_base_linears_mut`] advertises, by applying the
/// MMDiT's own [`key_remap`](Mmdit::key_remap) rules. The rules are anchored on
/// the trailing `.weight`/`.bias`/`.scale` segment, so a bare site path is
/// keyed with `.weight`, remapped, then un-keyed. Trunk/fusion sites pass
/// through unchanged; only the `nn.Sequential`-indexed projections (`tmlp.0` →
/// `tmlp.fc1`, `tproj.1` → `tproj.fc`, `txtmlp.1`/`.3` → `txtmlp.fc1`/`fc2`) are
/// rewritten.
fn remap_site_path<B: Backend>(site: &str) -> String {
    let mut keyed = format!("{site}.weight");
    for (pat, rep) in Mmdit::<B>::key_remap() {
        // key_remap patterns are fixed literals — a compile failure here is a
        // bug in key_remap, not user input.
        let re = Regex::new(pat).expect("Mmdit::key_remap has valid patterns");
        keyed = re.replace(&keyed, rep).into_owned();
    }
    keyed.strip_suffix(".weight").unwrap_or(&keyed).to_string()
}

/// Strip a leading `diffusion_model.` (the ComfyUI/ostris convention) so the
/// remaining key is a bare MMDiT module path.
fn strip_prefix(prefix: &str) -> &str {
    prefix.strip_prefix("diffusion_model.").unwrap_or(prefix)
}
