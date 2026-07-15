//! Interop export of a [`LoraAdapters`] set to a portable `.safetensors` — the
//! second half of milestone 6 (#17).
//!
//! The burn-native snapshot ([`crate::adapter`]) stays the *internal* checkpoint
//! format: it writes burn module paths verbatim, with no transpose and no scalar
//! tensors, which is exactly what a later `load_from` needs but is **not** what
//! the ecosystem's LoRA loaders (ComfyUI, Krea, kohya-ss) expect. This module is
//! the outward-facing bridge: it re-keys, transposes, and appends the `.alpha`
//! scalar so the exported file drops into those tools directly.
//!
//! ## Why a direct `safetensors` writer
//!
//! kohya-ss keys are arbitrary (`lora_<dots→underscores>.lora_down.weight`),
//! the tensors are stored **transposed** relative to burn's `Linear` layout, and
//! each adapter carries an `.alpha` **scalar** (`[1]`) tensor. burn-store's
//! snapshot-save can express none of those three, so this module reaches for the
//! `safetensors` crate's serializer directly and builds the on-disk tensors by
//! hand. `safetensors` already rides in transitively via burn-store, so this
//! adds no new external surface.
//!
//! ## The format seam
//!
//! [`AdapterNameMapper`] is a trait so a second convention (diffusers/PEFT
//! `lora_A`/`lora_B`) can be added later without touching the export machinery —
//! only [`KohyaMapper`] is implemented now, and [`ExportFormat`] has a single
//! `KohyaSs` arm with `PeftDiffusers` reserved. Locking the format contract now,
//! before the diffusion DiT it will ultimately serve exists, is a deliberate
//! early interop lock (see the milestone plan / ADR-0004).

use crate::adapters::LoraAdapters;
use anyhow::{Context, Result, bail};
use burn::module::Param;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use safetensors::tensor::{Dtype, View};
use std::borrow::Cow;
use std::path::Path;

/// Maps a burn module path (e.g. `transformer.h.0.attn.c_attn`) to an export
/// format's down/up/alpha tensor keys.
///
/// A trait so the transposing/writing machinery in [`export_adapters`] is
/// format-agnostic: adding diffusers/PEFT naming later is a new `impl`, not a
/// rewrite. Only [`KohyaMapper`] exists today.
pub trait AdapterNameMapper {
    /// Key for the down-projection tensor (`A`, transposed to `[rank, d_in]`).
    fn down_key(&self, path: &str) -> String;
    /// Key for the up-projection tensor (`B`, transposed to `[d_out, rank]`).
    fn up_key(&self, path: &str) -> String;
    /// Key for the `.alpha` scalar tensor (`[1]`).
    fn alpha_key(&self, path: &str) -> String;
}

/// kohya-ss naming: `lora_<path with dots→underscores>` prefix, then
/// `.lora_down.weight` / `.lora_up.weight` / `.alpha` — the convention
/// ComfyUI/Krea LoRA loaders key on.
pub struct KohyaMapper;

impl KohyaMapper {
    /// The shared `lora_<dots→underscores>` prefix for a module path.
    fn prefix(path: &str) -> String {
        format!("lora_{}", path.replace('.', "_"))
    }
}

impl AdapterNameMapper for KohyaMapper {
    fn down_key(&self, path: &str) -> String {
        format!("{}.lora_down.weight", Self::prefix(path))
    }
    fn up_key(&self, path: &str) -> String {
        format!("{}.lora_up.weight", Self::prefix(path))
    }
    fn alpha_key(&self, path: &str) -> String {
        format!("{}.alpha", Self::prefix(path))
    }
}

/// Krea 2 diffusers-style naming — the convention **ComfyUI's Krea 2 LoRA
/// loader actually accepts** (verified against `comfy/lora.py` +
/// `comfy/utils.py::krea2_to_diffusers`): base names are the diffusers-style
/// module paths (`transformer_blocks.{i}.attn.to_q`, `ff.up`, …), suffixed
/// kohya-style (`.lora_down.weight` / `.lora_up.weight` / `.alpha`), which
/// ComfyUI's weight adapters parse on top of its bare-key map. Native →
/// diffusers renames mirror `krea2_to_diffusers` exactly:
///
/// | native (site path)        | diffusers key                       |
/// |---------------------------|-------------------------------------|
/// | `blocks.{i}`              | `transformer_blocks.{i}`            |
/// | `txtfusion.*_blocks.{i}`  | `text_fusion.*_blocks.{i}`          |
/// | `attn.wq` / `wk` / `wv`   | `attn.to_q` / `to_k` / `to_v`       |
/// | `attn.gate` / `attn.wo`   | `attn.to_gate` / `attn.to_out.0`    |
/// | `mlp.gate` / `up` / `down`| `ff.gate` / `ff.up` / `ff.down`     |
pub struct Krea2DiffusersMapper;

impl Krea2DiffusersMapper {
    /// Translate a native injectable-site path into its diffusers-style name.
    fn diffusers_path(path: &str) -> String {
        let mut out = path.to_string();
        if let Some(rest) = out.strip_prefix("blocks.") {
            out = format!("transformer_blocks.{rest}");
        } else if let Some(rest) = out.strip_prefix("txtfusion.") {
            out = format!("text_fusion.{rest}");
        }
        for (native, diffusers) in [
            ("attn.wq", "attn.to_q"),
            ("attn.wk", "attn.to_k"),
            ("attn.wv", "attn.to_v"),
            ("attn.gate", "attn.to_gate"),
            ("attn.wo", "attn.to_out.0"),
            ("mlp.gate", "ff.gate"),
            ("mlp.up", "ff.up"),
            ("mlp.down", "ff.down"),
        ] {
            if out.ends_with(native) {
                out = format!("{}{}", &out[..out.len() - native.len()], diffusers);
                break;
            }
        }
        out
    }
}

impl AdapterNameMapper for Krea2DiffusersMapper {
    fn down_key(&self, path: &str) -> String {
        format!("{}.lora_down.weight", Self::diffusers_path(path))
    }
    fn up_key(&self, path: &str) -> String {
        format!("{}.lora_up.weight", Self::diffusers_path(path))
    }
    fn alpha_key(&self, path: &str) -> String {
        format!("{}.alpha", Self::diffusers_path(path))
    }
}

/// The interop export format for [`export_adapters`].
///
/// `KohyaSs` is the only variant implemented now; `PeftDiffusers` is reserved so
/// the diffusers/PEFT convention can be added behind the same
/// [`AdapterNameMapper`] seam without an API break.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// kohya-ss `.lora_down`/`.lora_up`/`.alpha` naming (see [`KohyaMapper`]).
    KohyaSs,
    /// Krea 2 diffusers-style naming — what ComfyUI's Krea 2 LoRA loader
    /// accepts (see [`Krea2DiffusersMapper`]).
    Krea2Diffusers,
}

impl ExportFormat {
    /// The name mapper for this format.
    fn mapper(self) -> Box<dyn AdapterNameMapper> {
        match self {
            ExportFormat::KohyaSs => Box::new(KohyaMapper),
            ExportFormat::Krea2Diffusers => Box::new(Krea2DiffusersMapper),
        }
    }
}

/// An owned f32 tensor in raw little-endian bytes, the unit
/// [`safetensors::serialize_to_file`] writes.
///
/// The exporter materializes each transposed burn tensor (and each `.alpha`
/// scalar) into one of these so the serializer borrows stable, owned bytes.
/// `pub(crate)`: the M12 dataset cache ([`crate::dataset`]) writes its latent
/// and conditioning tensors through the same unit.
pub(crate) struct OwnedF32Tensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) bytes: Vec<u8>,
}

impl View for &OwnedF32Tensor {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }
    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

/// Materialize a burn tensor into an [`OwnedF32Tensor`] (row-major f32 bytes of
/// its logical — i.e. post-transpose — layout).
pub(crate) fn to_owned_f32<B: Backend, const D: usize>(t: Tensor<B, D>) -> OwnedF32Tensor {
    let shape = t.dims().to_vec();
    let values: Vec<f32> = t
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("f32 tensor data");
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    OwnedF32Tensor { shape, bytes }
}

/// A single f32 `[1]` scalar tensor (the kohya `.alpha`).
fn scalar_f32(value: f32) -> OwnedF32Tensor {
    OwnedF32Tensor {
        shape: vec![1],
        bytes: value.to_le_bytes().to_vec(),
    }
}

/// Export a [`LoraAdapters`] set to `path` as a portable `.safetensors` in
/// `fmt`'s naming/layout convention.
///
/// For each delta, three tensors are written under the format's keys:
/// - **down** = `lora_a.weight` transposed to `[rank, d_in]`
/// - **up** = `lora_b.weight` transposed to `[d_out, rank]`
/// - **alpha** = the `[1]` scalar `scaling * rank` (recovering the original
///   `alpha`, since `scaling = alpha / rank`).
///
/// burn's `Linear.weight` is `[d_in, d_out]` and the LoRA loaders expect the
/// transposed `[out, in]`-style layout, so each factor is transposed on the way
/// out (mirroring how the GPT-2 loader keeps HF `Conv1D` weights un-transposed
/// on the way *in*). The burn-native snapshot ([`crate::adapter`]) remains the
/// internal checkpoint format — this is strictly the outward-facing copy.
pub fn export_adapters<B: Backend>(
    set: &LoraAdapters<B>,
    fmt: ExportFormat,
    path: &Path,
) -> Result<()> {
    let mapper = fmt.mapper();

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating export dir {}", parent.display()))?;
    }

    // (key, tensor) pairs; owned so the serializer borrows stable bytes.
    let mut tensors: Vec<(String, OwnedF32Tensor)> = Vec::with_capacity(set.deltas.len() * 3);
    for (delta, target) in set.deltas.iter().zip(&set.targets) {
        // `A` is [d_in, rank] in burn → transpose to kohya `[rank, d_in]`.
        let down = delta.lora_a.weight.val().transpose();
        // `B` is [rank, d_out] in burn → transpose to kohya `[d_out, rank]`.
        let up = delta.lora_b.weight.val().transpose();
        // scaling = alpha / rank ⇒ alpha = scaling * rank; rank is A's cols.
        let rank = delta.lora_a.weight.dims()[1];
        let alpha = (delta.scaling * rank as f64) as f32;

        tensors.push((mapper.down_key(target), to_owned_f32(down)));
        tensors.push((mapper.up_key(target), to_owned_f32(up)));
        tensors.push((mapper.alpha_key(target), scalar_f32(alpha)));
    }

    // Borrow the owned tensors as `View`s for the serializer.
    let views: Vec<(&str, &OwnedF32Tensor)> =
        tensors.iter().map(|(k, t)| (k.as_str(), t)).collect();
    safetensors::serialize_to_file(views, None, path)
        .with_context(|| format!("writing adapter export to {}", path.display()))?;

    Ok(())
}

/// Load a previously [`export_adapters`]-written file back into a freshly
/// built [`LoraAdapters`] set — the resume path: A/B round-trip through the
/// export's transposed layout, and each site's `.alpha` is checked against
/// the set's configured scaling (a drifted config must fail loudly, not
/// silently train at the wrong scale). Optimizer state is not part of the
/// export, so a resumed run re-warms its moments from zero.
///
/// Every target in `set` must be present in the file with matching shapes;
/// extra tensors in the file are ignored.
pub fn import_adapters<B: Backend>(
    set: &mut LoraAdapters<B>,
    fmt: ExportFormat,
    path: &Path,
) -> Result<usize> {
    let mapper = fmt.mapper();
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading adapter export {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parsing adapter export {}", path.display()))?;

    let read_matrix = |key: &str| -> Result<Tensor<B, 2>> {
        let view = st
            .tensor(key)
            .with_context(|| format!("adapter export is missing tensor {key}"))?;
        if view.dtype() != Dtype::F32 {
            bail!("tensor {key} is {:?}, expected F32", view.dtype());
        }
        let shape: Vec<usize> = view.shape().to_vec();
        let vals: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // `from_data` converts to the backend's working float dtype.
        Ok(Tensor::from_data(
            TensorData::new(vals, [shape[0], shape[1]]),
            &Default::default(),
        ))
    };

    for (delta, target) in set.deltas.iter_mut().zip(&set.targets) {
        // File layout is the export's: down `[rank, d_in]`, up `[d_out, rank]`
        // — transpose back to burn's `A: [d_in, rank]`, `B: [rank, d_out]`.
        let a = read_matrix(&mapper.down_key(target))?.transpose();
        let b = read_matrix(&mapper.up_key(target))?.transpose();
        if a.dims() != delta.lora_a.weight.dims() || b.dims() != delta.lora_b.weight.dims() {
            bail!(
                "resume shape mismatch at {target}: file A {:?} / B {:?} vs \
                 configured A {:?} / B {:?} — did lora.rank change?",
                a.dims(),
                b.dims(),
                delta.lora_a.weight.dims(),
                delta.lora_b.weight.dims()
            );
        }
        let alpha_key = mapper.alpha_key(target);
        let alpha_view = st
            .tensor(&alpha_key)
            .with_context(|| format!("adapter export is missing tensor {alpha_key}"))?;
        let alpha = f32::from_le_bytes(alpha_view.data()[..4].try_into().unwrap());
        let rank = delta.lora_a.weight.dims()[1];
        let expected = (delta.scaling * rank as f64) as f32;
        if (alpha - expected).abs() > 1e-3 {
            bail!(
                "resume alpha mismatch at {target}: file {alpha} vs configured \
                 {expected} — did lora.alpha change?"
            );
        }
        delta.lora_a.weight = Param::from_tensor(a);
        delta.lora_b.weight = Param::from_tensor(b);
    }
    Ok(set.deltas.len())
}
