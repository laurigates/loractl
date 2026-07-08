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
use anyhow::{Context, Result};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
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

/// The interop export format for [`export_adapters`].
///
/// `KohyaSs` is the only variant implemented now; `PeftDiffusers` is reserved so
/// the diffusers/PEFT convention can be added behind the same
/// [`AdapterNameMapper`] seam without an API break.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// kohya-ss `.lora_down`/`.lora_up`/`.alpha` naming (see [`KohyaMapper`]).
    KohyaSs,
}

impl ExportFormat {
    /// The name mapper for this format.
    fn mapper(self) -> Box<dyn AdapterNameMapper> {
        match self {
            ExportFormat::KohyaSs => Box::new(KohyaMapper),
        }
    }
}

/// An owned f32 tensor in raw little-endian bytes, the unit
/// [`safetensors::serialize_to_file`] writes.
///
/// The exporter materializes each transposed burn tensor (and each `.alpha`
/// scalar) into one of these so the serializer borrows stable, owned bytes.
struct OwnedF32Tensor {
    shape: Vec<usize>,
    bytes: Vec<u8>,
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
fn to_owned_f32<B: Backend, const D: usize>(t: Tensor<B, D>) -> OwnedF32Tensor {
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
