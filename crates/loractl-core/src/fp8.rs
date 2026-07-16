//! Scaled-fp8 (e4m3fn) safetensors loading — the custom snapshot source for
//! ComfyUI-style fp8 repacks of the Krea 2 denoiser (milestone 15, #82).
//!
//! burn-store 0.21's safetensors dtype map has no `F8_E4M3` arm: an fp8
//! tensor errors inside its snapshot-cache population (burn-store-0.21.0
//! `src/safetensors/store.rs:1000` → `safetensor_dtype_to_burn`) *before*
//! any `ModuleAdapter` runs, and burn's `DType` has no fp8 variant to map
//! to. So this module bypasses only the snapshot-construction step: it
//! mmaps the file with the `safetensors` crate (0.7 parses
//! `Dtype::F8_E4M3`), pairs each fp8 weight with its `<name>_scale`
//! sidecar, and yields lazily-dequantizing **f32** [`TensorSnapshot`]s.
//! Everything downstream — `KeyRemapper::remap`, the `PyTorchToBurnAdapter`
//! linear transpose, [`crate::CastFloatsAdapter`], `module.apply` — is the
//! exact machinery `SafetensorsStore::apply_to` already uses
//! (store.rs:686-716), reused verbatim by the trainer's fp8 load path.
//!
//! Memory: one `Arc<Mmap>` shared by every closure; each closure
//! re-deserializes the (cheap, mmap-backed) header and materializes exactly
//! one f32 tensor when the applier asks — mirroring burn-store's own lazy
//! file path (store.rs:984-1041). Peak stays ≈ the resident module + one
//! tensor (plus its lazy transpose copy inside `PyTorchToBurnAdapter`).

use anyhow::{Context, Result, bail};
use burn::module::ParamId;
use burn::tensor::{BoolStore, DType, TensorData};
use burn_store::{TensorSnapshot, TensorSnapshotError};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

/// The exact 256-entry fp8-e4m3fn → f32 decode table, indexed by the raw
/// byte. e4m3fn: 1 sign / 4 exponent (bias 7) / 3 mantissa bits; **no
/// infinities**; NaN only at 0x7f/0xff; max finite 448.0 (byte 0x7e).
/// Pinned bit-for-bit against torch's `float8_e4m3fn` by
/// `tests/fixtures/fp8_lut_golden.json`.
pub fn e4m3fn_lut() -> [f32; 256] {
    core::array::from_fn(|byte| decode_e4m3fn(byte as u8))
}

/// Decode one e4m3fn byte. Subnormals (exp==0): sign·(man/8)·2⁻⁶;
/// normals: sign·(1+man/8)·2^(exp−7); exp==15 && man==7 → NaN.
fn decode_e4m3fn(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = (byte >> 3) & 0x0f;
    let man = (byte & 0x07) as f32;
    if exp == 0x0f && (byte & 0x07) == 0x07 {
        f32::NAN
    } else if exp == 0 {
        sign * (man / 8.0) * (-6f32).exp2()
    } else {
        sign * (1.0 + man / 8.0) * (f32::from(exp) - 7.0).exp2()
    }
}

/// How a weight's `weight_scale` sidecar broadcasts.
#[derive(Clone, Copy)]
enum ScaleKind {
    /// Shape `[]` or `[1]` — one f32 multiplies every element (the verified
    /// local repack stores the 0-d `[]` form exclusively).
    Scalar,
    /// Shape `[out_features]` (== weight dim 0, the torch `[out, in]`
    /// layout's axis 0) — row `i` multiplies by `scale[i]`. Community
    /// per-output-channel requants. Any other shape is a hard error.
    PerChannel,
}

/// Header-only probe: does any tensor in `path`'s safetensors header carry
/// `Dtype::F8_E4M3`? Drives the auto-detect dispatch in the trainer;
/// bf16/f32 checkpoints return `false` and keep the existing burn-store
/// path.
pub fn is_fp8_checkpoint(path: &Path) -> Result<bool> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // SAFETY: same contract as burn-store's own mmap-backed loader — the
    // mapping is undefined only if the file is truncated/mutated while
    // mapped, which checkpoint files are not during a load.
    let mmap =
        unsafe { Mmap::map(&file) }.with_context(|| format!("mmapping {}", path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("parsing safetensors header of {}", path.display()))?;
    Ok(st.iter().any(|(_, v)| v.dtype() == Dtype::F8_E4M3))
}

/// Build lazy snapshots for every model tensor in a scaled-fp8 checkpoint.
///
/// Classification (hard errors in the order they are checked):
/// 1. A `scaled_fp8` marker key or any `*.scale_weight` key → error naming
///    the legacy ComfyUI "scaled fp8" convention (not supported).
/// 2. Every `F8_E4M3` tensor requires an F32 `<name>_scale` sidecar
///    (`blocks.0.attn.wq.weight` ↔ `blocks.0.attn.wq.weight_scale`);
///    missing sidecar, non-F32 sidecar, or a sidecar shape that is neither
///    scalar (`[]`/`[1]`) nor per-output-channel (`[weight.shape[0]]`) →
///    error. Valid sidecars are consumed (not emitted as snapshots) and the
///    weight emits a lazy dequant snapshot: `f32 = LUT[byte] * scale`.
/// 3. A `*.weight_scale` key with no fp8 base weight → error (orphan).
/// 4. `*.comfy_quant` / `*.input_scale` → dropped silently (inference-only
///    metadata; the local repack keeps its quant map in `__metadata__`).
/// 5. Any other dtype passes through as a lazy byte-copy snapshot with the
///    same 12-dtype map burn-store supports; an unmapped dtype (`F8_E5M2`,
///    …) → error naming tensor + dtype.
///
/// The returned snapshots use file-side dotted paths split like burn-store
/// does, empty container stacks (filled by the Applier during module
/// traversal), and [`DType::F32`] for every dequantized tensor — the
/// trainer's [`crate::CastFloatsAdapter`] then casts to the backend float
/// dtype exactly as for a stock checkpoint.
pub fn load_fp8_snapshots(path: &Path) -> Result<Vec<TensorSnapshot>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // SAFETY: see [`is_fp8_checkpoint`].
    let mmap: Arc<Mmap> = Arc::new(
        unsafe { Mmap::map(&file) }.with_context(|| format!("mmapping {}", path.display()))?,
    );
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("parsing safetensors header of {}", path.display()))?;
    let lut: Arc<[f32; 256]> = Arc::new(e4m3fn_lut());

    // ---- pass 1: classify every key from the header metadata alone. ----
    let header: HashMap<String, (Dtype, Vec<usize>)> = st
        .iter()
        .map(|(name, view)| (name.to_string(), (view.dtype(), view.shape().to_vec())))
        .collect();

    for name in header.keys() {
        if name == "scaled_fp8" || name.ends_with(".scale_weight") {
            bail!(
                "scaled-fp8 checkpoint {} uses the legacy ComfyUI 'scaled fp8' quant \
                 format ('scaled_fp8' marker / '.scale_weight' keys, found '{name}'): \
                 not supported — re-export with per-tensor '<name>.weight_scale' sidecars",
                path.display()
            );
        }
    }

    // fp8 weight name → its validated broadcast kind; `skip` collects the
    // consumed sidecars plus the dropped metadata keys so pass 2 emits
    // snapshots for model tensors only.
    let mut scale_kinds: HashMap<String, ScaleKind> = HashMap::new();
    let mut skip: HashSet<String> = HashSet::new();
    for (name, (dtype, wshape)) in &header {
        if *dtype != Dtype::F8_E4M3 {
            continue;
        }
        let scale_name = format!("{name}_scale");
        let Some((sdtype, sshape)) = header.get(&scale_name) else {
            bail!(
                "fp8 tensor '{name}' has no '{scale_name}' sidecar in {}",
                path.display()
            );
        };
        if *sdtype != Dtype::F32 {
            bail!(
                "'{scale_name}' dtype {sdtype:?} — expected F32 (in {})",
                path.display()
            );
        }
        let kind = if sshape.is_empty() || sshape[..] == [1] {
            ScaleKind::Scalar
        } else if sshape.len() == 1 && Some(&sshape[0]) == wshape.first() {
            // A `[0]` scale (degenerate zero-row weight) would divide by zero
            // in the dequant closure — reject here, where the key context is.
            if sshape[0] == 0 {
                bail!(
                    "'{scale_name}' is empty ([0]) for weight '{name}' {wshape:?} in {}",
                    path.display()
                );
            }
            ScaleKind::PerChannel
        } else {
            let out = wshape.first().copied().unwrap_or(0);
            bail!(
                "'{scale_name}' shape {sshape:?} is neither scalar ([]/[1]) nor \
                 per-output-channel ([{out}]) for weight '{name}' {wshape:?} in {}",
                path.display()
            );
        };
        scale_kinds.insert(name.clone(), kind);
        skip.insert(scale_name);
    }

    for (name, (dtype, _)) in &header {
        if skip.contains(name) || *dtype == Dtype::F8_E4M3 {
            continue;
        }
        if name.ends_with(".weight_scale") {
            let base = name.strip_suffix("_scale").expect("suffix just matched");
            bail!(
                "'{name}' has no fp8 base weight '{base}' in {}",
                path.display()
            );
        }
        if name.ends_with(".comfy_quant") || name.ends_with(".input_scale") {
            // Inference-only quant metadata some ComfyUI repacks carry
            // per-tensor (the local repack keeps its quant map in
            // `__metadata__` instead) — dropped, never a model param.
            skip.insert(name.clone());
            continue;
        }
        if map_dtype(*dtype).is_none() {
            bail!(
                "'{name}' has unsupported dtype {dtype:?} in {}",
                path.display()
            );
        }
    }

    // ---- pass 2: snapshot construction over the surviving model tensors. ----
    let mut snapshots = Vec::new();
    for (name, view) in st.iter() {
        if skip.contains(name) {
            continue;
        }
        if view.dtype() == Dtype::F8_E4M3 {
            snapshots.push(dequant_snapshot(
                Arc::clone(&mmap),
                Arc::clone(&lut),
                name.to_string(),
                format!("{name}_scale"),
                scale_kinds[name],
                view.shape().to_vec(),
            ));
        } else {
            let dtype = map_dtype(view.dtype()).expect("unmapped dtypes errored in pass 1");
            snapshots.push(plain_snapshot(
                Arc::clone(&mmap),
                name.to_string(),
                dtype,
                view.shape().to_vec(),
            ));
        }
    }
    Ok(snapshots)
}

/// Build lazy passthrough snapshots for every tensor in a **plain** (non-fp8)
/// safetensors checkpoint — the fp8-free twin of [`load_fp8_snapshots`], the
/// streaming source PR-B3's quantized loader ([`crate::diffusion_trainer`])
/// uses on the bf16/f32 path.
///
/// Each tensor becomes a lazy [`plain_snapshot`] over one shared `Arc<Mmap>`,
/// materializing its bytes only when the applier — or the quant pass — forces
/// it, so the loader never holds the whole f32 model at once. Unlike
/// `SafetensorsStore::from_file(..).load_from(..)` (which drives the whole
/// apply itself), this hands back the snapshot vector so the caller can
/// partition it (base-linear weight keys → the per-tensor quant pass,
/// everything else → the store applier). An unmapped dtype is a hard error
/// naming the tensor, exactly as in [`load_fp8_snapshots`].
pub(crate) fn load_plain_snapshots(path: &Path) -> Result<Vec<TensorSnapshot>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // SAFETY: see [`is_fp8_checkpoint`].
    let mmap: Arc<Mmap> = Arc::new(
        unsafe { Mmap::map(&file) }.with_context(|| format!("mmapping {}", path.display()))?,
    );
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("parsing safetensors header of {}", path.display()))?;

    let mut snapshots = Vec::new();
    for (name, view) in st.iter() {
        let Some(dtype) = map_dtype(view.dtype()) else {
            bail!(
                "'{name}' has unsupported dtype {:?} in {}",
                view.dtype(),
                path.display()
            );
        };
        snapshots.push(plain_snapshot(
            Arc::clone(&mmap),
            name.to_string(),
            dtype,
            view.shape().to_vec(),
        ));
    }
    Ok(snapshots)
}

/// One lazily-dequantizing snapshot: materializes `LUT[byte] * scale` as f32
/// only when the applier calls `to_data()`.
fn dequant_snapshot(
    mmap: Arc<Mmap>,
    lut: Arc<[f32; 256]>,
    name: String,
    scale_name: String,
    kind: ScaleKind,
    shape: Vec<usize>,
) -> TensorSnapshot {
    let path_parts: Vec<String> = name.split('.').map(str::to_string).collect();
    let data_shape = shape.clone();
    let data_fn = Rc::new(move || -> Result<TensorData, TensorSnapshotError> {
        let st = SafeTensors::deserialize(&mmap).map_err(|e| {
            TensorSnapshotError::IoError(format!("re-parsing safetensors header: {e}"))
        })?;
        let w = st.tensor(&name).map_err(|e| {
            TensorSnapshotError::DataError(format!("fp8 tensor '{name}' not found: {e}"))
        })?;
        let s = st.tensor(&scale_name).map_err(|e| {
            TensorSnapshotError::DataError(format!("scale tensor '{scale_name}' not found: {e}"))
        })?;
        let scale: Vec<f32> = s
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let bytes = w.data();
        let out: Vec<f32> = match kind {
            ScaleKind::Scalar => bytes.iter().map(|&b| lut[b as usize] * scale[0]).collect(),
            ScaleKind::PerChannel => {
                // Row-major [out, in, …]: `scale.len()` == out_features rows
                // of `row` contiguous elements each.
                let row = bytes.len() / scale.len();
                bytes
                    .iter()
                    .enumerate()
                    .map(|(i, &b)| lut[b as usize] * scale[i / row])
                    .collect()
            }
        };
        Ok(TensorData::new(out, data_shape.clone()))
    });
    TensorSnapshot::from_closure(
        data_fn,
        DType::F32,
        shape.into(),
        path_parts,
        // Empty container stack — the Applier fills it during traversal.
        vec![],
        ParamId::new(),
    )
}

/// The non-fp8 passthrough: a lazy byte-copy snapshot, byte-for-byte the
/// shape of burn-store's own lazy-file closure (store.rs:1008-1036).
fn plain_snapshot(
    mmap: Arc<Mmap>,
    name: String,
    dtype: DType,
    shape: Vec<usize>,
) -> TensorSnapshot {
    let path_parts: Vec<String> = name.split('.').map(str::to_string).collect();
    let data_shape = shape.clone();
    let data_fn = Rc::new(move || -> Result<TensorData, TensorSnapshotError> {
        let st = SafeTensors::deserialize(&mmap).map_err(|e| {
            TensorSnapshotError::IoError(format!("re-parsing safetensors header: {e}"))
        })?;
        let t = st.tensor(&name).map_err(|e| {
            TensorSnapshotError::DataError(format!("tensor '{name}' not found: {e}"))
        })?;
        Ok(TensorData::from_bytes_vec(
            t.data().to_vec(),
            data_shape.clone(),
            dtype,
        ))
    });
    TensorSnapshot::from_closure(
        data_fn,
        dtype,
        shape.into(),
        path_parts,
        vec![],
        ParamId::new(),
    )
}

/// The same 12-dtype safetensors→burn map burn-store's loader supports
/// (store.rs:1044-1064). `None` (F8_E5M2, F4, …) is a hard error at
/// classification time so a bad repack fails naming the tensor, not deep
/// inside the applier.
fn map_dtype(dtype: Dtype) -> Option<DType> {
    match dtype {
        Dtype::F64 => Some(DType::F64),
        Dtype::F32 => Some(DType::F32),
        Dtype::F16 => Some(DType::F16),
        Dtype::BF16 => Some(DType::BF16),
        Dtype::I64 => Some(DType::I64),
        Dtype::I32 => Some(DType::I32),
        Dtype::I16 => Some(DType::I16),
        Dtype::I8 => Some(DType::I8),
        Dtype::U64 => Some(DType::U64),
        Dtype::U32 => Some(DType::U32),
        Dtype::U8 => Some(DType::U8),
        Dtype::BOOL => Some(DType::Bool(BoolStore::Native)),
        _ => None,
    }
}
