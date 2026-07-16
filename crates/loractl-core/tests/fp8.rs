//! Scaled-fp8 loader tests (M15, #82), in two layers:
//!
//! - **Self-contained**: e4m3fn LUT spot values, the dequant broadcast
//!   semantics over synthesized in-test files, and every classification hard
//!   error — each test synthesizes its own minimal `.safetensors` in a
//!   tempdir and computes its expectations inline.
//! - **Golden / fixture-backed** (the trailing section; fixtures regenerate
//!   with `just fp8-reference`): the LUT bit-for-bit vs torch, dequant vs
//!   torch-computed goldens, header detection on the committed tiny-krea2
//!   files, the `load_fp8_module` unused-keys guard, and the fp8-vs-dequant
//!   twin-path forward agreement.

use burn::backend::NdArray;
use burn::tensor::DType;
use burn_store::TensorSnapshot;
use loractl_core::diffusion_trainer::load_fp8_module;
use loractl_core::fp8::{e4m3fn_lut, is_fp8_checkpoint, load_fp8_snapshots};
use loractl_core::mmdit::{Mmdit, MmditConfig};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, View};
use serde::Deserialize;
use std::borrow::Cow;
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("loractl-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create the tempdir");
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// An owned tensor of any safetensors dtype — the writer unit for
/// synthesizing minimal (and deliberately malformed) checkpoint files,
/// mirroring `export.rs`'s `OwnedF32Tensor` View pattern.
struct RawTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl View for &RawTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
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

fn fp8(shape: Vec<usize>, bytes: Vec<u8>) -> RawTensor {
    RawTensor {
        dtype: Dtype::F8_E4M3,
        shape,
        bytes,
    }
}

fn f32_tensor(shape: Vec<usize>, vals: &[f32]) -> RawTensor {
    RawTensor {
        dtype: Dtype::F32,
        shape,
        bytes: vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn write_st(dir: &TempDir, file: &str, tensors: &[(&str, RawTensor)]) -> PathBuf {
    let path = dir.0.join(file);
    let views: Vec<(&str, &RawTensor)> = tensors.iter().map(|(k, t)| (*k, t)).collect();
    safetensors::serialize_to_file(views, None, &path).expect("write synthesized safetensors");
    path
}

/// The fp8 byte encoding a given exactly-representable f32 value — the
/// inverse LUT lookup (positive zero wins the 0.0 tie against byte 0x80).
fn byte_for(lut: &[f32; 256], v: f32) -> u8 {
    (0u16..=255)
        .find(|&b| lut[b as usize] == v)
        .unwrap_or_else(|| panic!("{v} is not exactly representable in e4m3fn")) as u8
}

fn find<'a>(snaps: &'a [TensorSnapshot], name: &str) -> &'a TensorSnapshot {
    snaps
        .iter()
        .find(|s| s.full_path() == name)
        .unwrap_or_else(|| panic!("snapshot '{name}' present"))
}

fn values(snaps: &[TensorSnapshot], name: &str) -> Vec<f32> {
    find(snaps, name)
        .to_data()
        .expect("snapshot materializes")
        .into_vec::<f32>()
        .expect("f32 data")
}

/// 15 exactly-representable e4m3fn values in a non-square [5, 3] — the
/// asymmetric shape catches a rows/cols swap in the broadcast math.
const WEIGHT_5X3: [f32; 15] = [
    0.5,
    -0.5,
    1.0,
    1.5,
    -1.5,
    2.75,
    -3.0,
    448.0,
    0.0,
    0.001953125,
    -448.0,
    0.25,
    -2.0,
    3.5,
    -0.0625,
];

fn weight_bytes_5x3(lut: &[f32; 256]) -> Vec<u8> {
    WEIGHT_5X3.iter().map(|&v| byte_for(lut, v)).collect()
}

#[test]
fn e4m3fn_lut_spot_values() {
    let lut = e4m3fn_lut();
    // Boundary bytes of the format.
    assert_eq!(lut[0x00], 0.0);
    assert_eq!(lut[0x01], 0.001953125); // smallest subnormal, 2^-9
    assert_eq!(lut[0x7e], 448.0); // max finite
    assert!(lut[0x7f].is_nan());
    assert!(lut[0xff].is_nan());
    // Probe-verified byte ⇔ value pairs (torch `float8_e4m3fn` ground truth).
    assert_eq!(lut[0xb0], -0.5);
    assert_eq!(lut[0xbc], -1.5);
    assert_eq!(lut[0x43], 2.75);
    assert_eq!(lut[0xc4], -3.0);
    // The sign bit negates bit-exactly for every non-NaN byte.
    for b in 0x00..0x7f {
        assert_eq!(lut[b | 0x80], -lut[b], "byte {b:#04x}");
    }
}

#[test]
fn dequant_scalar_scale_over_synthesized_weights() {
    let lut = e4m3fn_lut();
    let dir = TempDir::new("fp8-scalar");
    let bytes = weight_bytes_5x3(&lut);
    let scale = 0.03125f32;
    let path = write_st(
        &dir,
        "scalar.safetensors",
        &[
            ("blk.w.weight", fp8(vec![5, 3], bytes.clone())),
            // The verified repack's 0-d `[]` scalar form.
            ("blk.w.weight_scale", f32_tensor(vec![], &[scale])),
        ],
    );
    let snaps = load_fp8_snapshots(&path).expect("clean scalar-scale file loads");
    assert_eq!(snaps.len(), 1, "the consumed sidecar is not emitted");
    let expected: Vec<f32> = bytes.iter().map(|&b| lut[b as usize] * scale).collect();
    assert_eq!(values(&snaps, "blk.w.weight"), expected);
    // Dequantized snapshots surface as f32 with the file-side shape.
    let snap = find(&snaps, "blk.w.weight");
    assert_eq!(snap.dtype, DType::F32);
    assert_eq!(snap.shape.iter().copied().collect::<Vec<_>>(), vec![5, 3]);
}

#[test]
fn dequant_per_channel_scale_broadcasts_along_axis0() {
    let lut = e4m3fn_lut();
    let dir = TempDir::new("fp8-per-channel");
    let bytes = weight_bytes_5x3(&lut);
    let scales = [0.5f32, 1.0, 2.0, -1.0, 0.25];
    let path = write_st(
        &dir,
        "per-channel.safetensors",
        &[
            ("blk.w.weight", fp8(vec![5, 3], bytes.clone())),
            ("blk.w.weight_scale", f32_tensor(vec![5], &scales)),
        ],
    );
    let snaps = load_fp8_snapshots(&path).expect("clean per-channel file loads");
    // Row-major [5, 3]: element i is scaled by its out-channel row, i / 3.
    let expected: Vec<f32> = bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| lut[b as usize] * scales[i / 3])
        .collect();
    assert_eq!(values(&snaps, "blk.w.weight"), expected);
}

#[test]
fn scale_shape_one_is_treated_as_scalar() {
    let lut = e4m3fn_lut();
    let dir = TempDir::new("fp8-shape-one");
    // weight.shape[0] == 2 ≠ 1, so a `[1]` scale must take the scalar
    // branch, not fail the per-channel length check.
    let vals = [1.0f32, -1.5, 0.5, 2.75, -3.0, 448.0];
    let bytes: Vec<u8> = vals.iter().map(|&v| byte_for(&lut, v)).collect();
    let scale = 0.75f32;
    let path = write_st(
        &dir,
        "shape-one.safetensors",
        &[
            ("blk.w.weight", fp8(vec![2, 3], bytes.clone())),
            ("blk.w.weight_scale", f32_tensor(vec![1], &[scale])),
        ],
    );
    let snaps = load_fp8_snapshots(&path).expect("[1]-shaped scale loads as scalar");
    let expected: Vec<f32> = bytes.iter().map(|&b| lut[b as usize] * scale).collect();
    assert_eq!(values(&snaps, "blk.w.weight"), expected);
}

#[test]
fn bad_scale_shape_is_hard_error() {
    let lut = e4m3fn_lut();
    let dir = TempDir::new("fp8-bad-scale-shape");
    let bytes = weight_bytes_5x3(&lut);

    // A [3] scale matches the IN-features dim of a [5, 3] weight — exactly
    // the rows/cols confusion the per-channel rule must reject.
    let path = write_st(
        &dir,
        "in-features.safetensors",
        &[
            ("blk.w.weight", fp8(vec![5, 3], bytes.clone())),
            ("blk.w.weight_scale", f32_tensor(vec![3], &[1.0, 2.0, 3.0])),
        ],
    );
    let err = load_fp8_snapshots(&path).unwrap_err().to_string();
    assert!(
        err.contains("blk.w.weight_scale"),
        "names the scale key: {err}"
    );
    assert!(err.contains("[3]"), "names the got-shape: {err}");
    assert!(
        err.contains("[5]"),
        "names the expected per-channel size: {err}"
    );

    // A 2-D scale is rejected even when its element count matches dim 0.
    let path = write_st(
        &dir,
        "two-d.safetensors",
        &[
            ("blk.w.weight", fp8(vec![5, 3], bytes)),
            (
                "blk.w.weight_scale",
                f32_tensor(vec![5, 1], &[1.0, 2.0, 3.0, 4.0, 5.0]),
            ),
        ],
    );
    let err = load_fp8_snapshots(&path).unwrap_err().to_string();
    assert!(
        err.contains("blk.w.weight_scale"),
        "names the scale key: {err}"
    );
    assert!(err.contains("[5, 1]"), "names the got-shape: {err}");
}

#[test]
fn empty_per_channel_scale_is_hard_error() {
    let dir = TempDir::new("fp8-empty-scale");
    // A [0] scale on a zero-row weight would divide by zero inside the lazy
    // dequant closure — classification must reject it with key context.
    let path = write_st(
        &dir,
        "empty.safetensors",
        &[
            ("blk.w.weight", fp8(vec![0, 3], vec![])),
            ("blk.w.weight_scale", f32_tensor(vec![0], &[])),
        ],
    );
    let err = load_fp8_snapshots(&path).unwrap_err().to_string();
    assert!(
        err.contains("blk.w.weight_scale"),
        "names the scale key: {err}"
    );
    assert!(err.contains("empty"), "names the defect: {err}");
}

#[test]
fn non_f32_scale_dtype_is_hard_error() {
    let lut = e4m3fn_lut();
    let dir = TempDir::new("fp8-scale-dtype");
    // A BF16 sidecar (2 bytes/elem) — dequant math is defined in f32 only.
    let path = write_st(
        &dir,
        "bf16-scale.safetensors",
        &[
            (
                "blk.w.weight",
                fp8(vec![2, 2], weight_bytes_5x3(&lut)[..4].to_vec()),
            ),
            (
                "blk.w.weight_scale",
                RawTensor {
                    dtype: Dtype::BF16,
                    shape: vec![],
                    bytes: vec![0x80, 0x3f], // bf16 LE 1.0
                },
            ),
        ],
    );
    let err = load_fp8_snapshots(&path).unwrap_err().to_string();
    assert!(
        err.contains("blk.w.weight_scale"),
        "names the scale key: {err}"
    );
    assert!(err.contains("expected F32"), "names the expectation: {err}");
}

#[test]
fn fp8_weight_without_scale_is_hard_error() {
    let lut = e4m3fn_lut();
    let dir = TempDir::new("fp8-missing-scale");
    let path = write_st(
        &dir,
        "missing.safetensors",
        &[
            (
                "blk.w.weight",
                fp8(vec![2, 2], weight_bytes_5x3(&lut)[..4].to_vec()),
            ),
            ("blk.norm.scale", f32_tensor(vec![2], &[1.0, 1.0])),
        ],
    );
    let err = load_fp8_snapshots(&path).unwrap_err().to_string();
    assert!(err.contains("blk.w.weight"), "names the fp8 tensor: {err}");
    assert!(
        err.contains("blk.w.weight_scale"),
        "names the missing sidecar: {err}"
    );
}

#[test]
fn orphan_weight_scale_is_hard_error() {
    let dir = TempDir::new("fp8-orphan-scale");
    // Case 1: no base tensor at all.
    let path = write_st(
        &dir,
        "no-base.safetensors",
        &[("blk.w.weight_scale", f32_tensor(vec![], &[1.0]))],
    );
    let err = load_fp8_snapshots(&path).unwrap_err().to_string();
    assert!(
        err.contains("blk.w.weight_scale"),
        "names the orphan: {err}"
    );
    assert!(
        err.contains("no fp8 base weight") && err.contains("blk.w.weight"),
        "names the missing base: {err}"
    );

    // Case 2: the base exists but is not fp8 — still an orphan sidecar.
    let path = write_st(
        &dir,
        "non-fp8-base.safetensors",
        &[
            ("blk.w.weight", f32_tensor(vec![2], &[1.0, 2.0])),
            ("blk.w.weight_scale", f32_tensor(vec![], &[1.0])),
        ],
    );
    let err = load_fp8_snapshots(&path).unwrap_err().to_string();
    assert!(
        err.contains("no fp8 base weight"),
        "non-fp8 base is an orphan: {err}"
    );
}

#[test]
fn legacy_scaled_fp8_marker_is_hard_error() {
    let lut = e4m3fn_lut();
    let dir = TempDir::new("fp8-legacy");
    // The legacy ComfyUI whole-file marker key.
    let path = write_st(
        &dir,
        "marker.safetensors",
        &[
            ("scaled_fp8", fp8(vec![0], vec![])),
            (
                "blk.w.weight",
                fp8(vec![2, 2], weight_bytes_5x3(&lut)[..4].to_vec()),
            ),
        ],
    );
    let err = load_fp8_snapshots(&path).unwrap_err().to_string();
    assert!(
        err.contains("legacy ComfyUI"),
        "names the convention: {err}"
    );

    // Sibling spelling: per-tensor '.scale_weight' keys.
    let path = write_st(
        &dir,
        "scale-weight.safetensors",
        &[("blk.w.scale_weight", f32_tensor(vec![1], &[1.0]))],
    );
    let err = load_fp8_snapshots(&path).unwrap_err().to_string();
    assert!(
        err.contains("legacy ComfyUI"),
        "names the convention: {err}"
    );
}

#[test]
fn comfy_quant_and_input_scale_sidecars_are_dropped() {
    let lut = e4m3fn_lut();
    let dir = TempDir::new("fp8-dropped-sidecars");
    let bytes = weight_bytes_5x3(&lut)[..4].to_vec();
    let path = write_st(
        &dir,
        "sidecars.safetensors",
        &[
            ("blk.w.weight", fp8(vec![2, 2], bytes.clone())),
            ("blk.w.weight_scale", f32_tensor(vec![], &[2.0])),
            (
                "blk.w.comfy_quant",
                RawTensor {
                    dtype: Dtype::U8,
                    shape: vec![2],
                    bytes: vec![1, 0],
                },
            ),
            ("blk.w.input_scale", f32_tensor(vec![], &[0.5])),
            ("blk.bias", f32_tensor(vec![2], &[3.0, -4.0])),
        ],
    );
    let snaps = load_fp8_snapshots(&path).expect("inference-only sidecars load cleanly");
    let mut names: Vec<String> = snaps.iter().map(|s| s.full_path()).collect();
    names.sort();
    assert_eq!(names, ["blk.bias", "blk.w.weight"]);
    // The dropped keys influence nothing: dequant still uses weight_scale.
    let expected: Vec<f32> = bytes.iter().map(|&b| lut[b as usize] * 2.0).collect();
    assert_eq!(values(&snaps, "blk.w.weight"), expected);
}

#[test]
fn non_fp8_tensors_pass_through_with_dtype_preserved() {
    let lut = e4m3fn_lut();
    let dir = TempDir::new("fp8-passthrough");
    let bf16_bytes = vec![0x80u8, 0x3f, 0xc0, 0xbf]; // bf16 LE: [1.0, -1.5]
    let path = write_st(
        &dir,
        "passthrough.safetensors",
        &[
            (
                "blk.w.weight",
                fp8(vec![2, 2], weight_bytes_5x3(&lut)[..4].to_vec()),
            ),
            ("blk.w.weight_scale", f32_tensor(vec![], &[1.0])),
            ("blk.bias", f32_tensor(vec![3], &[1.5, -2.5, 448.0])),
            (
                "blk.norm.scale",
                RawTensor {
                    dtype: Dtype::BF16,
                    shape: vec![2],
                    bytes: bf16_bytes.clone(),
                },
            ),
        ],
    );
    let snaps = load_fp8_snapshots(&path).expect("mixed-dtype file loads");
    // f32 passthrough is value-exact.
    assert_eq!(values(&snaps, "blk.bias"), vec![1.5, -2.5, 448.0]);
    assert_eq!(find(&snaps, "blk.bias").dtype, DType::F32);
    // Non-f32 dtypes survive as themselves (the trainer's cast adapter, not
    // this loader, owns dtype conversion) — byte-identical lazy copies.
    let norm = find(&snaps, "blk.norm.scale");
    assert_eq!(norm.dtype, DType::BF16);
    let data = norm.to_data().expect("bf16 snapshot materializes");
    assert_eq!(data.as_bytes(), &bf16_bytes[..]);
}

#[test]
fn unsupported_dtype_is_hard_error() {
    let dir = TempDir::new("fp8-unsupported-dtype");
    let path = write_st(
        &dir,
        "e5m2.safetensors",
        &[(
            "blk.w.weight",
            RawTensor {
                dtype: Dtype::F8_E5M2,
                shape: vec![2],
                bytes: vec![0x3c, 0xbe],
            },
        )],
    );
    let err = load_fp8_snapshots(&path).unwrap_err().to_string();
    assert!(err.contains("blk.w.weight"), "names the tensor: {err}");
    assert!(err.contains("F8_E5M2"), "names the dtype: {err}");
}

#[test]
fn is_fp8_checkpoint_detects_header() {
    let lut = e4m3fn_lut();
    let dir = TempDir::new("fp8-detect");
    let fp8_path = write_st(
        &dir,
        "quantized.safetensors",
        &[
            (
                "blk.w.weight",
                fp8(vec![2, 2], weight_bytes_5x3(&lut)[..4].to_vec()),
            ),
            ("blk.w.weight_scale", f32_tensor(vec![], &[1.0])),
        ],
    );
    let plain_path = write_st(
        &dir,
        "plain.safetensors",
        &[("blk.w.weight", f32_tensor(vec![2], &[1.0, 2.0]))],
    );
    assert!(is_fp8_checkpoint(&fp8_path).expect("header parses"));
    assert!(!is_fp8_checkpoint(&plain_path).expect("header parses"));
}

// ---------------------------------------------------------------------------
// Golden / fixture-backed tests. Torch is the ground truth: the goldens and
// the tiny-krea2 scaled-fp8 twin fixtures come from `just fp8-reference`.
// ---------------------------------------------------------------------------

const LUT_GOLDEN: &str = include_str!("fixtures/fp8_lut_golden.json");
const DEQUANT_GOLDEN: &str = include_str!("fixtures/fp8_dequant_golden.json");
const TURBO_FP8: &str = "tests/fixtures/tiny-krea2/turbo_fp8.safetensors";
const TURBO_DEQUANT: &str = "tests/fixtures/tiny-krea2/turbo_fp8_dequant.safetensors";
const RAW: &str = "tests/fixtures/tiny-krea2/raw.safetensors";

#[derive(Deserialize)]
struct LutGolden {
    /// `torch.arange(256, dtype=uint8).view(float8_e4m3fn).float()`, with
    /// the two NaN bytes (0x7f/0xff) serialized as null.
    lut: Vec<Option<f32>>,
}

/// A dequant golden case whose scale is the 0-d `[]` scalar form.
#[derive(Deserialize)]
struct ScalarScaleCase {
    weight_bytes: Vec<u8>,
    shape: Vec<usize>,
    scale: f32,
    expected: Vec<f32>,
}

/// A dequant golden case whose scale is a 1-D vector (`[out]` or `[1]`).
#[derive(Deserialize)]
struct VecScaleCase {
    weight_bytes: Vec<u8>,
    shape: Vec<usize>,
    scale: Vec<f32>,
    expected: Vec<f32>,
}

#[derive(Deserialize)]
struct DequantGolden {
    scalar: ScalarScaleCase,
    per_channel: VecScaleCase,
    scale_shape_one: VecScaleCase,
}

/// Write a `{weight, weight_scale}` pair from a golden case and assert the
/// loader reproduces torch's `w.float() * scale` exactly — f32 bit-equality,
/// since both sides compute the same `LUT[byte] · scale` product in f32.
fn assert_dequant_case(
    tag: &str,
    bytes: &[u8],
    shape: &[usize],
    scale: RawTensor,
    expected: &[f32],
) {
    let dir = TempDir::new(tag);
    let path = write_st(
        &dir,
        "golden.safetensors",
        &[
            ("blk.w.weight", fp8(shape.to_vec(), bytes.to_vec())),
            ("blk.w.weight_scale", scale),
        ],
    );
    let snaps = load_fp8_snapshots(&path).expect("golden case loads");
    assert_eq!(values(&snaps, "blk.w.weight"), expected);
}

#[test]
fn e4m3fn_lut_matches_torch_golden() {
    let golden: LutGolden = serde_json::from_str(LUT_GOLDEN).expect("parse the LUT golden");
    assert_eq!(golden.lut.len(), 256, "one golden entry per byte");
    let lut = e4m3fn_lut();
    for (byte, want) in golden.lut.iter().enumerate() {
        match want {
            None => assert!(lut[byte].is_nan(), "byte {byte:#04x} must decode to NaN"),
            Some(v) => assert_eq!(
                lut[byte].to_bits(),
                v.to_bits(),
                "byte {byte:#04x}: {} vs torch {v}",
                lut[byte]
            ),
        }
    }
}

#[test]
fn dequant_scalar_scale_matches_torch_golden() {
    let g: DequantGolden = serde_json::from_str(DEQUANT_GOLDEN).expect("parse the dequant golden");
    let c = g.scalar;
    // The verified local repack's 0-d `[]` scalar form.
    assert_dequant_case(
        "fp8-golden-scalar",
        &c.weight_bytes,
        &c.shape,
        f32_tensor(vec![], &[c.scale]),
        &c.expected,
    );
}

#[test]
fn dequant_per_channel_scale_matches_torch_golden() {
    let g: DequantGolden = serde_json::from_str(DEQUANT_GOLDEN).expect("parse the dequant golden");
    let c = g.per_channel;
    // Torch computed `expected` by broadcasting `scale.view(-1, 1)` over the
    // non-square [5, 3] weight — an independent pin of the axis-0 rule.
    assert_eq!(c.scale.len(), c.shape[0], "golden scale is per-out-channel");
    assert_dequant_case(
        "fp8-golden-per-channel",
        &c.weight_bytes,
        &c.shape,
        f32_tensor(vec![c.scale.len()], &c.scale),
        &c.expected,
    );
}

#[test]
fn dequant_scale_shape_one_matches_torch_golden() {
    let g: DequantGolden = serde_json::from_str(DEQUANT_GOLDEN).expect("parse the dequant golden");
    let c = g.scale_shape_one;
    assert_eq!(c.scale.len(), 1, "golden pins the [1]-is-scalar rule");
    assert_dequant_case(
        "fp8-golden-shape-one",
        &c.weight_bytes,
        &c.shape,
        f32_tensor(vec![1], &c.scale),
        &c.expected,
    );
}

#[test]
fn is_fp8_checkpoint_detects_fixture_headers() {
    // The exact dispatch input the trainer's auto-detect sees: the committed
    // scaled-fp8 repack vs its plain-f32 sibling of the same architecture.
    assert!(is_fp8_checkpoint(Path::new(TURBO_FP8)).expect("turbo_fp8 header parses"));
    assert!(!is_fp8_checkpoint(Path::new(RAW)).expect("raw header parses"));
}

#[test]
fn unexpected_keys_are_hard_error_listing_them() {
    // Rebuild the committed turbo fixture with two extra float tensors — an
    // fp8mixed-style repack (baked-in LoRA) that `load_fp8_module` must
    // reject via `ApplyResult::unused`, naming every leftover key.
    let dir = TempDir::new("fp8-unexpected-keys");
    let committed = std::fs::read(TURBO_FP8).expect("read the committed turbo fixture");
    let st = SafeTensors::deserialize(&committed).expect("parse the committed turbo fixture");
    let mut tensors: Vec<(String, RawTensor)> = st
        .iter()
        .map(|(name, view)| {
            (
                name.to_string(),
                RawTensor {
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    bytes: view.data().to_vec(),
                },
            )
        })
        .collect();
    tensors.push((
        "last.up".to_string(),
        f32_tensor(vec![2, 2], &[0.1, 0.2, 0.3, 0.4]),
    ));
    tensors.push((
        "last.down".to_string(),
        f32_tensor(vec![2, 2], &[0.5, 0.6, 0.7, 0.8]),
    ));
    let views: Vec<(&str, &RawTensor)> = tensors.iter().map(|(k, t)| (k.as_str(), t)).collect();
    let path = dir.0.join("extras.safetensors");
    safetensors::serialize_to_file(views, None, &path).expect("write the extended fixture");

    let device = Default::default();
    let init = Mmdit::<NdArray>::init(MmditConfig::tiny_krea2(), &device);
    let err = load_fp8_module(init, &path, &Mmdit::<NdArray>::key_remap(), "MMDiT")
        .unwrap_err()
        .to_string();
    assert!(err.contains("last.up"), "lists the first leftover: {err}");
    assert!(
        err.contains("last.down"),
        "lists the second leftover: {err}"
    );
}

/// Deterministic bounded input data: `sin` over a strided ramp.
fn ramp(n: usize, step: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * step).sin() * 0.5).collect()
}

#[test]
fn fp8_and_dequant_twin_paths_agree() {
    use burn::tensor::{Tensor, TensorData};
    use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

    let device = Default::default();
    let cfg = MmditConfig::tiny_krea2();

    // Path 1: the trainer's scaled-fp8 loader over the quantized fixture
    // (whose sidecars include one per-channel [256] scale, so this also
    // exercises that branch end-to-end).
    let init = Mmdit::<NdArray>::init(cfg.clone(), &device);
    let fp8_model = load_fp8_module(
        init,
        Path::new(TURBO_FP8),
        &Mmdit::<NdArray>::key_remap(),
        "tiny-krea2 fp8 MMDiT",
    )
    .expect("the scaled-fp8 fixture loads");

    // Path 2: the already-proven burn-store path over the f32 twin torch
    // dequantized from the same fp8 payload at fixture-generation time.
    let remapper = KeyRemapper::from_patterns(Mmdit::<NdArray>::key_remap().to_vec())
        .expect("valid remap patterns");
    // `BaseLinear` enum sites: skip the variant name in key paths (matches
    // `load_fp8_module`'s path, so the twin loads land identical keys).
    let mut store = SafetensorsStore::from_file(TURBO_DEQUANT)
        .remap(remapper)
        .skip_enum_variants(true)
        .with_from_adapter(PyTorchToBurnAdapter);
    let mut dequant_model = Mmdit::<NdArray>::init(cfg.clone(), &device);
    let result = dequant_model
        .load_from(&mut store)
        .expect("dequant twin loads");
    assert!(result.errors.is_empty(), "load errors: {:?}", result.errors);
    assert!(
        result.missing.is_empty(),
        "missing params: {:?}",
        result.missing
    );

    // One deterministic forward each over identical inputs: b = 1, 3 text
    // tokens, a 2×2 patch grid (4 image tokens of channels·patch² features).
    let (txtlen, imglen) = (3usize, 4usize);
    let imgdim = cfg.channels * cfg.patch * cfg.patch;
    let img = Tensor::<NdArray, 1>::from_data(
        TensorData::new(ramp(imglen * imgdim, 0.37), [imglen * imgdim]),
        &device,
    )
    .reshape([1, imglen, imgdim]);
    let ctx_len = txtlen * cfg.txtlayers * cfg.txtdim;
    let context =
        Tensor::<NdArray, 1>::from_data(TensorData::new(ramp(ctx_len, 0.23), [ctx_len]), &device)
            .reshape([1, txtlen, cfg.txtlayers, cfg.txtdim]);
    let t = Tensor::<NdArray, 1>::from_data(TensorData::new(vec![0.5f32], [1]), &device);
    // Text at the origin, image tokens on the 2×2 patch grid.
    let mut pos = vec![0.0f32; txtlen * 3];
    for i in 0..imglen {
        pos.extend([0.0, (i / 2) as f32, (i % 2) as f32]);
    }
    let pos =
        Tensor::<NdArray, 1>::from_data(TensorData::new(pos, [(txtlen + imglen) * 3]), &device)
            .reshape([1, txtlen + imglen, 3]);
    let mask = Tensor::<NdArray, 2>::ones([1, txtlen + imglen], &device);

    let out_fp8 = fp8_model.forward(
        img.clone(),
        context.clone(),
        t.clone(),
        pos.clone(),
        mask.clone(),
    );
    let out_dequant = dequant_model.forward(img, context, t, pos, mask);

    let a: Vec<f32> = out_fp8.into_data().convert::<f32>().into_vec().unwrap();
    let b: Vec<f32> = out_dequant.into_data().convert::<f32>().into_vec().unwrap();
    assert_eq!(a.len(), b.len());
    let max_diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff <= 1e-6,
        "twin paths diverge: max|Δ| = {max_diff:e}"
    );
    // The agreement must not be vacuous.
    assert!(
        a.iter().any(|v| *v != 0.0),
        "forward output must be non-zero"
    );
}
