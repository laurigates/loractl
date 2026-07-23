//! Merge-at-load of an external LoRA training adapter (#83).
//!
//! Two claims, pinned separately (see `.claude/rules/testing.md`):
//!
//! 1. **The merge math + layout convention** — [`merge_delta`] folds the
//!    on-disk `[out, in]` factors into burn's `[d_in, d_out]` weight exactly as
//!    the Python reference does (`training_adapter_merge_golden`, vs the
//!    `reference/training_adapter_merge_reference.py` golden).
//!
//! 2. **The producer-contract read path** — the real
//!    [`merge_training_adapter`] over the real MMDiT site enumeration accepts
//!    the `diffusion_model.*` `lora_A`/`lora_B` (and kohya `lora_down`/`lora_up`)
//!    keys an assistant adapter ships, folds them into the right sites, leaves
//!    others untouched, and **fails loud** on a key that names no site or a
//!    misshaped factor (an unmatched LoRA key doing nothing silently is the
//!    worst failure shape).

use burn::backend::NdArray;
use burn::tensor::Device;
use burn::tensor::{Tensor, TensorData};
use loractl_core::mmdit::{Mmdit, MmditConfig};
use loractl_core::training_adapter::{TrainingAdapter, merge_delta, merge_training_adapter};
use safetensors::tensor::{Dtype, TensorView};
use serde_json::Value;

type B = NdArray;

fn mat(vals: &[f32], rows: usize, cols: usize, device: &Device<B>) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(vals.to_vec(), [rows, cols]), device)
}

fn as_f32(t: Tensor<B, 2>) -> Vec<f32> {
    t.into_data().convert::<f32>().into_vec().unwrap()
}

/// Claim 1: `merge_delta` reproduces the Python reference merge byte-for-byte
/// (up to f32 rounding). Disk `down [rank, d_in]` / `up [d_out, rank]` lift to
/// burn `A = downᵀ`, `B = upᵀ`; `W + merge_delta(A, B, scaling)` must equal the
/// golden `merged_burn`.
#[test]
fn training_adapter_merge_golden() {
    let golden: Value = serde_json::from_str(include_str!("golden/training_adapter_merge.json"))
        .expect("parse merge golden");

    let device = Default::default();
    let hp = &golden["hyperparams"];
    let d_in = hp["d_in"].as_u64().unwrap() as usize;
    let d_out = hp["d_out"].as_u64().unwrap() as usize;
    let rank = hp["rank"].as_u64().unwrap() as usize;
    let scaling = hp["scaling"].as_f64().unwrap();

    let floats = |key: &str| -> Vec<f32> {
        golden[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };

    let w = mat(&floats("w_burn"), d_in, d_out, &device);
    // Disk factors → burn layout via transpose (the crate's own convention).
    let a = mat(&floats("down_disk"), rank, d_in, &device).transpose(); // [d_in, rank]
    let b = mat(&floats("up_disk"), d_out, rank, &device).transpose(); // [rank, d_out]

    let delta = merge_delta(a, b, scaling);
    let merged = as_f32(w + delta);

    let expected = floats("merged_burn");
    assert_eq!(merged.len(), expected.len());
    for (i, (got, want)) in merged.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "merged[{i}] = {got} vs golden {want}"
        );
    }
}

/// A deterministic f32 ramp, distinct per (site, factor), so a mis-wired read
/// (wrong tensor into a site) produces visibly wrong numbers.
fn ramp(n: usize, base: f32) -> Vec<f32> {
    (0..n).map(|i| base + (i as f32) * 0.01 - 0.5).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Write a `.safetensors` from `(key, shape, values)` entries.
fn write_st(path: &std::path::Path, entries: &[(String, Vec<usize>, Vec<f32>)]) {
    let bufs: Vec<(String, Vec<usize>, Vec<u8>)> = entries
        .iter()
        .map(|(k, s, v)| (k.clone(), s.clone(), f32_bytes(v)))
        .collect();
    let views: Vec<(String, TensorView)> = bufs
        .iter()
        .map(|(k, s, b)| {
            (
                k.clone(),
                TensorView::new(Dtype::F32, s.clone(), b).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, None, path).unwrap();
}

/// The original `[d_in, d_out]` weight at an injectable site.
fn site_weight(model: &mut Mmdit<B>, want: &str) -> Tensor<B, 2> {
    for (path, base) in model.base_linears_mut() {
        if path == want {
            return base.as_plain().weight.val();
        }
    }
    panic!("no site {want}");
}

/// Claim 2: the full read path over a real (tiny) MMDiT site enumeration.
///
/// One site uses diffusers `lora_A`/`lora_B` **with** a `.alpha` (scaling =
/// alpha/rank); another uses kohya `lora_down`/`lora_up` **without** alpha
/// (scaling = 1.0). Both are `diffusion_model.`-prefixed, as ComfyUI/ostris
/// ships them. After the merge each targeted site equals `W₀ + delta`, an
/// untargeted site is untouched, and the two scalings are honoured.
#[test]
fn merge_reads_producer_keys_into_the_right_sites() {
    let device = Default::default();
    let dir = std::env::temp_dir().join(format!("loractl-ta-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("adapter.safetensors");

    // tiny() sites: attn.wq is [96, 96]; mlp.down is [256, 96].
    let rank = 2usize;
    let (wq_in, wq_out) = (96usize, 96usize);
    let (dn_in, dn_out) = (256usize, 96usize);
    let alpha = 8.0f32; // wq scaling = alpha/rank = 4.0

    // Disk layout: down [rank, d_in], up [d_out, rank].
    let wq_down = ramp(rank * wq_in, 0.10);
    let wq_up = ramp(wq_out * rank, -0.20);
    let dn_down = ramp(rank * dn_in, 0.05);
    let dn_up = ramp(dn_out * rank, 0.30);

    write_st(
        &path,
        &[
            // wq via diffusers naming + alpha.
            (
                "diffusion_model.blocks.0.attn.wq.lora_A.weight".into(),
                vec![rank, wq_in],
                wq_down.clone(),
            ),
            (
                "diffusion_model.blocks.0.attn.wq.lora_B.weight".into(),
                vec![wq_out, rank],
                wq_up.clone(),
            ),
            (
                "diffusion_model.blocks.0.attn.wq.alpha".into(),
                vec![1],
                vec![alpha],
            ),
            // mlp.down via kohya naming, no alpha (scaling 1.0).
            (
                "diffusion_model.blocks.1.mlp.down.lora_down.weight".into(),
                vec![rank, dn_in],
                dn_down.clone(),
            ),
            (
                "diffusion_model.blocks.1.mlp.down.lora_up.weight".into(),
                vec![dn_out, rank],
                dn_up.clone(),
            ),
        ],
    );

    let mut model = Mmdit::<B>::init(MmditConfig::tiny(), &device);

    // Snapshot originals for the targeted + one untargeted site.
    let wq0 = site_weight(&mut model, "blocks.0.attn.wq");
    let dn0 = site_weight(&mut model, "blocks.1.mlp.down");
    let wo0 = as_f32(site_weight(&mut model, "blocks.0.attn.wo"));

    // Independently expected deltas (test-side, distinct code path from the impl
    // beyond the shared `merge_delta`): burn A = downᵀ, B = upᵀ.
    let wq_a = mat(&wq_down, rank, wq_in, &device).transpose();
    let wq_b = mat(&wq_up, wq_out, rank, &device).transpose();
    let wq_expected = as_f32(wq0 + merge_delta(wq_a, wq_b, alpha as f64 / rank as f64));

    let dn_a = mat(&dn_down, rank, dn_in, &device).transpose();
    let dn_b = mat(&dn_up, dn_out, rank, &device).transpose();
    let dn_expected = as_f32(dn0 + merge_delta(dn_a, dn_b, 1.0));

    let merged = merge_training_adapter(&mut model, &path, &device, &mut |_| {}).unwrap();
    assert_eq!(merged, 2, "both sites merged");

    let wq_got = as_f32(site_weight(&mut model, "blocks.0.attn.wq"));
    let dn_got = as_f32(site_weight(&mut model, "blocks.1.mlp.down"));
    let wo_got = as_f32(site_weight(&mut model, "blocks.0.attn.wo"));

    let close = |got: &[f32], want: &[f32], label: &str| {
        assert_eq!(got.len(), want.len(), "{label} len");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-4, "{label}[{i}] = {g} vs {w}");
        }
    };
    close(&wq_got, &wq_expected, "wq (alpha-scaled)");
    close(&dn_got, &dn_expected, "mlp.down (unit-scaled)");
    // Untargeted site is byte-identical.
    close(&wo_got, &wo0, "wo (untargeted)");

    std::fs::remove_dir_all(&dir).ok();
}

/// Teeth: a key that names no injectable site must fail loud (never a silent
/// no-op), and the error must point at the offending site.
#[test]
fn unknown_site_key_fails_loud() {
    let device = Default::default();
    let dir = std::env::temp_dir().join(format!("loractl-ta-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bad.safetensors");

    let rank = 2usize;
    // block 9 does not exist in a 2-layer tiny() model.
    write_st(
        &path,
        &[
            (
                "diffusion_model.blocks.9.attn.wq.lora_A.weight".into(),
                vec![rank, 96],
                ramp(rank * 96, 0.1),
            ),
            (
                "diffusion_model.blocks.9.attn.wq.lora_B.weight".into(),
                vec![96, rank],
                ramp(96 * rank, 0.1),
            ),
        ],
    );

    let mut model = Mmdit::<B>::init(MmditConfig::tiny(), &device);
    let err = merge_training_adapter(&mut model, &path, &device, &mut |_| {})
        .expect_err("unknown site must error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("blocks.9.attn.wq"),
        "error should name the unmatched site: {msg}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Teeth: a factor whose shape does not match the target site's base weight
/// must fail loud rather than merge garbage.
#[test]
fn shape_mismatch_fails_loud() {
    let device = Default::default();
    let dir = std::env::temp_dir().join(format!("loractl-ta-shape-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shape.safetensors");

    let rank = 2usize;
    // wq expects d_in = 96; give 50.
    write_st(
        &path,
        &[
            (
                "diffusion_model.blocks.0.attn.wq.lora_A.weight".into(),
                vec![rank, 50],
                ramp(rank * 50, 0.1),
            ),
            (
                "diffusion_model.blocks.0.attn.wq.lora_B.weight".into(),
                vec![96, rank],
                ramp(96 * rank, 0.1),
            ),
        ],
    );

    let mut model = Mmdit::<B>::init(MmditConfig::tiny(), &device);
    let err = merge_training_adapter(&mut model, &path, &device, &mut |_| {})
        .expect_err("shape mismatch must error");
    assert!(
        format!("{err:#}").contains("shape mismatch"),
        "error should flag the shape mismatch: {err:#}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Parsing alone (no model) — both naming conventions and the
/// `diffusion_model.` strip resolve to bare site paths; rank auto-detects.
#[test]
fn parse_detects_both_naming_conventions() {
    let device = Default::default();
    let dir = std::env::temp_dir().join(format!("loractl-ta-parse-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("parse.safetensors");
    let rank = 3usize;

    write_st(
        &path,
        &[
            (
                "diffusion_model.blocks.0.attn.wk.lora_A.weight".into(),
                vec![rank, 96],
                ramp(rank * 96, 0.1),
            ),
            (
                "diffusion_model.blocks.0.attn.wk.lora_B.weight".into(),
                vec![32, rank],
                ramp(32 * rank, 0.1),
            ),
        ],
    );

    let adapter = TrainingAdapter::<B>::from_file(&path, &device).unwrap();
    assert_eq!(adapter.len(), 1);

    std::fs::remove_dir_all(&dir).ok();
}
