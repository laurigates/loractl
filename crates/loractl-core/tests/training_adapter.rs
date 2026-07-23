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
use burn::module::Module;
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

/// Round f32 → bf16 (round-to-nearest-even) little-endian bytes — bf16 is the
/// top 16 bits of the f32 with rounding, the dtype real assistant adapters ship.
fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|f| {
            let bits = f.to_bits();
            let round = ((bits >> 16) & 1).wrapping_add(0x7fff);
            let bf = (bits.wrapping_add(round) >> 16) as u16;
            bf.to_le_bytes()
        })
        .collect()
}

/// Write a `.safetensors` from `(key, shape, values)` entries at `dtype`, with
/// optional `__metadata__`.
fn write_st_full(
    path: &std::path::Path,
    entries: &[(String, Vec<usize>, Vec<f32>)],
    dtype: Dtype,
    metadata: Option<std::collections::HashMap<String, String>>,
) {
    let bufs: Vec<(String, Vec<usize>, Vec<u8>)> = entries
        .iter()
        .map(|(k, s, v)| {
            let bytes = match dtype {
                Dtype::BF16 => bf16_bytes(v),
                _ => f32_bytes(v),
            };
            (k.clone(), s.clone(), bytes)
        })
        .collect();
    let views: Vec<(String, TensorView)> = bufs
        .iter()
        .map(|(k, s, b)| (k.clone(), TensorView::new(dtype, s.clone(), b).unwrap()))
        .collect();
    safetensors::serialize_to_file(views, metadata, path).unwrap();
}

/// Write an f32 `.safetensors` with no metadata (the common test case).
fn write_st(path: &std::path::Path, entries: &[(String, Vec<usize>, Vec<f32>)]) {
    write_st_full(path, entries, Dtype::F32, None);
}

/// The original `[d_in, d_out]` weight at any base-linear site.
fn site_weight(model: &mut Mmdit<B>, want: &str) -> Tensor<B, 2> {
    for (path, base) in model.all_base_linears_mut() {
        if path == want {
            return base.as_plain().weight.val();
        }
    }
    panic!("no site {want}");
}

/// Serialize one LoRA site's `lora_A`/`lora_B` (diffusers naming, no alpha) into
/// `entries`, `diffusion_model.`-prefixed, with deterministic ramps.
fn push_site(
    entries: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
    ckpt_site: &str,
    d_in: usize,
    d_out: usize,
    rank: usize,
) {
    entries.push((
        format!("diffusion_model.{ckpt_site}.lora_A.weight"),
        vec![rank, d_in],
        ramp(rank * d_in, 0.10),
    ));
    entries.push((
        format!("diffusion_model.{ckpt_site}.lora_B.weight"),
        vec![d_out, rank],
        ramp(d_out * rank, -0.20),
    ));
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
            // A second site with BARE (un-`diffusion_model.`-prefixed) keys, as
            // some tools emit — the `strip_prefix` fallthrough must accept it.
            (
                "blocks.0.attn.wv.lora_down.weight".into(),
                vec![rank, 96],
                ramp(rank * 96, 0.2),
            ),
            (
                "blocks.0.attn.wv.lora_up.weight".into(),
                vec![32, rank],
                ramp(32 * rank, 0.2),
            ),
        ],
    );

    let adapter = TrainingAdapter::<B>::from_file(&path, &device).unwrap();
    assert_eq!(adapter.len(), 2, "prefixed + bare-keyed sites both parse");

    std::fs::remove_dir_all(&dir).ok();
}

/// The bf16 dtype path (the format real assistant adapters ship): factors
/// written as bf16 must merge, matching the f32-computed delta within bf16
/// tolerance. Every other test feeds f32, so this pins `convert_dtype(F32)`.
#[test]
fn bf16_factors_merge() {
    let device = Default::default();
    let dir = std::env::temp_dir().join(format!("loractl-ta-bf16-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bf16.safetensors");
    let rank = 2usize;
    let (d_in, d_out) = (96usize, 96usize);

    let down = ramp(rank * d_in, 0.10);
    let up = ramp(d_out * rank, -0.20);
    write_st_full(
        &path,
        &[
            (
                "diffusion_model.blocks.0.attn.wq.lora_A.weight".into(),
                vec![rank, d_in],
                down.clone(),
            ),
            (
                "diffusion_model.blocks.0.attn.wq.lora_B.weight".into(),
                vec![d_out, rank],
                up.clone(),
            ),
        ],
        Dtype::BF16,
        None,
    );

    let mut model = Mmdit::<B>::init(MmditConfig::tiny(), &device);
    let w0 = site_weight(&mut model, "blocks.0.attn.wq");
    // Expected from the bf16-rounded factors (round-trip the ramps through bf16
    // so the comparison isn't chasing bf16's own quantization).
    let rt = |v: &[f32]| -> Vec<f32> {
        bf16_bytes(v)
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    };
    let a = mat(&rt(&down), rank, d_in, &device).transpose();
    let b = mat(&rt(&up), d_out, rank, &device).transpose();
    let expected = as_f32(w0 + merge_delta(a, b, 1.0));

    merge_training_adapter(&mut model, &path, &device, &mut |_| {}).unwrap();
    let got = as_f32(site_weight(&mut model, "blocks.0.attn.wq"));
    for (i, (g, w)) in got.iter().zip(&expected).enumerate() {
        assert!((g - w).abs() < 1e-2, "bf16 merged[{i}] = {g} vs {w}");
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// `__metadata__` `ss_network_alpha`/`ss_network_dim` (kohya/ai-toolkit, what
/// the real Krea-2-Turbo assistant adapter carries) sets the merge strength
/// when no per-site `.alpha` tensor is present — and no unit-scaling warning
/// fires.
#[test]
fn metadata_alpha_sets_scaling() {
    let device = Default::default();
    let dir = std::env::temp_dir().join(format!("loractl-ta-meta-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("meta.safetensors");
    let rank = 2usize;
    let (d_in, d_out) = (96usize, 96usize);
    // alpha=16, dim=4 → scaling 4.0 (deliberately != unit).
    let mut meta = std::collections::HashMap::new();
    meta.insert("ss_network_alpha".to_string(), "16".to_string());
    meta.insert("ss_network_dim".to_string(), "4".to_string());

    let down = ramp(rank * d_in, 0.10);
    let up = ramp(d_out * rank, -0.20);
    write_st_full(
        &path,
        &[
            (
                "diffusion_model.blocks.0.attn.wq.lora_A.weight".into(),
                vec![rank, d_in],
                down.clone(),
            ),
            (
                "diffusion_model.blocks.0.attn.wq.lora_B.weight".into(),
                vec![d_out, rank],
                up.clone(),
            ),
        ],
        Dtype::F32,
        Some(meta),
    );

    let mut model = Mmdit::<B>::init(MmditConfig::tiny(), &device);
    let w0 = site_weight(&mut model, "blocks.0.attn.wq");
    let a = mat(&down, rank, d_in, &device).transpose();
    let b = mat(&up, d_out, rank, &device).transpose();
    let expected = as_f32(w0 + merge_delta(a, b, 16.0 / 4.0)); // metadata scaling

    let mut warnings: Vec<String> = Vec::new();
    merge_training_adapter(&mut model, &path, &device, &mut |e| {
        if let loractl_core::TrainEvent::Warning { message } = e {
            warnings.push(message);
        }
    })
    .unwrap();

    let got = as_f32(site_weight(&mut model, "blocks.0.attn.wq"));
    for (i, (g, w)) in got.iter().zip(&expected).enumerate() {
        assert!((g - w).abs() < 1e-4, "meta-scaled merged[{i}] = {g} vs {w}");
    }
    assert!(
        !warnings.iter().any(|m| m.contains("unit scaling")),
        "metadata alpha present → no unit-scaling warning; got {warnings:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The frozen-base invariant, with teeth: after the merge, a merged Plain
/// site's weight `Param` must report `require_grad == false` on the Autodiff
/// backend — `Param::from_tensor` alone would re-track it (burn 0.21).
#[test]
fn merged_base_stays_frozen() {
    use burn::backend::Autodiff;
    type AD = Autodiff<NdArray>;

    let device = Default::default();
    let dir = std::env::temp_dir().join(format!("loractl-ta-frozen-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("frozen.safetensors");
    let rank = 2usize;

    write_st(
        &path,
        &[
            (
                "diffusion_model.blocks.0.attn.wq.lora_A.weight".into(),
                vec![rank, 96],
                ramp(rank * 96, 0.1),
            ),
            (
                "diffusion_model.blocks.0.attn.wq.lora_B.weight".into(),
                vec![96, rank],
                ramp(96 * rank, 0.1),
            ),
        ],
    );

    // Freeze the whole model (as the trainer does), then merge.
    let mut model = Mmdit::<AD>::init(MmditConfig::tiny(), &device).no_grad();
    merge_training_adapter(&mut model, &path, &device, &mut |_| {}).unwrap();

    for (p, base) in model.all_base_linears_mut() {
        if p == "blocks.0.attn.wq" {
            assert!(
                !base.as_plain().weight.val().is_require_grad(),
                "merged base weight must stay frozen (require_grad == false)"
            );
        }
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// Scope: a broad assistant adapter that targets base linears OUTSIDE the
/// LoRA-injectable subset — `attn.gate` (excluded) and `tmlp.0` (an
/// `nn.Sequential`-indexed projection that key_remap rewrites to `tmlp.fc1`) —
/// must still merge, not hard-error. Guards the run-blocking risk of scoping the
/// merge to the injectable subset only.
#[test]
fn merge_covers_non_injectable_base_linears() {
    let device = Default::default();
    let dir = std::env::temp_dir().join(format!("loractl-ta-scope-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("broad.safetensors");
    let rank = 2usize;

    // Read the real site dims from the model (config-determined, not guessed).
    let mut probe = Mmdit::<B>::init(MmditConfig::tiny(), &device);
    let gate = site_weight(&mut probe, "blocks.0.attn.gate").dims();
    let tmlp = site_weight(&mut probe, "tmlp.fc1").dims();

    let mut entries = Vec::new();
    push_site(&mut entries, "blocks.0.attn.gate", gate[0], gate[1], rank);
    push_site(&mut entries, "tmlp.0", tmlp[0], tmlp[1], rank); // remaps to tmlp.fc1
    write_st(&path, &entries);

    let mut model = Mmdit::<B>::init(MmditConfig::tiny(), &device);
    let gate0 = as_f32(site_weight(&mut model, "blocks.0.attn.gate"));
    let tmlp0 = as_f32(site_weight(&mut model, "tmlp.fc1"));

    let n = merge_training_adapter(&mut model, &path, &device, &mut |_| {}).unwrap();
    assert_eq!(n, 2, "both non-injectable sites merged");

    let gate1 = as_f32(site_weight(&mut model, "blocks.0.attn.gate"));
    let tmlp1 = as_f32(site_weight(&mut model, "tmlp.fc1"));
    assert!(
        gate0.iter().zip(&gate1).any(|(a, b)| (a - b).abs() > 1e-6),
        "attn.gate weight changed"
    );
    assert!(
        tmlp0.iter().zip(&tmlp1).any(|(a, b)| (a - b).abs() > 1e-6),
        "tmlp.fc1 (from tmlp.0) weight changed"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A DoRA adapter (magnitude vectors alongside `lora_A`/`lora_B`) must be
/// rejected, not silently merged as plain LoRA (a partial, wrong merge).
#[test]
fn dora_adapter_rejected() {
    let device = Default::default();
    let dir = std::env::temp_dir().join(format!("loractl-ta-dora-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("dora.safetensors");
    let rank = 2usize;

    write_st(
        &path,
        &[
            (
                "diffusion_model.blocks.0.attn.wq.lora_A.weight".into(),
                vec![rank, 96],
                ramp(rank * 96, 0.1),
            ),
            (
                "diffusion_model.blocks.0.attn.wq.lora_B.weight".into(),
                vec![96, rank],
                ramp(96 * rank, 0.1),
            ),
            (
                "diffusion_model.blocks.0.attn.wq.lora_magnitude_vector".into(),
                vec![96],
                ramp(96, 0.1),
            ),
        ],
    );

    let err = match TrainingAdapter::<B>::from_file(&path, &device) {
        Ok(_) => panic!("DoRA adapter must be rejected"),
        Err(e) => e,
    };
    assert!(
        format!("{err:#}").to_lowercase().contains("dora"),
        "error should flag DoRA: {err:#}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A file with no on-disk `.alpha` scalars merges at unit scaling AND surfaces a
/// warning, so a possibly-wrong global strength is never silent.
#[test]
fn absent_alpha_warns() {
    let device = Default::default();
    let dir = std::env::temp_dir().join(format!("loractl-ta-noalpha-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("noalpha.safetensors");
    let rank = 2usize;

    let mut entries = Vec::new();
    push_site(&mut entries, "blocks.0.attn.wq", 96, 96, rank);
    write_st(&path, &entries);

    let mut model = Mmdit::<B>::init(MmditConfig::tiny(), &device);
    let mut warnings: Vec<String> = Vec::new();
    merge_training_adapter(&mut model, &path, &device, &mut |e| {
        if let loractl_core::TrainEvent::Warning { message } = e {
            warnings.push(message);
        }
    })
    .unwrap();

    assert!(
        warnings.iter().any(|m| m.contains("unit scaling")),
        "a unit-scaling warning must be emitted; got {warnings:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
