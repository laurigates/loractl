//! ComfyUI-fp8 Qwen3-VL text-encoder load path (the ComfyUI-scattered-file
//! arc, Phase 3). A ComfyUI `models/text_encoders/qwen/qwen3vl_*_fp8_scaled`
//! file differs from a stock HF `text_encoder/model.safetensors` in three
//! ways that each block the existing burn-store load: HF-Qwen `model.*` keys
//! (vs loractl's `language_model.*`), `F8_E4M3` text weights burn-store 0.21
//! cannot decode, and a bundled `visual.*` vision tower the text-only encoder
//! never consumes. [`load_fp8_encoder`] dodges all three; these tests pin
//! that behavior offline.
//!
//! The fixture is synthesized in-test (the `tests/fp8.rs` pattern) — the fp8
//! bytes come from the torch-pinned [`e4m3fn_lut`], so no torch, no committed
//! binary, and no network are needed. The point under test is the
//! remap/filter/route composition, not the dequant numerics (already pinned
//! in `fp8.rs`), so exact values are irrelevant; the forward-finite backstop
//! is what proves the loaded module is usable.

use burn::backend::NdArray;
use burn_store::{KeyRemapper, TensorSnapshot};
use loractl_core::diffusion_trainer::load_fp8_encoder;
use loractl_core::fp8::{e4m3fn_lut, is_fp8_checkpoint, load_fp8_snapshots};
use loractl_core::qwen3vl::{Qwen3VlConfig, Qwen3VlEncoder};
use regex::Regex;
use safetensors::tensor::{Dtype, View};
use std::borrow::Cow;
use std::path::PathBuf;

type B = NdArray;

// --- minimal in-test safetensors synthesis (mirrors tests/fp8.rs) ----------

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

fn f32_tensor(shape: Vec<usize>, vals: &[f32]) -> RawTensor {
    RawTensor {
        dtype: Dtype::F32,
        shape,
        bytes: vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// An `F8_E4M3` weight whose `out*in` elements cycle through a handful of
/// exactly-representable e4m3fn values (so `LUT[byte]` is lossless).
fn fp8_weight(lut: &[f32; 256], out: usize, in_: usize) -> (RawTensor, f32) {
    const CYCLE: [f32; 8] = [0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 1.0, -1.0];
    let byte_for = |v: f32| -> u8 {
        (0u16..=255)
            .find(|&b| lut[b as usize] == v)
            .unwrap_or_else(|| panic!("{v} is not exactly representable in e4m3fn")) as u8
    };
    let bytes: Vec<u8> = (0..out * in_).map(|i| byte_for(CYCLE[i % 8])).collect();
    (
        RawTensor {
            dtype: Dtype::F8_E4M3,
            shape: vec![out, in_],
            bytes,
        },
        0.5, // scalar weight_scale
    )
}

/// A distinctive 1-D f32 vector used for the "was it actually applied?" probe.
fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.1 + i as f32 * 0.01).collect()
}

/// Build a tiny ComfyUI-style fp8 Qwen3-VL encoder file for `config`
/// (single decoder layer). Returns the path plus the exact f32 values written
/// to `model.layers.0.input_layernorm.weight` so the load can be verified.
fn write_fixture(dir: &TempDir, config: &Qwen3VlConfig) -> (PathBuf, Vec<f32>) {
    let lut = e4m3fn_lut();
    let (h, hd) = (config.hidden_size, config.head_dim);
    let (heads, kv) = (config.num_heads, config.num_kv_heads);
    let inter = config.intermediate_size;

    let mut tensors: Vec<(String, RawTensor)> = Vec::new();

    // Big projections as F8_E4M3 (PyTorch [out, in]) + scalar weight_scale;
    // one carries a `.comfy_quant` marker that must be dropped.
    let fp8_linears: [(&str, usize, usize); 7] = [
        ("self_attn.q_proj", heads * hd, h),
        ("self_attn.k_proj", kv * hd, h),
        ("self_attn.v_proj", kv * hd, h),
        ("self_attn.o_proj", h, heads * hd),
        ("mlp.gate_proj", inter, h),
        ("mlp.up_proj", inter, h),
        ("mlp.down_proj", h, inter),
    ];
    for (i, (name, out, in_)) in fp8_linears.iter().enumerate() {
        let (w, scale) = fp8_weight(&lut, *out, *in_);
        let key = format!("model.layers.0.{name}.weight");
        tensors.push((format!("{key}_scale"), f32_tensor(vec![], &[scale])));
        tensors.push((key.clone(), w));
        if i == 0 {
            // Inference-only quant metadata some repacks carry per tensor —
            // must be dropped, never applied (nor land in `unused`).
            tensors.push((
                format!("model.layers.0.{name}.comfy_quant"),
                RawTensor {
                    dtype: Dtype::U8,
                    shape: vec![64],
                    bytes: vec![0u8; 64],
                },
            ));
        }
    }

    // Norms stay full-precision F32 (real packs don't quantize them). The
    // input_layernorm value is the applied-load probe.
    let in_ln = ramp(h);
    tensors.push((
        "model.layers.0.input_layernorm.weight".into(),
        f32_tensor(vec![h], &in_ln),
    ));
    tensors.push((
        "model.layers.0.post_attention_layernorm.weight".into(),
        f32_tensor(vec![h], &vec![1.0f32; h]),
    ));
    tensors.push((
        "model.layers.0.self_attn.q_norm.weight".into(),
        f32_tensor(vec![hd], &vec![1.0f32; hd]),
    ));
    tensors.push((
        "model.layers.0.self_attn.k_norm.weight".into(),
        f32_tensor(vec![hd], &vec![1.0f32; hd]),
    ));

    // Embedding [vocab, hidden] (loads verbatim — no transpose).
    let vocab = config.vocab_size;
    let embed: Vec<f32> = (0..vocab * h)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    tensors.push((
        "model.embed_tokens.weight".into(),
        f32_tensor(vec![vocab, h], &embed),
    ));

    // Dead final `norm` (the module has no `norm` param): remaps to
    // `language_model.norm.weight`, survives the filter, lands in `unused` —
    // and MUST be tolerated.
    tensors.push((
        "model.norm.weight".into(),
        f32_tensor(vec![h], &vec![1.0f32; h]),
    ));

    // The vision tower: one representative `visual.*` key that the load filter
    // must drop BEFORE the applier (never reaching `unused`).
    tensors.push((
        "visual.blocks.0.attn.qkv.weight".into(),
        f32_tensor(vec![4, 4], &[0.25f32; 16]),
    ));

    let path = dir.0.join("qwen3vl_tiny_fp8_scaled.safetensors");
    let views: Vec<(&str, &RawTensor)> = tensors.iter().map(|(k, t)| (k.as_str(), t)).collect();
    safetensors::serialize_to_file(views, None, &path).expect("write synthesized encoder");
    (path, in_ln)
}

/// The 1-layer tiny config: `tiny()` dims, `select_layers = [1]` so the whole
/// trunk is `layers.0` (`num_layers() == 1`) — a complete, minimal fixture.
fn tiny_one_layer() -> Qwen3VlConfig {
    Qwen3VlConfig {
        select_layers: vec![1],
        ..Qwen3VlConfig::tiny()
    }
}

// --- tests -----------------------------------------------------------------

/// End to end: a synthesized ComfyUI fp8 encoder file loads through
/// [`load_fp8_encoder`] — the fp8 route is taken (header auto-detected), the
/// `model.*` keys are remapped, the vision tower + dead norm do NOT trigger an
/// `unused` bail, the applied params carry the file's values, and a forward on
/// the loaded encoder is finite.
#[test]
fn comfyui_fp8_encoder_loads_filters_and_forwards() {
    use burn::tensor::{Int, Tensor, TensorData};

    let dir = TempDir::new("fp8-encoder");
    let config = tiny_one_layer();
    let (path, in_ln) = write_fixture(&dir, &config);

    // The trainer's dispatch discriminator: this IS an fp8 checkpoint, so the
    // encoder closure routes here rather than to burn-store's `load_module`.
    assert!(
        is_fp8_checkpoint(&path).expect("header parses"),
        "the synthesized encoder must be detected as fp8"
    );

    let device = Default::default();
    let encoder = load_fp8_encoder(
        Qwen3VlEncoder::<B>::init(config.clone(), &device),
        &path,
        "text encoder",
    )
    .expect("the ComfyUI fp8 encoder loads (visual.* filtered, dead norm tolerated)");

    // The applied params carry the file's values: input_layernorm is F32,
    // 1-D, untouched by the transpose adapter — so it loads bit-for-bit.
    let loaded: Vec<f32> = encoder.language_model.layers[0]
        .input_layernorm
        .weight
        .val()
        .into_data()
        .convert::<f32>()
        .into_vec()
        .unwrap();
    assert_eq!(
        loaded, in_ln,
        "input_layernorm.weight must load the remapped model.* value verbatim"
    );

    // A forward on the loaded encoder is finite and non-degenerate — the proof
    // the fp8-dequanted, transposed linears actually landed in the module.
    let s = 4usize;
    let ids = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![1i64, 2, 3, 4], [1, s]), &device);
    let mask = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![1i64; s], [1, s]), &device);
    let cond = encoder.forward_conditioning(ids, mask, 0);
    assert_eq!(cond.dims(), [1, s, 1, config.hidden_size]);
    let values: Vec<f32> = cond.into_data().convert::<f32>().into_vec().unwrap();
    assert!(
        values.iter().all(|v| v.is_finite()),
        "the conditioning stack must be finite"
    );
    assert!(
        values.iter().any(|v| *v != 0.0),
        "the conditioning stack must be non-degenerate"
    );
}

/// The remap + filter + fp8-drop composition, at the snapshot level: after
/// `load_fp8_snapshots` (which drops `.comfy_quant` and consumes
/// `weight_scale`) and the `model.` → `language_model.` remap, the
/// `^language_model\.` filter keeps EXACTLY the text-trunk params and drops
/// the `visual.*` tower.
#[test]
fn comfyui_fp8_encoder_remap_and_filter_select_language_model() {
    let dir = TempDir::new("fp8-encoder-snap");
    let config = tiny_one_layer();
    let (path, _) = write_fixture(&dir, &config);

    // `load_fp8_snapshots` already drops the `.comfy_quant`/`.input_scale`
    // markers and consumes the `weight_scale` sidecars (pinned in fp8.rs).
    let snaps = load_fp8_snapshots(&path).expect("the fp8 encoder file classifies cleanly");
    let raw_keys: Vec<String> = snaps.iter().map(|s| s.full_path()).collect();
    assert!(
        raw_keys.iter().all(|k| !k.ends_with(".comfy_quant")),
        "comfy_quant markers are dropped by load_fp8_snapshots: {raw_keys:?}"
    );
    assert!(
        raw_keys.iter().all(|k| !k.ends_with(".weight_scale")),
        "weight_scale sidecars are consumed, not emitted: {raw_keys:?}"
    );
    assert!(
        raw_keys.iter().any(|k| k.starts_with("visual.")),
        "the vision tower is still present pre-filter (proving the filter is what drops it)"
    );

    // The trainer's remap + pre-filter.
    let remapper =
        KeyRemapper::from_patterns(vec![(r"^model\.", "language_model.")]).expect("valid remap");
    let (remapped, _) = remapper.remap(snaps);
    let filter = Regex::new(Qwen3VlEncoder::<B>::load_filter()).expect("valid filter regex");
    let kept: Vec<String> = remapped
        .into_iter()
        .map(|s: TensorSnapshot| s.full_path())
        .filter(|k| filter.is_match(k))
        .collect();

    // No vision tower survives; everything kept is under `language_model.`.
    assert!(
        kept.iter().all(|k| k.starts_with("language_model.")),
        "the filter keeps only the text trunk: {kept:?}"
    );
    assert!(
        !kept.iter().any(|k| k.starts_with("visual.")),
        "every visual.* key is dropped: {kept:?}"
    );

    // The kept set is exactly the trunk: embed + the dead final norm + the 11
    // per-layer params — all `model.*` keys, remapped to `language_model.*`.
    let mut expected = vec![
        "language_model.embed_tokens.weight".to_string(),
        "language_model.norm.weight".to_string(),
    ];
    for name in [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ] {
        expected.push(format!("language_model.layers.0.{name}"));
    }
    let mut kept_sorted = kept;
    kept_sorted.sort();
    expected.sort();
    assert_eq!(
        kept_sorted, expected,
        "the remap+filter keeps exactly the language_model.* trunk params"
    );
}
