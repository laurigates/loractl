//! Opt-in load-and-forward proof against the **real** scaled-fp8
//! Krea-2-Turbo checkpoint (M15, #82): the 13.1 GB ComfyUI-style repack
//! (686 keys = 256 `F8_E4M3` weights + 256 F32 0-d `weight_scale` sidecars +
//! 174 BF16 floats) classifies cleanly at full size and its dequantized
//! weights drive a finite forward at the real widths.
//!
//! **Depth-truncated by design**, like `tests/mmdit_real.rs`: the full
//! 28-block model dequantized to f32 is ~53 GB — beyond a 48 GiB host — so
//! this proof retains `blocks.{0,1}` plus every non-trunk tensor and loads
//! them into `MmditConfig::krea2_truncated(2)`. The truncation happens on
//! the Rust side by filtering the lazy snapshots (the dropped 26 blocks are
//! never materialized — no 13 GB re-dump), which is exactly what makes this
//! a streaming proof at real scale.
//!
//! Gated behind the `mmdit-real` feature AND `#[ignore]`; the checkpoint
//! path comes from `LORACTL_TURBO_FP8`:
//!
//! ```sh
//! just test-turbo-real <path/to/krea2_turbo_fp8_scaled.safetensors>
//! ```

#![cfg(feature = "mmdit-real")]

use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::{Distribution, Element, Tensor, TensorData};
use burn_store::{KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter};
use loractl_core::CastFloatsAdapter;
use loractl_core::fp8::load_fp8_snapshots;
use loractl_core::mmdit::{Mmdit, MmditConfig};
use std::path::Path;

type B = burn::backend::NdArray;

/// Trunk depth of the truncated proof (mirrors `mmdit_real.rs`).
const LAYERS: usize = 2;

/// Model tensors expected after truncation: the 28-block repack carries 430
/// model tensors (256 quantized + 174 float) — 13 per trunk block plus 66
/// shared/text-fusion — so 2·13 + 66 = 92 (the same count as the tiny
/// 2-block fixtures, which share the per-block tensor layout).
const EXPECTED_TENSORS: usize = 92;

#[test]
#[ignore = "opt-in: needs the local 13 GB scaled-fp8 Krea-2-Turbo checkpoint (run via `just test-turbo-real <path>`)"]
fn turbo_fp8_real_truncated_load_and_forward() {
    let path = std::env::var("LORACTL_TURBO_FP8").unwrap_or_else(|_| {
        panic!(
            "set LORACTL_TURBO_FP8 to the scaled-fp8 turbo checkpoint \
             (run via `just test-turbo-real <path>`)"
        )
    });
    let path = Path::new(&path);
    assert!(
        path.exists(),
        "LORACTL_TURBO_FP8 points at a missing file: {}",
        path.display()
    );

    // Full-file classification (every sidecar validated + consumed here),
    // then the Rust-side depth truncation to blocks.{0,1} + non-trunk keys.
    let snapshots = load_fp8_snapshots(path).expect("the full scaled-fp8 checkpoint classifies");
    let retained: Vec<_> = snapshots
        .into_iter()
        .filter(|s| {
            let p = s.full_path();
            !p.starts_with("blocks.") || p.starts_with("blocks.0.") || p.starts_with("blocks.1.")
        })
        .collect();
    assert_eq!(
        retained.len(),
        EXPECTED_TENSORS,
        "truncated model-tensor count (sidecars are consumed, never emitted)"
    );

    let device = Default::default();
    let mut model = Mmdit::<B>::init(MmditConfig::krea2_truncated(LAYERS), &device);
    let remapper =
        KeyRemapper::from_patterns(Mmdit::<B>::key_remap().to_vec()).expect("valid remap patterns");
    let (retained, _) = remapper.remap(retained);
    // The same adapter chain `load_fp8_module` drives: the nn.Linear
    // transpose, then a cast to the backend float dtype (the real file's
    // BF16 passthrough tensors must not survive as bf16 params).
    let cast = CastFloatsAdapter {
        target: <<B as BackendTypes>::FloatElem as Element>::dtype(),
    };
    let adapter: Box<dyn ModuleAdapter> = Box::new(PyTorchToBurnAdapter.chain(cast));
    let result = model.apply(retained, None, Some(adapter), false);

    assert!(
        result.errors.is_empty(),
        "apply errors: {:?}",
        result.errors
    );
    assert!(
        result.missing.is_empty(),
        "unexpected missing params: {:?}",
        result.missing
    );
    assert!(
        result.unused.is_empty(),
        "unexpected leftover tensors: {:?}",
        result.unused
    );
    assert_eq!(
        result.applied.len(),
        EXPECTED_TENSORS,
        "every truncated tensor applied"
    );

    // A seeded forward at the real widths (6144 features, 48/12 heads, the
    // full text-fusion transformer): fully finite output proves the dequant
    // + transpose + cast + apply pipeline end-to-end at scale.
    B::seed(&device, 15);
    let cfg = MmditConfig::krea2_truncated(LAYERS);
    let (txtlen, imglen) = (2usize, 4usize);
    let imgdim = cfg.channels * cfg.patch * cfg.patch;
    let img = Tensor::<B, 3>::random([1, imglen, imgdim], Distribution::Default, &device);
    let context = Tensor::<B, 4>::random(
        [1, txtlen, cfg.txtlayers, cfg.txtdim],
        Distribution::Default,
        &device,
    );
    let t = Tensor::<B, 1>::from_data(TensorData::new(vec![0.5f32], [1]), &device);
    // Text at the origin, image tokens on the 2×2 patch grid.
    let mut pos = vec![0.0f32; txtlen * 3];
    for i in 0..imglen {
        pos.extend([0.0, (i / 2) as f32, (i % 2) as f32]);
    }
    let pos = Tensor::<B, 1>::from_data(TensorData::new(pos, [(txtlen + imglen) * 3]), &device)
        .reshape([1, txtlen + imglen, 3]);
    let mask = Tensor::<B, 2>::ones([1, txtlen + imglen], &device);

    let out = model.forward(img, context, t, pos, mask);
    let vals: Vec<f32> = out.into_data().convert::<f32>().into_vec().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "forward output must be fully finite"
    );
    assert!(
        vals.iter().any(|v| *v != 0.0),
        "forward output must be non-zero"
    );
}
