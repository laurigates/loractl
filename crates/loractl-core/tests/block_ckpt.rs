//! #134 block-level gradient checkpointing: the two-phase step
//! (`block_ckpt::checkpointed_step`) must produce the SAME loss and the SAME
//! per-adapter gradients as the monolithic autodiff step it replaces — that
//! equality is the whole numerics gate for the memory restructuring.
//!
//! The comparisons are exact (`into_data` equality) on ndarray: both arms run
//! the identical kernel sequence — the capture forward is the same math as
//! the monolithic forward, and each block replay differentiates the same
//! local subgraph the monolithic graph contains, seeded by the exact
//! boundary cotangent (`d/d(out) Σ out⊙g = g`, and `1.0·g == g` in IEEE 754).
//! If a burn upgrade ever breaks bit-exactness for a legitimate reason
//! (accumulation-order change), relax to a tight relative tolerance with a
//! comment — but start from exact, so any drift is a loud signal.
//!
//! The completeness assertions are the teeth against the silent-skip hazard:
//! `GradientsParams` simply omits params nobody registered, and the optimizer
//! skips missing entries — a bug that dropped block-0's adapters (whose block
//! input is UNTRACKED in the monolithic graph — the text path is frozen)
//! would otherwise pass a value-only comparison vacuously.

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, Param};
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Element, Tensor, TensorData};
use burn_store::{
    KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use loractl_core::CastFloatsAdapter;
use loractl_core::adapters::{LoraAdapters, build_adapters};
use loractl_core::block_ckpt::checkpointed_step;
use loractl_core::config::{LoraConfig, TargetSpec};
use loractl_core::mmdit::{Mmdit, MmditConfig, krea2_positions, patchify};
use loractl_core::quant::QuantBackend;

const BUNDLE: &str = "tests/fixtures/tiny-krea2";
/// The trainer's initial loss scale — both arms fold it in identically, so
/// gradients stay directly comparable (and the comparison covers the scale
/// plumbing itself).
const S: f32 = 16384.0;

/// Deterministic pseudo-random values in [-1, 1] — no backend RNG, identical
/// everywhere (grad_compare's generator).
fn det_vals(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// Load the tiny-krea2 fixture denoiser on `AB`, frozen.
fn load_tiny<AB: AutodiffBackend + QuantBackend>(device: &AB::Device) -> Mmdit<AB> {
    let mut mmdit = Mmdit::<AB>::init(MmditConfig::tiny_krea2(), device);
    let remapper = KeyRemapper::from_patterns(Mmdit::<AB>::key_remap().to_vec()).unwrap();
    let mut store = SafetensorsStore::from_file(format!("{BUNDLE}/raw.safetensors"))
        .remap(remapper)
        .skip_enum_variants(true)
        .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatsAdapter {
            target: <AB::FloatElem as Element>::dtype(),
        }));
    let result = mmdit.load_from(&mut store).expect("fixture loads");
    assert!(result.errors.is_empty(), "load errors: {:?}", result.errors);
    mmdit.no_grad()
}

/// Adapters over every trunk site with BOTH factors seeded deterministically
/// off zero: with the stock zero-init `lora_b`, every `lora_a` gradient is
/// identically zero and the comparison would be toothless on half the params.
fn det_adapters<AB: AutodiffBackend + QuantBackend>(
    mmdit: &Mmdit<AB>,
    device: &AB::Device,
) -> LoraAdapters<AB> {
    let lora = LoraConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
        targets: vec![TargetSpec {
            pattern: r"blocks\.".into(),
            rank: None,
            alpha: None,
        }],
    };
    let mut set = build_adapters::<AB>(&mmdit.injectable_sites(), &lora, device);
    for (i, delta) in set.deltas.iter_mut().enumerate() {
        let [d_in, rank] = delta.lora_a.weight.dims();
        let a: Vec<f32> = det_vals(d_in * rank, 100 + i as u32)
            .iter()
            .map(|v| v * 0.05)
            .collect();
        delta.lora_a.weight =
            Param::from_tensor(Tensor::from_data(TensorData::new(a, [d_in, rank]), device));
        let [rank_b, d_out] = delta.lora_b.weight.dims();
        let b: Vec<f32> = det_vals(rank_b * d_out, 500 + i as u32)
            .iter()
            .map(|v| v * 0.02)
            .collect();
        delta.lora_b.weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(b, [rank_b, d_out]),
            device,
        ));
    }
    set
}

/// The deterministic flow-matching batch grad_compare uses: latent
/// `[1, 4, 8, 8]`, conditioning `[1, 16, 2, 32]`, full mask, t = 0.5.
#[allow(clippy::type_complexity)]
fn batch<AB: AutodiffBackend + QuantBackend>(
    device: &AB::Device,
) -> (
    Tensor<AB, 3>,
    Tensor<AB, 4>,
    Tensor<AB, 1>,
    Tensor<AB, 3>,
    Tensor<AB, 2>,
    Tensor<AB, 3>,
) {
    let patch = MmditConfig::tiny_krea2().patch;
    let (b, z, h, w) = (1usize, 4usize, 8usize, 8usize);
    let latent = Tensor::<AB, 4>::from_data(
        TensorData::new(det_vals(b * z * h * w, 7), [b, z, h, w]),
        device,
    );
    let cond = Tensor::<AB, 4>::from_data(
        TensorData::new(det_vals(16 * 2 * 32, 11), [b, 16, 2, 32]),
        device,
    );
    let eps = Tensor::<AB, 4>::from_data(
        TensorData::new(det_vals(b * z * h * w, 13), [b, z, h, w]),
        device,
    );
    let t_frac = 0.5f32;
    let xt = latent.clone() * (1.0 - t_frac) + eps.clone() * t_frac;
    let target = patchify(eps - latent, patch);
    let img = patchify(xt, patch);
    let (gh, gw) = (h / patch, w / patch);
    let pos = krea2_positions::<AB>(16, gh, gw, b, device);
    let mask = Tensor::ones([b, 16 + gh * gw], device);
    let t = Tensor::<AB, 1>::from_data(TensorData::new(vec![t_frac; b], [b]), device);
    (img, cond, t, pos, mask, target)
}

/// The monolithic step exactly as the trainer's knob-off arm runs it.
fn monolithic_step<AB: AutodiffBackend + QuantBackend>(
    mmdit: &Mmdit<AB>,
    set: &LoraAdapters<AB>,
    device: &AB::Device,
) -> (f32, GradientsParams) {
    let (img, cond, t, pos, mask, target) = batch::<AB>(device);
    let pred = mmdit.forward_with_adapters(img, cond, t, pos, mask, set);
    let diff = pred - target;
    let loss = diff.clone().mul(diff).mean();
    let loss_value: f32 = loss
        .clone()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap()[0];
    (
        loss_value,
        GradientsParams::from_grads((loss * S).backward(), set),
    )
}

/// The checkpointed step on the same inputs (`.inner()` of the same batch).
fn ckpt_step<AB: AutodiffBackend + QuantBackend>(
    mmdit: &Mmdit<AB>,
    set: &LoraAdapters<AB>,
    device: &AB::Device,
) -> (f32, GradientsParams)
where
    AB::InnerBackend: QuantBackend,
{
    let (img, cond, t, pos, mask, target) = batch::<AB>(device);
    let inner = mmdit.valid();
    checkpointed_step::<AB>(
        &inner,
        set,
        img.inner(),
        cond.inner(),
        t.inner(),
        pos.inner(),
        mask.inner(),
        target.inner(),
        S,
    )
}

/// Assert the two GradientsParams agree exactly on every adapter param, and
/// that BOTH are complete (2 grads per delta) and nowhere degenerate-zero.
fn assert_grads_match<AB: AutodiffBackend>(
    set: &LoraAdapters<AB>,
    mono: &GradientsParams,
    ckpt: &GradientsParams,
) {
    let mut compared = 0usize;
    for (delta, site) in set.deltas.iter().zip(&set.targets) {
        for (name, id) in [
            ("lora_a", delta.lora_a.weight.id),
            ("lora_b", delta.lora_b.weight.id),
        ] {
            let gm = mono
                .get::<AB::InnerBackend, 2>(id)
                .unwrap_or_else(|| panic!("monolithic grad missing for {site} {name}"));
            let gc = ckpt
                .get::<AB::InnerBackend, 2>(id)
                .unwrap_or_else(|| panic!("checkpointed grad missing for {site} {name}"));
            let vm = gm.into_data().convert::<f32>().into_vec::<f32>().unwrap();
            let vc = gc.into_data().convert::<f32>().into_vec::<f32>().unwrap();
            let max: f32 = vm.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                max > 0.0,
                "degenerate all-zero gradient at {site} {name} — the comparison \
                 would be vacuous (seed both LoRA factors off zero)"
            );
            assert_eq!(
                vm, vc,
                "gradient mismatch at {site} {name} (checkpointed vs monolithic)"
            );
            compared += 1;
        }
    }
    assert_eq!(
        compared,
        2 * set.deltas.len(),
        "every adapter param must be compared"
    );
}

/// The numerics gate on the fixture weights: identical loss, identical
/// gradients at every site of both trunk blocks — including block 0, whose
/// input is untracked in the monolithic graph (the text path is frozen).
#[test]
fn checkpointed_step_matches_monolithic() {
    type AB = Autodiff<NdArray>;
    let device = Default::default();
    let mmdit = load_tiny::<AB>(&device);
    let set = det_adapters(&mmdit, &device);

    let (loss_mono, grads_mono) = monolithic_step(&mmdit, &set, &device);
    let (loss_ckpt, grads_ckpt) = ckpt_step(&mmdit, &set, &device);

    assert!(
        loss_mono.is_finite() && loss_mono > 0.0,
        "loss: {loss_mono}"
    );
    // Bit-equal loss doubles as the forward-equality proof: the capture
    // forward + head replay is the same kernel sequence as the monolithic
    // forward.
    assert_eq!(loss_ckpt, loss_mono, "loss mismatch (forward diverged)");
    assert_grads_match(&set, &grads_mono, &grads_ckpt);
}

/// The same gate on a quantized (Q8S) frozen base — the real int4/int8
/// trainer path differentiates through `QuantMatmulT` inside the replayed
/// blocks exactly as the monolithic graph does.
#[test]
fn checkpointed_step_matches_monolithic_quantized() {
    use burn::tensor::quantization::QuantValue;
    type B = NdArray;
    type AB = Autodiff<NdArray>;
    let device = Default::default();

    // Random-init base is fine — both arms share the same materialized
    // weights; quantize on the inner backend and lift, the trainer's path.
    let base_inner = Mmdit::<B>::init(MmditConfig::tiny_krea2(), &device)
        .into_quantized(QuantValue::Q8S, &device);
    let mmdit = <Mmdit<AB> as AutodiffModule<AB>>::from_inner(base_inner).no_grad();
    let set = det_adapters(&mmdit, &device);

    let (loss_mono, grads_mono) = monolithic_step(&mmdit, &set, &device);
    let (loss_ckpt, grads_ckpt) = ckpt_step(&mmdit, &set, &device);

    assert!(loss_mono.is_finite(), "loss: {loss_mono}");
    assert_eq!(
        loss_ckpt, loss_mono,
        "loss mismatch (quantized forward diverged)"
    );
    assert_grads_match(&set, &grads_mono, &grads_ckpt);
}

/// The cuda twin of the fixture gate (#130's pattern): `--features cuda`,
/// `--ignored`, Linux + NVIDIA only.
///
///   cargo test -p loractl-core --features cuda --test block_ckpt -- --ignored
#[test]
#[ignore = "needs --features cuda and an NVIDIA GPU (run via just test-cuda)"]
#[cfg(feature = "cuda")]
fn checkpointed_step_matches_monolithic_cuda() {
    type AB = Autodiff<burn::backend::Cuda>;
    let device = burn::backend::cuda::CudaDevice::new(0);
    let mmdit = load_tiny::<AB>(&device);
    let set = det_adapters(&mmdit, &device);

    let (loss_mono, grads_mono) = monolithic_step(&mmdit, &set, &device);
    let (loss_ckpt, grads_ckpt) = ckpt_step(&mmdit, &set, &device);

    assert!(loss_mono.is_finite(), "loss: {loss_mono}");
    assert_eq!(loss_ckpt, loss_mono, "loss mismatch (forward diverged)");
    assert_grads_match(&set, &grads_mono, &grads_ckpt);
}
