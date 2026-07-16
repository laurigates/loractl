//! Diagnostic (wgpu feature): the same one training step on the tiny-krea2
//! bundle, once on ndarray f32 (ground truth) and once per GPU arm — Wgpu
//! f16/f32, candle-metal bf16 (`--features candle`), cuda f32/f16
//! (`--features cuda`) — deterministic identical inputs, then compare the
//! LoRA gradient magnitudes. Localizes backend-level gradient corruption
//! that same-backend comparisons (e.g. ckpt-vs-stored, both on wgpu)
//! structurally cannot see.
//!
//! Usage:
//!   cargo run --release -p loractl-core --features wgpu \
//!     --example grad_compare -- crates/loractl-core/tests/fixtures/tiny-krea2
//!   # on a Linux+NVIDIA host, add the cuda arms:
//!   cargo run --release -p loractl-core --features cuda,wgpu \
//!     --example grad_compare -- crates/loractl-core/tests/fixtures/tiny-krea2

fn main() -> anyhow::Result<()> {
    #[cfg(not(feature = "wgpu"))]
    anyhow::bail!("build with --features wgpu");
    #[cfg(feature = "wgpu")]
    run::main()
}

#[cfg(feature = "wgpu")]
mod run {
    use anyhow::{Context, Result, bail};
    use burn::backend::{Autodiff, NdArray, Wgpu};
    use burn::module::Module;
    use burn::optim::GradientsParams;
    use burn::tensor::backend::AutodiffBackend;
    use burn::tensor::{Element, Tensor, TensorData};
    use burn_store::{
        KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    };
    use loractl_core::CastFloatsAdapter;
    use loractl_core::adapters::build_adapters;
    use loractl_core::config::{LoraConfig, TargetSpec};
    use loractl_core::mmdit::{Mmdit, MmditConfig, krea2_positions, patchify};
    use loractl_core::quant::QuantBackend;
    use std::path::{Path, PathBuf};

    /// Deterministic pseudo-random values in [-1, 1] — identical on every
    /// backend (no backend RNG involved).
    fn det_vals(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// (site path, max |grad A|, max |grad B|) per adapter site.
    type SiteGrads = Vec<(String, f32, f32)>;

    /// One forward+backward on backend `AB`; returns (loss, per-site max |grad|
    /// over lora_a/lora_b) sorted by site path.
    fn one_step<AB: AutodiffBackend + QuantBackend>(base: &Path) -> Result<(f32, SiteGrads)> {
        one_step_on::<AB>(base, Default::default())
    }

    fn one_step_on<AB: AutodiffBackend + QuantBackend>(
        base: &Path,
        device: AB::Device,
    ) -> Result<(f32, SiteGrads)> {
        let cfg = MmditConfig::tiny_krea2();
        let patch = cfg.patch;

        let mut mmdit = Mmdit::<AB>::init(cfg, &device);
        let remapper = KeyRemapper::from_patterns(Mmdit::<AB>::key_remap().to_vec()).unwrap();
        let mut store = SafetensorsStore::from_file(base.join("raw.safetensors"))
            .remap(remapper)
            .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatsAdapter {
                target: <AB::FloatElem as Element>::dtype(),
            }));
        let result = mmdit.load_from(&mut store)?;
        if !result.errors.is_empty() {
            bail!("load errors: {:?}", result.errors);
        }
        let mmdit = mmdit.no_grad();

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
        let sites = mmdit.injectable_sites();
        let mut set = build_adapters::<AB>(&sites, &lora, &device);
        // Deterministic A-init: build_adapters draws from the backend RNG,
        // whose streams differ across backends — overwrite so every arm
        // trains the identical network and the grad ratios are meaningful.
        for (i, delta) in set.deltas.iter_mut().enumerate() {
            let [d_in, rank] = delta.lora_a.weight.dims();
            let vals: Vec<f32> = det_vals(d_in * rank, 100 + i as u32)
                .iter()
                .map(|v| v * 0.05)
                .collect();
            delta.lora_a.weight = burn::module::Param::from_tensor(Tensor::from_data(
                TensorData::new(vals, [d_in, rank]),
                &device,
            ));
        }

        // Deterministic batch: latent [1, 4, 8, 8], conditioning
        // [1, 16, 2, 32], full mask, t = 0.5.
        let (b, z, h, w) = (1usize, 4usize, 8usize, 8usize);
        let latent = Tensor::<AB, 4>::from_data(
            TensorData::new(det_vals(b * z * h * w, 7), [b, z, h, w]),
            &device,
        );
        let cond = Tensor::<AB, 4>::from_data(
            TensorData::new(det_vals(16 * 2 * 32, 11), [b, 16, 2, 32]),
            &device,
        );
        let eps = Tensor::<AB, 4>::from_data(
            TensorData::new(det_vals(b * z * h * w, 13), [b, z, h, w]),
            &device,
        );

        let t_frac = 0.5f32;
        let xt = latent.clone() * (1.0 - t_frac) + eps.clone() * t_frac;
        let target = patchify(eps - latent, patch);

        let img = patchify(xt, patch);
        let (gh, gw) = (h / patch, w / patch);
        let pos = krea2_positions::<AB>(16, gh, gw, b, &device);
        let mask = Tensor::ones([b, 16 + gh * gw], &device);
        let t = Tensor::<AB, 1>::from_data(TensorData::new(vec![t_frac; b], [b]), &device);

        let pred = mmdit.forward_with_adapters(img, cond, t, pos, mask, &set);
        let diff = pred - target;
        let loss = diff.clone().mul(diff).mean();
        let loss_v: f32 = loss
            .clone()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap()[0];

        // The trainer's loss scaling, applied on every backend so the
        // reported (unscaled) gradients stay directly comparable: without
        // it the tiny model's ~1e-6 gradients underflow f16 outright.
        const S: f32 = 16384.0;
        let grads = GradientsParams::from_grads((loss * S).backward(), &set);
        let mut out = Vec::new();
        for (delta, target) in set.deltas.iter().zip(&set.targets) {
            let ga = grads
                .get::<AB::InnerBackend, 2>(delta.lora_a.weight.id)
                .map(|g| max_abs(g) / S)
                .unwrap_or(f32::NAN);
            let gb = grads
                .get::<AB::InnerBackend, 2>(delta.lora_b.weight.id)
                .map(|g| max_abs(g) / S)
                .unwrap_or(f32::NAN);
            out.push((target.clone(), ga, gb));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok((loss_v, out))
    }

    fn max_abs<B: burn::tensor::backend::Backend>(t: Tensor<B, 2>) -> f32 {
        t.abs()
            .max()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap()[0]
    }

    /// NaN placeholders shaped like `like`, for arms whose feature is off —
    /// keeps the table column-comparable regardless of the build.
    fn nan_grads(like: &SiteGrads) -> SiteGrads {
        like.iter()
            .map(|(s, _, _)| (s.clone(), f32::NAN, f32::NAN))
            .collect()
    }

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let base = PathBuf::from(args.get(1).context("arg 1: tiny-krea2 bundle dir")?);

        println!("ndarray f32 (ground truth)...");
        let (loss_cpu, grads_cpu) = one_step::<Autodiff<NdArray>>(&base)?;
        println!("wgpu f16...");
        let (loss_f16, grads_f16) = one_step::<Autodiff<Wgpu<burn::tensor::f16>>>(&base)?;
        // The candle arm needs its own feature; without it, report NaN
        // placeholders so the table stays comparable.
        #[cfg(feature = "candle")]
        let (loss_cb, grads_cb) = {
            println!("candle-metal bf16...");
            #[allow(deprecated)]
            type CandleBf16 = burn::backend::Candle<burn::tensor::bf16>;
            #[allow(deprecated)]
            let device_cb = burn::backend::candle::CandleDevice::metal(0);
            one_step_on::<Autodiff<CandleBf16>>(&base, device_cb)?
        };
        #[cfg(not(feature = "candle"))]
        let (loss_cb, grads_cb) = {
            println!("candle-metal bf16... SKIPPED (build with --features candle)");
            (f32::NAN, nan_grads(&grads_cpu))
        };
        println!("wgpu f32...");
        let (loss_wf32, grads_wf32) = one_step::<Autodiff<Wgpu>>(&base)?;
        // The cuda arms need their own feature (Linux + NVIDIA + CUDA toolkit
        // only); without it, NaN placeholders keep the table comparable. The
        // f16 arm instantiates Cuda<f16> directly — the trainer's require_f32
        // gate is a config-surface rule, not a backend limitation, and the f16
        // data point is exactly what burn#5162 needs cross-platform.
        #[cfg(feature = "cuda")]
        let (loss_cu32, grads_cu32) = {
            println!("cuda f32...");
            one_step_on::<Autodiff<burn::backend::Cuda>>(
                &base,
                burn::backend::cuda::CudaDevice::new(0),
            )?
        };
        #[cfg(not(feature = "cuda"))]
        let (loss_cu32, grads_cu32) = {
            println!("cuda f32... SKIPPED (build with --features cuda)");
            (f32::NAN, nan_grads(&grads_cpu))
        };
        #[cfg(feature = "cuda")]
        let (loss_cu16, grads_cu16) = {
            println!("cuda f16...");
            one_step_on::<Autodiff<burn::backend::Cuda<burn::tensor::f16>>>(
                &base,
                burn::backend::cuda::CudaDevice::new(0),
            )?
        };
        #[cfg(not(feature = "cuda"))]
        let (loss_cu16, grads_cu16) = {
            println!("cuda f16... SKIPPED (build with --features cuda)");
            (f32::NAN, nan_grads(&grads_cpu))
        };

        println!(
            "\nloss: cpu={loss_cpu:.6} wgpu-f16={loss_f16:.6} candle-bf16={loss_cb:.6} wgpu-f32={loss_wf32:.6} cuda-f32={loss_cu32:.6} cuda-f16={loss_cu16:.6}"
        );
        println!(
            "\n{:24} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} | {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>8} {:>8}",
            "site",
            "A cpu",
            "A wf16",
            "A cbf16",
            "A wf32",
            "A cu32",
            "A cu16",
            "B cpu",
            "B wf16",
            "B cbf16",
            "B wf32",
            "B cu32",
            "B cu16",
            "cb/cpu",
            "cu32/cpu"
        );
        for (i, (s, a_c, b_c)) in grads_cpu.iter().enumerate() {
            let (_, a_16, b_16) = &grads_f16[i];
            let (_, a_cb, b_cb) = &grads_cb[i];
            let (_, a_32, b_32) = &grads_wf32[i];
            let (_, a_cu32, b_cu32) = &grads_cu32[i];
            let (_, a_cu16, b_cu16) = &grads_cu16[i];
            println!(
                "{:24} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} | {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>8.2} {:>8.2}",
                s,
                a_c,
                a_16,
                a_cb,
                a_32,
                a_cu32,
                a_cu16,
                b_c,
                b_16,
                b_cb,
                b_32,
                b_cu32,
                b_cu16,
                b_cb / b_c,
                b_cu32 / b_c
            );
        }
        Ok(())
    }
}
