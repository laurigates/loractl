//! Diagnostic (wgpu feature): the **quant twin** of `grad_compare` — the same
//! one training step on the tiny-krea2 bundle, but with the frozen base
//! `into_quantized` (int8, so every block-aligned base `Linear` routes through
//! the custom `QuantMatmulT` autodiff op) and lifted via `from_inner`. Run once
//! on ndarray f32 (ground truth) and once per GPU arm — Wgpu f16/f32, cuda
//! f32/f16 (`--features cuda`) — with deterministic identical inputs, then
//! compare the LoRA gradient magnitudes.
//!
//! ## Why this exists (burn#5162 probe)
//!
//! `grad_compare` shows burn 0.21's **generic** GPU autodiff returns
//! exactly-zero LoRA gradients on the f16 configs (Metal, Vulkan AND cuda).
//! `QuantMatmulT` (src/quant.rs) is a **hand-written** `Backward` impl that
//! prepares its op with the **activation `x` as the tracked parent** and never
//! calls `prep.checkpoint(...)` — structurally the same shape as the
//! "input-tracked → heals f32" row in the burn#5162 truth table
//! (`.claude/rules/burn-wgpu-metal-numerics.md`). This path has never run on the
//! broken configs: `grad_compare` exercises only `BaseLinear::Plain` sites, and
//! the int8 QLoRA trainer is guarded to `(ndarray|cuda, f32)`.
//!
//! So the question this answers: **does quantizing the base (routing it through
//! the activation-tracking custom op) change whether f16 adapter gradients
//! survive?** Compare this table to `grad_compare`'s (identical seeds → aligned):
//!   - quant f16 grads CLEAN where plain f16 grads were zero → the custom-op
//!     shape dodges the generic-autodiff defect (a decisive burn#5162 datum).
//!   - quant f16 grads ALSO zero → the "hand-write a custom backward to sidestep
//!     burn#5162" avenue is dead; the pruning defect is upstream of the op.
//! Each arm is `catch_unwind`-guarded: "config X can't even quantize/run" is
//! itself a recorded result (e.g. wgpu quant, or f16 dequant-to-f32).
//!
//! Usage (run alongside `grad_compare` for the plain-base contrast):
//!   cargo run --release -p loractl-core --features cuda,wgpu \
//!     --example quant_grad_compare -- crates/loractl-core/tests/fixtures/tiny-krea2

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
    use burn::module::{AutodiffModule, Module};
    use burn::optim::GradientsParams;
    use burn::tensor::backend::{AutodiffBackend, Backend};
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
    /// backend and identical to `grad_compare` (same LCG + seeds), so the
    /// quant table and the plain table line up column-for-column.
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

    /// One forward+backward on backend `AB`, **quantized base**; returns
    /// (loss, per-site max |grad| over lora_a/lora_b) sorted by site path.
    fn one_step_on<AB: AutodiffBackend + QuantBackend>(
        base: &Path,
        device: AB::Device,
    ) -> Result<(f32, SiteGrads)> {
        let cfg = MmditConfig::tiny_krea2();
        let patch = cfg.patch;

        // burn 0.21's Autodiff has no quantize op (todo!()), so quantize on the
        // INNER backend and lift with `from_inner` — the exact pattern
        // diffusion_trainer.rs / tests/quant_mmdit.rs use. Load real tiny-krea2
        // weights first so the int8 values are the trained ones, not random.
        let mut inner = Mmdit::<AB::InnerBackend>::init(cfg, &device);
        let remapper =
            KeyRemapper::from_patterns(Mmdit::<AB::InnerBackend>::key_remap().to_vec()).unwrap();
        let mut store = SafetensorsStore::from_file(base.join("raw.safetensors"))
            .remap(remapper)
            .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatsAdapter {
                // Autodiff shares its inner backend's float element, so this is
                // the dtype the inner-backend load casts to.
                target: <AB::FloatElem as Element>::dtype(),
            }));
        let result = inner.load_from(&mut store)?;
        if !result.errors.is_empty() {
            bail!("load errors: {:?}", result.errors);
        }
        // Route every block-aligned base linear through the int8 QuantMatmulT op.
        let inner = inner.into_quantized(&device);
        let mmdit = <Mmdit<AB> as AutodiffModule<AB>>::from_inner(inner).no_grad();

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
        // Deterministic A-init identical to grad_compare so the two tables align.
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

        // Deterministic batch — identical to grad_compare.
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

        // Same loss scaling grad_compare uses so the (unscaled) grads compare.
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

    fn max_abs<B: Backend>(t: Tensor<B, 2>) -> f32 {
        t.abs()
            .max()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap()[0]
    }

    fn nan_grads(like: &SiteGrads) -> SiteGrads {
        like.iter()
            .map(|(s, _, _)| (s.clone(), f32::NAN, f32::NAN))
            .collect()
    }

    /// Run one arm under `catch_unwind` so a config that cannot quantize/run
    /// (e.g. wgpu quant, or an f16 dequant-to-f32 path) is a recorded result,
    /// not an abort. Returns NaN placeholders and prints the reason on failure.
    fn run_arm<AB: AutodiffBackend + QuantBackend>(
        name: &str,
        base: &Path,
        device: AB::Device,
        like: &SiteGrads,
    ) -> (f32, SiteGrads) {
        println!("{name}...");
        let base = base.to_path_buf();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            one_step_on::<AB>(&base, device)
        })) {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                println!("  {name} ERRORED: {e:#}");
                (f32::NAN, nan_grads(like))
            }
            Err(panic) => {
                let msg = panic
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "<non-string panic>".to_string());
                println!("  {name} PANICKED: {msg}");
                (f32::NAN, nan_grads(like))
            }
        }
    }

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let base = PathBuf::from(args.get(1).context("arg 1: tiny-krea2 bundle dir")?);

        // ndarray f32 is the ground truth — must succeed (no catch_unwind).
        println!("ndarray f32 (quantized base, ground truth)...");
        let (loss_cpu, grads_cpu) = one_step_on::<Autodiff<NdArray>>(&base, Default::default())?;

        let (loss_f16, grads_f16) = run_arm::<Autodiff<Wgpu<burn::tensor::f16>>>(
            "wgpu f16 (quantized base)",
            &base,
            Default::default(),
            &grads_cpu,
        );
        let (loss_wf32, grads_wf32) = run_arm::<Autodiff<Wgpu>>(
            "wgpu f32 (quantized base)",
            &base,
            Default::default(),
            &grads_cpu,
        );

        #[cfg(feature = "cuda")]
        let (loss_cu32, grads_cu32) = run_arm::<Autodiff<burn::backend::Cuda>>(
            "cuda f32 (quantized base)",
            &base,
            burn::backend::cuda::CudaDevice::new(0),
            &grads_cpu,
        );
        #[cfg(not(feature = "cuda"))]
        let (loss_cu32, grads_cu32) = {
            println!("cuda f32... SKIPPED (build with --features cuda)");
            (f32::NAN, nan_grads(&grads_cpu))
        };
        #[cfg(feature = "cuda")]
        let (loss_cu16, grads_cu16) = run_arm::<Autodiff<burn::backend::Cuda<burn::tensor::f16>>>(
            "cuda f16 (quantized base)",
            &base,
            burn::backend::cuda::CudaDevice::new(0),
            &grads_cpu,
        );
        #[cfg(not(feature = "cuda"))]
        let (loss_cu16, grads_cu16) = {
            println!("cuda f16... SKIPPED (build with --features cuda)");
            (f32::NAN, nan_grads(&grads_cpu))
        };

        println!(
            "\nloss: cpu={loss_cpu:.6} wgpu-f16={loss_f16:.6} wgpu-f32={loss_wf32:.6} cuda-f32={loss_cu32:.6} cuda-f16={loss_cu16:.6}"
        );
        println!(
            "\n{:24} {:>11} {:>11} {:>11} {:>11} {:>11} | {:>11} {:>11} {:>11} {:>11} {:>11} {:>9} {:>9}",
            "site (QUANTIZED base)",
            "A cpu",
            "A wf16",
            "A wf32",
            "A cu32",
            "A cu16",
            "B cpu",
            "B wf16",
            "B wf32",
            "B cu32",
            "B cu16",
            "Bwf16/cpu",
            "Bcu16/cpu",
        );
        for (i, (s, a_c, b_c)) in grads_cpu.iter().enumerate() {
            let (_, a_16, b_16) = &grads_f16[i];
            let (_, a_32, b_32) = &grads_wf32[i];
            let (_, a_cu32, b_cu32) = &grads_cu32[i];
            let (_, a_cu16, b_cu16) = &grads_cu16[i];
            println!(
                "{:24} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} | {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>9.2} {:>9.2}",
                s,
                a_c,
                a_16,
                a_32,
                a_cu32,
                a_cu16,
                b_c,
                b_16,
                b_32,
                b_cu32,
                b_cu16,
                b_16 / b_c,
                b_cu16 / b_c,
            );
        }
        println!(
            "\nRead: B*/cpu ratios near 1.0 = clean; near 0.0 = the burn#5162 zero-grad defect.\n\
             Compare to grad_compare (plain base, identical seeds): if plain f16 B-grads were\n\
             zero and these quantized f16 B-grads are ~1.0, the QuantMatmulT activation-tracking\n\
             backward dodges the generic-autodiff defect."
        );
        Ok(())
    }
}
