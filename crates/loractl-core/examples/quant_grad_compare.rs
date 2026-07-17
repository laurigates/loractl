//! Diagnostic (wgpu feature): does routing the frozen base through the custom
//! `QuantMatmulT` autodiff op change whether f16 LoRA gradients survive the
//! burn#5162 defect? Self-contained: runs one training step on the tiny-krea2
//! bundle **twice per backend** — a **plain** base (generic burn autodiff, which
//! `grad_compare` shows returns exactly-zero f16 grads) and a **quantized** base
//! (`into_quantized` → `from_inner`, so every block-aligned base linear routes
//! through `QuantMatmulT`) — with identical deterministic inputs, then compares
//! the LoRA gradient magnitudes to the ndarray-f32 ground truth.
//!
//! ## Why (burn#5162 probe)
//!
//! `QuantMatmulT` (src/quant.rs) prepares its op with the **activation `x` as the
//! tracked parent** and never calls `prep.checkpoint(...)` — structurally the
//! "input-tracked → heals f32" row in the burn#5162 truth table
//! (`.claude/rules/burn-wgpu-metal-numerics.md`). This path has never run on the
//! broken GPU configs (the int8 QLoRA trainer is guarded to `(ndarray|cuda,
//! f32)`), so the quant-vs-plain contrast below is a genuinely new datum:
//!   - quant f16 B-grads ~1.0 where plain f16 B-grads are ~0.0 → the custom-op
//!     activation-tracking shape dodges the generic-autodiff defect.
//!   - quant f16 B-grads ALSO ~0.0 → the "hand-write a custom backward to
//!     sidestep burn#5162" avenue is dead; the pruning defect is upstream.
//! Each arm is `catch_unwind`-guarded: a config that cannot even quantize/run
//! (e.g. wgpu int8, or an f16 dequant-to-f32 path) is a recorded result.
//!
//! Usage:
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

    /// Deterministic pseudo-random values in [-1, 1] — identical on every backend
    /// (no backend RNG involved), so every arm trains the identical network.
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

    /// One forward+backward on backend `AB`. `quantize` picks the base op path:
    /// `false` → plain `Linear` (generic burn autodiff, the burn#5162 case),
    /// `true` → int8 `QuantMatmulT` (via `into_quantized` + `from_inner`).
    fn one_step_on<AB: AutodiffBackend + QuantBackend>(
        base: &Path,
        device: AB::Device,
        quantize: bool,
    ) -> Result<(f32, SiteGrads)> {
        let cfg = MmditConfig::tiny_krea2();
        let patch = cfg.patch;

        // Quantization (and the int8 op) needs a non-autodiff backend — burn
        // 0.21's Autodiff has no quantize op (todo!()) — so build+load on the
        // INNER backend, optionally quantize, then lift with `from_inner`.
        let mut inner = Mmdit::<AB::InnerBackend>::init(cfg, &device);
        let remapper =
            KeyRemapper::from_patterns(Mmdit::<AB::InnerBackend>::key_remap().to_vec()).unwrap();
        let mut store = SafetensorsStore::from_file(base.join("raw.safetensors"))
            .remap(remapper)
            // Bridge BaseLinear's `Plain`/`Quant` enum-variant path segment
            // (`blocks.0.attn.wq.Plain.weight`) so the checkpoint's
            // `blocks.0.attn.wq.weight` matches — same as `load_module`.
            .skip_enum_variants(true)
            .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatsAdapter {
                // Autodiff shares its inner backend's float element.
                target: <AB::FloatElem as Element>::dtype(),
            }));
        let result = inner.load_from(&mut store)?;
        if !result.errors.is_empty() {
            bail!("load errors: {:?}", result.errors);
        }
        let inner = if quantize {
            inner.into_quantized(&device)
        } else {
            inner
        };
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
        // Deterministic A-init so every arm trains the identical network.
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

        // Deterministic batch.
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

        // Loss scaling so the tiny model's ~1e-6 grads don't underflow f16; the
        // reported grads are unscaled, so f16-vs-cpu ratios stay comparable.
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

    /// Run one arm under `catch_unwind` so a config that cannot quantize/run is a
    /// recorded result, not an abort.
    fn run_arm<AB: AutodiffBackend + QuantBackend>(
        name: &str,
        base: &Path,
        device: AB::Device,
        quantize: bool,
        like: &SiteGrads,
    ) -> (f32, SiteGrads) {
        let base = base.to_path_buf();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            one_step_on::<AB>(&base, device, quantize)
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

    /// Mean per-site B-grad ratio vs the CPU ground truth (skips NaNs). ~1.0 =
    /// clean; ~0.0 = the burn#5162 zero-grad defect; NaN = the arm failed.
    fn mean_b_ratio(cpu: &SiteGrads, arm: &SiteGrads) -> f32 {
        let mut sum = 0.0;
        let mut n = 0;
        for ((_, _, bc), (_, _, ba)) in cpu.iter().zip(arm) {
            if bc.abs() > 0.0 && ba.is_finite() {
                sum += ba / bc;
                n += 1;
            }
        }
        if n == 0 { f32::NAN } else { sum / n as f32 }
    }

    /// Run every backend arm for one base mode (plain or quantized) and print a
    /// one-line-per-arm summary of the mean B-grad ratio vs CPU.
    fn run_mode(base: &Path, quantize: bool) -> Result<()> {
        let label = if quantize {
            "QUANTIZED base (QuantMatmulT)"
        } else {
            "PLAIN base (generic autodiff — burn#5162)"
        };
        println!("\n=== {label} ===");

        // CPU ground truth for this mode (must succeed).
        let (loss_cpu, cpu) = one_step_on::<Autodiff<NdArray>>(base, Default::default(), quantize)?;

        let mut rows: Vec<(&str, f32, f32)> = vec![("ndarray-f32 (truth)", loss_cpu, 1.0)];

        let (l, g) = run_arm::<Autodiff<Wgpu<burn::tensor::f16>>>(
            "wgpu-f16",
            base,
            Default::default(),
            quantize,
            &cpu,
        );
        rows.push(("wgpu-f16", l, mean_b_ratio(&cpu, &g)));

        let (l, g) =
            run_arm::<Autodiff<Wgpu>>("wgpu-f32", base, Default::default(), quantize, &cpu);
        rows.push(("wgpu-f32", l, mean_b_ratio(&cpu, &g)));

        #[cfg(feature = "cuda")]
        {
            let (l, g) = run_arm::<Autodiff<burn::backend::Cuda>>(
                "cuda-f32",
                base,
                burn::backend::cuda::CudaDevice::new(0),
                quantize,
                &cpu,
            );
            rows.push(("cuda-f32", l, mean_b_ratio(&cpu, &g)));
            let (l, g) = run_arm::<Autodiff<burn::backend::Cuda<burn::tensor::f16>>>(
                "cuda-f16",
                base,
                burn::backend::cuda::CudaDevice::new(0),
                quantize,
                &cpu,
            );
            rows.push(("cuda-f16", l, mean_b_ratio(&cpu, &g)));
        }
        #[cfg(not(feature = "cuda"))]
        println!("  (cuda arms SKIPPED — build with --features cuda)");

        println!("  {:<22} {:>12} {:>16}", "arm", "loss", "mean B/cpu");
        for (name, loss, ratio) in rows {
            println!("  {name:<22} {loss:>12.6} {ratio:>16.4}");
        }
        Ok(())
    }

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let base = PathBuf::from(args.get(1).context("arg 1: tiny-krea2 bundle dir")?);

        run_mode(&base, false)?; // plain — reproduces burn#5162
        run_mode(&base, true)?; // quantized — the QuantMatmulT probe

        println!(
            "\nVerdict: compare the f16 rows across the two tables.\n\
             If PLAIN f16 mean B/cpu ~0.0 and QUANTIZED f16 mean B/cpu ~1.0, the\n\
             QuantMatmulT activation-tracking backward dodges the burn#5162 defect.\n\
             If both ~0.0, a custom attention backward would not unblock training."
        );
        Ok(())
    }
}
