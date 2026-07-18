//! Diagnostic (wgpu feature): bisect the Metal gradient corruption that
//! `grad_compare` exposes, separating the burn-store load path from the
//! compute graph and localizing which backward variant breaks. This is the
//! fixture-free reproduction backing the upstream burn report
//! (tracel-ai/burn#5162, tracked on issue #25): on Apple Silicon, `no-load`
//! reports every LoRA gradient as NaN on wgpu (f16 AND f32) in a fresh
//! process while ndarray is finite. Whether tracking the model input avoids
//! the defect depends on the weight values — see `adapters` (random init:
//! yes) vs `workaround` (loaded weights: f32 becomes exactly correct, f16
//! shifts to a *wrong forward* with exactly-zero grads).
//!
//! Modes:
//!   verify-load <bundle>  — load the tiny MMDiT on wgpu-f32 AND ndarray-f32,
//!                           read every param back, compare elementwise.
//!                           Corruption here = bulk device-write bug.
//!                           (Observed clean: 92/92 tensors byte-identical.)
//!   no-load               — random-init tiny MMDiT per backend (no
//!                           safetensors, no burn-store), one forward+backward,
//!                           report loss finiteness + NaN grad count.
//!                           (Observed: 28/28 NaN grads on wgpu f16 + f32.)
//!   stages                — loss on each `MmditTrace` stage separately,
//!                           backward to the *inputs* (img/context/t).
//!                           (Observed finite everywhere on every backend.)
//!   adapters              — single-site adapter patterns × input-grad
//!                           on/off, random init. (Observed: with the input
//!                           tracked, all grads are finite; a prior
//!                           same-dtype full backward in the same process
//!                           also "heals" the params-only run — kernel/pool
//!                           state dependent.)
//!   workaround <bundle>   — loaded weights + input tracking, grads compared
//!                           NUMERICALLY vs CPU at two loss scales.
//!                           (Observed: wgpu-f32+tracking matches CPU to
//!                           ratio 1.000 on all 14 sites with a bit-identical
//!                           loss — while params-only f32 NaNs its forward.
//!                           wgpu-f16+tracking returns exactly-zero grads and
//!                           a wrong forward, 0.777 vs CPU 0.803, unchanged
//!                           at S=64 vs S=16384 — dropped values, not f16
//!                           range overflow.)
//!
//! Run: cargo run --release -p loractl-core --features wgpu \
//!        --example metal_bisect -- <mode> [bundle]

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

    fn det_vals(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    fn load_mmdit<B: Backend>(base: &Path, device: &B::Device) -> Result<Mmdit<B>> {
        let cfg = MmditConfig::tiny_krea2();
        let mut mmdit = Mmdit::<B>::init(cfg, device);
        let remapper = KeyRemapper::from_patterns(Mmdit::<B>::key_remap().to_vec()).unwrap();
        let mut store = SafetensorsStore::from_file(base.join("raw.safetensors"))
            .remap(remapper)
            // Bridge BaseLinear's `Plain`/`Quant` enum-variant path segment
            // (`blocks.0.attn.wq.Plain.weight`) so the checkpoint's
            // `blocks.0.attn.wq.weight` matches — as `load_module` does.
            .skip_enum_variants(true)
            .with_from_adapter(PyTorchToBurnAdapter.chain(CastFloatsAdapter {
                target: <B::FloatElem as Element>::dtype(),
            }));
        let result = mmdit.load_from(&mut store)?;
        if !result.errors.is_empty() {
            bail!("load errors: {:?}", result.errors);
        }
        Ok(mmdit)
    }

    fn verify_load(base: &Path) -> Result<()> {
        println!("loading on ndarray f32 (reference)...");
        let cpu: Mmdit<NdArray> = load_mmdit(base, &Default::default())?;
        println!("loading on wgpu f32...");
        let gpu: Mmdit<Wgpu> = load_mmdit(base, &Default::default())?;

        let cpu_snaps = cpu.collect(None, None, false);
        let gpu_snaps = gpu.collect(None, None, false);
        assert_eq!(cpu_snaps.len(), gpu_snaps.len());
        let (mut n_bad, mut n_nan, mut worst, mut worst_path) =
            (0usize, 0usize, 0f32, String::new());
        for (c, g) in cpu_snaps.iter().zip(gpu_snaps.iter()) {
            let cpath = c.full_path();
            assert_eq!(cpath, g.full_path());
            let cv: Vec<f32> = c.to_data().unwrap().convert::<f32>().into_vec().unwrap();
            let gv: Vec<f32> = g.to_data().unwrap().convert::<f32>().into_vec().unwrap();
            let mut tensor_bad = false;
            for (a, b) in cv.iter().zip(gv.iter()) {
                if !b.is_finite() {
                    n_nan += 1;
                    tensor_bad = true;
                    continue;
                }
                let d = (a - b).abs();
                if d > worst {
                    worst = d;
                    worst_path = cpath.clone();
                }
                if d > 1e-5 {
                    tensor_bad = true;
                }
            }
            if tensor_bad {
                n_bad += 1;
                if n_bad <= 8 {
                    println!("  CORRUPT: {cpath}");
                }
            }
        }
        println!(
            "\ntensors compared: {}  corrupted: {n_bad}  non-finite elems: {n_nan}  worst |cpu-gpu|: {worst:.3e} ({worst_path})",
            cpu_snaps.len()
        );
        Ok(())
    }

    /// Random-init forward+backward on one backend; no file I/O at all.
    fn no_load_step<AB: AutodiffBackend + QuantBackend>(label: &str) -> Result<()> {
        no_load_step_on::<AB>(label, r"blocks\.", false)
    }

    /// (site path, max |grad A|, max |grad B|) per adapter site.
    type SiteGrads = Vec<(String, f32, f32)>;

    /// Loaded-weights deterministic step (grad_compare's setup) with the
    /// input optionally tracked; returns per-site (path, max|dA|, max|dB|)
    /// so the input-tracking workaround can be verified *numerically*
    /// against CPU ground truth, not just for finiteness.
    fn loaded_step<AB: AutodiffBackend + QuantBackend>(
        base: &Path,
        input_grad: bool,
        loss_scale: f32,
    ) -> Result<(f32, SiteGrads)> {
        let device = AB::Device::default();
        let patch = MmditConfig::tiny_krea2().patch;
        let mmdit: Mmdit<AB> = load_mmdit(base, &device)?;
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

        let (b, z, h, w) = (1usize, 4usize, 8usize, 8usize);
        let latent = Tensor::<AB, 4>::from_data(
            TensorData::new(det_vals(b * z * h * w, 7), [b, z, h, w]),
            &device,
        );
        let latent = if input_grad {
            latent.require_grad()
        } else {
            latent
        };
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

        let s = loss_scale;
        let grads = GradientsParams::from_grads((loss * s).backward(), &set);
        let mut out = Vec::new();
        for (delta, site) in set.deltas.iter().zip(&set.targets) {
            let m = |id| {
                grads
                    .get::<AB::InnerBackend, 2>(id)
                    .map(|g| {
                        g.abs()
                            .max()
                            .into_data()
                            .convert::<f32>()
                            .into_vec::<f32>()
                            .unwrap()[0]
                            / s
                    })
                    .unwrap_or(f32::NAN)
            };
            out.push((
                site.clone(),
                m(delta.lora_a.weight.id),
                m(delta.lora_b.weight.id),
            ));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok((loss_v, out))
    }

    fn workaround(base: &Path) -> Result<()> {
        println!("ndarray f32 (ground truth), input tracked, S=16384...");
        let (loss_cpu, g_cpu) = loaded_step::<Autodiff<NdArray>>(base, true, 16384.0)?;
        println!("wgpu f32, input tracked, S=16384...");
        let (loss_32t, g_32t) = loaded_step::<Autodiff<Wgpu>>(base, true, 16384.0)?;
        println!("wgpu f16, input tracked, S=16384...");
        let (loss_16t, g_16t) =
            loaded_step::<Autodiff<Wgpu<burn::tensor::f16>>>(base, true, 16384.0)?;
        println!("wgpu f16, input tracked, S=64 (rule out f16 range overflow)...");
        let (loss_16s, g_16s) = loaded_step::<Autodiff<Wgpu<burn::tensor::f16>>>(base, true, 64.0)?;

        println!(
            "\nloss: cpu={loss_cpu:.6}  wgpu-f32(ig)={loss_32t:.6}  wgpu-f16(ig)={loss_16t:.6}  wgpu-f16(ig,S64)={loss_16s:.6}"
        );
        println!(
            "\n{:24} {:>11} {:>11} {:>11} {:>11} | {:>9} {:>9} {:>9}",
            "site", "B cpu", "B wf32+ig", "B wf16+ig", "B wf16+igS64", "r32", "r16", "r16s"
        );
        let mut bad32 = 0usize;
        for (((s, _a_c, b_c), (_, _, b_3)), ((_, _, b_t), (_, _, b_s))) in g_cpu
            .iter()
            .zip(g_32t.iter())
            .zip(g_16t.iter().zip(g_16s.iter()))
        {
            let (r3, rt, rs) = (b_3 / b_c, b_t / b_c, b_s / b_c);
            println!(
                "{s:24} {b_c:>11.3e} {b_3:>11.3e} {b_t:>11.3e} {b_s:>11.3e} | {r3:>9.3} {rt:>9.3} {rs:>9.3}"
            );
            if !r3.is_finite() || (r3 - 1.0).abs() > 0.05 {
                bad32 += 1;
            }
        }
        println!(
            "\nwgpu-f32 + input tracking vs CPU: {}",
            if bad32 == 0 {
                "MATCHES (any f16 NaN above is then range, not kernel)".to_string()
            } else {
                format!("{bad32} site(s) diverge — kernel defect persists with input tracking")
            }
        );
        Ok(())
    }

    fn no_load_step_on<AB: AutodiffBackend + QuantBackend>(
        label: &str,
        pattern: &str,
        input_grad: bool,
    ) -> Result<()> {
        let device = AB::Device::default();
        AB::seed(&device, 42);
        let cfg = MmditConfig::tiny_krea2();
        let patch = cfg.patch;
        let mmdit = Mmdit::<AB>::init(cfg, &device).no_grad();

        let lora = LoraConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            targets: vec![TargetSpec {
                pattern: pattern.into(),
                rank: None,
                alpha: None,
            }],
        };
        let sites = mmdit.injectable_sites();
        let mut set = build_adapters::<AB>(&sites, &lora, &device);
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

        let (b, z, h, w) = (1usize, 4usize, 8usize, 8usize);
        let latent = Tensor::<AB, 4>::from_data(
            TensorData::new(det_vals(b * z * h * w, 7), [b, z, h, w]),
            &device,
        );
        let latent = if input_grad {
            latent.require_grad()
        } else {
            latent
        };
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
        let target = patchify(eps - latent.clone(), patch);
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

        const S: f32 = 16384.0;
        let raw = (loss * S).backward();
        let d_img = if input_grad {
            match latent.grad(&raw) {
                Some(g) => {
                    let m: f32 = g
                        .abs()
                        .max()
                        .into_data()
                        .convert::<f32>()
                        .into_vec::<f32>()
                        .unwrap()[0];
                    if m.is_finite() {
                        format!("{m:.3e}")
                    } else {
                        "NaN".into()
                    }
                }
                None => "-".into(),
            }
        } else {
            "-".into()
        };
        let grads = GradientsParams::from_grads(raw, &set);
        let (mut n_grads, mut n_nan) = (0usize, 0usize);
        for delta in set.deltas.iter() {
            for id in [delta.lora_a.weight.id, delta.lora_b.weight.id] {
                if let Some(g) = grads.get::<AB::InnerBackend, 2>(id) {
                    n_grads += 1;
                    let m: f32 = g
                        .abs()
                        .max()
                        .into_data()
                        .convert::<f32>()
                        .into_vec::<f32>()
                        .unwrap()[0];
                    if !m.is_finite() {
                        n_nan += 1;
                    }
                }
            }
        }
        println!(
            "{label:34} loss={loss_v:>12.6}  loss_finite={:<5}  grads={n_grads}  nan_grads={n_nan}  d/img={d_img}",
            loss_v.is_finite()
        );
        Ok(())
    }

    /// Stage-level localization: loss on each trace stage separately, then
    /// backward to the *inputs* (img/context/t require_grad). The earliest
    /// stage whose input grads NaN bounds the broken op.
    fn stages<AB: AutodiffBackend + QuantBackend>(label: &str) -> Result<()> {
        let device = AB::Device::default();
        AB::seed(&device, 42);
        let cfg = MmditConfig::tiny_krea2();
        let patch = cfg.patch;
        let mmdit = Mmdit::<AB>::init(cfg, &device).no_grad();

        let (b, z, h, w) = (1usize, 4usize, 8usize, 8usize);
        let (gh, gw) = (h / patch, w / patch);
        println!("--- {label} ---");
        for stage in [
            "after_first",
            "tvec",
            "after_txtfusion",
            "after_txtmlp",
            "after_block0",
            "output",
        ] {
            // Fresh graph per stage (backward consumes it).
            let latent = Tensor::<AB, 4>::from_data(
                TensorData::new(det_vals(b * z * h * w, 7), [b, z, h, w]),
                &device,
            );
            let img = patchify(latent, patch).require_grad();
            let context = Tensor::<AB, 4>::from_data(
                TensorData::new(det_vals(16 * 2 * 32, 11), [b, 16, 2, 32]),
                &device,
            )
            .require_grad();
            let t = Tensor::<AB, 1>::from_data(TensorData::new(vec![0.5f32; b], [b]), &device)
                .require_grad();
            let pos = krea2_positions::<AB>(16, gh, gw, b, &device);
            let mask = Tensor::ones([b, 16 + gh * gw], &device);

            let trace = mmdit.forward_trace(img.clone(), context.clone(), t.clone(), pos, mask);
            let y = match stage {
                "after_first" => trace.after_first,
                "tvec" => trace.tvec,
                "after_txtfusion" => trace.after_txtfusion,
                "after_txtmlp" => trace.after_txtmlp,
                "after_block0" => trace.after_block0,
                _ => trace.output,
            };
            const S: f32 = 16384.0;
            let loss = (y.clone() * y).mean();
            let loss_v: f32 = loss
                .clone()
                .into_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .unwrap()[0];
            let grads = (loss * S).backward();
            let fmt = |m: Option<f32>| match m {
                None => "-".to_string(),
                Some(v) if v.is_finite() => format!("{v:.3e}"),
                Some(_) => "NaN".to_string(),
            };
            let g_img = img.grad(&grads).map(|g| max_abs3(g) / S);
            let g_ctx = context.grad(&grads).map(|g| max_abs4(g) / S);
            let g_t = t.grad(&grads).map(|g| max_abs1(g) / S);
            println!(
                "  {stage:16} loss={loss_v:>12.6e}  d/img={:>10}  d/ctx={:>10}  d/t={:>10}",
                fmt(g_img),
                fmt(g_ctx),
                fmt(g_t)
            );
        }
        Ok(())
    }

    fn max_abs1<Bk: Backend>(t: Tensor<Bk, 1>) -> f32 {
        t.abs()
            .max()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap()[0]
    }
    fn max_abs3<Bk: Backend>(t: Tensor<Bk, 3>) -> f32 {
        t.abs()
            .max()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap()[0]
    }
    fn max_abs4<Bk: Backend>(t: Tensor<Bk, 4>) -> f32 {
        t.abs()
            .max()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap()[0]
    }

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().collect();
        match args.get(1).map(String::as_str) {
            Some("verify-load") => {
                let base = PathBuf::from(args.get(2).context("arg 2: bundle dir")?);
                verify_load(&base)
            }
            Some("no-load") => {
                no_load_step::<Autodiff<NdArray>>("ndarray f32")?;
                no_load_step::<Autodiff<Wgpu<burn::tensor::f16>>>("wgpu f16")?;
                no_load_step::<Autodiff<Wgpu>>("wgpu f32")?;
                Ok(())
            }
            Some("stages") => {
                stages::<Autodiff<NdArray>>("ndarray f32")?;
                stages::<Autodiff<Wgpu>>("wgpu f32")?;
                stages::<Autodiff<Wgpu<burn::tensor::f16>>>("wgpu f16")?;
                Ok(())
            }
            Some("adapters") => {
                for (name, pat) in [
                    ("all", r"blocks\."),
                    ("b0.attn.wq", r"^blocks\.0\.attn\.wq$"),
                    ("b1.attn.wq", r"^blocks\.1\.attn\.wq$"),
                    ("b1.attn.wo", r"^blocks\.1\.attn\.wo$"),
                    ("b1.mlp.down", r"^blocks\.1\.mlp\.down$"),
                ] {
                    for ig in [true, false] {
                        let tag = if ig { "igrad" } else { "     " };
                        no_load_step_on::<Autodiff<NdArray>>(
                            &format!("cpu  {tag} {name}"),
                            pat,
                            ig,
                        )?;
                        no_load_step_on::<Autodiff<Wgpu>>(&format!("wgpu {tag} {name}"), pat, ig)?;
                    }
                }
                Ok(())
            }
            Some("workaround") => {
                let base = PathBuf::from(args.get(2).context("arg 2: bundle dir")?);
                workaround(&base)
            }
            _ => bail!(
                "mode: verify-load <bundle> | no-load | stages | adapters | workaround <bundle>"
            ),
        }
    }
}
