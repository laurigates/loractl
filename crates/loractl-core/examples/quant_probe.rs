//! On-box int8 validation (cuda feature) — the empirical proof behind #96:
//! that the ~12.8B Krea 2 base, quantized weight-only to int8, actually fits a
//! 24 GB GPU, and that the quantization is faithful on the REAL weights (not
//! just the tiny fixtures the offline tests bound).
//!
//! It loads a real Krea-2 denoiser (bf16 or scaled-fp8, auto-detected) at the
//! full `MmditConfig::krea2()` depth on cuda, quantized through the EXACT
//! trainer path ([`load_quant_module`]), then reports:
//!
//! 1. **Coverage** — how many base-linear sites quantized vs stayed full
//!    precision (every real site is block-aligned, so a nonzero "plain" count
//!    on the real model would be a bug, not a fixture quirk).
//! 2. **Resident VRAM** — read from `nvidia-smi` after the load. The claim is
//!    ~13–15 GB (12.8 GB int8 + ~1.6 GB f32 scales), well inside 24 GB.
//! 3. **Dequant error** — for a sample of sites, `max`/`mean` relative error of
//!    `dequantize(int8(W))` vs the checkpoint's own f32 `W`. Weight-only
//!    symmetric per-block-32 int8 should sit well under 1% — the
//!    bitsandbytes-int8 / Q8_0 regime.
//!
//! Usage (on a Linux + NVIDIA host, after `cargo build --features cuda`):
//!   cargo run --release -p loractl-core --features cuda --example quant_probe -- \
//!     /path/to/krea2_raw_fp8_scaled.safetensors
//!
//! Not a numerics-golden target and never run in CI — it needs real multi-GB
//! weights and a 24 GB GPU. It is the manual gate before the #25 acceptance run.

fn main() -> anyhow::Result<()> {
    #[cfg(not(feature = "cuda"))]
    anyhow::bail!("build with --features cuda on a Linux+NVIDIA host");
    #[cfg(feature = "cuda")]
    run::main()
}

#[cfg(feature = "cuda")]
mod run {
    use anyhow::{Context, Result};
    use burn::backend::{Cuda, cuda::CudaDevice};
    use burn::tensor::Tensor;
    use burn_store::ModuleSnapshot;
    use loractl_core::TrainEvent;
    use loractl_core::diffusion_trainer::load_quant_module;
    use loractl_core::mmdit::{BaseLinear, Mmdit, MmditConfig};
    use std::path::PathBuf;

    /// Resident VRAM (MiB) on device 0, via `nvidia-smi`. Best-effort — a
    /// missing tool degrades to `None`, not a probe failure.
    fn resident_vram_mib() -> Option<u64> {
        let out = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--id=0",
            ])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .lines()
            .next()?
            .trim()
            .parse()
            .ok()
    }

    /// `max` and `mean` relative error of `dequant(int8(w))` vs the f32 `w`,
    /// over the block-scale-normalized magnitude (guards against a
    /// near-zero-weight blowup).
    fn rel_error(quant_dequant: &[f32], reference: &[f32]) -> (f32, f32) {
        let denom = reference.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-8);
        let mut max = 0f32;
        let mut sum = 0f32;
        for (q, r) in quant_dequant.iter().zip(reference) {
            let e = (q - r).abs() / denom;
            max = max.max(e);
            sum += e;
        }
        (max, sum / reference.len().max(1) as f32)
    }

    pub fn main() -> Result<()> {
        let denoiser = PathBuf::from(
            std::env::args()
                .nth(1)
                .context("arg 1: path to a real Krea-2 denoiser (.safetensors)")?,
        );
        let device = CudaDevice::new(0);

        let base_vram = resident_vram_mib();
        println!(
            "loading {} as int8 at full krea2() depth...",
            denoiser.display()
        );

        // The exact trainer path: init on cuda, flip every aligned base linear
        // to a quantized placeholder, then stream the real weights in (one
        // transient f32 per site — never the full ~49 GB f32 model).
        let module = Mmdit::<Cuda>::init(MmditConfig::krea2(), &device);
        let module = module.into_quantized(&device);
        let mut sink = |event: TrainEvent| {
            if let TrainEvent::Warning { message } = event {
                println!("  {message}");
            }
        };
        let mut module = load_quant_module(
            module,
            &denoiser,
            &Mmdit::<Cuda>::key_remap(),
            "MMDiT",
            &device,
            &mut sink,
        )?;

        // Coverage.
        let sites = module.all_base_linears_mut();
        let quant = sites
            .iter()
            .filter(|(_, b)| matches!(b, BaseLinear::Quant(_)))
            .count();
        let plain = sites.len() - quant;
        println!(
            "coverage: {quant} sites int8, {plain} full-precision (of {})",
            sites.len()
        );

        // Resident VRAM.
        match (base_vram, resident_vram_mib()) {
            (Some(before), Some(after)) => {
                println!(
                    "resident VRAM: {after} MiB (int8 base ≈ {} MiB above the {before} MiB baseline)",
                    after.saturating_sub(before)
                );
            }
            (_, Some(after)) => println!("resident VRAM: {after} MiB"),
            _ => println!("resident VRAM: nvidia-smi unavailable"),
        }

        // Dequant fidelity on a spread of real sites: re-materialize each
        // sampled f32 weight from the checkpoint and compare to the int8 twin.
        // (Re-forcing a handful of tensors is cheap; the point is real-weight
        // error, which the tiny fixtures cannot show.)
        let mut snaps = if loractl_core::fp8::is_fp8_checkpoint(&denoiser)? {
            loractl_core::fp8::load_fp8_snapshots(&denoiser)?
        } else {
            // plain-safetensors snapshots aren't part of the public surface;
            // the real denoisers on the GPU host are all scaled-fp8 repacks, so
            // this arm is only reached by a bf16 snapshot — fall back to the
            // module's own store load for that case is out of scope here.
            anyhow::bail!(
                "dequant-error sampling supports scaled-fp8 checkpoints; got a plain checkpoint"
            )
        };
        let remapper = burn_store::KeyRemapper::from_patterns(Mmdit::<Cuda>::key_remap().to_vec())
            .expect("valid remap");
        snaps = remapper.remap(snaps).0;
        let by_key: std::collections::HashMap<String, _> =
            snaps.into_iter().map(|s| (s.full_path(), s)).collect();

        // Sample every ~40th quantized site for a spread across depth.
        let quant_sites: Vec<_> = sites
            .into_iter()
            .filter_map(|(p, b)| match b {
                BaseLinear::Quant(q) => Some((p, q)),
                BaseLinear::Plain(_) => None,
            })
            .collect();
        let step = (quant_sites.len() / 6).max(1);
        let mut worst = 0f32;
        println!("dequant error (int8 vs checkpoint f32), sampled:");
        for (path, q) in quant_sites.into_iter().step_by(step) {
            let key = format!("{path}.weight");
            let Some(snap) = by_key.get(&key) else {
                continue;
            };
            let ref_data = snap.to_data().map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let reference: Vec<f32> = ref_data.convert::<f32>().into_vec::<f32>().unwrap();
            let dq: Vec<f32> = q
                .weight
                .val()
                .dequantize()
                .into_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .unwrap();
            let (max, mean) = rel_error(&dq, &reference);
            worst = worst.max(max);
            println!("  {path:34} max {max:.4} mean {mean:.5}");
        }
        println!("worst sampled max-rel-error: {worst:.4}");

        Ok(())
    }
}
