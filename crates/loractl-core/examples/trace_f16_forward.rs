//! Diagnostic (wgpu feature): load the real Krea-2-Raw MMDiT on `Wgpu<f16>`,
//! run one traced forward over a real cached batch, and report the first
//! non-finite stage — localizes f16 numeric overflow that the offline f32
//! suite structurally cannot see. Not a test: run on demand while chasing
//! precision bugs.
//!
//! Usage:
//!   cargo run --release -p loractl-core --features wgpu \
//!     --example trace_f16_forward -- <snapshot-dir> <dataset-dir>

fn main() -> anyhow::Result<()> {
    #[cfg(not(feature = "wgpu"))]
    anyhow::bail!("build with --features wgpu");
    #[cfg(feature = "wgpu")]
    wgpu_main::main()
}

#[cfg(feature = "wgpu")]
mod wgpu_main {
    use anyhow::{Context, Result, bail};
    use burn::backend::Wgpu;
    use burn::module::Module;
    use burn::tensor::{DType, Tensor, TensorData};
    use burn_store::{
        KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    };
    use loractl_core::CastFloatsAdapter;
    use loractl_core::config::DatasetConfig;
    use loractl_core::dataset::prepare_dataset;
    use loractl_core::mmdit::{Mmdit, MmditConfig, krea2_positions, patchify};
    use std::path::PathBuf;

    type B = Wgpu<burn::tensor::f16>;

    fn stats<const D: usize>(name: &str, t: &Tensor<B, D>) -> bool {
        let vals: Vec<f32> = t.clone().into_data().convert::<f32>().into_vec().unwrap();
        let bad = vals.iter().filter(|x| !x.is_finite()).count();
        let mx = vals
            .iter()
            .filter(|x| x.is_finite())
            .fold(0.0f32, |a, &x| a.max(x.abs()));
        println!(
            "{name:16} bad={bad:9} max|finite|={mx:.3e}  dims={:?}",
            t.dims()
        );
        bad > 0
    }

    pub fn main() -> Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let base = PathBuf::from(args.get(1).context("arg 1: snapshot dir")?);
        let dataset = PathBuf::from(args.get(2).context("arg 2: dataset dir")?);
        let device = Default::default();

        // Real cached batch (warm cache required — closures bail).
        let prepared = prepare_dataset::<B>(
            &DatasetConfig {
                path: dataset,
                resolution: 256,
                batch_size: 1,
            },
            "krea2-ml512-enc32",
            &device,
            |_| bail!("cold latent cache — run the trainer's encode phase first"),
            |_| bail!("cold cond cache — run the trainer's encode phase first"),
        )?;
        let batches = prepared.batches(1);
        let batch = &batches[0];
        stats("latents", &batch.latents);
        stats("conditioning", &batch.conditioning);

        let cfg = MmditConfig::krea2();
        let patch = cfg.patch;
        println!("loading real MMDiT (f16)...");
        let mut mmdit = Mmdit::<B>::init(cfg, &device);
        let remapper = KeyRemapper::from_patterns(Mmdit::<B>::key_remap().to_vec()).unwrap();
        let mut store = SafetensorsStore::from_file(base.join("raw.safetensors"))
            .remap(remapper)
            // Bridge BaseLinear's `Plain`/`Quant` enum-variant path segment
            // (`blocks.0.attn.wq.Plain.weight`) so the checkpoint's
            // `blocks.0.attn.wq.weight` matches — as `load_module` does.
            .skip_enum_variants(true)
            .with_from_adapter(
                PyTorchToBurnAdapter.chain(CastFloatsAdapter { target: DType::F16 }),
            );
        let result = mmdit.load_from(&mut store)?;
        if !result.errors.is_empty() {
            bail!("load errors: {:?}", result.errors);
        }
        let mmdit = mmdit.no_grad();
        // File truth: first.weight is F32 max|W| 0.61, tmlp.0.weight F32 max
        // 0.39 — if the loaded values differ, the LOAD corrupts them.
        stats("W first (loaded)", &mmdit.first.weight.val());
        stats("W tmlp.0 (loaded)", &mmdit.tmlp.fc1.as_plain().weight.val());
        stats(
            "W blocks0.wq",
            &mmdit.blocks[0].attn.wq.as_plain().weight.val(),
        );

        let [b, z, h, w] = batch.latents.dims();
        let img = patchify(batch.latents.clone(), patch);
        let (gh, gw) = (h / patch, w / patch);
        let txt_len = batch.conditioning.dims()[1];
        let pos = krea2_positions::<B>(txt_len, gh, gw, b, &device);
        let mask = Tensor::cat(
            vec![
                batch.mask.clone().float(),
                Tensor::ones([b, gh * gw], &device),
            ],
            1,
        );
        let t = Tensor::<B, 1>::from_data(TensorData::new(vec![0.5f32; b], [b]), &device);
        let _ = z;

        println!("tracing forward...");
        let trace = mmdit.forward_trace(
            img.clone(),
            batch.conditioning.clone(),
            t.clone(),
            pos,
            mask,
        );
        stats("after_first", &trace.after_first);
        stats("tvec", &trace.tvec);
        stats("after_txtfusion", &trace.after_txtfusion);
        stats("after_txtmlp", &trace.after_txtmlp);
        stats("after_block0", &trace.after_block0);
        stats("output", &trace.output);

        // ---- Replicate block 0's forward step by step over the healthy
        // pre-trunk state (text ++ image, padded to 256), probing each
        // intermediate: localizes WHICH op first goes non-finite in f16.
        println!("block-0 internals...");
        let block = &mmdit.blocks[0];
        let features = trace.after_txtmlp.dims()[2];
        let combined = Tensor::cat(
            vec![trace.after_txtmlp.clone(), trace.after_first.clone()],
            1,
        );
        let fulllen = combined.dims()[1];
        let padlen = fulllen.div_ceil(256) * 256 - fulllen;
        let bsz = combined.dims()[0];
        let combined = if padlen > 0 {
            Tensor::cat(
                vec![combined, Tensor::zeros([bsz, padlen, features], &device)],
                1,
            )
        } else {
            combined
        };
        stats("x (trunk in)", &combined);

        // modulation chunks: tvec + lin, 6 × [b, 1, f]
        let modv = trace.tvec.clone() + block.modulation.lin.val().unsqueeze::<3>();
        let chunk = |i: usize| modv.clone().narrow(2, i * features, features);
        let (prescale, preshift) = (chunk(0), chunk(1));
        stats("prenorm(x)", &block.prenorm.forward(combined.clone()));
        let attn_in = (prescale + 1.0) * block.prenorm.forward(combined.clone()) + preshift;
        stats("attn_in", &attn_in);
        let q_flat = block.attn.wq.forward(attn_in.clone());
        stats("wq(attn_in)", &q_flat);
        let k_flat = block.attn.wk.forward(attn_in.clone());
        stats("wk(attn_in)", &k_flat);
        let gate = block.attn.gate.forward(attn_in.clone());
        stats("gate(attn_in)", &gate);
        // q/k norms over head_dim (128), pre-RoPE
        let l = attn_in.dims()[1];
        let (heads, kv, hd) = (48usize, 12usize, 128usize);
        let q = q_flat.reshape([bsz, l, heads, hd]).swap_dims(1, 2);
        let k = k_flat.reshape([bsz, l, kv, hd]).swap_dims(1, 2);
        let qn = block.attn.qknorm.qnorm.forward(q);
        let kn = block.attn.qknorm.knorm.forward(k);
        stats("qnorm(q)", &qn);
        stats("knorm(k)", &kn);
        // Raw (un-RoPE'd, un-masked) scores over the first kv group just to
        // see the matmul accumulator behavior on a 128-dim reduction.
        let scores = qn.narrow(1, 0, 12).matmul(kn.swap_dims(2, 3)) / (hd as f64).sqrt();
        stats("scores(no-rope)", &scores);
        // The MLP side over a healthy input: gate/up/product on attn_in.
        let g = block.mlp.gate.forward(attn_in.clone());
        let u = block.mlp.up.forward(attn_in.clone());
        stats("mlp.gate(x)", &g);
        stats("mlp.up(x)", &u);
        let prod = burn::tensor::activation::silu(g) * u;
        stats("silu(g)*u", &prod);
        stats("mlp.down(prod)", &block.mlp.down.forward(prod));
        Ok(())
    }
}
