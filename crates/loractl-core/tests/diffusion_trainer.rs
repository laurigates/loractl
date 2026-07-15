//! The M14 (#25) end-to-end proof, fully offline: the composed tiny Krea 2
//! bundle (`reference/krea2_reference.py` — real loading paths, matched
//! seams) trains a LoRA through the [`Trainer`] contract, emitting the same
//! `TrainEvent`s every other trainer emits, and exports a kohya-ss adapter
//! at every checkpoint.
//!
//! This is the whole stack in one test: M12 scans + buckets + caches the
//! dataset with the M9 VAE and M10 conditioner (then drops them), the M8
//! objective drives the M11 MMDiT via the M6 adapter injection, and the M6
//! kohya export writes the artifact ComfyUI loads. What it deliberately does
//! NOT claim: semantic quality (tiny random weights) — the real-weights
//! parity proofs live per-milestone, and the real training run is the
//! interop step tracked on #25.

use loractl_core::config::{
    ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, ModelVariant, OptimConfig,
    OutputConfig, TargetSpec, TaskKind,
};
use loractl_core::{DiffusionTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::{Path, PathBuf};

const BUNDLE: &str = "tests/fixtures/tiny-krea2";
const DATASET: &str = "tests/fixtures/dataset-tiny";
const STEPS: u64 = 12;

/// A unique temp dir, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("loractl-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Copy the checked-in tiny dataset into a temp dir — the pipeline writes its
/// `.loractl-cache/` INTO the dataset folder, and tests must not dirty the
/// repo tree.
///
/// On top of the four 48×32 landscapes, a fifth 32×48 portrait is generated
/// here (review): it lands in its own aspect bucket, so the trainer's step
/// loop sees heterogeneous batch geometry — two b=2 landscape batches
/// ([z, 4, 6] latents) plus a b=1 portrait remainder ([z, 6, 4]) — pinning
/// that gh/gw, positions, mask, and patchify are recomputed per batch, not
/// hoisted from batch 0.
fn staged_dataset(out: &TempDir) -> PathBuf {
    let dst = out.0.join("dataset");
    std::fs::create_dir_all(&dst).unwrap();
    for entry in std::fs::read_dir(DATASET).expect("checked-in dataset present") {
        let path = entry.unwrap().path();
        if path.is_file() {
            std::fs::copy(&path, dst.join(path.file_name().unwrap())).unwrap();
        }
    }
    let portrait = image::RgbImage::from_fn(32, 48, |x, y| {
        image::Rgb([(x * 8) as u8, (y * 5) as u8, 128])
    });
    portrait.save(dst.join("portrait.png")).unwrap();
    std::fs::write(dst.join("portrait.txt"), "a portrait gradient").unwrap();
    dst
}

/// The sorted `(file name, mtime)` listing of the dataset's cache dir — the
/// warm-rerun assertion below compares snapshots, so a run-unstable cache
/// fingerprint (which would silently re-encode the whole dataset every run)
/// shows up as new files or fresher mtimes.
fn cache_snapshot(dataset: &Path) -> Vec<(String, std::time::SystemTime)> {
    let mut listing: Vec<_> = std::fs::read_dir(dataset.join(".loractl-cache"))
        .expect("the cache dir exists after a run")
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().to_string_lossy().into_owned(),
                entry.metadata().unwrap().modified().unwrap(),
            )
        })
        .collect();
    listing.sort();
    listing
}

fn config(out: &TempDir, dataset: PathBuf) -> TrainConfig {
    TrainConfig {
        steps: STEPS,
        seed: 42,
        task: TaskKind::FlowMatching,
        model: ModelConfig {
            base: BUNDLE.into(),
            variant: ModelVariant::TinyKrea2,
        },
        lora: LoraConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            targets: vec![TargetSpec {
                pattern: r"blocks\.".into(),
                rank: None,
                alpha: None,
            }],
        },
        dataset: DatasetConfig {
            path: dataset,
            resolution: 32,
            batch_size: 2,
        },
        optim: OptimConfig {
            lr: 0.01,
            weight_decay: 0.0,
        },
        output: OutputConfig {
            dir: out.0.join("out"),
            name: "krea2-lora".into(),
            checkpoint_every: 5,
            sample_every: 0,
        },
        compute: ComputeConfig::default(),
        flow: FlowConfig::default(),
    }
}

#[test]
fn tiny_krea2_lora_trains_end_to_end_and_exports_kohya() {
    let out = TempDir::new("diffusion-e2e");
    let dataset = staged_dataset(&out);

    let mut started = None;
    let mut losses = Vec::new();
    let mut checkpoints = Vec::new();
    let mut finished = None;
    let adapter = DiffusionTrainer
        .train(&config(&out, dataset.clone()), &mut |event| match event {
            TrainEvent::Started { total_steps } => started = Some(total_steps),
            TrainEvent::Step { loss, .. } => losses.push(loss),
            TrainEvent::Checkpoint { step, path } => checkpoints.push((step, path)),
            TrainEvent::Finished { adapter_path } => finished = Some(adapter_path),
            _ => {}
        })
        .expect("the end-to-end tiny Krea 2 LoRA run completes");

    // The Trainer contract held: events framed the run.
    assert_eq!(started, Some(STEPS));
    assert_eq!(losses.len(), STEPS as usize, "one Step per step");
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "non-finite loss: {losses:?}"
    );
    // Deliberately NO loss-decrease assertion: each step draws fresh (t, ε),
    // so the per-step loss is dominated by the objective's irreducible noise
    // variance (v = ε − x₀ is mostly unpredictable at this scale) and is
    // non-monotone by construction — observed on this fixture: ~1.16 ± noise
    // either direction over dozens of steps. "Training happened" is asserted
    // deterministically below instead: the optimizer moved `B` off zero, and
    // the reseeded rerun is bit-identical.

    // The final artifact: the ComfyUI-loadable Krea2Diffusers export —
    // diffusers-style base names (verified against comfy/lora.py +
    // krea2_to_diffusers) with kohya suffixes.
    assert_eq!(finished.as_deref(), Some(adapter.as_path()));
    let keys = kohya_keys(&adapter);
    // 7 sites × 2 blocks × 3 tensors (down/up/alpha).
    assert_eq!(keys.len(), 42, "unexpected export keys: {keys:?}");
    for expect in [
        "transformer_blocks.0.attn.to_q.lora_down.weight",
        "transformer_blocks.0.attn.to_q.lora_up.weight",
        "transformer_blocks.0.attn.to_q.alpha",
        "transformer_blocks.0.attn.to_out.0.lora_up.weight",
        "transformer_blocks.1.ff.down.lora_up.weight",
    ] {
        assert!(keys.contains(&expect.to_string()), "missing key {expect}");
    }

    // Mid-run checkpoints are the SAME kohya export, not just files that
    // exist: each must deserialize to the identical key layout (review — a
    // checkpoint that silently switched to a native/resume format would
    // otherwise pass).
    assert_eq!(
        checkpoints.len(),
        2,
        "steps 5 and 10 (12 is the final save)"
    );
    for (step, path) in &checkpoints {
        assert_eq!(
            kohya_keys(path),
            keys,
            "checkpoint at step {step} must be the same kohya export layout"
        );
    }

    // The configured rank/alpha actually reached the adapters (review: the
    // key count alone is rank/alpha-invariant): lora_down is [rank, d_in],
    // lora_up [d_out, rank] for the 64-feature tiny model at rank 4, and the
    // `.alpha` scalar recovers the configured alpha = 8.0.
    let bytes = std::fs::read(&adapter).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let down = st
        .tensor("transformer_blocks.0.attn.to_q.lora_down.weight")
        .expect("down tensor present");
    assert_eq!(down.shape(), &[4, 64][..], "lora_down is [rank, d_in]");
    let up = st
        .tensor("transformer_blocks.0.attn.to_q.lora_up.weight")
        .expect("up tensor present");
    assert_eq!(up.shape(), &[64, 4][..], "lora_up is [d_out, rank]");
    let alpha = st
        .tensor("transformer_blocks.0.attn.to_q.alpha")
        .expect("alpha scalar present");
    assert_eq!(alpha.shape(), &[1][..]);
    let alpha = f32::from_le_bytes(alpha.data()[..4].try_into().unwrap());
    assert_eq!(alpha, 8.0, "the configured lora.alpha must round-trip");

    // The adapter genuinely trained: zero-init `B` (lora_up) moved off zero.
    let sum: f32 = up
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).abs())
        .sum();
    assert!(sum > 0.0, "lora_up must have moved off its zero init");

    // Warm-cache determinism: a second run re-reads the cached latents /
    // conditioning and, reseeded, reproduces the exact same loss stream —
    // the whole pipeline is reproducible. The cache snapshot proves the
    // warm run actually HIT the cache (review: identical losses alone
    // cannot distinguish a hit from a deterministic re-encode, so a
    // run-unstable fingerprint would otherwise go unnoticed): no cache file
    // may be added, removed, or rewritten by the second run.
    //
    // The rerun's bundle holds ONLY the MMDiT: with a warm cache the lazy
    // encode phase must never load the VAE / text encoder / tokenizer, and
    // the training phase reads the cache exclusively — so deleting them all
    // must not matter. (This pins the f32-encode/train split: the training
    // backend cannot quietly re-encode at its own precision.)
    let cache = cache_snapshot(&dataset);
    assert!(
        !cache.is_empty(),
        "the first run must have written the cache"
    );
    let stripped = out.0.join("bundle-stripped");
    std::fs::create_dir_all(&stripped).unwrap();
    std::fs::copy(
        Path::new(BUNDLE).join("raw.safetensors"),
        stripped.join("raw.safetensors"),
    )
    .unwrap();
    let mut config2 = config(&out, dataset.clone());
    config2.model.base = stripped.to_string_lossy().into_owned();
    // A fresh output dir: reusing run 1's would trigger the resume path
    // (exercised separately below) and break bit-identity.
    config2.output.dir = out.0.join("out2");
    let mut losses2 = Vec::new();
    DiffusionTrainer
        .train(&config2, &mut |event| {
            if let TrainEvent::Step { loss, .. } = event {
                losses2.push(loss);
            }
        })
        .expect("warm-cache rerun completes without any encoder files present");
    assert_eq!(losses, losses2, "reseeded rerun must be bit-identical");
    assert_eq!(
        cache_snapshot(&dataset),
        cache,
        "the rerun must hit the cache, not re-encode it"
    );

    // Resume: re-running against run 1's output dir loads the existing
    // adapter (announced via a Warning) and continues from it — the loss
    // stream must DIFFER from the fresh-start stream (the adapters no
    // longer begin at B = 0), and the export must stay loadable.
    let mut config3 = config(&out, dataset.clone());
    config3.model.base = stripped.to_string_lossy().into_owned();
    let mut resumed = false;
    let mut losses3 = Vec::new();
    let adapter3 = DiffusionTrainer
        .train(&config3, &mut |event| match event {
            TrainEvent::Warning { message } if message.contains("resuming") => resumed = true,
            TrainEvent::Step { loss, .. } => losses3.push(loss),
            _ => {}
        })
        .expect("the resume run completes");
    assert!(resumed, "the resume path must announce itself");
    assert_ne!(
        losses, losses3,
        "a resumed run continues from trained adapters, not from scratch"
    );
    assert_eq!(
        kohya_keys(&adapter3),
        keys,
        "resumed export layout unchanged"
    );
}

fn kohya_keys(path: &Path) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut keys: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();
    keys.sort();
    keys
}
