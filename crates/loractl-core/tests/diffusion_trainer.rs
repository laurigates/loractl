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

/// burn's backend RNG is process-global (`B::seed` swaps one shared seed),
/// so two trainings running in parallel interleave their draws and destroy
/// the reseeded determinism this file asserts. Every training test in this
/// binary serializes on this lock; a poisoned lock (a panicked sibling) is
/// safe to reuse — the guard only orders execution, it protects no data.
static TRAIN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
            checkpoint: None,
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
    let _rng = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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

/// M15 (#82): the `model.checkpoint` override routes the SAME e2e run
/// through the scaled-fp8 loader — `turbo_fp8.safetensors` is the fp8
/// quantization of the bundle's own seed-14 MMDiT weights, auto-detected
/// from the file header and dequantized at load. Two phases:
///
/// 1. A raw-checkpoint run populates the encoder cache.
/// 2. The fp8-override run (fresh output dir, same staged dataset) must
///    train end-to-end — Started/Step/Checkpoint/Finished, finite losses,
///    the same kohya export layout — while leaving the encoder cache
///    byte-untouched: the denoiser choice must not perturb the encode
///    phase (the fingerprint is encoder-derived, not denoiser-derived).
///
/// Shorter than the main e2e (6 steps): the full contract — checkpoints,
/// warm-cache determinism, resume — is pinned above; this test pins only
/// what the fp8 path adds.
#[test]
fn tiny_krea2_fp8_checkpoint_override_trains_e2e() {
    const FP8_STEPS: u64 = 6;
    let _rng = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("diffusion-fp8-e2e");
    let dataset = staged_dataset(&out);

    // Phase 1 — raw checkpoint, warming the encoder cache.
    let mut config_raw = config(&out, dataset.clone());
    config_raw.steps = FP8_STEPS;
    let mut losses_raw = Vec::new();
    DiffusionTrainer
        .train(&config_raw, &mut |event| {
            if let TrainEvent::Step { loss, .. } = event {
                losses_raw.push(loss);
            }
        })
        .expect("the raw-checkpoint run completes");
    let cache = cache_snapshot(&dataset);
    assert!(!cache.is_empty(), "the raw run must have written the cache");

    // Phase 2 — the fp8 override, against the already-warm cache.
    let mut config_fp8 = config(&out, dataset.clone());
    config_fp8.steps = FP8_STEPS;
    config_fp8.model.checkpoint = Some("turbo_fp8.safetensors".into());
    config_fp8.output.dir = out.0.join("out-fp8");
    let mut started = None;
    let mut losses = Vec::new();
    let mut checkpoints = Vec::new();
    let mut finished = None;
    let adapter = DiffusionTrainer
        .train(&config_fp8, &mut |event| match event {
            TrainEvent::Started { total_steps } => started = Some(total_steps),
            TrainEvent::Step { loss, .. } => losses.push(loss),
            TrainEvent::Checkpoint { step, path } => checkpoints.push((step, path)),
            TrainEvent::Finished { adapter_path } => finished = Some(adapter_path),
            _ => {}
        })
        .expect("the fp8-checkpoint run completes");

    // The Trainer contract held through the fp8 load path.
    assert_eq!(started, Some(FP8_STEPS));
    assert_eq!(losses.len(), FP8_STEPS as usize, "one Step per step");
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "non-finite loss: {losses:?}"
    );
    assert_eq!(
        checkpoints.len(),
        1,
        "step 5 (6 is the final save): {checkpoints:?}"
    );

    // The override actually took: fp8 quantization perturbs the weights, so
    // the reseeded loss stream must DIFFER from the raw run's — if the
    // checkpoint name were silently ignored (raw.safetensors loaded again),
    // the warm-cache determinism pinned above would make the two streams
    // bit-identical.
    assert_ne!(
        losses_raw, losses,
        "the fp8 checkpoint must load different (quantized) weights"
    );

    // The kohya export is the same ComfyUI-loadable layout as the raw run's.
    assert_eq!(finished.as_deref(), Some(adapter.as_path()));
    let keys = kohya_keys(&adapter);
    // 7 sites × 2 blocks × 3 tensors (down/up/alpha).
    assert_eq!(keys.len(), 42, "unexpected export keys: {keys:?}");
    for expect in [
        "transformer_blocks.0.attn.to_q.lora_down.weight",
        "transformer_blocks.0.attn.to_q.alpha",
        "transformer_blocks.1.ff.down.lora_up.weight",
    ] {
        assert!(keys.contains(&expect.to_string()), "missing key {expect}");
    }

    // The denoiser choice must not touch the encoder cache: no file added,
    // removed, or rewritten by the fp8 run.
    assert_eq!(
        cache_snapshot(&dataset),
        cache,
        "the fp8 run must reuse the raw run's encoder cache untouched"
    );
}

fn kohya_keys(path: &Path) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut keys: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();
    keys.sort();
    keys
}

// ---------------------------------------------------------------------------
// cuda backend (the guard is offline; the e2e is double-gated, box-only)
// ---------------------------------------------------------------------------

/// Selecting cuda in a binary built without the feature must bail with the
/// actionable not-built message (same convention `backend_dispatch.rs` pins
/// for the synthetic path), never the old "cuda isn't wired" catch-all.
#[cfg(not(feature = "cuda"))]
#[test]
fn diffusion_cuda_without_the_feature_names_the_fix() {
    use loractl_core::config::BackendKind;

    let out = TempDir::new("diffusion-cuda-unbuilt");
    let mut config = config(&out, PathBuf::from("unused-dataset"));
    config.compute.backend = BackendKind::Cuda;

    let err = DiffusionTrainer
        .train(&config, &mut |_event| {})
        .expect_err("cuda without the feature must refuse");
    let message = format!("{err:#}");
    assert!(
        message.contains("--features cuda"),
        "the error must name the rebuild fix, got: {message}"
    );
}

/// cuda is wired f32-only: f16 autodiff produces exactly-zero adapter
/// gradients on cuda (tracel-ai/burn#5162, validated on the RTX 4090), so
/// the guard must fail loudly before any GPU work. Cheap (bails pre-encode),
/// but compiled only under the cuda feature — runs via `just test-cuda`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "compiled only with the cuda feature; run via `just test-cuda`"]
fn diffusion_cuda_f16_bails_loudly() {
    use loractl_core::config::{BackendKind, Precision};

    let out = TempDir::new("diffusion-cuda-f16");
    let mut config = config(&out, PathBuf::from("unused-dataset"));
    config.compute.backend = BackendKind::Cuda;
    config.compute.precision = Precision::F16;

    let err = DiffusionTrainer
        .train(&config, &mut |_event| {})
        .expect_err("cuda f16 must refuse — burn#5162");
    let message = format!("{err:#}");
    assert!(
        message.contains("5162") && message.contains("f32"),
        "the error must cite the upstream defect and the fix, got: {message}"
    );
}

/// The tiny-krea2 e2e on real cuda hardware (M14 dispatch, cuda arm): the
/// whole diffusion stack trains through `DiffusionTrainer` on the GPU with
/// finite losses and exports the exact kohya layout. Portability asserts
/// only — GPU float-reduction order differs from ndarray (ADR-0001), so no
/// bit-identity and no loss-decrease bound (12 steps on random tiny weights).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an NVIDIA GPU (CUDA toolkit at build time); run via `just test-cuda`"]
fn tiny_krea2_cuda_f32_trains_e2e() {
    use loractl_core::config::BackendKind;

    let _rng = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("diffusion-cuda-e2e");
    let dataset = staged_dataset(&out);
    let mut config = config(&out, dataset);
    config.compute.backend = BackendKind::Cuda;

    let mut losses = Vec::new();
    let mut finished_path = None;
    let adapter = DiffusionTrainer
        .train(&config, &mut |event| match event {
            TrainEvent::Step { loss, .. } => losses.push(loss),
            TrainEvent::Finished { adapter_path } => finished_path = Some(adapter_path),
            _ => {}
        })
        .expect("cuda diffusion training should complete end-to-end");

    assert_eq!(losses.len() as u64, STEPS, "one Step event per step");
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "non-finite loss on cuda: {losses:?}"
    );

    // The exported adapter carries the exact kohya layout the offline e2e
    // pins: 7 sites × 2 blocks × 3 tensors = 42 keys.
    let adapter = finished_path.unwrap_or(adapter);
    let keys = kohya_keys(&adapter);
    assert_eq!(keys.len(), 42, "kohya export must carry 42 keys: {keys:?}");
    assert!(
        keys.iter()
            .any(|k| k == "transformer_blocks.0.attn.to_q.lora_down.weight"),
        "kohya naming must match the offline e2e's layout, got: {keys:?}"
    );
}
