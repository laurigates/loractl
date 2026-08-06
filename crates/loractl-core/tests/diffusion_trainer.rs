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
    BucketMode, DatasetConfig, LoraConfig, ModelConfig, ModelVariant, OptimConfig, OutputConfig,
    TargetSpec, TaskKind,
};
use loractl_core::{DiffusionTrainer, PhaseName, TrainConfig, TrainEvent, Trainer, read_metadata};
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
            denoiser: None,
            text_encoder: None,
            vae: None,
            tokenizer: None,
            training_adapter: None,
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
            no_upscale: false,
            bucketing: BucketMode::Aspects,
            min_bucket_resolution: None,
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
        ..Default::default()
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

    // The `__metadata__` wiring seam (#154). `build_metadata` is tested as a
    // pure function in tests/adapter_metadata.rs; what is only observable
    // HERE is that the trainer actually calls it, with facts drawn from the
    // real run — a mis-wired `Some(&metadata_for(..))` still compiles and
    // still exports a loadable file.
    let final_meta = read_metadata(&adapter).expect("the final export carries a header");
    assert_eq!(
        final_meta.get("ss_steps"),
        Some(STEPS.to_string().as_str()),
        "the final export records the completed run"
    );
    assert!(
        final_meta.get("ss_training_finished_at").is_some(),
        "only the final export claims a finish time"
    );
    // Facts that can only come from the live run, not from the config:
    // the dataset scan, the batch count, and the loaded checkpoint's name.
    // 5 = the 4 checked-in fixture images + the portrait `staged_dataset`
    // synthesizes, which is also why two buckets are populated below.
    assert_eq!(final_meta.get("ss_num_train_images"), Some("5"));
    assert_eq!(final_meta.get("ss_sd_model_name"), Some("raw.safetensors"));
    assert_eq!(final_meta.get("ss_network_dim"), Some("4"));
    let tags: serde_json::Value =
        serde_json::from_str(final_meta.get("ss_tag_frequency").expect("tag frequency"))
            .expect("ss_tag_frequency is JSON");
    let subset = tags
        .as_object()
        .and_then(|o| o.values().next())
        .and_then(|v| v.as_object())
        .expect("one subset of tags");
    assert_eq!(
        subset.len(),
        5,
        "one tag per staged caption, all distinct: {subset:?} — derived from \
         the scan, not from the config"
    );
    // Bucket info comes from the live assignment, not the configured
    // resolution: the square fixtures and the 32x48 portrait land in
    // different buckets.
    let buckets: serde_json::Value =
        serde_json::from_str(final_meta.get("ss_bucket_info").expect("bucket info"))
            .expect("ss_bucket_info is JSON");
    let populated = buckets["buckets"].as_object().expect("a bucket map").len();
    assert_eq!(
        populated, 2,
        "the portrait must be bucketed apart from the square images: {buckets}"
    );

    // Each checkpoint records ITS OWN step, and does not claim a finish time.
    for (step, path) in &checkpoints {
        let meta = read_metadata(path).expect("checkpoints carry a header too");
        assert_eq!(
            meta.get("ss_steps"),
            Some(step.to_string().as_str()),
            "checkpoint at step {step} must record its own progress"
        );
        assert_eq!(
            meta.get("ss_max_train_steps"),
            Some(STEPS.to_string().as_str())
        );
        assert!(
            meta.get("ss_training_finished_at").is_none(),
            "a mid-run checkpoint must not claim the run finished"
        );
    }

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

/// #84: `flow.shift_mode: resolution` (the ai-toolkit Krea 2 parity mode —
/// per-batch shift `exp(μ(gh·gw))`) is actually WIRED into the training
/// loop. A completes-with-finite-losses smoke cannot see the wiring (the
/// constant fallback also completes — verified by mutation), so this is a
/// differential kill-test in the `weight_decay_changes_the_loss_trajectory`
/// mold: two reseeded runs, identical except for `shift_mode`, must produce
/// DIFFERENT loss trajectories. The tiny buckets carry 6 image tokens, so
/// resolution mode resolves to `exp(μ(6)) ≈ 1.606` while constant mode uses
/// `shift = 3.0` — same seed, same RNG draw order, different `t` mapping. If
/// the per-batch `resolve_shift` call is ever dropped, both runs become
/// bit-identical and the inequality fails. (The resolved-shift *values* are
/// golden-pinned in `flow_reference.rs`; the run contract is pinned by the
/// main e2e above.)
#[test]
fn resolution_shift_mode_changes_the_loss_trajectory() {
    let _rng = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("diffusion-res-shift");
    let dataset = staged_dataset(&out);

    let run = |mode: loractl_core::config::ShiftMode, tag: &str| {
        let mut cfg = config(&out, dataset.clone());
        cfg.steps = 6;
        cfg.output.checkpoint_every = 10_000; // final save only
        cfg.output.dir = out.0.join(tag); // fresh dir: no resume cross-talk
        cfg.flow.shift_mode = mode;

        let mut losses = Vec::new();
        let adapter = DiffusionTrainer
            .train(&cfg, &mut |event| {
                if let TrainEvent::Step { loss, .. } = event {
                    losses.push(loss);
                }
            })
            .unwrap_or_else(|e| panic!("the {tag} run completes: {e:#}"));
        assert_eq!(losses.len(), 6, "{tag}: one Step per step");
        assert!(
            losses.iter().all(|l| l.is_finite()),
            "{tag}: non-finite loss: {losses:?}"
        );
        assert_eq!(kohya_keys(&adapter).len(), 42, "{tag}: kohya export layout");
        losses
    };

    let constant = run(loractl_core::config::ShiftMode::Constant, "constant");
    // The second run reuses the first run's encoder cache — the shift mode
    // must not perturb the cached latents/conditioning, only the sampled t.
    let resolution = run(loractl_core::config::ShiftMode::Resolution, "resolution");

    assert_ne!(
        constant, resolution,
        "shift_mode: resolution must change the sampled timesteps (and so the \
         loss trajectory) vs the constant shift under the same seed — identical \
         trajectories mean the per-batch resolve_shift is not wired into the \
         training loop"
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

/// #134: block checkpointing must reject LoRA dropout loudly — the replayed
/// backward would redraw masks the stored (plain-backend, dropout-identity)
/// forward never saw, silently corrupting the adapter gradients.
#[test]
fn block_checkpointing_rejects_dropout() {
    let out = TempDir::new("blockckpt-guard");
    let mut config = config(&out, PathBuf::from("unused-dataset"));
    config.compute.grad_checkpointing = true;
    config.lora.dropout = 0.1;
    let err = DiffusionTrainer
        .train(&config, &mut |_| {})
        .expect_err("dropout + block checkpointing must be rejected before any work");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("dropout"),
        "the error must name the conflicting knob: {msg}"
    );
}

/// #134 e2e: the block-checkpointed step trains the tiny fixture end to end,
/// emits its warning, and — because the per-step gradients are bit-identical
/// to the monolithic path's (tests/block_ckpt.rs) and both runs share the
/// seed and RNG stream — reproduces the monolithic loss trajectory.
#[test]
fn tiny_krea2_trains_with_block_checkpointing() {
    let _rng = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // Two runs in separate dirs (a shared output dir would make run 2 resume
    // from run 1's exported adapter).
    let run = |ckpt: bool, tag: &str| {
        let out = TempDir::new(tag);
        let dataset = staged_dataset(&out);
        let mut config = config(&out, dataset);
        config.compute.grad_checkpointing = ckpt;
        let mut losses = Vec::new();
        let mut warnings = Vec::new();
        DiffusionTrainer
            .train(&config, &mut |event| match event {
                TrainEvent::Step { loss, .. } => losses.push(loss),
                TrainEvent::Warning { message } => warnings.push(message),
                _ => {}
            })
            .expect("the block-checkpointed tiny run completes");
        (losses, warnings)
    };
    let (losses_off, _) = run(false, "blockckpt-off");
    let (losses_on, warnings) = run(true, "blockckpt-on");

    assert!(
        warnings
            .iter()
            .any(|w| w.contains("block-level gradient checkpointing")),
        "the knob must announce the block-checkpointed path: {warnings:?}"
    );
    assert_eq!(losses_on.len(), losses_off.len(), "one Step per step");
    for (i, (on, off)) in losses_on.iter().zip(&losses_off).enumerate() {
        assert!(on.is_finite() && off.is_finite(), "non-finite loss at {i}");
        let rel = (on - off).abs() / off.abs().max(1e-12);
        assert!(
            rel < 1e-5,
            "step {i}: checkpointed loss {on} vs monolithic {off} (rel {rel})"
        );
    }
}

/// The setup phases are actually reported (`TrainEvent::Phase`).
///
/// This is the deterministic proof for the thing that made a real run look
/// hung: everything before step 1 — the one-time dataset encode (minutes per
/// sample on the real 4B text encoder), the multi-gigabyte checkpoint loads,
/// the cache re-read, LoRA injection — used to emit nothing at all. Asserting
/// only "some Phase arrived" would pass on a single stray event, so this pins
/// the vocabulary, the counters, the ordering, and the cold/warm distinction.
#[test]
fn setup_phases_are_reported_before_the_first_step() {
    let _rng = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("diffusion-phases");
    let dataset = staged_dataset(&out);

    // (order, name, detail, done, total) — `order` is the index in the whole
    // event stream, so "before the first Step" is checkable.
    type Report = (usize, PhaseName, String, Option<u64>, Option<u64>);
    let mut phases: Vec<Report> = Vec::new();
    let mut first_step: Option<usize> = None;
    let mut seen = 0usize;
    DiffusionTrainer
        .train(&config(&out, dataset.clone()), &mut |event| {
            let idx = seen;
            seen += 1;
            match event {
                TrainEvent::Phase {
                    name,
                    detail,
                    counters,
                } => phases.push((
                    idx,
                    name,
                    detail,
                    counters.map(|c| c.done),
                    counters.map(|c| c.total),
                )),
                TrainEvent::Step { .. } => first_step = first_step.or(Some(idx)),
                _ => {}
            }
        })
        .expect("the cold run completes");

    let names: Vec<&str> = phases.iter().map(|p| p.1.as_str()).collect();
    for expected in ["encode", "load", "dataset", "inject"] {
        assert!(
            names.contains(&expected),
            "no `{expected}` phase in {names:?}"
        );
    }
    // The vocabulary is closed: a typo'd or ad-hoc phase name must fail here
    // rather than reach consumers keying on it.
    for (_, name, ..) in &phases {
        assert!(
            matches!(
                name.as_str(),
                "encode" | "dataset" | "load" | "quantize" | "merge" | "inject"
            ),
            "phase name outside the documented vocabulary: {name}"
        );
    }

    // Every phase precedes the first optimization step — these report SETUP,
    // and a Phase leaking into the step loop would be a different (and
    // bench-contaminating) thing.
    let first_step = first_step.expect("the run stepped");
    assert!(
        phases.iter().all(|p| p.0 < first_step),
        "a Phase arrived at or after the first Step: {phases:?}"
    );

    // The encode phase counts entries: 5 staged examples, `done` strictly
    // increasing from 0, `total` always 5, and each one named as an encode
    // (cold cache) rather than a cache hit.
    let encodes: Vec<_> = phases
        .iter()
        .filter(|p| p.1 == PhaseName::Encode && p.3.is_some())
        .collect();
    assert_eq!(
        encodes.len(),
        6,
        "5 per-entry reports + the closing summary: {encodes:?}"
    );
    for (i, p) in encodes.iter().enumerate() {
        assert_eq!(p.4, Some(5), "total must be the entry count");
        assert_eq!(p.3, Some(i.min(5) as u64), "done must count up from 0");
    }
    assert!(
        encodes[..5].iter().all(|p| p.2.starts_with("encoding ")),
        "a cold pass must report encodes, not cache hits: {encodes:?}"
    );

    // The checkpoint loads name what they are loading, and the injection
    // reports the site count the run actually adapted (7 sites × 2 blocks).
    let loads: Vec<&str> = phases
        .iter()
        .filter(|p| p.1 == PhaseName::Load)
        .map(|p| p.2.as_str())
        .collect();
    for what in ["VAE", "text encoder", "MMDiT"] {
        assert!(
            loads.iter().any(|d| d.starts_with(what)),
            "no `{what}` load phase in {loads:?}"
        );
    }
    let inject = phases
        .iter()
        .find(|p| p.1 == PhaseName::Inject)
        .expect("checked present above");
    assert!(
        inject.2.starts_with("14 LoRA adapters"),
        "inject must report the matched adapter count: {}",
        inject.2
    );

    // A warm rerun flips the encode reports to cache hits — the flag tracks
    // the cache rather than being hardcoded, and a run whose per-file encode
    // reports say "encoding" on a warm cache is reporting a lie.
    let mut config2 = config(&out, dataset);
    config2.output.dir = out.0.join("out2");
    let mut warm: Vec<String> = Vec::new();
    DiffusionTrainer
        .train(&config2, &mut |event| {
            if let TrainEvent::Phase {
                name,
                detail,
                counters,
            } = event
                && name == PhaseName::Encode
                && counters.is_some()
            {
                warm.push(detail);
            }
        })
        .expect("the warm rerun completes");
    assert!(
        warm[..5].iter().all(|d| d.starts_with("cached ")),
        "a warm pass must report cache hits: {warm:?}"
    );
}

/// A one-site LoRA training adapter (#83) written to `path`: diffusers
/// `lora_A`/`lora_B` naming, `diffusion_model.`-prefixed (as ostris/ComfyUI
/// ship them), with a per-site `.alpha` so nothing merges at fallback
/// scaling. Targets `blocks.0.attn.wq`, a `[64, 64]` base linear in the
/// tiny-Krea-2 architecture (`MmditConfig::tiny_krea2().features == 64`).
///
/// Generated here rather than checked in: the merge phase only fires when
/// `model.training_adapter` is set, and this test must stay offline and
/// fixture-free.
fn write_training_adapter(path: &Path) {
    const RANK: usize = 2;
    const FEATURES: usize = 64;
    // Small deterministic ramps: the merge only has to happen, not to move
    // the loss anywhere in particular.
    let ramp =
        |n: usize, base: f32| -> Vec<f32> { (0..n).map(|i| base + (i as f32) * 0.001).collect() };
    let entries: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
        (
            "diffusion_model.blocks.0.attn.wq.lora_A.weight".into(),
            vec![RANK, FEATURES],
            ramp(RANK * FEATURES, -0.01),
        ),
        (
            "diffusion_model.blocks.0.attn.wq.lora_B.weight".into(),
            vec![FEATURES, RANK],
            ramp(FEATURES * RANK, 0.01),
        ),
        (
            "diffusion_model.blocks.0.attn.wq.alpha".into(),
            vec![1],
            vec![RANK as f32],
        ),
    ];
    let bufs: Vec<(String, Vec<usize>, Vec<u8>)> = entries
        .into_iter()
        .map(|(k, shape, vals)| {
            (
                k,
                shape,
                vals.iter().flat_map(|f| f.to_le_bytes()).collect(),
            )
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView)> = bufs
        .iter()
        .map(|(k, shape, bytes)| {
            (
                k.clone(),
                safetensors::tensor::TensorView::new(
                    safetensors::tensor::Dtype::F32,
                    shape.clone(),
                    bytes,
                )
                .unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, None, path).unwrap();
}

/// The `merge` phase (#83's merge-at-load) is pinned by EXECUTION, not by
/// reading the source (#170).
///
/// `setup_phases_are_reported_before_the_first_step` above cannot reach it:
/// its config leaves `model.training_adapter` unset, so the emission never
/// runs — and `tests/training_adapter.rs` calls `merge_training_adapter`
/// directly, never through `.train()`. That left the one emission deletable
/// with the whole suite staying green, which is the exact failure shape #165
/// already shipped once (a `quantize` phase that never emitted its terminal
/// snapshot).
///
/// So: drive a real (1-step) run with a generated training adapter and pin
/// both that the phase arrives and WHERE it arrives — before LoRA injection
/// (it folds into the frozen base, which injection then wraps) and therefore
/// before the first `Step`.
///
/// The MNIST `dataset` half of #170 stays uncovered by design: that emission
/// is behind `--features mnist`, whose loader downloads the dataset, so it
/// can never join the default offline suite.
#[test]
fn merge_phase_is_emitted_before_injection_and_the_first_step() {
    let _rng = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("diffusion-merge-phase");
    let dataset = staged_dataset(&out);
    let adapter_path = out.0.join("assistant.safetensors");
    write_training_adapter(&adapter_path);

    let mut cfg = config(&out, dataset);
    cfg.steps = 1; // the phase is setup-time; one step is enough to have a "first Step"
    cfg.model.training_adapter = Some(adapter_path.clone());

    // Index in the whole event stream, so "before" is checkable.
    let mut merge: Option<(usize, String)> = None;
    let mut inject: Option<usize> = None;
    let mut first_step: Option<usize> = None;
    let mut seen = 0usize;
    DiffusionTrainer
        .train(&cfg, &mut |event| {
            let idx = seen;
            seen += 1;
            match event {
                TrainEvent::Phase {
                    name: PhaseName::Merge,
                    detail,
                    ..
                } => {
                    if merge.is_none() {
                        merge = Some((idx, detail));
                    }
                }
                TrainEvent::Phase {
                    name: PhaseName::Inject,
                    ..
                } => inject = inject.or(Some(idx)),
                TrainEvent::Step { .. } => first_step = first_step.or(Some(idx)),
                _ => {}
            }
        })
        .expect("the training-adapter run completes");

    let (merge_idx, merge_detail) =
        merge.expect("a `merge` phase must be emitted when model.training_adapter is set");
    let inject_idx = inject.expect("the run injected adapters");
    let first_step = first_step.expect("the run stepped");

    assert!(
        merge_idx < inject_idx,
        "merge folds into the frozen base BEFORE LoRA injection wraps it \
         (merge at {merge_idx}, inject at {inject_idx})"
    );
    assert!(
        merge_idx < first_step,
        "the merge phase must precede the first Step (merge at {merge_idx}, \
         first Step at {first_step})"
    );
    // The detail names what is being merged — a phase that fired with the
    // wrong (or an empty) subject would pass a bare presence check.
    assert!(
        merge_detail.starts_with("training adapter")
            && merge_detail.ends_with("from assistant.safetensors"),
        "merge must report the adapter it folds in: {merge_detail}"
    );
}

/// #175 **at the call site**: the trainer reads a batch's cache files *inside*
/// the step loop, not once up front.
///
/// `tests/dataset_residency.rs` proves `PreparedDataset` cannot hold the
/// dataset — it is a plan with no `B: Backend` parameter, and `load_batch`
/// costs one batch. None of that constrains the **caller**. Hoisting
/// `let batches: Vec<_> = plans.iter().map(|p| prepared.load_batch(p, &device))
/// .collect::<Result<_>>()?;` above `for step in 1..=total` compiles, keeps
/// the type non-generic, yields bit-identical losses, and restores exactly
/// the O(dataset) residency #175 removed — with every other test green.
///
/// So this pins the observable consequence of loading per step: the cache
/// files have to still be there. One full epoch in, every batch has been
/// visited once, and then the latents are deleted. A trainer that reads per
/// step fails on the next one, loudly, naming the file and the step. A
/// trainer that hoisted them — or that memoized them in process, the same
/// regression wearing a cache — finishes the run and fails this test.
#[test]
fn the_trainer_reads_the_cache_inside_the_step_loop() {
    let _rng = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("diffusion-per-step-load");
    let dataset = staged_dataset(&out);

    // Four landscapes + one portrait at batch_size 2 = 3 batches per epoch,
    // and STEPS is 12, so there are several epochs to work with. Asserted
    // below rather than assumed: if the fixture ever changes shape, this test
    // must fail rather than quietly delete at the wrong moment.
    const BATCHES_PER_EPOCH: usize = 3;

    let mut steps_seen = 0usize;
    let mut epoch_reported: Option<String> = None;
    let result = DiffusionTrainer.train(&config(&out, dataset.clone()), &mut |event| match event {
        TrainEvent::Phase {
            name: PhaseName::Dataset,
            detail,
            ..
        } => {
            epoch_reported = Some(detail);
        }
        TrainEvent::Step { .. } => {
            steps_seen += 1;
            if steps_seen == BATCHES_PER_EPOCH + 1 {
                // A full epoch has been consumed, so an in-process cache
                // would be warm for every plan by now — and the batch for
                // the NEXT step is one this run has already loaded once.
                let cache = dataset.join(".loractl-cache");
                let mut deleted = 0usize;
                for entry in std::fs::read_dir(&cache).expect("cache dir") {
                    let path = entry.unwrap().path();
                    if path.to_string_lossy().ends_with(".latent.safetensors") {
                        std::fs::remove_file(&path).unwrap();
                        deleted += 1;
                    }
                }
                assert!(
                    deleted >= 5,
                    "expected the latents to delete, got {deleted}"
                );
            }
        }
        _ => {}
    });

    assert!(
        epoch_reported
            .as_deref()
            .is_some_and(|d| d.contains(&format!("{BATCHES_PER_EPOCH} batches per epoch"))),
        "the fixture no longer produces {BATCHES_PER_EPOCH} batches per epoch, so the \
         deletion point above is wrong: {epoch_reported:?}"
    );

    let err = format!(
        "{:#}",
        result.expect_err(
            "deleting the latents mid-run must fail the NEXT step — a run that finished \
             was not reading the cache per step"
        )
    );
    assert!(
        err.contains("disappeared mid-run") && err.contains(".latent.safetensors"),
        "the failure must name the vanished file: {err}"
    );
    assert!(
        err.contains(&format!(
            "loading the batch for step {}",
            BATCHES_PER_EPOCH + 2
        )),
        "the failure must name the step that tried to load: {err}"
    );
    // …and it got that far, so the steps before the deletion ran normally.
    assert_eq!(
        steps_seen,
        BATCHES_PER_EPOCH + 1,
        "steps before the failure"
    );
}
