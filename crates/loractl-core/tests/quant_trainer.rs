//! PR-B3 (#96): int8 frozen-base quantization wired through the diffusion
//! trainer, fully offline (ndarray).
//!
//! Three things are pinned here:
//!
//! 1. **The guard matrix** — `compute.quant: int8` is legal only on the two
//!    numerically-clean f32 paths (`ndarray` for CI, `cuda` for the real run);
//!    every other backend/precision combination bails with an actionable
//!    message, and the synthetic `BurnTrainer` rejects the knob outright.
//! 2. **The streaming quantized loader, end to end** — the tiny-krea2 bundle
//!    trains through `DiffusionTrainer` with `quant: int8`: the loader
//!    quantizes every block-aligned base site (52 of 53 on tiny-krea2, leaving
//!    only the unaligned `tmlp.fc1`), the forward runs (a wrong weight
//!    orientation would shape-mismatch on the non-square projections), losses
//!    are finite, the kohya export is the same 42-key layout, and the resume
//!    path works. This proves the loader yields a **correct trainable**
//!    quantized model — it does NOT prove on-box memory (that is PR-B4).
//! 3. **The fp8 → int8 requant path** — the M15 scaled-fp8 tiny checkpoint
//!    loaded with `quant: int8` trains e2e, exercising `load_quant_module`'s
//!    fp8-snapshot branch (fp8 → f32 → int8).

use loractl_core::config::{
    ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, ModelVariant, OptimConfig,
    OutputConfig, TargetSpec, TaskKind,
};
use loractl_core::{
    BackendKind, BurnTrainer, DiffusionTrainer, Precision, Quant, TrainConfig, TrainEvent, Trainer,
};
use std::path::{Path, PathBuf};

const BUNDLE: &str = "tests/fixtures/tiny-krea2";
const DATASET: &str = "tests/fixtures/dataset-tiny";

/// burn's backend RNG is process-global, so training tests in this binary
/// serialize on this lock (see `tests/diffusion_trainer.rs`). A poisoned lock
/// is safe to reuse — it only orders execution.
static TRAIN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A unique temp dir, removed on drop.
struct TempDir(PathBuf);

/// Per-process monotonic counter making every `TempDir` path unique even when
/// two threads construct one at the same nanosecond. The guard-matrix tests run
/// fully parallel (no `TRAIN_LOCK`), so a `pid+nanos`-only name collided under
/// macOS's coarse clock — two tests then shared an `out/` dir and raced in
/// `create_dir_all`, masking the guard message. The counter removes the race.
static TEMPDIR_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = TEMPDIR_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "loractl-{tag}-{}-{nanos}-{seq}",
            std::process::id()
        ));
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
/// repo tree (mirrors `tests/diffusion_trainer.rs`).
fn staged_dataset(out: &TempDir) -> PathBuf {
    let dst = out.0.join("dataset");
    std::fs::create_dir_all(&dst).unwrap();
    for entry in std::fs::read_dir(DATASET).expect("checked-in dataset present") {
        let path = entry.unwrap().path();
        if path.is_file() {
            std::fs::copy(&path, dst.join(path.file_name().unwrap())).unwrap();
        }
    }
    dst
}

/// The tiny-krea2 diffusion config (mirrors `tests/diffusion_trainer.rs`), with
/// `quant` left at its default `None` — callers flip the compute knobs.
fn config(out: &TempDir, dataset: PathBuf, steps: u64) -> TrainConfig {
    TrainConfig {
        steps,
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

fn kohya_keys(path: &Path) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut keys: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();
    keys.sort();
    keys
}

// ---------------------------------------------------------------------------
// 1. Guard matrix — every illegal combo bails with its actionable message.
//    These bail inside `DiffusionTrainer::train` BEFORE any encode/backend
//    work (a pure config check compiled on every backend), so they need no
//    dataset and run under the default (offline) features.
// ---------------------------------------------------------------------------

fn quant_bail(backend: BackendKind, precision: Precision, quant: Quant) -> String {
    let out = TempDir::new("quant-guard");
    let mut cfg = config(&out, PathBuf::from("unused-dataset"), 4);
    cfg.compute.backend = backend;
    cfg.compute.precision = precision;
    cfg.compute.quant = quant;
    let err = DiffusionTrainer
        .train(&cfg, &mut |_event| {})
        .expect_err("the illegal quant combo must refuse");
    format!("{err:#}")
}

#[test]
fn quant_int8_on_wgpu_is_rejected_as_untested() {
    let message = quant_bail(BackendKind::Wgpu, Precision::F32, Quant::Int8);
    assert!(
        message.contains("wgpu is untested") && message.contains("f16"),
        "wgpu+int8 must point at f16 for wgpu memory savings, got: {message}"
    );
}

#[test]
fn quant_int8_on_candle_is_rejected() {
    let message = quant_bail(BackendKind::Candle, Precision::F32, Quant::Int8);
    assert!(
        message.contains("not supported on the Candle") && message.contains("int8"),
        "candle+int8 must name the unsupported backend, got: {message}"
    );
}

#[test]
fn quant_int8_on_tch_is_rejected() {
    let message = quant_bail(BackendKind::Tch, Precision::F32, Quant::Int8);
    assert!(
        message.contains("not supported on the Tch"),
        "tch+int8 must name the unsupported backend, got: {message}"
    );
}

#[test]
fn quant_int8_with_non_f32_precision_is_rejected() {
    // ndarray passes the backend check; f16 then trips the precision guard.
    let message = quant_bail(BackendKind::Ndarray, Precision::F16, Quant::Int8);
    assert!(
        message.contains("dequantizes to f32") && message.contains("f32"),
        "int8+f16 must say quantization dequantizes to f32, got: {message}"
    );
}

// int4 gets the identical guard matrix as int8 (same dequant-to-f32 path,
// restricted to the two numerically-clean f32 backends).

#[test]
fn quant_int4_on_wgpu_is_rejected_as_untested() {
    let message = quant_bail(BackendKind::Wgpu, Precision::F32, Quant::Int4);
    assert!(
        message.contains("wgpu is untested") && message.contains("f16"),
        "wgpu+int4 must point at f16 for wgpu memory savings, got: {message}"
    );
}

#[test]
fn quant_int4_on_candle_is_rejected() {
    let message = quant_bail(BackendKind::Candle, Precision::F32, Quant::Int4);
    assert!(
        message.contains("not supported on the Candle") && message.contains("int4"),
        "candle+int4 must name the unsupported backend, got: {message}"
    );
}

#[test]
fn quant_int4_on_tch_is_rejected() {
    let message = quant_bail(BackendKind::Tch, Precision::F32, Quant::Int4);
    assert!(
        message.contains("not supported on the Tch"),
        "tch+int4 must name the unsupported backend, got: {message}"
    );
}

#[test]
fn quant_int4_with_non_f32_precision_is_rejected() {
    // ndarray passes the backend check; f16 then trips the precision guard.
    let message = quant_bail(BackendKind::Ndarray, Precision::F16, Quant::Int4);
    assert!(
        message.contains("dequantizes to f32") && message.contains("f32"),
        "int4+f16 must say quantization dequantizes to f32, got: {message}"
    );
}

/// #83: `model.training_adapter` needs a full-precision base — combining it with
/// `compute.quant` must bail up front (a quantized base stores no weight the
/// `W += (alpha/rank)·B·A` merge can fold into; silently loading an unmerged
/// base is exactly the failure this guard prevents). Reaches the guard on the
/// otherwise-legal ndarray/f32/int8 path, so no dataset is needed.
#[test]
fn quant_with_training_adapter_is_rejected() {
    let out = TempDir::new("quant-ta-guard");
    let mut cfg = config(&out, PathBuf::from("unused-dataset"), 4);
    cfg.compute.backend = BackendKind::Ndarray;
    cfg.compute.precision = Precision::F32;
    cfg.compute.quant = Quant::Int8;
    cfg.model.training_adapter = Some(PathBuf::from("some/assistant.safetensors"));
    let err = DiffusionTrainer
        .train(&cfg, &mut |_event| {})
        .expect_err("training_adapter + quant must refuse");
    let message = format!("{err:#}");
    assert!(
        message.contains("training_adapter") && message.contains("int8"),
        "must name training_adapter and the quant scheme, got: {message}"
    );
}

/// #128: the chunk knob defaults to 512 MiB — through `Default` AND through a
/// YAML that carries a `compute:` block without the field (the struct-level
/// `#[serde(default)]` constructs missing fields from `Self::default()`, which
/// is exactly why `ComputeConfig`'s `Default` is hand-written; a derived
/// all-zeros default would silently disable chunking). `0` must parse as the
/// explicit off switch.
#[test]
fn dequant_chunk_mib_defaults_to_512_and_zero_disables() {
    use figment::Figment;
    use figment::providers::{Format, Yaml};
    use loractl_core::config::DEFAULT_DEQUANT_CHUNK_MIB;

    assert_eq!(DEFAULT_DEQUANT_CHUNK_MIB, 512);
    assert_eq!(ComputeConfig::default().dequant_chunk_mib, 512);

    // A compute block WITHOUT the field → the default.
    let cfg: ComputeConfig = Figment::new()
        .merge(Yaml::string("backend: ndarray\nquant: int8\n"))
        .extract()
        .expect("compute block without dequant_chunk_mib must parse");
    assert_eq!(cfg.dequant_chunk_mib, 512);

    // The explicit off switch.
    let cfg: ComputeConfig = Figment::new()
        .merge(Yaml::string("dequant_chunk_mib: 0\n"))
        .extract()
        .expect("dequant_chunk_mib: 0 must parse");
    assert_eq!(cfg.dequant_chunk_mib, 0);

    // An explicit non-default value.
    let cfg: ComputeConfig = Figment::new()
        .merge(Yaml::string("dequant_chunk_mib: 16\n"))
        .extract()
        .expect("dequant_chunk_mib: 16 must parse");
    assert_eq!(cfg.dequant_chunk_mib, 16);
}

#[test]
fn burn_trainer_rejects_quant_knob() {
    // The synthetic/MNIST trainer has no frozen base worth quantizing.
    let out = TempDir::new("quant-burn-bail");
    let mut cfg = config(&out, PathBuf::from("unused-dataset"), 4);
    cfg.model.base = "synthetic".into();
    cfg.task = TaskKind::Classification;
    cfg.output.sample_every = 0;
    cfg.compute.quant = Quant::Int8;
    let err = BurnTrainer
        .train(&cfg, &mut |_event| {})
        .expect_err("the synthetic trainer must refuse the quant knob");
    let message = format!("{err:#}");
    assert!(
        message.contains("diffusion trainer's frozen base") && message.contains("compute.quant"),
        "BurnTrainer+int8 must name the diffusion trainer as the right place, got: {message}"
    );
}

// ---------------------------------------------------------------------------
// 2. ndarray int8 end-to-end — the streaming loader produces a correct,
//    trainable quantized model.
// ---------------------------------------------------------------------------

#[test]
fn tiny_krea2_int8_trains_end_to_end_and_exports_kohya() {
    const STEPS: u64 = 12;
    let _rng = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("quant-int8-e2e");
    let dataset = staged_dataset(&out);

    let mut cfg = config(&out, dataset.clone(), STEPS);
    cfg.compute.quant = Quant::Int8; // ndarray + f32 + int8 — the legal CI path

    let mut started = None;
    let mut losses = Vec::new();
    let mut checkpoints = Vec::new();
    let mut finished = None;
    let mut quant_accounting = None;
    let adapter = DiffusionTrainer
        .train(&cfg, &mut |event| match event {
            TrainEvent::Started { total_steps } => started = Some(total_steps),
            TrainEvent::Step { loss, .. } => losses.push(loss),
            TrainEvent::Checkpoint { step, path } => checkpoints.push((step, path)),
            TrainEvent::Finished { adapter_path } => finished = Some(adapter_path),
            TrainEvent::Warning { message } if message.contains("int8-quantized") => {
                quant_accounting = Some(message)
            }
            _ => {}
        })
        .expect("the ndarray int8 tiny Krea 2 run completes");

    // The Trainer contract held through the quantized load path.
    assert_eq!(started, Some(STEPS));
    assert_eq!(losses.len(), STEPS as usize, "one Step per step");
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "non-finite loss with int8 base: {losses:?}"
    );

    // Loud accounting proves the streaming loader quantized the RIGHT set:
    // tiny-krea2 has 53 base linears, all block-aligned except `tmlp.fc1`
    // (tdim = 16), so 52 quantize and 1 stays full-precision. A wrong
    // orientation would have shape-mismatched on a non-square projection long
    // before here (the run completing is itself the orientation proof).
    let accounting = quant_accounting.expect("the int8 loader must emit its accounting Warning");
    assert!(
        accounting.contains("int8-quantized 52 frozen-base linear sites")
            && accounting.contains("1 left full-precision"),
        "unexpected quant accounting: {accounting}"
    );

    // The final artifact is the same ComfyUI-loadable kohya export as the
    // full-precision run: 7 injectable sites × 2 trunk blocks × 3 tensors = 42.
    assert_eq!(finished.as_deref(), Some(adapter.as_path()));
    let keys = kohya_keys(&adapter);
    assert_eq!(keys.len(), 42, "unexpected export keys: {keys:?}");
    for expect in [
        "transformer_blocks.0.attn.to_q.lora_down.weight",
        "transformer_blocks.0.attn.to_q.lora_up.weight",
        "transformer_blocks.0.attn.to_q.alpha",
        "transformer_blocks.1.ff.down.lora_up.weight",
    ] {
        assert!(keys.contains(&expect.to_string()), "missing key {expect}");
    }

    // Mid-run checkpoints are the same kohya layout (steps 5, 10; 12 is final).
    assert_eq!(checkpoints.len(), 2, "checkpoints at steps 5 and 10");
    for (step, path) in &checkpoints {
        assert_eq!(
            kohya_keys(path),
            keys,
            "checkpoint at step {step} must be the same kohya layout"
        );
    }

    // The adapter genuinely trained on top of the frozen int8 base: zero-init
    // `B` (lora_up) moved off zero (gradients flow through the custom quant op
    // to the adapters, never to the frozen QFloat weight).
    let bytes = std::fs::read(&adapter).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let up = st
        .tensor("transformer_blocks.0.attn.to_q.lora_up.weight")
        .expect("up tensor present");
    let sum: f32 = up
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).abs())
        .sum();
    assert!(sum > 0.0, "lora_up must have moved off its zero init");

    // Resume: re-running against the same output dir loads the existing adapter
    // (announced via a Warning) and continues from it — the loss stream differs
    // from a fresh int8 start and the export layout is unchanged.
    let mut cfg_resume = config(&out, dataset, STEPS);
    cfg_resume.compute.quant = Quant::Int8;
    let mut resumed = false;
    let mut losses_resume = Vec::new();
    let adapter_resume = DiffusionTrainer
        .train(&cfg_resume, &mut |event| match event {
            TrainEvent::Warning { message } if message.contains("resuming") => resumed = true,
            TrainEvent::Step { loss, .. } => losses_resume.push(loss),
            _ => {}
        })
        .expect("the int8 resume run completes");
    assert!(resumed, "the int8 resume path must announce itself");
    assert_ne!(
        losses, losses_resume,
        "a resumed int8 run continues from trained adapters, not from scratch"
    );
    assert_eq!(
        kohya_keys(&adapter_resume),
        keys,
        "resumed int8 export layout unchanged"
    );
}

/// The int4 twin of the ndarray e2e: `quant: int4` streams the base as `Q4S`
/// through the SAME loader, the forward runs (a wrong orientation would
/// shape-mismatch on the non-square projections), losses are finite, and the
/// kohya export is the same 42-key layout. Proves int4 is a correct trainable
/// scheme parametrization of the int8 path — on-box memory fit is the separate
/// PR-B4 probe. Needs burn-ndarray's `export_tests` (the dev-dependency) so
/// ndarray can quantize `Q4S`.
#[test]
fn tiny_krea2_int4_trains_end_to_end_and_exports_kohya() {
    const STEPS: u64 = 12;
    let _rng = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("quant-int4-e2e");
    let dataset = staged_dataset(&out);

    let mut cfg = config(&out, dataset, STEPS);
    cfg.compute.quant = Quant::Int4; // ndarray + f32 + int4 — the legal CI path

    let mut losses = Vec::new();
    let mut checkpoints = Vec::new();
    let mut finished = None;
    let mut quant_accounting = None;
    let adapter = DiffusionTrainer
        .train(&cfg, &mut |event| match event {
            TrainEvent::Step { loss, .. } => losses.push(loss),
            TrainEvent::Checkpoint { step, path } => checkpoints.push((step, path)),
            TrainEvent::Finished { adapter_path } => finished = Some(adapter_path),
            TrainEvent::Warning { message } if message.contains("int4-quantized") => {
                quant_accounting = Some(message)
            }
            _ => {}
        })
        .expect("the ndarray int4 tiny Krea 2 run completes");

    assert_eq!(losses.len(), STEPS as usize, "one Step per step");
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "non-finite loss with int4 base: {losses:?}"
    );

    // Same 52/1 split as int8 — block alignment is scheme-independent — but the
    // accounting must name int4 (the scheme actually used).
    let accounting = quant_accounting.expect("the int4 loader must emit its accounting Warning");
    assert!(
        accounting.contains("int4-quantized 52 frozen-base linear sites")
            && accounting.contains("1 left full-precision"),
        "unexpected int4 quant accounting: {accounting}"
    );

    // Same ComfyUI-loadable kohya export as every other tiny-krea2 run (42 keys).
    let adapter = finished.unwrap_or(adapter);
    let keys = kohya_keys(&adapter);
    assert_eq!(keys.len(), 42, "unexpected int4 export keys: {keys:?}");
    assert_eq!(checkpoints.len(), 2, "checkpoints at steps 5 and 10");

    // The adapter genuinely trained on top of the frozen int4 base: zero-init
    // `B` (lora_up) moved off zero (gradients flow through the custom quant op).
    let bytes = std::fs::read(&adapter).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let up = st
        .tensor("transformer_blocks.0.attn.to_q.lora_up.weight")
        .expect("up tensor present");
    let sum: f32 = up
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).abs())
        .sum();
    assert!(
        sum > 0.0,
        "lora_up must have moved off its zero init (int4)"
    );
}

// ---------------------------------------------------------------------------
// 3. fp8 → int8 requant — the scaled-fp8 tiny checkpoint loaded with int8.
// ---------------------------------------------------------------------------

#[test]
fn tiny_krea2_fp8_checkpoint_int8_requant_trains_e2e() {
    const STEPS: u64 = 6;
    let _rng = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("quant-fp8-int8");
    let dataset = staged_dataset(&out);

    // The M15 fp8 tiny checkpoint, loaded and int8-quantized: exercises
    // `load_quant_module`'s fp8-snapshot branch (fp8 → f32 → int8).
    let mut cfg = config(&out, dataset, STEPS);
    cfg.model.checkpoint = Some("turbo_fp8.safetensors".into());
    cfg.compute.quant = Quant::Int8;

    let mut started = None;
    let mut losses = Vec::new();
    let mut finished = None;
    let mut quant_accounting = None;
    let adapter = DiffusionTrainer
        .train(&cfg, &mut |event| match event {
            TrainEvent::Started { total_steps } => started = Some(total_steps),
            TrainEvent::Step { loss, .. } => losses.push(loss),
            TrainEvent::Finished { adapter_path } => finished = Some(adapter_path),
            TrainEvent::Warning { message } if message.contains("int8-quantized") => {
                quant_accounting = Some(message)
            }
            _ => {}
        })
        .expect("the fp8→int8 run completes");

    assert_eq!(started, Some(STEPS));
    assert_eq!(losses.len(), STEPS as usize, "one Step per step");
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "non-finite loss with fp8→int8 base: {losses:?}"
    );
    let accounting =
        quant_accounting.expect("the fp8→int8 loader must emit its accounting Warning");
    assert!(
        accounting.contains("int8-quantized 52 frozen-base linear sites"),
        "unexpected fp8→int8 quant accounting: {accounting}"
    );

    // Same ComfyUI-loadable kohya layout as every other tiny-krea2 run.
    let adapter = finished.unwrap_or(adapter);
    let keys = kohya_keys(&adapter);
    assert_eq!(keys.len(), 42, "unexpected fp8→int8 export keys: {keys:?}");
}

// ---------------------------------------------------------------------------
// 4. cuda + int8 on real hardware — the quantized training path on the GPU.
// ---------------------------------------------------------------------------

/// The full quantized diffusion stack on real cuda hardware (`cuda + f32 +
/// int8`) — the exact production configuration for the #25 real run, exercised
/// on the tiny bundle so it needs no multi-GB weights. Distinct from the
/// ndarray e2e (which proves correctness) and the `quant_probe` example (which
/// proves the real model's memory fit): this proves the quantized *training*
/// path — the custom autodiff op, the streaming load, the kohya export — runs
/// on the cuda backend. Compiled only under the cuda feature; run via
/// `just test-cuda`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an NVIDIA GPU (CUDA toolkit at build time); run via `just test-cuda`"]
fn tiny_krea2_cuda_int8_trains_e2e() {
    const STEPS: u64 = 12;
    let _rng = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("quant-cuda-int8");
    let dataset = staged_dataset(&out);

    let mut cfg = config(&out, dataset, STEPS);
    cfg.compute.backend = BackendKind::Cuda; // cuda + f32 (default precision) + int8
    cfg.compute.quant = Quant::Int8;

    let mut losses = Vec::new();
    let mut finished = None;
    let mut quant_accounting = None;
    let adapter = DiffusionTrainer
        .train(&cfg, &mut |event| match event {
            TrainEvent::Step { loss, .. } => losses.push(loss),
            TrainEvent::Finished { adapter_path } => finished = Some(adapter_path),
            TrainEvent::Warning { message } if message.contains("int8-quantized") => {
                quant_accounting = Some(message)
            }
            _ => {}
        })
        .expect("the cuda int8 tiny Krea 2 run completes");

    assert_eq!(losses.len() as u64, STEPS, "one Step per step");
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "non-finite loss on cuda int8: {losses:?}"
    );
    assert!(
        quant_accounting
            .as_deref()
            .is_some_and(|m| m.contains("int8-quantized 52 frozen-base linear sites")),
        "cuda int8 quant accounting missing/unexpected: {quant_accounting:?}"
    );
    let keys = kohya_keys(&finished.unwrap_or(adapter));
    assert_eq!(keys.len(), 42, "unexpected cuda int8 export keys: {keys:?}");
}

/// The int4 twin of the cuda e2e (`cuda + f32 + int4`) — the exact production
/// configuration for the #25 24 GB real run, exercised on the tiny bundle. cuda
/// quantizes `Q4S` natively via cubecl's generic `PackedU32` store (no
/// `export_tests` needed, unlike ndarray). Run via `just test-cuda`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an NVIDIA GPU (CUDA toolkit at build time); run via `just test-cuda`"]
fn tiny_krea2_cuda_int4_trains_e2e() {
    const STEPS: u64 = 12;
    let _rng = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("quant-cuda-int4");
    let dataset = staged_dataset(&out);

    let mut cfg = config(&out, dataset, STEPS);
    cfg.compute.backend = BackendKind::Cuda; // cuda + f32 (default precision) + int4
    cfg.compute.quant = Quant::Int4;

    let mut losses = Vec::new();
    let mut finished = None;
    let mut quant_accounting = None;
    let adapter = DiffusionTrainer
        .train(&cfg, &mut |event| match event {
            TrainEvent::Step { loss, .. } => losses.push(loss),
            TrainEvent::Finished { adapter_path } => finished = Some(adapter_path),
            TrainEvent::Warning { message } if message.contains("int4-quantized") => {
                quant_accounting = Some(message)
            }
            _ => {}
        })
        .expect("the cuda int4 tiny Krea 2 run completes");

    assert_eq!(losses.len() as u64, STEPS, "one Step per step");
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "non-finite loss on cuda int4: {losses:?}"
    );
    assert!(
        quant_accounting
            .as_deref()
            .is_some_and(|m| m.contains("int4-quantized 52 frozen-base linear sites")),
        "cuda int4 quant accounting missing/unexpected: {quant_accounting:?}"
    );
    let keys = kohya_keys(&finished.unwrap_or(adapter));
    assert_eq!(keys.len(), 42, "unexpected cuda int4 export keys: {keys:?}");
}
