//! Explicit, lossless resume on the diffusion path (#180) — offline, ndarray,
//! against the checked-in tiny Krea 2 bundle.
//!
//! The *headline* acceptance case (train N, stop, continue) and the
//! rank-mismatch refusal are asserted where the issue asked for them, on the
//! existing int8 e2e (`tests/quant_trainer.rs`) and the fp32 e2e
//! (`tests/diffusion_trainer.rs`), rather than duplicated here. What this file
//! adds is everything those two cannot reach without becoming something else:
//! the optimizer sidecar's *effect*, the three provenance states the
//! `__metadata__` header can express, the corrupt-source refusal, and the
//! recorded decision that the synthetic path does not resume.
//!
//! Every assertion here is about a *trajectory*, never about a file existing:
//! a sidecar that is written and then ignored is precisely the failure this
//! feature is meant to be immune to, and "the file is there" cannot see it.

use loractl_core::config::{
    DatasetConfig, LoraConfig, ModelConfig, ModelVariant, OptimConfig, OutputConfig, TargetSpec,
    TaskKind,
};
use loractl_core::{
    BurnTrainer, DiffusionTrainer, TrainConfig, TrainEvent, Trainer, read_metadata,
};
use std::path::{Path, PathBuf};

const BUNDLE: &str = "tests/fixtures/tiny-krea2";
const DATASET: &str = "tests/fixtures/dataset-tiny";
/// The kohya key whose weights every trajectory assertion below compares.
/// `lora_up` starts at zero, so any movement in it is training, not init.
const PROBE: &str = "transformer_blocks.0.attn.to_q.lora_up.weight";

/// burn's backend RNG is process-global, so trainings must not interleave
/// (mirrors `tests/diffusion_trainer.rs`).
static TRAIN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

/// The pipeline writes `.loractl-cache/` INTO the dataset folder, so tests
/// stage a copy rather than dirtying the repo tree.
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

fn config(dir: PathBuf, dataset: PathBuf, steps: u64) -> TrainConfig {
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
            dir,
            name: "krea2-lora".into(),
            // Larger than any `steps` here unless a test lowers it, so mid-run
            // checkpoints only appear where a test wants them.
            checkpoint_every: 10_000,
            sample_every: 0,
        },
        ..Default::default()
    }
}

/// One training run, returning the step numbers it emitted and the resume
/// Warning it emitted (if any).
fn run(config: &TrainConfig) -> (Vec<u64>, Option<String>, PathBuf) {
    let mut steps = Vec::new();
    let mut note = None;
    let adapter = DiffusionTrainer
        .train(config, &mut |event| match event {
            TrainEvent::Step { step, .. } => steps.push(step),
            TrainEvent::Warning { message } if message.contains("resuming") => note = Some(message),
            _ => {}
        })
        .expect("the run completes");
    (steps, note, adapter)
}

fn tensor_f32(path: &Path, key: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read the artifact");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse the artifact");
    st.tensor(key)
        .unwrap_or_else(|_| panic!("{key} present in {}", path.display()))
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Copy every regular file of `src` into a fresh `dst` — used to fork one
/// training state into two independent continuations.
fn fork_state(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            std::fs::copy(&path, dst.join(path.file_name().unwrap())).unwrap();
        }
    }
}

/// Overwrite the first `f32` of `tensor` in place (mirrors
/// `tests/adapter_guards.rs`): the container stays structurally valid, so only
/// the *value* is under test.
fn poison_first_element(path: &Path, tensor: &str, poison: f32) {
    let mut bytes = std::fs::read(path).expect("read the adapter file");
    let header_len =
        u64::from_le_bytes(bytes[..8].try_into().expect("8-byte header length")) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + header_len]).expect("safetensors header parses");
    let start = header[tensor]["data_offsets"][0]
        .as_u64()
        .expect("tensor is present with data_offsets") as usize;
    let at = 8 + header_len + start;
    bytes[at..at + 4].copy_from_slice(&poison.to_le_bytes());
    std::fs::write(path, bytes).expect("write the poisoned adapter file");
}

/// The optimizer sidecar is not decoration: restoring AdamW's moments changes
/// the very next update, and the bias-correction step counter round-trips.
///
/// Two continuations are forked from the *same* trained state — one with the
/// sidecar beside it, one with it deleted — so the only difference between
/// them is the optimizer state. If `load_optimizer_state` were a no-op the two
/// would be identical, which is the kill-test for this whole feature (a file
/// that is written and then ignored looks exactly like a working one from the
/// outside).
#[test]
fn restored_optimizer_moments_change_the_next_update() {
    const BASE: u64 = 4;
    let _lock = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("resume-moments");
    let dataset = staged_dataset(&out);

    let base_dir = out.0.join("base");
    let (steps, note, _) = run(&config(base_dir.clone(), dataset.clone(), BASE));
    assert_eq!(steps, (1..=BASE).collect::<Vec<_>>());
    assert!(note.is_none(), "a fresh run must not announce a resume");
    let sidecar_name = "krea2-lora.optim.safetensors";
    assert_eq!(
        read_metadata(&base_dir.join(sidecar_name))
            .unwrap()
            .get("loractl_adamw_time"),
        Some("4"),
        "the sidecar records AdamW's step count, which no moment tensor implies"
    );

    let with_dir = out.0.join("with");
    let without_dir = out.0.join("without");
    fork_state(&base_dir, &with_dir);
    fork_state(&base_dir, &without_dir);
    std::fs::remove_file(without_dir.join(sidecar_name)).unwrap();

    // `resume.allow_unfinished` is left at its default `false` here: a genuine
    // FINISHED artifact resumes without the escape hatch, so the refusal below
    // cannot be passing merely because "auto resume is refused".
    let (steps_with, note_with, adapter_with) =
        run(&config(with_dir.clone(), dataset.clone(), BASE + 1));
    let (steps_without, note_without, adapter_without) =
        run(&config(without_dir.clone(), dataset.clone(), BASE + 1));

    assert_eq!(
        steps_with,
        vec![BASE + 1],
        "the continuation runs step 5 only"
    );
    assert_eq!(steps_without, vec![BASE + 1]);

    let note_with = note_with.expect("the resumed run announces itself");
    assert!(
        note_with.contains("AdamW moments for") && note_with.contains("at step 4"),
        "{note_with}"
    );
    let note_without = note_without.expect("the resumed run announces itself");
    assert!(
        note_without.contains("NOT restored: AdamW's moments"),
        "a run with no sidecar must SAY the moments were not restored: {note_without}"
    );

    // The trajectories diverge: same weights in, different optimizer state,
    // different weights out.
    assert_ne!(
        tensor_f32(&adapter_with, PROBE),
        tensor_f32(&adapter_without, PROBE),
        "restoring AdamW's moments must change the update — if these are equal \
         the sidecar is being written and ignored"
    );
    // ...and the step counter continued rather than restarting from 1.
    assert_eq!(
        read_metadata(&with_dir.join(sidecar_name))
            .unwrap()
            .get("loractl_adamw_time"),
        Some("5"),
        "a restored optimizer continues its bias correction"
    );
    assert_eq!(
        read_metadata(&without_dir.join(sidecar_name))
            .unwrap()
            .get("loractl_adamw_time"),
        Some("1"),
        "an unrestored optimizer starts its bias correction over"
    );
}

/// An UNFINISHED artifact at the automatic path is refused by name, and the
/// documented escape hatch gets an operator through.
///
/// The final export is only ever written with a finish timestamp, so an
/// unfinished file there means a checkpoint was copied into place (exactly the
/// recovery workflow the issue describes) or a write was interrupted. Refusing
/// costs one flag; not refusing costs a run that silently trains on a file
/// nobody chose.
#[test]
fn an_unfinished_auto_target_is_refused_by_name_with_a_documented_way_through() {
    const BASE: u64 = 4;
    let _lock = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("resume-unfinished");
    let dataset = staged_dataset(&out);

    let dir = out.0.join("out");
    let mut first = config(dir.clone(), dataset.clone(), BASE);
    first.output.checkpoint_every = 2;
    run(&first);

    // The issue's workflow verbatim: copy a mid-run checkpoint into place.
    let checkpoint = dir.join("checkpoint-2.safetensors");
    std::fs::copy(&checkpoint, dir.join("krea2-lora.safetensors")).unwrap();

    let refused = config(dir.clone(), dataset.clone(), BASE + 2);
    let err = DiffusionTrainer
        .train(&refused, &mut |_| {})
        .expect_err("an unfinished auto-target must be refused");
    let message = format!("{err:#}");
    assert!(message.contains("krea2-lora.safetensors"), "{message}");
    assert!(
        message.contains("records 2 of 4 steps"),
        "the refusal must quote the file's own ss_steps/ss_max_train_steps: {message}"
    );
    assert!(
        message.contains("resume.allow_unfinished"),
        "the refusal must name the way through: {message}"
    );

    // The way through works, and says the source was unfinished.
    let mut allowed = config(dir.clone(), dataset.clone(), BASE + 2);
    allowed.resume.allow_unfinished = true;
    let (steps, note, _) = run(&allowed);
    assert_eq!(
        steps,
        (3..=BASE + 2).collect::<Vec<_>>(),
        "the run continues from the checkpoint's step 2"
    );
    let note = note.expect("announced");
    assert!(note.contains("did not finish"), "{note}");
    // The copy moved the WEIGHTS (step 2) and left the finished run's sidecar
    // (step 4) beside them. Pairing those is invisible — same shapes, same
    // sites — so the mismatch is caught on the two recorded step counts and
    // reported here rather than restored (`resume::stale_sidecar`).
    assert!(
        note.contains("NOT restored: AdamW's moments"),
        "a step-4 sidecar must not be paired with step-2 weights: {note}"
    );
    assert!(note.contains("krea2-lora.optim.safetensors"), "{note}");
    assert!(
        note.contains("records 4 steps") && note.contains("the source records 2"),
        "the skip must quote both step counts: {note}"
    );
}

/// Naming a checkpoint explicitly is itself the statement of intent, so
/// `resume.from` does NOT require `allow_unfinished` — every mid-run
/// checkpoint is unfinished by construction, and requiring the flag would make
/// the flag mandatory for the feature's main use.
///
/// This is the twin of the test above: the two paths deliberately differ, and
/// applying the finished-check to both would break exactly this.
#[test]
fn an_explicitly_named_checkpoint_resumes_without_the_escape_hatch() {
    const BASE: u64 = 4;
    let _lock = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("resume-explicit");
    let dataset = staged_dataset(&out);

    let dir = out.0.join("out");
    let mut first = config(dir.clone(), dataset.clone(), BASE);
    first.output.checkpoint_every = 2;
    run(&first);

    // Into a CLEAN output directory, from a source in another one — the
    // "recover into a fresh dir" workflow. The sidecar is looked for beside
    // the SOURCE, so the moments still come back.
    let mut explicit = config(out.0.join("recovered"), dataset.clone(), BASE + 1);
    explicit.resume.from = Some(dir.join("checkpoint-2.safetensors"));
    assert!(!explicit.resume.allow_unfinished);
    let (steps, note, _) = run(&explicit);
    assert_eq!(steps, (3..=BASE + 1).collect::<Vec<_>>());
    let note = note.expect("announced");
    assert!(note.contains("resume.from"), "{note}");
    assert!(note.contains("checkpoint-2.safetensors"), "{note}");
    assert!(
        note.contains("AdamW moments for"),
        "the sidecar beside the SOURCE must still be found: {note}"
    );
}

/// `metadata.embed: false` writes no `__metadata__` at all, so "no header" is
/// a THIRD state, not a synonym for unfinished. It must proceed — refusing
/// would break every `--no-metadata` user — while saying plainly that the step
/// count was not restored.
#[test]
fn an_export_without_metadata_resumes_but_cannot_restore_the_step_count() {
    const BASE: u64 = 3;
    let _lock = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("resume-nometa");
    let dataset = staged_dataset(&out);

    let dir = out.0.join("out");
    let mut first = config(dir.clone(), dataset.clone(), BASE);
    first.metadata.embed = false;
    run(&first);
    assert!(
        read_metadata(&dir.join("krea2-lora.safetensors"))
            .unwrap()
            .get("ss_steps")
            .is_none(),
        "the premise: metadata.embed = false writes no ss_steps"
    );

    let mut second = config(dir.clone(), dataset.clone(), BASE);
    second.metadata.embed = false;
    let (steps, note, _) = run(&second);
    assert_eq!(
        steps,
        (1..=BASE).collect::<Vec<_>>(),
        "an unrecorded source numbers its steps from 1"
    );
    let note = note.expect("announced");
    assert!(note.contains("no ss_steps"), "{note}");
    assert!(note.contains("NOT"), "{note}");
}

/// A resume source with a non-finite weight is refused at import, by tensor
/// name. Without this the run proceeds and every subsequent loss is NaN — a
/// forward that "works" on poison (ADR-0010's shape).
#[test]
fn a_corrupt_resume_source_is_refused_by_tensor_name() {
    const BASE: u64 = 2;
    let _lock = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("resume-corrupt");
    let dataset = staged_dataset(&out);

    let dir = out.0.join("out");
    run(&config(dir.clone(), dataset.clone(), BASE));
    let adapter = dir.join("krea2-lora.safetensors");
    poison_first_element(&adapter, PROBE, f32::NAN);

    let mut resumed = config(dir.clone(), dataset.clone(), BASE + 1);
    // The escape hatch is about PROVENANCE, not content: it must not wave a
    // corrupt file through.
    resumed.resume.allow_unfinished = true;
    let err = DiffusionTrainer
        .train(&resumed, &mut |_| {})
        .expect_err("a NaN weight must not be resumed from");
    let message = format!("{err:#}");
    assert!(message.contains(PROBE), "{message}");
    assert!(message.contains("not finite"), "{message}");
}

/// The sidecar twin of the test above: a non-finite *moment* is refused by key,
/// not merely a non-finite weight.
///
/// The asymmetry is what makes this worth its own test. `import_adapters`
/// guards the weights, so a corrupt adapter dies at the read with the tensor
/// named. A corrupt moment restores cleanly, poisons the first update, and the
/// run dies one step later inside `check_step_loss` — whose message blames f16
/// range unconditionally (`.claude/rules/gpu-runner-failure-signatures.md`),
/// sending the operator to `compute.precision` instead of to this file. The
/// source is reachable: `save_optimizer_state` writes in place with no
/// temp-and-rename, so an interrupted write leaves a structurally valid file
/// with garbage in it.
#[test]
fn a_corrupt_optimizer_sidecar_is_refused_by_key() {
    const BASE: u64 = 2;
    let _lock = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("resume-corrupt-optim");
    let dataset = staged_dataset(&out);

    let dir = out.0.join("out");
    run(&config(dir.clone(), dataset.clone(), BASE));

    // The sidecar is keyed by SITE PATH (`blocks.0.attn.wq`), where the export
    // is keyed by kohya name (`transformer_blocks.0.attn.to_q`) — that
    // difference is deliberate (`save_optimizer_state`: this file is ours
    // alone), so this is PROBE's site-path twin rather than PROBE itself.
    let moment = "blocks.0.attn.wq.lora_up.exp_avg";
    let sidecar = dir.join("krea2-lora.optim.safetensors");
    poison_first_element(&sidecar, moment, f32::NAN);

    let mut resumed = config(dir.clone(), dataset.clone(), BASE + 1);
    resumed.resume.allow_unfinished = true;
    let err = DiffusionTrainer
        .train(&resumed, &mut |_| {})
        .expect_err("a NaN moment must not be resumed from");
    let message = format!("{err:#}");
    assert!(message.contains(moment), "{message}");
    assert!(message.contains("not finite"), "{message}");
    // Named as the sidecar, not as the adapter: the whole point is that the
    // operator is sent to the right file.
    assert!(message.contains("sidecar"), "{message}");
}

/// `resume.auto: false` (`--no-resume`) forces a fresh start into a directory
/// that already holds an adapter — the surprise the implicit trigger used to
/// have no answer for.
#[test]
fn no_resume_starts_over_in_a_populated_directory() {
    const BASE: u64 = 3;
    let _lock = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("resume-off");
    let dataset = staged_dataset(&out);

    let dir = out.0.join("out");
    run(&config(dir.clone(), dataset.clone(), BASE));

    let mut fresh = config(dir.clone(), dataset.clone(), BASE);
    fresh.resume.auto = false;
    let (steps, note, _) = run(&fresh);
    assert_eq!(steps, (1..=BASE).collect::<Vec<_>>());
    assert!(note.is_none(), "--no-resume must not resume: {note:?}");
}

/// The recorded decision (#180): the synthetic/MNIST path has **no** resume,
/// intentionally — see the comment in `burn_trainer.rs`'s training loop for
/// the three reasons. Made observable so "intentional" cannot decay into
/// "someone added it and nobody noticed": two runs into the same output dir
/// must emit no resume Warning and the identical loss stream.
#[test]
fn the_synthetic_trainer_does_not_resume() {
    let _lock = TRAIN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = TempDir::new("resume-synthetic");
    let dir = out.0.join("out");
    let mut cfg = config(dir, PathBuf::from("unused"), 4);
    cfg.model.base = "synthetic".into();
    cfg.model.variant = ModelVariant::Krea2;
    cfg.task = TaskKind::Classification;

    let collect = || {
        let mut losses = Vec::new();
        let mut note = None;
        BurnTrainer
            .train(&cfg, &mut |event| match event {
                TrainEvent::Step { loss, .. } => losses.push(loss),
                TrainEvent::Warning { message } if message.contains("resuming") => {
                    note = Some(message)
                }
                _ => {}
            })
            .expect("the synthetic run completes");
        (losses, note)
    };

    let (first, note_first) = collect();
    let (second, note_second) = collect();
    assert!(note_first.is_none() && note_second.is_none());
    assert_eq!(
        first, second,
        "the synthetic path restarts from scratch every run — a warm start \
         would diverge here (and `load_adapter` regenerates the frozen base \
         from the seed, which burn 0.21's lazy Param init makes \
         RNG-history-dependent)"
    );
}
