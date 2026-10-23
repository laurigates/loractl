//! `optim.schedule` end to end (#212): the scheduled rate reaches the
//! optimizer AND the `Step` event, a resumed run continues on the curve, and
//! the default config is untouched.
//!
//! The curve itself (hand-computed values, edge cases, validation) is pinned
//! in `src/schedule.rs`'s unit tests. What only a trainer run can show — and
//! what this file pins — is wiring: a schedule that is computed correctly and
//! then not handed to `optim.step`, or handed over but not reported, passes
//! every unit test. So the trajectory assertions here are about *behavior*
//! (the loss stream moves; the event stream carries the rate), never about a
//! helper being called.

use loractl_core::config::{
    DatasetConfig, LoraConfig, ModelConfig, ModelVariant, OptimConfig, OutputConfig,
    ScheduleConfig, ScheduleKind, TargetSpec, TaskKind,
};
use loractl_core::{BurnTrainer, DiffusionTrainer, MockTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::PathBuf;

const BUNDLE: &str = "tests/fixtures/tiny-krea2";
const DATASET: &str = "tests/fixtures/dataset-tiny";

/// burn's backend RNG is process-global, so trainings must not interleave.
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

fn cosine_with_warmup(warmup_steps: u64) -> ScheduleConfig {
    ScheduleConfig {
        kind: ScheduleKind::Cosine,
        warmup_steps,
        ..ScheduleConfig::default()
    }
}

/// The synthetic classification demo (BurnTrainer), seed 42.
fn synthetic(steps: u64, schedule: ScheduleConfig, out: &TempDir) -> TrainConfig {
    TrainConfig {
        steps,
        seed: 42,
        task: TaskKind::Classification,
        lora: LoraConfig {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            targets: vec![],
        },
        dataset: DatasetConfig {
            path: PathBuf::from("unused"),
            resolution: 28,
            batch_size: 1,
            ..Default::default()
        },
        optim: OptimConfig {
            lr: 0.01,
            weight_decay: 0.0,
            schedule,
        },
        output: OutputConfig {
            dir: out.0.clone(),
            name: "adapter".into(),
            checkpoint_every: 10_000,
            sample_every: 0,
        },
        ..Default::default()
    }
}

/// `(step, loss, lr)` for every Step event of a run.
fn steps_of(trainer: &mut dyn Trainer, config: &TrainConfig) -> Vec<(u64, f32, f64)> {
    let mut out = Vec::new();
    trainer
        .train(config, &mut |event| {
            if let TrainEvent::Step { step, loss, lr } = event {
                out.push((step, loss, lr));
            }
        })
        .expect("the run completes");
    out
}

/// The default config applies `optim.lr` unchanged: every `Step` event carries
/// it bit-for-bit. (The loss trajectory itself is pinned against the PyTorch
/// golden by `burn_trainer_reference.rs`, which now runs through the scheduled
/// code path with the default schedule.)
#[test]
fn default_schedule_reports_the_bare_lr_on_every_step() {
    let _lock = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let out = TempDir::new("lr-default");
    let config = synthetic(12, ScheduleConfig::default(), &out);
    let got = steps_of(&mut BurnTrainer, &config);
    assert_eq!(got.len(), 12);
    for (step, _, lr) in got {
        assert_eq!(lr.to_bits(), 0.01f64.to_bits(), "step {step}");
    }
}

/// The `Step` event carries the SCHEDULED rate, equal to the pure function.
#[test]
fn step_events_carry_the_scheduled_lr() {
    let _lock = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let out = TempDir::new("lr-events");
    let config = synthetic(10, cosine_with_warmup(3), &out);
    let got = steps_of(&mut BurnTrainer, &config);
    assert_eq!(got.len(), 10);
    for (step, _, lr) in &got {
        assert_eq!(*lr, config.optim.lr_at(*step, 10), "step {step}");
    }
    // Warmup ramps to the peak, then decay strictly lowers it.
    assert!((got[0].2 - 0.01 / 3.0).abs() < 1e-12);
    assert_eq!(got[2].2, 0.01);
    assert!(got[9].2 < got[3].2 && got[9].2 > 0.0);

    // MockTrainer reports through the same function.
    let mock = steps_of(&mut MockTrainer, &config);
    for (step, _, lr) in &mock {
        assert_eq!(*lr, config.optim.lr_at(*step, 10), "mock step {step}");
    }
}

/// The scheduled rate must reach `optim.step`, not only the event: a cosine
/// run and a constant run from the same seed share step 1's loss (the loss is
/// recorded before the update) and then diverge.
#[test]
fn the_schedule_changes_training_not_just_the_report() {
    let _lock = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let out_a = TempDir::new("lr-train-const");
    let out_b = TempDir::new("lr-train-cos");
    let constant = steps_of(
        &mut BurnTrainer,
        &synthetic(20, ScheduleConfig::default(), &out_a),
    );
    let cosine = steps_of(
        &mut BurnTrainer,
        &synthetic(20, cosine_with_warmup(5), &out_b),
    );
    assert_eq!(
        constant[0].1, cosine[0].1,
        "step 1's loss is measured before any update"
    );
    assert_ne!(
        constant.iter().map(|s| s.1).collect::<Vec<_>>(),
        cosine.iter().map(|s| s.1).collect::<Vec<_>>(),
        "identical loss streams mean the scheduled lr never reached optim.step"
    );
    assert!(cosine.iter().all(|s| s.1.is_finite()));
}

/// A malformed schedule is refused before any event, on every trainer.
#[test]
fn an_invalid_schedule_is_refused_before_any_event() {
    let _lock = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let out = TempDir::new("lr-invalid");
    // warmup consumes the whole run; and `exponential` with no gamma.
    for bad in [
        ScheduleConfig {
            warmup_steps: 5,
            ..ScheduleConfig::default()
        },
        ScheduleConfig {
            kind: ScheduleKind::Exponential,
            ..ScheduleConfig::default()
        },
    ] {
        let config = synthetic(5, bad, &out);
        let mut trainers: Vec<(&str, Box<dyn Trainer>)> = vec![
            ("burn", Box::new(BurnTrainer)),
            ("mock", Box::new(MockTrainer)),
            ("diffusion", Box::new(DiffusionTrainer)),
        ];
        for (name, trainer) in &mut trainers {
            let mut events = 0;
            let err = trainer
                .train(&config, &mut |_| events += 1)
                .expect_err(name);
            // The diffusion trainer's own earlier checks (e.g. task) may fire
            // first on this synthetic config; for the other two the schedule
            // is the only thing wrong.
            if *name != "diffusion" {
                assert!(err.to_string().contains("optim.schedule"), "{name}: {err}");
            }
            assert_eq!(events, 0, "{name} emitted before refusing");
        }
    }
}

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

fn diffusion(dir: PathBuf, dataset: PathBuf, steps: u64, schedule: ScheduleConfig) -> TrainConfig {
    TrainConfig {
        steps,
        seed: 42,
        task: TaskKind::FlowMatching,
        model: ModelConfig {
            base: BUNDLE.into(),
            variant: ModelVariant::TinyKrea2,
            ..ModelConfig::default()
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
            ..Default::default()
        },
        optim: OptimConfig {
            lr: 0.01,
            weight_decay: 0.0,
            schedule,
        },
        output: OutputConfig {
            dir,
            name: "krea2-lora".into(),
            checkpoint_every: 2,
            sample_every: 0,
        },
        ..Default::default()
    }
}

/// **The resume contract.** A run resumed from `checkpoint-2` of a 6-step run
/// reports, for steps 3..=6, exactly the rates an uninterrupted run did — the
/// curve continues; the warmup does not restart.
///
/// The schedule has a 2-step warmup so a restart is unmistakable: a restarted
/// run would report 0.005 (warmup step 1) at step 3, where the correct rate is
/// the 0.01 peak. That teeth-check is asserted explicitly so the equality
/// cannot pass vacuously (e.g. a schedule flat over these steps).
#[test]
fn a_resumed_run_continues_on_the_curve() {
    let _lock = TRAIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let out = TempDir::new("lr-resume");
    let dataset = staged_dataset(&out);
    let schedule = cosine_with_warmup(2);

    let dir = out.0.join("full");
    let full = steps_of(
        &mut DiffusionTrainer,
        &diffusion(dir.clone(), dataset.clone(), 6, schedule.clone()),
    );
    assert_eq!(
        full.iter().map(|s| s.0).collect::<Vec<_>>(),
        (1..=6).collect::<Vec<_>>()
    );
    // Teeth: warmup is visibly different from the peak.
    assert_eq!(full[0].2, 0.005);
    assert_eq!(full[2].2, 0.01);

    let mut resumed = diffusion(out.0.join("recovered"), dataset, 6, schedule);
    resumed.resume.from = Some(dir.join("checkpoint-2.safetensors"));
    let tail = steps_of(&mut DiffusionTrainer, &resumed);
    assert_eq!(
        tail.iter().map(|s| s.0).collect::<Vec<_>>(),
        vec![3, 4, 5, 6]
    );
    for ((step, _, lr), (full_step, _, full_lr)) in tail.iter().zip(&full[2..]) {
        assert_eq!(step, full_step);
        assert_eq!(
            lr.to_bits(),
            full_lr.to_bits(),
            "step {step}: a resumed run must continue on the curve, not restart warmup"
        );
    }
    assert_ne!(
        tail[0].2, full[0].2,
        "the resumed first step must NOT be the warmup's first step"
    );
}
