//! `BurnTrainer` step-loss golden vs PyTorch (#49 H9).
//!
//! `convergence.rs` only asserts a *trend* (`last < 0.7 * first`) — a trainer
//! that trains fast but **wrong** (a mis-scaled LoRA delta, a dropped
//! `weight_decay`, a coupled-instead-of-decoupled decay, a loss read after the
//! optimizer step) sails through it. This test pins the *exact* per-step loss
//! the public `Trainer` emits, against a checked-in golden computed
//! independently by PyTorch (`reference/burn_trainer_reference.py`), the way
//! `lora_reference.rs` pins the toy forward pass.
//!
//! ## What is and isn't independent
//!
//! burn's frozen base, LoRA `A` init, and Gaussian-blob dataset all come from
//! its seeded ChaCha `StdRng`, which PyTorch cannot reproduce — so the golden
//! cannot be derived end-to-end in Python. Instead, `just
//! burn-trainer-reference` dumps burn's **actual** init + batches
//! (`examples/dump_synthetic_run.rs`, via `burn_trainer::synthetic_run_inputs`)
//! and torch recomputes the **losses** from them:
//!
//! * **Independent** (this is the numerics proof): the forward pass, the
//!   `alpha/rank` scaling, the cross-entropy, the AdamW update incl. decoupled
//!   weight decay, the freeze, and the record-loss-*before*-step ordering.
//! * **Shared**: the initial weights and the training data. A bug in burn's
//!   *data generation* is a shared input and is not caught here — the black-box
//!   `convergence.rs` covers that half (it would stop converging).
//!
//! The dump path is itself gated by this test: `synthetic_run_inputs` has to
//! reproduce `run_classification`'s init and RNG ordering exactly (see
//! `.claude/rules/burn-lazy-param-init.md`) or torch is fed weights the trainer
//! never had and every loss below mismatches.
//!
//! Always-run, offline, no Python at `cargo test` time. Regenerate with
//! `just burn-trainer-reference` when the trainer's math, burn's version, or the
//! constants below change.

use burn::tensor::{TensorData, Tolerance};
use loractl_core::adapter::load_adapter;
use loractl_core::config::{
    ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig, OutputConfig,
    TaskKind,
};
use loractl_core::{BurnTrainer, Device, LoraMlp, NdArray, TrainConfig, TrainEvent, Trainer};
use std::path::{Path, PathBuf};

// The run under test. MUST match `examples/dump_synthetic_run.rs` — and the
// `hyperparams` assertions below mechanically enforce it: the golden records the
// values the dump was taken with, so a drift between the two fails loudly here
// instead of silently pinning a different run.
const SEED: u64 = 7;
const STEPS: u64 = 12;
const RANK: u32 = 4;
const ALPHA: f32 = 8.0;
const LR: f64 = 0.01;
/// `NUM_CLASSES` in `burn_trainer.rs` — `lora_b` is `[rank, out]`.
const NUM_CLASSES: usize = 10;

/// Tolerance: burn (ndarray) and torch (Accelerate) do the same arithmetic in a
/// different summation order over 784-wide dots, and the difference compounds
/// across 12 AdamW steps. Measured max |diff| on this machine: ~1e-5 (the suite
/// still passes at `absolute(1e-4)`), so 1e-3 leaves ~100x headroom for a CI
/// host with a different BLAS while staying a real gate — the mutants this test
/// is aimed at move the losses by 1e-2 or more (dropping `weight_decay` alone
/// separates the trajectories by 5e-2).
fn tol() -> Tolerance<f32> {
    Tolerance::absolute(1e-3)
}

#[derive(serde::Deserialize)]
struct Golden {
    hyperparams: Hyperparams,
    trajectories: Vec<Trajectory>,
}

#[derive(serde::Deserialize)]
struct Hyperparams {
    seed: u64,
    steps: u64,
    rank: u32,
    alpha: f32,
    lr: f64,
}

#[derive(serde::Deserialize)]
struct Trajectory {
    weight_decay: f64,
    losses: Vec<f32>,
    lora_b_final: Vec<f32>,
}

/// A unique temp output dir, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("loractl-{tag}-{}-{nanos}", std::process::id()));
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn config(out_dir: &Path, weight_decay: f64) -> TrainConfig {
    TrainConfig {
        steps: STEPS,
        seed: SEED,
        task: TaskKind::Classification,
        model: ModelConfig {
            base: "synthetic".into(),
        },
        lora: LoraConfig {
            rank: RANK,
            alpha: ALPHA,
            // Non-zero dropout would draw RNG inside the training loop, which
            // torch cannot replay — the golden run is dropout-free by design.
            dropout: 0.0,
            targets: vec![],
        },
        dataset: DatasetConfig {
            path: PathBuf::from("unused"),
            resolution: 28,
        },
        optim: OptimConfig {
            lr: LR,
            weight_decay,
        },
        output: OutputConfig {
            dir: out_dir.to_path_buf(),
            name: "adapter".into(),
            checkpoint_every: u64::MAX,
            sample_every: 0,
        },
        compute: ComputeConfig::default(),
        flow: FlowConfig::default(),
    }
}

#[test]
fn burn_trainer_step_losses_match_pytorch_golden() {
    let golden: Golden = serde_json::from_str(include_str!("golden/burn_trainer_steps.json"))
        .expect("golden fixture parses");

    // Drift guard: the golden is only meaningful for the run it was generated
    // from. If `examples/dump_synthetic_run.rs` and this file disagree about the
    // run, fail here — not with a confusing numerics mismatch 40 lines down.
    let hp = &golden.hyperparams;
    assert_eq!(
        (hp.seed, hp.steps, hp.rank, hp.alpha, hp.lr),
        (SEED, STEPS, RANK, ALPHA, LR),
        "the golden was generated for a different run — regenerate with \
         `just burn-trainer-reference` after changing the constants"
    );
    assert_eq!(
        golden.trajectories.len(),
        2,
        "the golden pins two weight-decay trajectories (0.0 and the kill-test value)"
    );

    let device: Device<NdArray> = Default::default();

    for (i, traj) in golden.trajectories.iter().enumerate() {
        let out = TempDir::new(&format!("h9-{i}"));
        let config = config(&out.0, traj.weight_decay);

        // Black-box: observe ONLY the public event stream, exactly as the CLI
        // and the API do.
        let mut losses = Vec::new();
        let mut trainer = BurnTrainer;
        let adapter = trainer
            .train(&config, &mut |event| {
                if let TrainEvent::Step { loss, .. } = event {
                    losses.push(loss);
                }
            })
            .expect("the training run succeeds");

        assert_eq!(
            losses.len(),
            STEPS as usize,
            "exactly one Step event per step"
        );
        TensorData::new(losses, [STEPS as usize]).assert_approx_eq::<f32>(
            &TensorData::new(traj.losses.clone(), [STEPS as usize]),
            tol(),
        );

        // The final trained `lora_b`, read back through the public adapter path.
        // It is zero-initialized, so it carries the ENTIRE learned update — and
        // pinning it catches an error in the last optimizer step, which no
        // recorded loss ever observes.
        let reloaded: LoraMlp<NdArray> =
            load_adapter(&adapter, &device).expect("the final adapter loads");
        reloaded
            .fc2
            .lora_b
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::new(traj.lora_b_final.clone(), [RANK as usize, NUM_CLASSES]),
                tol(),
            );
    }
}
