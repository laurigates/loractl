//! End-to-end convergence proof for the flow-matching (rectified-flow) task
//! (issue #19, M8 — the black-box half, mirroring `convergence.rs`).
//!
//! This drives the *public* `BurnTrainer` with `task: flow-matching` over its
//! seeded synthetic latent toy as a black box — observing only the
//! `TrainEvent` stream, never reaching into the model — and asserts that real
//! v-prediction training happens: the loss stream trends decisively downward,
//! the run brackets its work with `Started`/`Finished`, the adapter + sidecar
//! exist on disk, the sidecar records the flow-matching task, the adapter
//! reloads and forwards, and classifier sampling REFUSES the flow adapter.
//!
//! Also pins the toy's sign conventions: the flow toy is exactly
//! sign-symmetric (a flipped velocity target converges identically), so the
//! convergence gate alone cannot see a flipped `ε − x₀` or a flipped
//! interpolation — the [`flow_batches_routes_through_the_pinned_conventions`]
//! identity test is what pins them, on top of the golden-pinned helpers the
//! trainer routes through (`flow_reference.rs`).

use burn::backend::NdArray;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use loractl_core::adapter::{AdapterMeta, load_adapter};
use loractl_core::burn_trainer::flow_batches;
use loractl_core::config::{
    ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig, OutputConfig,
    TaskKind,
};
use loractl_core::sample::sample_adapter;
use loractl_core::{BurnTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::PathBuf;
use std::sync::Mutex;

/// See `adapter_roundtrip.rs`: the ndarray RNG is a process-global static, so
/// tests that seed it and rely on what gets drawn afterward serialize against
/// each other.
static RNG_LOCK: Mutex<()> = Mutex::new(());

/// Plain CPU backend — reload/forward and the batch identity checks are
/// inference-only (the alias also lets `TB::seed` resolve NdArray's default
/// generics).
type TB = NdArray;

/// A unique temp output dir so concurrent test runs don't collide or litter the
/// repo. Removed on drop — same convention as `convergence.rs`.
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

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len() as f32
}

/// OBSERVED (ndarray, seed 42, 150 steps): first-third mean 1.4395,
/// last-third mean 0.8260, ratio 0.574 vs the 0.7 bound (losses[0] = 3.488) —
/// consistent with the design review's torch-replica spread of 0.547–0.586
/// over 7 seeds.
///
/// Margin discipline: the model's representational floor (frozen random
/// features + rank-8 readout) is nonzero, so the criterion is a loss RATIO,
/// never an absolute near-zero loss. If the 0.7 margin ever needs widening,
/// raise `steps` (toward ~200), NEVER `lr` — against a nonzero floor a higher
/// lr shrinks the first-third mean and pushes the ratio TOWARD 0.7.
#[test]
fn flow_training_converges() {
    let _guard = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let steps = 150u64;
    let out = TempDir::new("flow-conv");
    let config = TrainConfig {
        steps,
        seed: 42,
        task: TaskKind::FlowMatching,
        model: ModelConfig {
            base: "synthetic".into(),
            variant: Default::default(),
            checkpoint: None,
            denoiser: None,
            text_encoder: None,
            vae: None,
            tokenizer: None,
        },
        lora: LoraConfig {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            targets: vec![],
        },
        dataset: DatasetConfig {
            // The flow toy generates synthetic latents; the dataset is unused.
            path: PathBuf::from("unused"),
            resolution: 28,
            batch_size: 1,
        },
        optim: OptimConfig {
            lr: 0.01,
            weight_decay: 0.0,
        },
        output: OutputConfig {
            dir: out.0.clone(),
            name: "flow-adapter".into(),
            // Larger than `steps` so no mid-run checkpoints — keeps the test fast
            // while still exercising the final-adapter write.
            checkpoint_every: 10_000,
            // MUST stay 0: validation sampling is classification-specific and
            // the trainer bails on flow + sample_every > 0 (see flow_task.rs).
            sample_every: 0,
        },
        // Default (ndarray) backend — this offline convergence proof must stay
        // on CPU; it doubles as the regression pin that the default is ndarray.
        compute: ComputeConfig::default(),
        // SD3/kohya sampler defaults (logit-normal N(0,1), shift 3.0).
        flow: FlowConfig::default(),
    };

    let mut losses = Vec::new();
    let mut started = false;
    let mut finished_path = None;

    let mut trainer = BurnTrainer;
    let adapter = trainer
        .train(&config, &mut |event| match event {
            TrainEvent::Started { total_steps } => {
                assert_eq!(total_steps, steps);
                started = true;
            }
            TrainEvent::Step { loss, .. } => losses.push(loss),
            TrainEvent::Finished { adapter_path } => finished_path = Some(adapter_path),
            _ => {}
        })
        .expect("flow training run succeeds");

    assert!(started, "a Started event must be emitted");
    assert_eq!(
        finished_path.as_ref(),
        Some(&adapter),
        "the Finished event's path must equal the returned adapter path"
    );
    assert!(
        adapter.exists(),
        "the final adapter record must exist on disk at {}",
        adapter.display()
    );
    assert_eq!(
        losses.len(),
        steps as usize,
        "exactly one Step event per step"
    );
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "all losses must be finite"
    );
    // Untrained-regime anchor: with the LoRA readout still ~zero, the MSE
    // against v = ε − c (unit noise + the ±1.5 point mass) sits well above 2 —
    // a run that starts already-converged would mean the toy lost its signal.
    assert!(
        losses[0] > 2.0,
        "first loss should be in the untrained regime, got {}",
        losses[0]
    );

    // Black-box convergence: the mean of the last third of losses must be well
    // below the mean of the first third. See the margin discipline in the doc
    // comment — raise steps, never lr, if this ever tightens.
    let third = losses.len() / 3;
    let first = mean(&losses[..third]);
    let last = mean(&losses[losses.len() - third..]);
    assert!(
        last < 0.7 * first,
        "loss should trend down: first-third mean {first:.4}, last-third mean {last:.4}"
    );

    // The sidecar must record the flow-matching task so downstream consumers
    // (the `sample` refusal below) can tell a velocity net from a classifier.
    let mut sidecar = adapter.clone().into_os_string();
    sidecar.push(".json");
    let json = std::fs::read_to_string(&sidecar).expect("adapter sidecar exists");
    let meta: AdapterMeta = serde_json::from_str(&json).expect("sidecar parses");
    assert_eq!(
        meta.task,
        TaskKind::FlowMatching,
        "the sidecar's task field must record the flow-matching task"
    );
    // MUST match burn_trainer.rs: FLOW_LATENT_DIM = 16, FLOW_HIDDEN = 64, and
    // the velocity net's input is concat[x_t, t] (one column wider than the
    // latent it predicts).
    assert_eq!(
        meta.d_in, 17,
        "velocity-net input width = latent + t column"
    );
    assert_eq!(meta.hidden, 64);
    assert_eq!(meta.out, 16, "velocity-net output width = latent dim");

    // The adapter reloads through the standard path and forwards finitely.
    let device = Default::default();
    let reloaded = load_adapter::<TB>(&adapter, &device).expect("flow adapter reloads");
    let d_in = reloaded.fc1.weight.dims()[0];
    let probe = Tensor::<TB, 2>::from_data(TensorData::new(vec![0.1f32; d_in], [1, d_in]), &device);
    let v = reloaded.forward(probe);
    assert_eq!(v.dims(), [1, 16], "predicted velocity is [1, LATENT_DIM]");
    let v: Vec<f32> = v.into_data().convert::<f32>().into_vec().unwrap();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "reloaded velocity output must be finite, got {v:?}"
    );

    // Fail-fast: classifier sampling must REFUSE the flow adapter instead of
    // printing a confidently-wrong "predicted class" from a velocity net.
    let err = sample_adapter::<TB>(&adapter, 0, &device)
        .expect_err("classifier sampling must refuse a flow-matching adapter");
    assert!(
        err.to_string().contains("flow"),
        "the refusal should name the flow-matching task, got: {err}"
    );
}

/// Sign-pinning identity test (review major): the flow toy is exactly
/// sign-symmetric — training against a flipped target `x₀ − ε` (or a flipped
/// interpolation) converges identically, so [`flow_training_converges`] cannot
/// see either flip. This test pins the conventions on `flow_batches`' real
/// output via identities that are asymmetric in the signs:
///
/// - the input's last column is the (0,1) timestep `t` — the SAME shifted `t`
///   used by the interpolation;
/// - recovering `ε = target + c` (valid only for `v = ε − x₀` with the known
///   point-mass `x₀ ≡ c`) and substituting into `x_t = (1−t)·c + t·ε` must
///   reproduce the input's x_t columns elementwise.
///
/// A flipped target makes the recovered ε wrong, a flipped interpolation makes
/// the reconstruction wrong, and an inconsistent t column breaks both — each
/// fails the elementwise comparison.
#[test]
fn flow_batches_routes_through_the_pinned_conventions() {
    let _guard = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let device = Default::default();
    TB::seed(&device, 7);

    let batches = flow_batches::<TB>(1, 8, FlowConfig::default(), &device);
    assert_eq!(batches.len(), 1);
    let (input, target) = &batches[0];

    let [batch, width] = input.dims();
    assert_eq!(batch, 8);
    let latent = width - 1;
    assert_eq!(latent, 16, "MUST match burn_trainer.rs FLOW_LATENT_DIM");
    assert_eq!(target.dims(), [batch, latent]);

    let input: Vec<f32> = input.to_data().convert::<f32>().into_vec().unwrap();
    let target: Vec<f32> = target.to_data().convert::<f32>().into_vec().unwrap();

    // The fixed point-mass data constant: c[j] = ±1.5 alternating. MUST match
    // burn_trainer.rs's `flow_batches`.
    let c: Vec<f32> = (0..latent)
        .map(|j| if j % 2 == 0 { 1.5 } else { -1.5 })
        .collect();

    for i in 0..batch {
        let t = input[i * width + latent];
        assert!(
            t > 0.0 && t < 1.0,
            "the t column must hold (0,1) timesteps, got {t} in row {i}"
        );
        for j in 0..latent {
            let x_t = input[i * width + j];
            let eps = target[i * latent + j] + c[j]; // v = ε − c  ⇒  ε = v + c
            let expected = (1.0 - t) * c[j] + t * eps;
            assert!(
                (x_t - expected).abs() < 1e-5,
                "x_t[{i}][{j}] = {x_t} but (1−t)·c + t·(target + c) = {expected} \
                 (t = {t}) — a sign convention flipped"
            );
        }
    }
}
