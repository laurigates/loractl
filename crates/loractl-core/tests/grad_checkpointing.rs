//! The M13 (#24) gradient-checkpointing knob: `compute.grad_checkpointing`
//! swaps burn's `Autodiff` strategy to `BalancedCheckpointing` (recompute
//! activations during backward instead of storing them).
//!
//! The invariant that makes this testable offline: recomputation replays the
//! SAME deterministic f32 ops the stored-activation path recorded, so the two
//! strategies must produce **bit-identical** loss trajectories — any
//! divergence means the knob changed the math, not just the memory profile.
//! (The memory saving itself is a profile property, not assertable in a unit
//! test; this pins that turning the knob is numerically free.)
//!
//! Also pins the f16 guard: `precision: f16` on the ndarray backend must be a
//! loud error (the M7 no-silent-fallback rule), never a quiet f32 run.

use loractl_core::config::{
    ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig, OutputConfig,
    Precision, TaskKind,
};
use loractl_core::{BurnTrainer, TrainConfig, TrainEvent, Trainer};
use std::path::PathBuf;

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

fn config(compute: ComputeConfig, out: &TempDir) -> TrainConfig {
    TrainConfig {
        steps: 8,
        seed: 42,
        task: TaskKind::Classification,
        model: ModelConfig {
            base: "synthetic".into(),
            variant: Default::default(),
            checkpoint: None,
            denoiser: None,
            text_encoder: None,
            vae: None,
            tokenizer: None,
            training_adapter: None,
        },
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
        },
        optim: OptimConfig {
            lr: 0.01,
            weight_decay: 0.0,
        },
        output: OutputConfig {
            dir: out.0.clone(),
            name: "adapter".into(),
            checkpoint_every: 10_000,
            sample_every: 0,
        },
        compute,
        flow: FlowConfig::default(),
    }
}

/// Run the synthetic demo with the given checkpointing setting; collect
/// losses and any advisory messages.
fn run_losses(grad_checkpointing: bool, out: &TempDir) -> (Vec<f32>, Vec<String>) {
    let compute = ComputeConfig {
        grad_checkpointing,
        ..ComputeConfig::default()
    };
    let mut losses = Vec::new();
    let mut warnings = Vec::new();
    BurnTrainer
        .train(&config(compute, out), &mut |event| match event {
            TrainEvent::Step { loss, .. } => losses.push(loss),
            TrainEvent::Warning { message } => warnings.push(message),
            _ => {}
        })
        .expect("training run succeeds");
    (losses, warnings)
}

#[test]
fn checkpointing_is_numerically_identical_to_stored_activations() {
    let out_off = TempDir::new("ckpt-off");
    let out_on = TempDir::new("ckpt-on");

    let (stored, off_warnings) = run_losses(false, &out_off);
    let (recomputed, on_warnings) = run_losses(true, &out_on);

    assert_eq!(stored.len(), 8, "one Step event per step");
    assert_eq!(
        stored, recomputed,
        "BalancedCheckpointing must replay the exact same math — bit-identical losses"
    );

    // Bit-identity alone cannot distinguish a working knob from a dead one
    // (a knob that never reached the dispatch would ALSO produce equal
    // losses). The advisory emitted from INSIDE the checkpointing branch is
    // the discriminator: present exactly when the flag took the
    // BalancedCheckpointing path.
    assert!(
        on_warnings.iter().any(|m| m.contains("checkpointing")),
        "the checkpointing branch must announce itself: {on_warnings:?}"
    );
    assert!(
        !off_warnings.iter().any(|m| m.contains("checkpointing")),
        "the stored-activation path must not: {off_warnings:?}"
    );
}

#[test]
fn f16_on_ndarray_fails_loudly() {
    let out = TempDir::new("f16-guard");
    let compute = ComputeConfig {
        precision: Precision::F16,
        ..ComputeConfig::default()
    };
    let err = BurnTrainer
        .train(&config(compute, &out), &mut |_| {})
        .expect_err("f16 on ndarray must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("f16") && msg.contains("wgpu"),
        "the error should name the fix (switch to wgpu): {msg}"
    );
}

#[test]
fn precision_parses_all_config_layers_accept() {
    // The FromStr the YAML/env/flag layers all route through.
    assert_eq!("f16".parse::<Precision>().unwrap(), Precision::F16);
    assert_eq!("FP16".parse::<Precision>().unwrap(), Precision::F16);
    assert_eq!("half".parse::<Precision>().unwrap(), Precision::F16);
    assert_eq!("f32".parse::<Precision>().unwrap(), Precision::F32);
    assert!("f64".parse::<Precision>().is_err());

    // And deserialization routes through the same parser (format-agnostic —
    // the YAML/env layers use this exact Deserialize impl via figment).
    let compute: ComputeConfig =
        serde_json::from_str(r#"{"precision":"FP16","grad_checkpointing":true}"#)
            .expect("config parses");
    assert_eq!(compute.precision, Precision::F16);
    assert!(compute.grad_checkpointing);
    assert!(
        serde_json::from_str::<ComputeConfig>(r#"{"precision":"f64"}"#).is_err(),
        "unknown precisions are rejected with the FromStr error"
    );
}
