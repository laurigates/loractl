//! Fail-fast backend dispatch contract (M7, #18).
//!
//! Selecting a GPU backend that this binary was **not** built with must return
//! an error — never a silent fall-back to CPU (which would defeat "a training
//! run executes on a GPU backend selected from config" and violate the
//! project's fail-fast rule). These cases run offline on the default (ndarray)
//! feature set, so they execute in CI and pin the `#[cfg(not(feature = ...))]
//! => bail!` arms in `burn_trainer.rs` cheaply, without any GPU.

use loractl_core::config::{
    BackendKind, ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig,
    OutputConfig, TaskKind,
};
use loractl_core::{BurnTrainer, TrainConfig, Trainer};
use std::path::PathBuf;

/// A minimal synthetic-demo config with the compute backend overridden. Kept
/// tiny (`steps: 1`) because the run must error at dispatch, before training.
fn cfg(backend: BackendKind) -> TrainConfig {
    TrainConfig {
        steps: 1,
        seed: 0,
        task: TaskKind::Classification,
        model: ModelConfig {
            base: "synthetic".into(),
            variant: Default::default(),
        },
        lora: LoraConfig {
            rank: 2,
            alpha: 4.0,
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
            dir: std::env::temp_dir().join("loractl-backend-dispatch"),
            name: "unused".into(),
            checkpoint_every: 10_000,
            sample_every: 0,
        },
        compute: ComputeConfig {
            backend,
            ..ComputeConfig::default()
        },
        flow: FlowConfig::default(),
    }
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn selecting_wgpu_without_the_feature_bails() {
    let err = BurnTrainer
        .train(&cfg(BackendKind::Wgpu), &mut |_event| {})
        .expect_err("wgpu selected without --features wgpu must error, not silently run on CPU");
    let msg = err.to_string();
    assert!(
        msg.contains("wgpu"),
        "the error should name the missing backend, got: {msg}"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn selecting_cuda_without_the_feature_bails() {
    let err = BurnTrainer
        .train(&cfg(BackendKind::Cuda), &mut |_event| {})
        .expect_err("cuda selected without --features cuda must error");
    assert!(err.to_string().contains("cuda"));
}

#[cfg(not(feature = "tch"))]
#[test]
fn selecting_tch_without_the_feature_bails() {
    let err = BurnTrainer
        .train(&cfg(BackendKind::Tch), &mut |_event| {})
        .expect_err("tch selected without --features tch must error");
    assert!(err.to_string().contains("tch"));
}

/// Backward-compat: a config with NO `compute:` block must deserialize onto the
/// ndarray CPU backend. This is the `#[serde(default)]` contract that keeps
/// every pre-M7 config working and `just test`/CI offline (acceptance #2 of
/// #18) — pinned so a future removal of the default can't silently regress it.
#[test]
fn config_without_compute_block_defaults_to_ndarray() {
    let json = r#"{
        "model": { "base": "synthetic" },
        "lora": {},
        "dataset": { "path": "unused" }
    }"#;
    let config: TrainConfig =
        serde_json::from_str(json).expect("a config without a compute block should deserialize");
    assert_eq!(config.compute.backend, BackendKind::Ndarray);
    assert_eq!(config.compute.device, 0);
}

/// The YAML/env deserialization surface accepts the same spellings as the
/// `--backend` flag — case-insensitive, plus the `libtorch` alias for `tch`
/// (`BackendKind`'s `Deserialize` routes through its `FromStr`) — so the three
/// config layers (YAML → env → flag) stay an interchangeable surface.
#[test]
fn backend_deserialize_matches_the_cli_vocabulary() {
    let parse = |s: &str| -> BackendKind {
        serde_json::from_value(serde_json::Value::String(s.into())).expect("known backend spelling")
    };
    assert_eq!(parse("wgpu"), BackendKind::Wgpu);
    assert_eq!(parse("WGPU"), BackendKind::Wgpu); // case-insensitive, like the flag
    assert_eq!(parse("libtorch"), BackendKind::Tch); // alias, like the flag
    assert!(
        serde_json::from_value::<BackendKind>(serde_json::Value::String("bogus".into())).is_err(),
        "an unknown backend spelling must be a clear error"
    );
}
