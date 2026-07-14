//! `TaskKind` config-surface contract + the flow-task validation bail (M8, #19).
//!
//! Mirrors `backend_dispatch.rs`: the task enum and its serde surface are
//! always compiled, the YAML/env/flag layers accept the same spellings, an
//! existing config with no `task:`/`flow:` keys keeps parsing unchanged, and
//! an unsupported config combination (flow-matching + validation sampling)
//! bails loudly — before the backend dispatch, before any event, never a
//! silent fallback.

use loractl_core::adapter::AdapterMeta;
use loractl_core::config::{
    BackendKind, ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, ModelConfig, OptimConfig,
    OutputConfig, TaskKind,
};
use loractl_core::{BurnTrainer, TrainConfig, Trainer};
use std::path::PathBuf;

/// A minimal flow-matching config with validation sampling (invalidly) enabled,
/// on an arbitrary backend. Kept tiny (`steps: 1`) because the run must error
/// at validation, before training.
fn flow_sampling_cfg(backend: BackendKind) -> TrainConfig {
    TrainConfig {
        steps: 1,
        seed: 0,
        task: TaskKind::FlowMatching,
        model: ModelConfig {
            base: "synthetic".into(),
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
        },
        optim: OptimConfig {
            lr: 0.01,
            weight_decay: 0.0,
        },
        output: OutputConfig {
            dir: std::env::temp_dir().join("loractl-flow-task"),
            name: "unused".into(),
            checkpoint_every: 10_000,
            // The invalid combination under test: sampling is classification-
            // specific and must be rejected for the flow task.
            sample_every: 5,
        },
        compute: ComputeConfig {
            backend,
            ..ComputeConfig::default()
        },
        flow: FlowConfig::default(),
    }
}

/// Flow-matching + `sample_every > 0` must bail with a clear error — a
/// velocity net has no classifier sample path, and silently skipping the
/// samples (or writing "predicted class" reports from velocities) would
/// violate fail-fast. The sink must never fire: the config is rejected before
/// any `TrainEvent`, before `B::seed`, before any filesystem I/O.
#[test]
fn flow_task_with_validation_sampling_bails() {
    let mut events = 0usize;
    let err = BurnTrainer
        .train(&flow_sampling_cfg(BackendKind::Ndarray), &mut |_event| {
            events += 1;
        })
        .expect_err("flow-matching + sample_every > 0 must error, not silently skip sampling");
    let msg = err.to_string();
    assert!(
        msg.contains("sample"),
        "the error should name the invalid sampling setting, got: {msg}"
    );
    assert_eq!(
        events, 0,
        "the invalid config must be rejected before any TrainEvent is emitted"
    );
}

/// The validation must run BEFORE the backend match: a config that is *both*
/// invalid (flow + sampling) *and* names a backend this binary was not built
/// with must report the validation error, not the missing-backend one — the
/// combination is invalid on every backend, so it fails identically
/// everywhere.
#[cfg(not(feature = "wgpu"))]
#[test]
fn flow_sampling_validation_precedes_backend_dispatch() {
    let err = BurnTrainer
        .train(&flow_sampling_cfg(BackendKind::Wgpu), &mut |_event| {})
        .expect_err("the invalid task/sampling combination must error");
    let msg = err.to_string();
    assert!(
        msg.contains("sample") && !msg.contains("wgpu"),
        "validation must precede backend dispatch — expected the sampling error, got: {msg}"
    );
}

/// `FlowMatching` must serialize as kebab-case `"flow-matching"` — a plain
/// lowercase rename would emit `"flowmatching"`, which round-trips (it's a
/// belt-and-braces `FromStr` spelling) but diverges from the documented
/// config vocabulary.
#[test]
fn task_kind_serializes_kebab_case() {
    assert_eq!(
        serde_json::to_value(TaskKind::Classification).unwrap(),
        serde_json::Value::String("classification".into())
    );
    assert_eq!(
        serde_json::to_value(TaskKind::FlowMatching).unwrap(),
        serde_json::Value::String("flow-matching".into())
    );
}

/// Serialize → Deserialize is stable for both variants (the hand-written
/// `Deserialize` routes through `FromStr`, which must accept everything the
/// derived kebab-case `Serialize` emits).
#[test]
fn task_kind_round_trips_through_serde() {
    for task in [TaskKind::Classification, TaskKind::FlowMatching] {
        let json = serde_json::to_string(&task).expect("task serializes");
        let back: TaskKind = serde_json::from_str(&json).expect("serialized task deserializes");
        assert_eq!(back, task, "round-trip must be lossless for {json}");
    }
}

/// The YAML/env deserialization surface accepts the same spellings as the
/// `--task` flag — case-insensitive, with the underscore/joined/`flow`
/// aliases — so the three config layers (YAML → env → flag) stay an
/// interchangeable surface, exactly like `BackendKind`.
#[test]
fn task_deserialize_matches_the_cli_vocabulary() {
    let parse = |s: &str| -> TaskKind {
        serde_json::from_value(serde_json::Value::String(s.into())).expect("known task spelling")
    };
    assert_eq!(parse("classification"), TaskKind::Classification);
    assert_eq!(parse("Classification"), TaskKind::Classification); // case-insensitive
    assert_eq!(parse("flow-matching"), TaskKind::FlowMatching);
    assert_eq!(parse("FLOW-MATCHING"), TaskKind::FlowMatching); // case-insensitive
    assert_eq!(parse("flow_matching"), TaskKind::FlowMatching); // env-var friendly
    assert_eq!(parse("flowmatching"), TaskKind::FlowMatching); // lowercase-rename belt
    assert_eq!(parse("flow"), TaskKind::FlowMatching); // short alias
    assert!(
        serde_json::from_value::<TaskKind>(serde_json::Value::String("bogus".into())).is_err(),
        "an unknown task spelling must be a clear error"
    );
}

/// Backward-compat: a config with NO `task:` or `flow:` keys must deserialize
/// onto the classification task with the SD3/kohya flow defaults. This is the
/// `#[serde(default)]` contract that keeps every pre-M8 config working —
/// pinned so a future removal of the defaults can't silently regress it
/// (mirrors `config_without_compute_block_defaults_to_ndarray`).
#[test]
fn config_without_task_or_flow_blocks_defaults_to_classification() {
    let json = r#"{
        "model": { "base": "synthetic" },
        "lora": {},
        "dataset": { "path": "unused" }
    }"#;
    let config: TrainConfig =
        serde_json::from_str(json).expect("a config without task/flow blocks should deserialize");
    assert_eq!(config.task, TaskKind::Classification);
    assert_eq!(config.flow, FlowConfig::default());
    assert_eq!(config.flow.logit_mean, 0.0);
    assert_eq!(config.flow.logit_std, 1.0);
    assert_eq!(config.flow.shift, 3.0);
}

/// Backward-compat for adapters already on disk: a pre-M8 sidecar carries no
/// `task` field and must parse as a classification adapter.
#[test]
fn sidecar_without_task_field_defaults_to_classification() {
    let json = r#"{ "seed": 1, "rank": 2, "alpha": 8.0, "d_in": 8, "hidden": 6, "out": 4 }"#;
    let meta: AdapterMeta = serde_json::from_str(json).expect("a pre-M8 sidecar still parses");
    assert_eq!(meta.task, TaskKind::Classification);
}
