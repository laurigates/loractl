//! Pins the shipped default example config (review): `just train` defaults
//! to `config/examples/lora.yaml`, documented as the offline synthetic demo
//! — an M14 factory change once routed it to the diffusion trainer, which
//! bails on the default classification task, breaking the documented entry
//! point with the whole suite green. This test parses the real file the way
//! `load_config` does and drives it through core's `select_trainer`.

use figment::Figment;
use figment::providers::{Format, Yaml};
use loractl_core::{TrainConfig, TrainEvent, select_trainer};
use std::path::PathBuf;

/// A unique temp output dir, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "loractl-example-config-{}-{nanos}",
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

#[test]
fn the_example_config_runs_the_synthetic_demo() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../config/examples/lora.yaml"
    );
    let mut config: TrainConfig = Figment::new()
        .merge(Yaml::file(path))
        .extract()
        .expect("config/examples/lora.yaml parses into TrainConfig");
    assert_eq!(
        config.model.base, "synthetic",
        "the shipped default example must stay on the demo trainer"
    );

    // Shrink the run the way CLI flag overrides would, and keep all writes
    // in a temp dir; everything else runs exactly as shipped.
    let out = TempDir::new();
    config.steps = 2;
    config.output.dir = out.0.clone();
    config.output.checkpoint_every = 10_000;

    let mut steps = 0u64;
    select_trainer(&config)
        .train(&config, &mut |event| {
            if let TrainEvent::Step { .. } = event {
                steps += 1;
            }
        })
        .expect("the default example config must run the synthetic demo");
    assert_eq!(steps, config.steps, "one Step per step through the demo");
}
