//! Non-finite guard at the adapter save boundary (#49 H11).
//!
//! `save_adapter` refuses to persist an adapter whose LoRA weights contain
//! `NaN`/`Inf` (`adapter.rs`'s `all_finite` guard) — but no test fed it a
//! non-finite weight, so a mutant deleting the guard passed. `run_sample`'s
//! sibling guard *is* exercised (`sample.rs`); this pins the save side.

use burn::module::Param;
use burn::tensor::TensorData;
use loractl_core::adapter::save_adapter;
use loractl_core::config::TaskKind;
use loractl_core::{Device, LoraMlp, NdArray};
use std::path::PathBuf;

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

#[test]
fn save_adapter_refuses_non_finite_weights() {
    let device: Device<NdArray> = Default::default();
    let mut model = LoraMlp::<NdArray>::new(8, 6, 4, 2, 8.0, 0.0, &device);

    // Simulate a diverged run: a NaN in a trainable LoRA factor.
    let [rank, out] = model.fc2.lora_b.weight.dims();
    let mut data = vec![0.0f32; rank * out];
    data[0] = f32::NAN;
    model.fc2.lora_b.weight = Param::from_data(TensorData::new(data, [rank, out]), &device);

    let dir = TempDir::new("adapter-guard");
    let path = dir.0.join("adapter.safetensors");
    let err = save_adapter(&model, &path, 0, TaskKind::Classification)
        .expect_err("save_adapter must refuse non-finite weights");
    assert!(
        err.to_string().contains("non-finite"),
        "the error must name the non-finite cause, got: {err}"
    );
    // The guard runs before any I/O, so no partial artifact is left behind.
    assert!(
        !path.exists(),
        "a rejected save must not leave a partial adapter file"
    );
}
