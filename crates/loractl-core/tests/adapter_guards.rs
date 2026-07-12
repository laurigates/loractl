//! Non-finite guards at BOTH adapter I/O boundaries (#49 H11).
//!
//! `save_adapter` refuses to persist an adapter whose LoRA weights contain
//! `NaN`/`Inf`, and `load_adapter` refuses to hand back a model reconstructed
//! from a file that contains them (`adapter.rs`'s two `all_finite` guards) —
//! but no test fed either one a non-finite weight, so a mutant deleting them
//! passed. `run_sample`'s sibling guard *is* exercised (`sample.rs`); this file
//! pins the save side and the load side.
//!
//! The load side needs a poisoned file, and `save_adapter` (correctly) will not
//! write one — so the test byte-patches a NaN/Inf into a cleanly-saved
//! `.safetensors`, which is exactly the "corrupted / hand-edited file" the load
//! guard exists for.

use burn::module::Param;
use burn::tensor::TensorData;
use loractl_core::adapter::{load_adapter, save_adapter};
use loractl_core::config::TaskKind;
use loractl_core::{Device, LoraMlp, NdArray};
use std::path::{Path, PathBuf};

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

/// Overwrite the first `f32` of `tensor` inside a `.safetensors` file with
/// `poison`, simulating a corrupted / hand-edited adapter.
///
/// safetensors layout: `[u64 LE header length][JSON header][raw tensor data]`,
/// where each entry's `data_offsets` are relative to the start of the data
/// section. Patching 4 bytes in place keeps the container structurally valid —
/// only the *value* is poisoned, which is what isolates the finite guard.
fn poison_first_element(path: &Path, tensor: &str, poison: f32) {
    let mut bytes = std::fs::read(path).expect("read the adapter file");
    let header_len =
        u64::from_le_bytes(bytes[..8].try_into().expect("8-byte header length")) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + header_len]).expect("safetensors header parses");
    let entry = &header[tensor];
    assert_eq!(
        entry["dtype"], "F32",
        "{tensor} must be F32 for a 4-byte patch to be meaningful"
    );
    let start = entry["data_offsets"][0]
        .as_u64()
        .expect("tensor is present with data_offsets") as usize;
    let at = 8 + header_len + start;
    bytes[at..at + 4].copy_from_slice(&poison.to_le_bytes());
    std::fs::write(path, bytes).expect("write the poisoned adapter file");
}

#[test]
fn load_adapter_refuses_non_finite_weights() {
    let device: Device<NdArray> = Default::default();
    let dir = TempDir::new("adapter-load-guard");

    // Both trainable tensors, both flavours of non-finite — the guard checks
    // `lora_a` AND `lora_b`, so poisoning only one would let a half-deleted
    // guard survive.
    for (label, tensor, poison) in [
        ("nan", "fc2.lora_a.weight", f32::NAN),
        ("inf", "fc2.lora_b.weight", f32::INFINITY),
    ] {
        let model = LoraMlp::<NdArray>::new(8, 6, 4, 2, 8.0, 0.0, &device);
        let path = dir.0.join(format!("{label}.safetensors"));
        save_adapter(&model, &path, 7, TaskKind::Classification).expect("the clean save succeeds");

        // Non-vacuity: the file loads fine BEFORE the patch, so the `Err` below
        // is the finite guard firing — not the byte patch having broken the
        // container out from under the loader.
        load_adapter::<NdArray>(&path, &device).expect("the clean adapter loads");

        poison_first_element(&path, tensor, poison);

        let err = load_adapter::<NdArray>(&path, &device)
            .expect_err("load_adapter must refuse a non-finite LoRA weight");
        assert!(
            err.to_string().contains("non-finite"),
            "the error must name the non-finite cause ({label} in {tensor}), got: {err}"
        );
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
