//! The **consumer contract** for the Krea 2 adapter export (#137).
//!
//! `tests/adapter_export.rs` proves the export matches *our own* golden — it
//! pins our convention, so it cannot tell us whether the real consumer accepts
//! it. That gap is what made #137 possible: the export was believed broken
//! because its key names differ from community LoRAs, and nothing in the repo
//! could settle the question mechanically. (It was not broken — ComfyUI accepts
//! both forms — but the belief cost a full re-diagnosis.)
//!
//! This test closes the gap from the other side. It runs the **real export
//! path** over the **real Krea 2 site enumeration** and asserts every key that
//! lands on disk is one ComfyUI's LoRA key map actually contains, per a golden
//! generated from pinned upstream ComfyUI source
//! (`reference/krea2_lora_keys_reference.py`, regenerate with
//! `just krea2-lora-keys-reference`).
//!
//! Offline and fast: `build_adapters` is config-derived, so the full 196-site
//! set is built from `MmditConfig::krea2()` without instantiating the ~12.8B
//! model, and the LoRA factors are rank-4 slivers.
//!
//! **What breaks this test, and what to do:** if upstream ComfyUI drops the
//! bare `key_map[key_lora] = to` alias in `model_lora_keys_unet`'s Krea2
//! branch, the *reference script* fails first (it asserts that line is present
//! before emitting a golden). If it is gone, `export.rs` must switch to a
//! surviving alias — the native `diffusion_model.blocks.N.*` form is the
//! obvious one, and is what #137 originally proposed.

use burn::backend::NdArray;
use loractl_core::adapters::build_adapters;
use loractl_core::config::{LoraConfig, TargetSpec};
use loractl_core::export::{ExportFormat, export_adapters};
use loractl_core::mmdit::MmditConfig;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::path::PathBuf;

type TB = NdArray;

/// A unique temp dir, removed on drop (same convention as the other tests).
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

/// The LoRA suffixes `export_adapters` writes, and that ComfyUI's weight
/// adapters look for on top of a bare key-map entry.
const SUFFIXES: [&str; 3] = [".lora_down.weight", ".lora_up.weight", ".alpha"];

/// Krea 2's trunk: 28 blocks x 7 injectable projections.
const EXPECTED_SITES: usize = 196;

#[derive(Deserialize)]
struct Golden {
    comfyui_commit: String,
    layers: usize,
    accepted_keys: Vec<String>,
}

fn golden() -> Golden {
    serde_json::from_str(include_str!("golden/krea2_lora_keys.json"))
        .expect("krea2_lora_keys.json parses — regenerate with `just krea2-lora-keys-reference`")
}

/// Export the full Krea 2 adapter set and return the on-disk tensor keys.
fn exported_keys(dir: &std::path::Path) -> Vec<String> {
    let sites = MmditConfig::krea2().injectable_sites();
    assert_eq!(
        sites.len(),
        EXPECTED_SITES,
        "Krea 2 site enumeration changed; the contract below covers whatever \
         `injectable_sites` advertises, but a surprise here means the model \
         changed shape — re-read `MmditConfig::injectable_sites`"
    );

    let cfg = LoraConfig {
        rank: 4, // key names are rank-independent; keep the factors tiny
        alpha: 8.0,
        dropout: 0.0,
        targets: vec![TargetSpec {
            pattern: ".*".to_string(),
            rank: None,
            alpha: None,
        }],
    };
    let set = build_adapters::<TB>(&sites, &cfg, &Default::default());
    assert_eq!(set.deltas.len(), EXPECTED_SITES, "every site gets a delta");

    let path = dir.join("krea2-contract.safetensors");
    export_adapters(&set, ExportFormat::Krea2Diffusers, &path).expect("export succeeds");

    let bytes = std::fs::read(&path).expect("export is readable");
    let st = SafeTensors::deserialize(&bytes).expect("export parses");
    st.names().into_iter().map(str::to_string).collect()
}

/// Strip a known LoRA suffix, yielding the site key ComfyUI matches on.
fn site_of(key: &str) -> &str {
    for suffix in SUFFIXES {
        if let Some(base) = key.strip_suffix(suffix) {
            return base;
        }
    }
    panic!("exported key {key:?} has none of the expected LoRA suffixes {SUFFIXES:?}");
}

/// Every exported key must be one ComfyUI's Krea 2 key map accepts.
///
/// This is the assertion #137 needed and did not have.
#[test]
fn every_exported_key_is_accepted_by_comfyui() {
    let g = golden();
    assert_eq!(
        g.layers,
        MmditConfig::krea2().layers,
        "golden was generated for a different trunk depth than krea2() has — \
         regenerate with `just krea2-lora-keys-reference`"
    );
    let accepted: std::collections::HashSet<&str> =
        g.accepted_keys.iter().map(String::as_str).collect();

    let dir = TempDir::new("krea2-keys");
    let keys = exported_keys(&dir.0);
    assert_eq!(
        keys.len(),
        EXPECTED_SITES * SUFFIXES.len(),
        "expected down/up/alpha per site"
    );

    let unmatched: Vec<&String> = keys
        .iter()
        .filter(|k| !accepted.contains(site_of(k)))
        .collect();

    assert!(
        unmatched.is_empty(),
        "{} of {} exported keys are NOT in ComfyUI's Krea 2 key map (commit {}). \
         A LoRA with unmatched keys loads WITHOUT ERROR and silently does nothing \
         — the worst failure shape, which is exactly why this test exists.\n\
         First few unmatched: {:?}",
        unmatched.len(),
        keys.len(),
        g.comfyui_commit,
        &unmatched[..unmatched.len().min(5)],
    );
}

/// Teeth for the assertion above: the *un-renamed* native path must NOT be
/// accepted as a bare key.
///
/// Without this, a mapper regressed to the identity function would still pass
/// the test above if the golden happened to contain both forms bare. It does
/// not — ComfyUI registers the native form only under the `diffusion_model.`
/// prefix — and this test pins that, so the contract test above has real
/// discriminating power rather than passing vacuously.
#[test]
fn bare_native_path_is_not_accepted() {
    let g = golden();
    let accepted: std::collections::HashSet<&str> =
        g.accepted_keys.iter().map(String::as_str).collect();

    // What an identity (broken) mapper would emit.
    assert!(
        !accepted.contains("blocks.0.attn.wq"),
        "bare native paths are unexpectedly accepted — the contract test above \
         would no longer catch a mapper regressed to the identity function"
    );
    // The same site under the two forms that ARE accepted, as documentation:
    // the diffusers key loractl emits, and the native key community LoRAs use.
    assert!(
        accepted.contains("transformer_blocks.0.attn.to_q"),
        "the bare diffusers key loractl emits must be accepted"
    );
    assert!(
        accepted.contains("diffusion_model.blocks.0.attn.wq"),
        "the native key community LoRAs use must also be accepted — both forms \
         are valid, which is the finding that closed #137"
    );
}
