//! The **consumer contract** for the exported LoRA's `__metadata__` header.
//!
//! `tests/adapter_metadata.rs` pins the keys loractl *chooses* to write — a
//! self-golden, which by construction cannot disagree with itself. It says
//! nothing about whether any real consumer reads those keys, and the gap is
//! silent in the worst way: a header full of keys nobody reads loads without
//! error and shows nothing. That is the #137 failure shape one layer up from
//! tensor names (see `.claude/rules/testing.md`).
//!
//! So this test comes from the other side. It runs the **real export path**
//! and asserts every metadata key a real consumer reads is either written or
//! **deliberately** not written, with the reason recorded here. The consumer's
//! key list is generated from pinned upstream source
//! (`reference/lora_metadata_keys_reference.py`, regenerate with
//! `just lora-metadata-keys-reference`) — AUTOMATIC1111's Lora extension at
//! tag `v1.10.1`, the open-source consumer that actually parses this header
//! (ComfyUI ignores `__metadata__`; Civitai is closed). Forge and the
//! reForge/SD-Next descendants inherit that code.
//!
//! Offline and fast: the golden is checked in, and a two-delta adapter is
//! enough to exercise the writer.
//!
//! **What breaks this test, and what to do:** a new key in the golden after a
//! regeneration means the consumer started reading something we do not write
//! — either write it, or add it to [`DELIBERATELY_OMITTED`] with a reason
//! that is true. Never silence it by trimming the golden.

use burn::backend::NdArray;
use burn::module::Param;
use burn::nn::{DropoutConfig, Linear};
use burn::tensor::TensorData;
use loractl_core::LoraDelta;
use loractl_core::adapters::LoraAdapters;
use loractl_core::config::{
    BucketMode, DatasetConfig, LoraConfig, MetadataConfig, ModelConfig, ModelVariant, OptimConfig,
    OutputConfig, TargetSpec, TaskKind, TrainConfig,
};
use loractl_core::dataset::{Bucket, DatasetEntry};
use loractl_core::export::{ExportFormat, export_adapters};
use loractl_core::metadata::{DatasetFacts, LoraMetadata, RunFacts, build_metadata, read_metadata};
use serde::Deserialize;
use std::path::PathBuf;

type TB = NdArray;

/// Consumer keys loractl does **not** write, each with the reason it would be
/// wrong to. These are claims about Krea 2 and about loractl's design, not
/// conveniences — if one stops being true, the key belongs in the header.
const DELIBERATELY_OMITTED: [(&str, &str); 2] = [
    (
        "ss_clip_skip",
        "a CLIP text-encoder layer offset. Krea 2 conditions on Qwen3-VL \
         hidden states — there is no CLIP and no layer to skip, so any value \
         here would be a number a UI displays and a user acts on.",
    ),
    (
        "ss_v2",
        "sd-scripts' Stable Diffusion 2.x marker. A1111 reads it as \
         `== \"True\"`, so its absence already means 'not SD2', which is the \
         truth for a Krea 2 adapter.",
    ),
];

/// The generated contract. `ref`/`sources` are in the JSON for a human
/// reading the golden; serde ignores what is not named here.
#[derive(Deserialize)]
struct Golden {
    /// Which consumer this contract is from — quoted in failure messages so a
    /// failure names the tool whose behavior is at stake.
    consumer: String,
    /// Every metadata key that consumer reads.
    keys_read: Vec<String>,
}

fn golden() -> Golden {
    serde_json::from_str(include_str!("golden/lora_metadata_keys.json")).expect(
        "lora_metadata_keys.json parses — regenerate with `just lora-metadata-keys-reference`",
    )
}

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

fn adapters() -> LoraAdapters<TB> {
    let device = Default::default();
    let delta = LoraDelta {
        lora_a: Linear::<TB> {
            weight: Param::from_data(TensorData::new(vec![0.1f32; 8], [4, 2]), &device),
            bias: None,
        },
        lora_b: Linear::<TB> {
            weight: Param::from_data(TensorData::new(vec![0.2f32; 12], [2, 6]), &device),
            bias: None,
        },
        scaling: 4.0,
        dropout: DropoutConfig::new(0.0).init(),
    };
    LoraAdapters {
        deltas: vec![delta],
        targets: vec!["blocks.0.attn.wq".to_string()],
    }
}

/// A fully-populated run — every optional field set, so a key missing from the
/// export is missing because loractl never writes it, not because this config
/// left it unconfigured.
fn config() -> TrainConfig {
    TrainConfig {
        steps: 100,
        seed: 1,
        task: TaskKind::FlowMatching,
        model: ModelConfig {
            base: "/models/Krea-2-Raw".into(),
            variant: ModelVariant::Krea2,
            checkpoint: None,
            denoiser: None,
            text_encoder: None,
            vae: None,
            tokenizer: None,
            training_adapter: None,
        },
        lora: LoraConfig {
            rank: 16,
            alpha: 16.0,
            dropout: 0.0,
            targets: vec![TargetSpec {
                pattern: "blocks\\.".into(),
                rank: None,
                alpha: None,
            }],
        },
        dataset: DatasetConfig {
            path: PathBuf::from("/data/subject"),
            resolution: 512,
            batch_size: 1,
            no_upscale: false,
            bucketing: BucketMode::Aspects,
            min_bucket_resolution: None,
        },
        optim: OptimConfig {
            lr: 1e-4,
            weight_decay: 0.0,
        },
        output: OutputConfig {
            dir: PathBuf::from("output"),
            name: "subject".into(),
            checkpoint_every: 50,
            sample_every: 0,
        },
        resume: Default::default(),
        compute: Default::default(),
        flow: Default::default(),
        metadata: MetadataConfig {
            embed: true,
            trigger_words: vec!["sks subject".into()],
            title: Some("Subject".into()),
            description: Some("desc".into()),
            author: Some("author".into()),
            license: Some("apache-2.0".into()),
            tags: vec!["tag".into()],
            comment: Some("comment".into()),
        },
    }
}

const BUCKETS: [Bucket; 1] = [Bucket {
    width: 512,
    height: 512,
}];

/// **The contract itself**, as one function: every key the consumer reads
/// that `written` does not carry and that is not a recorded omission.
///
/// Both the contract test and its kill-test call this rather than each
/// spelling the rule out — a duplicated rule is one the sabotage can drift
/// away from, leaving a kill-test that proves only that its own copy still
/// works. (Same shape as `krea2_lora_keys.rs`.)
fn unwritten_keys<'a>(written: &LoraMetadata, golden: &'a Golden) -> Vec<&'a str> {
    golden
        .keys_read
        .iter()
        .map(String::as_str)
        .filter(|key| written.get(key).is_none())
        .filter(|key| {
            !DELIBERATELY_OMITTED
                .iter()
                .any(|(omitted, _)| omitted == key)
        })
        .collect()
}

#[test]
fn every_key_the_consumer_reads_is_written_or_deliberately_omitted() {
    let golden = golden();
    let config = config();
    let entries = vec![DatasetEntry {
        image_path: PathBuf::from("/data/subject/a.png"),
        caption: "sks subject, studio".into(),
        bucket: 0,
    }];
    let meta = build_metadata(&RunFacts {
        config: &config,
        steps_done: 100,
        dataset: Some(DatasetFacts {
            name: "subject".into(),
            entries: &entries,
            buckets: &BUCKETS,
            batches_per_epoch: 1,
        }),
        base_model_file: Some("raw.safetensors".into()),
        started_at: Some(1_785_069_296),
        finished_at: Some(1_785_069_356),
    });

    // Read the keys back off a REAL export, not from the builder's return
    // value: `sshs_model_hash` is added by the writer, so a builder-only
    // check would report the consumer's identity key as unwritten.
    let out = TempDir::new("metadata-consumer");
    let path = out.0.join("lora.safetensors");
    export_adapters(
        &adapters(),
        ExportFormat::Krea2Diffusers,
        Some(&meta),
        &path,
    )
    .expect("export succeeds");
    let written = read_metadata(&path).expect("the export carries a header");

    assert!(
        !golden.keys_read.is_empty(),
        "the golden is empty — regenerate it; an empty contract passes vacuously"
    );

    let unwritten = unwritten_keys(&written, &golden);
    assert!(
        unwritten.is_empty(),
        "{} reads these metadata keys, which loractl neither writes nor \
         records a reason for omitting: {unwritten:?}. Write them, or add \
         them to DELIBERATELY_OMITTED with a reason that is true.",
        golden.consumer
    );
}

/// The omission list must not rot into a dumping ground: every entry has to
/// still be a key the consumer actually reads, or it is dead weight asserting
/// nothing.
#[test]
fn every_omission_is_still_a_key_the_consumer_reads() {
    let golden = golden();
    for (key, _) in DELIBERATELY_OMITTED {
        assert!(
            golden.keys_read.iter().any(|k| k == key),
            "{key} is recorded as a deliberate omission, but {} no longer \
             reads it — drop the entry",
            golden.consumer
        );
    }
}

/// Teeth for the contract above: it must be able to FAIL. A header that is
/// missing a key the consumer reads, and has no recorded reason, has to be
/// detected — otherwise the assertion is decorative.
#[test]
fn the_contract_detects_a_missing_key() {
    let golden = golden();
    // The metadata-free export is the sabotage: no header at all.
    let out = TempDir::new("metadata-consumer-kill");
    let path = out.0.join("bare.safetensors");
    export_adapters(&adapters(), ExportFormat::Krea2Diffusers, None, &path).expect("export");
    let written = read_metadata(&path).expect("reads");

    // Guards the subtraction below, which would otherwise underflow and
    // panic on a trimmed golden instead of reporting why.
    assert!(
        golden.keys_read.len() > DELIBERATELY_OMITTED.len(),
        "the golden has no non-omitted keys left — regenerate it; there is \
         nothing for this test to detect"
    );
    // Deliberately the SAME function the contract test calls, so sabotage
    // exercises the real rule rather than a copy of it.
    let unwritten = unwritten_keys(&written, &golden);
    assert_eq!(
        unwritten.len(),
        golden.keys_read.len() - DELIBERATELY_OMITTED.len(),
        "with no header written, EVERY non-omitted consumer key must come \
         back missing — if this count is 0, the check above passes vacuously"
    );
}
