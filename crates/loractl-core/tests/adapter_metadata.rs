//! The exported LoRA's `__metadata__` header (#154) — what a UI reads back.
//!
//! Three claims, each with its own teeth:
//!
//! 1. **The builder derives the right facts.** `build_metadata` turns a
//!    [`TrainConfig`] plus a scanned dataset into the `ss_*`/`modelspec.*`
//!    keys — trigger words, tag frequency, bucket info, network topology,
//!    optimizer, epochs — in the JSON-in-a-string shapes kohya's consumers
//!    parse.
//! 2. **The header survives the round trip.** Every key the builder produced
//!    is readable back off the on-disk file, and the tensors are untouched by
//!    its presence: a metadata-carrying export is byte-for-byte the same
//!    weights, and `import_adapters` (the resume path) still reads it.
//! 3. **The hashes are the ones a consumer computes.** `sshs_model_hash` is
//!    recomputed here **independently** — SHA-256 over the finished file's
//!    tensor-data region, which is exactly what
//!    sd-webui-additional-networks' `addnet_hash_safetensors` does — rather
//!    than compared against a golden that the exporter itself produced.
//!
//! Offline, no fixtures, milliseconds.

use burn::backend::NdArray;
use burn::module::Param;
use burn::nn::{DropoutConfig, Linear};
use burn::tensor::TensorData;
use loractl_core::LoraDelta;
use loractl_core::adapters::LoraAdapters;
use loractl_core::config::{
    ComputeConfig, DatasetConfig, FlowConfig, LoraConfig, MetadataConfig, ModelConfig,
    ModelVariant, OptimConfig, OutputConfig, Precision, Quant, ShiftMode, TargetSpec, TaskKind,
    TrainConfig,
};
use loractl_core::dataset::{Bucket, DatasetEntry};
use loractl_core::export::{ExportFormat, export_adapters, import_adapters};
use loractl_core::metadata::{DatasetFacts, LoraMetadata, RunFacts, build_metadata, read_metadata};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

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

/// A two-delta adapter set with fixed weights (no RNG — every byte of the
/// export is a function of the inputs).
fn adapters() -> LoraAdapters<TB> {
    let device = Default::default();
    let mut deltas = Vec::new();
    for scale in [1.0f32, -1.0] {
        let a: Vec<f32> = (0..8).map(|i| scale * i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..12).map(|i| scale * i as f32 * -0.02).collect();
        deltas.push(LoraDelta {
            lora_a: Linear::<TB> {
                weight: Param::from_data(TensorData::new(a, [4, 2]), &device),
                bias: None,
            },
            lora_b: Linear::<TB> {
                weight: Param::from_data(TensorData::new(b, [2, 6]), &device),
                bias: None,
            },
            scaling: 8.0 / 2.0,
            dropout: DropoutConfig::new(0.0).init(),
        });
    }
    LoraAdapters {
        deltas,
        targets: vec![
            "blocks.0.attn.wq".to_string(),
            "blocks.1.mlp.up".to_string(),
        ],
    }
}

/// A config exercising every metadata-bearing knob.
fn config() -> TrainConfig {
    TrainConfig {
        steps: 10,
        seed: 7,
        task: TaskKind::FlowMatching,
        model: ModelConfig {
            base: "/models/Krea-2-Turbo".into(),
            variant: ModelVariant::Krea2Turbo,
            checkpoint: None,
            denoiser: None,
            text_encoder: None,
            vae: None,
            tokenizer: None,
            training_adapter: Some(PathBuf::from(
                "/adapters/krea2_turbo_training_adapter.safetensors",
            )),
        },
        lora: LoraConfig {
            rank: 16,
            alpha: 8.0,
            dropout: 0.0,
            targets: vec![TargetSpec {
                pattern: "blocks\\..*attn".to_string(),
                rank: Some(32),
                alpha: None,
            }],
        },
        dataset: DatasetConfig {
            path: PathBuf::from("/data/sks-dog"),
            resolution: 512,
            batch_size: 2,
        },
        optim: OptimConfig {
            lr: 1e-4,
            weight_decay: 0.01,
        },
        output: OutputConfig {
            dir: PathBuf::from("output"),
            name: "sks-dog".into(),
            checkpoint_every: 5,
            sample_every: 0,
        },
        resume: Default::default(),
        compute: ComputeConfig {
            precision: Precision::F32,
            quant: Quant::Int4,
            grad_checkpointing: true,
            ..ComputeConfig::default()
        },
        flow: FlowConfig {
            shift_mode: ShiftMode::Resolution,
            ..FlowConfig::default()
        },
        metadata: MetadataConfig {
            embed: true,
            trigger_words: vec!["sks dog".into(), "in the style of sks".into()],
            title: Some("SKS Dog".into()),
            description: Some("A dog LoRA".into()),
            author: Some("someone".into()),
            license: Some("apache-2.0".into()),
            tags: vec!["dog".into(), "pet".into()],
            comment: Some("second attempt".into()),
        },
    }
}

/// Three images across two buckets, with comma-separated captions.
fn entries() -> Vec<DatasetEntry> {
    vec![
        DatasetEntry {
            image_path: PathBuf::from("/data/sks-dog/a.png"),
            caption: "sks dog, park, sunny".into(),
            bucket: 0,
        },
        DatasetEntry {
            image_path: PathBuf::from("/data/sks-dog/b.png"),
            caption: "sks dog,  park ".into(),
            bucket: 0,
        },
        DatasetEntry {
            image_path: PathBuf::from("/data/sks-dog/c.png"),
            caption: "sks dog, indoors".into(),
            bucket: 1,
        },
    ]
}

const BUCKETS: [Bucket; 3] = [
    Bucket {
        width: 512,
        height: 512,
    },
    Bucket {
        width: 576,
        height: 448,
    },
    // Deliberately empty — no entry lands here.
    Bucket {
        width: 448,
        height: 576,
    },
];

/// Build the metadata for `config`, with fixed timestamps so assertions are
/// deterministic.
fn metadata_for(config: &TrainConfig, entries: &[DatasetEntry]) -> LoraMetadata {
    build_metadata(&RunFacts {
        config,
        steps_done: 10,
        dataset: Some(DatasetFacts {
            name: "sks-dog".into(),
            entries,
            buckets: &BUCKETS,
            batches_per_epoch: 3,
        }),
        base_model_file: Some("turbo.safetensors".into()),
        started_at: Some(1_785_069_296),
        finished_at: Some(1_785_069_356),
    })
}

/// Parse a metadata value that is itself JSON (kohya's convention).
fn nested(meta: &LoraMetadata, key: &str) -> Value {
    let raw = meta
        .get(key)
        .unwrap_or_else(|| panic!("metadata has no {key}"));
    serde_json::from_str(raw).unwrap_or_else(|e| panic!("{key} is not JSON ({e}): {raw}"))
}

#[test]
fn builder_derives_the_training_record() {
    let config = config();
    let entries = entries();
    let meta = metadata_for(&config, &entries);

    // Trigger words: the full list under kohya's key, the first under
    // ModelSpec's single-string one.
    assert_eq!(
        nested(&meta, "ss_trained_words"),
        serde_json::json!(["sks dog", "in the style of sks"])
    );
    assert_eq!(meta.get("modelspec.trigger_phrase"), Some("sks dog"));

    // Network topology. `alpha` renders as `8`, not `8.0` — a config value a
    // reader can paste back.
    assert_eq!(meta.get("ss_network_module"), Some("networks.lora"));
    assert_eq!(meta.get("ss_network_dim"), Some("16"));
    assert_eq!(meta.get("ss_network_alpha"), Some("8"));
    assert_eq!(meta.get("ss_network_dropout"), Some("0"));
    assert_eq!(
        nested(&meta, "ss_network_args"),
        serde_json::json!({"targets": [{"pattern": "blocks\\..*attn", "rank": 32}]}),
        "per-target overrides ride in kohya's network-args slot"
    );

    // Base model + optimization.
    assert_eq!(meta.get("ss_base_model_version"), Some("krea-2-turbo"));
    assert_eq!(
        meta.get("modelspec.architecture"),
        Some("krea-2-turbo/lora")
    );
    assert_eq!(meta.get("ss_sd_model_name"), Some("turbo.safetensors"));
    assert_eq!(
        meta.get("ss_training_adapter"),
        Some("krea2_turbo_training_adapter.safetensors"),
        "the merged assistant LoRA is part of what produced these weights"
    );
    assert_eq!(
        meta.get("ss_optimizer"),
        Some("burn.optim.AdamW(weight_decay=0.01)")
    );
    assert_eq!(meta.get("ss_learning_rate"), Some("0.0001"));
    assert_eq!(meta.get("ss_unet_lr"), Some("0.0001"));
    assert_eq!(meta.get("ss_text_encoder_lr"), Some("0.0"));
    assert_eq!(meta.get("ss_max_train_steps"), Some("10"));
    assert_eq!(meta.get("ss_seed"), Some("7"));
    assert_eq!(meta.get("ss_mixed_precision"), Some("no"));
    assert_eq!(meta.get("ss_gradient_checkpointing"), Some("True"));
    assert_eq!(meta.get("ss_base_model_quant"), Some("int4"));

    // Dataset: images, buckets, tags, epochs.
    assert_eq!(meta.get("ss_num_train_images"), Some("3"));
    assert_eq!(meta.get("ss_resolution"), Some("(512, 512)"));
    assert_eq!(meta.get("ss_batch_size_per_device"), Some("2"));
    assert_eq!(
        meta.get("ss_num_epochs"),
        Some("4"),
        "10 steps over 3 batches/epoch touches 4 epochs (ceiling, as kohya counts)"
    );
    assert_eq!(
        nested(&meta, "ss_bucket_info"),
        serde_json::json!({
            "buckets": {
                "0": {"resolution": [512, 512], "count": 2},
                "1": {"resolution": [576, 448], "count": 1},
            }
        }),
        "empty buckets are not part of this run"
    );
    assert_eq!(
        nested(&meta, "ss_tag_frequency"),
        serde_json::json!({"sks-dog": {"sks dog": 3, "park": 2, "sunny": 1, "indoors": 1}}),
        "captions split on commas and trim, keyed by subset dir — kohya's shape"
    );
    assert_eq!(
        nested(&meta, "ss_dataset_dirs"),
        serde_json::json!({"sks-dog": {"n_repeats": 1, "img_count": 3}}),
        "A1111's metadata panel reads this; loractl has no repeat mechanism, \
         so n_repeats is truthfully 1"
    );

    // The resolution-dependent shift (#84) has no single value to report, so
    // its anchors are what a reader needs instead.
    assert!(
        meta.get("ss_discrete_flow_shift").is_none(),
        "a constant shift would be a lie under shift_mode: resolution"
    );
    assert_eq!(
        nested(&meta, "ss_dynamic_shift")["max_shift"],
        serde_json::json!(1.15)
    );

    // Provenance.
    assert_eq!(meta.get("modelspec.title"), Some("SKS Dog"));
    assert_eq!(meta.get("modelspec.author"), Some("someone"));
    assert_eq!(meta.get("modelspec.license"), Some("apache-2.0"));
    assert_eq!(meta.get("modelspec.tags"), Some("dog,pet"));
    assert_eq!(meta.get("modelspec.date"), Some("2026-07-26T12:35:56Z"));
    assert_eq!(meta.get("ss_training_started_at"), Some("1785069296"));
    assert_eq!(meta.get("ss_training_comment"), Some("second attempt"));

    // Deliberate omissions (see the module docs) — a UI acting on a
    // fabricated value is worse than one showing nothing.
    for absent in [
        "ss_clip_skip",
        "civitai_model_id",
        "civitai_version_id",
        "ss_datasets",
    ] {
        assert!(meta.get(absent).is_none(), "{absent} must not be written");
    }
}

/// Non-whole `f32` config values must reach the header at **f32 precision**.
///
/// `num(v as f64)` would widen first and print the f64's shortest decimal —
/// `alpha: 12.8` landing as `12.800000190734863`, `dropout: 0.05` as
/// `0.05000000074505806`. Every other test here uses whole values (8.0, 0.0),
/// which are byte-identical under both spellings, so this is the only case
/// that can tell the two apart — the defaults-hide-the-bug shape
/// `.claude/rules/burn-optimizer-and-dropout.md` describes. The same trap
/// exists one layer down in `serde_json`'s `From<f32>` (it stores `f as f64`),
/// which is why the per-target alpha is checked too.
#[test]
fn non_exact_f32_config_values_keep_f32_precision() {
    let mut config = config();
    config.lora.alpha = 12.8;
    config.lora.dropout = 0.05;
    config.lora.targets[0].alpha = Some(2.7);
    let entries = entries();
    let meta = metadata_for(&config, &entries);

    assert_eq!(meta.get("ss_network_alpha"), Some("12.8"));
    assert_eq!(meta.get("ss_network_dropout"), Some("0.05"));
    assert_eq!(
        nested(&meta, "ss_network_args")["targets"][0]["alpha"].to_string(),
        "2.7",
        "the JSON number must render at f32 precision too"
    );
}

#[test]
fn constant_shift_mode_reports_its_shift() {
    let mut config = config();
    config.flow.shift_mode = ShiftMode::Constant;
    config.flow.shift = 3.0;
    let entries = entries();
    let meta = metadata_for(&config, &entries);
    assert_eq!(meta.get("ss_discrete_flow_shift"), Some("3"));
    assert!(meta.get("ss_dynamic_shift").is_none());
}

#[test]
fn defaults_omit_optional_fields_rather_than_writing_blanks() {
    let mut config = config();
    config.metadata = MetadataConfig::default();
    config.model.training_adapter = None;
    config.compute.quant = Quant::None;
    let entries = entries();
    let meta = metadata_for(&config, &entries);

    for absent in [
        "ss_trained_words",
        "modelspec.trigger_phrase",
        "modelspec.author",
        "modelspec.license",
        "modelspec.description",
        "modelspec.tags",
        "ss_training_comment",
        "ss_training_adapter",
        "ss_base_model_quant",
    ] {
        assert!(
            meta.get(absent).is_none(),
            "{absent} should be omitted, not blank"
        );
    }
    // The derived record still lands — a user who configures nothing still
    // ships a LoRA that says what it is.
    assert_eq!(meta.get("ss_network_dim"), Some("16"));
    assert_eq!(meta.get("ss_base_model_version"), Some("krea-2-turbo"));
    assert_eq!(
        meta.get("modelspec.title"),
        Some("sks-dog"),
        "title falls back to the output name"
    );
}

/// SHA-256 over a safetensors file's **tensor-data region** — an independent
/// reimplementation of sd-webui-additional-networks' `addnet_hash_safetensors`
/// (which the exporter must agree with), computed here from the *finished
/// file* exactly as a consumer would.
fn data_region_sha256(path: &Path) -> String {
    let bytes = std::fs::read(path).expect("read export");
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut hasher = Sha256::new();
    hasher.update(&bytes[8 + n..]);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn header_round_trips_and_carries_consumer_hashes() {
    let config = config();
    let entries = entries();
    let meta = metadata_for(&config, &entries);
    let set = adapters();

    let out = TempDir::new("adapter-metadata");
    let path = out.0.join("lora.safetensors");
    export_adapters(&set, ExportFormat::Krea2Diffusers, Some(&meta), &path).expect("export");

    // Header-only read: every built key comes back verbatim.
    let read = read_metadata(&path).expect("metadata reads back");
    for (key, value) in meta.iter() {
        assert_eq!(read.get(key), Some(value), "{key} did not round-trip");
    }

    // …plus the two hashes, which only exist once the tensors are laid out.
    let model_hash = read
        .get("sshs_model_hash")
        .expect("sshs_model_hash written");
    assert_eq!(model_hash.len(), 64, "a full SHA-256 hex digest");
    assert_eq!(
        model_hash,
        data_region_sha256(&path),
        "sshs_model_hash must be what a consumer recomputes from the finished \
         file — the whole point of the key"
    );
    assert_eq!(
        read.get("sshs_legacy_hash").expect("legacy hash").len(),
        8,
        "the legacy hash is the truncated form sd-webui indexes by"
    );
    assert!(
        meta.get("sshs_model_hash").is_none(),
        "the builder must not invent a hash it cannot compute"
    );
}

#[test]
fn metadata_does_not_disturb_the_tensors() {
    let config = config();
    let entries = entries();
    let meta = metadata_for(&config, &entries);
    let set = adapters();

    let out = TempDir::new("adapter-metadata-tensors");
    let with = out.0.join("with.safetensors");
    let without = out.0.join("without.safetensors");
    export_adapters(&set, ExportFormat::Krea2Diffusers, Some(&meta), &with).expect("export");
    export_adapters(&set, ExportFormat::Krea2Diffusers, None, &without).expect("export");

    let with_bytes = std::fs::read(&with).unwrap();
    let without_bytes = std::fs::read(&without).unwrap();
    let a = safetensors::SafeTensors::deserialize(&with_bytes).expect("parses");
    let b = safetensors::SafeTensors::deserialize(&without_bytes).expect("parses");

    let mut a_names: Vec<&str> = a.names();
    let mut b_names: Vec<&str> = b.names();
    a_names.sort_unstable();
    b_names.sort_unstable();
    assert_eq!(a_names, b_names, "the same six tensors either way");
    for name in a_names {
        assert_eq!(
            a.tensor(name).unwrap().data(),
            b.tensor(name).unwrap().data(),
            "{name} differs — the header must not touch the weights"
        );
    }

    // The data region is byte-identical, which is *why* `sshs_model_hash` is
    // recomputable from a file carrying more metadata than was hashed.
    assert_eq!(data_region_sha256(&with), data_region_sha256(&without));

    // And the resume path still reads a metadata-carrying export.
    let mut resumed = adapters();
    for delta in &mut resumed.deltas {
        delta.lora_a.weight = Param::from_data(TensorData::new(vec![0.0f32; 8], [4, 2]), &{
            Default::default()
        });
    }
    let loaded = import_adapters(&mut resumed, ExportFormat::Krea2Diffusers, &with)
        .expect("import from a metadata-carrying export");
    assert_eq!(loaded, 2);
    resumed.deltas[0]
        .lora_a
        .weight
        .val()
        .into_data()
        .assert_eq(&set.deltas[0].lora_a.weight.val().into_data(), true);
}

#[test]
fn embed_false_writes_no_header_at_all() {
    let mut config = config();
    config.metadata.embed = false;
    let entries = entries();
    let meta = metadata_for(&config, &entries);
    assert!(meta.is_empty(), "the opt-out short-circuits the builder");

    let set = adapters();
    let out = TempDir::new("adapter-metadata-optout");
    let opted_out = out.0.join("opted-out.safetensors");
    let none = out.0.join("none.safetensors");
    export_adapters(&set, ExportFormat::Krea2Diffusers, Some(&meta), &opted_out).expect("export");
    export_adapters(&set, ExportFormat::Krea2Diffusers, None, &none).expect("export");

    assert!(
        read_metadata(&opted_out).expect("reads").is_empty(),
        "an empty map must write no __metadata__, not an empty one"
    );
    assert_eq!(
        std::fs::read(&opted_out).unwrap(),
        std::fs::read(&none).unwrap(),
        "the opt-out is byte-identical to a metadata-free export — that is \
         what makes it a reproducible-build knob"
    );
}

#[test]
fn read_metadata_rejects_non_safetensors_and_tolerates_headerless_files() {
    let out = TempDir::new("adapter-metadata-read");
    std::fs::create_dir_all(&out.0).unwrap();

    // A file with tensors but no `__metadata__` — what diffusers scripts and
    // minimal trainers write. Not an error; just nothing to report.
    let bare = out.0.join("bare.safetensors");
    export_adapters(&adapters(), ExportFormat::KohyaSs, None, &bare).expect("export");
    assert!(read_metadata(&bare).expect("reads").is_empty());

    // Garbage must fail loudly rather than yield an empty map that reads as
    // "this LoRA has no trigger words".
    let junk = out.0.join("junk.safetensors");
    std::fs::write(&junk, b"not a safetensors file at all").unwrap();
    assert!(read_metadata(&junk).is_err());

    let truncated = out.0.join("truncated.safetensors");
    std::fs::write(&truncated, b"abc").unwrap();
    assert!(read_metadata(&truncated).is_err());

    // A file that lies about its header length with a PLAUSIBLE number. This
    // is the only case that reaches the file-size bound: `junk` above trips
    // the 100 MB ceiling first (its first 8 bytes read as an astronomically
    // large LE u64), so without this the size check would be dead code that
    // still looks covered.
    let lying = out.0.join("lying.safetensors");
    let mut bytes = 500u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}"); // claims a 500-byte header; the file is 10
    std::fs::write(&lying, bytes).unwrap();
    let err = read_metadata(&lying)
        .expect_err("a header longer than the file must be rejected")
        .to_string();
    // Match the PHRASE, not the two numbers: the message begins with the temp
    // path, whose nanosecond timestamp contains "10" nearly always and "500"
    // now and then — so a digits-only check could pass on the `read_exact`
    // error that would replace this one if the bound were removed.
    assert!(
        err.contains("claims 500 bytes but the file is 10"),
        "the error must come from the size bound, naming both sizes: {err}"
    );
}
