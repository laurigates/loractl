//! The exported LoRA's `__metadata__` header — trigger words, dataset tags,
//! and the training record every ecosystem UI reads back.
//!
//! A `.safetensors` file begins with an 8-byte little-endian header length,
//! then a JSON header, then the tensor data. The header may carry a
//! `__metadata__` object of **string → string** pairs, so a trainer can embed
//! provenance in the artifact itself: no sidecar to lose, no cost to read (a
//! consumer parses a few kB and never touches the weights).
//!
//! loractl writes that header on every interop export ([`crate::export`]) —
//! the checkpoints and the final adapter. Two vocabularies, both of which the
//! ecosystem already reads:
//!
//! | prefix | who reads it | what we write |
//! |---|---|---|
//! | `ss_*` | kohya-ss / sd-scripts convention; A1111, Forge, ComfyUI tag auto-complete, Civitai | the training record: network topology, optimizer, dataset buckets, [`ss_tag_frequency`](build_metadata) |
//! | `modelspec.*` | Stability AI's [ModelSpec] schema | title/author/license/date/architecture/`trigger_phrase` |
//! | `sshs_*` | sd-webui-additional-networks | the two file hashes (see [`crate::export`]) |
//!
//! ## Which consumer reads what (verified, 2026-07-25)
//!
//! An interop key nobody reads is indistinguishable from one nobody wrote,
//! and the failure is silent — the file loads and shows nothing (the #137
//! shape; see `.claude/rules/testing.md`). So the contract is **pinned
//! mechanically**, not just described: `tests/lora_metadata_keys.rs` asserts
//! every key a real consumer reads is either written here or carries a
//! recorded reason for its absence, against a golden generated from that
//! consumer's own source at a pinned tag
//! (`reference/lora_metadata_keys_reference.py`, `just
//! lora-metadata-keys-reference`).
//!
//! The consumer is AUTOMATIC1111's Lora extension at tag `v1.10.1`
//! (`extensions-builtin/Lora/`) — the open-source tool that actually parses
//! this header. ComfyUI ignores `__metadata__` entirely (its contract is the
//! *tensor* keys, pinned by `tests/krea2_lora_keys.rs`) and Civitai is closed
//! source, so neither can pin anything here. What that consumer does with
//! each key:
//!
//! | key | read by |
//! |---|---|
//! | `ss_tag_frequency` | `ui_edit_user_metadata.py::build_tags` — ranks the tags and offers them for the LoRA's activation text. **This is the load-bearing trigger-word path**: A1111's own activation text comes from its user-metadata JSON, not from the file, and this is what populates the suggestions. |
//! | `sshs_model_hash` | `network.py` — the file's identity |
//! | `ss_output_name` | `network.py` — the alias shown in the UI |
//! | `ss_base_model_version`, `ss_v2` | `network.py` — SD-version detection (both prefix/equality tests we correctly do not match: Krea 2 is neither SDXL nor SD2, and `ss_v2` is left unwritten) |
//! | `ss_sd_model_name`, `ss_clip_skip`, `ss_network_module`, `ss_training_started_at`, `ss_bucket_info`, `ss_dataset_dirs`, `ss_resolution`, `ss_num_train_images` | `ui_edit_user_metadata.py::get_metadata_table` / `network.py`'s `metadata_tags_order` — the metadata panel |
//!
//! `ss_trained_words` and `modelspec.trigger_phrase` are **not** read by that
//! extension; they are ai-toolkit/Civitai-side conventions, written because
//! they are the explicit, unambiguous statement of the trigger phrase and
//! cost nothing. The honest summary: `ss_tag_frequency` is what makes a
//! trigger word discoverable in A1111 today, and it works because the
//! trigger phrase is in the captions the frequency is derived from.
//!
//! ## What is derived vs. configured
//!
//! Anything a run already knows is **derived** here from the [`TrainConfig`]
//! and the scanned dataset — rank, alpha, learning rate, optimizer, steps,
//! epochs, resolution, bucket counts, tag frequency, base-model variant. Only
//! what a run genuinely cannot infer is configured, in the `metadata:` block
//! ([`MetadataConfig`]): trigger words, title, author, license, tags,
//! description. That split is the point — a field a user has to retype is a
//! field that goes stale.
//!
//! ## Fields deliberately NOT written
//!
//! - **`ss_clip_skip`** — a CLIP-text-encoder offset. Krea 2 conditions on
//!   Qwen3-VL hidden states ([`crate::qwen3vl`]); there is no CLIP and no
//!   layer to skip, so emitting it would be a lie a UI would act on.
//! - **`ss_text_encoder_lr`** — loractl trains adapters on the MMDiT trunk
//!   only; the text encoder is frozen. Emitted as `0.0` rather than omitted,
//!   because "0" is the true value and reads correctly in a UI.
//! - **`civitai_model_id` / `civitai_version_id`** — injected by download
//!   helpers *after* a file is published. A trainer inventing them would
//!   point at someone else's page.
//! - **`ss_datasets`** — kohya's per-subset JSON blob. With one dataset root
//!   it is a strictly redundant re-encoding of `ss_bucket_info` +
//!   `ss_tag_frequency` + `ss_resolution`, and a second copy is a second
//!   thing to drift.
//!
//! ## Scope: the interop export, not the burn-native snapshot
//!
//! This header rides on [`crate::export`]'s outward-facing files — the ones a
//! ComfyUI/Krea/Civitai user actually loads. The synthetic `BurnTrainer`
//! demo's adapter ([`crate::adapter`]) keeps its JSON sidecar instead: it is
//! attached to no public base model, so it is not an ecosystem artifact, and
//! its sidecar already carries the reconstruction facts a reload needs.
//! `loractl inspect` still reads it — and correctly reports that it has no
//! `__metadata__`.
//!
//! ## Values are strings
//!
//! The header is `string → string`; numbers are written in their natural
//! decimal form and structured values (tag frequency, bucket info, trained
//! words, network args) as **JSON-encoded strings**, exactly as kohya does —
//! that is what the consuming UIs `json.loads()`.
//!
//! [ModelSpec]: https://github.com/Stability-AI/ModelSpec

use crate::config::{ModelVariant, Precision, Quant, ShiftMode, TrainConfig};
use crate::dataset::{Bucket, DatasetEntry};
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// The `__metadata__` map of a LoRA `.safetensors`.
///
/// A `BTreeMap` (not a `HashMap`) so iteration is sorted and every reader of
/// this type — `loractl inspect`, the tests, the `ss_`-prefix filter the
/// hashes are computed over — sees a stable order.
///
/// It does **not** make the on-disk key order deterministic: `safetensors`
/// takes the header map as a `HashMap` and serializes it in hash-iteration
/// order, which `RandomState` randomizes. Nothing depends on that order —
/// a JSON object with the same pairs has the same byte *length* whatever the
/// order, so the tensor-data offsets, and therefore both `sshs_*` hashes, are
/// unaffected.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoraMetadata(BTreeMap<String, String>);

impl LoraMetadata {
    /// An empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a key. Values are always strings — see the module
    /// docs.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.0.insert(key.into(), value.into());
    }

    /// Insert only when `value` is `Some` and non-empty — the "omit rather
    /// than write a blank" rule every optional field here follows.
    pub(crate) fn set_opt(&mut self, key: impl Into<String>, value: Option<impl Into<String>>) {
        if let Some(v) = value {
            let v: String = v.into();
            if !v.is_empty() {
                self.0.insert(key.into(), v);
            }
        }
    }

    /// Look a key up.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    /// Iterate the pairs in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Number of pairs.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the map carries no pairs (a header loractl would not write).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The pairs whose key starts with `prefix` — how a consumer picks out
    /// the `ss_`/`modelspec.` families, and how [`crate::export`] isolates
    /// the `ss_*` subset the sd-webui hashes are computed over.
    pub fn with_prefix(&self, prefix: &str) -> Self {
        Self(
            self.0
                .iter()
                .filter(|(k, _)| k.starts_with(prefix))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    }

    /// Borrow the underlying map.
    pub fn as_map(&self) -> &BTreeMap<String, String> {
        &self.0
    }

    /// Consume into the underlying map.
    pub fn into_map(self) -> BTreeMap<String, String> {
        self.0
    }
}

impl From<BTreeMap<String, String>> for LoraMetadata {
    fn from(map: BTreeMap<String, String>) -> Self {
        Self(map)
    }
}

impl FromIterator<(String, String)> for LoraMetadata {
    fn from_iter<I: IntoIterator<Item = (String, String)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// What the dataset contributed to a run — the bucket and caption facts
/// [`build_metadata`] turns into `ss_bucket_info` / `ss_tag_frequency` /
/// `ss_num_train_images`.
///
/// Borrowed from the already-scanned dataset
/// ([`PreparedDataset`](crate::dataset::PreparedDataset)), never re-read from
/// disk: the metadata must describe *this* run's data, and a second scan
/// could see a folder edited mid-run.
#[derive(Clone)]
pub struct DatasetFacts<'a> {
    /// The dataset folder's own name (not its full path — a published
    /// adapter should not carry the trainer's directory layout). kohya keys
    /// `ss_tag_frequency` by subset directory, and this is that key.
    pub name: String,
    /// One entry per training image, with its caption and bucket.
    pub entries: &'a [DatasetEntry],
    /// The bucket set `entries` index into.
    pub buckets: &'a [Bucket],
    /// Batches the loader produced per epoch — the divisor turning
    /// `steps` into `ss_num_epochs`.
    pub batches_per_epoch: usize,
}

/// Everything [`build_metadata`] needs that is not in the [`TrainConfig`].
///
/// A struct rather than a long argument list so a new fact is an additive
/// field, and so a caller cannot silently swap two `&str`s.
pub struct RunFacts<'a> {
    /// The run's config — the source of every derived hyperparameter.
    pub config: &'a TrainConfig,
    /// Steps actually completed at the moment of writing. Mid-run
    /// checkpoints record their own step; the final export records
    /// `config.steps`. (kohya's `ss_steps` vs `ss_max_train_steps`.)
    pub steps_done: u64,
    /// The dataset, when the run has one (the synthetic tasks do not).
    pub dataset: Option<DatasetFacts<'a>>,
    /// The denoiser checkpoint's **file name** (e.g. `turbo.safetensors`) —
    /// `ss_sd_model_name`. A bare name, never a path.
    pub base_model_file: Option<String>,
    /// Unix seconds when the run started, and when this file was written.
    /// `None` leaves the timestamps (and `modelspec.date`) out entirely,
    /// which is what makes a metadata-carrying export reproducible in tests.
    pub started_at: Option<u64>,
    /// Unix seconds at write time — see [`Self::started_at`].
    pub finished_at: Option<u64>,
}

/// Wall-clock now, in Unix seconds — the value a trainer passes as
/// [`RunFacts::started_at`]/[`finished_at`](RunFacts::finished_at).
///
/// Returns `None` if the clock is before the epoch, so a misconfigured
/// machine drops the timestamp rather than recording a nonsense one.
pub fn unix_now() -> Option<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

/// Format Unix seconds as RFC 3339 UTC (`2026-07-25T09:41:00Z`) — the form
/// ModelSpec's `modelspec.date` takes.
///
/// Hand-rolled (Howard Hinnant's `civil_from_days`) rather than pulling a
/// date crate in for one field: the conversion is proleptic-Gregorian
/// arithmetic with no locale, timezone, or leap-second subtleties, and core's
/// dependency surface is deliberately small.
pub(crate) fn rfc3339_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let tod = secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Days since 1970-01-01 → `(year, month, day)`, proleptic Gregorian.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Split a caption into kohya-style tags: comma-separated, trimmed, empties
/// dropped.
///
/// This is the convention `ss_tag_frequency` is defined in — sd-scripts
/// splits a caption on `,` and counts the pieces, which is why a UI can offer
/// a LoRA's tags as auto-complete. A natural-language caption with no commas
/// therefore counts as one long "tag"; that is the format's behavior, not a
/// bug here.
pub(crate) fn caption_tags(caption: &str) -> Vec<&str> {
    caption
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .collect()
}

/// Tag → occurrence count over every caption, in sorted key order.
pub(crate) fn tag_frequency<'a>(
    captions: impl IntoIterator<Item = &'a str>,
) -> BTreeMap<String, usize> {
    let mut freq: BTreeMap<String, usize> = BTreeMap::new();
    for caption in captions {
        for tag in caption_tags(caption) {
            *freq.entry(tag.to_string()).or_insert(0) += 1;
        }
    }
    freq
}

/// The `ss_base_model_version` label for a model variant.
fn base_model_version(variant: ModelVariant) -> &'static str {
    match variant {
        ModelVariant::Krea2 => "krea-2-raw",
        ModelVariant::Krea2Turbo => "krea-2-turbo",
        ModelVariant::TinyKrea2 => "tiny-krea2",
    }
}

/// `ss_mixed_precision`, in kohya's vocabulary (`no` for full precision).
fn mixed_precision(precision: Precision) -> &'static str {
    match precision {
        Precision::F32 => "no",
        Precision::F16 => "fp16",
        Precision::Bf16 => "bf16",
    }
}

/// The frozen-base quantization label, or `None` when unquantized.
fn quant_label(quant: Quant) -> Option<&'static str> {
    match quant {
        Quant::None => None,
        Quant::Int8 => Some("int8"),
        Quant::Int4 => Some("int4"),
    }
}

/// Format an `f64` the way a config reader expects to see it back: `16` not
/// `16.0` for whole values, the shortest round-tripping decimal otherwise.
fn num(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// [`num`] for an `f32`, formatting at **f32 precision**.
///
/// Not `num(v as f64)`: `{}` prints the shortest decimal that round-trips the
/// type it is given, so widening first prints the f64 that the f32 happens to
/// equal — `alpha: 12.8` would land in the header as `12.800000190734863`,
/// and `dropout: 0.05` as `0.05000000074505806`. The config said `12.8`; the
/// metadata must say `12.8`. Whole values (the defaults, and every value the
/// example configs use) are identical either way, which is exactly why this
/// needs a test with a non-exact value — see
/// `.claude/rules/burn-optimizer-and-dropout.md` on defaults that make right
/// and wrong wiring indistinguishable.
fn num32(v: f32) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// An `f32` as a JSON **number** at f32 precision.
///
/// `serde_json`'s `From<f32>` stores `f as f64` (`Number::from_f32`), so
/// `json!(12.8f32)` renders `12.800000190734863` — the same widening trap as
/// [`num32`], one layer down. Routing through the f32's shortest decimal and
/// back gives the f64 that *prints* as `12.8`.
fn json_f32(v: f32) -> serde_json::Value {
    num32(v)
        .parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map_or(serde_json::Value::Null, serde_json::Value::Number)
}

/// Build the `__metadata__` map for an exported adapter.
///
/// Returns an **empty** map when `config.metadata.embed` is false — the
/// exporter then writes no header at all, keeping the opt-out a real one.
///
/// Every value is a string (see the module docs); structured ones
/// (`ss_tag_frequency`, `ss_bucket_info`, `ss_trained_words`,
/// `ss_network_args`) are JSON-encoded strings, matching kohya so the same
/// consumers parse them. The two `sshs_*` hashes are NOT set here — they can
/// only be computed once the tensors are serialized, so [`crate::export`]
/// adds them at write time.
pub fn build_metadata(facts: &RunFacts<'_>) -> LoraMetadata {
    let mut m = LoraMetadata::new();
    let config = facts.config;
    if !config.metadata.embed {
        return m;
    }
    let meta = &config.metadata;

    // ---- Author-supplied provenance (ModelSpec). ----
    m.set("modelspec.sai_model_spec", "1.0.0");
    m.set(
        "modelspec.architecture",
        format!("{}/lora", base_model_version(config.model.variant)),
    );
    m.set(
        "modelspec.implementation",
        "https://github.com/laurigates/loractl",
    );
    m.set(
        "modelspec.title",
        meta.title
            .clone()
            .unwrap_or_else(|| config.output.name.clone()),
    );
    m.set_opt("modelspec.description", meta.description.clone());
    m.set_opt("modelspec.author", meta.author.clone());
    m.set_opt("modelspec.license", meta.license.clone());
    if !meta.tags.is_empty() {
        m.set("modelspec.tags", meta.tags.join(","));
    }
    // Rectified flow is the only objective the interop export can be produced
    // by (the classifier demo writes a burn-native adapter, not this file).
    m.set("modelspec.prediction_type", "flow");
    m.set(
        "modelspec.resolution",
        format!("{r}x{r}", r = config.dataset.resolution),
    );
    if let Some(at) = facts.finished_at.or(facts.started_at) {
        m.set("modelspec.date", rfc3339_utc(at));
    }

    // ---- Trigger words. `modelspec.trigger_phrase` is a single string, so
    // it takes the first; `ss_trained_words` carries the full list. ----
    if !meta.trigger_words.is_empty() {
        m.set("modelspec.trigger_phrase", meta.trigger_words[0].clone());
        m.set(
            "ss_trained_words",
            serde_json::to_string(&meta.trigger_words).expect("string list serializes"),
        );
    }

    // ---- Network topology. ----
    m.set("ss_network_module", "networks.lora");
    m.set("ss_network_dim", config.lora.rank.to_string());
    m.set("ss_network_alpha", num32(config.lora.alpha));
    m.set("ss_network_dropout", num32(config.lora.dropout));
    // Per-target rank/alpha overrides and the target patterns themselves have
    // no kohya key of their own; kohya's `ss_network_args` is exactly the
    // "arguments this network was built with" slot, so they ride there.
    let targets: Vec<serde_json::Value> = config
        .lora
        .targets
        .iter()
        .map(|t| {
            let mut o = serde_json::Map::new();
            o.insert("pattern".into(), t.pattern.clone().into());
            if let Some(r) = t.rank {
                o.insert("rank".into(), r.into());
            }
            if let Some(a) = t.alpha {
                o.insert("alpha".into(), json_f32(a));
            }
            serde_json::Value::Object(o)
        })
        .collect();
    if !targets.is_empty() {
        m.set(
            "ss_network_args",
            serde_json::json!({ "targets": targets }).to_string(),
        );
    }

    // ---- Base model. ----
    m.set(
        "ss_base_model_version",
        base_model_version(config.model.variant),
    );
    m.set_opt("ss_sd_model_name", facts.base_model_file.clone());
    if let Some(ta) = &config.model.training_adapter {
        // The turbo assistant-LoRA (#83) is merged into the frozen base
        // before injection, so it is part of what produced these weights.
        m.set_opt(
            "ss_training_adapter",
            ta.file_name().map(|n| n.to_string_lossy().into_owned()),
        );
    }

    // ---- Optimization. ----
    m.set(
        "ss_optimizer",
        format!(
            "burn.optim.AdamW(weight_decay={})",
            num(config.optim.weight_decay)
        ),
    );
    m.set("ss_learning_rate", num(config.optim.lr));
    m.set("ss_unet_lr", num(config.optim.lr));
    // Always frozen — see the module docs' "deliberately NOT written".
    m.set("ss_text_encoder_lr", "0.0");
    m.set("ss_max_train_steps", config.steps.to_string());
    m.set("ss_steps", facts.steps_done.to_string());
    m.set("ss_seed", config.seed.to_string());
    m.set(
        "ss_mixed_precision",
        mixed_precision(config.compute.precision),
    );
    m.set(
        "ss_gradient_checkpointing",
        bool_str(config.compute.grad_checkpointing),
    );
    m.set_opt("ss_base_model_quant", quant_label(config.compute.quant));
    m.set("ss_output_name", config.output.name.clone());
    m.set_opt("ss_training_comment", meta.comment.clone());
    m.set("ss_loractl_version", env!("CARGO_PKG_VERSION"));
    if let Some(at) = facts.started_at {
        m.set("ss_training_started_at", at.to_string());
    }
    if let Some(at) = facts.finished_at {
        m.set("ss_training_finished_at", at.to_string());
    }

    // ---- The flow-matching sampler (sd-scripts' flux/SD3 keys). ----
    m.set("ss_timestep_sampling", "logit_normal");
    m.set("ss_logit_mean", num(config.flow.logit_mean));
    m.set("ss_logit_std", num(config.flow.logit_std));
    match config.flow.shift_mode {
        ShiftMode::Constant => {
            m.set("ss_discrete_flow_shift", num(config.flow.shift));
        }
        ShiftMode::Resolution => {
            // Per-batch `exp(μ)` — there is no single shift to report, so the
            // μ-line anchors are what a reader needs to reproduce it (#84).
            m.set(
                "ss_dynamic_shift",
                serde_json::json!({
                    "base_image_seq_len": config.flow.base_image_seq_len,
                    "max_image_seq_len": config.flow.max_image_seq_len,
                    "base_shift": config.flow.base_shift,
                    "max_shift": config.flow.max_shift,
                })
                .to_string(),
            );
        }
    }

    // ---- Dataset: resolution, buckets, tags. ----
    m.set(
        "ss_resolution",
        format!("({r}, {r})", r = config.dataset.resolution),
    );
    m.set(
        "ss_batch_size_per_device",
        config.dataset.batch_size.to_string(),
    );
    if let Some(ds) = &facts.dataset {
        m.set("ss_num_train_images", ds.entries.len().to_string());
        m.set("ss_enable_bucket", "True");
        m.set("ss_num_batches_per_epoch", ds.batches_per_epoch.to_string());
        if ds.batches_per_epoch > 0 {
            // Ceiling division: a run that stops mid-epoch still touched that
            // epoch, which is how kohya counts.
            let epochs = config.steps.div_ceil(ds.batches_per_epoch as u64);
            m.set("ss_num_epochs", epochs.to_string());
        }
        m.set("ss_bucket_info", bucket_info_json(ds));
        // kohya's per-subset counts, which A1111's metadata panel displays.
        // loractl has no repeat mechanism, so `n_repeats` is truthfully 1.
        let mut dirs = serde_json::Map::new();
        dirs.insert(
            ds.name.clone(),
            serde_json::json!({"n_repeats": 1, "img_count": ds.entries.len()}),
        );
        m.set(
            "ss_dataset_dirs",
            serde_json::Value::Object(dirs).to_string(),
        );
        // Keyed by subset directory, as kohya writes it — a UI shows the
        // frequencies per folder.
        let mut by_subset = serde_json::Map::new();
        by_subset.insert(
            ds.name.clone(),
            serde_json::json!(tag_frequency(ds.entries.iter().map(|e| e.caption.as_str()))),
        );
        m.set(
            "ss_tag_frequency",
            serde_json::Value::Object(by_subset).to_string(),
        );
    }

    m
}

/// kohya's `"True"`/`"False"` (Python `str(bool)`) — the spelling the UIs
/// that read these keys already parse.
fn bool_str(v: bool) -> &'static str {
    if v { "True" } else { "False" }
}

/// kohya's `ss_bucket_info`: `{"buckets": {"<i>": {"resolution": [w, h],
/// "count": n}}}`, with empty buckets omitted (a bucket no image landed in is
/// not part of this run).
fn bucket_info_json(ds: &DatasetFacts<'_>) -> String {
    let mut buckets = serde_json::Map::new();
    for (i, bucket) in ds.buckets.iter().enumerate() {
        let count = ds.entries.iter().filter(|e| e.bucket == i).count();
        if count == 0 {
            continue;
        }
        buckets.insert(
            i.to_string(),
            serde_json::json!({
                "resolution": [bucket.width, bucket.height],
                "count": count,
            }),
        );
    }
    serde_json::json!({ "buckets": buckets }).to_string()
}

/// Read a `.safetensors` file's `__metadata__` **without loading any
/// tensors**.
///
/// Reads the 8-byte length prefix and just the JSON header — bytes, not
/// gigabytes — so inspecting a multi-GB checkpoint is instant. Returns an
/// empty map for a file that carries no `__metadata__` (a valid, common
/// case: diffusers scripts and older ai-toolkit write none), and errors only
/// on a file that is not a readable safetensors header.
///
/// Non-string metadata values are rejected rather than coerced: the format
/// defines the map as string → string, and a file that violates it is one
/// whose other claims should not be trusted either.
pub fn read_metadata(path: &Path) -> Result<LoraMetadata> {
    let mut file =
        File::open(path).with_context(|| format!("opening {} for metadata", path.display()))?;

    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .with_context(|| format!("{} is too short to be a safetensors file", path.display()))?;
    let n = u64::from_le_bytes(len_bytes);
    // Validate the claimed header length BEFORE allocating for it. Two
    // bounds, because either alone lets a hostile or corrupt file steer the
    // allocation: the format's own 100 MB ceiling, and the file's actual
    // size — an 8-byte file claiming a 100 MB header must cost 8 bytes to
    // reject, not 100 MB.
    if n > 100_000_000 {
        bail!(
            "{}: safetensors header claims {n} bytes — not a safetensors file",
            path.display()
        );
    }
    let file_len = file
        .metadata()
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    if n + 8 > file_len {
        bail!(
            "{}: safetensors header claims {n} bytes but the file is {file_len} — truncated or \
             not a safetensors file",
            path.display()
        );
    }
    let mut header = vec![0u8; n as usize];
    file.read_exact(&mut header)
        .with_context(|| format!("reading the safetensors header of {}", path.display()))?;

    let json: serde_json::Value = serde_json::from_slice(&header)
        .with_context(|| format!("parsing the safetensors header of {}", path.display()))?;
    let Some(meta) = json.get("__metadata__") else {
        return Ok(LoraMetadata::new());
    };
    let Some(obj) = meta.as_object() else {
        bail!("{}: __metadata__ is not an object", path.display());
    };
    let mut out = LoraMetadata::new();
    for (k, v) in obj {
        let Some(s) = v.as_str() else {
            bail!(
                "{}: __metadata__[{k:?}] is {v}, but safetensors metadata values must be strings",
                path.display()
            );
        };
        out.set(k.clone(), s);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_dates_round_trip_known_points() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        // 2026-07-25T12:34:56Z
        assert_eq!(rfc3339_utc(1_784_982_896), "2026-07-25T12:34:56Z");
        // A leap day.
        assert_eq!(rfc3339_utc(1_709_164_800), "2024-02-29T00:00:00Z");
    }

    #[test]
    fn tags_split_on_commas_and_count() {
        let freq = tag_frequency(["sks dog, park, sunny", "sks dog,  park ", "", "  "]);
        assert_eq!(freq["sks dog"], 2);
        assert_eq!(freq["park"], 2);
        assert_eq!(freq["sunny"], 1);
        assert_eq!(freq.len(), 3, "blank captions contribute no tags");
    }

    #[test]
    fn numbers_render_without_trailing_zeros() {
        assert_eq!(num(16.0), "16");
        assert_eq!(num(0.0001), "0.0001");
        assert_eq!(num32(16.0), "16");
    }

    /// Pins BOTH sides of the widening trap: what the f32 formatters produce,
    /// and what the `as f64` / `json!(f32)` spellings they replace produce.
    /// If `serde_json` ever renders `From<f32>` at f32 precision, the last
    /// assertion fails and [`json_f32`] can be deleted — a loud signal rather
    /// than a workaround that outlives its cause.
    #[test]
    fn f32_values_are_not_widened_before_formatting() {
        assert_eq!(num32(12.8), "12.8");
        assert_eq!(
            num(12.8f32 as f64),
            "12.800000190734863",
            "the trap num32 exists to avoid"
        );
        assert_eq!(num32(0.05), "0.05");
        assert_eq!(json_f32(2.7).to_string(), "2.7");
        assert_eq!(
            serde_json::json!(2.7f32).to_string(),
            "2.700000047683716",
            "serde_json's From<f32> still stores `f as f64`"
        );
    }
}
