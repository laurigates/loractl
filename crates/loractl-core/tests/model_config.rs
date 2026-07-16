//! `ModelVariant`/`ModelConfig` config-surface contract for M15 (#82):
//! the `krea2-turbo` variant spellings, the kebab-case serde surface, the
//! backward-compatible `model.checkpoint` override, the variant → denoiser
//! filename precedence, and the shared Krea2/Turbo encoder-cache
//! fingerprint.
//!
//! Mirrors `flow_task.rs`: the enum and its serde surface are always
//! compiled, the YAML/env/flag layers accept the same spellings, and an
//! existing config with no `checkpoint:` key keeps parsing unchanged.

use loractl_core::TrainConfig;
use loractl_core::config::{ModelConfig, ModelVariant};
use loractl_core::diffusion_trainer::{denoiser_filename, encoder_fingerprint};

/// The `--variant`/YAML/env vocabulary accepts the turbo spellings —
/// kebab-case (the canonical Serialize form), joined, the short alias, and
/// case-insensitively — and an unknown spelling is a clear error naming the
/// full vocabulary, not a creative fallback.
#[test]
fn model_variant_from_str_accepts_turbo_spellings() {
    for spelling in ["krea2-turbo", "krea2turbo", "turbo", "KREA2-TURBO"] {
        assert_eq!(
            spelling.parse::<ModelVariant>(),
            Ok(ModelVariant::Krea2Turbo),
            "spelling {spelling:?} must parse as Krea2Turbo"
        );
    }
    let err = "bogus"
        .parse::<ModelVariant>()
        .expect_err("an unknown variant spelling must be a clear error");
    assert!(
        err.contains("krea2") && err.contains("krea2-turbo") && err.contains("tiny-krea2"),
        "the error should name the full variant vocabulary, got: {err}"
    );
}

/// All three variants serialize as their kebab-case names — a plain
/// lowercase rename would emit `"krea2turbo"`, which round-trips (it's a
/// belt-and-braces `FromStr` spelling) but diverges from the documented
/// config vocabulary — and Serialize → Deserialize is lossless (the
/// hand-written `Deserialize` routes through `FromStr`, which must accept
/// everything the derived `Serialize` emits).
#[test]
fn model_variant_serializes_kebab_case_and_round_trips() {
    for (variant, spelling) in [
        (ModelVariant::Krea2, "krea2"),
        (ModelVariant::Krea2Turbo, "krea2-turbo"),
        (ModelVariant::TinyKrea2, "tiny-krea2"),
    ] {
        assert_eq!(
            serde_json::to_value(variant).unwrap(),
            serde_json::Value::String(spelling.into()),
            "{variant:?} must serialize as {spelling:?}"
        );
        let json = serde_json::to_string(&variant).expect("variant serializes");
        let back: ModelVariant =
            serde_json::from_str(&json).expect("serialized variant deserializes");
        assert_eq!(back, variant, "round-trip must be lossless for {json}");
    }
}

/// Backward-compat: a config with NO `checkpoint:` key must deserialize with
/// `checkpoint == None` (the variant default filename applies). This is the
/// `#[serde(default)]` contract that keeps every pre-M15 config working —
/// pinned so a future removal of the default can't silently regress it
/// (mirrors `config_without_task_or_flow_blocks_defaults_to_classification`).
#[test]
fn config_without_checkpoint_defaults_to_none() {
    let json = r#"{
        "model": { "base": "synthetic", "variant": "krea2-turbo" },
        "lora": {},
        "dataset": { "path": "unused" }
    }"#;
    let config: TrainConfig =
        serde_json::from_str(json).expect("a config without a checkpoint key should deserialize");
    assert_eq!(config.model.variant, ModelVariant::Krea2Turbo);
    assert_eq!(config.model.checkpoint, None);
}

/// The `checkpoint` field parses from a model block — the override surface
/// for local repacks like `krea2_turbo_fp8_scaled.safetensors`.
#[test]
fn checkpoint_field_parses() {
    let model: ModelConfig = serde_json::from_str(
        r#"{ "base": "some/dir", "variant": "krea2", "checkpoint": "x.safetensors" }"#,
    )
    .expect("a model block with a checkpoint key should deserialize");
    assert_eq!(model.checkpoint.as_deref(), Some("x.safetensors"));
}

/// The denoiser filename precedence: each variant supplies its default
/// (`raw.safetensors` for Krea2/TinyKrea2, `turbo.safetensors` for Turbo),
/// and an explicit `model.checkpoint` beats every variant default.
#[test]
fn denoiser_filename_precedence() {
    for (variant, default) in [
        (ModelVariant::Krea2, "raw.safetensors"),
        (ModelVariant::TinyKrea2, "raw.safetensors"),
        (ModelVariant::Krea2Turbo, "turbo.safetensors"),
    ] {
        let mut model = ModelConfig {
            base: "unused".into(),
            variant,
            checkpoint: None,
            denoiser: None,
            text_encoder: None,
            vae: None,
            tokenizer: None,
        };
        assert_eq!(
            denoiser_filename(&model),
            default,
            "{variant:?} default filename"
        );
        model.checkpoint = Some("x.safetensors".into());
        assert_eq!(
            denoiser_filename(&model),
            "x.safetensors",
            "an explicit checkpoint must beat the {variant:?} default"
        );
    }
}

/// Turbo shares Krea2's encoder-cache fingerprint: the two variants read the
/// same encoder files from the same `base`, so their caches must be
/// interchangeable — a variant-derived fingerprint would force a pointless
/// full re-encode when switching raw ↔ turbo. The historical literals are
/// pinned too: the emitted strings must stay byte-identical to the pre-M15
/// `{variant:?}`-derived form, or every existing on-disk cache is silently
/// invalidated.
#[test]
fn turbo_shares_the_krea2_encoder_cache_fingerprint() {
    for max_length in [16, 512] {
        assert_eq!(
            encoder_fingerprint(ModelVariant::Krea2Turbo, max_length),
            encoder_fingerprint(ModelVariant::Krea2, max_length),
            "turbo and krea2 caches must be interchangeable at ml{max_length}"
        );
    }
    assert_eq!(
        encoder_fingerprint(ModelVariant::Krea2, 512),
        "krea2-ml512-enc32"
    );
    assert_eq!(
        encoder_fingerprint(ModelVariant::TinyKrea2, 16),
        "tinykrea2-ml16-enc32"
    );
}
