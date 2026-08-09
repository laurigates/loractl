//! The measured-fit envelope, and the pre-flight advisory that names it.
//!
//! Everything in this repo that says a Krea 2 step *fits* says it about one
//! point: **512px, int4 frozen base, block-level gradient checkpointing, on a
//! 24 GB RTX 4090** — 19.4 GB peak, zero panics, 3/3 steps, 196/196 sites
//! (ADR-0005 Addendum 3). Every other resolution is arithmetic. Until #179
//! nothing checked a config against that, so a `resolution: 1024` edit — one
//! line, no flag, no warning — bought a 3× trunk sequence and found out at
//! OOM time, minutes into an encode phase.
//!
//! This module is the check. It is **advisory, never fatal**: the hard
//! `bail!`s in [`DiffusionTrainer`](crate::DiffusionTrainer) are for illegal
//! *combinations*, and a large resolution is legal — it is merely unmeasured.
//! The output is a [`TrainEvent::Warning`](crate::TrainEvent::Warning), the
//! variant already documented as "a non-fatal issue worth surfacing to the
//! operator", so this adds nothing to the wire contract.
//!
//! Two rules the text obeys, both inherited from ADR-0005's withdrawal sweep
//! (`docs/adrs/0005-int4-training-vram-bound.md`, "Provenance and units"):
//!
//! 1. **No predicted peak.** The 19.4 GB figure carries two recorded, still
//!    open ambiguities — GB vs GiB (~7% at this magnitude) and total vs
//!    above-baseline. It is an anchor, not a budget, and subtracting a
//!    derived cost from it would restate an inference as a measurement.
//! 2. **What changes, and by how much — then go probe.** The advisory names
//!    the sequence at both resolutions and the retained block-input set at
//!    both, labels them derived, and hands over to `just step-probe`.
//!
//! Like the rest of `loractl-core`, this module renders nothing: it returns a
//! `String` and the caller decides whether anyone sees it.

use crate::config::{BackendKind, ModelVariant, Quant, TrainConfig};
use crate::mmdit::MmditConfig;

/// The only resolution the block-checkpointed int4 route has ever been
/// measured at: 512px on a 24 GB RTX 4090, 19.4 GB peak, **zero panics**,
/// 3/3 steps, 196/196 sites, ~4 GB headroom (ADR-0005 Addendum 3, from the
/// on-box run reported on #134/PR #135; the 300-step real run that closed
/// M14 is #25/PR #150, same route, same resolution). Read the addendum
/// before quoting the peak — its unit basis is explicitly unresolved.
pub const MEASURED_RESOLUTION: u32 = 512;

/// Peak device memory at [`MEASURED_RESOLUTION`], in GB as reported
/// (ADR-0005 Addendum 3). Quoted here only to say *which* run the envelope
/// is; the advisory never does arithmetic on it, because the addendum
/// records that neither the GB/GiB basis nor total-vs-above-baseline was
/// pinned.
pub const MEASURED_PEAK_GB: f64 = 19.4;

/// Headroom left on the 24 GB card at that peak (ADR-0005 Addendum 3).
/// Same caveat: an anchor, not a budget.
pub const MEASURED_HEADROOM_GB: f64 = 4.0;

/// VRAM one cached example's conditioning holds for the **whole run**:
/// `[1, 512, 12, 2560]` f32 = 60 MiB. Fixed-length by construction —
/// `Qwen3VlConditioner::tokenize` right-pads every caption to `max_length`
/// regardless of what it says — so this is a rate, not an average
/// (#175, ADR-0010 claim ledger #5, DERIVED).
pub const CONDITIONING_MIB_PER_EXAMPLE: u64 = 60;

/// Trip point for the residency half of the advisory: half of
/// [`MEASURED_HEADROOM_GB`] read as GiB, i.e. ~35 examples.
///
/// A judgement call, and the weakest number in this module — so it is pinned
/// in **both** directions by `residency_advisory_fires_at_49_examples_and_not_at_5`
/// rather than left free to drift. Half rather than all of the headroom
/// because ADR-0010 puts *exhaustion* at ~65 examples, and an advisory that
/// only fires once the card is already full has no one left to advise. The
/// GB-vs-GiB choice is deliberately unimportant here: this gates a sentence,
/// it does not predict a fit.
const RESIDENCY_ADVISORY_MIB: u64 = 2048;

/// VRAM the eager dataset read holds resident for `entries` examples.
///
/// This depends on a specific, current behaviour: `dataset::prepare_dataset`
/// returns a `PreparedDataset` of **live device tensors**, so the whole
/// cache is resident at once rather than streamed per batch (#175, ADR-0010
/// ledger #5). If that read ever becomes per-batch, this function — and the
/// residency half of [`preflight_advisory`] — describe behaviour that no
/// longer exists and should be deleted together. Kept as one named function
/// so that deletion is one site.
fn dataset_residency_mib(entries: usize) -> u64 {
    entries as u64 * CONDITIONING_MIB_PER_EXAMPLE
}

/// Bytes `checkpointed_step`'s capture phase retains at the peak: exactly
/// `layers × [batch, seq, features]` block-input residual streams, f32, all
/// co-resident because the reverse sweep drains them last-to-first
/// (`block_ckpt.rs:13-16`; the table is ADR-0008's, whose 384px and 512px
/// rows reproduce ADR-0005 Addendum 2's two measured anchors).
///
/// `batch` is a factor and not a constant on purpose: ADR-0008's figures are
/// batch-1 and it is explicit that they scale linearly, so dropping the term
/// would understate every non-default config by exactly the batch size.
fn retained_block_input_bytes(cfg: &MmditConfig, sequence_len: usize, batch: usize) -> u64 {
    (cfg.layers * sequence_len * cfg.features * 4 * batch) as u64
}

/// The pre-flight advisory for `config`, or `None` when it is inside the
/// measured envelope.
///
/// `dataset_entries` is the scanned example count when the caller has one
/// (the encode phase does, [`Trainer::train`](crate::Trainer::train) does
/// not); `None` simply skips the residency half.
///
/// Returns **one** message or none — never a list. Two independent findings
/// concatenate into a single warning, because the failure mode this whole
/// mechanism has to avoid is becoming a lint that fires on every non-default
/// config and gets tuned out.
pub fn preflight_advisory(config: &TrainConfig, dataset_entries: Option<usize>) -> Option<String> {
    // The envelope is a statement about the real 12.8B model on a 24 GB card.
    // `TinyKrea2` is the offline fixture — a 2-block toy whose whole point is
    // that it fits anywhere — so quoting 28 layers and 6144 features at it
    // would be a lie, and every offline test would carry a warning it cannot
    // act on.
    if !matches!(
        config.model.variant,
        ModelVariant::Krea2 | ModelVariant::Krea2Turbo
    ) {
        return None;
    }

    let mut parts: Vec<String> = Vec::new();

    // The quant conjunct is not redundant. `quant: none` is not "inside the
    // envelope at any resolution" — it is outside it at *every* resolution,
    // and loudly so already (ADR-0005 Addendum 2: the unquantized step pinned
    // 67.9 GiB, int8's ~17.1 GB step OOMed). Warning about resolution there
    // would name the second-largest problem.
    if config.dataset.resolution > MEASURED_RESOLUTION && config.compute.quant != Quant::None {
        let (mmdit, _, vae, caption_tokens) =
            crate::diffusion_trainer::variant_configs(config.model.variant);
        let compression = vae.spatial_compression();
        let batch = config.dataset.batch_size.max(1) as usize;
        let here = crate::mmdit::token_geometry(
            config.dataset.resolution,
            compression,
            mmdit.patch,
            caption_tokens,
        );
        let measured = crate::mmdit::token_geometry(
            MEASURED_RESOLUTION,
            compression,
            mmdit.patch,
            caption_tokens,
        );
        let here_gb = retained_block_input_bytes(&mmdit, here.sequence_len, batch) as f64 / 1e9;
        let measured_gb =
            retained_block_input_bytes(&mmdit, measured.sequence_len, batch) as f64 / 1e9;
        parts.push(format!(
            "dataset.resolution {} with compute.quant {} is outside the fit this repo has \
             measured. The one measured point is {MEASURED_RESOLUTION}px \
             ({MEASURED_PEAK_GB} GB peak, zero panics, 196/196 sites — ADR-0005 Addendum 3, \
             whose unit basis is unresolved, so it is an anchor and not a budget). At {}px the \
             trunk attends over {} tokens against {} at {MEASURED_RESOLUTION}px ({} image + {} \
             caption, padded to a multiple of {}), and the block inputs gradient checkpointing \
             retains grow from ~{measured_gb:.2} GB to ~{here_gb:.2} GB at batch {batch} \
             (derived from the capture contract, not measured — ADR-0008). Nothing at this size \
             has been measured: probe it first with `just step-probe <config> --steps 3`, and \
             read the result as zero-panic-or-not, never as a survived OOM storm.",
            config.dataset.resolution,
            // Lowercased so the advisory names the value the way a config
            // spells it (`quant: int4`), not the way `Debug` does.
            format!("{:?}", config.compute.quant).to_lowercase(),
            config.dataset.resolution,
            here.sequence_len,
            measured.sequence_len,
            here.image_tokens,
            caption_tokens,
            crate::mmdit::SEQUENCE_PAD,
        ));
    }

    // Gated on a device backend: on ndarray the "VRAM" this describes is host
    // RAM against no 24 GB bound at all, so every offline run and every CPU
    // smoke would carry an advisory about a card it is not using.
    if let Some(entries) = dataset_entries {
        let mib = dataset_residency_mib(entries);
        if mib > RESIDENCY_ADVISORY_MIB && config.compute.backend != BackendKind::Ndarray {
            parts.push(format!(
                "The dataset read is eager: all {entries} examples' conditioning stays \
                 device-resident for the whole run at {CONDITIONING_MIB_PER_EXAMPLE} MiB each \
                 ([1, 512, 12, 2560] f32, fixed-length however short the caption), \
                 {:.2} GiB in total — against the ~{MEASURED_HEADROOM_GB} GB of headroom the \
                 {MEASURED_RESOLUTION}px measurement left, and ~65 examples exhausts that on \
                 residency alone (#175, ADR-0010 ledger #5, derived).",
                mib as f64 / 1024.0,
            ));
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ComputeConfig, DatasetConfig, ModelConfig};
    use crate::mmdit::token_geometry;
    use figment::Figment;
    use figment::providers::{Format, Yaml};

    /// Krea 2's own geometry, spelled out so the table rows below read as the
    /// ADR's table and not as a call into the code they are checking.
    const KREA2_COMPRESSION: usize = 8;
    const KREA2_PATCH: usize = 2;
    const KREA2_CAPTION: usize = 512;

    #[test]
    fn token_geometry_reproduces_adr0008_table() {
        // docs/adrs/0008-host-offload-mechanism-and-scope.md, the resolution
        // table. 384px is the row with teeth: it is the ONLY one where the
        // 256-pad is not a no-op (1088 → 1280), so dropping the pad, or
        // dropping the fixed caption block, is invisible at 512/1024 and
        // fails here. All three rows are asserted for that reason.
        for (resolution, latent, image_tokens, sequence_len) in [
            (384, 48, 576, 1280),
            (512, 64, 1024, 1536),
            (1024, 128, 4096, 4608),
        ] {
            let g = token_geometry(resolution, KREA2_COMPRESSION, KREA2_PATCH, KREA2_CAPTION);
            assert_eq!(
                (g.latent, g.image_tokens, g.sequence_len),
                (latent, image_tokens, sequence_len),
                "{resolution}px"
            );
        }
    }

    #[test]
    fn token_geometry_is_the_derivation_the_bench_already_used() {
        // The extraction's contract with `StepWork::for_config`: image-only
        // (caption 0) must be exactly what the bench derived before, INCLUDING
        // the degenerate case it relies on to refuse a zero denominator. A
        // `max(1)` anywhere in the helper turns that refusal into a
        // fabricated `tok_s=`.
        assert_eq!(token_geometry(32, 8, 2, 0).image_tokens, 4);
        assert_eq!(token_geometry(8, 8, 2, 0).image_tokens, 0);
    }

    #[test]
    fn retained_block_inputs_reproduce_adr0008_bytes() {
        let cfg = MmditConfig::krea2();
        // ADR-0008: 28 × seq × 6144 × 4 B at batch 1 — 1.057 GB at 512px,
        // 3.171 GB at 1024px. Using `txtdim` (2560) instead of `features`
        // (6144), or dropping the ×4, misses both by more than 2×.
        let at = |seq, batch| retained_block_input_bytes(&cfg, seq, batch) as f64 / 1e9;
        assert!((at(1536, 1) - 1.057).abs() < 0.001, "{}", at(1536, 1));
        assert!((at(4608, 1) - 3.171).abs() < 0.001, "{}", at(4608, 1));
        // Batch is a factor, not a constant (ADR-0008 is explicit that its
        // figures are batch-1 and scale linearly). Without this row the term
        // could be dropped and every row above would still pass.
        assert!((at(1536, 2) - 2.114).abs() < 0.001, "{}", at(1536, 2));
    }

    /// A `krea2-comfyui.yaml`-shaped config: the real variant on the measured
    /// route, with `resolution` and `batch_size` as the variables.
    fn krea2_config(resolution: u32, quant: Quant) -> TrainConfig {
        TrainConfig {
            model: ModelConfig {
                base: "/path/to/comfyui/models".into(),
                variant: ModelVariant::Krea2,
                ..Default::default()
            },
            dataset: DatasetConfig {
                path: "/path/to/dataset".into(),
                resolution,
                batch_size: 1,
                ..Default::default()
            },
            compute: ComputeConfig {
                backend: BackendKind::Cuda,
                quant,
                grad_checkpointing: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn advisory_fires_for_1024px_int4() {
        // The acceptance criterion of #179, asserted as a string contract:
        // the derived sequence at this resolution, the measured point it is
        // being compared against, and the command that settles it.
        let message = preflight_advisory(&krea2_config(1024, Quant::Int4), None)
            .expect("1024px int4 is outside the measured envelope");
        assert!(message.contains("4608"), "{message}");
        assert!(message.contains("1536"), "{message}");
        assert!(message.contains("512px"), "{message}");
        assert!(message.contains("just step-probe"), "{message}");
        // Derived figures must be labelled as derived — ADR-0005 carries a
        // withdrawal sweep because inferred numbers were once stated as
        // measured.
        assert!(message.contains("derived"), "{message}");
        // And no predicted peak: the 19.4 GB anchor is quoted, never spent.
        assert!(!message.contains("will need"), "{message}");
    }

    #[test]
    fn no_advisory_at_the_measured_resolution() {
        // The false-positive gate at the boundary. `>` not `>=`: 512px IS the
        // measured point, so it is the one config that must never warn.
        assert!(preflight_advisory(&krea2_config(512, Quant::Int4), None).is_none());
        assert!(preflight_advisory(&krea2_config(512, Quant::Int4), Some(5)).is_none());
    }

    #[test]
    fn no_advisory_without_quant() {
        // `quant: none` is outside the envelope at every resolution and has
        // its own loud measured record (ADR-0005 Addendum 2); naming the
        // resolution there would name the second-largest problem.
        assert!(preflight_advisory(&krea2_config(1024, Quant::None), None).is_none());
    }

    #[test]
    fn no_advisory_for_the_tiny_fixture_variant() {
        // What keeps the whole offline suite silent — and honest: 28 layers
        // and 6144 features are a lie about a 2-block, 64-feature fixture.
        let mut config = krea2_config(1024, Quant::Int4);
        config.model.variant = ModelVariant::TinyKrea2;
        assert!(preflight_advisory(&config, Some(49)).is_none());
    }

    #[test]
    fn residency_advisory_fires_at_49_examples_and_not_at_5() {
        // Both directions, so the threshold cannot be quietly widened into
        // noise (5 examples is krea2-dog's actual DreamBooth set) nor raised
        // until it never fires. 49 × 60 MiB = 2.87 GiB.
        let config = krea2_config(512, Quant::Int4);
        let message = preflight_advisory(&config, Some(49)).expect("49 × 60 MiB is 2.87 GiB");
        assert!(message.contains("60 MiB"), "{message}");
        assert!(message.contains("#175"), "{message}");
        assert!(message.contains("2.87 GiB"), "{message}");
        assert!(preflight_advisory(&config, Some(5)).is_none());

        // On ndarray there is no card to fill, so the same 49 examples say
        // nothing. This is what keeps every CPU run and every offline
        // Krea2-variant test quiet.
        let cpu = TrainConfig {
            compute: ComputeConfig {
                backend: BackendKind::Ndarray,
                ..config.compute
            },
            ..config
        };
        assert!(preflight_advisory(&cpu, Some(49)).is_none());
    }

    #[test]
    fn both_conditions_produce_one_warning() {
        // The motivating run. `Option<String>`, not `Vec`: the type is what
        // stops this becoming a lint that emits a list.
        let message = preflight_advisory(&krea2_config(1024, Quant::Int4), Some(49))
            .expect("both halves apply");
        assert!(message.contains("4608"), "{message}");
        assert!(message.contains("2.87 GiB"), "{message}");
    }

    #[test]
    fn no_advisory_for_any_shipped_example_config() {
        // Reads the real files rather than a copy, so raising a shipped
        // example's resolution fails HERE instead of shipping an example that
        // warns about itself.
        //
        // Checked with no dataset knowledge (`None`, the config-only claim
        // #179 asks for) and with `Some(5)`, krea2-dog's actual DreamBooth
        // set. Deliberately NOT with #179's 49-image set: residency is a
        // property of the user's dataset, not of these files, and a 49-image
        // run on `krea2-lora.yaml`'s wgpu backend genuinely does hold
        // 2.87 GiB — firing there is the motivating case, not a false
        // positive. `residency_advisory_fires_at_49_examples_and_not_at_5`
        // pins that half.
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/examples");
        let mut checked = 0;
        for entry in std::fs::read_dir(dir).expect("config/examples exists") {
            let path = entry.expect("readable entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            let config: TrainConfig = Figment::new()
                .merge(Yaml::file(&path))
                .extract()
                .unwrap_or_else(|e| panic!("{} parses into TrainConfig: {e}", path.display()));
            for entries in [None, Some(5)] {
                assert_eq!(
                    preflight_advisory(&config, entries),
                    None,
                    "{} advises at {entries:?} examples",
                    path.display()
                );
            }
            checked += 1;
        }
        // The loop passes vacuously over an empty or renamed directory.
        assert!(checked >= 6, "only {checked} example configs were checked");
    }
}
