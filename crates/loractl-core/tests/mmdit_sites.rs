//! Pins the config-derived LoRA-site enumeration (ADR-0005 tooling seam):
//! [`MmditConfig::injectable_sites`] must enumerate an architecture's sites
//! **without instantiating the model** (the real config is ~12.8B params —
//! target-set tooling like `examples/step_probe.rs` must never have to build
//! it just to count matches), and [`Mmdit::injectable_sites`] must be pure
//! delegation to it. Path strings and ORDER are additionally pinned against
//! `base_linears_mut` by `tests/quant_mmdit.rs`.

use loractl_core::NdArray;
use loractl_core::config::ModelVariant;
use loractl_core::mmdit::{Mmdit, MmditConfig};

/// The real-architecture site count, computed from the config's own
/// constants (7 trunk projections per layer), never hardcoded from a guess.
#[test]
fn krea2_config_sites_are_seven_per_layer() {
    let cfg = MmditConfig::krea2();
    let sites = cfg.injectable_sites();
    assert_eq!(
        sites.len(),
        cfg.layers * 7,
        "7 injectable trunk projections per layer"
    );
    // Spot-pin the path scheme and dims the target regexes match against.
    assert_eq!(sites[0].path, "blocks.0.attn.wq");
    assert_eq!(sites[0].d_in, cfg.features);
    assert_eq!(sites[0].d_out, cfg.features);
    let last = sites.last().expect("non-empty");
    assert_eq!(last.path, format!("blocks.{}.mlp.down", cfg.layers - 1));
    assert_eq!(
        last.d_in,
        MmditConfig::swiglu_dim(cfg.features, cfg.multiplier)
    );
    assert_eq!(last.d_out, cfg.features);
}

/// The model-side enumeration is the config-side one — same paths, same
/// dims, same order (built on the tiny config; instantiating krea2() here
/// would allocate the ~12.8B model).
#[test]
fn model_sites_delegate_to_config_sites() {
    let device = Default::default();
    let cfg = MmditConfig::tiny_krea2();
    let model = Mmdit::<NdArray>::init(cfg.clone(), &device);
    let from_model: Vec<_> = model
        .injectable_sites()
        .into_iter()
        .map(|s| (s.path, s.d_in, s.d_out))
        .collect();
    let from_config: Vec<_> = cfg
        .injectable_sites()
        .into_iter()
        .map(|s| (s.path, s.d_in, s.d_out))
        .collect();
    assert_eq!(cfg.layers * 7, from_config.len());
    assert_eq!(from_model, from_config);
}

/// `for_variant` is the one home of the variant → architecture mapping
/// (`diffusion_trainer::variant_configs` delegates to it): Krea2 and the
/// architecturally-identical Krea2Turbo map to `krea2()`, TinyKrea2 to
/// `tiny_krea2()`.
#[test]
fn for_variant_maps_each_variant_to_its_architecture() {
    assert_eq!(
        MmditConfig::for_variant(ModelVariant::Krea2),
        MmditConfig::krea2()
    );
    assert_eq!(
        MmditConfig::for_variant(ModelVariant::Krea2Turbo),
        MmditConfig::krea2()
    );
    assert_eq!(
        MmditConfig::for_variant(ModelVariant::TinyKrea2),
        MmditConfig::tiny_krea2()
    );
}
