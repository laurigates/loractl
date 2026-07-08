//! Multi-site grad-routing proof for generic LoRA injection (milestone 6, #17).
//!
//! Freezes the loaded tiny GPT-2, builds a [`LoraAdapters`] set covering *all*
//! injectable sites across *all* blocks via [`build_adapters`], runs ONE real
//! training step, and proves the mechanism trains the right parameters and only
//! those:
//!
//! - every `deltas[i].lora_b` (and `lora_a`) receives a gradient,
//! - every base site weight (the frozen `c_attn`/`c_proj`/`c_fc`) receives none,
//! - `GradientsParams::from_grads(grads, &set)` isolates exactly `2·N` tensors
//!   (each delta contributes its two bias-less factors),
//! - after one Adam step every delta's `B` has moved off its zero init.
//!
//! This is how M6 proves trainability without adding a GPT-2 training loop to
//! `BurnTrainer` (that is M11/M14): a single grad-routed step through the full
//! transformer with the generic name-keyed set.
//!
//! Offline and always-run; reuses the checked-in parity fixture. Serialized on
//! `RNG_LOCK` because each `LoraDelta`'s `A` draws from the process-global
//! ndarray RNG.

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Tensor, TensorData};
use loractl_core::adapters::{LoraAdapters, build_adapters};
use loractl_core::config::{LoraConfig, TargetSpec};
use loractl_core::gpt2::{Gpt2, Gpt2Config};
use std::sync::Mutex;

/// See `adapter_roundtrip.rs`: the ndarray RNG is a process-global static, so
/// tests that draw from it serialize against each other.
static RNG_LOCK: Mutex<()> = Mutex::new(());

type AB = Autodiff<NdArray>;

const SAFETENSORS: &str = "tests/fixtures/tiny-gpt2/model.safetensors";
const INPUT_IDS: [i64; 8] = [5, 12, 7, 3, 42, 1, 0, 9];

/// Load the tiny GPT-2 and freeze it — the base rides unmodified while only the
/// injected deltas train (the standard LoRA setup).
fn load_frozen_tiny() -> Gpt2<AB> {
    use burn_store::{KeyRemapper, ModuleSnapshot, SafetensorsStore};

    let device = Default::default();
    let mut model = Gpt2::<AB>::init(Gpt2Config::tiny(), &device);
    let remapper = KeyRemapper::from_patterns(Gpt2::<AB>::layernorm_key_remap().to_vec())
        .expect("valid remap patterns");
    let mut store = SafetensorsStore::from_file(SAFETENSORS)
        .allow_partial(true)
        .remap(remapper);
    let result = model.load_from(&mut store).expect("safetensors load");
    assert!(result.errors.is_empty(), "load errors: {:?}", result.errors);
    model.no_grad()
}

/// A LoRA config whose single `.*` target matches every injectable site.
fn all_sites_config() -> LoraConfig {
    LoraConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
        targets: vec![TargetSpec {
            pattern: r".*".to_string(),
            rank: None,
            alpha: None,
        }],
    }
}

#[test]
fn multi_site_step_routes_grads_to_every_delta_only() {
    let _guard = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let device = Default::default();
    let model = load_frozen_tiny();

    // Build adapters at every injectable site (4 per block × n_layer blocks).
    let sites = model.injectable_sites();
    let mut set: LoraAdapters<AB> = build_adapters(&sites, &all_sites_config(), &device);
    let n = set.deltas.len();
    assert_eq!(
        n,
        Gpt2Config::tiny().n_layer * 4,
        "all four sites per block should be targeted"
    );
    assert_eq!(
        set.targets,
        sites.iter().map(|s| s.path.clone()).collect::<Vec<_>>()
    );

    let seq = INPUT_IDS.len();
    let vocab = model.config.vocab_size;
    let ids = Tensor::<AB, 1, Int>::from_data(TensorData::new(INPUT_IDS.to_vec(), [seq]), &device)
        .reshape([1, seq]);

    // One real step: cross-entropy against an arbitrary fixed target sequence.
    let targets = Tensor::<AB, 1, Int>::from_data(
        TensorData::new(vec![1i64, 2, 3, 4, 5, 6, 7, 8], [seq]),
        &device,
    );
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let logits2d = model.forward_with_adapters(ids, &set).reshape([seq, vocab]);
    let loss = loss_fn.forward(logits2d, targets);
    assert!(
        loss.clone().into_scalar().is_finite(),
        "loss must be finite"
    );

    let grads = loss.backward();

    // Every delta's factors receive a gradient.
    for (i, delta) in set.deltas.iter().enumerate() {
        assert!(
            delta.lora_b.weight.val().grad(&grads).is_some(),
            "delta {i} ({}) lora_b must receive a gradient",
            set.targets[i]
        );
        assert!(
            delta.lora_a.weight.val().grad(&grads).is_some(),
            "delta {i} ({}) lora_a must receive a gradient",
            set.targets[i]
        );
    }

    // Every base site weight receives NONE (the base is frozen).
    for block in model.transformer.h.iter() {
        for w in [
            &block.attn.c_attn.weight,
            &block.attn.c_proj.weight,
            &block.mlp.c_fc.weight,
            &block.mlp.c_proj.weight,
        ] {
            assert!(
                w.val().grad(&grads).is_none(),
                "a frozen base site weight must receive NO gradient"
            );
        }
    }

    // from_grads over the set isolates exactly the 2·N delta factors.
    let grad_params = GradientsParams::from_grads(grads, &set);
    assert_eq!(
        grad_params.len(),
        2 * n,
        "from_grads(&set) must isolate exactly 2·N delta tensors, got {}",
        grad_params.len()
    );

    // One Adam step; every delta's B moves off its zero init.
    let mut optim = AdamConfig::new().init::<AB, LoraAdapters<AB>>();
    set = optim.step(1e-3, set, grad_params);
    for (i, delta) in set.deltas.iter().enumerate() {
        let b_sum: f32 = delta.lora_b.weight.val().abs().sum().into_scalar();
        assert!(
            b_sum > 0.0,
            "delta {i} ({}) B must have moved off zero after one step",
            set.targets[i]
        );
    }
}
