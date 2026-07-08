//! Generic LoRA injection + one training step on the loaded real GPT-2 (M6,
//! #17 — re-expressed from the M3 single-`c_attn` attach).
//!
//! Loads the checked-in tiny GPT-2 on the autodiff backend, builds a
//! [`LoraAdapters`] set with a single delta targeting block 0's `c_attn`
//! projection, and runs ONE genuine training step through the full transformer
//! via [`Gpt2::forward_with_adapters`]. Proves the generalized attach is sound
//! and trainable:
//!
//! - **Zero-init no-op:** because each delta's `B` factor starts at zero, the
//!   pre-step adapted logits equal the base golden bit-for-bit — a free
//!   attach-integrity check that the injection didn't perturb the forward.
//! - **Gradient routing:** after backprop, the trainable `deltas[0].lora_b` has
//!   a gradient and the (still in-place) frozen-in-spirit base `c_attn` weight
//!   has none — the delta rides on top of an unmodified base.
//! - **A real step runs:** the loss is finite and one Adam step on the adapter
//!   set completes without panic, moving `B` off zero.
//!
//! Offline and always-run: it reuses the same checked-in fixture as the parity
//! test.

use burn::backend::{Autodiff, NdArray};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Tensor, TensorData};
use loractl_core::adapters::LoraAdapters;
use loractl_core::gpt2::{Gpt2, Gpt2Config};
use loractl_core::lora::LoraDelta;
use serde::Deserialize;

/// Autodiff-wrapped CPU backend — a real training step needs gradients.
type AB = Autodiff<NdArray>;

const GOLDEN: &str = include_str!("fixtures/gpt2_tiny_golden.json");
const SAFETENSORS: &str = "tests/fixtures/tiny-gpt2/model.safetensors";
const INPUT_IDS: [i64; 8] = [5, 12, 7, 3, 42, 1, 0, 9];
/// The single injection target for this test — block 0's fused QKV projection.
const TARGET: &str = "transformer.h.0.attn.c_attn";

#[derive(Deserialize)]
struct Golden {
    logits: Vec<f32>,
    logits_shape: Vec<usize>,
}

/// Load the tiny GPT-2 and **freeze** it (`no_grad`): the generic injection adds
/// each base `Linear`'s output into the autodiff graph (unlike the old attach
/// that bypassed the block's own `c_attn`), so training the adapters on an
/// unmodified base means freezing that base — the standard LoRA setup, and what
/// a future GPT-2 trainer will do. Freezing is what makes the "base site weight
/// gets no gradient" assertion below true.
fn load_tiny() -> Gpt2<AB> {
    use burn::module::Module;
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

fn input_ids(device: &burn::tensor::Device<AB>) -> Tensor<AB, 2, Int> {
    Tensor::<AB, 1, Int>::from_data(
        TensorData::new(INPUT_IDS.to_vec(), [INPUT_IDS.len()]),
        device,
    )
    .reshape([1, INPUT_IDS.len()])
}

/// Build a one-delta adapter set targeting block 0's `c_attn` (`n_embd ->
/// 3·n_embd`, sized from the tiny config).
fn single_c_attn_adapters(device: &burn::tensor::Device<AB>) -> LoraAdapters<AB> {
    let e = Gpt2Config::tiny().n_embd;
    let delta = LoraDelta::<AB>::new(e, 3 * e, /* rank */ 4, /* alpha */ 8.0, device);
    LoraAdapters {
        deltas: vec![delta],
        targets: vec![TARGET.to_string()],
    }
}

#[test]
fn lora_attach_one_step_on_loaded_gpt2() {
    let device = Default::default();
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden");
    let model = load_tiny();
    let ids = input_ids(&device);

    let mut set = single_c_attn_adapters(&device);

    // --- Attach-integrity: zero-init B => adapted forward == base golden. ---
    let adapted = model.forward_with_adapters(ids.clone(), &set);
    let [b, s, v] = adapted.dims();
    assert_eq!(vec![s, v], golden.logits_shape, "logits shape");
    assert_eq!(b, 1);
    let adapted_flat: Vec<f32> = adapted
        .clone()
        .into_data()
        .convert::<f32>()
        .into_vec()
        .unwrap();
    let max_diff = adapted_flat
        .iter()
        .zip(&golden.logits)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff <= 1e-4,
        "zero-init adapted logits must equal the base golden: max|Δ| = {max_diff:e}"
    );

    // --- One real training step on the adapter set. ---
    // Cross-entropy against an arbitrary fixed target sequence.
    let targets = Tensor::<AB, 1, Int>::from_data(
        TensorData::new(vec![1i64, 2, 3, 4, 5, 6, 7, 8], [s]),
        &device,
    );
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let logits2d = model.forward_with_adapters(ids, &set).reshape([s, v]);
    let loss = loss_fn.forward(logits2d, targets);
    let loss_value: f32 = loss.clone().into_scalar();
    assert!(
        loss_value.is_finite() && loss_value > 0.0,
        "loss must be finite and positive, got {loss_value}"
    );

    // Backprop, then check gradient routing BEFORE consuming the grads.
    let grads = loss.backward();
    assert!(
        set.deltas[0].lora_b.weight.val().grad(&grads).is_some(),
        "the trainable LoRA `B` factor must receive a gradient"
    );
    assert!(
        set.deltas[0].lora_a.weight.val().grad(&grads).is_some(),
        "the trainable LoRA `A` factor must receive a gradient"
    );
    assert!(
        model.transformer.h[0]
            .attn
            .c_attn
            .weight
            .val()
            .grad(&grads)
            .is_none(),
        "the frozen base c_attn weight must receive NO gradient"
    );

    // One Adam step on the adapter set only — trains the deltas, leaves the base
    // model untouched. Must complete without panic.
    let mut optim = AdamConfig::new().init::<AB, LoraAdapters<AB>>();
    let grad_params = GradientsParams::from_grads(grads, &set);
    set = optim.step(1e-3, set, grad_params);

    // After the step, B has moved off zero, so the adapter is no longer a no-op.
    let b_sum: f32 = set.deltas[0].lora_b.weight.val().abs().sum().into_scalar();
    assert!(
        b_sum > 0.0,
        "after one step the LoRA `B` factor must have moved off zero"
    );
}
