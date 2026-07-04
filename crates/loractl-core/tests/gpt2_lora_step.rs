//! LoRA attach + one training step on the loaded real GPT-2 (M3, #2 —
//! acceptance c).
//!
//! Loads the checked-in tiny GPT-2 on the autodiff backend, wraps block 0's
//! `c_attn` projection with [`LoraLinear::from_base`] (the M3 attach entry
//! point), and runs ONE genuine training step through the full transformer.
//! Proves the attach is sound and trainable:
//!
//! - **Zero-init no-op:** because the adapter's `B` factor starts at zero, the
//!   pre-step adapted logits equal the base golden bit-for-bit — a free
//!   attach-integrity check that the wrapping didn't perturb the forward.
//! - **Gradient routing:** after backprop, the trainable `lora_b` has a
//!   gradient and the frozen base `c_attn` weight has none.
//! - **A real step runs:** the loss is finite and one Adam step on the adapter
//!   completes without panic.
//!
//! Offline and always-run: it reuses the same checked-in fixture as the parity
//! test.

use burn::backend::{Autodiff, NdArray};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Tensor, TensorData};
use loractl_core::gpt2::{Gpt2, Gpt2Config};
use loractl_core::lora::LoraLinear;
use serde::Deserialize;

/// Autodiff-wrapped CPU backend — a real training step needs gradients.
type AB = Autodiff<NdArray>;

const GOLDEN: &str = include_str!("fixtures/gpt2_tiny_golden.json");
const SAFETENSORS: &str = "tests/fixtures/tiny-gpt2/model.safetensors";
const INPUT_IDS: [i64; 8] = [5, 12, 7, 3, 42, 1, 0, 9];

#[derive(Deserialize)]
struct Golden {
    logits: Vec<f32>,
    logits_shape: Vec<usize>,
}

fn load_tiny() -> Gpt2<AB> {
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
    model
}

fn input_ids(device: &burn::tensor::Device<AB>) -> Tensor<AB, 2, Int> {
    Tensor::<AB, 1, Int>::from_data(
        TensorData::new(INPUT_IDS.to_vec(), [INPUT_IDS.len()]),
        device,
    )
    .reshape([1, INPUT_IDS.len()])
}

#[test]
fn lora_attach_one_step_on_loaded_gpt2() {
    let device = Default::default();
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden");
    let model = load_tiny();
    let ids = input_ids(&device);

    // Attach: freeze the loaded c_attn and wrap it with a fresh LoRA adapter.
    let base_c_attn = model.transformer.h[0].attn.c_attn.clone();
    let mut lora = LoraLinear::<AB>::from_base(
        base_c_attn,
        /* rank */ 4,
        /* alpha */ 8.0,
        &device,
    );

    // --- Attach-integrity: zero-init B => adapted forward == base golden. ---
    let adapted = model.forward_with_lora_c_attn(ids.clone(), &lora);
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

    // --- One real training step on the adapter. ---
    // Cross-entropy against an arbitrary fixed target sequence.
    let targets = Tensor::<AB, 1, Int>::from_data(
        TensorData::new(vec![1i64, 2, 3, 4, 5, 6, 7, 8], [s]),
        &device,
    );
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let logits2d = model.forward_with_lora_c_attn(ids, &lora).reshape([s, v]);
    let loss = loss_fn.forward(logits2d, targets);
    let loss_value: f32 = loss.clone().into_scalar();
    assert!(
        loss_value.is_finite() && loss_value > 0.0,
        "loss must be finite and positive, got {loss_value}"
    );

    // Backprop, then check gradient routing BEFORE consuming the grads.
    let grads = loss.backward();
    assert!(
        lora.lora_b.weight.val().grad(&grads).is_some(),
        "the trainable LoRA `B` factor must receive a gradient"
    );
    assert!(
        lora.base.weight.val().grad(&grads).is_none(),
        "the frozen base c_attn weight must receive NO gradient"
    );
    // lora_a should also be trainable.
    assert!(
        lora.lora_a.weight.val().grad(&grads).is_some(),
        "the trainable LoRA `A` factor must receive a gradient"
    );

    // One Adam step on the adapter only — trains the LoRA, leaves the frozen
    // base model untouched. Must complete without panic.
    let mut optim = AdamConfig::new().init::<AB, LoraLinear<AB>>();
    let grad_params = GradientsParams::from_grads(grads, &lora);
    lora = optim.step(1e-3, lora, grad_params);

    // After the step, B has moved off zero, so the adapter is no longer a no-op.
    let b_sum: f32 = lora.lora_b.weight.val().abs().sum().into_scalar();
    assert!(
        b_sum > 0.0,
        "after one step the LoRA `B` factor must have moved off zero"
    );
}
