# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Deterministic reference for the training-adapter merge-at-load (#83).

Pins the exact merge math + layout convention loractl's
`training_adapter::merge_delta` computes: folding an external LoRA training
adapter into a frozen base weight,

    W_burn += (alpha / rank) · (A_burn · B_burn)

where the *on-disk* factors (PyTorch/diffusers `[out, in]` convention, as in
`ostris/krea2_turbo_training_adapter`) are

    down = A_disk  [rank, d_in]   (`lora_A.weight` / `lora_down.weight`)
    up   = B_disk  [d_out, rank]  (`lora_B.weight` / `lora_up.weight`)

and burn's `Linear.weight` is `[d_in, d_out]`, so the merge lifts the factors to
burn layout `A_burn = A_diskᵀ [d_in, rank]`, `B_burn = B_diskᵀ [rank, d_out]`.
The delta `A_burn · B_burn` `[d_in, d_out]` is identically `(B_disk · A_disk)ᵀ`
— the standard PyTorch weight delta `(alpha/rank)·B·A` `[d_out, d_in]`
transposed into burn's weight layout.

Every value is a FIXED constant (no RNG), so the Rust merge and this script
compute identical bytes up to f32 rounding. This derives the checked-in golden;
`cargo test` reads the JSON and never runs this script. Regenerate with
`just training-adapter-merge-reference`.
"""

import json
import platform
import sys

import numpy as np

# ---- FIXED constants; duplicated verbatim in tests/training_adapter.rs with a
# ---- "MUST match reference/training_adapter_merge_reference.py" comment. ------
D_IN, D_OUT, RANK = 4, 6, 2
ALPHA = 8.0  # scaling = alpha / rank = 4.0

# Frozen base weight in burn `[d_in, d_out]` layout.
W = np.array(
    [[0.10, -0.20, 0.30, -0.40, 0.50, -0.60],
     [-0.11, 0.21, -0.31, 0.41, -0.51, 0.61],
     [0.12, -0.22, 0.32, -0.42, 0.52, -0.62],
     [-0.13, 0.23, -0.33, 0.43, -0.53, 0.63]], dtype=np.float32)

# On-disk down `A_disk [rank, d_in]` (`lora_A.weight`).
DOWN = np.array(
    [[0.10, 0.30, -0.50, 0.70],
     [-0.20, 0.40, 0.60, -0.80]], dtype=np.float32)

# On-disk up `B_disk [d_out, rank]` (`lora_B.weight`), non-zero so the transpose
# is observable (a trained assistant B is off its zero init).
UP = np.array(
    [[0.01, 0.07],
     [-0.02, -0.08],
     [0.03, 0.09],
     [-0.04, -0.10],
     [0.05, 0.11],
     [-0.06, -0.12]], dtype=np.float32)

scaling = ALPHA / RANK

# PyTorch weight delta `[d_out, d_in]`, then transposed into burn `[d_in, d_out]`.
delta_pytorch = scaling * (UP @ DOWN)          # [d_out, d_in]
delta_burn = delta_pytorch.T.astype(np.float32)  # [d_in, d_out]

# Cross-check: burn-layout factors give the identical delta directly.
a_burn = DOWN.T.astype(np.float32)             # [d_in, rank]
b_burn = UP.T.astype(np.float32)               # [rank, d_out]
delta_burn_direct = (scaling * (a_burn @ b_burn)).astype(np.float32)
assert np.allclose(delta_burn, delta_burn_direct, atol=1e-6), "layout identity broke"

merged_burn = (W + delta_burn).astype(np.float32)

print("=== training-adapter merge reference ===", file=sys.stderr)
print(f"d_in={D_IN} d_out={D_OUT} rank={RANK} alpha={ALPHA} scaling={scaling}", file=sys.stderr)

golden = {
    "provenance": {
        "platform": platform.platform(),
        "note": "training-adapter merge-at-load (#83); disk A/B are [out,in]-style, "
        "burn W/delta are [d_in,d_out]; regenerate with "
        "`just training-adapter-merge-reference`",
    },
    "hyperparams": {
        "d_in": D_IN,
        "d_out": D_OUT,
        "rank": RANK,
        "alpha": ALPHA,
        "scaling": scaling,
    },
    "w_burn": W.flatten().tolist(),
    "w_shape": list(W.shape),
    "down_disk": DOWN.flatten().tolist(),
    "down_shape": list(DOWN.shape),
    "up_disk": UP.flatten().tolist(),
    "up_shape": list(UP.shape),
    "delta_burn": delta_burn.flatten().tolist(),
    "merged_burn": merged_burn.flatten().tolist(),
    "merged_shape": list(merged_burn.shape),
}
print(json.dumps(golden, indent=2))
