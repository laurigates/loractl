# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Deterministic reference for the kohya-ss adapter export test (issue #17, M6).

Every weight is a FIXED constant (no RNG), so the burn exporter and this script
compute the identical bytes and differ only by f32 rounding. This derives the
checked-in golden fixture; `cargo test` reads the JSON and never runs this
script.

The export contract (kohya-ss): for a delta at module path P, the on-disk file
carries three tensors under the `lora_<P with dots→underscores>` prefix:

  - `<prefix>.lora_down.weight` = A transposed to [rank, d_in]
  - `<prefix>.lora_up.weight`   = B transposed to [d_out, rank]
  - `<prefix>.alpha`            = the [1] scalar `alpha`

Convention: A/B below are in **burn** layout ([d_in, d_out]); the exporter
transposes on the way out (burn `Linear.weight` is [d_in, d_out]; the LoRA
loaders want the [out, in]-style transpose), so this script transposes here and
the Rust test compares with ZERO transposing.
"""

import json
import platform
import sys

import numpy as np

# ---- FIXED constants; duplicated verbatim in tests/adapter_export.rs with a
# ---- "MUST match reference/lora_export_reference.py" comment. -----------------
PATH = "transformer.h.0.attn.c_attn"
D_IN, D_OUT, RANK = 4, 6, 2
ALPHA = 8.0

# A [d_in, rank] = [4, 2] (burn layout), trainable down-projection.
A = np.array(
    [[0.10, -0.20],
     [0.30, 0.40],
     [-0.50, 0.60],
     [0.70, -0.80]], dtype=np.float32)

# B [rank, d_out] = [2, 6] (burn layout), non-zero so the transpose is
# observable (a real exported B has been trained off its zero init).
B = np.array(
    [[0.01, -0.02, 0.03, -0.04, 0.05, -0.06],
     [0.07, -0.08, 0.09, -0.10, 0.11, -0.12]], dtype=np.float32)


def kohya_prefix(path: str) -> str:
    return "lora_" + path.replace(".", "_")


prefix = kohya_prefix(PATH)
down_key = f"{prefix}.lora_down.weight"  # [rank, d_in]
up_key = f"{prefix}.lora_up.weight"      # [d_out, rank]
alpha_key = f"{prefix}.alpha"            # [1]

a_transposed = A.T.copy()  # [2, 4]
b_transposed = B.T.copy()  # [6, 2]

kohya_keys = sorted([down_key, up_key, alpha_key])

print("=== kohya-ss export reference ===", file=sys.stderr)
print(f"path: {PATH}  prefix: {prefix}", file=sys.stderr)
print(f"down {a_transposed.shape}  up {b_transposed.shape}  alpha {ALPHA}", file=sys.stderr)

golden = {
    "provenance": {
        "platform": platform.platform(),
        "note": "kohya-ss export; A/B burn-layout [d_in,d_out], transposed here; "
        "regenerate with `just export-reference`",
    },
    "hyperparams": {"d_in": D_IN, "d_out": D_OUT, "rank": RANK, "alpha": ALPHA, "path": PATH},
    "kohya_keys": kohya_keys,
    "alpha_value": ALPHA,
    "a_transposed": a_transposed.flatten().tolist(),
    "a_shape": list(a_transposed.shape),
    "b_transposed": b_transposed.flatten().tolist(),
    "b_shape": list(b_transposed.shape),
}
print(json.dumps(golden, indent=2))
