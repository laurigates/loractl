# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.*", "numpy"]
# ///
"""Per-block symmetric int4 golden for loractl's frozen-base quantization (#96).

The int4 twin of `quant_reference.py`. Pins the numerics `src/quant.rs` must
reproduce for `Quant::Int4`: weight-only int4, symmetric (zero-point-free), one
f32 scale per contiguous block of 32 along the INPUT dimension of a weight held
in file layout `[d_out, d_in]` — exactly what burn 0.21 computes for
`QuantValue::Q4S` + `QuantLevel::block([1, 32])` under min-max calibration.

burn's symmetric scale is `2·absmax / (b − a)` where `(a, b)` is the value's
range; for `Q4S` that range is `(−7, 7)`, so `scale = 2·absmax/14 = absmax/7`
(the int8 `Q8S` range is `(−127, 127)` → `absmax/127`). Quantized values are
`round(w/scale)` clamped to `[−7, 7]` (15 levels), dequantized as `q·scale`.

Emits `crates/loractl-core/tests/fixtures/quant_int4_golden.json`:

- `w`  — seed-5 float32 weight `[8, 64]` (2 blocks of 32 per output row),
- `x`  — seed-6 float32 activations `[4, 64]`,
- `dq` — dequantize(quantize(w)): `round(w/scale)·scale` per block (clamp ±7),
- `y`  — `x @ dq.T`, the forward the Rust `quant_matmul_t` must match.

Rounding note: torch.round is round-half-to-even; Rust `round()` is
half-away-from-zero. The generator ASSERTS no |w|/scale value lands within
1e-4 of a .5 tie, so both conventions produce identical `dq` and the Rust
test can assert tight (1e-6) agreement instead of a mushy 1-ulp bound. int4's
15 levels tie more readily than int8's 255, so this guard matters more here.

Deterministic, offline, tiny. Regenerate: `just quant-int4-reference`.
"""

import json
import pathlib

import numpy as np
import torch

BLOCK = 32
QMAX = 7.0  # Q4S range max — symmetric int4 clamps to [-7, 7]
OUT_PATH = pathlib.Path("crates/loractl-core/tests/fixtures/quant_int4_golden.json")


def quantize_per_block(w: torch.Tensor) -> torch.Tensor:
    """Symmetric int4 per contiguous block of BLOCK along the last dim."""
    d_out, d_in = w.shape
    assert d_in % BLOCK == 0
    blocks = w.reshape(d_out, d_in // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=-1, keepdim=True) / QMAX
    ratio = blocks / scale
    # Tie guard: keep the fixture rounding-convention-agnostic (see docstring).
    frac = (ratio - ratio.floor() - 0.5).abs()
    assert frac.min() > 1e-4, "regenerate with another seed: a .5 rounding tie"
    q = ratio.round().clamp(-QMAX, QMAX)
    return (q * scale).reshape(d_out, d_in)


def main() -> None:
    torch.manual_seed(5)
    w = torch.randn(8, 64, dtype=torch.float32)
    torch.manual_seed(6)
    x = torch.randn(4, 64, dtype=torch.float32)

    dq = quantize_per_block(w)
    y = x @ dq.T

    golden = {
        "block": BLOCK,
        "w": w.numpy().astype(np.float32).tolist(),
        "x": x.numpy().astype(np.float32).tolist(),
        "dq": dq.numpy().astype(np.float32).tolist(),
        "y": y.numpy().astype(np.float32).tolist(),
    }
    OUT_PATH.write_text(json.dumps(golden))
    print(f"wrote {OUT_PATH} (w {list(w.shape)}, x {list(x.shape)}, y {list(y.shape)})")


if __name__ == "__main__":
    main()
