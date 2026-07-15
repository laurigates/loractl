# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.*", "einops", "safetensors", "numpy", "requests", "transformers>=4.57", "diffusers>=0.35"]
# ///
"""Scaled-fp8 (float8_e4m3fn + weight_scale) fixtures for loractl M15 (issue #82).

Krea-2-Turbo ships as a ComfyUI-style scaled-fp8 repack: quantized 2-D
projections stored as F8_E4M3 with a per-tensor F32 `<name>.weight_scale`
sidecar (`dequant = fp8.float() * scale`), everything else float. This script
pins that format's numerics for the Rust loader (`src/fp8.rs`) at three levels:

1. `fp8_lut_golden.json` — all 256 e4m3fn byte decodings from torch itself
   (NaN at 0x7f/0xff serialized as null; the Rust LUT must match bit-for-bit).
2. `fp8_dequant_golden.json` — seed-8 non-square dequant cases covering the
   0-d scalar scale, the per-channel `[out]` scale (broadcast along axis 0,
   i.e. rows of the file's [out, in] orientation), and the `[1]`-is-scalar rule.
3. Model-level twins:
   - `tiny-mmdit/model_fp8.safetensors` + `fp8_mmdit_golden.json` — the
     checked-in seed-11 tiny fixture quantized in the real repack's key set,
     with the staged forward golden computed by the official pinned-commit
     `SingleStreamDiT` over the quantize-then-DEQUANTIZED weights (so the fp8
     load path is compared against official numerics on mathematically
     identical weights). Schema is identical to `mmdit_tiny_golden.json`.
   - `tiny-krea2/turbo_fp8.safetensors` + `turbo_fp8_dequant.safetensors` —
     the e2e bundle's MMDiT (the committed `raw.safetensors`, cross-checked
     against krea2_reference's seed-14 `mmdit_state()` below) as a
     quantized/dequantized twin pair for the two-path forward-agreement test.
     One designated non-square tensor carries a per-channel scale so the e2e
     also exercises that branch.

Quantization matches the verified real repack (krea2_turbo_fp8_scaled
header, 2026-07-15): per-tensor `scale = amax/448` (0-d F32), quantized stems
are exactly the 2-D attn wq/wk/wv/gate/wo + mlp gate/up/down projections in
`blocks.*` / `txtfusion.layerwise_blocks.*` / `txtfusion.refiner_blocks.*`;
first/last/tmlp/tproj/txtmlp/norm scales/mod.lin/projector stay float.

Network is needed only at fixture-REGENERATION time (the pinned-commit
mmdit.py fetch); all emitted fixtures are checked in and the Rust tests run
offline. transformers/diffusers are dependencies only because importing
krea2_reference (for `mmdit_state()`) imports its sibling reference modules
at module scope.
"""

import argparse
import json
import math
import os
import platform
import re
import sys
import tempfile

# The reference decorates forwards with @torch.compile; run eagerly on CPU.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

# Single source of truth for the model helpers and dims: the sibling
# reference scripts whose outputs these fixtures extend.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmdit_reference  # noqa: E402
from krea2_reference import MMDIT, mmdit_state  # noqa: E402
from qwen3vl_reference import TINY_SELECT_LAYERS, TINY_TEXT  # noqa: E402
from qwen_vae_reference import TINY_CFG as VAE_TINY_CFG  # noqa: E402

FP8_MAX = 448.0  # e4m3fn max normal (byte 0x7e)

# The real repack's quantized key set: every 2-D attn/mlp projection in the
# trunk and both text-fusion groups (28*8 + 4*8 = 256 tensors at real depth;
# 48 at the tiny configs' 2 trunk + 2+2 fusion blocks).
QUANT_RE = re.compile(
    r"^(?:blocks|txtfusion\.(?:layerwise_blocks|refiner_blocks))\.\d+"
    r"\.(?:attn\.(?:wq|wk|wv|gate|wo)|mlp\.(?:gate|up|down))\.weight$"
)

# The tiny-krea2 tensor designated to carry a per-channel scale ([out], from
# amax over dim=1) so the e2e twins exercise that branch; non-square [256, 64]
# so an axis swap cannot cancel out.
PER_CHANNEL_KEY = "blocks.0.mlp.gate.weight"


def quantize(w, per_channel=False):
    """fp8 repack of one weight: scale = amax/448 (0-d, or [out] per-channel)."""
    if per_channel:
        scale = (w.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
        q = (w / scale[:, None]).to(torch.float8_e4m3fn)
    else:
        scale = (w.abs().amax() / FP8_MAX).clamp(min=1e-12)
        q = (w / scale).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


def dequant(q, scale):
    """The loader's contract: lut(byte) * scale, per-channel along axis 0."""
    if scale.ndim == 1 and scale.numel() > 1:
        return q.float() * scale[:, None]
    return q.float() * scale  # 0-d, or [1] treated as scalar


def quantize_state(state, per_channel_key=None):
    """Split a state dict into the fp8 repack and its dequantized f32 twin."""
    fp8_sd, dq = {}, {}
    for k, v in state.items():
        if QUANT_RE.match(k):
            assert v.ndim == 2, f"quantized stem must be 2-D: {k}"
            q, s = quantize(v, per_channel=(k == per_channel_key))
            fp8_sd[k] = q
            fp8_sd[k + "_scale"] = s  # "<n>.weight" -> "<n>.weight_scale"
            dq[k] = dequant(q, s).contiguous()
        else:
            fp8_sd[k] = v
            dq[k] = v
    return fp8_sd, dq


def write_lut_golden(out):
    lut = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    vals = [None if math.isnan(v) else v for v in lut.tolist()]
    assert vals[0x7E] == 448.0 and vals[0x01] == 0.001953125
    assert vals[0x7F] is None and vals[0xFF] is None
    assert sum(v is None for v in vals) == 2
    path = os.path.join(out, "fp8_lut_golden.json")
    with open(path, "w") as f:
        json.dump({"lut": vals}, f)
    print(f"lut golden written to {path}", file=sys.stderr)


def write_dequant_golden(out):
    g = torch.Generator().manual_seed(8)

    def case(per_channel=False, shape_one=False):
        w = torch.randn(5, 3, generator=g)  # non-square: catches an axis swap
        q, s = quantize(w, per_channel=per_channel)
        if shape_one:
            s = s.reshape(1)
        expected = dequant(q, s)
        assert torch.isfinite(expected).all()
        return {
            "weight_bytes": q.view(torch.uint8).flatten().tolist(),
            "shape": list(q.shape),
            "scale": s.item() if s.ndim == 0 else s.tolist(),
            "expected": expected.flatten().tolist(),
        }

    golden = {
        "scalar": case(),
        "per_channel": case(per_channel=True),
        "scale_shape_one": case(shape_one=True),
    }
    assert len(golden["per_channel"]["scale"]) == 5
    path = os.path.join(out, "fp8_dequant_golden.json")
    with open(path, "w") as f:
        json.dump(golden, f)
    print(f"dequant golden written to {path}", file=sys.stderr)


def write_tiny_krea2_twins(out):
    bundle = os.path.join(out, "tiny-krea2")

    # The twins must model the SAME network as the committed e2e bundle, so
    # they quantize the committed raw.safetensors tensors directly. The
    # krea2_reference mmdit_state() (seed 14) cross-check below tolerates
    # last-ulp drift: the scalar `randn * 0.02` rounds differently across
    # torch builds (the bundle was authored on macOS/arm64), so bytes are
    # only reproducible on the authoring platform — layout and values are.
    state = load_file(os.path.join(bundle, "raw.safetensors"))
    fresh = mmdit_state()
    assert sorted(fresh) == sorted(state), "mmdit_state drifted from raw.safetensors"
    for k, v in state.items():
        assert fresh[k].dtype == v.dtype and fresh[k].shape == v.shape
        # atol only: near-zero entries make rtol meaningless, and observed
        # cross-build drift is ≤5e-8 abs while a wrong stream would be ~2e-2.
        assert torch.allclose(fresh[k], v, rtol=0.0, atol=1e-6), (
            f"mmdit_state diverged from raw.safetensors at {k}"
        )

    # Same seams krea2_reference.py asserts — the bundle stays composed.
    assert VAE_TINY_CFG["z_dim"] == MMDIT["channels"], (
        "vae z_dim must equal mmdit channels"
    )
    assert TINY_TEXT["hidden_size"] == MMDIT["txtdim"], (
        "encoder hidden must equal mmdit txtdim"
    )
    assert len(TINY_SELECT_LAYERS) == MMDIT["txtlayers"], (
        "select count must equal txtlayers"
    )

    fp8_sd, dq = quantize_state(state, per_channel_key=PER_CHANNEL_KEY)
    n_scales = sum(1 for k in fp8_sd if k.endswith(".weight_scale"))
    assert n_scales == 48, f"expected 48 quantized tensors, got {n_scales}"
    assert fp8_sd[PER_CHANNEL_KEY + "_scale"].shape == (256,)

    save_file(
        {k: v.contiguous() for k, v in fp8_sd.items()},
        os.path.join(bundle, "turbo_fp8.safetensors"),
    )
    save_file(
        {k: v.contiguous() for k, v in dq.items()},
        os.path.join(bundle, "turbo_fp8_dequant.safetensors"),
    )
    print(
        f"tiny-krea2 twins written to {bundle} ({n_scales} quantized tensors, "
        f"per-channel: {PER_CHANNEL_KEY})",
        file=sys.stderr,
    )


def write_fp8_mmdit(out, mmdit):
    """Quantize the checked-in tiny-mmdit fixture + golden its official forward.

    The golden is computed by loading the quantize-then-DEQUANTIZED weights
    into the pinned-commit SingleStreamDiT: the Rust fp8 load path and the
    official reference then run over mathematically identical weights, so the
    parity thresholds can match mmdit_parity.rs's.
    """
    src_dir = os.path.join(out, "tiny-mmdit")
    state = load_file(os.path.join(src_dir, "model.safetensors"))
    fp8_sd, dq = quantize_state(state)
    n_scales = sum(1 for k in fp8_sd if k.endswith(".weight_scale"))
    assert n_scales == 48, f"expected 48 quantized tensors, got {n_scales}"
    save_file(
        {k: v.contiguous() for k, v in fp8_sd.items()},
        os.path.join(src_dir, "model_fp8.safetensors"),
    )

    cfg_dict = mmdit_reference.TINY
    config = mmdit.SingleMMDiTConfig(**cfg_dict)
    with torch.device("meta"):
        model = mmdit.SingleStreamDiT(config)
    model.load_state_dict(dq, strict=True, assign=True)
    model = model.float().eval()

    # Inputs drawn exactly like mmdit_reference.run(), under this milestone's
    # seed (the golden carries its own inputs; only determinism matters).
    torch.manual_seed(15)  # M15
    b = 1
    latent = (
        torch.rand(
            b,
            cfg_dict["channels"],
            mmdit_reference.TINY_LATENT,
            mmdit_reference.TINY_LATENT,
        )
        * 2
        - 1
    )
    txtlen = mmdit_reference.TINY_TXTLEN
    txtmask = torch.ones(b, txtlen, dtype=torch.bool)
    txtmask[0, -1] = False  # one masked text position: the key mask must bite
    img_tokens, pos, mask = mmdit_reference.prepare(
        latent, txtlen, cfg_dict["patch"], txtmask
    )
    context = torch.randn(b, txtlen, cfg_dict["txtlayers"], cfg_dict["txtdim"])
    t = torch.full((b,), 0.5)

    stages = mmdit_reference.staged_forward(model, img_tokens, context, t, pos, mask)

    mmdit_reference.dump(
        os.path.join(out, "fp8_mmdit_golden.json"),
        {
            "latent": latent,
            "img_tokens": img_tokens,
            "context": context,
            "pos": pos,
            **stages,
        },
        {
            "config": cfg_dict,
            "txtmask": txtmask.long().flatten().tolist(),
            "t": t.tolist(),
            "krea2_commit": mmdit_reference.KREA2_COMMIT,
            "safetensors_keys": sorted(fp8_sd.keys()),
            "provenance": {
                "torch": torch.__version__,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        },
    )
    print(
        f"fp8 tiny-mmdit fixture saved to {src_dir} ({len(fp8_sd)} tensors)",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="crates/loractl-core/tests/fixtures")
    args = parser.parse_args()

    write_lut_golden(args.out)
    write_dequant_golden(args.out)
    write_tiny_krea2_twins(args.out)
    with tempfile.TemporaryDirectory() as tmp:
        mmdit = mmdit_reference.fetch_krea2(tmp)
        write_fp8_mmdit(args.out, mmdit)


if __name__ == "__main__":
    main()
