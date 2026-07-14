# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.*", "diffusers>=0.35", "transformers>=4.57", "tokenizers", "safetensors", "numpy", "pillow"]
# ///
"""The composed tiny-Krea-2 bundle for loractl M14 (issue #25) — the offline
end-to-end `DiffusionTrainer` fixture.

The M9–M11 tiny fixtures each pin one component's parity, but their dims were
chosen per-milestone (deliberately non-degenerate) and do NOT compose. This
script assembles a **dimension-matched** bundle in the exact `krea/Krea-2-Raw`
HF layout, so the trainer's real loading path runs unchanged against it:

```text
tiny-krea2/
  raw.safetensors                          the MMDiT (bundle dims below)
  text_encoder/model.safetensors           the tiny Qwen3-VL (M10's TINY_TEXT)
  tokenizer/tokenizer.json                 a tiny WordLevel tokenizer
  vae/diffusion_pytorch_model.safetensors  the tiny Qwen-Image VAE (M9's TINY_CFG)
dataset-tiny/                              4 images + captions (the e2e dataset)
```

Matched seams (each asserted below):
- VAE `z_dim` (4)          == MMDiT `channels`
- encoder `hidden` (32)    == MMDiT `txtdim`
- encoder `len(select)` (2) == MMDiT `txtlayers`

The MMDiT weights here are plain seeded-random tensors written key-by-key in
the verified `raw.safetensors` layout — **no parity claim** (that is
`mmdit_reference.py`'s job); the e2e test only needs a trainable,
correctly-shaped model. Norm `scale` params are ZERO (the zero-centered
convention) and modulation `lin`s are zero, matching the real init semantics.

The tokenizer is a WordLevel over a tiny vocab with `<unk>` — the chat
template's exotic tokens mostly map to `<unk>`, which is fine: the tiny e2e
proves the PIPELINE (tokenize → condition → denoise → step), not semantic
template alignment (that is pinned by the real-weights M10 proof).
"""

import os
import sys

import torch
from safetensors.torch import save_file

# The single source of truth for the tiny component dims: the sibling
# reference scripts whose dicts the Rust tiny configs are parity-pinned to.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen3vl_reference import TINY_SELECT_LAYERS, TINY_TEXT, TINY_VISION  # noqa: E402
from qwen_vae_reference import TINY_CFG as VAE_TINY_CFG  # noqa: E402

# The bundle MMDiT (MmditConfig::tiny_krea2() in Rust — keep in sync).
MMDIT = dict(
    features=64,
    tdim=16,
    txtdim=32,
    heads=4,
    kvheads=2,
    multiplier=4,
    layers=2,
    patch=2,
    channels=4,
    txtheads=2,
    txtkvheads=2,
    txtlayers=2,
)

OUT = "crates/loractl-core/tests/fixtures"


def swiglu_dim(features: int, multiplier: int) -> int:
    raw = (2 * features // 3) * multiplier
    return -(-raw // 128) * 128


def mmdit_state() -> dict:
    """Seeded-random tensors in the verified raw.safetensors key layout."""
    g = torch.Generator().manual_seed(14)  # M14

    def lin(out_d, in_d, scale=0.02):
        return torch.randn(out_d, in_d, generator=g) * scale

    f = MMDIT["features"]
    hd = f // MMDIT["heads"]
    kv_out = hd * MMDIT["kvheads"]
    inner = swiglu_dim(f, MMDIT["multiplier"])
    td = MMDIT["txtdim"]
    thd = td // MMDIT["txtheads"]
    tkv_out = thd * MMDIT["txtkvheads"]
    tinner = swiglu_dim(td, MMDIT["multiplier"])

    state = {
        "first.weight": lin(f, MMDIT["channels"] * MMDIT["patch"] ** 2),
        "first.bias": torch.zeros(f),
        "tmlp.0.weight": lin(f, MMDIT["tdim"]),
        "tmlp.0.bias": torch.zeros(f),
        "tmlp.2.weight": lin(f, f),
        "tmlp.2.bias": torch.zeros(f),
        "tproj.1.weight": lin(6 * f, f),
        "tproj.1.bias": torch.zeros(6 * f),
        "txtmlp.0.scale": torch.zeros(td),
        "txtmlp.1.weight": lin(f, td),
        "txtmlp.1.bias": torch.zeros(f),
        "txtmlp.3.weight": lin(f, f),
        "txtmlp.3.bias": torch.zeros(f),
        "txtfusion.projector.weight": lin(1, MMDIT["txtlayers"], scale=0.5),
        "last.norm.scale": torch.zeros(f),
        "last.linear.weight": lin(MMDIT["patch"] ** 2 * MMDIT["channels"], f),
        "last.linear.bias": torch.zeros(MMDIT["patch"] ** 2 * MMDIT["channels"]),
        "last.modulation.lin": torch.zeros(2, f),
    }

    def attn(prefix, dim, n_heads, n_kv):
        head_dim = dim // n_heads
        state[f"{prefix}.wq.weight"] = lin(head_dim * n_heads, dim)
        state[f"{prefix}.wk.weight"] = lin(head_dim * n_kv, dim)
        state[f"{prefix}.wv.weight"] = lin(head_dim * n_kv, dim)
        state[f"{prefix}.gate.weight"] = lin(dim, dim)
        state[f"{prefix}.wo.weight"] = lin(dim, dim)
        state[f"{prefix}.qknorm.qnorm.scale"] = torch.zeros(head_dim)
        state[f"{prefix}.qknorm.knorm.scale"] = torch.zeros(head_dim)

    def swiglu(prefix, dim, inner_dim):
        state[f"{prefix}.gate.weight"] = lin(inner_dim, dim)
        state[f"{prefix}.up.weight"] = lin(inner_dim, dim)
        state[f"{prefix}.down.weight"] = lin(dim, inner_dim)

    for i in range(MMDIT["layers"]):
        p = f"blocks.{i}"
        state[f"{p}.mod.lin"] = torch.zeros(6 * f)
        state[f"{p}.prenorm.scale"] = torch.zeros(f)
        state[f"{p}.postnorm.scale"] = torch.zeros(f)
        attn(f"{p}.attn", f, MMDIT["heads"], MMDIT["kvheads"])
        swiglu(f"{p}.mlp", f, inner)

    for group in ("layerwise_blocks", "refiner_blocks"):
        for i in range(2):
            p = f"txtfusion.{group}.{i}"
            state[f"{p}.prenorm.scale"] = torch.zeros(td)
            state[f"{p}.postnorm.scale"] = torch.zeros(td)
            attn(f"{p}.attn", td, MMDIT["txtheads"], MMDIT["txtkvheads"])
            swiglu(f"{p}.mlp", td, tinner)

    _ = (kv_out, tkv_out)  # shapes derive inside attn(); kept for readability
    return {k: v.contiguous() for k, v in state.items()}


def build_tokenizer(path: str):
    """A tiny WordLevel tokenizer with the caption vocabulary + pad token."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    words = [
        "<unk>",
        "<|endoftext|>",
        "a",
        "red",
        "fox",
        "green",
        "field",
        "blue",
        "sky",
        "gold",
        "sand",
        "photo",
        "of",
    ]
    vocab = {w: i for i, w in enumerate(words)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tokenizer.save(path)


def build_dataset(path: str):
    """Four tiny distinct gradient PNGs + captions from the tokenizer vocab."""
    from PIL import Image
    import numpy as np

    os.makedirs(path, exist_ok=True)
    specs = [
        ("fox.png", (200, 60, 30), "a photo of a red fox"),
        ("field.png", (40, 180, 60), "a photo of a green field"),
        ("sky.png", (50, 90, 220), "a photo of a blue sky"),
        ("sand.png", (220, 190, 90), "a photo of gold sand"),
    ]
    for name, (r, g, b), caption in specs:
        # A 48x32 gradient tinted per image (distinct, deterministic).
        x = np.linspace(0.3, 1.0, 48)[None, :, None]
        y = np.linspace(0.5, 1.0, 32)[:, None, None]
        base = np.array([r, g, b])[None, None, :] * x * y
        img = Image.fromarray(base.astype("uint8"), "RGB")
        img.save(os.path.join(path, name))
        with open(os.path.join(path, name.replace(".png", ".txt")), "w") as fh:
            fh.write(caption + "\n")


def main():
    torch.manual_seed(14)

    # --- Assert the seams the Rust configs rely on. ---
    assert VAE_TINY_CFG["z_dim"] == MMDIT["channels"], (
        "vae z_dim must equal mmdit channels"
    )
    assert TINY_TEXT["hidden_size"] == MMDIT["txtdim"], (
        "encoder hidden must equal mmdit txtdim"
    )
    assert len(TINY_SELECT_LAYERS) == MMDIT["txtlayers"], (
        "select count must equal txtlayers"
    )

    bundle = os.path.join(OUT, "tiny-krea2")
    os.makedirs(bundle, exist_ok=True)

    # 1. MMDiT — random weights in the verified raw.safetensors layout.
    save_file(mmdit_state(), os.path.join(bundle, "raw.safetensors"))

    # 2. Text encoder — the SAME class + dims as the M10 tiny fixture (its
    # dims are what the Rust Qwen3VlConfig::tiny() is pinned to).
    from transformers import Qwen3VLModel
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    encoder = Qwen3VLModel(
        Qwen3VLConfig(
            text_config=Qwen3VLTextConfig(**TINY_TEXT),
            vision_config=Qwen3VLVisionConfig(**TINY_VISION),
        )
    ).eval()
    te_dir = os.path.join(bundle, "text_encoder")
    os.makedirs(te_dir, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in encoder.state_dict().items()},
        os.path.join(te_dir, "model.safetensors"),
    )

    # 3. VAE — the SAME class + dims as the M9 tiny fixture.
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage(**VAE_TINY_CFG).eval()
    vae_dir = os.path.join(bundle, "vae")
    os.makedirs(vae_dir, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in vae.state_dict().items()},
        os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"),
    )

    # 4. Tokenizer + 5. dataset.
    build_tokenizer(os.path.join(bundle, "tokenizer", "tokenizer.json"))
    build_dataset(os.path.join(OUT, "dataset-tiny"))

    n = len(mmdit_state())
    print(
        f"tiny-krea2 bundle written to {bundle} (mmdit tensors: {n})", file=sys.stderr
    )


if __name__ == "__main__":
    main()
