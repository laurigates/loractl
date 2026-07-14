# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.*", "transformers>=4.57", "safetensors", "numpy"]
# ///
"""Qwen3-VL text-conditioner PyTorch reference for loractl M10 (issue #21).

Krea 2's conditioner (`krea-ai/krea-2` encoder.py) is a frozen Qwen3-VL run
TEXT-ONLY: forward templated caption ids with output_hidden_states=True, stack
hidden_states at select_layers (2,5,8,...,35) on dim=2, slice off the template
prefix. This script generates the tiny fixture + staged golden for the burn
port (M3/M9 methodology).

Tiny mode: a REAL Qwen3VLForConditionalGeneration at a tiny config — WITH a
tiny vision tower, so the fixture carries `visual.*` keys and the burn load
test proves the text-only drop-filter. Skips the tokenizer (fixed in-range
ids); tokenizer parity is a real-mode concern. The golden includes a
RIGHT-PADDED batch row (mask with interior zeros, suffix-after-padding, like
encoder.py's padding="max_length" + suffix concat) so the port must get
key-padding masking + arange position ids right, not just causal masking.

Real mode (--real): downloads krea/Krea-2-Raw's OWN text_encoder + tokenizer
(the exact shipped conditioning weights), reproduces encoder.py's
template/tokenize/pad/concat path on a fixed caption, and dumps the golden
conditioning stack [b, seq, 12, 2560] + mask (as safetensors — too big for
JSON) plus the token ids (so the Rust tokenizer is parity-tested against the
same caption string). Tiny mode uses Qwen3VLModel — the same class as that
checkpoint — so the two key spaces match.
"""

import argparse
import json
import os
import platform
import sys

import torch
from transformers import Qwen3VLModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

# Tiny architecture — real Qwen3-VL structure at minimal dims. Duplicated in
# the Rust tiny config. GQA, per-head q/k RMS-norm, SwiGLU, real rope_theta;
# mrope_section sums to head_dim/2 (structure faithful; in text-only mode all
# three streams share position ids so it collapses to plain half-split RoPE).
#
# Deliberately NON-DEGENERATE, like the real 4B config (head_dim 128 !=
# 2560/32): head_dim (6) != hidden/heads (32/6), heads (6) != kv (2) !=
# GQA groups (3), and heads*head_dim (36) != hidden (32) so the projections
# are non-square. A port that conflates any of these passes a "square"
# fixture and only fails on the real model — this fixture makes the
# always-run parity test catch it.
TINY_TEXT = dict(
    hidden_size=32,
    num_hidden_layers=4,
    num_attention_heads=6,
    num_key_value_heads=2,
    head_dim=6,
    intermediate_size=64,
    vocab_size=93,
    max_position_embeddings=64,
    rms_norm_eps=1e-6,
    rope_theta=5_000_000,
    rope_scaling=dict(
        rope_type="default", mrope_section=[1, 1, 1], mrope_interleaved=True
    ),
    attention_bias=False,
    tie_word_embeddings=True,
)
# The vision tower is present ONLY so the fixture carries visual.* keys the
# text-only load must drop; it is never run.
TINY_VISION = dict(
    depth=2,
    hidden_size=16,
    intermediate_size=32,
    num_heads=2,
    out_hidden_size=32,
    patch_size=4,
    spatial_merge_size=1,
    temporal_patch_size=1,
    num_position_embeddings=16,
    deepstack_visual_indexes=[0, 1],
)
TINY_SELECT_LAYERS = (1, 3)  # of 4 layers; hidden_states[0] is the embedding
TINY_PREFIX_IDX = 3  # the template-prefix slice analogue of the real 34
SEQ = 12


def staged_forward(model, input_ids, attention_mask, select_layers, prefix_idx):
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask.bool(),
            output_hidden_states=True,
        )
    hs = out.hidden_states
    stacked = torch.stack([hs[i] for i in select_layers], dim=2)
    return {
        "after_embed": hs[0],
        "hidden_first_select": hs[select_layers[0]],
        "hidden_last_select": hs[select_layers[-1]],
        "conditioning": stacked[:, prefix_idx:],
    }, len(hs)


def dump(path, tensors, extra):
    golden = dict(extra)
    for name, t in tensors.items():
        golden[name] = t.flatten().tolist()
        golden[f"{name}_shape"] = list(t.shape)
    with open(path, "w") as f:
        json.dump(golden, f)
    print(f"golden written to {path}", file=sys.stderr)


# encoder.py's template constants, verbatim (krea-ai/krea-2).
REAL_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
REAL_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
REAL_PREFIX_IDX = 34
REAL_SUFFIX_START_IDX = 5
REAL_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
# Reduced from encoder.py's 512 to keep the golden a manageable size — the
# path under test (template, right-padding, suffix-after-padding concat,
# select-layer stacking, prefix slice) is length-independent.
REAL_MAX_LENGTH = 64
REAL_CAPTION = "a watercolor painting of a red fox curled up on mossy stones"


def real_mode(args):
    """Reproduce encoder.py's exact conditioning path on the REAL Krea-2-Raw
    text encoder + tokenizer, and dump fixtures for the opt-in Rust test.

    Emits (all uncommitted, gitignored):
      qwen3vl-real/            f32 re-save of Krea-2-Raw's text_encoder
      qwen3vl-real/tokenizer/  Krea-2-Raw's tokenizer files (tokenizer.json …)
      qwen3vl_real_golden.safetensors  conditioning stack + sliced mask
      qwen3vl_real_golden.json         token ids, mask, shapes, provenance
    """
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    model = Qwen3VLModel.from_pretrained("krea/Krea-2-Raw", subfolder="text_encoder")
    model = model.to(torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained("krea/Krea-2-Raw", subfolder="tokenizer")

    save_dir = os.path.join(args.out, "qwen3vl-real")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

    # encoder.py's forward, verbatim modulo max_length: right-pad the
    # templated body to (max_length + prefix_idx - suffix_start_idx), then
    # CONCATENATE the separately-tokenized suffix after the padding.
    text = [REAL_PREFIX + REAL_CAPTION]
    body = tokenizer(
        text,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        padding="max_length",
        max_length=REAL_MAX_LENGTH + REAL_PREFIX_IDX - REAL_SUFFIX_START_IDX,
        return_tensors="pt",
    )
    suffix = tokenizer(text=[REAL_SUFFIX], return_tensors="pt")
    input_ids = torch.cat([body["input_ids"], suffix["input_ids"]], dim=1)
    mask = torch.cat([body["attention_mask"], suffix["attention_mask"]], dim=1)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=mask.bool(),
            output_hidden_states=True,
        )
    hiddens = torch.stack([out.hidden_states[i] for i in REAL_SELECT_LAYERS], dim=2)[
        :, REAL_PREFIX_IDX:
    ]
    mask_sliced = mask[:, REAL_PREFIX_IDX:]

    # The conditioning stack is too large for JSON — safetensors carries the
    # tensors; JSON carries the ids/shapes/provenance the Rust test needs.
    save_file(
        {
            "conditioning": hiddens.contiguous(),
            "mask_sliced": mask_sliced.contiguous(),
        },
        os.path.join(args.out, "qwen3vl_real_golden.safetensors"),
    )
    from safetensors import safe_open

    with safe_open(os.path.join(save_dir, "model.safetensors"), framework="pt") as f:
        keys = sorted(f.keys())
    with open(os.path.join(args.out, "qwen3vl_real_golden.json"), "w") as f:
        json.dump(
            {
                "caption": REAL_CAPTION,
                "input_ids": input_ids.flatten().tolist(),
                "input_shape": list(input_ids.shape),
                "attention_mask": mask.flatten().tolist(),
                "select_layers": list(REAL_SELECT_LAYERS),
                "prefix_idx": REAL_PREFIX_IDX,
                "max_length": REAL_MAX_LENGTH,
                "conditioning_shape": list(hiddens.shape),
                "safetensors_keys": keys,
                "provenance": {
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                },
            },
            f,
        )
    print(f"real qwen3vl saved to {save_dir} ({len(keys)} tensors)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".")
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(10)  # M10

    if args.real:
        real_mode(args)
        return

    # Qwen3VLModel — the bare trunk (visual + language_model, no lm_head):
    # the SAME class Krea-2-Raw ships as its text_encoder/, so the fixture's
    # key space (`language_model.*`, `visual.*`) matches the real checkpoint.
    config = Qwen3VLConfig(
        text_config=Qwen3VLTextConfig(**TINY_TEXT),
        vision_config=Qwen3VLVisionConfig(**TINY_VISION),
    )
    model = Qwen3VLModel(config).eval()

    save_dir = os.path.join(args.out, "tiny-qwen3vl")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)

    from safetensors import safe_open

    st_path = os.path.join(save_dir, "model.safetensors")
    with safe_open(st_path, framework="pt") as f:
        keys = sorted(f.keys())

    vocab = TINY_TEXT["vocab_size"]
    gen = torch.Generator().manual_seed(1010)
    input_ids = torch.randint(0, vocab, (2, SEQ), generator=gen)
    # Row 0: full attention. Row 1: right-padded body with a live tail —
    # positions 7..9 masked, 10..11 live (the suffix-after-padding shape).
    attention_mask = torch.ones(2, SEQ, dtype=torch.long)
    attention_mask[1, 7:10] = 0

    stages, n_hidden = staged_forward(
        model, input_ids, attention_mask, TINY_SELECT_LAYERS, TINY_PREFIX_IDX
    )
    mask_sliced = attention_mask[:, TINY_PREFIX_IDX:]

    dump(
        os.path.join(args.out, "qwen3vl_tiny_golden.json"),
        {**stages, "mask_sliced": mask_sliced},
        {
            "input_ids": input_ids.flatten().tolist(),
            "input_shape": list(input_ids.shape),
            "attention_mask": attention_mask.flatten().tolist(),
            "select_layers": list(TINY_SELECT_LAYERS),
            "prefix_idx": TINY_PREFIX_IDX,
            "num_hidden_states": n_hidden,
            "safetensors_keys": keys,
            "provenance": {
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        },
    )
    n_visual = sum(1 for k in keys if k.startswith("visual."))
    print(
        f"tiny qwen3vl saved to {save_dir} ({len(keys)} tensors, {n_visual} visual)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
