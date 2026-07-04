# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.*", "transformers", "safetensors", "numpy"]
# ///
"""Tiny-GPT-2 PyTorch reference for loractl M3 (issue #2) forward-parity.

Builds a REAL HF GPT-2 (`GPT2LMHeadModel`) at a *tiny* fixed config with seeded
random weights, saves its safetensors + config, and dumps golden logits plus
localizing intermediate activations on a FIXED token-ID input. Because the burn
implementation loads the SAME checked-in safetensors, both frameworks run
identical weights and differ only by f32 rounding — the exact M2 toy strategy,
lifted to a real transformer architecture.

The tiny weights + goldens are small enough to check in, so `cargo test` proves
load + forward parity fully offline. The opt-in test downloads the real
`openai-community/gpt2` and compares against a separately-generated golden.

Emitted (to --out dir): tiny-gpt2/{model.safetensors, config.json} and
gpt2_tiny_golden.json {config, input_ids, logits, hidden_after_embed,
hidden_after_block0, hidden_after_lnf, safetensors_keys, provenance}.
"""

import argparse
import json
import platform
import sys

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

# Fixed tiny architecture — real GPT-2 shape, minimal dims. Duplicated in the
# Rust test's config so the burn module tree matches exactly.
CFG = dict(
    vocab_size=61,
    n_positions=16,
    n_embd=32,
    n_layer=2,
    n_head=2,
    n_inner=64,           # 4 * n_embd
    activation_function="gelu_new",  # tanh-approx GELU — MUST match burn Gelu::new_approximate
    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,  # no dropout — deterministic
    layer_norm_epsilon=1e-5,
    tie_word_embeddings=True,
)
INPUT_IDS = [5, 12, 7, 3, 42, 1, 0, 9]  # fixed, len 8 < n_positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(1234)
    torch.set_default_dtype(torch.float32)

    config = GPT2Config(**CFG)
    model = GPT2LMHeadModel(config).eval()

    import os
    save_dir = os.path.join(args.out, "tiny-gpt2")
    os.makedirs(save_dir, exist_ok=True)
    # save_pretrained writes model.safetensors + config.json.
    model.save_pretrained(save_dir, safe_serialization=True)

    ids = torch.tensor([INPUT_IDS], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        logits = out.logits[0]                   # [seq, vocab]
        hidden = out.hidden_states               # tuple: embed, block0, block1(=pre-ln_f input)
        after_embed = hidden[0][0]               # [seq, n_embd]
        after_block0 = hidden[1][0]
        # ln_f applied to the last hidden state to get the pre-head normed features:
        after_lnf = model.transformer.ln_f(hidden[-1])[0]

    from safetensors import safe_open
    st_path = os.path.join(save_dir, "model.safetensors")
    with safe_open(st_path, framework="pt") as f:
        keys = sorted(f.keys())

    golden = {
        "provenance": {
            "torch": torch.__version__,
            "platform": platform.platform(),
            "note": "tiny HF GPT2LMHeadModel, seed 1234; burn loads the same safetensors",
        },
        "config": CFG,
        "input_ids": INPUT_IDS,
        "logits": logits.numpy().astype(np.float32).flatten().tolist(),
        "logits_shape": list(logits.shape),
        "hidden_after_embed": after_embed.numpy().astype(np.float32).flatten().tolist(),
        "hidden_after_block0": after_block0.numpy().astype(np.float32).flatten().tolist(),
        "hidden_after_lnf": after_lnf.numpy().astype(np.float32).flatten().tolist(),
        "hidden_shape": list(after_embed.shape),
        "safetensors_keys": keys,
    }
    with open(os.path.join(args.out, "gpt2_tiny_golden.json"), "w") as f:
        json.dump(golden, f, indent=2)

    top1 = int(logits[-1].argmax())
    print(f"tiny-gpt2 saved to {save_dir}", file=sys.stderr)
    print(f"safetensors keys ({len(keys)}):", file=sys.stderr)
    for k in keys:
        print(f"  {k}", file=sys.stderr)
    print(f"logits shape {list(logits.shape)}; last-token top1 = {top1}", file=sys.stderr)
    print(f"logits[-1][:5] = {logits[-1][:5].tolist()}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
