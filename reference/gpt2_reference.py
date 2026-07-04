# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.*", "transformers", "safetensors", "numpy"]
# ///
"""Real pretrained GPT-2 reference for loractl M3 (issue #2) forward-parity.

Downloads `openai-community/gpt2` (GPT-2 small, 124M), saves its safetensors,
and dumps golden logits (+ localizing intermediate activations) on a FIXED
token-ID input. The burn `gpt2-real` test (`tests/gpt2_real.rs`) loads the SAME
safetensors and asserts logit parity — the pretrained-weights counterpart of the
always-run tiny-fixture parity test.

The real weights (~500 MB) and their golden are large, so — unlike the tiny
fixture — they are NOT checked in. This script writes them under
`crates/loractl-core/tests/fixtures/` where the opt-in test reads them:

    crates/loractl-core/tests/fixtures/gpt2-real/model.safetensors
    crates/loractl-core/tests/fixtures/gpt2_real_golden.json

Run via `just gpt2-reference` (needs network + torch/transformers via uv), then
`just test-gpt2-real`.

NOTE (parity semantics, learned from the tiny reference): HF's last
`output_hidden_states` entry is ALREADY `ln_f`-applied, so `hidden_states[-1] @
wteᵀ` reproduces the logits exactly. We therefore emit `hidden_after_lnf =
model.transformer.ln_f(hidden_states[-1])` to mirror the tiny golden's
(double-normed) convention; the burn test's authoritative check is the logits.
"""

import argparse
import json
import os
import platform
import sys

import numpy as np
import torch
from transformers import GPT2LMHeadModel

MODEL = "openai-community/gpt2"
# Fixed, deterministic input (short; well within n_positions=1024).
INPUT_IDS = [15496, 11, 314, 1101, 257, 1332, 286]  # "Hello, I'm a test of"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="crates/loractl-core/tests/fixtures",
        help="fixtures dir (default: crates/loractl-core/tests/fixtures)",
    )
    args = ap.parse_args()

    torch.set_default_dtype(torch.float32)
    model = GPT2LMHeadModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

    save_dir = os.path.join(args.out, "gpt2-real")
    os.makedirs(save_dir, exist_ok=True)
    # Writes model.safetensors + config.json (tied head is not duplicated).
    model.save_pretrained(save_dir, safe_serialization=True)

    ids = torch.tensor([INPUT_IDS], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        logits = out.logits[0]  # [seq, vocab]
        hidden = out.hidden_states
        after_embed = hidden[0][0]
        after_block0 = hidden[1][0]
        # Mirror the tiny golden's convention (ln_f re-applied to hidden[-1]).
        after_lnf = model.transformer.ln_f(hidden[-1])[0]

    from safetensors import safe_open

    with safe_open(os.path.join(save_dir, "model.safetensors"), framework="pt") as f:
        keys = sorted(f.keys())

    golden = {
        "provenance": {
            "model": MODEL,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "note": "real pretrained GPT-2 small; burn loads the same safetensors",
        },
        "input_ids": INPUT_IDS,
        "logits": logits.numpy().astype(np.float32).flatten().tolist(),
        "logits_shape": list(logits.shape),
        "hidden_after_embed": after_embed.numpy().astype(np.float32).flatten().tolist(),
        "hidden_after_block0": after_block0.numpy().astype(np.float32).flatten().tolist(),
        "hidden_after_lnf": after_lnf.numpy().astype(np.float32).flatten().tolist(),
        "hidden_shape": list(after_embed.shape),
        "safetensors_keys": keys,
    }
    with open(os.path.join(args.out, "gpt2_real_golden.json"), "w") as f:
        json.dump(golden, f)

    top1 = int(logits[-1].argmax())
    print(f"real gpt2 saved to {save_dir} ({len(keys)} tensors)", file=sys.stderr)
    print(f"logits shape {list(logits.shape)}; last-token top1 = {top1}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
