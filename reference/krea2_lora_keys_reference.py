#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate the golden set of LoRA keys ComfyUI's Krea 2 loader accepts.

This is the **consumer contract** for `export_adapters(.., Krea2Diffusers, ..)`:
loractl's exported key names are only useful if ComfyUI's LoRA key map contains
them. Issue #137 was filed on the belief that it did not; the belief came from
comparing our key names against community LoRAs rather than against ComfyUI's
own map, and it was wrong (our keys match 196/196). This golden exists so that
question is settled mechanically, in either direction, forever after.

Rather than transcribe ComfyUI's mapping (a hand-copy that silently drifts), we
download `comfy/utils.py` and `comfy/lora.py` at a PINNED COMMIT, extract the
real `krea2_to_diffusers` source with `ast`, and execute it. It is pure
dict/str/range code, so it runs standalone without torch.

The `Krea2` registration block in `model_lora_keys_unet` cannot be executed
standalone (it needs a live model), so instead we *assert* the four alias lines
we depend on are still present in the pinned source, then apply them ourselves.
If upstream restructures that block, this script fails loudly rather than
emitting a golden that silently no longer reflects the consumer.

Usage: just krea2-lora-keys-reference
"""

import argparse
import ast
import json
import sys
import urllib.request
from pathlib import Path

# ComfyUI master @ 2026-07-20. Bump deliberately, and re-read the Krea2 block in
# `comfy/lora.py::model_lora_keys_unet` when you do.
COMFY_COMMIT = "5697b970173bc0c16a05c30d509d0911f2b84822"
COMFY_RAW = "https://raw.githubusercontent.com/comfyanonymous/ComfyUI/{commit}/{path}"

# The four aliases the Krea2 branch registers per diffusers key. We assert each
# of these appears verbatim in the pinned source before relying on it.
REQUIRED_ALIAS_LINES = [
    'key_map["diffusion_model.{}".format(key_lora)] = to',
    'key_map["transformer.{}".format(key_lora)] = to',
    'key_map["lycoris_{}".format(key_lora.replace(".", "_"))] = to',
    "key_map[key_lora] = to",
]


def fetch(path: str) -> str:
    url = COMFY_RAW.format(commit=COMFY_COMMIT, path=path)
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def extract_function(source: str, name: str) -> str:
    """Return the source of a top-level function, via AST (no import needed)."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise SystemExit(f"FAIL: {name} not found in the pinned ComfyUI source")


def load_krea2_to_diffusers(utils_src: str):
    """Exec the real `krea2_to_diffusers` in an empty namespace and return it."""
    ns: dict = {}
    exec(extract_function(utils_src, "krea2_to_diffusers"), ns)  # noqa: S102
    return ns["krea2_to_diffusers"]


def assert_krea2_branch(lora_src: str) -> None:
    """Fail loudly if the Krea2 alias registration is no longer what we model."""
    marker = "if isinstance(model, comfy.model_base.Krea2):"
    if marker not in lora_src:
        raise SystemExit(
            "FAIL: the Krea2 branch is gone from model_lora_keys_unet — "
            "the export's consumer contract has changed; re-read comfy/lora.py"
        )
    block = lora_src.split(marker, 1)[1].split("\n    if isinstance(", 1)[0]
    missing = [line for line in REQUIRED_ALIAS_LINES if line not in block]
    if missing:
        raise SystemExit(
            "FAIL: the Krea2 branch no longer registers these aliases:\n  "
            + "\n  ".join(missing)
            + "\n\nloractl emits the BARE diffusers key; if that alias is gone, "
            "export.rs must switch to a surviving form."
        )


def build_accepted_keys(krea2_to_diffusers, layers: int) -> list[str]:
    """Every LoRA site key ComfyUI's Krea 2 path accepts, at this depth.

    `krea2_to_diffusers` returns `{diffusers_key.weight: diffusion_model.native.weight}`.
    ComfyUI registers four aliases per diffusers key (the Krea2 branch), and the
    generic loop at the top of `model_lora_keys_unet` separately registers every
    state-dict key bare — and this map's *values* are state-dict keys by
    construction, so the native form is derivable here without the checkpoint.
    """
    cfg = {"layers": layers}
    dkeys = krea2_to_diffusers(cfg, output_prefix="diffusion_model.")

    accepted: set[str] = set()
    for k, to in dkeys.items():
        if not k.endswith(".weight"):
            continue
        key_lora = k[: -len(".weight")]
        # The Krea2 branch's four aliases.
        accepted.add(f"diffusion_model.{key_lora}")
        accepted.add(f"transformer.{key_lora}")
        accepted.add("lycoris_" + key_lora.replace(".", "_"))
        accepted.add(key_lora)  # <-- the bare diffusers key loractl emits
        # The generic loop: every state-dict key, bare.
        accepted.add(to[: -len(".weight")])
    return sorted(accepted)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="tests/golden dir")
    ap.add_argument("--layers", type=int, default=28, help="Krea 2 trunk depth")
    args = ap.parse_args()

    utils_src = fetch("comfy/utils.py")
    lora_src = fetch("comfy/lora.py")
    assert_krea2_branch(lora_src)
    krea2_to_diffusers = load_krea2_to_diffusers(utils_src)

    accepted = build_accepted_keys(krea2_to_diffusers, args.layers)

    golden = {
        "_comment": (
            "LoRA site keys ComfyUI's Krea 2 loader accepts. Generated from the "
            "pinned ComfyUI source by reference/krea2_lora_keys_reference.py — "
            "do not hand-edit. Regenerate: just krea2-lora-keys-reference"
        ),
        "comfyui_commit": COMFY_COMMIT,
        "layers": args.layers,
        "accepted_keys": accepted,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "krea2_lora_keys.json"
    dest.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {dest} ({len(accepted)} accepted keys, layers={args.layers})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
