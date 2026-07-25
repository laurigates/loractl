#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate the golden set of LoRA `__metadata__` keys a real consumer reads.

This is the **consumer contract** for `crates/loractl-core/src/metadata.rs`,
the sibling of `krea2_lora_keys_reference.py`. That one answers "does the
consumer accept our tensor keys"; this one answers "does the consumer read the
metadata keys we write" — the same silent failure mode one layer up, where the
file loads fine and the UI shows nothing.

## Why AUTOMATIC1111's Lora extension

It is the consumer that actually *reads* LoRA metadata and is open source:

- **ComfyUI** loads the tensors and ignores `__metadata__` entirely, so it can
  neither confirm nor deny a metadata key (its tensor-key contract is the
  other reference script's job).
- **Civitai** is closed source; its keys cannot be pinned to anything.
- **A1111 / Forge** parse `ss_*` and `sshs_*` in `extensions-builtin/Lora/`,
  display them, and — the load-bearing one — turn `ss_tag_frequency` into the
  suggested activation text for a LoRA. Forge and the various reForge/SD-Next
  descendants inherit this code.

## What is extracted

Every string literal matching `ss_*` / `sshs_*` / `modelspec.*` in the two
files that touch metadata. Deliberately broader than "arguments to
`metadata.get()`": the display table builds its rows from tuple literals, not
`.get` calls, and a narrow extractor that silently missed those would produce
a golden that *understates* the contract — the failure this script exists to
prevent. Over-collection is harmless: the Rust test asserts every collected
key is either written by loractl or carries a recorded reason for not being.

The `REQUIRED_KEYS` assertion below is the kill-switch: if a restructure (or a
wrong path, or an HTML error page) means those never appear, this fails loudly
rather than emitting a thin golden that trivially passes.

Usage: just lora-metadata-keys-reference
"""

import argparse
import ast
import json
import re
import sys
import urllib.request
from pathlib import Path

# AUTOMATIC1111/stable-diffusion-webui @ v1.10.1 (a release TAG, not a moving
# branch). Bump deliberately, and re-read `build_tags` in
# ui_edit_user_metadata.py when you do — that function is the trigger-word path.
WEBUI_REF = "v1.10.1"
WEBUI_RAW = "https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/{ref}/{path}"

# The files in the Lora extension that touch `__metadata__`.
SOURCES = [
    "extensions-builtin/Lora/network.py",
    "extensions-builtin/Lora/ui_edit_user_metadata.py",
]

# Keys we KNOW this consumer reads, asserted present before a golden is
# emitted. `ss_tag_frequency` is the activation-text path and `sshs_model_hash`
# the file identity — if either is missing, we are not looking at the code we
# think we are.
REQUIRED_KEYS = ["ss_tag_frequency", "sshs_model_hash", "ss_output_name"]

# A metadata key literal: the two kohya families plus ModelSpec. Applied with
# `fullmatch`, not `match` — Python's `$` also matches BEFORE a trailing
# newline, so `"ss_output_name\n"` would sneak into the golden under `^...$`
# and then fail the Rust contract test as an unwritten key whose name looks
# identical to one we write.
KEY_RE = re.compile(r"ss_[a-z0-9_]+|sshs_[a-z0-9_]+|modelspec\.[a-z0-9_]+")


def fetch(path: str) -> str:
    url = WEBUI_RAW.format(ref=WEBUI_REF, path=path)
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def metadata_keys(source: str, path: str) -> set[str]:
    """Every metadata-shaped string literal in `source`, via AST.

    Two filters, doing different jobs:

    - **AST** drops comments (they are not nodes at all), so a key named in a
      `# TODO: also read ss_foo` cannot inflate the contract.
    - **`KEY_RE.fullmatch`** is what handles docstrings — those ARE
      `ast.Constant` nodes and `ast.walk` visits them, but a key mentioned
      inside a sentence is part of a longer string and cannot match.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:  # a fetched error page, most likely
        raise SystemExit(f"FAIL: {path} did not parse as Python ({e})") from e
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and KEY_RE.fullmatch(node.value)
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="golden output dir")
    args = ap.parse_args()

    keys: set[str] = set()
    for path in SOURCES:
        src = fetch(path)
        found = metadata_keys(src, path)
        print(f"  {path}: {len(found)} metadata keys", file=sys.stderr)
        keys |= found

    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        raise SystemExit(
            f"FAIL: pinned source no longer mentions {missing}. Upstream "
            f"restructured the Lora extension, or SOURCES is stale — re-read "
            f"the files before emitting a golden."
        )

    golden = {
        "consumer": "AUTOMATIC1111/stable-diffusion-webui extensions-builtin/Lora",
        "ref": WEBUI_REF,
        "sources": SOURCES,
        "keys_read": sorted(keys),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "lora_metadata_keys.json"
    out.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"wrote {out} ({len(keys)} keys)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
