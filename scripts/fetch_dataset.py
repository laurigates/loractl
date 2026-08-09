#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub>=0.24"]
# ///
"""Materialize a PUBLIC captioned image dataset into loractl's on-disk layout.

loractl reads a kohya-style folder: every `.png`/`.jpg`/`.jpeg` with an
optional same-stem `.txt` caption, scanned in filename order
(`dataset.rs::scan_dataset`). Hugging Face publishes the same content as an
`imagefolder` repo (loose images + a `metadata.jsonl` mapping file_name →
caption). This script is the adapter between the two, so a measurement run is
reproducible by someone who does not have the operator's disk.

Why a *public* set rather than a local folder of generations: #175 and #178
both close on before/after numbers, and a number nobody else can reproduce is
an assertion. A pinned public dataset lets a third party re-run the same
matrix and compare, and lets us compare a loractl LoRA against the many
published LoRAs trained on the same images.

Datasets are PINNED BY REVISION, not by tag or branch. An upstream re-upload
that silently changed the images would otherwise invalidate a recorded
measurement with nothing to show for it. Bump a pin deliberately, and re-run
the matrix when you do.

Determinism, which the measurement depends on:

  * Entries are sorted by file name, so `--limit 8` is a strict PREFIX of
    `--limit 56`. The small arm of the residency matrix is therefore a subset
    of the large arm, and the two differ only in example count -- which is the
    single variable #175's acceptance criterion is about.
  * Duplicate `file_name` rows keep the FIRST occurrence in sorted order
    (linoyts/Tuxemon has one: `Snorilla_upscaled.jpg` appears twice with
    different captions). Reported, not silently dropped.
  * Metadata rows naming a file that is not in the repo are skipped and
    reported (Tuxemon's metadata.jsonl has 254 rows against 251 images).

Only the files actually needed are downloaded (`--limit N` fetches N images,
not the whole repo), so a small arm costs a small download.

Usage:
    scripts/fetch_dataset.py --dataset tuxemon --out <dir> --limit 56
    scripts/fetch_dataset.py --list

Set HF_HOME to a disk with room before running -- the default
(~/.cache/huggingface) is on the small root disk on the GPU host.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    """A vetted public dataset, pinned to an immutable revision."""

    repo_id: str
    revision: str
    caption_field: str
    license: str
    attribution: str
    note: str


# Each entry is pinned to a commit sha, NOT to `main`. Verify a new pin with:
#   curl -s https://huggingface.co/api/datasets/<repo_id>/revision/main | jq .sha
REGISTRY: dict[str, Dataset] = {
    "tuxemon": Dataset(
        repo_id="linoyts/Tuxemon",
        revision="c12eb30553ed267fd9d68716f11d1c9725426be7",
        caption_field="prompt",
        license="cc-by-sa-3.0",
        attribution=(
            "Tuxemon monster art from https://wiki.tuxemon.org/Category:Monster "
            "(CC BY-SA 3.0); some images upscaled, captions generated with "
            "BLIP-large. Packaged as linoyts/Tuxemon."
        ),
        note=(
            "251 JPEGs, 45 MB, mean 2.4 Mpx. Measured shape (not assumed -- an "
            "early 10-image sample read as uniformly 2048x2048 and was wrong): "
            "226/251 are square and 206/251 are at least 2048 on both sides, so "
            "decode + Lanczos resize dominates the cold encode. That is exactly "
            "the work #178 parallelized, which is why this set was chosen. A "
            "~10% non-square tail spans aspect 0.58-1.67 and two images fall "
            "below 512 on a side, so bucket assignment is exercised but NOT "
            "stressed -- do not read a run over this set as evidence about #147 "
            "(no_upscale) or #148 (grid bucketing); those stay covered by the "
            "offline tests."
        ),
    ),
}

METADATA_FILE = "metadata.jsonl"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
# loractl scans these three extensions only; anything else is invisible to it.
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def sanitize(name: str) -> str:
    """Map a file name to the safe subset, preserving sort order within the set.

    Tuxemon carries two names with parentheses. Substitution is per-character
    so it cannot merge two distinct names into one unless they differed only in
    unsafe characters -- and that case is caught by the collision check below.
    """
    return _SAFE.sub("_", name)


def load_entries(meta_path: Path, caption_field: str, present: set[str]):
    """Parse metadata.jsonl into a deterministic (file_name, caption) list.

    Returns the entries plus the two classes of dropped row, so the caller can
    report them rather than have the counts quietly not add up.
    """
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    missing: list[str] = []

    with meta_path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            name = row.get("file_name")
            if not name:
                raise SystemExit(f"{meta_path}:{lineno}: row has no 'file_name'")
            caption = (row.get(caption_field) or "").strip()
            if not caption:
                raise SystemExit(f"{meta_path}:{lineno}: {name} has an empty caption")
            if name not in present:
                missing.append(name)
                continue
            if name in seen:
                duplicates.append(name)
                continue
            seen[name] = caption

    entries = sorted(seen.items())
    return entries, sorted(set(duplicates)), sorted(set(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="tuxemon", help="registry key")
    parser.add_argument("--out", type=Path, help="destination directory")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="materialize the first N entries in filename order (0 = all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-materialize even if the destination already holds the right count",
    )
    parser.add_argument(
        "--list", action="store_true", help="list the registry and exit"
    )
    args = parser.parse_args()

    if args.list:
        for key, spec in REGISTRY.items():
            print(f"{key}\t{spec.repo_id}@{spec.revision[:12]}\t{spec.license}")
            print(f"\t{spec.note}")
        return 0

    if args.dataset not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise SystemExit(f"unknown dataset {args.dataset!r}; known: {known}")
    if args.out is None:
        raise SystemExit("--out is required")

    spec = REGISTRY[args.dataset]

    # Imported here so `--list` and the argument errors stay instant and do not
    # need the dependency resolved.
    from huggingface_hub import HfApi, hf_hub_download

    if not os.environ.get("HF_HOME"):
        print(
            "warning: HF_HOME is unset -- the download cache goes to "
            "~/.cache/huggingface on the root disk",
            file=sys.stderr,
        )

    api = HfApi()
    files = api.list_repo_files(
        spec.repo_id, revision=spec.revision, repo_type="dataset"
    )
    present = {f for f in files if Path(f).suffix.lower() in IMAGE_SUFFIXES}
    if METADATA_FILE not in files:
        raise SystemExit(f"{spec.repo_id}@{spec.revision} has no {METADATA_FILE}")

    meta_path = Path(
        hf_hub_download(
            spec.repo_id,
            METADATA_FILE,
            revision=spec.revision,
            repo_type="dataset",
        )
    )
    entries, duplicates, missing = load_entries(meta_path, spec.caption_field, present)

    if args.limit:
        if args.limit > len(entries):
            raise SystemExit(
                f"--limit {args.limit} exceeds the {len(entries)} usable entries in "
                f"{spec.repo_id}@{spec.revision[:12]}"
            )
        entries = entries[: args.limit]

    out: Path = args.out
    # A destination that already holds exactly the requested pairs is left
    # alone, so the matrix can call this per arm without re-downloading.
    existing = sorted(p for p in out.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if (
        not args.force
        and len(existing) == len(entries)
        and (out / "PROVENANCE.md").exists()
    ):
        print(f"up to date: {out} already holds {len(entries)} images", file=sys.stderr)
        emit_rollup(spec, out, entries, duplicates, missing, downloaded=0)
        return 0

    if out.exists() and args.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for name, caption in entries:
        safe = sanitize(name)
        if safe in written:
            raise SystemExit(
                f"file name collision after sanitizing: {name!r} and "
                f"{written[safe]!r} both map to {safe!r}"
            )
        written[safe] = name
        cached = hf_hub_download(
            spec.repo_id, name, revision=spec.revision, repo_type="dataset"
        )
        image_path = out / safe
        shutil.copyfile(cached, image_path)
        image_path.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")

    (out / "PROVENANCE.md").write_text(
        provenance(spec, entries, duplicates, missing), encoding="utf-8"
    )
    emit_rollup(spec, out, entries, duplicates, missing, downloaded=len(entries))
    return 0


def provenance(spec: Dataset, entries, duplicates, missing) -> str:
    """The attribution + pin record that travels with the materialized folder.

    cc-by-sa-3.0 requires attribution, and a measurement folder with no record
    of which revision produced it cannot be re-derived.
    """
    lines = [
        "# Dataset provenance",
        "",
        f"- Source: `{spec.repo_id}`",
        f"- Revision (pinned): `{spec.revision}`",
        f"- License: {spec.license}",
        f"- Images materialized: {len(entries)}",
        "",
        "## Attribution",
        "",
        spec.attribution,
        "",
        "## Notes",
        "",
        spec.note,
        "",
        "## Rows dropped from metadata.jsonl",
        "",
        f"- Duplicate `file_name` rows (first kept): {len(duplicates)}",
    ]
    lines += [f"  - `{d}`" for d in duplicates]
    lines.append(f"- Rows naming a file absent from the repo: {len(missing)}")
    lines += [f"  - `{m}`" for m in missing]
    lines += [
        "",
        "Regenerate with:",
        "",
        "```",
        f"just dataset-fetch <key> <dir> {len(entries)}",
        "```",
        "",
    ]
    return "\n".join(lines)


def emit_rollup(spec, out, entries, duplicates, missing, downloaded: int) -> None:
    """`KEY=VALUE` + `STATUS=` rollup, so a calling script reads a verdict."""
    print(f"REPO={spec.repo_id}")
    print(f"REVISION={spec.revision}")
    print(f"LICENSE={spec.license}")
    print(f"OUT={out}")
    print(f"IMAGES={len(entries)}")
    print(f"DOWNLOADED={downloaded}")
    print(f"DUPLICATE_ROWS={len(duplicates)}")
    print(f"MISSING_ROWS={len(missing)}")
    print("STATUS=OK")


if __name__ == "__main__":
    sys.exit(main())
