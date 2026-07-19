#!/usr/bin/env python3
"""Summarize a burn-autodiff retention ledger (#132, ADR-0005 attribution).

Input: the file named by LORACTL_RETENTION_LEDGER during a training run,
written by the burn-autodiff fork pin (event lines) and loractl's
`probe::phase` (PHASE markers). See the fork's `ledger.rs` module docs for
the line grammar.

Output: a per-op-class attribution of eagerly pinned (`Computed`) checkpoint
bytes, the forward-pinned logical peak, the backward live-curve, fallback
counts, and a `STATUS=`/`KEY=VALUE` rollup — the numbers the #132 decision
rule consumes.

Usage: python3 scripts/ledger-report.py <ledger-file> [--top N]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

GIB = 1024**3


def short_op(type_name: str) -> str:
    """`…::<impl …>::float_matmul::Matmul` -> `Matmul`. The `<impl …>` path
    segment burn's fn-local op structs carry means generics must be stripped
    from the LEAF segment, not the whole path."""
    leaf = type_name.rsplit("::", 1)[-1]
    return leaf.split("<")[0] or leaf or type_name


@dataclass
class Node:
    op: str = "?"
    shape: str = "?"
    pinned_bytes: int = 0  # nonzero iff a Computed action pinned it
    explicit: int = 0
    backup: int = 0
    built: str = ""  # "", "Computed", "Recompute"
    n_required: int = 0
    dropped: bool = False


@dataclass
class Phase:
    name: str
    step: str
    # events observed while this phase was current
    ckpt_bytes_new: int = 0  # newly pinned (deduped) Computed bytes
    saves: int = 0
    save_bytes: int = 0
    consumes: int = 0
    frees: int = 0


def parse(path: str):
    nodes: dict[str, Node] = defaultdict(Node)
    phases: list[Phase] = [Phase("PRE", "0")]
    fallbacks: dict[str, int] = defaultdict(int)
    pinned_nodes: set[str] = set()
    # live tracking during backward: node -> bytes currently materialized
    live_bytes = 0
    live_peak = 0
    live_peak_phase = "?"
    node_live: dict[str, int] = {}

    def node_bytes(nid: str) -> int:
        n = nodes.get(nid)
        if n is None:
            return 0
        if n.pinned_bytes:
            return n.pinned_bytes
        # Recompute nodes: bytes from the OP line's shape (f32 assumed)
        if n.shape and n.shape not in ("?", "[]"):
            dims = [int(x) for x in re.findall(r"\d+", n.shape)]
            elems = 1
            for d in dims:
                elems *= d
            return elems * 4 if dims else 0
        return 0

    with open(path, errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            tag = parts[0]
            ph = phases[-1]
            if tag == "PHASE" and len(parts) >= 3:
                phases.append(Phase(parts[1], parts[2]))
            elif tag == "OP" and len(parts) >= 4:
                n = nodes[parts[1]]
                n.op = short_op(parts[2])
                n.shape = parts[3]
            elif tag == "CKPT" and len(parts) >= 6:
                nid, action, kind, nbytes = parts[1], parts[2], parts[3], int(parts[4])
                n = nodes[nid]
                if action == "Explicit":
                    n.explicit += 1
                else:
                    n.backup += 1
                if kind == "Computed":
                    if n.shape == "?":
                        n.shape = parts[5]
                    if nid not in pinned_nodes:
                        pinned_nodes.add(nid)
                        n.pinned_bytes = nbytes
                        ph.ckpt_bytes_new += nbytes
            elif tag == "FALLBACK" and len(parts) >= 2:
                fallbacks[short_op(parts[1])] += 1
            elif tag == "BUILD" and len(parts) >= 4:
                n = nodes[parts[1]]
                n.built = parts[2]
                n.n_required = int(parts[3])
                if parts[2] == "Computed":
                    node_live[parts[1]] = node_bytes(parts[1])
            elif tag == "DROP" and len(parts) >= 2:
                nodes[parts[1]].dropped = True
            elif tag == "SAVE" and len(parts) >= 3:
                nid = parts[1]
                b = node_bytes(nid)
                if nid not in node_live:
                    node_live[nid] = b
                    live_bytes += b
                ph.saves += 1
                ph.save_bytes += b
            elif tag == "CONSUME" and len(parts) >= 3:
                nid, remaining = parts[1], int(parts[2])
                ph.consumes += 1
                if remaining == 0 and nid in node_live:
                    live_bytes -= node_live.pop(nid)
                    ph.frees += 1
            # track backward-phase live peak (BUILD initializes at backward
            # start; SAVE/CONSUME move it)
            if tag in ("BUILD", "SAVE", "CONSUME"):
                cur = sum(node_live.values()) if tag == "BUILD" else live_bytes
                if tag == "BUILD":
                    live_bytes = cur
                if live_bytes > live_peak:
                    live_peak = live_bytes
                    live_peak_phase = f"{ph.name}/{ph.step}"

    return nodes, phases, fallbacks, live_peak, live_peak_phase


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    nodes, phases, fallbacks, live_peak, live_peak_phase = parse(args.ledger)

    # --- per op x shape class over PINNED (Computed) bytes -----------------
    classes: dict[tuple[str, str], list[int]] = defaultdict(list)
    for n in nodes.values():
        if n.pinned_bytes:
            classes[(n.op, n.shape)].append(n.pinned_bytes)

    total_pinned = sum(sum(v) for v in classes.values())
    dropped_bytes = sum(
        n.pinned_bytes for n in nodes.values() if n.pinned_bytes and n.dropped
    )
    built_computed = sum(
        n.pinned_bytes for n in nodes.values() if n.built == "Computed"
    )
    recompute_nodes = sum(1 for n in nodes.values() if n.built == "Recompute")

    print("=== PINNED (eager Computed clones registered during forward) ===")
    print(f"{'GiB':>8} {'count':>6} {'each-MiB':>9}  op / shape")
    rows = sorted(classes.items(), key=lambda kv: -sum(kv[1]))
    for (op, shape), sizes in rows[: args.top]:
        tot = sum(sizes)
        print(
            f"{tot / GIB:8.3f} {len(sizes):6d} {sizes[0] / 1024**2:9.1f}  {op} {shape}"
        )
    if len(rows) > args.top:
        rest = sum(sum(s) for _, s in rows[args.top :])
        print(f"{rest / GIB:8.3f}    ...  (+{len(rows) - args.top} more classes)")

    print("\n=== FALLBACKS (memory-bound op -> ComputeBound: untracked parent) ===")
    for op, count in sorted(fallbacks.items(), key=lambda kv: -kv[1]):
        print(f"{count:8d}  {op}")

    print("\n=== PHASES ===")
    print(f"{'phase':>16} {'step':>5} {'new-pinned-GiB':>15} {'saves':>6} {'consumes':>9} {'frees':>6}")
    for ph in phases:
        if ph.ckpt_bytes_new or ph.saves or ph.consumes:
            print(
                f"{ph.name:>16} {ph.step:>5} {ph.ckpt_bytes_new / GIB:15.3f}"
                f" {ph.saves:6d} {ph.consumes:9d} {ph.frees:6d}"
            )

    print("\n=== ROLLUP ===")
    print(f"STATUS=ok")
    print(f"TOTAL_PINNED_GIB={total_pinned / GIB:.3f}")
    print(f"BUILT_COMPUTED_GIB={built_computed / GIB:.3f}")
    print(f"DROPPED_AT_BUILD_GIB={dropped_bytes / GIB:.3f}")
    print(f"RECOMPUTE_NODES={recompute_nodes}")
    print(f"BACKWARD_LIVE_PEAK_GIB={live_peak / GIB:.3f} (at {live_peak_phase})")
    print(f"FALLBACK_EVENTS={sum(fallbacks.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
