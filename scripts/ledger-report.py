#!/usr/bin/env python3
"""Summarize a burn-autodiff retention ledger (#132, ADR-0005 attribution).

Input: the file named by LORACTL_RETENTION_LEDGER during a training run,
written by the burn-autodiff fork pin (event lines) and loractl's
`probe::phase` (PHASE markers). See the fork's `ledger.rs` module docs for
the line grammar.

Output: a **per-step** attribution of eagerly pinned (`Computed`) checkpoint
bytes by op-class, fallback counts, build-time drops, the backward
live-curve, and a `STATUS=`/`KEY=VALUE` rollup — the numbers the #132
decision rule consumes.

Aggregation is segmented by step (the PHASE markers' step field): burn node
ids are globally unique across steps, so an unsegmented sum over an N-step
ledger would be ~N x the per-step working set (review finding, 2026-07-19).
The rollup reports the LAST complete step as the steady state. A node that
was BUILT and also has leftover duplicate actions gets both BUILD and DROP
events; `built` is authoritative — DROP bytes count only never-built nodes.

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
    step: str = "0"  # the step whose forward first pinned/registered it
    pinned_bytes: int = 0  # nonzero iff a Computed action pinned it
    explicit: int = 0
    backup: int = 0
    built: str = ""  # "", "Computed", "Recompute"
    n_required: int = 0
    dropped: bool = False


@dataclass
class StepAgg:
    pinned_bytes: int = 0
    built_computed: int = 0
    dropped_never_built: int = 0
    recompute_nodes: int = 0
    fallbacks: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    classes: dict[tuple[str, str], list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )
    saves: int = 0
    consumes: int = 0


def parse(path: str):
    nodes: dict[str, Node] = {}
    steps: dict[str, StepAgg] = defaultdict(StepAgg)
    current_step = "0"
    phase_rows: list[tuple[str, str, int, int, int, int]] = []
    # current-phase accumulators
    cur_phase = ("PRE", "0")
    cur = [0, 0, 0, 0]  # new-pinned bytes, saves, consumes, frees

    live_bytes = 0
    live_peak = 0
    live_peak_phase = "?"
    node_live: dict[str, int] = {}

    def node(nid: str) -> Node:
        n = nodes.get(nid)
        if n is None:
            n = nodes[nid] = Node(step=current_step)
        return n

    def node_bytes(nid: str) -> int:
        n = nodes.get(nid)
        if n is None:
            return 0
        if n.pinned_bytes:
            return n.pinned_bytes
        # Recompute nodes: bytes from the OP line's shape (f32 assumed; the
        # f32-only cuda/ndarray arms this probe targets make that exact)
        if n.shape and n.shape not in ("?", "[]"):
            dims = [int(x) for x in re.findall(r"\d+", n.shape)]
            elems = 1
            for d in dims:
                elems *= d
            return elems * 4 if dims else 0
        return 0

    def flush_phase():
        if any(cur):
            phase_rows.append((*cur_phase, *cur))
        cur[0] = cur[1] = cur[2] = cur[3] = 0

    with open(path, errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            tag = parts[0]
            if tag == "PHASE" and len(parts) >= 3:
                flush_phase()
                cur_phase = (parts[1], parts[2])
                current_step = parts[2]
            elif tag == "OP" and len(parts) >= 5:
                n = node(parts[1])
                n.op = short_op(parts[2])
                n.shape = parts[3]
            elif tag == "CKPT" and len(parts) >= 6:
                nid, action, kind, nbytes = parts[1], parts[2], parts[3], int(parts[4])
                n = node(nid)
                if action == "Explicit":
                    n.explicit += 1
                else:
                    n.backup += 1
                if kind == "Computed" and not n.pinned_bytes:
                    if n.shape == "?":
                        n.shape = parts[5]
                    n.pinned_bytes = nbytes
                    agg = steps[n.step]
                    agg.pinned_bytes += nbytes
                    agg.classes[(n.op, n.shape)].append(nbytes)
                    cur[0] += nbytes
            elif tag == "FALLBACK" and len(parts) >= 2:
                steps[current_step].fallbacks[short_op(parts[1])] += 1
            elif tag == "BUILD" and len(parts) >= 4:
                n = node(parts[1])
                n.built = parts[2]
                n.n_required = int(parts[3])
                if parts[2] == "Computed":
                    node_live[parts[1]] = node_bytes(parts[1])
                    steps[n.step].built_computed += n.pinned_bytes
                else:
                    steps[n.step].recompute_nodes += 1
            elif tag == "DROP" and len(parts) >= 2:
                node(parts[1]).dropped = True
            elif tag == "SAVE" and len(parts) >= 3:
                nid = parts[1]
                b = node_bytes(nid)
                if nid not in node_live:
                    node_live[nid] = b
                    live_bytes += b
                steps[current_step].saves += 1
                cur[1] += 1
            elif tag == "CONSUME" and len(parts) >= 3:
                nid, remaining = parts[1], int(parts[2])
                steps[current_step].consumes += 1
                cur[2] += 1
                if remaining == 0 and nid in node_live:
                    live_bytes -= node_live.pop(nid)
                    cur[3] += 1
            if tag == "BUILD":
                live_bytes = sum(node_live.values())
            if tag in ("BUILD", "SAVE", "CONSUME") and live_bytes > live_peak:
                live_peak = live_bytes
                live_peak_phase = f"{cur_phase[0]}/{cur_phase[1]}"
    flush_phase()

    # dropped-never-built, attributed to the node's own step. `built` is
    # authoritative: a duplicated-but-required node emits BUILD + DROP, and
    # its bytes were retained into backward, not released at build.
    for n in nodes.values():
        if n.pinned_bytes and n.dropped and not n.built:
            steps[n.step].dropped_never_built += n.pinned_bytes

    return steps, phase_rows, live_peak, live_peak_phase


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    steps, phase_rows, live_peak, live_peak_phase = parse(args.ledger)

    # steady state = the last step with pinned activity
    active = [s for s, a in steps.items() if a.pinned_bytes]
    last = max(active, key=lambda s: int(s) if s.isdigit() else -1) if active else "0"
    agg = steps[last]

    print(f"=== PINNED per op x shape, step {last} (eager Computed clones) ===")
    print(f"{'GiB':>8} {'count':>6} {'each-MiB':>9}  op / shape")
    rows = sorted(agg.classes.items(), key=lambda kv: -sum(kv[1]))
    for (op, shape), sizes in rows[: args.top]:
        tot = sum(sizes)
        print(
            f"{tot / GIB:8.3f} {len(sizes):6d} {sizes[0] / 1024**2:9.1f}  {op} {shape}"
        )
    if len(rows) > args.top:
        rest = sum(sum(s) for _, s in rows[args.top :])
        print(f"{rest / GIB:8.3f}    ...  (+{len(rows) - args.top} more classes)")

    print(f"\n=== FALLBACKS step {last} (memory-bound op -> ComputeBound) ===")
    for op, count in sorted(agg.fallbacks.items(), key=lambda kv: -kv[1]):
        print(f"{count:8d}  {op}")

    print("\n=== PHASES ===")
    print(
        f"{'phase':>16} {'step':>5} {'new-pinned-GiB':>15} {'saves':>6} {'consumes':>9} {'frees':>6}"
    )
    for name, step, pinned, saves, consumes, frees in phase_rows:
        print(
            f"{name:>16} {step:>5} {pinned / GIB:15.3f} {saves:6d} {consumes:9d} {frees:6d}"
        )

    print("\n=== PER-STEP ROLLUP ===")
    print(
        f"{'step':>5} {'pinned-GiB':>11} {'built-GiB':>10} {'dropped-GiB':>12} "
        f"{'recompute':>10} {'fallbacks':>10}"
    )
    for s in sorted(steps, key=lambda x: int(x) if x.isdigit() else -1):
        a = steps[s]
        if not (a.pinned_bytes or a.fallbacks):
            continue
        print(
            f"{s:>5} {a.pinned_bytes / GIB:11.3f} {a.built_computed / GIB:10.3f} "
            f"{a.dropped_never_built / GIB:12.3f} {a.recompute_nodes:10d} "
            f"{sum(a.fallbacks.values()):10d}"
        )

    print("\n=== ROLLUP (steady state = step "f"{last}) ===")
    print("STATUS=ok")
    print(f"STEP={last}")
    print(f"STEP_PINNED_GIB={agg.pinned_bytes / GIB:.3f}")
    print(f"STEP_BUILT_COMPUTED_GIB={agg.built_computed / GIB:.3f}")
    print(f"STEP_DROPPED_NEVER_BUILT_GIB={agg.dropped_never_built / GIB:.3f}")
    print(f"STEP_RECOMPUTE_NODES={agg.recompute_nodes}")
    print(f"STEP_FALLBACK_EVENTS={sum(agg.fallbacks.values())}")
    print(f"BACKWARD_LIVE_PEAK_GIB={live_peak / GIB:.3f} (at {live_peak_phase})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
