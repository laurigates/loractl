#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Golden LR multipliers from diffusers' own schedule definitions (#247).

`optim.schedule` (#212) was first verified only against hand-computed values,
which pin loractl's *own* reading of each curve and cannot disagree with it.
The consumer-style reference is diffusers' `get_scheduler`, which kohya-ss and
most LoRA trainers call, so a loractl run and a kohya run with the same
`linear`/`cosine` settings should follow the same curve.

Rather than transcribe the formulas, we download `src/diffusers/optimization.py`
at a PINNED TAG, extract the three schedule factories with `ast`, and execute
them with `LambdaLR` stubbed to capture the `lr_lambda` closure. The closures
are pure `math`/float code, so no torch is needed. Before trusting them we
assert the lines whose semantics the Rust test relies on are still present, so
an upstream change fails here instead of producing a golden that silently no
longer reflects diffusers.

Pinned: `constant_with_warmup`, `linear`, `cosine` (num_cycles = 0.5).
Not pinned: `exponential`, `step` — diffusers has no equivalent (they mirror
torch's `ExponentialLR` / `StepLR`) and stay covered by the hand-computed
cases in `tests/lr_schedule.rs`.

Output: `lambda[k]` for k = 0..=total, diffusers' 0-based scheduler index.
The Rust side maps its 1-based step onto that index; see
`tests/lr_schedule_reference.rs` for the mapping and the one deliberate
divergence (warmup).

Usage: just lr-schedule-reference
"""

import ast
import json
import urllib.request

# Bump deliberately, and re-read the three factories when you do.
DIFFUSERS_TAG = "v0.35.1"
URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/"
    f"{DIFFUSERS_TAG}/src/diffusers/optimization.py"
)

FACTORIES = {
    "constant": "get_constant_schedule_with_warmup",
    "linear": "get_linear_schedule_with_warmup",
    "cosine": "get_cosine_schedule_with_warmup",
}

# The semantics the Rust mapping depends on: warmup is `k / W` on a 0-based
# index, linear decays to 0 over `total - W`, cosine is the half-cosine.
REQUIRED_LINES = [
    "if current_step < num_warmup_steps:",
    "return float(current_step) / float(max(1, num_warmup_steps))",
    "float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))",
    "progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))",
    "return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))",
]

# (warmup, total) shapes: no warmup, short warmup, warmup = total - 1 (the
# largest `validate` allows), and an odd total so cosine's midpoint is off-grid.
CASES = [(0, 10), (3, 10), (9, 10), (5, 37), (0, 1)]


def fetch() -> str:
    with urllib.request.urlopen(URL) as r:
        return r.read().decode("utf-8")


def extract(source: str, name: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise SystemExit(f"FAIL: {name} not found in diffusers {DIFFUSERS_TAG}")


def load_factories(source: str) -> dict:
    for line in REQUIRED_LINES:
        if line not in source:
            raise SystemExit(
                f"FAIL: diffusers {DIFFUSERS_TAG} no longer contains {line!r}; "
                "the schedule contract moved — re-read optimization.py"
            )
    import math

    ns: dict = {
        "math": math,
        "Optimizer": object,
        "Optional": object,
        "Union": object,
        # Capture the closure instead of building a torch scheduler.
        "LambdaLR": lambda _opt, fn, last_epoch=-1: fn,
    }
    for name in FACTORIES.values():
        exec(extract(source, name), ns)  # noqa: S102
    return ns


def main() -> None:
    ns = load_factories(fetch())
    cases = []
    for kind, name in FACTORIES.items():
        for warmup, total in CASES:
            if kind == "constant":
                fn = ns[name](None, num_warmup_steps=warmup)
            else:
                fn = ns[name](None, num_warmup_steps=warmup, num_training_steps=total)
            cases.append(
                {
                    "kind": kind,
                    "warmup_steps": warmup,
                    "total": total,
                    "lambda": [float(fn(k)) for k in range(total + 1)],
                }
            )
    print(
        json.dumps(
            {"source": f"diffusers {DIFFUSERS_TAG} src/diffusers/optimization.py", "cases": cases},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
