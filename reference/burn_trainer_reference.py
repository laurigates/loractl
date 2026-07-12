# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.*", "numpy"]
# ///
"""PyTorch reference for the `BurnTrainer` step-loss golden (#49 H9).

The convergence tests only assert a *trend* (`last < 0.7 * first`), which a
trainer that "trains fast but wrong" passes. This script pins the trainer's
exact per-step losses to an INDEPENDENT PyTorch computation, the way
`lora_reference.py` pins the toy forward pass.

## Why this one needs a dump

Every other reference here derives its inputs from fixed constants, so torch and
burn start from identical numbers by construction. `BurnTrainer`'s synthetic run
cannot: its frozen base, its LoRA `A` init, and its Gaussian-blob dataset all
come from burn's seeded `StdRng` (ChaCha), which PyTorch cannot reproduce. So we
feed torch burn's *actual* initial tensors and data — dumped by
`cargo run -p loractl-core --example dump_synthetic_run` — and have torch
independently recompute the **losses** from them.

What that does and does not prove:

  * INDEPENDENT: the forward (frozen `fc1` -> ReLU -> base + (alpha/rank)·B(A(h))),
    the cross-entropy, the AdamW update, the freeze (only A/B are optimized), and
    the record-loss-before-step ordering are all recomputed by torch. A bug in any
    of them shows up as a loss mismatch. This is a real numerics proof.
  * SHARED: the initial weights and the training data come from burn. A bug in
    burn's *data generation* (e.g. centroids drawn with the wrong scale) is not
    caught here — it is a shared input to both sides. That half is covered by the
    black-box convergence test, which would stop converging.

## AdamW: burn == torch, on purpose

burn's `AdamW` (burn-optim 0.21 `adamw.rs`) computes
`p := p*(1 - lr*wd) - lr * m̂/(sqrt(v̂) + eps)` with defaults
`beta_1=0.9, beta_2=0.999, epsilon=1e-5` — i.e. decoupled decay, identical in
form to `torch.optim.AdamW`. torch's eps default is 1e-8, so it is set to 1e-5
explicitly below. The golden pins two trajectories, `weight_decay = 0.0` and
`0.05`, so the *decoupled* decay semantics are pinned against torch and not just
asserted (see `.claude/rules/burn-optimizer-and-dropout.md`).

Convention: all matrices in the dump are burn layout `[d_in, d_out]`. This script
owns every transpose (torch `nn.Linear.weight` is `[out, in]`) and transposes back
before dumping, so the Rust test does ZERO transposing.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

# burn-optim 0.21 AdamWConfig defaults (adamw.rs) — NOT torch's defaults.
BETAS = (0.9, 0.999)
EPS = 1e-5


def load(dump: Path, name: str, shapes: dict, dtype: str) -> np.ndarray:
    suffix = "f32" if dtype == "f32" else "i64"
    raw = np.fromfile(dump / f"{name}.{suffix}.bin", dtype="<f4" if dtype == "f32" else "<i8")
    return raw.reshape(shapes[name])


def build(dump: Path, manifest: dict):
    shapes = manifest["shapes"]
    hp = manifest["hyperparams"]
    rank = int(hp["rank"])

    fc1_w = load(dump, "fc1_weight", shapes, "f32")  # [784, 256] burn layout
    fc1_b = load(dump, "fc1_bias", shapes, "f32")  # [256]
    base_w = load(dump, "fc2_base_weight", shapes, "f32")  # [256, 10]
    base_b = load(dump, "fc2_base_bias", shapes, "f32")  # [10]
    a_w = load(dump, "lora_a_weight", shapes, "f32")  # [256, rank]
    b_w = load(dump, "lora_b_weight", shapes, "f32")  # [rank, 10]

    d_in, hidden = fc1_w.shape
    out = base_w.shape[1]

    fc1 = torch.nn.Linear(d_in, hidden, bias=True)
    fc1.weight.data = torch.tensor(fc1_w.T.copy())
    fc1.bias.data = torch.tensor(fc1_b.copy())
    fc1.weight.requires_grad_(False)  # FROZEN — excluded from the optimizer
    fc1.bias.requires_grad_(False)

    base = torch.nn.Linear(hidden, out, bias=True)
    base.weight.data = torch.tensor(base_w.T.copy())
    base.bias.data = torch.tensor(base_b.copy())
    base.weight.requires_grad_(False)  # FROZEN
    base.bias.requires_grad_(False)

    lora_a = torch.nn.Linear(hidden, rank, bias=False)
    lora_a.weight.data = torch.tensor(a_w.T.copy())
    lora_b = torch.nn.Linear(rank, out, bias=False)
    lora_b.weight.data = torch.tensor(b_w.T.copy())

    batches = []
    for i in range(int(manifest["batches_dumped"])):
        x = torch.tensor(load(dump, f"batch_{i}_x", shapes, "f32").copy())
        y = torch.tensor(load(dump, f"batch_{i}_y", shapes, "i64").copy())
        batches.append((x, y))

    return fc1, base, lora_a, lora_b, batches


def run(dump: Path, manifest: dict, weight_decay: float):
    """Replay `run_classification`'s loop in torch and return its losses."""
    hp = manifest["hyperparams"]
    steps, lr, scaling = int(hp["steps"]), float(hp["lr"]), float(hp["scaling"])
    fc1, base, lora_a, lora_b, batches = build(dump, manifest)

    opt = torch.optim.AdamW(
        [lora_a.weight, lora_b.weight],
        lr=lr,
        betas=BETAS,
        eps=EPS,
        weight_decay=weight_decay,
        amsgrad=False,
    )

    losses = []
    for step in range(steps):
        x, y = batches[step % len(batches)]
        h = torch.relu(fc1(x))
        # LoRA forward: base(h) + (alpha/rank) · B(A(h)). Dropout is 0.0, and
        # burn's Dropout is identity at prob 0, so there is nothing to model.
        logits = base(h) + scaling * lora_b(lora_a(h))
        # burn's CrossEntropyLoss (no weights, no smoothing) is
        # `-mean(log_softmax(logits)[target])` == torch's mean-reduction CE.
        loss = F.cross_entropy(logits, y)
        losses.append(float(loss.item()))  # record BEFORE backward/step
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Frozen base must be untouched — the strongest form of the LoRA claim.
    assert fc1.weight.grad is None and base.weight.grad is None

    # Back to burn layout `[d_in, d_out]` before dumping.
    b_final = lora_b.weight.detach().cpu().numpy().T.copy()
    return losses, b_final


parser = argparse.ArgumentParser()
parser.add_argument("--dump", required=True, type=Path, help="dir written by dump_synthetic_run")
args = parser.parse_args()

manifest = json.loads((args.dump / "manifest.json").read_text())
hp = manifest["hyperparams"]

trajectories = {}
for wd in hp["weight_decays"]:
    losses, b_final = run(args.dump, manifest, float(wd))
    trajectories[f"wd_{wd}"] = {
        "weight_decay": float(wd),
        "losses": losses,
        "lora_b_final": b_final.flatten().tolist(),
    }

print("=== BurnTrainer step-loss reference (torch replay of burn's init + data) ===", file=sys.stderr)
print(f"torch: {torch.__version__}  platform: {platform.platform()}", file=sys.stderr)
for key, t in trajectories.items():
    ls = t["losses"]
    print(f"{key}: {ls[0]:.6f} -> {ls[-1]:.6f} over {len(ls)} steps", file=sys.stderr)
a, b = (trajectories[f"wd_{wd}"]["losses"] for wd in hp["weight_decays"])
print(f"max |loss diff| between the two decay settings: {max(abs(x - y) for x, y in zip(a, b)):.3e}", file=sys.stderr)

golden = {
    "provenance": {
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "note": "burn-layout [d_in,d_out]; regenerate with `just burn-trainer-reference`",
    },
    "hyperparams": {
        "seed": int(hp["seed"]),
        "steps": int(hp["steps"]),
        "rank": int(hp["rank"]),
        "alpha": float(hp["alpha"]),
        "lr": float(hp["lr"]),
        "scaling": float(hp["scaling"]),
    },
    "trajectories": [trajectories[f"wd_{wd}"] for wd in hp["weight_decays"]],
}
print(json.dumps(golden, indent=2))
