# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.*", "numpy"]
# ///
"""Deterministic PyTorch reference for the loractl LoRA numerics test (issue #1, M2).

Every weight/input is a FIXED constant (no RNG anywhere), so PyTorch and burn
compute the identical arithmetic sequence and differ only by f32 rounding. This
is the derivation tool for the checked-in golden fixture; `cargo test` reads the
JSON and never runs this script.

Convention: all matrices below are defined in **burn** layout `[d_in, d_out]`
(burn `Linear.weight` is `[d_in, d_out]` and computes `x @ W`). PyTorch
`nn.Linear.weight` is `[out, in]` and computes `x @ Wᵀ`, so this script installs
`tensor(W).T` on the way in and transposes back to burn layout before dumping —
the Rust test does ZERO transposing.

Also runs a pure-numpy analytic-gradient replica (standing in for burn's f32 CPU
arithmetic) and prints the max abs diff vs torch, so the achievable tolerance is
measured, not guessed.
"""

import json
import platform

import numpy as np
import torch

torch.set_default_dtype(torch.float32)

# ---- FIXED constants (burn layout [d_in, d_out]); duplicated verbatim in the
# ---- Rust test with a "MUST match reference/lora_reference.py" comment. -------
D_IN, D_OUT, RANK = 4, 3, 2
ALPHA = 2.0
SCALING = ALPHA / RANK  # = 1.0
LR = 0.1
STEPS = 20

W_BASE = np.array(
    [[0.10, -0.20, 0.30],
     [0.40, 0.50, -0.60],
     [-0.70, 0.80, 0.90],
     [0.15, -0.25, 0.35]], dtype=np.float32)  # [4,3] frozen

A_INIT = np.array(
    [[0.20, 0.10],
     [-0.30, 0.40],
     [0.50, -0.60],
     [0.70, 0.80]], dtype=np.float32)  # [4,2] trainable

# B is zero-initialized [2,3] (LoRA no-op at step 0), trainable.

X = np.array(
    [[1.0, 0.5, -0.5, 0.25],
     [-1.0, 0.3, 0.8, -0.2],
     [0.6, -0.7, 0.9, 0.1],
     [0.2, 0.4, -0.1, 1.0],
     [-0.4, 0.6, 0.3, -0.8]], dtype=np.float32)  # [5,4]

T = np.array(
    [[0.5, -0.5, 1.0],
     [1.0, 0.0, -1.0],
     [-0.3, 0.7, 0.2],
     [0.9, -0.1, 0.4],
     [0.1, 0.6, -0.6]], dtype=np.float32)  # [5,3]


def run_torch():
    """torch autograd + torch.optim.SGD (defaults => p -= lr*grad)."""
    base = torch.nn.Linear(D_IN, D_OUT, bias=False)
    base.weight.data = torch.tensor(W_BASE.T.copy())      # [out,in] = W_BASEᵀ
    base.weight.requires_grad_(False)                      # frozen; excluded from optim

    lora_a = torch.nn.Linear(D_IN, RANK, bias=False)
    lora_a.weight.data = torch.tensor(A_INIT.T.copy())     # [rank,in]
    lora_b = torch.nn.Linear(RANK, D_OUT, bias=False)
    lora_b.weight.data = torch.zeros(D_OUT, RANK)          # [out,rank]

    x = torch.tensor(X.copy())
    t = torch.tensor(T.copy())
    opt = torch.optim.SGD([lora_a.weight, lora_b.weight], lr=LR,
                          momentum=0, dampening=0, weight_decay=0, nesterov=False)

    losses = []
    for _ in range(STEPS):
        pred = base(x) + SCALING * lora_b(lora_a(x))
        loss = ((pred - t) ** 2).mean()                    # manual MSE
        losses.append(float(loss.item()))                  # record BEFORE backward/step
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Transpose torch weights BACK to burn layout before dumping.
    a_final = lora_a.weight.detach().cpu().numpy().T.copy()  # [4,2]
    b_final = lora_b.weight.detach().cpu().numpy().T.copy()  # [2,3]
    base_final = base.weight.detach().cpu().numpy().T.copy()  # [4,3]
    return losses, a_final, b_final, base_final


def run_numpy():
    """Pure-numpy analytic-gradient replica in burn layout (x @ W). Stands in
    for burn's f32 CPU arithmetic to measure the achievable tolerance."""
    A = A_INIT.copy()
    B = np.zeros((RANK, D_OUT), dtype=np.float32)
    n = X.shape[0] * D_OUT  # elements in the mean

    losses = []
    for _ in range(STEPS):
        H = X @ A                      # [5,2]
        pred = X @ W_BASE + SCALING * (H @ B)   # [5,3]
        R = pred - T
        losses.append(float((R ** 2).mean()))
        dP = (2.0 / n) * R             # [5,3]
        dB = SCALING * (H.T @ dP)      # [2,3]
        dH = SCALING * (dP @ B.T)      # [5,2]
        dA = X.T @ dH                  # [4,2]
        A = (A - LR * dA).astype(np.float32)
        B = (B - LR * dB).astype(np.float32)
    return losses, A.astype(np.float32), B.astype(np.float32), W_BASE.copy()


t_losses, t_a, t_b, t_base = run_torch()
n_losses, n_a, n_b, n_base = run_numpy()

max_loss_diff = max(abs(a - b) for a, b in zip(t_losses, n_losses))
max_a_diff = float(np.max(np.abs(t_a - n_a)))
max_b_diff = float(np.max(np.abs(t_b - n_b)))
base_bitexact = bool(np.array_equal(t_base, W_BASE))

import sys
print("=== feasibility: torch autograd vs numpy analytic (burn stand-in) ===", file=sys.stderr)
print(f"torch: {sys.modules['torch'].__version__}  platform: {platform.platform()}", file=sys.stderr)
print(f"max |loss diff| over {STEPS} steps: {max_loss_diff:.3e}", file=sys.stderr)
print(f"max |lora_a diff|: {max_a_diff:.3e}   max |lora_b diff|: {max_b_diff:.3e}", file=sys.stderr)
print(f"base bit-exact (frozen): {base_bitexact}", file=sys.stderr)
print(f"first loss {t_losses[0]:.6f} -> last loss {t_losses[-1]:.6f} (converges: {t_losses[-1] < t_losses[0]})", file=sys.stderr)

golden = {
    "provenance": {
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "note": "burn-layout [d_in,d_out], row-major; regenerate with `uv run reference/lora_reference.py`",
    },
    "hyperparams": {"d_in": D_IN, "d_out": D_OUT, "rank": RANK, "alpha": ALPHA,
                    "scaling": SCALING, "lr": LR, "steps": STEPS},
    "base_final": t_base.flatten().tolist(),
    "lora_a_final": t_a.flatten().tolist(),
    "lora_b_final": t_b.flatten().tolist(),
    "losses": t_losses,
}
print(json.dumps(golden, indent=2))
