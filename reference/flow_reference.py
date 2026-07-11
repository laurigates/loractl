# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.*", "numpy"]
# ///
"""Deterministic PyTorch reference for the loractl flow-matching numerics test (issue #19, M8).

Every weight/input is a FIXED constant (no RNG anywhere), so PyTorch and burn
compute the identical arithmetic sequence and differ only by f32 rounding. This
is the derivation tool for the checked-in golden fixture; `cargo test` reads the
JSON and never runs this script.

Pinned math conventions (SD3 paper + diffusers + kohya-ss — all agree):

- t = 0 is DATA, t = 1 is NOISE (the opposite of the original Lipman/Liu
  flow-matching papers — do not mix).
- Interpolation (SD3 Eq. 13): `x_t = (1 - t)*x_0 + t*eps`, `eps ~ N(0, I)`.
- Velocity / v-prediction target: `v = dx_t/dt = eps - x_0` (noise minus data).
- Loss: plain `mean((v_pred - v)**2)` — weighting is identically 1.0 for the
  logit-normal scheme (the t/(1-t) emphasis is delivered by the *sampling
  density*, never a multiplicative weight).
- Timestep sampling: `u ~ Normal(mean, std)`, `t = sigmoid(u)`, then the
  constant shift `t' = shift*t / (1 + (shift - 1)*t)`.
- FLUX resolution-dependent shift: mu is linear in image_seq_len through
  (256, 0.5) and (4096, 1.15); the LINEAR shift consumable by the constant
  form is `exp(mu)` (mu itself is a LOG shift).

Convention: all matrices below are defined in **burn** layout `[d_in, d_out]`
(burn `Linear.weight` is `[d_in, d_out]` and computes `x @ W`). PyTorch
`nn.Linear.weight` is `[out, in]` and computes `x @ Wᵀ`, so this script installs
`tensor(W).T` on the way in and transposes back to burn layout before dumping —
the Rust test does ZERO transposing.

Also runs a pure-numpy analytic-gradient replica (standing in for burn's f32 CPU
arithmetic) and prints the max abs diff vs torch, so the achievable tolerance is
measured, not guessed. A generation-time self-check asserts every frozen fc1
preactivation is safely away from zero (min |preactivation| > 1e-3): a ~0
preactivation is a knife-edge ReLU gate whose sign torch and burn could round
differently, and that divergence would compound over the 20 training steps.
"""

import json
import platform
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float32)

# ---- FIXED constants (burn layout [d_in, d_out]); duplicated verbatim in the
# ---- Rust test with a "MUST match reference/flow_reference.py" comment. -------
LATENT_DIM = 3  # toy latent width (the trainer's synthetic toy uses 16)
HIDDEN = 4
RANK = 2
ALPHA = 4.0
SCALING = ALPHA / RANK  # = 2.0
LR = 0.1
STEPS = 20
BATCH = 4
D_IN = LATENT_DIM + 1  # velocity-net input = concat[x_t, t]

# Sampler pins: raw standard-normal draws pushed through the FULL
# `logit_to_t` composition (sigmoid(u*std + mean) then the constant shift).
U = np.array([-2.5, -1.0, -0.3, 0.0, 0.4, 1.0, 2.0, 3.0], dtype=np.float64)
# (a) symmetric config: a sigmoid(-u) error cancels at mean 0 / std 1 — which is
# why (b) exists: mean 0.5 / std 2.0 / shift 1.5 breaks every sign symmetry.
SYM = {"logit_mean": 0.0, "logit_std": 1.0, "shift": 3.0}
ASYM = {"logit_mean": 0.5, "logit_std": 2.0, "shift": 1.5}

# FLUX resolution-dependent shift anchors: (256, 0.5) and (4096, 1.15) pin the
# mu line; 1024 is an interior point. The golden stores exp(mu) — the LINEAR
# shift — because that is what `shift_timesteps` consumes.
SEQ_LENS = [256, 1024, 4096]

X0 = np.array(
    [[1.5, -0.6, 0.8],
     [-1.2, 0.9, -0.4],
     [0.3, -1.0, 1.1],
     [-0.7, 0.5, -1.3]], dtype=np.float32)  # [4,3] data points

EPS = np.array(
    [[0.4, 1.1, -0.9],
     [-0.5, -1.4, 0.7],
     [1.2, 0.2, -0.6],
     [-0.3, 0.8, 1.0]], dtype=np.float32)  # [4,3] "noise" draws (fixed)

T_INTERP = np.array([0.15, 0.4, 0.65, 0.9], dtype=np.float32)  # [4] timesteps

W1 = np.array(
    [[0.30, -0.20, 0.45, 0.10],
     [-0.25, 0.35, -0.15, 0.40],
     [0.20, 0.15, -0.30, -0.35],
     [0.50, -0.40, 0.25, 0.30]], dtype=np.float32)  # [4,4] frozen fc1 weight

B1 = np.array([0.10, -0.15, 0.20, 0.05], dtype=np.float32)  # [4] frozen fc1 bias

W_BASE = np.array(
    [[0.10, -0.20, 0.30],
     [0.40, 0.50, -0.60],
     [-0.70, 0.80, 0.90],
     [0.15, -0.25, 0.35]], dtype=np.float32)  # [4,3] frozen readout base

A_INIT = np.array(
    [[0.20, 0.10],
     [-0.30, 0.40],
     [0.50, -0.60],
     [0.70, 0.80]], dtype=np.float32)  # [4,2] trainable

# B is zero-initialized [2,3] (LoRA no-op at step 0), trainable.


def logit_to_t(u: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """The full production transform: sigmoid(u*std + mean), then the constant
    shift `t' = shift*t / (1 + (shift - 1)*t)`. Returns (t_sigmoid, t_shifted)."""
    t = 1.0 / (1.0 + np.exp(-(u * cfg["logit_std"] + cfg["logit_mean"])))
    shift = cfg["shift"]
    return t, shift * t / (1.0 + (shift - 1.0) * t)


def resolution_shift(image_seq_len: int) -> float:
    """FLUX resolution-dependent shift: mu linear through (256, 0.5) and
    (4096, 1.15); returns the LINEAR shift exp(mu), directly consumable by the
    constant-form shift transform."""
    mu = 0.5 + (1.15 - 0.5) * (image_seq_len - 256) / (4096 - 256)
    return float(np.exp(mu))


# Interpolation + velocity target on the fixed batch (f32, matching burn).
X_T = (1.0 - T_INTERP[:, None]) * X0 + T_INTERP[:, None] * EPS  # [4,3]
V_TARGET = EPS - X0  # [4,3] — noise minus data; the sign convention under test

# Velocity-net training input: concat[x_t, t] — the t column is the SAME t used
# by the interpolation.
X_TRAIN = np.concatenate([X_T, T_INTERP[:, None]], axis=1).astype(np.float32)  # [4,4]

# Generation-time self-check: no knife-edge ReLU gates (see module docstring).
preact = X_TRAIN @ W1 + B1
min_preact = float(np.min(np.abs(preact)))
assert min_preact > 1e-3, (
    f"min |fc1 preactivation| = {min_preact:.3e} <= 1e-3 — pick different fixed "
    "constants; a ~0 preactivation is a knife-edge ReLU gate torch and burn "
    "could disagree on"
)


def run_torch():
    """torch autograd + torch.optim.SGD (defaults => p -= lr*grad)."""
    fc1 = torch.nn.Linear(D_IN, HIDDEN, bias=True)
    fc1.weight.data = torch.tensor(W1.T.copy())            # [out,in] = W1ᵀ
    fc1.bias.data = torch.tensor(B1.copy())
    fc1.weight.requires_grad_(False)                        # frozen
    fc1.bias.requires_grad_(False)

    base = torch.nn.Linear(HIDDEN, LATENT_DIM, bias=False)
    base.weight.data = torch.tensor(W_BASE.T.copy())        # [out,in] = W_BASEᵀ
    base.weight.requires_grad_(False)                       # frozen

    lora_a = torch.nn.Linear(HIDDEN, RANK, bias=False)
    lora_a.weight.data = torch.tensor(A_INIT.T.copy())      # [rank,in]
    lora_b = torch.nn.Linear(RANK, LATENT_DIM, bias=False)
    lora_b.weight.data = torch.zeros(LATENT_DIM, RANK)      # [out,rank]

    x = torch.tensor(X_TRAIN.copy())
    v = torch.tensor(V_TARGET.copy())
    opt = torch.optim.SGD([lora_a.weight, lora_b.weight], lr=LR,
                          momentum=0, dampening=0, weight_decay=0, nesterov=False)

    losses = []
    for _ in range(STEPS):
        h = torch.relu(fc1(x))
        pred = base(h) + SCALING * lora_b(lora_a(h))
        loss = ((pred - v) ** 2).mean()                     # plain MSE, weight = 1.0
        losses.append(float(loss.item()))                   # record BEFORE backward/step
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Transpose torch weights BACK to burn layout before dumping.
    a_final = lora_a.weight.detach().cpu().numpy().T.copy()      # [4,2]
    b_final = lora_b.weight.detach().cpu().numpy().T.copy()      # [2,3]
    fc1_w_final = fc1.weight.detach().cpu().numpy().T.copy()     # [4,4]
    fc1_b_final = fc1.bias.detach().cpu().numpy().copy()         # [4]
    base_final = base.weight.detach().cpu().numpy().T.copy()     # [4,3]
    return losses, a_final, b_final, fc1_w_final, fc1_b_final, base_final


def run_numpy():
    """Pure-numpy analytic-gradient replica in burn layout (x @ W). Stands in
    for burn's f32 CPU arithmetic to measure the achievable tolerance."""
    A = A_INIT.copy()
    Bm = np.zeros((RANK, LATENT_DIM), dtype=np.float32)
    n = BATCH * LATENT_DIM  # elements in the mean

    losses = []
    for _ in range(STEPS):
        H = np.maximum(X_TRAIN @ W1 + B1, 0.0)          # [4,4] relu(fc1)
        HA = H @ A                                       # [4,2]
        pred = H @ W_BASE + SCALING * (HA @ Bm)          # [4,3]
        R = pred - V_TARGET
        losses.append(float((R ** 2).mean()))
        dP = (2.0 / n) * R                               # [4,3]
        dB = SCALING * (HA.T @ dP)                       # [2,3]
        dA = SCALING * (H.T @ (dP @ Bm.T))               # [4,2]
        A = (A - LR * dA).astype(np.float32)
        Bm = (Bm - LR * dB).astype(np.float32)
    return losses, A.astype(np.float32), Bm.astype(np.float32)


t_losses, t_a, t_b, t_fc1_w, t_fc1_b, t_base = run_torch()
n_losses, n_a, n_b = run_numpy()

max_loss_diff = max(abs(a - b) for a, b in zip(t_losses, n_losses))
max_a_diff = float(np.max(np.abs(t_a - n_a)))
max_b_diff = float(np.max(np.abs(t_b - n_b)))
frozen_bitexact = (
    bool(np.array_equal(t_fc1_w, W1))
    and bool(np.array_equal(t_fc1_b, B1))
    and bool(np.array_equal(t_base, W_BASE))
)

print("=== feasibility: torch autograd vs numpy analytic (burn stand-in) ===", file=sys.stderr)
print(f"torch: {torch.__version__}  platform: {platform.platform()}", file=sys.stderr)
print(f"min |fc1 preactivation| over the fixed batch: {min_preact:.3e} (> 1e-3 OK)", file=sys.stderr)
print(f"max |loss diff| over {STEPS} steps: {max_loss_diff:.3e}", file=sys.stderr)
print(f"max |lora_a diff|: {max_a_diff:.3e}   max |lora_b diff|: {max_b_diff:.3e}", file=sys.stderr)
print(f"frozen fc1/base bit-exact: {frozen_bitexact}", file=sys.stderr)
print(f"first loss {t_losses[0]:.6f} -> last loss {t_losses[-1]:.6f} (converges: {t_losses[-1] < t_losses[0]})", file=sys.stderr)

sym_sigmoid, sym_shifted = logit_to_t(U, SYM)
_, asym_shifted = logit_to_t(U, ASYM)

golden = {
    "provenance": {
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "note": "burn-layout [d_in,d_out], row-major; regenerate with `uv run reference/flow_reference.py` (`just flow-reference`)",
    },
    "hyperparams": {
        "latent_dim": LATENT_DIM, "hidden": HIDDEN, "rank": RANK, "alpha": ALPHA,
        "scaling": SCALING, "lr": LR, "steps": STEPS, "batch": BATCH,
        "sampler_symmetric": SYM, "sampler_asymmetric": ASYM,
        "resolution_seq_lens": SEQ_LENS,
    },
    "sampler": {
        "u": U.tolist(),
        "symmetric": {
            "t_sigmoid": sym_sigmoid.tolist(),
            "t_shifted": sym_shifted.tolist(),
        },
        "asymmetric": {
            "t_shifted": asym_shifted.tolist(),
        },
    },
    "resolution_shift": [
        {"seq_len": s, "shift": resolution_shift(s)} for s in SEQ_LENS
    ],
    "interp": {
        "x0": X0.flatten().tolist(),
        "eps": EPS.flatten().tolist(),
        "t": T_INTERP.tolist(),
        "x_t": X_T.flatten().tolist(),
        "v_target": V_TARGET.flatten().tolist(),
    },
    "train": {
        "fc1_weight_final": t_fc1_w.flatten().tolist(),
        "fc1_bias_final": t_fc1_b.flatten().tolist(),
        "base_final": t_base.flatten().tolist(),
        "lora_a_final": t_a.flatten().tolist(),
        "lora_b_final": t_b.flatten().tolist(),
        "losses": t_losses,
    },
}
print(json.dumps(golden, indent=2))
