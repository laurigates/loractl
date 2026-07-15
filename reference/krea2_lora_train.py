# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.4",
#   "safetensors",
#   "numpy",
#   "einops",
# ]
# ///
"""Reference LoRA trainer for Krea-2-Raw on torch/MPS — the known-good
substrate for the M14 real-run proof while burn's Metal backends are blocked
(wgpu autodiff produces corrupt gradients; candle's Metal allocator cannot
run the ~25 GB model — see crates/loractl-core/examples/grad_compare.rs and
the fix/m14-f32-encode-phase branch).

Everything except the backward pass is loractl's own work product:

- **Inputs**: reads loractl's `.loractl-cache/` directly — the latents and
  conditioning stacks the M12 pipeline encoded with the parity-proven M9
  VAE and M10 conditioner on CPU f32.
- **Model**: the OFFICIAL krea-2 `mmdit.py` (the file M11's parity golden
  pins loractl's port against), loaded from the same raw.safetensors.
- **Objective**: the M8 rectified-flow recipe, formula-identical to
  `flow.rs` (logit-normal + shift timesteps, x_t = (1-t)x0 + t*eps,
  v = eps - x0) and `sampling.py`'s token layout (channel-major patchify,
  text at the origin, image at (0, row, col)).
- **Export**: the Krea2Diffusers naming loractl's exporter emits and
  ComfyUI's `krea2_to_diffusers` accepts (verified key-for-key).

Usage:
  uv run reference/krea2_lora_train.py \
    --snapshot crates/loractl-core/tests/fixtures/krea2-raw \
    --dataset tmp/dataset-skslora --out tmp/krea2-skslora \
    --steps 300 --checkpoint-every 50
"""

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # mmdit.py uses @torch.compile

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


def load_mmdit_module(mmdit_py: Path):
    """Import the official mmdit.py and patch its CUDNN-pinned attention for MPS."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("krea2_mmdit", mmdit_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["krea2_mmdit"] = mod
    spec.loader.exec_module(mod)

    from einops import rearrange

    def attention(q, k, v, mask=None, scale=None, gqa=False):
        # The original pins SDPBackend.CUDNN_ATTENTION (CUDA-only) and
        # relies on SDPA's enable_gqa. Manual KV expansion + plain SDPA is
        # the same math and runs on MPS.
        if gqa and k.shape[1] != q.shape[1]:
            rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        return rearrange(x, "B H L D -> B L (H D)")

    mod.attention = attention

    # rope builds its tables in float64, which MPS lacks. The tables depend
    # only on the (tiny) position tensor, so compute them on CPU with the
    # original f64 semantics and move the f32 result to the device.
    orig_rope = mod.rope

    def rope(pos, dim, theta=1e4, ntk=1.0):
        return orig_rope(pos.cpu(), dim, theta, ntk).to(pos.device)

    mod.rope = rope
    # Both patched names are bound at call time (module-global lookup), so
    # patching the module attributes is sufficient.
    return mod


def load_cache(dataset: Path, device, dtype):
    """Read loractl's .loractl-cache: (latent [16,h,w], cond [512,12,2560], mask [512]) per example."""
    cache = dataset / ".loractl-cache"
    items = []
    for lat_path in sorted(glob.glob(str(cache / "*.latent.safetensors"))):
        name = Path(lat_path).name
        stem = name.split(".")[0]
        cond_glob = glob.glob(str(cache / f"{stem}.*.cond.safetensors"))
        if not cond_glob:
            continue
        with safe_open(lat_path, framework="pt") as f:
            latent = f.get_tensor("latent")[0]
        with safe_open(cond_glob[0], framework="pt") as f:
            cond = f.get_tensor("conditioning")[0]
            mask = f.get_tensor("mask")[0]
        items.append(
            (
                latent.to(device, dtype),
                cond.to(device, dtype),
                mask.to(device, torch.bool),
            )
        )
    if not items:
        raise SystemExit(f"no cached examples under {cache} — run loractl's encode first")
    return items


def patchify(x, p):
    from einops import rearrange

    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=p, pw=p)


def positions(txt_len, gh, gw, device):
    pos = torch.zeros(txt_len + gh * gw, 3, device=device)
    rows = torch.arange(gh, device=device).repeat_interleave(gw)
    cols = torch.arange(gw, device=device).repeat(gh)
    pos[txt_len:, 1] = rows.float()
    pos[txt_len:, 2] = cols.float()
    return pos.unsqueeze(0)


def sample_t(shift=3.0, logit_mean=0.0, logit_std=1.0, device="cpu"):
    """flow.rs logit_to_t: sigmoid(u*std+mean), then shift*t/(1+(shift-1)*t)."""
    u = torch.randn(1, device=device)
    t = torch.sigmoid(u * logit_std + logit_mean)
    return shift * t / (1.0 + (shift - 1.0) * t)


LORA_NAME_MAP = {
    "attn.wq": "attn.to_q",
    "attn.wk": "attn.to_k",
    "attn.wv": "attn.to_v",
    "attn.wo": "attn.to_out.0",
    "mlp.gate": "ff.gate",
    "mlp.up": "ff.up",
    "mlp.down": "ff.down",
}


class LoraHook:
    """A/B in fp32 attached to a frozen Linear via forward hook."""

    def __init__(self, linear, rank, alpha, device):
        d_out, d_in = linear.weight.shape
        self.a = torch.nn.Parameter(
            torch.randn(rank, d_in, device=device, dtype=torch.float32)
            / math.sqrt(d_in)
        )
        self.b = torch.nn.Parameter(
            torch.zeros(d_out, rank, device=device, dtype=torch.float32)
        )
        self.scaling = alpha / rank
        self.handle = linear.register_forward_hook(self._hook, with_kwargs=False)

    def _hook(self, module, inputs, output):
        x = inputs[0]
        delta = (x.float() @ self.a.t() @ self.b.t()) * self.scaling
        return output + delta.to(output.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mmdit-py", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    mmdit_py = args.mmdit_py or (args.snapshot / "mmdit.py")
    mod = load_mmdit_module(mmdit_py)

    cfg = mod.SingleMMDiTConfig(
        features=6144,
        tdim=256,
        txtdim=2560,
        heads=48,
        multiplier=4,
        layers=28,
        patch=2,
        channels=16,
        kvheads=12,
        txtlayers=12,
    )
    print(f"loading MMDiT on {device} ({dtype})...", flush=True)
    model = mod.SingleStreamDiT(cfg)
    with safe_open(args.snapshot / "raw.safetensors", framework="pt") as f:
        state = {k: f.get_tensor(k) for k in f.keys()}
    model.load_state_dict(state, strict=True)
    model = model.to(device, dtype).eval().requires_grad_(False)
    del state

    # LoRA attach on every trunk projection (blocks\. — 7 sites × 28 blocks).
    hooks = {}
    for i, block in enumerate(model.blocks):
        for path, mapped in LORA_NAME_MAP.items():
            parent, leaf = path.split(".")
            linear = getattr(getattr(block, parent), leaf)
            hooks[f"transformer_blocks.{i}.{mapped}"] = LoraHook(
                linear, args.rank, args.alpha, device
            )
    params = [p for h in hooks.values() for p in (h.a, h.b)]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    print(f"attached {len(hooks)} LoRA sites ({sum(p.numel() for p in params)/1e6:.1f}M params)", flush=True)

    items = load_cache(args.dataset, device, dtype)
    print(f"dataset: {len(items)} cached examples", flush=True)

    def export(path):
        tensors = {}
        for base, h in hooks.items():
            tensors[f"{base}.lora_down.weight"] = h.a.detach().float().cpu()
            tensors[f"{base}.lora_up.weight"] = h.b.detach().float().cpu()
            tensors[f"{base}.alpha"] = torch.tensor([args.alpha], dtype=torch.float32)
        save_file(tensors, str(path))
        print(f"exported {len(tensors)} tensors -> {path}", flush=True)

    for step in range(1, args.steps + 1):
        latent, cond, mask = items[(step - 1) % len(items)]
        x0 = latent.unsqueeze(0)
        _, c, h, w = x0.shape
        eps = torch.randn_like(x0)
        t = sample_t(shift=args.shift, device=device).to(dtype)
        xt = (1.0 - t) * x0 + t * eps
        target = patchify(eps - x0, cfg.patch)

        img = patchify(xt, cfg.patch)
        gh, gw = h // cfg.patch, w // cfg.patch
        txt_len = cond.shape[0]
        pos = positions(txt_len, gh, gw, device)
        full_mask = torch.cat(
            [mask.unsqueeze(0), torch.ones(1, gh * gw, device=device, dtype=torch.bool)],
            dim=1,
        )

        pred = model(img, cond.unsqueeze(0), t, pos, full_mask)
        loss = F.mse_loss(pred.float(), target.float())
        lv = loss.item()
        if not math.isfinite(lv):
            raise SystemExit(f"non-finite loss at step {step}")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        print(f"step {step}/{args.steps} loss {lv:.4f}", flush=True)

        if step % args.checkpoint_every == 0 and step != args.steps:
            export(args.out / f"checkpoint-{step}.safetensors")

    export(args.out / "skslora.safetensors")


if __name__ == "__main__":
    main()
