# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.*", "diffusers>=0.35", "safetensors", "numpy"]
# ///
"""Qwen-Image VAE PyTorch reference for loractl M9 (issue #20) encode/decode parity.

Krea 2's autoencoder is the STOCK Qwen-Image VAE: `krea-ai/krea-2`'s
`autoencoder.py` is 22 lines that wrap
`diffusers.AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae")`
plus per-channel `latents_mean`/`latents_std` (de)normalization. So the
authoritative architecture source for the burn port is diffusers'
`autoencoder_kl_qwenimage.py`, and this script is its golden generator.

Default (tiny) mode: constructs a REAL `AutoencoderKLQwenImage` at a tiny fixed
config with seeded random weights, saves its safetensors, and dumps staged
golden activations for a FIXED seeded input image: encode stages
(conv_in -> down blocks -> mid block -> quant_conv moments -> latent mode ->
normalized latent) and decode stages (conv_in -> mid block -> clamped image).
Because the burn implementation loads the SAME checked-in safetensors, both
frameworks run identical weights and differ only by f32 rounding — the exact
M3 tiny-GPT-2 strategy, transplanted to the VAE.

--real mode: downloads the real `Qwen/Qwen-Image` VAE (the exact checkpoint
Krea 2 uses), re-saves it as float32 safetensors (the shipped file is BF16;
re-saving sidesteps loader dtype conversion as a variable), and dumps the same
staged golden on a seeded 64x64 input. Output is gitignored — regenerate with
`just vae-real-reference`.

Emitted (to --out dir):
  tiny: tiny-qwen-vae/model.safetensors + qwen_vae_tiny_golden.json
  real: qwen-vae-real/model.safetensors + qwen_vae_real_golden.json
"""

import argparse
import json
import os
import platform
import sys

import torch
from diffusers import AutoencoderKLQwenImage

# Fixed tiny architecture — real AutoencoderKLQwenImage shape, minimal dims.
# Duplicated in the Rust tiny config (QwenVaeConfig::tiny()) so the burn module
# tree matches exactly. Chosen to exercise every layer kind the real config
# (base_dim=96, dim_mult=[1,2,4,4], temperal_downsample=[F,T,T], z_dim=16)
# uses: downsample2d AND downsample3d (+time_conv), upsample3d AND upsample2d,
# res blocks with and without conv_shortcut, and the mid-block attention.
TINY_CFG = dict(
    base_dim=8,
    z_dim=4,
    dim_mult=[1, 2, 2],
    # 2, like the real config: the second same-stage residual block is what
    # exercises the constructors' in_dim -> out_dim advance in the ALWAYS-RUN
    # parity test (at 1, that path would only ever run via the opt-in
    # real-weights test).
    num_res_blocks=2,
    attn_scales=[],
    temperal_downsample=[False, True],
    dropout=0.0,
    # Arbitrary fixed per-channel stats — the tiny analogue of the real
    # config's measured latent statistics. Must match QwenVaeConfig::tiny().
    latents_mean=[0.1, -0.2, 0.3, -0.4],
    latents_std=[1.5, 0.8, 1.2, 2.0],
)
TINY_IMAGE = 24  # divisible by the tiny spatial compression (2 downsamples = 4)
REAL_IMAGE = 64  # divisible by the real spatial compression (3 downsamples = 8)


def staged_forward(vae: AutoencoderKLQwenImage, x: torch.Tensor) -> dict:
    """Run encode->normalize->denormalize->decode, capturing localizing stages.

    `x` is a 4D image batch [b, 3, h, w]; the VAE is a video model, so a T=1
    frame axis is added (exactly what krea-2's wrapper does via einops).
    Returns a dict of stage name -> tensor.
    """
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)

    stages: dict[str, torch.Tensor] = {}

    def grab(name):
        def hook(_module, inputs, output):
            # Also keep the mid block's *input*: it is the down/up trunk's
            # output, letting the parity test split trunk vs mid failures.
            if name == "enc_mid":
                stages["enc_down"] = inputs[0].detach().clone()
            stages[name] = output.detach().clone()

        return hook

    handles = [
        vae.encoder.conv_in.register_forward_hook(grab("enc_conv_in")),
        vae.encoder.mid_block.register_forward_hook(grab("enc_mid")),
        vae.quant_conv.register_forward_hook(grab("moments")),
        vae.decoder.conv_in.register_forward_hook(grab("dec_conv_in")),
        vae.decoder.mid_block.register_forward_hook(grab("dec_mid")),
    ]
    try:
        with torch.no_grad():
            x5 = x.unsqueeze(2)  # [b, 3, 1, h, w]
            posterior = vae.encode(x5).latent_dist
            latent = posterior.mode()  # deterministic: the distribution mean
            latent_norm = (latent - mean) / std
            # Decode takes a *normalized* latent (krea-2 convention): denorm,
            # then the diffusers decode (which clamps to [-1, 1] internally).
            decoded = vae.decode(latent_norm * std + mean).sample
    finally:
        for h in handles:
            h.remove()

    stages["latent_mode"] = latent
    stages["latent_norm"] = latent_norm
    stages["decoded"] = decoded
    return stages


def dump(stages: dict, x: torch.Tensor, vae, path: str, extra: dict) -> None:
    golden = {
        "input": x.flatten().tolist(),
        "input_shape": list(x.shape),
        "latents_mean": list(vae.config.latents_mean),
        "latents_std": list(vae.config.latents_std),
        "provenance": {
            "torch": torch.__version__,
            "diffusers": __import__("diffusers").__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        **extra,
    }
    for name, t in stages.items():
        golden[name] = t.flatten().tolist()
        golden[f"{name}_shape"] = list(t.shape)
    with open(path, "w") as f:
        json.dump(golden, f)
    print(f"golden written to {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="crates/loractl-core/tests/fixtures")
    parser.add_argument(
        "--real",
        action="store_true",
        help="use the real Qwen/Qwen-Image VAE instead of the tiny seeded one",
    )
    args = parser.parse_args()
    torch.manual_seed(9)  # M9

    if args.real:
        vae = AutoencoderKLQwenImage.from_pretrained(
            "Qwen/Qwen-Image", subfolder="vae"
        ).to(torch.float32)
        save_dir = os.path.join(args.out, "qwen-vae-real")
        golden_path = os.path.join(args.out, "qwen_vae_real_golden.json")
        size = REAL_IMAGE
    else:
        vae = AutoencoderKLQwenImage(**TINY_CFG)
        save_dir = os.path.join(args.out, "tiny-qwen-vae")
        golden_path = os.path.join(args.out, "qwen_vae_tiny_golden.json")
        size = TINY_IMAGE

    vae.eval()
    os.makedirs(save_dir, exist_ok=True)
    vae.save_pretrained(save_dir, safe_serialization=True)

    from safetensors import safe_open

    st_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
    with safe_open(st_path, framework="pt") as f:
        keys = sorted(f.keys())

    # A fixed seeded image in [-1, 1] — the VAE's expected input range.
    x = torch.rand(1, 3, size, size) * 2.0 - 1.0
    stages = staged_forward(vae, x)
    dump(stages, x, vae, golden_path, {"safetensors_keys": keys})
    print(f"vae saved to {save_dir} ({len(keys)} tensors)", file=sys.stderr)


if __name__ == "__main__":
    main()
