# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "safetensors"]
# ///
"""Re-key the tiny diffusers Qwen-Image VAE fixture into ComfyUI-native (Qwen/WAN)
naming for the M-Phase-4 native-VAE parity guard (issue #25, ComfyUI-support arc).

A ComfyUI Qwen-Image VAE file (`models/vae/qwen/qwen_image_vae.safetensors`) names
the SAME weights as the diffusers `AutoencoderKLQwenImage` file this port mirrors,
but under the original Wan-VAE state-dict scheme (top-level `conv1`/`conv2`,
`{enc,dec}.conv1`, `{…}.head.{0,2}`, `{…}.middle.{0,1,2}`, `.residual.{0,2,3,6}` /
`.shortcut`, a flat encoder `downsamples.N` and a flat decoder `upsamples.N`).

This script produces the tiny native fixture purely by re-keying the committed
tiny diffusers fixture's tensors — no torch, no network, no re-training — so the
encode-parity test loads the SAME weights under BOTH schemes and asserts identical
latents. The `diffusers → native` map here is the exact inverse of diffusers' own
`convert_wan_vae_to_diffusers` (`loaders/single_file_utils.py`); the script
round-trips every produced native key back through a verbatim copy of that
converter and asserts it reproduces the original diffusers key set, so a bug in
the inverse fails the generator, not silently the fixture.

Output (to --out dir): tiny-qwen-vae-native/qwen_image_vae.safetensors
"""

import argparse
import os
import re
import sys

import numpy as np
from safetensors.numpy import load_file, save_file

DIFFUSERS_FIXTURE = "crates/loractl-core/tests/fixtures/tiny-qwen-vae/diffusion_pytorch_model.safetensors"


# ---- diffusers → native (the inverse of convert_wan_vae_to_diffusers) --------
def _res_sub_to_native(sub: str) -> str:
    """Res-block sub-path: diffusers norm/conv/shortcut → native residual index.

    resample.1.* / time_conv.* pass through unchanged (native keeps them)."""
    return {
        "norm1.gamma": "residual.0.gamma",
        "conv1.weight": "residual.2.weight",
        "conv1.bias": "residual.2.bias",
        "norm2.gamma": "residual.3.gamma",
        "conv2.weight": "residual.6.weight",
        "conv2.bias": "residual.6.bias",
        "conv_shortcut.weight": "shortcut.weight",
        "conv_shortcut.bias": "shortcut.bias",
    }.get(sub, sub)


def diffusers_to_native(k: str) -> str:
    # Top-level quant convs.
    if k.startswith("quant_conv."):
        return "conv1." + k[len("quant_conv.") :]
    if k.startswith("post_quant_conv."):
        return "conv2." + k[len("post_quant_conv.") :]
    for pre in ("encoder", "decoder"):
        if k == f"{pre}.conv_in.weight":
            return f"{pre}.conv1.weight"
        if k == f"{pre}.conv_in.bias":
            return f"{pre}.conv1.bias"
        if k == f"{pre}.norm_out.gamma":
            return f"{pre}.head.0.gamma"
        if k == f"{pre}.conv_out.weight":
            return f"{pre}.head.2.weight"
        if k == f"{pre}.conv_out.bias":
            return f"{pre}.head.2.bias"
        m = re.match(rf"^{pre}\.mid_block\.resnets\.(\d+)\.(.+)$", k)
        if m:
            native_mid = 0 if int(m.group(1)) == 0 else 2
            return f"{pre}.middle.{native_mid}.{_res_sub_to_native(m.group(2))}"
        m = re.match(rf"^{pre}\.mid_block\.attentions\.0\.(.+)$", k)
        if m:
            return f"{pre}.middle.1.{m.group(1)}"
    # Encoder trunk is flat: down_blocks.N → downsamples.N.
    m = re.match(r"^encoder\.down_blocks\.(\d+)\.(.+)$", k)
    if m:
        return f"encoder.downsamples.{m.group(1)}.{_res_sub_to_native(m.group(2))}"
    # Decoder trunk regroups: up_blocks.X.resnets.Y → flat upsamples.{X*4+Y}
    # (num_res_blocks=2 ⇒ stride 4), up_blocks.X.upsamplers.0 → upsamples.{X*4+3}.
    m = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", k)
    if m:
        flat = int(m.group(1)) * 4 + int(m.group(2))
        return f"decoder.upsamples.{flat}.{_res_sub_to_native(m.group(3))}"
    m = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.+)$", k)
    if m:
        flat = int(m.group(1)) * 4 + 3
        return f"decoder.upsamples.{flat}.{m.group(2)}"
    raise ValueError(f"no native mapping for diffusers key {k!r}")


# ---- verbatim copy of diffusers' convert_wan_vae_to_diffusers (the ground -----
#      truth forward map — used only to round-trip-verify this script's inverse).
def convert_wan_vae_to_diffusers(checkpoint):
    converted_state_dict = {}
    middle_key_mapping = {}
    for pre in ("encoder", "decoder"):
        middle_key_mapping.update(
            {
                f"{pre}.middle.0.residual.0.gamma": f"{pre}.mid_block.resnets.0.norm1.gamma",
                f"{pre}.middle.0.residual.2.bias": f"{pre}.mid_block.resnets.0.conv1.bias",
                f"{pre}.middle.0.residual.2.weight": f"{pre}.mid_block.resnets.0.conv1.weight",
                f"{pre}.middle.0.residual.3.gamma": f"{pre}.mid_block.resnets.0.norm2.gamma",
                f"{pre}.middle.0.residual.6.bias": f"{pre}.mid_block.resnets.0.conv2.bias",
                f"{pre}.middle.0.residual.6.weight": f"{pre}.mid_block.resnets.0.conv2.weight",
                f"{pre}.middle.2.residual.0.gamma": f"{pre}.mid_block.resnets.1.norm1.gamma",
                f"{pre}.middle.2.residual.2.bias": f"{pre}.mid_block.resnets.1.conv1.bias",
                f"{pre}.middle.2.residual.2.weight": f"{pre}.mid_block.resnets.1.conv1.weight",
                f"{pre}.middle.2.residual.3.gamma": f"{pre}.mid_block.resnets.1.norm2.gamma",
                f"{pre}.middle.2.residual.6.bias": f"{pre}.mid_block.resnets.1.conv2.bias",
                f"{pre}.middle.2.residual.6.weight": f"{pre}.mid_block.resnets.1.conv2.weight",
            }
        )
    attention_mapping = {}
    for pre in ("encoder", "decoder"):
        for suf in ("norm.gamma", "to_qkv.weight", "to_qkv.bias", "proj.weight", "proj.bias"):
            attention_mapping[f"{pre}.middle.1.{suf}"] = f"{pre}.mid_block.attentions.0.{suf}"
    head_mapping = {}
    for pre in ("encoder", "decoder"):
        head_mapping[f"{pre}.head.0.gamma"] = f"{pre}.norm_out.gamma"
        head_mapping[f"{pre}.head.2.bias"] = f"{pre}.conv_out.bias"
        head_mapping[f"{pre}.head.2.weight"] = f"{pre}.conv_out.weight"
    quant_mapping = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
    }
    for key, value in checkpoint.items():
        if key in middle_key_mapping:
            converted_state_dict[middle_key_mapping[key]] = value
        elif key in attention_mapping:
            converted_state_dict[attention_mapping[key]] = value
        elif key in head_mapping:
            converted_state_dict[head_mapping[key]] = value
        elif key in quant_mapping:
            converted_state_dict[quant_mapping[key]] = value
        elif key == "encoder.conv1.weight":
            converted_state_dict["encoder.conv_in.weight"] = value
        elif key == "encoder.conv1.bias":
            converted_state_dict["encoder.conv_in.bias"] = value
        elif key == "decoder.conv1.weight":
            converted_state_dict["decoder.conv_in.weight"] = value
        elif key == "decoder.conv1.bias":
            converted_state_dict["decoder.conv_in.bias"] = value
        elif key.startswith("encoder.downsamples."):
            new_key = key.replace("encoder.downsamples.", "encoder.down_blocks.")
            for a, b in (
                (".residual.0.gamma", ".norm1.gamma"),
                (".residual.2.bias", ".conv1.bias"),
                (".residual.2.weight", ".conv1.weight"),
                (".residual.3.gamma", ".norm2.gamma"),
                (".residual.6.bias", ".conv2.bias"),
                (".residual.6.weight", ".conv2.weight"),
                (".shortcut.bias", ".conv_shortcut.bias"),
                (".shortcut.weight", ".conv_shortcut.weight"),
            ):
                if a in new_key:
                    new_key = new_key.replace(a, b)
                    break
            converted_state_dict[new_key] = value
        elif key.startswith("decoder.upsamples."):
            block_idx = int(key.split(".")[2])
            if "residual" in key:
                if block_idx in [0, 1, 2]:
                    nb, ri = 0, block_idx
                elif block_idx in [4, 5, 6]:
                    nb, ri = 1, block_idx - 4
                elif block_idx in [8, 9, 10]:
                    nb, ri = 2, block_idx - 8
                elif block_idx in [12, 13, 14]:
                    nb, ri = 3, block_idx - 12
                else:
                    converted_state_dict[key] = value
                    continue
                if ".residual.0.gamma" in key:
                    new_key = f"decoder.up_blocks.{nb}.resnets.{ri}.norm1.gamma"
                elif ".residual.2.bias" in key:
                    new_key = f"decoder.up_blocks.{nb}.resnets.{ri}.conv1.bias"
                elif ".residual.2.weight" in key:
                    new_key = f"decoder.up_blocks.{nb}.resnets.{ri}.conv1.weight"
                elif ".residual.3.gamma" in key:
                    new_key = f"decoder.up_blocks.{nb}.resnets.{ri}.norm2.gamma"
                elif ".residual.6.bias" in key:
                    new_key = f"decoder.up_blocks.{nb}.resnets.{ri}.conv2.bias"
                elif ".residual.6.weight" in key:
                    new_key = f"decoder.up_blocks.{nb}.resnets.{ri}.conv2.weight"
                else:
                    new_key = key
                converted_state_dict[new_key] = value
            elif ".shortcut." in key:
                if block_idx == 4:
                    new_key = key.replace(".shortcut.", ".resnets.0.conv_shortcut.").replace(
                        "decoder.upsamples.4", "decoder.up_blocks.1"
                    )
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.").replace(
                        ".shortcut.", ".conv_shortcut."
                    )
                converted_state_dict[new_key] = value
            elif ".resample." in key or ".time_conv." in key:
                if block_idx == 3:
                    new_key = key.replace("decoder.upsamples.3", "decoder.up_blocks.0.upsamplers.0")
                elif block_idx == 7:
                    new_key = key.replace("decoder.upsamples.7", "decoder.up_blocks.1.upsamplers.0")
                elif block_idx == 11:
                    new_key = key.replace("decoder.upsamples.11", "decoder.up_blocks.2.upsamplers.0")
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                converted_state_dict[new_key] = value
            else:
                converted_state_dict[key.replace("decoder.upsamples.", "decoder.up_blocks.")] = value
        else:
            converted_state_dict[key] = value
    return converted_state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="crates/loractl-core/tests/fixtures")
    args = parser.parse_args()

    diffusers = load_file(DIFFUSERS_FIXTURE)
    native = {diffusers_to_native(k): v for k, v in diffusers.items()}
    assert len(native) == len(diffusers), (
        f"diffusers→native is not injective: {len(native)} native vs {len(diffusers)} diffusers keys"
    )

    # Round-trip: the authoritative forward converter must turn our native keys
    # back into EXACTLY the diffusers key set, with byte-identical tensors.
    back = convert_wan_vae_to_diffusers(native)
    assert set(back.keys()) == set(diffusers.keys()), (
        f"round-trip key mismatch: extra={sorted(set(back) - set(diffusers))[:5]} "
        f"missing={sorted(set(diffusers) - set(back))[:5]}"
    )
    for k in diffusers:
        assert np.array_equal(back[k], diffusers[k]), f"round-trip tensor mismatch for {k}"

    save_dir = os.path.join(args.out, "tiny-qwen-vae-native")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "qwen_image_vae.safetensors")
    # `contiguous`-safe: values from load_file are already plain np arrays.
    save_file(native, out_path)
    print(
        f"native VAE fixture written to {out_path} ({len(native)} tensors, "
        f"re-keyed from {DIFFUSERS_FIXTURE} and round-trip-verified)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
