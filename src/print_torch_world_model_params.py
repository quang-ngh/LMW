#!/usr/bin/env python
"""Create SolarisMPModelTorch (no checkpoint load) and print parameter names and shapes.

Usage:
    python -m src.print_torch_world_model_params
    python -m src.print_torch_world_model_params -o torch_world_model_params.txt
"""

from __future__ import annotations

import argparse
import os
import sys

# Repo root on path for src.models
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch


def main():
    parser = argparse.ArgumentParser(description="Print PyTorch SolarisMPModelTorch parameter names/shapes (no load)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Write parameter list to this txt file")
    args = parser.parse_args()

    from src.models.torch.world_model_mp_torch import SolarisMPModelTorch

    action_config = {
        "mouse_dim_in": 2,
        "keyboard_dim_in": 23,
        "hidden_size": 128,
        "img_hidden_size": 1536,
        "keyboard_hidden_dim": 1024,
        "mouse_hidden_dim": 1024,
        "vae_time_compression_ratio": 4,
        "windows_size": 3,
        "heads_num": 16,
        "patch_size": [1, 2, 2],
        "qk_norm": True,
        "qkv_bias": False,
        "rope_dim_list": [8, 28, 28],
        "rope_theta": 256,
        "mouse_qk_dim_list": [8, 28, 28],
        "enable_mouse": True,
        "enable_keyboard": True,
        "left_action_padding": 11,
    }

    print("Creating SolarisMPModelTorch (no checkpoint load)...")
    model = SolarisMPModelTorch(
        model_type="i2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=36,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=12,
        num_layers=30,
        local_attn_size=6,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        action_config=action_config,
        inject_sample_info=False,
        eps=1e-6,
        multiplayer_method="multiplayer_attn",
        num_players=2,
    )

    sd = model.state_dict()
    keys_sorted = sorted(sd.keys())

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("PYTORCH SolarisMPModelTorch PARAMETERS (created, not loaded)")
    lines.append("=" * 80)
    lines.append("Format: key -> shape, dtype")
    lines.append("")
    for key in keys_sorted:
        t = sd[key]
        lines.append(f"  {key}")
        lines.append(f"    -> shape {tuple(t.shape)}  dtype {t.dtype}")
    lines.append("")
    lines.append(f"Total parameters: {len(keys_sorted)}")

    for line in lines:
        print(line)

    if args.output:
        out_path = os.path.abspath(args.output)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\nWrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
