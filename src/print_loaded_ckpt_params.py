#!/usr/bin/env python
"""Load a converted PyTorch world-model checkpoint and print parameter names and shapes.

Usage:
    python -m src.print_loaded_ckpt_params checkpoint_pt/solaris.pt
    python -m src.print_loaded_ckpt_params checkpoint_pt/solaris.pt -o loaded_ckpt_params.txt
"""

from __future__ import annotations

import argparse
import os
import sys

# Repo root on path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Load a PyTorch checkpoint and print parameter names/shapes"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",
        default="checkpoint_pt/solaris.pt",
        help="Path to .pt checkpoint (default: checkpoint_pt/solaris.pt)",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Write parameter list to this txt file")
    args = parser.parse_args()

    path = args.checkpoint
    if not os.path.isfile(path):
        print(f"Error: checkpoint not found: {path}")
        sys.exit(1)

    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Unwrap state_dict if saved as {"state_dict": ..., ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        # Full model
        if hasattr(ckpt, "state_dict"):
            state = ckpt.state_dict()
        else:
            print("Error: checkpoint is not a state dict or model")
            sys.exit(1)

    if not isinstance(state, dict):
        print("Error: state is not a dict")
        sys.exit(1)

    keys_sorted = sorted(state.keys())
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("LOADED CHECKPOINT PARAMETERS")
    lines.append("=" * 80)
    lines.append(f"Source: {path}")
    lines.append("Format: key -> shape, dtype")
    lines.append("")
    for key in keys_sorted:
        v = state[key]
        if isinstance(v, torch.Tensor):
            lines.append(f"  {key}")
            lines.append(f"    -> shape {tuple(v.shape)}  dtype {v.dtype}")
        else:
            lines.append(f"  {key}")
            lines.append(f"    -> (not a tensor: {type(v).__name__})")
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
