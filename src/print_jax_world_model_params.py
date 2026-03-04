#!/usr/bin/env python
"""Load the world model with JAX and print parameter names (and shapes) for comparison with the PyTorch model.

Usage:
    # From repo root, with an Orbax checkpoint directory (e.g. output of JAX training):
    python -m src.print_jax_world_model_params --checkpoint_dir /path/to/checkpoint_dir

    # Optional: also print PyTorch model keys for side-by-side comparison
    python -m src.print_jax_world_model_params --checkpoint_dir /path/to/checkpoint_dir --print_torch

The script loads the JAX SolarisMPModel from the checkpoint, flattens the state to dotted keys
(e.g. blocks.0.norm1.scale, blocks.0.self_attn.q.kernel), and prints each key with its shape.
This helps align JAX/Flax param names with the PyTorch state_dict keys used in inference.py.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root on path before other imports that need convert_checkpoints
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
os.chdir(_repo_root)

import numpy as np

# Single-device JAX for local inspection (no multi-host)
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from convert_checkpoints import _make_mp_model, flatten_state

# Subdir names under the parent folder where the world model Orbax checkpoint may live (same as convert_checkpoints)
WORLD_MODEL_CANDIDATES = ("solaris.pt")


def resolve_world_model_checkpoint(parent_dir):
    """Return path to world model checkpoint under parent_dir, or None."""
    if not os.path.isdir(parent_dir):
        return None
    for name in WORLD_MODEL_CANDIDATES:
        path = os.path.join(parent_dir, name)
        if os.path.isdir(path):
            return path
    # Try any subdirectory
    try:
        for name in sorted(os.listdir(parent_dir)):
            path = os.path.join(parent_dir, name)
            if os.path.isdir(path):
                return path
    except OSError:
        pass
    return None


def print_jax_params(ckpt_path, print_torch=False):
    """Load JAX world model from checkpoint and print all parameter names and shapes.
    ckpt_path can be the parent folder (e.g. checkpoints) or the exact world-model checkpoint subdir (e.g. checkpoints/solaris.pt).
    """
    if not os.path.exists(ckpt_path):
        print(f"Path does not exist: {ckpt_path}")
        return
    if not os.path.isdir(ckpt_path):
        print(f"Checkpoint path is not a directory: {ckpt_path}")
        print("Pass the parent folder (e.g. checkpoints) or the world-model subdir (e.g. checkpoints/solaris.pt).")
        return

    # Accept parent folder: resolve to world-model checkpoint subdir if present
    actual_ckpt = ckpt_path
    resolved = resolve_world_model_checkpoint(ckpt_path)
    if resolved:
        actual_ckpt = resolved
        print(f"Using world model checkpoint: {actual_ckpt}")

    print("Creating JAX SolarisMPModel (multiplayer_attn)...")
    model = _make_mp_model(multiplayer_method="multiplayer_attn")

    print("Splitting model to get state structure...")
    _graph, state = nnx.split(model)

    print(f"Restoring from: {actual_ckpt}")
    checkpointer = ocp.StandardCheckpointer()
    try:
        restored = checkpointer.restore(actual_ckpt, state)
    except FileNotFoundError as e:
        print(f"Orbax could not find a valid checkpoint at: {actual_ckpt}")
        if actual_ckpt != ckpt_path:
            print(f"(resolved from parent: {ckpt_path})")
        print()
        print("Pass the parent folder containing the world model subdir (e.g. checkpoints/ with checkpoints/solaris.pt inside),")
        print("or the exact Orbax checkpoint directory. If you only have a converted PyTorch .pt file, use inference.py with LMW_DEBUG_LOAD=1.")
        if os.path.isdir(ckpt_path):
            try:
                subdirs = [d for d in os.listdir(ckpt_path) if os.path.isdir(os.path.join(ckpt_path, d))]
                if subdirs:
                    print(f"Subdirs under {ckpt_path}: {subdirs[:10]}")
            except OSError:
                pass
        raise SystemExit(1) from e

    print("Flattening state to dotted keys...")
    flat = flatten_state(restored)

    print()
    print("=" * 80)
    print("JAX WORLD MODEL PARAMETER NAMES (after restore + flatten)")
    print("=" * 80)
    print("Format: key -> shape (dtype)")
    print()

    keys_sorted = sorted(flat.keys())
    for key in keys_sorted:
        arr = flat[key]
        shape = np.asarray(arr).shape
        dtype = np.asarray(arr).dtype
        print(f"  {key}")
        print(f"    -> shape {shape}  dtype {dtype}")

    print()
    print(f"Total keys: {len(keys_sorted)}")
    print()

    if print_torch:
        print("=" * 80)
        print("PYTORCH WORLD MODEL PARAMETER NAMES (SolarisMPModelTorch.state_dict())")
        print("=" * 80)
        print("Format: key -> shape")
        print()

        try:
            import torch
            from src.models.torch.world_model_mp_torch import SolarisMPModelTorch
        except Exception as e:
            print(f"Could not load PyTorch model for comparison: {e}")
            return

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
        torch_model = SolarisMPModelTorch(
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
            cross_attn_norm=False,
            action_config=action_config,
            inject_sample_info=False,
            eps=1e-6,
            multiplayer_method="multiplayer_attn",
            num_players=2,
        )
        sd = torch_model.state_dict()
        for key in sorted(sd.keys()):
            t = sd[key]
            print(f"  {key}")
            print(f"    -> shape {tuple(t.shape)}  dtype {t.dtype}")

        print()
        print(f"Total PyTorch keys: {len(sd)}")
        print()
        print("Use the JAX keys above to see the exact names/shapes from the checkpoint.")
        print("inference.py remaps: .kernel -> .weight, .scale -> .weight, blocks.N.0 -> ffn.0, etc.")


def main():
    parser = argparse.ArgumentParser(
        description="Load JAX world model and print parameter names for Torch alignment"
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        nargs="?",
        default=None,
        help="Path to Orbax checkpoint directory (e.g. .../solaris.pt or .../matrix-game-init.pt)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        dest="ckpt_dir_opt",
        help="Same as positional checkpoint_dir",
    )
    parser.add_argument(
        "--print_torch",
        action="store_true",
        help="Also print PyTorch SolarisMPModelTorch state_dict keys for comparison",
    )
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir or args.ckpt_dir_opt
    if not ckpt_dir:
        print("Usage: python -m src.print_jax_world_model_params <checkpoint_dir> [--print_torch]")
        print("   or: python -m src.print_jax_world_model_params --checkpoint_dir <path> [--print_torch]")
        sys.exit(1)

    print_jax_params(ckpt_dir, print_torch=args.print_torch)


if __name__ == "__main__":
    main()
