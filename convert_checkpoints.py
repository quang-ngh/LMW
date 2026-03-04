#!/usr/bin/env python
"""Convert Orbax/OCDBT checkpoints to PyTorch .pt state dicts.

This script loads JAX/Flax NNX model checkpoints saved in Orbax format
and converts them to PyTorch .pt files that can be loaded with torch.load().

Usage:
    python convert_checkpoints.py [--checkpoint_dir PATH] [--output_dir PATH]

The script handles three types of models:
    - World model (SolarisMPModel / SolarisSPModel)
    - VAE (WanVAE_)
    - CLIP (CLIPModel)
"""

import argparse
import functools
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from src.models.model_loaders import get_jax_clip_model, get_vae_model

ACTION_CONFIG = OmegaConf.create(
    {
        "blocks": list(range(30)),
        "enable_keyboard": True,
        "enable_mouse": True,
        "heads_num": 16,
        "hidden_size": 128,
        "img_hidden_size": 1536,
        "keyboard_dim_in": 23,
        "keyboard_hidden_dim": 1024,
        "mouse_dim_in": 2,
        "mouse_hidden_dim": 1024,
        "mouse_qk_dim_list": [8, 28, 28],
        "patch_size": [1, 2, 2],
        "qk_norm": True,
        "qkv_bias": False,
        "rope_dim_list": [8, 28, 28],
        "rope_theta": 256,
        "vae_time_compression_ratio": 4,
        "windows_size": 3,
        "left_action_padding": 11,
    }
)


def _make_sp_model():
    from src.models.singleplayer.world_model import SolarisSPModel

    return SolarisSPModel(
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
        action_config=ACTION_CONFIG,
        inject_sample_info=False,
        eps=1e-6,
        rngs=nnx.Rngs(0),
        platform="gpu",
    )


def _make_mp_model(multiplayer_method="multiplayer_attn"):
    from src.models.multiplayer.world_model import SolarisMPModel

    return SolarisMPModel(
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
        action_config=ACTION_CONFIG,
        inject_sample_info=False,
        eps=1e-6,
        rngs=nnx.Rngs(0),
        multiplayer_method=multiplayer_method,
        num_players=2,
        platform="gpu",
    )


def flatten_state(state, prefix=""):
    """Recursively flatten a Flax NNX state into ``{dotted_path: numpy_array}``.
    Skips RNG/PRNG key arrays (they cannot be converted to NumPy and are not needed for PyTorch).
    """
    result = {}
    if isinstance(state, nnx.VariableState):
        val = state.value
        try:
            result[prefix] = np.asarray(val)
        except TypeError:
            # Skip PRNG keys and other non-numeric state (e.g. JAX key dtype)
            pass
    elif hasattr(state, "items"):
        for key, subtree in state.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_state(subtree, child))
    elif isinstance(state, (list, tuple)):
        for idx, subtree in enumerate(state):
            child = f"{prefix}.{idx}" if prefix else str(idx)
            result.update(flatten_state(subtree, child))
    return result


def _numpy_to_torch_supported(arr):
    """Convert numpy array to a torch tensor. Converts unsupported dtypes (e.g. bfloat16) to float32."""
    arr = np.asarray(arr)
    try:
        return torch.from_numpy(arr.copy())
    except TypeError:
        # e.g. ml_dtypes.bfloat16 not supported by torch.from_numpy
        arr = arr.astype(np.float32)
        return torch.from_numpy(arr.copy())


def convert_checkpoint(ckpt_path, model, out_path):
    """Load an Orbax checkpoint into *model*'s state and save as a torch file."""
    print(f"  Loading  : {ckpt_path}")
    _graph, state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(ckpt_path, state)

    flat = flatten_state(restored)
    torch_state = {k: _numpy_to_torch_supported(v) for k, v in flat.items()}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(torch_state, out_path)

    n_params = sum(t.numel() for t in torch_state.values())
    mb = sum(t.numel() * t.element_size() for t in torch_state.values()) / 1024**2
    print(f"  Saved    : {out_path}")
    print(f"  Keys     : {len(torch_state)}")
    print(f"  Params   : {n_params:,}  ({mb:.1f} MB)")
    return torch_state


def main():
    parser = argparse.ArgumentParser(
        description="Convert Orbax/OCDBT checkpoints to PyTorch .pt state dicts"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing orbax checkpoint folders (default: checkpoints)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints_torch",
        help="Output directory for torch .pt files (default: checkpoints_torch)",
    )
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    out_dir = args.output_dir

    if not os.path.isdir(ckpt_dir):
        print(f"ERROR: checkpoint directory not found: {ckpt_dir}")
        sys.exit(1)

    ckpt_dir = os.path.abspath(ckpt_dir)
    out_dir = os.path.abspath(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Checkpoint dir : {ckpt_dir}")
    print(f"Output dir     : {out_dir}")
    print(f"JAX devices    : {jax.devices()}")
    print()

    # ------------------------------------------------------------------
    # 1. VAE
    # ------------------------------------------------------------------
    vae_path = os.path.join(ckpt_dir, "vae.pt")
    if os.path.isdir(vae_path):
        print("[VAE]")
        vae_model = get_vae_model()
        convert_checkpoint(vae_path, vae_model, os.path.join(out_dir, "vae.pt"))
        del vae_model
        print()

    # ------------------------------------------------------------------
    # 2. CLIP
    # ------------------------------------------------------------------
    clip_path = os.path.join(ckpt_dir, "clip.pt")
    if os.path.isdir(clip_path):
        print("[CLIP]")
        clip_model = get_jax_clip_model()
        convert_checkpoint(clip_path, clip_model, os.path.join(out_dir, "clip.pt"))
        del clip_model
        print()

    # ------------------------------------------------------------------
    # 3. Multiplayer world-model checkpoints
    # ------------------------------------------------------------------
    # mp_names = [
    #     "solaris.pt",
    #     # "mp_bidirectional_120000.pt",
    #     # "mp_causal_60000.pt",
    # ]
    # for name in mp_names:
    #     path = os.path.join(ckpt_dir, name)
    #     if os.path.isdir(path):
    #         print(f"[MP world-model] {name}")
    #         model = _make_mp_model(multiplayer_method="multiplayer_attn")
    #         try:
    #             convert_checkpoint(path, model, os.path.join(out_dir, name))
    #         except Exception as exc:
    #             print(f"  FAILED: {exc}")
    #         del model
    #         print()

    # # ------------------------------------------------------------------
    # # 4. Single-player world-model checkpoints
    # # ------------------------------------------------------------------
    # sp_names = [
    #     "sp_bidirectional_pretrain_120000.pt",
    # ]
    # for name in sp_names:
    #     path = os.path.join(ckpt_dir, name)
    #     if os.path.isdir(path):
    #         print(f"[SP world-model] {name}")
    #         model = _make_sp_model()
    #         try:
    #             convert_checkpoint(path, model, os.path.join(out_dir, name))
    #         except Exception as exc:
    #             print(f"  FAILED: {exc}")
    #         del model
    #         print()

    # ------------------------------------------------------------------
    # 5. matrix-game-init  (try MP first, fallback to SP)
    # ------------------------------------------------------------------
    # init_path = os.path.join(ckpt_dir, "matrix-game-init.pt")
    # if os.path.isdir(init_path):
    #     print("[matrix-game-init]")
    #     converted = False
    #     for label, factory in [("MP", _make_mp_model), ("SP", _make_sp_model)]:
    #         try:
    #             model = factory()
    #             convert_checkpoint(
    #                 init_path, model, os.path.join(out_dir, "matrix-game-init.pt")
    #             )
    #             converted = True
    #             del model
    #             break
    #         except Exception as exc:
    #             print(f"  {label} attempt failed: {exc}")
    #             del model
    #     if not converted:
    #         print("  Skipping matrix-game-init.pt (could not match model).")
    #     print()

    print("Done.")


if __name__ == "__main__":
    main()
