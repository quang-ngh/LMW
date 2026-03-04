#!/usr/bin/env python
"""Load an Orbax checkpoint and print all parameter names and shapes (no JAX model).

Uses only Orbax + the checkpoint's _METADATA to build a skeleton PyTree, restore
the checkpoint into it, then flatten and print keys/shapes. No SolarisMPModel or
any other model class is created.

Usage:
    python -m src.print_ckpt_params checkpoints/solaris.pt
    python -m src.print_ckpt_params checkpoints
    python -m src.print_ckpt_params checkpoints/solaris.pt -o checkpoint_params.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Single-device JAX
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax.numpy as jnp
import orbax.checkpoint as ocp


def _nested_set(d, path, value):
    """Set d[p0][p1]...[pN] = value for path = (p0, p1, ..., pN)."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def _build_skeleton_from_metadata(metadata_path):
    """Build a nested-dict skeleton with jnp.zeros(shape) at each leaf from _METADATA."""
    with open(metadata_path) as f:
        meta = json.load(f)
    tree_metadata = meta.get("tree_metadata") or {}
    skeleton = {}
    for key_str, val_meta in tree_metadata.items():
        try:
            key_tuple = eval(key_str)  # e.g. "('blocks', 'cross_attn', 'q', 'kernel', 'value')"
        except Exception:
            continue
        if not isinstance(key_tuple, (list, tuple)) or len(key_tuple) < 2:
            continue
        # path is all but last (usually 'value'); shape from value_metadata
        path = tuple(key_tuple[:-1])
        vm = val_meta.get("value_metadata") or {}
        shape = vm.get("write_shape")
        if not shape:
            continue
        leaf = jnp.zeros(shape, dtype=jnp.float32)
        _nested_set(skeleton, path, leaf)
    return skeleton


def _flatten_tree(tree, prefix=""):
    """Flatten nested dict to dotted keys -> array."""
    out = {}
    for k, v in tree.items():
        key = f"{prefix}.{k}" if prefix else k
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            out[key] = v
        elif isinstance(v, dict):
            out.update(_flatten_tree(v, key))
        else:
            out[key] = v
    return out


def resolve_checkpoint_dir(path):
    """If path is a parent dir, try to find solaris.pt or similar inside."""
    if not os.path.isdir(path):
        return path
    for name in ("solaris.pt", "matrix-game-init.pt"):
        sub = os.path.join(path, name)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "_METADATA")):
            return sub
    for name in sorted(os.listdir(path)):
        sub = os.path.join(path, name)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "_METADATA")):
            return sub
    return path


def main():
    parser = argparse.ArgumentParser(description="Load Orbax checkpoint and print parameter names/shapes (no model)")
    parser.add_argument("checkpoint_dir", type=str, help="Path to Orbax checkpoint directory (e.g. checkpoints/solaris.pt)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Write parameter list to this txt file")
    args = parser.parse_args()

    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    ckpt_dir = resolve_checkpoint_dir(ckpt_dir)

    metadata_path = os.path.join(ckpt_dir, "_METADATA")
    if not os.path.isfile(metadata_path):
        print(f"_METADATA not found at {metadata_path}")
        print("Pass an Orbax checkpoint directory that contains _METADATA.")
        sys.exit(1)

    print(f"Reading metadata: {metadata_path}")
    with open(metadata_path) as f:
        meta = json.load(f)
    tree_metadata = meta.get("tree_metadata") or {}
    if not tree_metadata:
        print("No tree_metadata found in _METADATA.")
        sys.exit(1)

    # Try restore into skeleton; if it fails, print from metadata only
    skeleton = _build_skeleton_from_metadata(metadata_path)
    restored = None
    try:
        print(f"Restoring checkpoint: {ckpt_dir}")
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(ckpt_dir, skeleton)
    except Exception as e:
        print(f"Restore failed ({e}); printing parameter names/shapes from _METADATA only.")
        restored = None

    lines = []
    if restored is not None:
        flat = _flatten_tree(restored)
        keys_sorted = sorted(flat.keys())
        lines.append("")
        lines.append("=" * 80)
        lines.append("CHECKPOINT PARAMETERS (from Orbax restore)")
        lines.append("=" * 80)
        lines.append("Format: key -> shape, dtype")
        lines.append("")
        for key in keys_sorted:
            arr = flat[key]
            if hasattr(arr, "shape") and hasattr(arr, "dtype"):
                lines.append(f"  {key}")
                lines.append(f"    -> shape {tuple(arr.shape)}  dtype {arr.dtype}")
            else:
                lines.append(f"  {key}  -> {type(arr)}")
    else:
        # Metadata-only: key tuple (without 'value') -> write_shape
        keys_shapes = []
        for key_str, val_meta in tree_metadata.items():
            try:
                key_tuple = eval(key_str)
            except Exception:
                continue
            if not isinstance(key_tuple, (list, tuple)) or len(key_tuple) < 2:
                continue
            path = ".".join(str(k) for k in key_tuple[:-1])
            vm = val_meta.get("value_metadata") or {}
            shape = vm.get("write_shape", ())
            keys_shapes.append((path, tuple(shape)))
        keys_shapes.sort(key=lambda x: x[0])
        keys_sorted = [p for p, _ in keys_shapes]
        lines.append("")
        lines.append("=" * 80)
        lines.append("CHECKPOINT PARAMETERS (from _METADATA only)")
        lines.append("=" * 80)
        lines.append("Format: key -> write_shape")
        lines.append("")
        for path, shape in keys_shapes:
            lines.append(f"  {path}")
            lines.append(f"    -> shape {shape}")

    lines.append("")
    lines.append(f"Total parameters: {len(keys_sorted)}")

    # Print to stdout
    for line in lines:
        print(line)

    # Write to file if requested
    if args.output:
        out_path = os.path.abspath(args.output)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\nWrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
