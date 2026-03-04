"""
Standalone PyTorch inference: load models with torch.load and run evaluation.
No hydra/config — run from repo root with: python -m src.inference [args]
Or from solaris/src: python inference.py [args]

Offloading (--offload): keep models on CPU and move to GPU only when needed, to reduce VRAM use.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import math
import os
import sys
from tqdm import tqdm
import numpy as np
import torch
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from torchvision.io import write_video

# Ensure parent is on path when run as script
if __name__ == "__main__" and "__file__" in dir():
    _src = os.path.dirname(os.path.abspath(__file__))
    if _src not in sys.path:
        sys.path.insert(0, _src)

from data.dataset import (
    DatasetMultiplayer,
    VideoReadError,
    collate_segments_to_batch,
)
from data.batch_sampler import EvalBatchSampler
from models.wan_vae import VAE_SCALE
from utils.multiplayer import handle_multiplayer_input, handle_multiplayer_output


def wan_image_condition_preprocess_torch(image_BPFHWC, target_height, target_width):
    """Crop, resize, normalize. image: (B, P, F, H, W, C)."""
    B, P, F, H, W, C = image_BPFHWC.shape
    device = image_BPFHWC.device
    dtype = image_BPFHWC.dtype
    if image_BPFHWC.dtype == torch.uint8:
        image_BPFHWC = image_BPFHWC.float() / 255.0
    src_aspect = H / W
    tgt_aspect = target_height / target_width
    if src_aspect > tgt_aspect:
        new_w = W
        new_h = int(new_w * tgt_aspect)
    else:
        new_h = H
        new_w = int(new_h / tgt_aspect)
    h_start = (H - new_h) // 2
    w_start = (W - new_w) // 2
    cropped = image_BPFHWC[:, :, :, h_start : h_start + new_h, w_start : w_start + new_w, :]
    cropped = cropped.permute(0, 1, 2, 5, 3, 4)
    resized = torch.nn.functional.interpolate(
        cropped.reshape(B * P * F, C, new_h, new_w),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.view(B, P, F, C, target_height, target_width)
    resized = resized.permute(0, 1, 2, 4, 5, 3)
    if resized.max() <= 1.0:
        resized = resized * 2.0 - 1.0
    return resized.to(dtype=dtype, device=device)


def change_tensor_range_torch(x, in_range, out_range, dtype=torch.uint8):
    low_in, high_in = in_range
    low_out, high_out = out_range
    x = (x - low_in) / (high_in - low_in)
    x = x * (high_out - low_out) + low_out
    return x.clamp(low_out, high_out).to(dtype)


def build_eval_dataloader(
    eval_data_dir,
    test_dataset_name,
    dataset_name,
    num_frames,
    batch_size=1,
    eval_num_samples=None,
    bot1_name="Alpha",
    bot2_name="Bravo",
    converters=None,
    obs_resize=(352, 640),
):
    """Build a single eval DataLoader without config/JAX. Uses data.dataset and data.batch_sampler."""
    if converters is None:
        converters = ["CameraLinearConverterMatrixGame2"]
    data_dir = os.path.join(eval_data_dir, test_dataset_name)
    dataset = DatasetMultiplayer(
        data_dir=data_dir,
        dataset_name=dataset_name,
        bot1_name=bot1_name,
        bot2_name=bot2_name,
        converters=converters,
        obs_resize=obs_resize,
        shuffle_bots=True,
    )
    sampler = EvalBatchSampler(
        dataset,
        rank=0,
        num_replicas=1,
        batch_size=batch_size,
        num_frames=num_frames,
        num_global_samples=eval_num_samples,
    )
    pad_batch_to = None
    if eval_num_samples is not None:
        pad_batch_to = math.ceil(eval_num_samples / batch_size)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=functools.partial(
            collate_segments_to_batch,
            num_frames,
            pad_batch_to,
        ),
    )
    return loader


def load_models(model_weights_path, clip_checkpoint_path, vae_checkpoint_path, device, offload=False):
    """Load CLIP, VAE, and world model from .pt files.
    Supports both full nn.Module (torch.save(model, path)) and state_dict (torch.save(model.state_dict(), path)).
    Checkpoints are always loaded to CPU first to avoid GPU OOM; models are moved to device only after loading.
    If offload=True, models stay on CPU and are moved to device only when needed by the caller.
    """
    def _is_state_dict(obj):
        if not isinstance(obj, dict):
            return False
        if len(obj) == 0:
            return False
        k = next(iter(obj))
        return isinstance(k, str) and ("." in k or k in ("state_dict",))

    # Load all checkpoints on CPU to avoid GPU OOM (never materialize full checkpoint on GPU).
    load_device = torch.device("cpu")

    from models.torch.clip_torch import CLIPModel as TorchCLIPModel
    from models.torch.wan_vae_torch import WanVAETorch
    from models.torch.world_model_mp_torch import SolarisMPModelTorch
    try:
        from models.torch.state_dict_utils import flax_state_dict_to_torch
    except Exception:
        flax_state_dict_to_torch = None

    def _needs_flax_conversion(state):
        """True if state dict uses Flax/NNX naming (.kernel, .scale, .embedding)."""
        if not isinstance(state, dict):
            return False
        for k in state:
            if isinstance(k, str) and (
                k.endswith(".kernel") or k.endswith(".scale") or k.endswith(".embedding")
            ):
                return True
        return False

    def _prepare_state(state):
        """Convert Flax-style keys/shapes to PyTorch only when needed."""
        if not isinstance(state, dict):
            return state
        if flax_state_dict_to_torch is not None and _needs_flax_conversion(state):
            return flax_state_dict_to_torch(state)
        return state

    def _clip_state_align_keys(state):
        """Remap checkpoint keys to match Torch CLIP: mlp.layers.0/2 -> mlp.0/2. Drop keys model doesn't have."""
        if not isinstance(state, dict):
            return state
        out = {}
        for k, v in state.items():
            new_k = k.replace(".mlp.layers.0.", ".mlp.0.").replace(".mlp.layers.2.", ".mlp.2.")
            out[new_k] = v
        return out

    def _vae_state_align_shapes(state, model):
        """Align .gamma/.bias shapes: checkpoint may have (C,1,1,1) or (C,1,1); match model param shape."""
        model_sd = model.state_dict()
        out = {}
        for k, v in state.items():
            if not isinstance(v, torch.Tensor):
                out[k] = v
                continue
            if k not in model_sd:
                out[k] = v
                continue
            target = model_sd[k].shape
            if v.shape == target:
                out[k] = v
                continue
            # Differ by one trailing dimension?
            if v.dim() == 4 and model_sd[k].dim() == 3 and v.shape[:3] == target:
                out[k] = v.squeeze(-1)
            elif v.dim() == 3 and model_sd[k].dim() == 4 and v.shape == target[:3] and target[3:] == (1,):
                out[k] = v.unsqueeze(-1)
            else:
                out[k] = v
        return out

    def _vae_state_align_conv_keys(state):
        """Remap VAE checkpoint keys: CausalConv3d uses .conv submodule, so X.weight -> X.conv.weight."""
        if not isinstance(state, dict):
            return state
        out = {}
        for k, v in state.items():
            if not isinstance(v, torch.Tensor):
                continue
            if ".conv." in k:
                out[k] = v
                continue
            if k.endswith(".weight"):
                prefix = k[:-7]
            elif k.endswith(".bias"):
                prefix = k[:-5]
            else:
                out[k] = v
                continue
            need_conv = (
                prefix == "conv1" or prefix == "conv2"
                or prefix.endswith(".conv1") or prefix.endswith(".conv2")
                or prefix.endswith(".residual.2") or prefix.endswith(".residual.6")
                or prefix.endswith(".shortcut") or prefix.endswith(".time_conv")
                or prefix.endswith(".head.2")
            )
            if need_conv:
                new_k = prefix + ".conv." + ("weight" if k.endswith(".weight") else "bias")
                out[new_k] = v
            else:
                out[k] = v
        return out

    def _world_model_mp_state_align(state, num_blocks=30):
        """Align Flax/JAX-style world model checkpoint keys to Torch SolarisMPModelTorch.
        Expects state already passed through _prepare_state (Flax .kernel->.weight etc).
        Converts .layers.N->.N, patch_embedding->patch_embedding.conv,
        img_emb.proj.N->img_emb.norm0/fc1/fc2/norm1, and expands blocks.XXX (no index)
        to blocks.0.XXX ... blocks.(num_blocks-1).XXX.
        If checkpoint has only blocks.0..blocks.K (e.g. K=3 from .layers.0..3), replicates
        to fill blocks.0..blocks.(num_blocks-1) by copying from the first available block."""
        if not isinstance(state, dict):
            return state
        out = {}
        for k, v in state.items():
            if not isinstance(k, str):
                out[k] = v
                continue
            # .layers.0 -> .0, .layers.1 -> .1, .layers.2 -> .2, .layers.3 -> .3
            key = k.replace(".layers.0.", ".0.").replace(".layers.1.", ".1.").replace(".layers.2.", ".2.").replace(".layers.3.", ".3.")
            # time_embedding: JAX and PyTorch both use (Linear, SiLU, Linear) so layers.0→0, layers.2→2; no remap needed.
            # img_emb.proj.0 -> norm0, proj.1 -> fc1, proj.2 -> fc2, proj.3 -> norm1
            if key.startswith("img_emb.proj."):
                key = key.replace("img_emb.proj.0.", "img_emb.norm0.").replace("img_emb.proj.1.", "img_emb.fc1.").replace("img_emb.proj.2.", "img_emb.fc2.").replace("img_emb.proj.3.", "img_emb.norm1.")
            # patch_embedding.weight/bias -> patch_embedding.conv.weight/bias
            if key == "patch_embedding.weight":
                key = "patch_embedding.conv.weight"
            elif key == "patch_embedding.bias":
                key = "patch_embedding.conv.bias"
            # head.0.* (norm) / head.1.* (linear) -> head.norm.* / head.head.* if export uses indices
            if key.startswith("head.0."):
                key = key.replace("head.0.", "head.norm.", 1)
            elif key.startswith("head.1."):
                key = key.replace("head.1.", "head.head.", 1)
            # blocks.XXX (no block index) -> blocks.0.XXX, blocks.1.XXX, ... blocks.29.XXX
            if key.startswith("blocks."):
                parts = key.split(".", 2)
                # blocks.modulation or blocks.norm1.weight etc. (exactly two segments)
                if len(parts) == 2:
                    suffix = parts[1]
                    v_tensor = v if isinstance(v, torch.Tensor) else None
                    if v_tensor is not None and v_tensor.dim() >= 1 and v_tensor.shape[0] == num_blocks:
                        for i in range(num_blocks):
                            out[f"blocks.{i}.{suffix}"] = v_tensor[i].clone()
                    else:
                        for i in range(num_blocks):
                            out[f"blocks.{i}.{suffix}"] = v
                    continue
                # blocks.norm1.weight, blocks.cross_attn.q.weight, blocks.ffn.0.bias etc. (non-digit middle part)
                if len(parts) >= 3 and not parts[1].isdigit():
                    suffix = ".".join(parts[1:])  # full path after blocks. e.g. ffn.0.bias, cross_attn.q.kernel
                    v_tensor = v if isinstance(v, torch.Tensor) else None
                    if v_tensor is not None and v_tensor.dim() >= 1 and v_tensor.shape[0] == num_blocks:
                        for i in range(num_blocks):
                            out[f"blocks.{i}.{suffix}"] = v_tensor[i].clone()
                    else:
                        for i in range(num_blocks):
                            out[f"blocks.{i}.{suffix}"] = v
                    continue
            out[key] = v
        # Replicate block params if checkpoint has only a subset of block indices (e.g. blocks.0..3 from .layers.0..3)
        block_suffix_to_idx = {}
        for key in list(out.keys()):
            if not key.startswith("blocks."):
                continue
            parts = key.split(".", 2)
            if len(parts) >= 3 and parts[1].isdigit():
                i, suffix = int(parts[1]), parts[2]
                block_suffix_to_idx.setdefault(suffix, set()).add(i)
        for suffix, indices in block_suffix_to_idx.items():
            missing = set(range(num_blocks)) - indices
            if not missing:
                continue
            source = min(indices)
            for j in missing:
                out[f"blocks.{j}.{suffix}"] = out[f"blocks.{source}.{suffix}"]
        return out

    def _world_model_mp_kernel_to_weight(state):
        """Convert any remaining Flax .kernel keys to PyTorch .weight (e.g. if state was nested and not converted by _prepare_state)."""
        if not isinstance(state, dict):
            return state
        out = {}
        for k, v in state.items():
            if not isinstance(k, str):
                out[k] = v
                continue
            if k.endswith(".kernel") and isinstance(v, torch.Tensor):
                new_k = k.replace(".kernel", ".weight")
                if v.dim() == 2:
                    out[new_k] = v.t().contiguous()
                elif v.dim() == 4:
                    out[new_k] = v.permute(3, 2, 0, 1).contiguous()
                elif v.dim() == 5:
                    out[new_k] = v.permute(4, 3, 0, 1, 2).contiguous()
                else:
                    out[new_k] = v
            else:
                out[k] = v
        return out

    def _world_model_mp_remap_block_keys(state):
        """Remap checkpoint keys to match PyTorch block structure.
        Checkpoint (Flax/NNX) block layer order: 0=norm1, 1=self_attn, 2=norm3, 3=cross_attn, 4=norm2, 5=ffn.0, 6=ffn.2.
        Also supports named keys: blocks.N.norm1.*, blocks.N.cross_attn.*, blocks.N.mouse_mlp.* -> action_model, etc."""
        if not isinstance(state, dict):
            return state
        action_model_parts = {
            "mouse_mlp", "t_qkv", "keyboard_embed", "img_attn_q_norm", "img_attn_k_norm",
            "proj_mouse", "key_attn_q_norm", "key_attn_k_norm", "mouse_attn_q",
            "keyboard_attn_kv", "proj_keyboard",
        }
        self_attn_parts = {"q", "k", "v", "o", "norm_q", "norm_k"}
        cross_attn_parts = {"q", "k", "v", "o", "norm_q", "norm_k"}
        # Alternate names some checkpoints use for cross-attention (remap to cross_attn)
        cross_attn_alternate_names = {"enc_attn", "context_attn", "i2v_cross_attn", "cross_attention"}
        # Flax/NNX block definition order: norm1, self_attn, norm3, cross_attn, norm2, ffn.0, ffn.2
        # Set LMW_WORLD_MODEL_OLD_BLOCK_ORDER=1 if your checkpoint uses 0=ffn.0, 1=self_attn, 2=ffn.2, 3=cross_attn, 4=norm2 (no norm1/norm3).
        if os.environ.get("LMW_WORLD_MODEL_OLD_BLOCK_ORDER"):
            layer_idx_to_module = {"0": "ffn.0", "1": "self_attn", "2": "ffn.2", "3": "cross_attn", "4": "norm2", "5": "ffn"}
        else:
            layer_idx_to_module = {
                "0": "norm1", "1": "self_attn", "2": "norm3", "3": "cross_attn", "4": "norm2",
                "5": "ffn.0", "6": "ffn.2",
            }
        out = {}
        for k, v in state.items():
            if not isinstance(k, str):
                out[k] = v
                continue
            if not k.startswith("blocks."):
                out[k] = v
                continue
            parts = k.split(".")
            if len(parts) < 3:
                out[k] = v
                continue
            try:
                block_idx = int(parts[1])
            except ValueError:
                out[k] = v
                continue
            part = parts[2]
            rest = ".".join(parts[3:]) if len(parts) > 3 else ""
            prefix = f"blocks.{block_idx}."
            if part in layer_idx_to_module:
                sub = layer_idx_to_module[part]
                # Checkpoint may use numeric indices 0/2 for FFN linears instead of norm1/norm3 (e.g. blocks.0.0.weight = ffn.0).
                # Use shape: LayerNorm = 1D with dim (48 or 1536); FFN = 2D or 1D with ffn_dim (280 or 8960).
                if part == "0" and isinstance(v, torch.Tensor):
                    if v.dim() == 2 or (v.dim() == 1 and v.shape[0] in (280, 8960)):
                        sub = "ffn.0"
                elif part == "2" and isinstance(v, torch.Tensor):
                    if v.dim() == 2:
                        sub = "ffn.2"
                    # else: 1D (48 or 1536) stays norm3
                new_k = f"{prefix}{sub}.{rest}" if rest else f"{prefix}{sub}"
            elif part in action_model_parts:
                new_k = f"{prefix}action_model.{part}.{rest}" if rest else f"{prefix}action_model.{part}"
            elif part in self_attn_parts:
                new_k = f"{prefix}self_attn.{part}.{rest}" if rest else f"{prefix}self_attn.{part}"
            elif part in cross_attn_parts:
                new_k = f"{prefix}cross_attn.{part}.{rest}" if rest else f"{prefix}cross_attn.{part}"
            elif part in cross_attn_alternate_names:
                # Checkpoint may name cross-attention enc_attn, context_attn, etc.
                new_k = f"{prefix}cross_attn.{rest}" if rest else f"{prefix}cross_attn"
            elif part in ("weight", "bias") and len(parts) == 3:
                new_k = f"{prefix}norm2.{part}"
            elif part == "norm1" and isinstance(v, torch.Tensor):
                # Checkpoint may name ffn.0 (Linear) as norm1: weight (8960,1536) or (1536,8960), bias (8960)
                if v.dim() == 2 or (v.dim() == 1 and v.shape[0] == 8960):
                    sub = "ffn.0"
                    new_k = f"{prefix}{sub}.{rest}" if rest else f"{prefix}{sub}"
                else:
                    new_k = k
            elif part == "norm3" and isinstance(v, torch.Tensor):
                # Checkpoint may name ffn.2 (Linear) as norm3: weight (1536,8960) or (8960,1536), bias (1536)
                if v.dim() == 2 or (v.dim() == 1 and v.shape[0] == 1536):
                    sub = "ffn.2"
                    new_k = f"{prefix}{sub}.{rest}" if rest else f"{prefix}{sub}"
                else:
                    new_k = k
            elif part in ("norm2", "cross_attn"):
                new_k = k
            else:
                new_k = k
            out[new_k] = v
        return out

    def _world_model_mp_unstack_block_params(state, num_blocks=30):
        """Split checkpoint params that are stacked per block (first dim = num_blocks) into one tensor per block.
        Checkpoint has e.g. blocks.0.action_model.mouse_mlp.0.weight with shape (30, in, out); model expects
        blocks.i.action_model.mouse_mlp.0.weight with shape (out, in) for each i. So we slice and transpose."""
        if not isinstance(state, dict):
            return state
        out = {}
        for k, v in state.items():
            if not isinstance(k, str) or not k.startswith("blocks."):
                out[k] = v
                continue
            if not isinstance(v, torch.Tensor):
                out[k] = v
                continue
            parts = k.split(".")
            if len(parts) < 3:
                out[k] = v
                continue
            try:
                _ = int(parts[1])
            except ValueError:
                out[k] = v
                continue
            suffix = ".".join(parts[2:])
            if v.dim() >= 1 and v.shape[0] == num_blocks:
                for i in range(num_blocks):
                    block_key = f"blocks.{i}.{suffix}"
                    vi = v[i]
                    if vi.dim() == 2:
                        # Linear weight: checkpoint (in, out), PyTorch wants (out, in)
                        out[block_key] = vi.t().contiguous()
                    else:
                        out[block_key] = vi.contiguous()
            else:
                out[k] = v
        return out

    def _world_model_mp_fill_missing_norms(state, num_blocks=30):
        """Fill only params that are still missing after remap/align (fallback so load can succeed).
        Logs when filling so you can confirm checkpoint has norm1/norm3/cross_attn/head if needed."""
        if not isinstance(state, dict):
            return state
        dim = None
        device = torch.device("cpu")
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                device = v.device
                if dim is None and k.startswith("blocks.0.self_attn.q.weight") and v.dim() == 2:
                    dim = v.shape[0]
                    break
        if dim is None:
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and "blocks." in k and v.dim() == 2:
                    dim = v.shape[0]
                    break
        if dim is None:
            return state
        filled = []
        # norm1, norm3 (LayerNorm)
        for i in range(num_blocks):
            for name, default in (("norm1.weight", torch.ones(dim)), ("norm1.bias", torch.zeros(dim)),
                                  ("norm3.weight", torch.ones(dim)), ("norm3.bias", torch.zeros(dim))):
                key = f"blocks.{i}.{name}"
                if key not in state:
                    state[key] = default.clone().to(device=device)
                    filled.append(key)
        # cross_attn (Linear q,k,v,o + RMSNorm norm_q, norm_k) — same layout as self_attn
        for i in range(num_blocks):
            for name in ("q", "k", "v", "o"):
                for wb in ("weight", "bias"):
                    key = f"blocks.{i}.cross_attn.{name}.{wb}"
                    if key not in state:
                        state[key] = (torch.eye(dim) if wb == "weight" else torch.zeros(dim)).clone().to(device=device)
                        filled.append(key)
            for name in ("norm_q.weight", "norm_k.weight"):
                key = f"blocks.{i}.cross_attn.{name}"
                if key not in state:
                    state[key] = torch.ones(dim).clone().to(device=device)
                    filled.append(key)
        # head.norm (LayerNorm)
        for name in ("head.norm.weight", "head.norm.bias"):
            if name not in state:
                state[name] = (torch.ones(dim) if "weight" in name else torch.zeros(dim)).clone().to(device=device)
                filled.append(name)
        if filled:
            print(f"[LMW] World model: filled {len(filled)} missing params (not in checkpoint): {filled[:20]}{'...' if len(filled) > 20 else ''}")
        return state

    # Load one model at a time and free checkpoint after load to reduce peak memory.
    clip_raw = torch.load(clip_checkpoint_path, map_location=load_device, weights_only=False)
    clip_is_sd = _is_state_dict(clip_raw)
    if clip_is_sd:
        clip_state = _prepare_state(clip_raw.get("state_dict", clip_raw))
        clip_state = _clip_state_align_keys(clip_state)
        clip_model = TorchCLIPModel(
            embed_dim=1024,
            image_size=224,
            patch_size=14,
            vision_dim=1280,
            vision_mlp_ratio=4,
            vision_heads=16,
            vision_layers=32,
            vision_pool="token",
            vision_pre_norm=True,
            vision_post_norm=False,
            activation="gelu",
            attn_dropout=0.0,
            proj_dropout=0.0,
            embedding_dropout=0.0,
            norm_eps=1e-5,
        )
        clip_model.load_state_dict(clip_state, strict=True)
        del clip_raw
    else:
        if not isinstance(clip_raw, torch.nn.Module):
            raise ValueError(f"CLIP checkpoint is not an nn.Module or state dict: {type(clip_raw)}")
        clip_model = clip_raw
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vae_raw = torch.load(vae_checkpoint_path, map_location=load_device, weights_only=True)
    vae_is_sd = _is_state_dict(vae_raw)
    if vae_is_sd:
        vae_state = _prepare_state(vae_raw.get("state_dict", vae_raw))
        vae_state = _vae_state_align_conv_keys(vae_state)
        vae_model = WanVAETorch(
            dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
        )
        vae_state = _vae_state_align_shapes(vae_state, vae_model)
        # Drop Flax-only keys (e.g. rngs.default.count) so strict load does not see unexpected keys
        model_keys = set(vae_model.state_dict().keys())
        vae_state = {k: v for k, v in vae_state.items() if k in model_keys}
        vae_model.load_state_dict(vae_state, strict=True)
        del vae_raw
    else:
        if not isinstance(vae_raw, torch.nn.Module):
            raise ValueError(f"VAE checkpoint is not an nn.Module or state dict: {type(vae_raw)}")
        vae_model = vae_raw
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # World model: same as above — checkpoint on CPU only.
    world_raw = torch.load(model_weights_path, map_location=load_device, weights_only=False)
    world_is_sd = _is_state_dict(world_raw)
    if world_is_sd:
        world_state = _prepare_state(world_raw.get("state_dict", world_raw))
        world_state = _world_model_mp_state_align(world_state, num_blocks=30)
        world_state = _world_model_mp_kernel_to_weight(world_state)
        world_state = _world_model_mp_remap_block_keys(world_state)
        world_state = _world_model_mp_unstack_block_params(world_state, num_blocks=30)
        world_state = _world_model_mp_fill_missing_norms(world_state, num_blocks=30)
        # World model config: must match checkpoint shapes for load_state_dict(strict=True).
        # Defaults are for the full model (dim=1536). For solaris.pt / 48-dim checkpoint set:
        #   LMW_WORLD_MODEL_DIM=48
        #   LMW_WORLD_MODEL_FFN_DIM=280   (from blocks.ffn.layers.0.bias (30,280); kernel has 8960 but bias defines out_dim)
        #   LMW_WORLD_MODEL_FREQ_DIM=8    (from time_embedding.layers.0.kernel (8, 1536))
        #   LMW_WORLD_MODEL_NUM_HEADS=12  (dim 48 / 12 = 4; or 6 for 8)
        #   LMW_ACTION_MOUSE_HIDDEN_DIM=32
        #   LMW_ACTION_KEYBOARD_HIDDEN_DIM=32
        #   LMW_ACTION_HIDDEN_SIZE=4       (keyboard_embed layers.0 out in checkpoint)
        _dim = int(os.environ.get("LMW_WORLD_MODEL_DIM", "1536"))
        _ffn_dim = int(os.environ.get("LMW_WORLD_MODEL_FFN_DIM", "8960"))
        _freq_dim = int(os.environ.get("LMW_WORLD_MODEL_FREQ_DIM", "256"))
        _num_heads = int(os.environ.get("LMW_WORLD_MODEL_NUM_HEADS", "12"))
        # Action module: optional overrides to match small checkpoints (e.g. mouse_hidden_dim=32, hidden_size=4).
        _action_hidden = int(os.environ.get("LMW_ACTION_HIDDEN_SIZE", "128"))
        _action_img = int(os.environ.get("LMW_ACTION_IMG_HIDDEN_SIZE", "1536"))  # CLIP/context dim, usually 1536
        _action_kb = int(os.environ.get("LMW_ACTION_KEYBOARD_HIDDEN_DIM", "1024"))
        _action_mouse = int(os.environ.get("LMW_ACTION_MOUSE_HIDDEN_DIM", "1024"))
        world_model = SolarisMPModelTorch(
            model_type="i2v",
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=36,
            dim=_dim,
            ffn_dim=_ffn_dim,
            freq_dim=_freq_dim,
            # text_dim=4096,
            out_dim=16,
            num_heads=_num_heads,
            num_layers=30,
            local_attn_size=6,
            sink_size=0,
            qk_norm=True,
            cross_attn_norm=False,
            action_config=dict(
                mouse_dim_in=2,
                keyboard_dim_in=23,  # match action_BPFD[:,:,:,:-2] (all but last 2 mouse dims)
                hidden_size=_action_hidden,
                img_hidden_size=_action_img,
                keyboard_hidden_dim=_action_kb,
                mouse_hidden_dim=_action_mouse,
                vae_time_compression_ratio=4,
                windows_size=3,
                heads_num=16,
                patch_size=[1, 2, 2],
                qk_norm=True,
                qkv_bias=False,
                rope_dim_list=[8, 28, 28],
                rope_theta=256,
                mouse_qk_dim_list=[8, 28, 28],
                enable_mouse=True,
                enable_keyboard=True,
                left_action_padding=11,
            ),
            inject_sample_info=False,
            eps=1e-6,
            multiplayer_method="multiplayer_attn",
            num_players=2,
        )
        # Optional: set LMW_DEBUG_LOAD=1 to log missing/unexpected keys (sanity check for garbage video)
        if os.environ.get("LMW_DEBUG_LOAD"):
            missing, unexpected = world_model.load_state_dict(world_state, strict=False)
            if missing:
                print(f"[LMW] World model missing_keys (first 30): {missing[:30]}")
            if unexpected:
                print(f"[LMW] World model unexpected_keys (first 30): {unexpected[:30]}")
        else:
            world_model.load_state_dict(world_state, strict=True)
        del world_raw
    else:
        if not isinstance(world_raw, torch.nn.Module):
            raise ValueError(f"World model checkpoint is not an nn.Module or state dict: {type(world_raw)}")
        world_model = world_raw
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if offload:
        for m in (clip_model, vae_model, world_model):
            m.eval()
        return clip_model, vae_model, world_model
    clip_model = clip_model.to(device).eval()
    vae_model = vae_model.to(device).eval()
    world_model = world_model.to(device).eval()
    return clip_model, vae_model, world_model


@contextlib.contextmanager
def offload_to(device, *models):
    """Temporarily move models to device; move back to CPU on exit (for memory offloading)."""
    if not models or device.type == "cpu":
        yield
        return
    try:
        for m in models:
            m.to(device)
        yield
    finally:
        for m in models:
            m.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_evaluate(
    clip_model,
    vae_model,
    world_model,
    eval_loader,
    device,
    eval_save_dir,
    left_action_padding=11,
    multiplayer_method="multiplayer_attn",
    num_denoising_steps=None,
    offload=False,
    model_dtype=torch.bfloat16,
):
    """Run one eval run: one batch from loader, rollout, decode, save videos and metrics.
    If offload=True, models are moved to device only when needed (CLIP+VAE for inputs, world_model for rollout, VAE for decode).
    """
    from utils.rollout_torch import (
        get_model_inputs_for_eval_torch,
        perform_multiplayer_rollout_torch,
    )

    loader_iter = iter(eval_loader)
    batch = next(loader_iter)
    if hasattr(batch, "to_dict"):
        batch = batch.to_dict()
    real_lengths = batch["real_lengths"]
    if isinstance(real_lengths, torch.Tensor):
        real_lengths = real_lengths.numpy()
    obs = batch["obs"]
    act = batch["act"]
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)
    if isinstance(act, np.ndarray):
        act = torch.from_numpy(act)
    video_BPFHWC = rearrange(obs.float(), "b f p h w c -> b p f h w c")
    action_BPFD = rearrange(act.float(), "b f p d -> b p f d")
    actions_mouse_BPFD = action_BPFD[:, :, :, -2:]
    actions_keyboard_BPFD = action_BPFD[:, :, :, :-2]
    video_BPFHWC = wan_image_condition_preprocess_torch(video_BPFHWC, 352, 640)
    video_BPFHWC = video_BPFHWC.to(device)
    processed = wan_image_condition_preprocess_torch(video_BPFHWC, 352, 640)
    video_uint8 = change_tensor_range_torch(processed, [-1.0, 1.0], [0, 255])
    first_frame_BPHWC = processed[:, :, 0, :, :, :]
    n_frames_decode = video_uint8.shape[2]
    first_frames_BPFHWC = repeat(
        first_frame_BPHWC, "b p h w c -> b p f h w c", f=n_frames_decode
    ).to(device=device, dtype=model_dtype)
    actions_mouse_BPFD = actions_mouse_BPFD.to(device=device, dtype=model_dtype)
    actions_keyboard_BPFD = actions_keyboard_BPFD.to(device=device, dtype=model_dtype)
    scale_torch = [
        torch.from_numpy(np.asarray(s).copy()).to(device=device, dtype=model_dtype)
        for s in VAE_SCALE
    ]

    if offload:
        with offload_to(device, clip_model, vae_model):
            (
                cond_concat_BPFHWC,
                visual_context_BPFD,
                mouse_BPTD,
                keyboard_BPTD,
            ) = get_model_inputs_for_eval_torch(
                first_frames_BPFHWC,
                clip_model,
                vae_model,
                actions_mouse_BPFD,
                actions_keyboard_BPFD,
                multiplayer_method,
                device,
                vae_scale=scale_torch,
            )
    else:
        (
            cond_concat_BPFHWC,
            visual_context_BPFD,
            mouse_BPTD,
            keyboard_BPTD,
        ) = get_model_inputs_for_eval_torch(
            first_frames_BPFHWC,
            clip_model,
            vae_model,
            actions_mouse_BPFD,
            actions_keyboard_BPFD,
            multiplayer_method,
            device,
            vae_scale=scale_torch,
        )

    if offload:
        with offload_to(device, world_model):
            final_frame_BPFHWC = perform_multiplayer_rollout_torch(
                world_model,
                vae_model,
                clip_model,
                cond_concat_BPFHWC,
                visual_context_BPFD,
                mouse_BPTD,
                keyboard_BPTD,
                left_action_padding,
                multiplayer_method,
                device,
                num_denoising_steps=num_denoising_steps,
            )
    else:
        final_frame_BPFHWC = perform_multiplayer_rollout_torch(
            world_model,
            vae_model,
            clip_model,
            cond_concat_BPFHWC,
            visual_context_BPFD,
            mouse_BPTD,
            keyboard_BPTD,
            left_action_padding,
            multiplayer_method,
            device,
            num_denoising_steps=num_denoising_steps,
        )

    final_frame_BPFHWC = handle_multiplayer_output(
        final_frame_BPFHWC.cpu().float().numpy(),
        multiplayer_method,
        num_players=2,
    )
    final_frame_BPFHWC = torch.from_numpy(final_frame_BPFHWC).to(
        device=device, dtype=model_dtype
    )

    del clip_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    B = final_frame_BPFHWC.shape[0]
    P = final_frame_BPFHWC.shape[1]
    F_latent = final_frame_BPFHWC.shape[2]
    CHUNK_FRAMES = 4

    decoded_list = []
    for b in tqdm(range(B)):
        for p in range(P):
            latent_bpf = final_frame_BPFHWC[b, p]  # (F, h, w, c)
            chunks_out = []
            for start in range(0, F_latent, CHUNK_FRAMES):
                end = min(start + CHUNK_FRAMES, F_latent)
                chunk = latent_bpf[start:end].unsqueeze(0)  # (1, chunk_len, h, w, c)
                chunk = chunk.to(device=device, dtype=final_frame_BPFHWC.dtype)
                if offload:
                    with offload_to(device, vae_model):
                        out_chunk = vae_model.decode(chunk, scale=scale_torch)
                else:
                    out_chunk = vae_model.decode(chunk, scale=scale_torch)
                chunks_out.append(out_chunk.cpu())
            decoded_bpf = torch.cat(chunks_out, dim=1)
            decoded_list.append(decoded_bpf)
    decoded = torch.cat(decoded_list, dim=0)
    decoded = rearrange(decoded, "(b p) f h w c -> b p f h w c", b=B)
    decoded_uint8 = change_tensor_range_torch(decoded, [-1.0, 1.0], [0, 255])
    os.makedirs(eval_save_dir, exist_ok=True)
    rollout_np = decoded_uint8.cpu().numpy()
    # Save final decoded video only (no concat with gt to avoid shape mismatch)
    rollout_BFHWC = rearrange(rollout_np, "b p f h w c -> b f (p h) w c")
    for i in range(rollout_BFHWC.shape[0]):
        write_video(
            os.path.join(eval_save_dir, f"video_{i}_rollout.mp4"),
            rollout_BFHWC[i],
            fps=20,
        )
    gt_np = video_uint8.cpu().numpy()
    from metrics.compute_metrics import FIDCalculator, calculate_metrics_from_batch
    metrics = ["fid"]
    n_prompt_frames = 1
    fid_calculator = FIDCalculator(num_sources=2)
    pred_frames = rearrange(decoded_uint8.cpu().numpy(), "b p f h w c -> b f p h w c")[
        :, n_prompt_frames:
    ]
    gt_frames = rearrange(video_uint8.cpu().numpy(), "b p f h w c -> b f p h w c")[
        :, n_prompt_frames:
    ]
    calculate_metrics_from_batch(
        preds=pred_frames,
        targets=gt_frames,
        metrics_to_cal=metrics,
        real_lengths=real_lengths - n_prompt_frames,
        fid_calculator=fid_calculator,
    )
    fid_curve = np.array(fid_calculator.get_fid_curve_jax())
    print("FID curve:", fid_curve)
    return fid_curve


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="Load PyTorch checkpoints and run evaluation (no config/hydra)."
    )
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default="checkpoints_torch/solaris.pt",
        help="Path to world model .pt",
    )
    parser.add_argument(
        "--clip_checkpoint_path",
        type=str,
        default="checkpoints_torch/clip.pt",
        help="Path to CLIP .pt",
    )
    parser.add_argument(
        "--vae_checkpoint_path",
        type=str,
        default="checkpoints_torch/vae.pt",
        help="Path to VAE .pt",
    )
    parser.add_argument(
        "--eval_save_dir",
        type=str,
        default="eval_out",
        help="Directory to save eval videos and outputs",
    )
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default=None,
        help="Root dir for eval data (e.g. path containing eval/structureEval/test). If not set, models are loaded only.",
    )
    parser.add_argument(
        "--test_dataset_name",
        type=str,
        default="eval/structureEval/test",
        help="Subdir under eval_data_dir for the test set",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="eval_structure",
        help="Dataset name (used for eval_ids file)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=257,
        help="Number of frames per segment",
    )
    parser.add_argument(
        "--eval_num_samples",
        type=int,
        default=1,
        help="Max number of eval samples",
    )
    parser.add_argument(
        "--left_action_padding",
        type=int,
        default=11,
        help="Left action padding for rollout",
    )
    parser.add_argument(
        "--multiplayer_method",
        type=str,
        default="multiplayer_attn",
        choices=["multiplayer_attn", "concat_c"],
    )
    parser.add_argument(
        "--num_denoising_steps",
        type=int,
        default=None,
        help="Denoising steps (default 10)",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Keep models on CPU and move to GPU only when needed (reduces VRAM, slower).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("Loading models...")
    clip_model, vae_model, world_model = load_models(
        args.model_weights_path,
        args.clip_checkpoint_path,
        args.vae_checkpoint_path,
        device,
        offload=args.offload,
    )
    clip_model = clip_model.to(dtype=torch.bfloat16)
    vae_model = vae_model.to(dtype=torch.bfloat16)
    world_model = world_model.to(dtype=torch.bfloat16)

    print("Models loaded." + (" (CPU offload enabled)" if args.offload else ""))

    if args.eval_data_dir is None:
        print("No --eval_data_dir provided. Exiting. Run with --eval_data_dir <path> to evaluate.")
        return

    print("Building eval dataloader...")
    eval_loader = build_eval_dataloader(
        eval_data_dir=args.eval_data_dir,
        test_dataset_name=args.test_dataset_name,
        dataset_name=args.dataset_name,
        num_frames=args.num_frames,
        batch_size=1,
        eval_num_samples=args.eval_num_samples,
    )
    print("Running evaluation...")
    run_evaluate(
        clip_model,
        vae_model,
        world_model,
        eval_loader,
        device,
        args.eval_save_dir,
        left_action_padding=args.left_action_padding,
        multiplayer_method=args.multiplayer_method,
        num_denoising_steps=args.num_denoising_steps or 10,
        offload=args.offload,
        model_dtype=torch.bfloat16,
    )
    print("Done. Outputs saved to", args.eval_save_dir)


if __name__ == "__main__":
    main()
