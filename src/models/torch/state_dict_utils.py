"""Convert Flax/NNX checkpoint state dict to PyTorch state dict format.

The conversion script saves parameters with Flax naming:
  - Linear: .kernel (in_features, out_features), .bias
  - Conv: .kernel (D, H, W, in_c, out_c), .bias
  - LayerNorm: .scale, .bias (Flax nnx.LayerNorm uses scale/bias)
  - Embedding: .embedding (num_embeddings, features)
  - RMSNorm / custom: .weight

PyTorch expects:
  - nn.Linear: .weight (out_features, in_features), .bias
  - nn.Conv3d: .weight (out_c, in_c, D, H, W), .bias
  - nn.LayerNorm: .weight (scale), .bias
  - nn.Embedding: .weight (num_embeddings, features)
"""

from collections import OrderedDict

import torch


def _is_linear_kernel(key: str, tensor: torch.Tensor) -> bool:
    if not key.endswith(".kernel"):
        return False
    return tensor.dim() == 2


def _is_conv2d_kernel(key: str, tensor: torch.Tensor) -> bool:
    if not key.endswith(".kernel"):
        return False
    return tensor.dim() == 4


def _is_conv_kernel(key: str, tensor: torch.Tensor) -> bool:
    if not key.endswith(".kernel"):
        return False
    return tensor.dim() == 5


def _is_embedding(key: str, tensor: torch.Tensor) -> bool:
    if not key.endswith(".embedding"):
        return False
    return tensor.dim() == 2


def flax_state_dict_to_torch(flat_state_dict):
    """Convert flat Flax/NNX state dict to PyTorch state dict."""
    out = OrderedDict()
    for key, tensor in flat_state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        tensor = tensor.clone()

        if _is_linear_kernel(key, tensor):
            new_key = key.replace(".kernel", ".weight")
            out[new_key] = tensor.t().contiguous()
        elif _is_conv2d_kernel(key, tensor):
            new_key = key.replace(".kernel", ".weight")
            out[new_key] = tensor.permute(3, 2, 0, 1).contiguous()
        elif _is_conv_kernel(key, tensor):
            new_key = key.replace(".kernel", ".weight")
            out[new_key] = tensor.permute(4, 3, 0, 1, 2).contiguous()
        elif key.endswith(".bias"):
            out[key] = tensor
        elif key.endswith(".scale"):
            new_key = key.replace(".scale", ".weight")
            out[new_key] = tensor
        elif _is_embedding(key, tensor):
            new_key = key.replace(".embedding", ".weight")
            out[new_key] = tensor
        else:
            out[key] = tensor
    return out


def load_flax_ckpt_for_torch(model, ckpt_path, strict=False, map_location=None):
    """Load a converted (Flax-style) .pt checkpoint into a PyTorch model."""
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint must be a state dict")
    torch_sd = flax_state_dict_to_torch(raw)
    missing, unexpected = model.load_state_dict(torch_sd, strict=strict)
    return missing, unexpected
