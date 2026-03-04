"""
Test chunked VAE decoding as used in inference.py.

Loads only the VAE, builds a dummy latent tensor, runs the same chunked
decode loop (chunk frames -> decode each chunk -> concat), and checks shapes.

Run from repo root:
  python -m src.tests.test_vae_decode_chunks [--vae_checkpoint_path PATH]
  pytest src/tests/test_vae_decode_chunks.py -v
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

import torch
from einops import rearrange

# Repo root / src on path
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
if _src not in sys.path:
    sys.path.insert(0, _src)

from models.wan_vae import VAE_SCALE
from models.torch.wan_vae_torch import WanVAETorch


def _is_state_dict(obj):
    return isinstance(obj, dict) and (not hasattr(obj, "keys") or any(
        k in obj for k in ("state_dict", "params", "model")
    ))


def _prepare_state(raw):
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    return raw


def load_vae_only(vae_checkpoint_path, device="cpu"):
    """Load only the VAE from a .pt checkpoint. Returns (vae_model, scale_torch)."""
    load_device = torch.device("cpu")
    vae_raw = torch.load(vae_checkpoint_path, map_location=load_device, weights_only=False)
    vae_is_sd = _is_state_dict(vae_raw)
    if vae_is_sd:
        vae_state = _prepare_state(vae_raw.get("state_dict", vae_raw))

        def _vae_state_align_shapes(state, model):
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
                if v.dim() == 4 and model_sd[k].dim() == 3 and v.shape[:3] == target:
                    out[k] = v.squeeze(-1)
                elif v.dim() == 3 and model_sd[k].dim() == 4 and v.shape == target[:3] and target[3:] == (1,):
                    out[k] = v.unsqueeze(-1)
                else:
                    out[k] = v
            return out

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
        vae_model.load_state_dict(vae_state, strict=False)
    else:
        if not isinstance(vae_raw, torch.nn.Module):
            raise ValueError(f"VAE checkpoint is not an nn.Module or state dict: {type(vae_raw)}")
        vae_model = vae_raw
    del vae_raw
    gc.collect()
    vae_model = vae_model.to(device).eval()
    scale_torch = [
        torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s
        for s in VAE_SCALE
    ]
    return vae_model, scale_torch


def decode_chunks_same_as_inference(vae_model, final_frame_BPFHWC, scale_torch, device, offload=False):
    """
    Same chunked decode logic as inference.py (lines ~469–491):
    chunk latent frames, decode each chunk with vae_model.decode(), concat along time.
    """
    B = final_frame_BPFHWC.shape[0]
    P = final_frame_BPFHWC.shape[1]
    F_latent = final_frame_BPFHWC.shape[2]
    CHUNK_FRAMES = 4

    decoded_list = []
    for b in range(B):
        for p in range(P):
            latent_bpf = final_frame_BPFHWC[b, p]  # (F, h, w, c)
            chunks_out = []
            for start in range(0, F_latent, CHUNK_FRAMES):
                end = min(start + CHUNK_FRAMES, F_latent)
                chunk = latent_bpf[start:end].unsqueeze(0)  # (1, chunk_len, h, w, c)
                chunk = chunk.to(device=device, dtype=final_frame_BPFHWC.dtype)
                out_chunk = vae_model.decode(chunk, scale=scale_torch)
                chunks_out.append(out_chunk.cpu())
            decoded_bpf = torch.cat(chunks_out, dim=1)
            decoded_list.append(decoded_bpf)
    decoded = torch.cat(decoded_list, dim=0)
    decoded = rearrange(decoded, "(b p) f h w c -> b p f h w c", b=B)
    return decoded


def test_decode_chunks(vae_checkpoint_path=None, device="cpu", B=1, P=2, F_latent=8):
    """
    Load VAE (from checkpoint or random init), create dummy latent, run chunked decode, check shapes.
    When run via pytest, no checkpoint is required (uses random-init VAE).
    """
    # Latent spatial size: 352/8=44, 640/8=80 typical for this VAE
    H_latent, W_latent, C_latent = 44, 80, 16

    if vae_checkpoint_path and os.path.isfile(vae_checkpoint_path):
        vae_model, scale_torch = load_vae_only(vae_checkpoint_path, device=device)
    else:
        vae_model = WanVAETorch(
            dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
        )
        vae_model = vae_model.to(device).eval()
        scale_torch = [
            torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s
            for s in VAE_SCALE
        ]

    # Dummy latent: (B, P, F_latent, h, w, c), random values
    final_frame_BPFHWC = torch.randn(
        B, P, F_latent, H_latent, W_latent, C_latent,
        device=device, dtype=torch.bfloat16,
    )

    decoded = decode_chunks_same_as_inference(
        vae_model, final_frame_BPFHWC, scale_torch, device=device, offload=False
    )

    # Expected: each chunk of 4 latents -> (4-1)*4+1 = 13 decoded frames; 2 chunks -> 26
    CHUNK_FRAMES = 4
    num_chunks = (F_latent + CHUNK_FRAMES - 1) // CHUNK_FRAMES
    expected_frames_per_chunk = (CHUNK_FRAMES - 1) * 4 + 1
    last_chunk_len = F_latent - (num_chunks - 1) * CHUNK_FRAMES
    expected_last = (last_chunk_len - 1) * 4 + 1
    expected_F_dec = (num_chunks - 1) * expected_frames_per_chunk + expected_last

    assert decoded.shape[0] == B
    assert decoded.shape[1] == P
    assert decoded.shape[2] == expected_F_dec, (
        f"decoded time dim: expected {expected_F_dec}, got {decoded.shape[2]}"
    )
    # Decoder outputs 352x640 for 44x80 latent at 8x upsample
    assert decoded.shape[3] == 352
    assert decoded.shape[4] == 640
    assert decoded.shape[5] == 3

    return decoded


def main():
    parser = argparse.ArgumentParser(description="Test chunked VAE decoding")
    parser.add_argument(
        "--vae_checkpoint_path",
        type=str,
        default="checkpoints_torch/vae.pt",
        help="Path to VAE .pt (optional; if missing, uses random init)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Testing chunked VAE decode (same logic as inference.py)...")
    decoded = test_decode_chunks(
        vae_checkpoint_path=args.vae_checkpoint_path,
        device=args.device,
        B=1,
        P=2,
        F_latent=8,
    )
    print(f"decoded shape: {decoded.shape}")
    print("All good.")


if __name__ == "__main__":
    main()


def test_vae_decode_chunks_no_checkpoint():
    """Chunked decode with random-init VAE (no checkpoint)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_decode_chunks(vae_checkpoint_path=None, device=device)
