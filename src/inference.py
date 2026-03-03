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
import math
import os
import sys

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
    If offload=True, load to CPU and do not move to device (caller moves when needed).
    """
    def _is_state_dict(obj):
        if not isinstance(obj, dict):
            return False
        if len(obj) == 0:
            return False
        k = next(iter(obj))
        return isinstance(k, str) and ("." in k or k in ("state_dict",))

    load_device = torch.device("cpu") if offload else device

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

    clip_raw = torch.load(clip_checkpoint_path, map_location=load_device, weights_only=False)
    vae_raw = torch.load(vae_checkpoint_path, map_location=load_device, weights_only=False)
    world_raw = torch.load(model_weights_path, map_location=load_device, weights_only=False)

    clip_is_sd = _is_state_dict(clip_raw)
    vae_is_sd = _is_state_dict(vae_raw)
    world_is_sd = _is_state_dict(world_raw)

    if clip_is_sd:
        clip_state = _prepare_state(clip_raw.get("state_dict", clip_raw))
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
        clip_model.load_state_dict(clip_state, strict=False)
    else:
        if not isinstance(clip_raw, torch.nn.Module):
            raise ValueError(f"CLIP checkpoint is not an nn.Module or state dict: {type(clip_raw)}")
        clip_model = clip_raw

    if vae_is_sd:
        vae_state = _prepare_state(vae_raw.get("state_dict", vae_raw))
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

    if world_is_sd:
        world_state = _prepare_state(world_raw.get("state_dict", world_raw))
        world_model = SolarisMPModelTorch(
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
            action_config=dict(
                mouse_dim_in=2,
                keyboard_dim_in=6,
                hidden_size=128,
                img_hidden_size=1536,
                keyboard_hidden_dim=1024,
                mouse_hidden_dim=1024,
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
        world_model.load_state_dict(world_state, strict=False)
    else:
        if not isinstance(world_raw, torch.nn.Module):
            raise ValueError(f"World model checkpoint is not an nn.Module or state dict: {type(world_raw)}")
        world_model = world_raw

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
        final_frame_BPFHWC.cpu().numpy(),
        multiplayer_method,
        num_players=2,
    )
    final_frame_BPFHWC = torch.from_numpy(final_frame_BPFHWC).to(
        device=device, dtype=model_dtype
    )
    B = final_frame_BPFHWC.shape[0]

    if offload:
        with offload_to(device, vae_model):
            decoded = vae_model.decode(
                rearrange(final_frame_BPFHWC, "b p f h w c -> (b p) f h w c"),
                scale=scale_torch,
            )
    else:
        decoded = vae_model.decode(
            rearrange(final_frame_BPFHWC, "b p f h w c -> (b p) f h w c"),
            scale=scale_torch,
        )

    decoded = rearrange(decoded, "(b p) f h w c -> b p f h w c", b=B)
    decoded_uint8 = change_tensor_range_torch(decoded, [-1.0, 1.0], [0, 255])
    os.makedirs(eval_save_dir, exist_ok=True)
    rollout_np = decoded_uint8.cpu().numpy()
    gt_np = video_uint8.cpu().numpy()
    side_by_side = np.concatenate([gt_np, rollout_np], axis=4)
    side_by_side_BFHWC = rearrange(side_by_side, "b p f h w c -> b f (p h) w c")
    for i in range(side_by_side_BFHWC.shape[0]):
        write_video(
            os.path.join(eval_save_dir, f"video_{i}_side_by_side.mp4"),
            side_by_side_BFHWC[i],
            fps=20,
        )
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
