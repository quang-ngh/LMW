"""PyTorch inference runner: loads converted .pt checkpoints and runs eval (no JAX)."""

import os
from collections import defaultdict

import numpy as np
import torch
from absl import logging
from einops import rearrange, repeat
from torchvision.io import write_video

from data.dataset import VideoReadError
from metrics.compute_metrics import FIDCalculator, calculate_metrics_from_batch
from models.utils import jax_to_torch
from models.wan_vae import VAE_SCALE
from utils.config import get_obj_from_str, instantiate_from_config
from utils.multiplayer import handle_multiplayer_input, handle_multiplayer_output


def _build_eval_dataloaders_torch(
    eval_datasets,
    eval_data_dir,
    obs_resolution,
    converters,
    eval_dataloader_config,
    allow_additional_params=True,
):
    """Build eval dataloaders without JAX sharding (single process)."""
    eval_dataloaders = {}
    for eval_dataset_config in eval_datasets.values():
        eval_dataset_cls = get_obj_from_str(eval_dataset_config["class"])
        eval_dataset_name = eval_dataset_config["name"]
        dataset_kwargs = {
            "data_dir": os.path.join(
                eval_data_dir, eval_dataset_config["test_dataset_name"]
            ),
            "dataset_name": eval_dataset_name,
            "obs_resize": obs_resolution,
            "converters": converters,
        }
        if allow_additional_params:
            dataset_kwargs.update(eval_dataset_config.get("additional_params", {}))

        eval_dataset = eval_dataset_cls(**dataset_kwargs)
        eval_dataloader, eval_local_num_batches = instantiate_from_config(
            eval_dataloader_config, dataset=eval_dataset
        )
        eval_dataloaders[eval_dataset_name] = {
            "dataloader": eval_dataloader,
            "local_num_batches": eval_local_num_batches,
        }
    return eval_dataloaders


def wan_image_condition_preprocess_torch(image_BPFHWC, target_height, target_width):
    """Torch version of wan_image_condition_preprocess (crop, resize, normalize)."""
    # image: (B, P, F, H, W, C)
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
    cropped = cropped.permute(0, 1, 2, 5, 3, 4)  # B P F C H W
    resized = torch.nn.functional.interpolate(
        cropped.reshape(B * P * F, C, new_h, new_w),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.view(B, P, F, C, target_height, target_width)
    resized = resized.permute(0, 1, 2, 4, 5, 3)  # B P F H W C
    if resized.max() <= 1.0:
        resized = resized * 2.0 - 1.0
    return resized.to(dtype=dtype, device=device)


def change_tensor_range_torch(x, in_range, out_range, dtype=torch.uint8):
    """Torch version of change_tensor_range."""
    low_in, high_in = in_range
    low_out, high_out = out_range
    x = (x - low_in) / (high_in - low_in)
    x = x * (high_out - low_out) + low_out
    return x.clamp(low_out, high_out).to(dtype)


class InferenceTorch:
    """Inference runner that loads converted PyTorch checkpoints and runs eval (no JAX)."""

    def __init__(
        self,
        model_weights_path,
        clip_checkpoint_path,
        vae_checkpoint_path,
        eval_save_dir,
        eval_dataloader_config,
        eval_datasets,
        eval_data_dir,
        obs_resolution,
        converters,
        experiment_name,
        left_action_padding=11,
        multiplayer_method="multiplayer_attn",
        network_config=None,
        base_seed=4,
        eval_num_samples=None,
        sharding_config=None,
        **kwargs,
    ):
        self.model_weights_path = model_weights_path
        self.clip_checkpoint_path = clip_checkpoint_path
        self.vae_checkpoint_path = vae_checkpoint_path
        self.eval_save_dir = eval_save_dir
        self.experiment_name = experiment_name
        self.left_action_padding = left_action_padding
        self.multiplayer_method = multiplayer_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("InferenceTorch device: %s", self.device)

        self.eval_dataloaders = _build_eval_dataloaders_torch(
            eval_datasets=eval_datasets,
            eval_data_dir=eval_data_dir,
            obs_resolution=obs_resolution,
            converters=converters,
            eval_dataloader_config=eval_dataloader_config,
            allow_additional_params=True,
        )

        self.clip_model, self.vae_model, self.world_model = self._load_models()

    def _load_models(self):
        """Load CLIP, VAE, and world model directly from .pt files (torch.load)."""
        # torch.load with weights_only=False to allow loading full model objects
        clip_model = torch.load(
            self.clip_checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(clip_model, torch.nn.Module):
            raise ValueError(
                "clip_checkpoint_path must point to a saved nn.Module (e.g. torch.save(model, path)). "
                "Got state dict or other object."
            )
        clip_model = clip_model.to(self.device)
        clip_model.eval()

        vae_model = torch.load(
            self.vae_checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(vae_model, torch.nn.Module):
            raise ValueError(
                "vae_checkpoint_path must point to a saved nn.Module. Got state dict or other object."
            )
        vae_model = vae_model.to(self.device)
        vae_model.eval()

        world_model = torch.load(
            self.model_weights_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(world_model, torch.nn.Module):
            raise ValueError(
                "model_weights_path must point to a saved nn.Module. Got state dict or other object."
            )
        world_model = world_model.to(self.device)
        world_model.eval()

        # Override from loaded model if present
        if hasattr(world_model, "left_action_padding"):
            self.left_action_padding = world_model.left_action_padding
        if hasattr(world_model, "multiplayer_method"):
            self.multiplayer_method = world_model.multiplayer_method

        return clip_model, vae_model, world_model

    def _robust_batch_sample(self, loader_iter, num_retries=5):
        for _ in range(num_retries):
            try:
                return next(loader_iter)
            except VideoReadError:
                logging.info("Retrying to load batch")
        raise RuntimeError("Failed to load batch")

    def _get_curr_batch_torch(self, loader_iter):
        batch = self._robust_batch_sample(loader_iter)
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
        video_BPFHWC_unprocessed = video_BPFHWC
        video_BPFHWC = wan_image_condition_preprocess_torch(video_BPFHWC, 352, 640)
        return (
            video_BPFHWC,
            video_BPFHWC_unprocessed,
            actions_mouse_BPFD,
            actions_keyboard_BPFD,
            real_lengths,
        )

    def run(self):
        self.run_evals()

    def run_evals(self):
        for eval_dataset_name, eval_dataloader_info in self.eval_dataloaders.items():
            logging.info("Running eval on %s", eval_dataset_name)
            self.run_eval(
                eval_dataloader_info=eval_dataloader_info,
                eval_dir_name=eval_dataset_name,
            )

    def run_eval(self, eval_dataloader_info, eval_dir_name, num_denoising_steps=None):
        (
            video_BPFHWC,
            video_unprocessed,
            actions_mouse,
            actions_keyboard,
            real_lengths,
        ) = self._get_curr_batch_torch(iter(eval_dataloader_info["dataloader"]))

        evaluation_output_directory = os.path.join(self.eval_save_dir, eval_dir_name)
        os.makedirs(evaluation_output_directory, exist_ok=True)

        metric_curve = self._evaluate_torch(
            video_BPFHWC=video_BPFHWC,
            video_unprocessed=video_unprocessed,
            actions_mouse=actions_mouse,
            actions_keyboard=actions_keyboard,
            real_lengths=real_lengths,
            eval_dir=evaluation_output_directory,
            num_denoising_steps=num_denoising_steps,
        )
        for k, v in metric_curve.items():
            if hasattr(v, "mean"):
                logging.info("test_%s: %s", k, v.mean().item())
            else:
                logging.info("test_%s: %s", k, v)
        return metric_curve

    def _evaluate_torch(
        self,
        video_BPFHWC,
        video_unprocessed,
        actions_mouse,
        actions_keyboard,
        real_lengths,
        eval_dir,
        num_denoising_steps=None,
    ):
        """Run rollout and metrics in PyTorch (mirrors base_mp_runner.evaluate_mp)."""
        from utils.rollout_torch import (
            perform_multiplayer_rollout_torch,
            get_model_inputs_for_eval_torch,
        )

        num_eval_frames = video_BPFHWC.shape[2]
        video_BPFHWC = video_BPFHWC.to(self.device)
        processed = wan_image_condition_preprocess_torch(video_BPFHWC, 352, 640)
        video_uint8 = change_tensor_range_torch(processed, [-1.0, 1.0], [0, 255])
        first_frame_BPHWC = processed[:, :, 0, :, :, :]
        n_frames_decode = video_uint8.shape[2]

        first_frames_BPFHWC = repeat(
            first_frame_BPHWC, "b p h w c -> b p f h w c", f=n_frames_decode
        ).to(self.device)
        actions_mouse = actions_mouse.to(self.device)
        actions_keyboard = actions_keyboard.to(self.device)

        scale_torch = [
            torch.tensor(s, dtype=torch.float32, device=self.device)
            for s in VAE_SCALE
        ]
        (
            cond_concat_BPFHWC,
            visual_context_BPFD,
            mouse_BPTD,
            keyboard_BPTD,
        ) = get_model_inputs_for_eval_torch(
            first_frames_BPFHWC,
            self.clip_model,
            self.vae_model,
            actions_mouse,
            actions_keyboard,
            self.multiplayer_method,
            self.device,
            vae_scale=scale_torch,
        )

        final_frame_BPFHWC = perform_multiplayer_rollout_torch(
            self.world_model,
            self.vae_model,
            self.clip_model,
            cond_concat_BPFHWC,
            visual_context_BPFD,
            mouse_BPTD,
            keyboard_BPTD,
            self.left_action_padding,
            self.multiplayer_method,
            self.device,
            num_denoising_steps=num_denoising_steps,
        )

        final_frame_BPFHWC = handle_multiplayer_output(
            final_frame_BPFHWC.cpu().numpy(),
            self.multiplayer_method,
            num_players=2,
        )
        final_frame_BPFHWC = torch.from_numpy(final_frame_BPFHWC).float().to(self.device)

        B = final_frame_BPFHWC.shape[0]
        scale_torch = [
            torch.tensor(s, dtype=torch.float32, device=self.device)
            for s in VAE_SCALE
        ]
        decoded = self.vae_model.decode(
            rearrange(final_frame_BPFHWC, "b p f h w c -> (b p) f h w c"),
            scale=scale_torch,
        )
        decoded = rearrange(decoded, "(b p) f h w c -> b p f h w c", b=B)
        decoded_uint8 = change_tensor_range_torch(decoded, [-1.0, 1.0], [0, 255])

        if eval_dir:
            rollout_np = decoded_uint8.cpu().numpy()
            gt_np = video_uint8.cpu().numpy()
            side_by_side = np.concatenate([gt_np, rollout_np], axis=4)
            side_by_side_BFHWC = rearrange(side_by_side, "b p f h w c -> b f (p h) w c")
            for i in range(side_by_side_BFHWC.shape[0]):
                write_video(
                    f"{eval_dir}/video_{i}_side_by_side.mp4",
                    side_by_side_BFHWC[i],
                    fps=20,
                )

        metrics = ["fid"]
        n_prompt_frames = 1
        fid_calculator = FIDCalculator(num_sources=2) if "fid" in metrics else None
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
        all_metrics = {}
        if fid_calculator is not None:
            all_metrics["fid"] = np.array(fid_calculator.get_fid_curve_jax())
        return all_metrics
