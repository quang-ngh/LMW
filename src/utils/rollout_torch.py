"""Torch versions of eval model inputs and multiplayer rollout (bidirectional)."""

import torch
from einops import rearrange

from utils.multiplayer import handle_multiplayer_input


def get_model_inputs_for_eval_torch(
    video_BPFHWC,
    clip_model,
    vae_model,
    actions_mouse_BPFD,
    actions_keyboard_BPFD,
    multiplayer_method,
    device,
    vae_scale=None,
):
    """Build conditioning tensors for eval (mirrors get_model_inputs_for_eval)."""
    B, P, F, H, W, C = video_BPFHWC.shape
    video_BPFHWC = video_BPFHWC.to(device)
    first_frame_BPFHWC = video_BPFHWC[:, :, 0:1, :, :, :]

    with torch.no_grad():
        visual_context_BPFD = clip_model.encode_video(
            rearrange(first_frame_BPFHWC, "b p f h w c -> (b p) c f h w")
        )
    visual_context_BPFD = rearrange(
        visual_context_BPFD.float(), "(b p) f d -> b p f d", b=B
    ).to(video_BPFHWC.dtype)

    compress = lambda x: rearrange(x, "b p f h w c -> (b p) f h w c")
    uncompress = lambda x: rearrange(x, "(b p) f h w c -> b p f h w c", b=B)
    padded = torch.cat(
        [
            video_BPFHWC[:, :, :1, :, :, :],
            torch.zeros(B, P, F - 1, H, W, C, dtype=video_BPFHWC.dtype, device=device),
        ],
        dim=2,
    )
    with torch.no_grad():
        enc = vae_model.encode(compress(padded), scale=vae_scale)
    img_cond_BPFHWC = uncompress(enc)
    mask_cond_BPFHWC = torch.ones_like(img_cond_BPFHWC, dtype=img_cond_BPFHWC.dtype, device=device)
    mask_cond_BPFHWC[:, :, 1:] = 0
    cond_concat_BPFHWC = torch.cat(
        [mask_cond_BPFHWC[:, :, :, :, :, :4], img_cond_BPFHWC], dim=-1
    )

    (
        cond_concat_BPFHWC,
        actions_mouse_BPFD,
        actions_keyboard_BPFD,
        visual_context_BPFD,
        _,
    ) = handle_multiplayer_input(
        cond_concat_BPFHWC,
        actions_mouse_BPFD,
        actions_keyboard_BPFD,
        multiplayer_method,
        visual_context_BPFD,
        video_BPFHWC=None,
    )
    return (
        cond_concat_BPFHWC,
        visual_context_BPFD,
        actions_mouse_BPFD,
        actions_keyboard_BPFD,
    )


def left_repeat_padding_torch(x, pad, axis):
    """Repeat first slice along axis and concat to the left."""
    if axis == 1:
        return torch.cat([x[:, 0:1].repeat(1, pad, *([1] * (x.dim() - 2))), x], dim=1)
    elif axis == 2:
        return torch.cat([x[:, :, 0:1].repeat(1, 1, pad, *([1] * (x.dim() - 3))), x], dim=2)
    raise ValueError(f"Invalid axis: {axis}")


def flow_prediction_to_x0_torch(flow_pred, x_t, sigma):
    return x_t - sigma * flow_pred


def flow_match_inference_timesteps_torch(
    num_inference_steps,
    timestep_shift=5.0,
    sigma_min=0.0,
    sigma_max=1.0,
    num_train_timesteps=1000,
    denoising_strength=1.0,
    extra_one_step=True,
    device=None,
):
    sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
    if extra_one_step:
        sigmas = torch.linspace(
            sigma_start, sigma_min, num_inference_steps + 1, device=device
        )[:-1]
    else:
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps, device=device)
    sigmas = timestep_shift * sigmas / (1.0 + (timestep_shift - 1.0) * sigmas)
    timesteps = sigmas * num_train_timesteps
    return torch.cat([timesteps, torch.zeros(1, device=device, dtype=timesteps.dtype)])


def fully_denoise_frame_bidirectional_multiplayer_torch(
    world_model,
    visual_context_BPFD,
    cond_concat_BPFHWC,
    frame_noise_BPFHWC,
    mouse_actions_BPTD,
    keyboard_actions_BPTD,
    left_action_padding,
    device,
    num_denoising_steps=10,
):
    """Fully denoise one full frame sequence (bidirectional, no KV cache)."""
    if num_denoising_steps is None:
        num_denoising_steps = 10
    inference_timesteps = flow_match_inference_timesteps_torch(
        num_denoising_steps, device=device
    )
    print(f"inference_timesteps: {inference_timesteps}")
    sigma_t = inference_timesteps / 1000.0
    B, P, F, Hl, Wl, latent_c = frame_noise_BPFHWC.shape

    padded_mouse = left_repeat_padding_torch(mouse_actions_BPTD, left_action_padding, axis=2)
    padded_keyboard = left_repeat_padding_torch(keyboard_actions_BPTD, left_action_padding, axis=2)

    frame = frame_noise_BPFHWC
    for i in range(num_denoising_steps):
        t = inference_timesteps[i].item()
        t_batch = torch.full((B, P, F), t, dtype=torch.long, device=device)
        v, _, _, _ = world_model(
            frame,
            t_batch,
            visual_context_BPFD,
            cond_concat_BPFHWC,
            padded_mouse,
            padded_keyboard,
            bidirectional=True,
        )
        x_start_f32 = flow_prediction_to_x0_torch(
            v.float(), frame.float(), sigma_t[i].item()
        )
        if i == num_denoising_steps - 1:
            frame = x_start_f32.to(frame.dtype)
        else:
            noise = torch.randn_like(x_start_f32, device=device)
            frame = (
                (1 - sigma_t[i + 1]) * x_start_f32
                + sigma_t[i + 1] * noise
            ).to(frame.dtype)
    return frame


def perform_multiplayer_rollout_torch(
    world_model,
    vae_model,
    clip_model,
    cond_concat_BPFHWC,
    visual_context_BPFD,
    mouse_actions_BPTD,
    keyboard_actions_BPTD,
    left_action_padding,
    multiplayer_method,
    device,
    num_denoising_steps=None,
):
    """Bidirectional multiplayer rollout: denoise full sequence then return latent (B,P,F,H,W,C)."""
    B, P_eff, F, Hl, Wl, _ = cond_concat_BPFHWC.shape
    num_players = 2
    latent_channels = 16 * num_players if multiplayer_method == "concat_c" else 16
    frame_noise_BPFHWC = torch.randn(
        B, P_eff, F, Hl, Wl, latent_channels,
        dtype=cond_concat_BPFHWC.dtype,
        device=device,
    )

    frame = fully_denoise_frame_bidirectional_multiplayer_torch(
        world_model,
        visual_context_BPFD,
        cond_concat_BPFHWC,
        frame_noise_BPFHWC,
        mouse_actions_BPTD,
        keyboard_actions_BPTD,
        left_action_padding,
        device,
        num_denoising_steps=num_denoising_steps,
    )
    return frame
