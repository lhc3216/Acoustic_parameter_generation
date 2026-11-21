import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.amp import autocast
from tqdm import tqdm

from tools.utils import save_residual_heatmaps


def _apply_condition_mode(
    ct: torch.Tensor,
    mr: torch.Tensor,
    condition_mode: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if condition_mode is None:
        return ct, mr
    mode = str(condition_mode).strip().lower().replace("-", "_")
    if mode in {"both", "ct_mri", "ct+mri"}:
        return ct, mr
    if mode in {"ct_only", "ct"}:
        return ct, torch.zeros_like(mr)
    if mode in {"mri_only", "mr_only", "mri", "mr"}:
        return torch.zeros_like(ct), mr
    if mode in {"uncond", "none"}:
        return torch.zeros_like(ct), torch.zeros_like(mr)
    raise ValueError(f"Unsupported validation condition_mode: {condition_mode}")


@torch.no_grad()
def validate_a_noise(
    val_loader,
    model,
    vae,
    noise_scheduler,
    cfg,
    device,
    use_amp=True,
    condition_mode: Optional[str] = None,
):
    """
    Fast validation on one-step noise prediction.

    Aggregate by sample count instead of batch count so the final partial batch
    does not bias the metric.
    """
    model.eval()
    vae.eval()

    val_loss_total = 0.0
    total_samples = 0

    for batch in tqdm(val_loader, desc="val-a"):
        condition = batch["condition"].to(device, non_blocking=True)
        slice_idx = batch["slice_idx"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        ct, mr = torch.chunk(condition, 2, dim=1)
        ct, mr = _apply_condition_mode(ct, mr, condition_mode)
        batch_size = target.shape[0]

        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            h = vae.encoder(target)
            clean_z0, _ = torch.chunk(h, 2, dim=1)
            clean_z0 = clean_z0 * cfg.diffusion.latent_scale_factor
            clean_z0 = torch.clamp(
                    clean_z0, 
                    -cfg.diffusion.static_threshold, 
                    cfg.diffusion.static_threshold
                )
        noise = torch.randn_like(clean_z0)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        z_t = noise_scheduler.add_noise(clean_z0, noise, timesteps)

        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            noise_pred = model(
                noisy_latent=z_t,
                timesteps=timesteps,
                ct_images=ct,
                mri_images=mr,
                slice_idx=slice_idx,
            )
            loss = F.mse_loss(noise_pred, noise)

        val_loss_total += loss.item() * batch_size
        total_samples += batch_size

    return val_loss_total / max(total_samples, 1)


@torch.no_grad()
def validate_b_physics(
    val_loader,
    model,
    vae,
    val_ddim_scheduler,
    cfg,
    device,
    num_steps=50,
    use_amp=True,
    save_path=None,
    steps=None,
    max_save_images=8,
    condition_mode: Optional[str] = None,
):
    """
    Full DDIM validation in physics space.

    PSNR/SSIM are accumulated by sample count to avoid the last smaller batch
    skewing the final mean.
    """
    model.eval()
    vae.eval()

    val_ddim_scheduler.set_timesteps(num_steps, device=device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    saved_images = 0
    step_tag = str(steps) if steps is not None else "na"

    for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"val-b {num_steps} steps")):
        condition = batch["condition"].to(device, non_blocking=True)
        slice_idx = batch["slice_idx"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        ct, mr = torch.chunk(condition, 2, dim=1)
        ct, mr = _apply_condition_mode(ct, mr, condition_mode)
        batch_size = target.shape[0]

        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            dummy_h = vae.encoder(target)
            dummy_z, _ = torch.chunk(dummy_h, 2, dim=1)

        latents = torch.randn_like(dummy_z)
        pred_z0 = None
        ct_zeros = torch.zeros_like(ct)
        mr_zeros = torch.zeros_like(mr)
        for t in val_ddim_scheduler.timesteps:
            batch_timesteps = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
            with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                noise_pred = model(
                    noisy_latent=latents,
                    timesteps=batch_timesteps,
                    ct_images=ct,
                    mri_images=mr,
                    slice_idx=slice_idx,
                )
                if cfg.diffusion.cfg_scale != 1.0:
                    noise_pred_uncond = model(
                        noisy_latent=latents,
                        timesteps=batch_timesteps,
                        ct_images=ct_zeros,
                        mri_images=mr_zeros,
                        slice_idx=slice_idx,
                    )
                    noise_pred = noise_pred_uncond + cfg.diffusion.cfg_scale * (noise_pred - noise_pred_uncond)
                else:
                    noise_pred = noise_pred

            step_out = val_ddim_scheduler.step(
                noise_pred,
                t,
                latents,
                use_clipped_model_output=True,
            )
            latents = step_out.prev_sample
            pred_z0 = step_out.pred_original_sample

        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            if pred_z0 is None:
                pred_z0 = latents
            pred_z0 = torch.clamp(
                pred_z0,
                -cfg.diffusion.static_threshold,
                cfg.diffusion.static_threshold,
            )
            pred_z0_unscaled = pred_z0 / cfg.diffusion.latent_scale_factor
            pred_physics = vae.decoder(pred_z0_unscaled)

        pred_physics = torch.clamp(pred_physics.float(), -1.0, 1.0)
        target = target.float()
        speed_pred, density_pred = torch.chunk(pred_physics, 2, dim=1)
        speed_target, density_target = torch.chunk(target, 2, dim=1)

        speed_pred_norm = (speed_pred + 1.0) / 2.0
        density_pred_norm = (density_pred + 1.0) / 2.0
        speed_target_norm = (speed_target + 1.0) / 2.0
        density_target_norm = (density_target + 1.0) / 2.0

        mse_speed_per_img = torch.mean((speed_pred_norm - speed_target_norm) ** 2, dim=[1, 2, 3])
        mse_density_per_img = torch.mean((density_pred_norm - density_target_norm) ** 2, dim=[1, 2, 3])
        eps = 1e-8
        psnr_speed = 10.0 * torch.log10(1.0 / (mse_speed_per_img + eps))
        psnr_density = 10.0 * torch.log10(1.0 / (mse_density_per_img + eps))
        batch_psnr = 0.5 * (psnr_speed + psnr_density)

        speed_ssim = ssim(speed_pred_norm, speed_target_norm, data_range=1.0, size_average=False)
        density_ssim = ssim(density_pred_norm, density_target_norm, data_range=1.0, size_average=False)
        batch_ssim = 0.5 * (speed_ssim + density_ssim)

        total_psnr += batch_psnr.sum().item()
        total_ssim += batch_ssim.sum().item()
        total_samples += batch_size

        if save_path is not None and saved_images < max_save_images:
            index = np.random.randint(0, batch_size)
            save_residual_heatmaps(
                target[index, 0, ...].cpu().numpy(),
                pred_physics[index, 0, ...].cpu().numpy(),
                save_path / f"speed_step{step_tag}_batch{batch_idx}_idx{index}.png",
                "Speed",
            )
            save_residual_heatmaps(
                target[index, 1, ...].cpu().numpy(),
                pred_physics[index, 1, ...].cpu().numpy(),
                save_path / f"density_step{step_tag}_batch{batch_idx}_idx{index}.png",
                "Density",
            )
            saved_images += 1

    return total_psnr / max(total_samples, 1), total_ssim / max(total_samples, 1)
