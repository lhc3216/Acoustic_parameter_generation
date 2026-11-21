import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import hydra
import csv
import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler
from omegaconf import DictConfig
from torch.amp import autocast
from tqdm import tqdm

from data.test_sd_dataset import build_dataloader
from models.sd.sd import HybridAcousticDiffusion
from models.vae.vae import AcousticVAE
from tools.utils import PHYSICS_RANGE, denormalize
import time

DENSITY_MASK_THRESHOLD = 1005.0
SPEED_MASK_THRESHOLD = 1485.0
MIN_MASK_PIXELS = 100
METRIC_NAMES = (
    "Speed MAE (m/s)",
    "Speed RMSE (m/s)",
    "Density MAE (kg/m³)",
    "Density RMSE (kg/m³)",
    "Impedance MAPE (%)",
)

def _build_brain_mask(speed_gt: np.ndarray, density_gt: np.ndarray) -> np.ndarray:
    mask = (speed_gt > SPEED_MASK_THRESHOLD) | (density_gt > DENSITY_MASK_THRESHOLD)
    if int(mask.sum()) < MIN_MASK_PIXELS:
        return np.ones_like(mask, dtype=bool)
    return mask.astype(bool)


def _load_vae(cfg: DictConfig, device: torch.device) -> AcousticVAE:
    vae = AcousticVAE().to(device)
    ckpt = torch.load(cfg.trainer.vae_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    vae.load_state_dict(state_dict, strict=True)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


def _resolve_test_checkpoint(cfg: DictConfig) -> Path:
    test_pt = str(cfg.tester.test_pt)
    test_path = Path(test_pt)
    if test_path.is_absolute():
        return test_path
    return ROOT / cfg.trainer.save_dir / cfg.task / test_path


def _normalize_state_dict_for_model(state_dict):
    """
    Make checkpoint keys compatible with plain HybridAcousticDiffusion.
    Handles EMA/DDP style keys such as:
    - `module.xxx`
    - extra `n_averaged` from AveragedModel
    """
    normalized = {}
    for key, value in state_dict.items():
        if key == "n_averaged":
            continue
        new_key = key[7:] if key.startswith("module.") else key
        normalized[new_key] = value
    return normalized


def _sample_id_and_slice(batch, index: int) -> tuple[str, int]:
    patient_id = batch["patient_id"]
    slice_idx = batch["slice_idx"]

    patient_id_i = patient_id[index] if isinstance(patient_id, (list, tuple)) else patient_id
    slice_idx_i = slice_idx[index] if isinstance(slice_idx, (list, tuple, torch.Tensor)) else slice_idx
    if isinstance(slice_idx_i, torch.Tensor):
        slice_idx_i = int(slice_idx_i.item())
    else:
        slice_idx_i = int(slice_idx_i)
    return str(patient_id_i), slice_idx_i


def _denormalize_speed_density(sample: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    sample_np = sample.detach().cpu().numpy()
    speed = sample_np[0, ...].squeeze()
    density = sample_np[1, ...].squeeze()
    speed_phys = denormalize(speed, PHYSICS_RANGE["speed"]["min"], PHYSICS_RANGE["speed"]["max"])
    density_phys = denormalize(density, PHYSICS_RANGE["density"]["min"], PHYSICS_RANGE["density"]["max"])
    return speed_phys, density_phys


@torch.no_grad()
def _evaluate_with_cfg(
    val_loader,
    model,
    vae,
    scheduler,
    cfg: DictConfig,
    device: torch.device,
    num_steps: int,
    cfg_scale: float,
    use_amp: bool,
):
    model.eval()
    vae.eval()
    scheduler.set_timesteps(num_steps, device=device)

    # Compute one metric value per slice, then summarize across slices.
    # MAE/RMSE use denormalized physical values, separately for each channel.
    metric_values = {name: [] for name in METRIC_NAMES}

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    for batch in tqdm(val_loader, desc=f"test {scheduler.__class__.__name__} steps={num_steps}"):
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        slice_idx = batch["slice_idx"].to(device, non_blocking=True)
        ct, mr = torch.chunk(condition, 2, dim=1)
        batch_size = target.shape[0]
        with autocast(device_type=device.type, enabled=use_amp):
            dummy_h = vae.encoder(target)
            dummy_z, _ = torch.chunk(dummy_h, 2, dim=1)

        latents = torch.randn_like(dummy_z)
        pred_z0 = None
        ct_zeros = torch.zeros_like(ct)
        mr_zeros = torch.zeros_like(mr)
        for t in scheduler.timesteps:
            batch_timesteps = torch.full((batch_size,), int(t), device=device, dtype=torch.long)

            with autocast(device_type=device.type, enabled=use_amp):
                noise_pred_cond = model(
                    noisy_latent=latents,
                    timesteps=batch_timesteps,
                    ct_images=ct_zeros,
                    mri_images=mr,
                    slice_idx=slice_idx,
                )

                if cfg_scale != 1.0:
                    noise_pred_uncond = model(
                        noisy_latent=latents,
                        timesteps=batch_timesteps,
                        ct_images=ct_zeros,
                        mri_images=mr_zeros,
                        slice_idx=slice_idx,
                    )
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

            step_out = scheduler.step(
                noise_pred,
                t,
                latents,
                # use_clipped_model_output=True,
            )
            latents = step_out.prev_sample
            pred_z0 = step_out.pred_original_sample

        with autocast(device_type=device.type, enabled=use_amp):
            if pred_z0 is None:
                pred_z0 = latents
            pred_z0 = torch.clamp(
                pred_z0,
                -cfg.diffusion.static_threshold,
                cfg.diffusion.static_threshold,
            )
            pred_z0_unscaled = pred_z0 / cfg.diffusion.latent_scale_factor
            pred_physics = vae.decoder(pred_z0_unscaled)

        pred_physics = pred_physics.float()
        target = target.float()
        pred_physics = torch.clamp(pred_physics, -1.0, 1.0)

        for index in range(target.shape[0]):
            speed_orig, density_orig = _denormalize_speed_density(target[index])
            speed_recon, density_recon = _denormalize_speed_density(pred_physics[index])

            brain_mask = _build_brain_mask(speed_orig, density_orig)
            speed_err = (speed_orig - speed_recon)[brain_mask].astype(np.float64)
            density_err = (density_orig - density_recon)[brain_mask].astype(np.float64)

            impedance_orig = density_orig * speed_orig
            impedance_recon = density_recon * speed_recon
            impedance_relative_error = (
                np.abs(impedance_orig - impedance_recon)[brain_mask]
                / (np.abs(impedance_orig[brain_mask]) + 1e-8)
            )
            metric_values["Speed MAE (m/s)"].append(float(np.mean(np.abs(speed_err))))
            metric_values["Speed RMSE (m/s)"].append(float(np.sqrt(np.mean(np.square(speed_err)))))
            metric_values["Density MAE (kg/m³)"].append(float(np.mean(np.abs(density_err))))
            metric_values["Density RMSE (kg/m³)"].append(float(np.sqrt(np.mean(np.square(density_err)))))
            metric_values["Impedance MAPE (%)"].append(float(np.mean(impedance_relative_error) * 100.0))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_time = time.perf_counter() - start_time

    if not metric_values[METRIC_NAMES[0]]:
        raise ValueError("The test loader produced no valid evaluation pixels")

    metrics = {
        f"{key}_mean": float(np.mean(values))
        for key, values in metric_values.items()
    }
    metrics.update({
        f"{key}_std": float(np.std(values, ddof=0))
        for key, values in metric_values.items()
    })
    metrics["Time (s)"] = elapsed_time
    return metrics


@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="sd_config")
def test_sd(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.trainer.use_amp and device.type == "cuda")

    save_dir = ROOT / cfg.tester.test_save_dir / cfg.tester.now
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = _resolve_test_checkpoint(cfg)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Test checkpoint not found: {ckpt_path}")

    test_cache_dir = Path(cfg.data_root) / "sd_persistent_cache_512_test"
    # test_cache_dir.mkdir(parents=True, exist_ok=True)
    test_loader = build_dataloader(
        root_dir=cfg.data_root,
        split=str(cfg.tester.get("split", "test")),
        cache_dir=test_cache_dir,
        batch_size=cfg.tester.test_batch,
        num_workers=cfg.num_workers,
    )

    vae = _load_vae(cfg, device)
    model = HybridAcousticDiffusion(
        pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
        latent_channels=cfg.model.latent_channels,
        cond_channels=cfg.model.cond_channels,
        max_z_slices=int(cfg.model.get("max_z_slices", 256)),
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("ema_model_state_dict") or checkpoint.get("model_state_dict") or checkpoint
    state_dict = _normalize_state_dict_for_model(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    del state_dict
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
        clip_sample=bool(cfg.diffusion.get("clip_sample", True)),
        clip_sample_range=cfg.diffusion.static_threshold,
        thresholding=bool(cfg.diffusion.get("thresholding", False)),
        dynamic_thresholding_ratio=float(cfg.diffusion.get("dynamic_thresholding_ratio", 0.995)),
    )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
        clip_sample=bool(cfg.diffusion.get("clip_sample", True)),
        clip_sample_range=cfg.diffusion.static_threshold,
    )

    samplers = {
        "DDPM": ddpm_scheduler,
        "DDIM": ddim_scheduler,
    }
    step_values = [4]
    results_path = save_dir / "summary.csv"
    fieldnames = [
        "Sampler", "Steps",
        *(f"{name} (mean±std)" for name in METRIC_NAMES),
        "Time (s)",
    ]
    with results_path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for sampler_name, scheduler in samplers.items():
        for steps in step_values:
            if steps <= 0 or steps > int(cfg.diffusion.num_train_timesteps):
                raise ValueError(
                    f"Sampling steps must be in [1, {cfg.diffusion.num_train_timesteps}], got {steps}"
                )
            cfg_scale = float(cfg.diffusion.get("cfg_scale", 1.0))
            sample_steps = int(steps)
            # Reset RNGs for each configuration so runs start from the same
            # random stream (assuming the test loader has deterministic order).
            test_seed = int(cfg.diffusion.get("test_seed", 42))
            torch.manual_seed(test_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(test_seed)
            metrics = _evaluate_with_cfg(
                val_loader=test_loader,
                model=model,
                vae=vae,
                scheduler=scheduler,
                cfg=cfg,
                device=device,
                num_steps=sample_steps,
                cfg_scale=cfg_scale,
                use_amp=use_amp,
            )
            row = {
                "Sampler": sampler_name,
                "Steps": sample_steps,
                **{
                    f"{name} (mean±std)": (
                        f"{metrics[f'{name}_mean']:.6f} ± {metrics[f'{name}_std']:.6f}"
                    )
                    for name in METRIC_NAMES
                },
                "Time (s)": f"{metrics['Time (s)']:.2f}",
            }
            with results_path.open("a", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
            metric_summary = ", ".join(
                f"{name}={row[f'{name} (mean±std)']}"
                for name in METRIC_NAMES
            )
            print(f"{sampler_name} steps={sample_steps}: {metric_summary}, Time={row['Time (s)']}s")

    print(f"Saved summary metrics to: {results_path}")


if __name__ == "__main__":
    test_sd()
