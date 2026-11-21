import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import json
import os
from typing import Dict

import hydra
import numpy as np
import torch
from diffusers import DDPMScheduler
from omegaconf import DictConfig
from torch.amp import autocast
from tqdm import tqdm

from data.test_sd_dataset import build_dataloader
from models.dmd2.dmd2 import build_generator_model
from models.dmd2.dmd2_loss import sample_terminal_latent
from models.vae.vae import AcousticVAE
from tools.utils import PHYSICS_RANGE, compute_metrics, denormalize, print_log, save_images


def _build_scheduler(cfg: DictConfig) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
    )


def _load_vae(cfg: DictConfig, device: torch.device) -> AcousticVAE:
    vae = AcousticVAE().to(device)
    ckpt = torch.load(cfg.paths.vae_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    vae.load_state_dict(state_dict, strict=True)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


def _resolve_dmd2_checkpoint(cfg: DictConfig) -> str:
    best_ckpt = os.path.join(cfg.paths.save_dir, "dmd2_best.pth")
    if os.path.exists(best_ckpt):
        return best_ckpt
    resume_from = str(cfg.trainer.get("resume_from", "")).strip()
    if resume_from:
        if os.path.isabs(resume_from):
            return resume_from
        return os.path.join(cfg.paths.save_dir, resume_from)
    raise FileNotFoundError(
        f"No DMD2 checkpoint found. Expected {best_ckpt} or cfg.trainer.resume_from."
    )


@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="dmd2")
def test_dmd2(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.trainer.use_amp and device.type == "cuda")

    test_cache_dir = Path(cfg.data_root) / "sd_persistent_cache_512_test"
    test_cache_dir.mkdir(parents=True, exist_ok=True)
    test_loader = build_dataloader(
        root_dir=cfg.data_root,
        split="test",
        cache_dir=test_cache_dir,
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.num_workers,
    )

    scheduler = _build_scheduler(cfg)
    vae = _load_vae(cfg, device)
    generator = build_generator_model(cfg, device)
    dmd2_ckpt = _resolve_dmd2_checkpoint(cfg)
    ckpt = torch.load(dmd2_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt.get("ema_generator_state_dict") or ckpt.get("generator_state_dict")
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()

    out_root = os.path.join(str(ROOT), "test_outputs", "dmd2_eval")
    img_root = os.path.join(out_root, "images")
    speed_img_root = os.path.join(img_root, "speed")
    density_img_root = os.path.join(img_root, "density")
    log_root = os.path.join(out_root, "logs")
    os.makedirs(speed_img_root, exist_ok=True)
    os.makedirs(density_img_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    sample_csv_path = os.path.join(log_root, "dmd2_eval_per_sample.csv")
    summary_json_path = os.path.join(log_root, "dmd2_eval_summary.json")

    metrics: Dict[str, list] = {
        "norm_speed_psnr": [],
        "norm_speed_ssim": [],
        "norm_density_psnr": [],
        "norm_density_ssim": [],
        "speed_mae": [],
        "speed_rmse": [],
        "density_mae": [],
        "density_rmse": [],
    }

    save_vis_max = int(cfg.get("eval", {}).get("save_vis_max", 20))
    vis_count = 0
    sample_idx = 0

    log_file = open(sample_csv_path, "w", encoding="utf-8")
    print_log(
        log_file,
        "idx,norm_speed_psnr,norm_speed_ssim,norm_density_psnr,norm_density_ssim,"
        "speed_mae,speed_rmse,density_mae,density_rmse",
    )

    print(f"Using DMD2 checkpoint: {dmd2_ckpt}")
    print("Start DMD2 one-step inference on test split...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="DMD2 Test"):
            condition = batch["condition"].to(device, non_blocking=True)
            ct, mr = torch.chunk(condition, 2, dim=1)
            target = batch["target"].to(device, non_blocking=True)
            slice_idx = batch["slice_idx"].to(device, non_blocking=True)

            latent_shape = (
                target.shape[0],
                cfg.model.latent_channels,
                target.shape[-2] // 8,
                target.shape[-1] // 8,
            )

            with autocast(device_type=device.type, enabled=use_amp):
                _, _, z0_pred, _ = sample_terminal_latent(
                    model=generator,
                    scheduler=scheduler,
                    latent_shape=torch.Size(latent_shape),
                    ct_cond=ct,
                    mr_cond=mr,
                    slice_idx=slice_idx,
                )
                pred = vae.decoder(z0_pred / cfg.loss.latent_scale_factor)
            pred = torch.clamp(pred.float(), -1.0, 1.0)

            target_np = target.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()

            for i in range(target_np.shape[0]):
                gt_speed = target_np[i, 0]
                gt_density = target_np[i, 1]
                pred_speed = pred_np[i, 0]
                pred_density = pred_np[i, 1]

                p_speed, s_speed = compute_metrics(gt_speed, pred_speed, data_range=2.0)
                p_density, s_density = compute_metrics(gt_density, pred_density, data_range=2.0)

                metrics["norm_speed_psnr"].append(float(p_speed))
                metrics["norm_speed_ssim"].append(float(s_speed))
                metrics["norm_density_psnr"].append(float(p_density))
                metrics["norm_density_ssim"].append(float(s_density))

                speed_gt_phys = denormalize(
                    gt_speed, PHYSICS_RANGE["speed"]["min"], PHYSICS_RANGE["speed"]["max"]
                )
                speed_pred_phys = denormalize(
                    pred_speed, PHYSICS_RANGE["speed"]["min"], PHYSICS_RANGE["speed"]["max"]
                )
                density_gt_phys = denormalize(
                    gt_density, PHYSICS_RANGE["density"]["min"], PHYSICS_RANGE["density"]["max"]
                )
                density_pred_phys = denormalize(
                    pred_density, PHYSICS_RANGE["density"]["min"], PHYSICS_RANGE["density"]["max"]
                )

                speed_err = np.abs(speed_gt_phys - speed_pred_phys)
                density_err = np.abs(density_gt_phys - density_pred_phys)

                speed_mae = float(speed_err.mean())
                speed_rmse = float(np.sqrt((speed_err ** 2).mean()))
                density_mae = float(density_err.mean())
                density_rmse = float(np.sqrt((density_err ** 2).mean()))

                metrics["speed_mae"].append(speed_mae)
                metrics["speed_rmse"].append(speed_rmse)
                metrics["density_mae"].append(density_mae)
                metrics["density_rmse"].append(density_rmse)

                print_log(
                    log_file,
                    (
                        f"{sample_idx},{p_speed:.6f},{s_speed:.6f},{p_density:.6f},{s_density:.6f},"
                        f"{speed_mae:.6f},{speed_rmse:.6f},{density_mae:.6f},{density_rmse:.6f}"
                    ),
                )

                if vis_count < save_vis_max:
                    save_images(gt_speed, pred_speed, speed_img_root, save_num=sample_idx)
                    save_images(gt_density, pred_density, density_img_root, save_num=sample_idx)
                    vis_count += 1

                sample_idx += 1

    log_file.close()

    summary = {key: float(np.mean(value)) if value else float("nan") for key, value in metrics.items()}
    summary["num_test_samples"] = int(sample_idx)
    summary["num_visualized_samples"] = int(vis_count)
    summary["dmd2_checkpoint"] = dmd2_ckpt
    summary["oneshot_timestep"] = int(cfg.diffusion.num_train_timesteps - 1)

    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)

    print("\nDMD2 test completed. Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"Per-sample metrics: {sample_csv_path}")
    print(f"Summary metrics: {summary_json_path}")

    return summary


if __name__ == "__main__":
    test_dmd2()
