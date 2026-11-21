import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(Path(__file__).parent.parent))
import os
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from models.vae.vae import AcousticVAE
from tools.utils import compute_metrics, denormalize, PHYSICS_RANGE
from data.test_vae_dataset import get_vae_data_dicts_all_slices, get_static_cache_transforms
from monai.data import DataLoader, CacheDataset, PersistentDataset
from tqdm import tqdm
from tools.utils import print_log, save_residual_heatmaps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DENSITY_MASK_THRESHOLD = 1005.0
SPEED_MASK_THRESHOLD = 1485.0
MIN_MASK_PIXELS = 100


def build_brain_mask(speed_gt, density_gt):
    mask = (speed_gt > SPEED_MASK_THRESHOLD) | (density_gt > DENSITY_MASK_THRESHOLD)
    if int(mask.sum()) < MIN_MASK_PIXELS:
        return np.ones_like(mask, dtype=bool)
    return mask.astype(bool)


def compute_masked_metrics(original, reconstructed, mask, data_range):
    mask = mask.astype(bool)
    if int(mask.sum()) < MIN_MASK_PIXELS:
        mask = np.ones_like(mask, dtype=bool)

    masked_mse = float(np.mean((original[mask] - reconstructed[mask]) ** 2))
    if masked_mse <= 0:
        psnr_val = float("inf")
    else:
        psnr_val = float(10.0 * np.log10((data_range ** 2) / masked_mse))

    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    original_crop = original[r0:r1, c0:c1]
    reconstructed_crop = reconstructed[r0:r1, c0:c1]
    mask_crop = mask[r0:r1, c0:c1]
    reconstructed_crop = np.where(mask_crop, reconstructed_crop, original_crop)
    _, ssim_val = compute_metrics(original_crop, reconstructed_crop, data_range=data_range)

    return psnr_val, float(ssim_val)


@hydra.main(version_base=None, config_path=str(ROOT/'configs'), config_name='vae_config')
def test_vae(cfg: DictConfig):

    ckpt_path = Path(ROOT) / cfg.trainer.save_dir / cfg.task / cfg.tester.test_pt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    save_path = Path(ROOT) / cfg.tester.test_save_dir / cfg.task
    save_path.mkdir(parents=True, exist_ok=True)

    # img_dir = save_path / "images"
    # img_dir.mkdir(parents=True, exist_ok=True)

    np_dir = save_path / "np"
    np_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(save_path / "test_vae.log", "a")
    print_log(log_file, f"test ave on {ckpt_path}")

    model = AcousticVAE().to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if 'epoch' in checkpoint:
        print_log(log_file, f'load_epochs: {checkpoint["epoch"]}')
    model.eval()

    print("Loading test dataset...")
    test_dicts = get_vae_data_dicts_all_slices(cfg.data_root, split="test")
    static_transforms = get_static_cache_transforms()
    
    test_dataset = PersistentDataset(
        data=test_dicts,
        transform=static_transforms,
        cache_dir=os.path.join(cfg.data_root, "vae_persistent_cache_512_test")
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.tester.batch_size,  
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
    )

    print("🚀 calculating...")
    metrics = {
        "norm_speed_psnr": [], "norm_speed_ssim": [],
        "norm_density_psnr": [], "norm_density_ssim": [],
        "speed_mae": [], "speed_mse": [], "speed_rmse": [],
        "density_mae": [], "density_mse": [], "density_rmse": [],
        "impedance_mae": [], "impedance_mape_percent": []
    }
   
    vis_count = 0
    total_samples = 0
    # all_mu = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            inputs = batch["target"].to(DEVICE)
            per_id = batch["patient_id"]
            slice_idx = batch["slice_idx"] 
 
            outputs, mu, _ = model(inputs, sample=False) 
            # all_mu.append(mu.cpu())
            total_samples += inputs.shape[0]

            
            batch_metrics = {k: [] for k in metrics}
            for i in range(inputs.shape[0]):
                per_id_i = per_id[i] if isinstance(per_id, (list, tuple)) else per_id
                slice_idx_i = slice_idx[i] if isinstance(slice_idx, (list, tuple, torch.Tensor)) else slice_idx
                if isinstance(slice_idx_i, torch.Tensor):
                    slice_idx_i = int(slice_idx_i.item())
                else:
                    slice_idx_i = int(slice_idx_i)

                input_np = inputs[i].detach().cpu().numpy()
                output_np = outputs[i].detach().cpu().numpy()
                input_np_speed = input_np[0, ...].squeeze()
                input_np_density = input_np[1, ...].squeeze()
                output_np_speed = output_np[0, ...].squeeze()
                output_np_density = output_np[1, ...].squeeze()

                # save_residual_heatmaps(input_np_speed, output_np_speed, os.path.join(img_dir, f"{per_id_i}_{slice_idx_i:03d}_residual_speed.png"), "Speed", "m/s")
                # save_residual_heatmaps(input_np_density, output_np_density, os.path.join(img_dir, f"{per_id_i}_{slice_idx_i:03d}_residual_density.png"), "Density", "kg/m^3")

                speed_orig = denormalize(input_np_speed, PHYSICS_RANGE["speed"]["min"], PHYSICS_RANGE["speed"]["max"])
                speed_recon = denormalize(output_np_speed, PHYSICS_RANGE["speed"]["min"], PHYSICS_RANGE["speed"]["max"])

            
                
                density_orig = denormalize(input_np_density, PHYSICS_RANGE["density"]["min"], PHYSICS_RANGE["density"]["max"])
                density_recon = denormalize(output_np_density, PHYSICS_RANGE["density"]["min"], PHYSICS_RANGE["density"]["max"])
                brain_mask = build_brain_mask(speed_orig, density_orig)
                p_speed, s_speed = compute_masked_metrics(input_np_speed, output_np_speed, brain_mask, data_range=2.0)
                p_density, s_density = compute_masked_metrics(input_np_density, output_np_density, brain_mask, data_range=2.0)
                batch_metrics["norm_speed_psnr"].append(float(p_speed))
                batch_metrics["norm_speed_ssim"].append(float(s_speed))
                batch_metrics["norm_density_psnr"].append(float(p_density))
                batch_metrics["norm_density_ssim"].append(float(s_density))
                # np.save(os.path.join(np_dir, f"{per_id_i}_{slice_idx_i:03d}_speed_orig.npy"), speed_orig)
                # np.save(os.path.join(np_dir, f"{per_id_i}_{slice_idx_i:03d}_speed_recon.npy"), speed_recon)
                # np.save(os.path.join(np_dir, f"{per_id_i}_{slice_idx_i:03d}_density_orig.npy"), density_orig)
                # np.save(os.path.join(np_dir, f"{per_id_i}_{slice_idx_i:03d}_density_recon.npy"), density_recon)

                speed_err = np.abs(speed_orig - speed_recon)[brain_mask]
                density_err = np.abs(density_orig - density_recon)[brain_mask]
                speed_sq_err = ((speed_orig - speed_recon) ** 2)[brain_mask]
                density_sq_err = ((density_orig - density_recon) ** 2)[brain_mask]
                speed_mse = float(speed_sq_err.mean())
                density_mse = float(density_sq_err.mean())
                impedance_orig = density_orig * speed_orig
                impedance_recon = density_recon * speed_recon
                impedance_orig_masked = impedance_orig[brain_mask]
                impedance_abs_err = np.abs(impedance_orig - impedance_recon)[brain_mask]
                impedance_mape_percent = float(np.mean(impedance_abs_err / (np.abs(impedance_orig_masked) + 1e-8)) * 100.0)

                batch_metrics["speed_mae"].append(float(speed_err.mean()))
                batch_metrics["speed_mse"].append(speed_mse)
                batch_metrics["speed_rmse"].append(float(np.sqrt(speed_mse)))
                batch_metrics["density_mae"].append(float(density_err.mean()))
                batch_metrics["density_mse"].append(density_mse)
                batch_metrics["density_rmse"].append(float(np.sqrt(density_mse)))
                batch_metrics["impedance_mae"].append(float(impedance_abs_err.mean()))
                batch_metrics["impedance_mape_percent"].append(impedance_mape_percent)
                vis_count += 1

            batch_summary = {}
            for key, values in batch_metrics.items():
                metrics[key].extend(values)
                batch_summary[key] = float(np.mean(values)) if len(values) > 0 else float("nan")

            print_log(
                log_file,
                (
                    f"{batch_summary['norm_speed_psnr']:.6f},{batch_summary['norm_speed_ssim']:.6f},"
                    f"{batch_summary['norm_density_psnr']:.6f},{batch_summary['norm_density_ssim']:.6f},"
                    f"{batch_summary['speed_mae']:.6f},{batch_summary['speed_mse']:.6f},"
                    f"{batch_summary['speed_rmse']:.6f},{batch_summary['density_mae']:.6f},"
                    f"{batch_summary['density_mse']:.6f},{batch_summary['density_rmse']:.6f},"
                    f"{batch_summary['impedance_mae']:.6f},{batch_summary['impedance_mape_percent']:.6f}"
                )
            )
    summary = {}
    for k, v in metrics.items():
        values = np.asarray(v, dtype=np.float64)
        summary[f"{k}_mean"] = float(np.mean(values)) if len(values) > 0 else float("nan")
        summary[f"{k}_var"] = float(np.var(values)) if len(values) > 0 else float("nan")
    summary["num_test_samples"] = total_samples
    summary["num_visualized_samples"] = vis_count

    print("\nEvaluation complete. Metric summary:")
    for k, v in summary.items():
        print_log(log_file, (f"{k}: {v:.6f}" if isinstance(v, float) and not np.isnan(v) else f"{k}: {v}"))
                

  
    # all_mu = torch.cat(all_mu, dim=0).cpu()
    # global_std = torch.std(all_mu) # [N, 4, 64, 64]
    # latent_scale_factor = 1.0 / global_std.item()
    # channel_stds = torch.std(all_mu, dim=(0, 2, 3))

    # scaled_mu = all_mu * latent_scale_factor
    # abs_scaled_mu = torch.abs(scaled_mu).float()

    # flat_mu = abs_scaled_mu.view(-1)

    # active_mu = flat_mu[flat_mu > 1e-2]

    # indices = torch.randperm(active_mu.numel())[:10000000]
    # final_samples = active_mu[indices]

    # q_995 = torch.quantile(final_samples, 0.995).item()
    # q_999 = torch.quantile(final_samples, 0.999).item()

    # max_val = torch.max(final_samples).item()
    
    
    # recommended_threshold = round(q_999, 1)
    # log_file.close()


if __name__ == "__main__":
    test_vae()
