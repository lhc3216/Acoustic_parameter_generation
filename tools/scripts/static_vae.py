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
from data.vae_dataset import (
    get_static_cache_transforms,
    get_vae_data_dicts_all_slices,
)
from monai.data import DataLoader, CacheDataset, PersistentDataset
from tqdm import tqdm
from tools.utils import print_log, save_residual_heatmaps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path=str(ROOT/'configs'), config_name='vae_config')
def test_vae(cfg: DictConfig):

    ckpt_path = Path(ROOT) / cfg.trainer.save_dir / cfg.task / cfg.tester.test_pt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    save_path = Path(ROOT) / cfg.tester.test_save_dir / cfg.task
    save_path.mkdir(parents=True, exist_ok=True)

    log_file = open(save_path / "static_vae.log", "a")
    print_log(log_file, f"static vae on {ckpt_path}")

    model = AcousticVAE().to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if 'epoch' in checkpoint:
        print_log(log_file, f'load_epochs: {checkpoint["epoch"]}')
    model.eval()

    print("Loading training dataset...")
    train_cache_dir = os.path.join(cfg.data_root, "vae_persistent_cache_512_train")
    train_dicts = get_vae_data_dicts_all_slices(cfg.data_root, split="train")
    static_transforms = get_static_cache_transforms()

    train_dataset = PersistentDataset(
        data=train_dicts,
        transform=static_transforms,
        cache_dir=train_cache_dir,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    print("🚀 calculating...")

    all_mu = []
    sum_mu = torch.zeros(4, device=DEVICE)
    sum_squares_mu = torch.zeros(4, device=DEVICE)
    total_pixels_per_channel = 0
    sampled_active_points = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            inputs = batch["target"].to(DEVICE)
 
            _, mu, _ = model(inputs, sample=False)
            if batch_idx < 100:
                print_log(log_file, f'mu shape: {mu.shape}') 

            B, C, H, W = mu.shape
            mu_permuted = mu.permute(1, 0, 2, 3).reshape(C, -1)

            sum_mu += mu_permuted.sum(dim=1)
            sum_squares_mu += (mu_permuted ** 2).sum(dim=1)
            total_pixels_per_channel += mu_permuted.shape[1]

            flat_batch_mu = mu.detach().cpu().view(-1)
            active_batch_mu = flat_batch_mu[torch.abs(flat_batch_mu) > 1e-2]
            if active_batch_mu.numel() > 0:
                sample_size = max(1, int(active_batch_mu.numel() * 0.01))
                perm = torch.randperm(active_batch_mu.numel())[:sample_size]
                sampled_active_points.append(active_batch_mu[perm])
    total_pixels_global = total_pixels_per_channel * C
    
    channel_means = sum_mu / total_pixels_per_channel
    channel_vars = (sum_squares_mu / total_pixels_per_channel) - (channel_means ** 2)
    channel_stds = torch.sqrt(channel_vars).cpu().numpy()
    
    global_mean = sum_mu.sum() / total_pixels_global
    global_var = (sum_squares_mu.sum() / total_pixels_global) - (global_mean ** 2)
    global_std = torch.sqrt(global_var).item()
    
    latent_scale_factor = 1.0 / global_std
    
    print_log(log_file, f'Global standard deviation: {global_std:.6f}, scale factor: {latent_scale_factor:.6f}')
    print_log(log_file, f"Standard deviations for channels 0-3: {channel_stds}")

    final_samples = torch.cat(sampled_active_points, dim=0)
    final_samples_scaled = torch.abs(final_samples * latent_scale_factor).float()

    q_995 = torch.quantile(final_samples_scaled, 0.995).item()
    q_999 = torch.quantile(final_samples_scaled, 0.999).item()
    max_val = torch.max(final_samples_scaled).item()
    
    print_log(log_file, "\n--- Static threshold analysis ---")
    print_log(log_file, f"Maximum absolute scaled latent value: {max_val:.3f}")
    print_log(log_file, f"99.5th percentile of absolute latent values: {q_995:.3f}")
    print_log(log_file, f"99.9th percentile of absolute latent values: {q_999:.3f}")
    
    recommended_threshold = round(q_999, 1)
    print_log(log_file, f"\n=> Recommended settings: latent_scale_factor = {latent_scale_factor:.5f}, static_threshold = {recommended_threshold}")
    log_file.close()


if __name__ == "__main__":
    test_vae()
    print("cal end!!!")
