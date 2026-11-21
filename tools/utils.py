import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import torch

PHYSICS_RANGE = {
    "speed": {"min": 1482.3, "max": 3100.0},
    "density": {"min": 1000.0, "max": 1950.0}
}

def denormalize(tensor, p_min, p_max):
    real = (tensor + 1.0) * 0.5 * (p_max - p_min) + p_min
    real = np.clip(real, p_min, p_max)
    return real

def print_log(logger, message):
    if logger:
        logger.write(str(message) + '\n')
        logger.flush()

def save_residual_heatmaps(gt, pred, save_path, title_prefix="Speed", unit="m/s"):
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError(f"gt and pred must have matching 2D shapes (gt: {gt.shape}, pred: {pred.shape}).")

    gt = np.clip(gt, -1.0, 1.0)
    pred = np.clip(pred, -1.0, 1.0)
    
    if title_prefix == "Speed":
        phys_min = PHYSICS_RANGE["speed"]["min"]
        phys_max = PHYSICS_RANGE["speed"]["max"]
    elif title_prefix == "Density":
        phys_min = PHYSICS_RANGE["density"]["min"]
        phys_max = PHYSICS_RANGE["density"]["max"]
    else:
        raise ValueError(f"Invalid title_prefix: {title_prefix}")

    gt_phys = ((gt + 1.0) / 2.0) * (phys_max - phys_min) + phys_min
    pred_phys = ((pred + 1.0) / 2.0) * (phys_max - phys_min) + phys_min
    
    residual = np.abs(gt_phys - pred_phys)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin, vmax = gt_phys.min(), gt_phys.max()
    
    im0 = axes[0].imshow(gt_phys, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Ground Truth {title_prefix}')
    fig.colorbar(im0, ax=axes[0], label=unit)
    
    im1 = axes[1].imshow(pred_phys, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Reconstructed {title_prefix}')
    fig.colorbar(im1, ax=axes[1], label=unit)
    
    im2 = axes[2].imshow(residual, cmap='jet') 

    err_max = residual.max()
    err_mean = residual.mean()
    
    axes[2].set_title(f'{title_prefix} Residual\n(MAE: {err_mean:.2f} | Max: {err_max:.2f})')
    
    fig.colorbar(im2, ax=axes[2], label=f'Error ({unit})')

    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        raise ValueError("save_path is required to save the image.")
        
    plt.close(fig)


def compute_metrics(original, reconstructed, data_range):
    orig_np = np.asarray(original)
    recon_np = np.asarray(reconstructed)
    
    psnr_val = psnr(orig_np, recon_np, data_range=data_range)

    channel_axis = None
    if orig_np.ndim == 3 and orig_np.shape[0] <= 4:
        channel_axis = 0

    spatial_shape = [d for i, d in enumerate(orig_np.shape) if i != channel_axis]
    min_spatial = int(min(spatial_shape))
    if min_spatial < 7:
        win_size = min_spatial if min_spatial % 2 == 1 else max(min_spatial - 1, 3)
        ssim_val = ssim(
            orig_np,
            recon_np,
            data_range=data_range,
            channel_axis=channel_axis,
            win_size=win_size,
        )
    else:
        ssim_val = ssim(orig_np, recon_np, data_range=data_range, channel_axis=channel_axis)
    
    return psnr_val, ssim_val


def save_images(original, reconstructed, save_path, save_num):

    if torch.is_tensor(original):
        original = original.detach().cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.detach().cpu().numpy()

    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    if original.ndim > 2:
        original = np.squeeze(original)
    if reconstructed.ndim > 2:
        reconstructed = np.squeeze(reconstructed)

    original = ((original + 1) *0.5 * 255.0).astype(np.uint8)
    original = np.clip(original, 0, 255)    
    reconstructed = ((reconstructed + 1) *0.5 * 255.0).astype(np.uint8)
    reconstructed = np.clip(reconstructed, 0, 255)
    original = Image.fromarray(original)
    reconstructed = Image.fromarray(reconstructed)
    original.save(os.path.join(save_path, f"original_{save_num}.png"))
    reconstructed.save(os.path.join(save_path, f"reconstructed_{save_num}.png"))


