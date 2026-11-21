import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(
        self,
        alphas_cumprod,
        lambda_noise=1.0,
        lambda_l1=0.1,
    ):
        super().__init__()
        self.lambda_noise = lambda_noise
        self.lambda_l1 = lambda_l1
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(self, noise_pred, noise_gt, z_t, timesteps, clean_z0):
        # 1. Standard epsilon-prediction loss.
        loss_noise = F.mse_loss(noise_pred, noise_gt)

        # 2. Recover x0/z0 from the predicted epsilon.
        # Clamp avoids numerical issues when alpha_bar is extremely small.
        alpha_bar_t = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1).clamp_min(1e-6)
        z0_pred = (z_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

        # 3. Per-sample latent reconstruction loss.
        loss_l1_batch = F.l1_loss(z0_pred, clean_z0, reduction="none").mean(dim=(1, 2, 3))

        # 4. Timestep weighting must stay in shape [B]; otherwise broadcasting
        # with loss_l1_batch will silently produce incorrect weights.
        alpha_bar_batch = alpha_bar_t.view(-1)
        snr = alpha_bar_batch / (1 - alpha_bar_batch + 1e-6)
        weight_b = torch.clamp(snr, max=5.0)
        weight_b = weight_b / weight_b.mean().clamp_min(1e-6)
        weighted_loss_l1 = (weight_b * loss_l1_batch).mean()

        total_loss = self.lambda_noise * loss_noise + self.lambda_l1 * weighted_loss_l1
        loss_dict = {
            "loss_noise": self.lambda_noise * loss_noise.item(),
            "loss_l1": (self.lambda_l1 * weighted_loss_l1).item(),
            "avg_time_weight": weight_b.mean().item(),
        }
        return total_loss, loss_dict
