from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


MODE_BOTH = 0
MODE_CT_ONLY = 1
MODE_MRI_ONLY = 2
MODE_UNCOND = 3


class ModalityEncoder(nn.Module):
    """
    Lightweight encoder for a single CT or MRI slice.

    The three stride-2 convolutions map 512x512 CT/MRI slices to the 64x64 VAE
    latent resolution. GroupNorm keeps CT and MRI feature magnitudes closer
    before fusion, which reduces the chance that CT dominates only because its
    feature scale is larger.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.encoder[0].in_channels:
            raise ValueError(f"Expected [B, {self.encoder[0].in_channels}, H, W] modality input, got {tuple(x.shape)}.")
        return self.encoder(x)


class CrossModalityAttention(nn.Module):
    """
    CT/MRI fusion by concatenation and projection.

    The old softmax fusion forced CT and MRI to compete for a sum-to-one weight,
    which made MRI easy to suppress when CT features were stronger. This module
    learns a fused condition directly from both feature tensors and does not
    explicitly output modality ratios.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
        )

    def forward(self, feat_ct: Optional[torch.Tensor], feat_mri: Optional[torch.Tensor]) -> torch.Tensor:
        if feat_ct is None:
            return feat_mri
        if feat_mri is None:
            return feat_ct
        return self.fusion_proj(torch.cat([feat_ct, feat_mri], dim=1))


class HybridAcousticDiffusion(nn.Module):
    """
    Hybrid acoustic diffusion model.

    CT/MRI are encoded to latent-resolution condition maps and concatenated with
    the noisy acoustic latent. The U-Net also receives a slice-position token and
    a mode token: both, CT-only, MRI-only, or unconditioned.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        latent_channels: int = 4,
        cond_channels: int = 4,
        max_z_slices: int = 256,
    ):
        super().__init__()
        self.ct_encoder = ModalityEncoder(in_channels=1, out_channels=cond_channels)
        self.mri_encoder = ModalityEncoder(in_channels=1, out_channels=cond_channels)
        self.fusion_module = CrossModalityAttention(channels=cond_channels)
        self.cond_channels = cond_channels
        if max_z_slices <= 0:
            raise ValueError("max_z_slices must be positive.")
        self.max_z_slices = max_z_slices

        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
        )
        cross_attention_dim = self.unet.config.cross_attention_dim

        self.z_position_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, cross_attention_dim),
        )
        # Learn one task token for each input mode. This makes modality dropout
        # explicit instead of asking the model to infer the task only from
        # zero-filled CT/MRI tensors.
        self.mode_embedding = nn.Embedding(4, cross_attention_dim)

        self._modify_unet_input(latent_channels, cond_channels)

    def _modify_unet_input(self, latent_channels: int, cond_channels: int) -> None:
        """
        Expand the SD U-Net input from latent channels to latent+condition.

        The original latent channels inherit pretrained SD weights. The new
        condition channels start at zero so early training behaves like the
        pretrained denoiser before learning to use CT/MRI conditions.
        """
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            latent_channels + cond_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )

        with torch.no_grad():
            new_conv.weight[:, :latent_channels, :, :] = old_conv.weight.clone()
            nn.init.zeros_(new_conv.weight[:, latent_channels:, :, :])
            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.unet.conv_in = new_conv

    @staticmethod
    def _presence_mask(images: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if images is None:
            return torch.zeros(batch_size, 1, 1, 1, device=device, dtype=torch.bool)
        return images.abs().sum(dim=(1, 2, 3), keepdim=True) > 0

    @staticmethod
    def _infer_mode_id(ct_present: torch.Tensor, mri_present: torch.Tensor) -> torch.Tensor:
        """
        Infer per-sample mode id from modality presence.

        0=both, 1=ct_only, 2=mri_only, 3=uncond.
        """
        both = (ct_present & mri_present).view(-1)
        ct_only = (ct_present & (~mri_present)).view(-1)
        mri_only = ((~ct_present) & mri_present).view(-1)

        mode_id = torch.full((both.shape[0],), MODE_UNCOND, device=both.device, dtype=torch.long)
        mode_id[both] = MODE_BOTH
        mode_id[ct_only] = MODE_CT_ONLY
        mode_id[mri_only] = MODE_MRI_ONLY
        return mode_id

    def _build_fused_condition(
        self,
        noisy_latent: torch.Tensor,
        ct_images: Optional[torch.Tensor],
        mri_images: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, latent_h, latent_w = noisy_latent.shape
        device = noisy_latent.device
        dtype = noisy_latent.dtype

        feat_ct = self.ct_encoder(ct_images) if ct_images is not None else None
        feat_mri = self.mri_encoder(mri_images) if mri_images is not None else None

        ct_present = self._presence_mask(ct_images, batch_size, device)
        mri_present = self._presence_mask(mri_images, batch_size, device)

        if feat_ct is None:
            feat_ct = torch.zeros(batch_size, self.cond_channels, latent_h, latent_w, device=device, dtype=dtype)
        if feat_mri is None:
            feat_mri = torch.zeros(batch_size, self.cond_channels, latent_h, latent_w, device=device, dtype=dtype)

        feat_ct = feat_ct * ct_present.to(feat_ct.dtype)
        feat_mri = feat_mri * mri_present.to(feat_mri.dtype)

        fused_cond = torch.zeros_like(feat_ct)
        both_mask = (ct_present & mri_present).view(-1)
        ct_only_mask = (ct_present & (~mri_present)).view(-1)
        mri_only_mask = ((~ct_present) & mri_present).view(-1)

        if both_mask.any():
            fused_both = self.fusion_module(feat_ct[both_mask], feat_mri[both_mask])
            fused_cond[both_mask] = fused_both.type_as(fused_cond)
        if ct_only_mask.any():
            fused_cond[ct_only_mask] = feat_ct[ct_only_mask].type_as(fused_cond)
        if mri_only_mask.any():
            fused_cond[mri_only_mask] = feat_mri[mri_only_mask].type_as(fused_cond)

        return fused_cond, ct_present, mri_present

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        ct_images: Optional[torch.Tensor] = None,
        mri_images: Optional[torch.Tensor] = None,
        slice_idx: Optional[torch.Tensor] = None,
        mode_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = noisy_latent.shape[0]

        fused_cond, ct_present, mri_present = self._build_fused_condition(noisy_latent, ct_images, mri_images)
        net_input = torch.cat([noisy_latent, fused_cond], dim=1)

        if slice_idx is None:
            slice_idx = torch.zeros(batch_size, device=noisy_latent.device, dtype=torch.long)
        if mode_id is None:
            mode_id = self._infer_mode_id(ct_present, mri_present)
        else:
            mode_id = mode_id.to(device=noisy_latent.device, dtype=torch.long).view(-1)

        z_norm = slice_idx.float().view(-1, 1) / float(self.max_z_slices)
        z_hidden_states = self.z_position_encoder(z_norm)
        mode_hidden_states = self.mode_embedding(mode_id)
        encoder_hidden_states = (z_hidden_states + mode_hidden_states).unsqueeze(1)

        return self.unet(
            net_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample


def apply_modality_dropout(
    ct: torch.Tensor,
    mr: torch.Tensor,
    probs: Dict[str, float],
    weaken_probs: Optional[Dict[str, float]] = None,
    weaken_range: Tuple[float, float] = (0.35, 0.80),
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    if ct.shape != mr.shape:
        raise ValueError(f"CT/MRI shape mismatch: {ct.shape} vs {mr.shape}")

    batch_size = ct.shape[0]
    rnd = torch.rand(batch_size, device=ct.device)

    p_both = probs["both"]
    p_mri_only = p_both + probs["mri_only"]
    p_ct_only = p_mri_only + probs["ct_only"]

    mode_both = rnd < p_both
    mode_mri_only = (rnd >= p_both) & (rnd < p_mri_only)
    mode_ct_only = (rnd >= p_mri_only) & (rnd < p_ct_only)
    mode_uncond = rnd >= p_ct_only

    ct_mask = torch.ones(batch_size, 1, 1, 1, device=ct.device, dtype=ct.dtype)
    mr_mask = torch.ones(batch_size, 1, 1, 1, device=mr.device, dtype=mr.dtype)
    ct_mask[mode_mri_only | mode_uncond] = 0.0
    mr_mask[mode_ct_only | mode_uncond] = 0.0

    ct_dropped = ct * ct_mask
    mr_dropped = mr * mr_mask

    counts = {
        "both": int(mode_both.sum().item()),
        "mri_only": int(mode_mri_only.sum().item()),
        "ct_only": int(mode_ct_only.sum().item()),
        "uncond": int(mode_uncond.sum().item()),
    }
    if weaken_probs is not None:
        low, high = weaken_range
        if not 0 < low <= high <= 1:
            raise ValueError("weaken_range must satisfy 0 < low <= high <= 1.")
        for name, values, present in (
            ("ct", ct_dropped, mode_both | mode_ct_only),
            ("mri", mr_dropped, mode_both | mode_mri_only),
        ):
            probability = float(weaken_probs.get(name, 0.0))
            if not 0 <= probability <= 1:
                raise ValueError(f"Invalid {name} weakening probability: {probability}")
            weakened = present & (torch.rand(batch_size, device=values.device) < probability)
            keep = torch.empty(batch_size, 1, 1, 1, device=values.device, dtype=values.dtype).uniform_(low, high)
            values.mul_(torch.where(weakened.view(-1, 1, 1, 1), keep, torch.ones_like(keep)))
            counts[f"{name}_weakened"] = int(weakened.sum().item())
    return ct_dropped, mr_dropped, counts
