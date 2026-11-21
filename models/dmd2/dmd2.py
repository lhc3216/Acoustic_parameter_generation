from typing import Optional

import torch
import torch.nn as nn

from models.sd.sd import HybridAcousticDiffusion


def _normalize_state_dict(state_dict: dict) -> dict:
    """Normalize DDP/EMA checkpoint keys."""
    normalized = {}
    for key, value in state_dict.items():
        if key == "n_averaged":
            continue
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def _load_pretrained_diffusion_state(cfg, device: torch.device) -> dict:
    """Prefer EMA weights, and fall back to raw model weights."""
    ckpt = torch.load(cfg.paths.teacher_sd_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt.get("ema_model_state_dict") or ckpt.get("model_state_dict") or ckpt
    return _normalize_state_dict(state_dict)


def _build_base_model(cfg, device: torch.device) -> HybridAcousticDiffusion:
    model = HybridAcousticDiffusion(
        pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
        latent_channels=int(cfg.model.latent_channels),
        cond_channels=int(cfg.model.cond_channels),
        max_z_slices=int(cfg.model.get("max_z_slices", 256)),
    ).to(device)
    if bool(cfg.model.get("gradient_checkpointing", False)) and hasattr(model.unet, "enable_gradient_checkpointing"):
        model.unet.enable_gradient_checkpointing()
    return model


def _extract_hook_tensor(output) -> torch.Tensor:
    """
    Normalize different Diffusers block outputs into a plain feature tensor.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if hasattr(output, "sample") and isinstance(output.sample, torch.Tensor):
        return output.sample
    raise TypeError(f"Unsupported hook output type: {type(output)}")


class SharedFeatureDiscriminatorHead(nn.Module):
    """
    Lightweight readout head on top of shared U-Net features.

    This is intentionally only a projection/classification head. The actual
    discriminative representation comes from the hooked diffusion backbone,
    which matches the parameter-sharing spirit of DMD2.
    """

    def __init__(self, feature_channels: list[int], max_z_slices: int):
        super().__init__()
        self.max_z_slices = max_z_slices
        self.feature_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(max(1, min(32, channels // 8)), channels),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(channels, 256),
                )
                for channels in feature_channels
            ]
        )
        merged_dim = 256 * len(feature_channels)
        self.slice_mlp = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Linear(merged_dim + 256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features: list[torch.Tensor], slice_idx: torch.Tensor) -> torch.Tensor:
        if len(features) != len(self.feature_projs):
            raise ValueError(f"Expected {len(self.feature_projs)} hooked features, got {len(features)}")

        pooled = [proj(feature) for proj, feature in zip(self.feature_projs, features)]
        merged = torch.cat(pooled, dim=1)
        z_norm = slice_idx.float().view(-1, 1) / float(self.max_z_slices)
        slice_embed = self.slice_mlp(z_norm)
        logits = self.classifier(torch.cat([merged, slice_embed], dim=1))
        return logits.view(-1)


class FakeModelWithDiscriminator(HybridAcousticDiffusion):
    """
    Fake model used by DMD2.

    It keeps the original diffusion backbone for score regression and reads out
    discriminator logits from hooked internal U-Net features, so the fake model
    and discriminator truly share the same Diffusers backbone.
    """

    def __init__(
        self,
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        latent_channels: int = 4,
        cond_channels: int = 4,
        max_z_slices: int = 256,
    ):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            latent_channels=latent_channels,
            cond_channels=cond_channels,
            max_z_slices=max_z_slices,
        )
        self.latent_channels = latent_channels
        self.cond_channels = cond_channels
        block_channels = list(self.unet.config.block_out_channels)
        shared_feature_channels = [block_channels[-1], block_channels[0]]
        self.d_head = SharedFeatureDiscriminatorHead(
            feature_channels=shared_feature_channels,
            max_z_slices=max_z_slices,
        )
        self._hooked_feature_names = ("mid_block", "up_block_last")
        self._hooked_features: dict[str, torch.Tensor] = {}
        self._capture_enabled = False
        self._hook_handles = self._register_shared_feature_hooks()

    def _register_shared_feature_hooks(self) -> list[torch.utils.hooks.RemovableHandle]:
        handles = []

        def _make_hook(name: str):
            def _hook(_module, _inputs, output):
                if self._capture_enabled:
                    self._hooked_features[name] = _extract_hook_tensor(output)

            return _hook

        handles.append(self.unet.mid_block.register_forward_hook(_make_hook("mid_block")))
        handles.append(self.unet.up_blocks[-1].register_forward_hook(_make_hook("up_block_last")))
        return handles

    def _clear_hook_cache(self) -> None:
        self._hooked_features.clear()

    def _run_unet_with_feature_capture(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        ct_images: Optional[torch.Tensor] = None,
        mri_images: Optional[torch.Tensor] = None,
        slice_idx: Optional[torch.Tensor] = None,
        mode_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._clear_hook_cache()
        self._capture_enabled = True
        try:
            noise_pred = super().forward(
                noisy_latent=noisy_latent,
                timesteps=timesteps,
                ct_images=ct_images,
                mri_images=mri_images,
                slice_idx=slice_idx,
                mode_id=mode_id,
            )
        finally:
            self._capture_enabled = False
        return noise_pred

    def _get_hooked_features(self) -> list[torch.Tensor]:
        missing = [name for name in self._hooked_feature_names if name not in self._hooked_features]
        if missing:
            raise RuntimeError(f"Missing hooked U-Net features: {missing}")
        return [self._hooked_features[name] for name in self._hooked_feature_names]

    def forward(self, noisy_latent, timesteps, ct_images=None, mri_images=None, slice_idx=None, mode_id=None):
        return super().forward(
            noisy_latent=noisy_latent,
            timesteps=timesteps,
            ct_images=ct_images,
            mri_images=mri_images,
            slice_idx=slice_idx,
            mode_id=mode_id,
        )

    def _build_fused_condition(
        self,
        reference_latent: torch.Tensor,
        ct_images: Optional[torch.Tensor] = None,
        mri_images: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Keep compatibility with HybridAcousticDiffusion.forward.

        The parent forward expects this method to return the fused condition and
        the per-sample modality-presence masks used to infer the mode token.
        """
        feat_ct = self.ct_encoder(ct_images) if ct_images is not None else None
        feat_mri = self.mri_encoder(mri_images) if mri_images is not None else None

        batch_size, _, latent_h, latent_w = reference_latent.shape
        device = reference_latent.device
        dtype = reference_latent.dtype

        if ct_images is not None:
            ct_present = ct_images.abs().sum(dim=(1, 2, 3), keepdim=True) > 0
        else:
            ct_present = torch.zeros(batch_size, 1, 1, 1, device=device, dtype=torch.bool)
        if mri_images is not None:
            mri_present = mri_images.abs().sum(dim=(1, 2, 3), keepdim=True) > 0
        else:
            mri_present = torch.zeros(batch_size, 1, 1, 1, device=device, dtype=torch.bool)

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

    def discriminate(
        self,
        latent: torch.Tensor,
        ct_images: Optional[torch.Tensor] = None,
        mri_images: Optional[torch.Tensor] = None,
        slice_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if slice_idx is None:
            slice_idx = torch.zeros(latent.shape[0], device=latent.device, dtype=torch.long)
        # Reuse the same diffusion backbone at t=0 and read discriminator logits
        # from hooked internal features instead of running a separate discriminator.
        disc_timesteps = torch.zeros(latent.shape[0], device=latent.device, dtype=torch.long)
        _ = self._run_unet_with_feature_capture(
            noisy_latent=latent,
            timesteps=disc_timesteps,
            ct_images=ct_images,
            mri_images=mri_images,
            slice_idx=slice_idx,
        )
        logits = self.d_head(self._get_hooked_features(), slice_idx=slice_idx)
        self._clear_hook_cache()
        return logits


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        for name in self.shadow:
            self.shadow[name].mul_(self.decay).add_(current[name], alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        current = model.state_dict()
        for name in self.shadow:
            current[name].copy_(self.shadow[name])

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self.backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.backup:
            return
        current = model.state_dict()
        for name, value in self.backup.items():
            current[name].copy_(value)
        self.backup = {}

    @torch.no_grad()
    def as_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        ema_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        for name, value in self.shadow.items():
            ema_state[name] = value.detach().clone()
        return ema_state


def _load_fake_model_backbone(model: FakeModelWithDiscriminator, cfg, device: torch.device) -> None:
    """
    Load the pretrained diffusion backbone while intentionally leaving the new
    discriminator head randomly initialized.
    """
    state_dict = _load_pretrained_diffusion_state(cfg, device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        "d_head.",
        "ct_encoder.encoder.5.",
        "mri_encoder.encoder.5.",
        "fusion_module.fusion_proj.",
        "mode_embedding.",
        "control_conditioner.",
    )
    allowed_unexpected_prefixes = ("fusion_module.attention_net.", "control_conditioner.")
    allowed_missing = {
        key for key in incompatible.missing_keys
        if key.startswith(allowed_missing_prefixes)
    }
    unexpected = [
        key for key in incompatible.unexpected_keys
        if not key.startswith(allowed_unexpected_prefixes)
    ]
    if set(incompatible.missing_keys) != allowed_missing:
        raise RuntimeError(f"Unexpected missing keys when loading fake model: {incompatible.missing_keys}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading fake model: {unexpected}")


def _load_diffusion_backbone(model: HybridAcousticDiffusion, cfg, device: torch.device, strict: bool = True) -> None:
    state_dict = _load_pretrained_diffusion_state(cfg, device)
    if strict:
        model.load_state_dict(state_dict, strict=True)
        return

    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        "ct_encoder.encoder.5.",
        "mri_encoder.encoder.5.",
        "fusion_module.fusion_proj.",
        "mode_embedding.",
        "control_conditioner.",
    )
    allowed_unexpected_prefixes = ("fusion_module.attention_net.", "control_conditioner.")
    unexpected = [
        key for key in incompatible.unexpected_keys
        if not key.startswith(allowed_unexpected_prefixes)
    ]
    missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Unexpected incompatibility when loading diffusion backbone: "
            f"missing={missing}, unexpected={unexpected}"
        )


def build_teacher_model(cfg, device: torch.device) -> HybridAcousticDiffusion:
    model = _build_base_model(cfg, device)
    _load_diffusion_backbone(model, cfg, device, strict=bool(cfg.get("strict_teacher_load", False)))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_generator_model(cfg, device: torch.device) -> HybridAcousticDiffusion:
    model = _build_base_model(cfg, device)
    _load_diffusion_backbone(model, cfg, device, strict=bool(cfg.get("strict_teacher_load", False)))
    return model


def build_fake_model(cfg, device: torch.device) -> FakeModelWithDiscriminator:
    model = FakeModelWithDiscriminator(
        pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
        latent_channels=int(cfg.model.latent_channels),
        cond_channels=int(cfg.model.cond_channels),
        max_z_slices=int(cfg.model.get("max_z_slices", 256)),
    ).to(device)
    if bool(cfg.model.get("gradient_checkpointing", False)) and hasattr(model.unet, "enable_gradient_checkpointing"):
        model.unet.enable_gradient_checkpointing()
    _load_fake_model_backbone(model, cfg, device)
    return model
