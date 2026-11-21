from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler


def _alphas_cumprod_for_timesteps(
    scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    alphas_cumprod = scheduler.alphas_cumprod.to(device=timesteps.device)
    return alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(
        device=reference.device,
        dtype=reference.dtype,
    )


def predict_x0_from_eps(
    scheduler: DDPMScheduler,
    z_t: torch.Tensor,
    eps_pred: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alpha_bar = _alphas_cumprod_for_timesteps(scheduler, timesteps, z_t)
    return (z_t - torch.sqrt(1.0 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)


def _sample_randn(
    shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
    seed: int | None = None,
) -> torch.Tensor:
    if seed is None:
        return torch.randn(shape, device=device, dtype=dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    sample_dtype = dtype if dtype not in (torch.float16, torch.bfloat16) else torch.float32
    return torch.randn(shape, generator=generator, device="cpu", dtype=sample_dtype).to(device=device, dtype=dtype)


def _sample_randint(
    low: int,
    high: int,
    size: tuple[int, ...],
    device: torch.device,
    seed: int | None = None,
) -> torch.Tensor:
    if seed is None:
        return torch.randint(low, high, size, device=device, dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randint(low, high, size, generator=generator, device="cpu", dtype=torch.long).to(device=device)


def sample_terminal_latent(
    model,
    scheduler: DDPMScheduler,
    latent_shape: torch.Size,
    ct_cond: torch.Tensor,
    mr_cond: torch.Tensor,
    slice_idx: torch.Tensor,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one terminal denoising step and return the predicted one-step latent."""
    z_t = _sample_randn(latent_shape, device=ct_cond.device, dtype=ct_cond.dtype, seed=seed)
    timesteps = torch.full(
        (latent_shape[0],),
        int(scheduler.config.num_train_timesteps - 1),
        device=ct_cond.device,
        dtype=torch.long,
    )
    eps_pred = model(
        noisy_latent=z_t,
        timesteps=timesteps,
        ct_images=ct_cond,
        mri_images=mr_cond,
        slice_idx=slice_idx,
    )
    x0_pred = predict_x0_from_eps(scheduler, z_t, eps_pred, timesteps)
    return z_t, eps_pred, x0_pred, timesteps


def _sample_noisy_latent(
    scheduler: DDPMScheduler,
    clean_latent: torch.Tensor,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = clean_latent.shape[0]
    total_steps = int(scheduler.config.num_train_timesteps)
    noise = _sample_randn(
        clean_latent.shape,
        device=clean_latent.device,
        dtype=clean_latent.dtype,
        seed=seed,
    )
    timesteps = _sample_randint(
        1,
        total_steps,
        (bsz,),
        device=clean_latent.device,
        seed=None if seed is None else seed + 1,
    )
    noisy_latent = scheduler.add_noise(clean_latent, noise, timesteps)
    return noisy_latent, noise, timesteps


def _extract_alpha_bar(
    scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    return _alphas_cumprod_for_timesteps(scheduler, timesteps, reference)


def _cfg_loss_value(cfg, key: str, default: float) -> float:
    value = cfg.loss.get(key, default)
    return float(value)


def _truncated_gradient_proxy_loss(source: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Inject a pre-computed gradient into `source` while keeping the forward value
    as a standard positive loss scalar.

    target = stop_grad(source - grad)
    0.5 * ||source - target||^2  ==>  dL/dsource = grad
    """
    target = (source - grad).detach()
    return 0.5 * ((source - target) ** 2).sum() / source.shape[0]


def _clip_per_sample_gradient(grad: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0:
        return grad
    flat_grad = grad.flatten(start_dim=1)
    grad_norm = flat_grad.norm(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (max_norm / grad_norm).clamp(max=1.0)
    view_shape = [grad.shape[0]] + [1] * (grad.ndim - 1)
    return grad * scale.view(*view_shape)


def _hinge_discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    loss_real = F.relu(1.0 - real_logits).mean()
    loss_fake = F.relu(1.0 + fake_logits).mean()
    return 0.5 * (loss_real + loss_fake)


@dataclass
class DMD2GeneratorLossOutput:
    total: torch.Tensor
    x0_pred: torch.Tensor
    loss_dict: Dict[str, float]


def compute_fake_model_loss(
    fake_model,
    scheduler: DDPMScheduler,
    generated_latent: torch.Tensor,
    real_latent: torch.Tensor,
    ct_cond: torch.Tensor,
    mr_cond: torch.Tensor,
    slice_idx: torch.Tensor,
    cfg,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Train the fake model with two signals:
    1. Score regression on generator samples.
    2. A conditional GAN discriminator objective in latent space.
    """
    noisy_latent, noise, timesteps = _sample_noisy_latent(scheduler, generated_latent)
    noise_pred = fake_model(
        noisy_latent=noisy_latent,
        timesteps=timesteps,
        ct_images=ct_cond,
        mri_images=mr_cond,
        slice_idx=slice_idx,
    )
    loss_score = F.mse_loss(noise_pred, noise)

    disc_latent = torch.cat([real_latent.detach(), generated_latent.detach()], dim=0)
    disc_ct = torch.cat([ct_cond, ct_cond], dim=0)
    disc_mr = torch.cat([mr_cond, mr_cond], dim=0)
    disc_slice_idx = torch.cat([slice_idx, slice_idx], dim=0)
    disc_logits = fake_model.discriminate(
        latent=disc_latent,
        ct_images=disc_ct,
        mri_images=disc_mr,
        slice_idx=disc_slice_idx,
    )
    real_logits, fake_logits = disc_logits.chunk(2, dim=0)
    loss_disc = _hinge_discriminator_loss(real_logits=real_logits, fake_logits=fake_logits)

    lambda_fake_score = _cfg_loss_value(cfg, "lambda_fake_score", 1.0)
    lambda_disc = _cfg_loss_value(cfg, "lambda_disc", 1.0)
    total = lambda_fake_score * loss_score + lambda_disc * loss_disc
    return total, {
        "loss_fake": float(total.item()),
        "loss_fake_score": float(loss_score.item()),
        "loss_disc": float(loss_disc.item()),
        "real_logit": float(real_logits.mean().item()),
        "fake_logit": float(fake_logits.mean().item()),
    }


def compute_generator_loss(
    generator,
    teacher,
    fake_model,
    scheduler: DDPMScheduler,
    clean_z0: torch.Tensor,
    ct_cond: torch.Tensor,
    mr_cond: torch.Tensor,
    slice_idx: torch.Tensor,
    cfg,
    dm_loss_scale: float = 1.0,
    gan_loss_scale: float = 1.0,
    random_seed_base: int | None = None,
) -> DMD2GeneratorLossOutput:
    """
    Generator objective:
    1. One-step teacher alignment at the terminal timestep.
    2. Paired latent reconstruction.
    3. DMD2 score-difference guidance via a truncated-gradient proxy loss.
    4. GAN fooling loss from the fake model's external discriminator head.
    """
    terminal_z_t, generator_eps, x0_pred, terminal_t = sample_terminal_latent(
        model=generator,
        scheduler=scheduler,
        latent_shape=clean_z0.shape,
        ct_cond=ct_cond,
        mr_cond=mr_cond,
        slice_idx=slice_idx,
        seed=random_seed_base,
    )

    with torch.no_grad():
        teacher_eps_t = teacher(
            noisy_latent=terminal_z_t,
            timesteps=terminal_t,
            ct_images=ct_cond,
            mri_images=mr_cond,
            slice_idx=slice_idx,
        )

    loss_teacher = F.mse_loss(generator_eps, teacher_eps_t)
    loss_recon = F.smooth_l1_loss(x0_pred, clean_z0)

    lambda_dm = _cfg_loss_value(cfg, "lambda_dm", 1.0) * float(dm_loss_scale)
    lambda_gan = _cfg_loss_value(cfg, "lambda_gan", 0.1) * float(gan_loss_scale)

    loss_dm = clean_z0.new_zeros(())
    grad_dm = torch.zeros_like(x0_pred)
    grad_dm_raw = torch.zeros_like(x0_pred)
    if lambda_dm > 0.0:
        # Explicitly stop the graph before teacher/fake-model evaluation.
        noisy_fake, _, timesteps = _sample_noisy_latent(
            scheduler,
            x0_pred.detach(),
            seed=None if random_seed_base is None else random_seed_base + 1,
        )
        with torch.no_grad():
            teacher_eps_fake = teacher(
                noisy_latent=noisy_fake,
                timesteps=timesteps,
                ct_images=ct_cond,
                mri_images=mr_cond,
                slice_idx=slice_idx,
            )
            fake_eps_fake = fake_model(
                noisy_latent=noisy_fake,
                timesteps=timesteps,
                ct_images=ct_cond,
                mri_images=mr_cond,
                slice_idx=slice_idx,
            )

        # Convert the epsilon gap into an x0-space update direction, then inject it
        # back to the generator with a stop-gradient proxy loss.
        alpha_bar = _extract_alpha_bar(scheduler, timesteps, x0_pred).clamp(min=1e-6, max=1.0)
        dm_weight = torch.sqrt((1.0 - alpha_bar) / alpha_bar)
        grad_dm_raw = dm_weight * (fake_eps_fake - teacher_eps_fake)
        dm_grad_clip = _cfg_loss_value(cfg, "dm_grad_clip", 0.0)
        grad_dm = _clip_per_sample_gradient(grad_dm_raw, dm_grad_clip)
        loss_dm = _truncated_gradient_proxy_loss(x0_pred, grad_dm)

    gan_logits = clean_z0.new_zeros((clean_z0.shape[0],))
    loss_gan = clean_z0.new_zeros(())
    if lambda_gan > 0.0:
        gan_logits = fake_model.discriminate(
            latent=x0_pred,
            ct_images=ct_cond,
            mri_images=mr_cond,
            slice_idx=slice_idx,
        )
        loss_gan = -gan_logits.mean()

    total = (
        _cfg_loss_value(cfg, "lambda_teacher", 1.0) * loss_teacher
        + _cfg_loss_value(cfg, "lambda_recon", 0.5) * loss_recon
        + lambda_dm * loss_dm
        + lambda_gan * loss_gan
    )
    return DMD2GeneratorLossOutput(
        total=total,
        x0_pred=x0_pred,
        loss_dict={
            "loss_total": float(total.item()),
            "loss_teacher": float(loss_teacher.item()),
            "loss_recon": float(loss_recon.item()),
            "loss_dm": float(loss_dm.item()),
            "loss_gan": float(loss_gan.item()),
            "gan_logit": float(gan_logits.mean().item()),
            "lambda_dm_active": float(lambda_dm),
            "lambda_gan_active": float(lambda_gan),
            "dm_grad_norm": float(grad_dm.norm().item() / max(grad_dm.shape[0], 1)),
            "dm_grad_raw_norm": float(grad_dm_raw.norm().item() / max(grad_dm_raw.shape[0], 1)),
        },
    )


def compute_generator_eval_loss(
    generator,
    teacher,
    fake_model,
    scheduler: DDPMScheduler,
    clean_z0: torch.Tensor,
    ct_cond: torch.Tensor,
    mr_cond: torch.Tensor,
    slice_idx: torch.Tensor,
    cfg,
    dm_loss_scale: float = 1.0,
    gan_loss_scale: float = 1.0,
    random_seed_base: int | None = None,
) -> Dict[str, torch.Tensor]:
    out = compute_generator_loss(
        generator=generator,
        teacher=teacher,
        fake_model=fake_model,
        scheduler=scheduler,
        clean_z0=clean_z0,
        ct_cond=ct_cond,
        mr_cond=mr_cond,
        slice_idx=slice_idx,
        cfg=cfg,
        dm_loss_scale=dm_loss_scale,
        gan_loss_scale=gan_loss_scale,
        random_seed_base=random_seed_base,
    )
    return {
        "total": out.total.detach(),
        "teacher": clean_z0.new_tensor(out.loss_dict["loss_teacher"]),
        "recon": clean_z0.new_tensor(out.loss_dict["loss_recon"]),
        "dm": clean_z0.new_tensor(out.loss_dict["loss_dm"]),
        "gan": clean_z0.new_tensor(out.loss_dict["loss_gan"]),
    }
