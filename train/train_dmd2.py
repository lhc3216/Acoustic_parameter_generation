import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import os
from typing import Dict, Tuple

import hydra
import torch
from diffusers import DDPMScheduler
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.sd_dataset import build_train_dataloader
from data.test_sd_dataset import build_dataloader
from models.dmd2.dmd2 import EMA, build_fake_model, build_generator_model, build_teacher_model
from models.dmd2.dmd2_loss import (
    compute_fake_model_loss,
    compute_generator_eval_loss,
    compute_generator_loss,
    sample_terminal_latent,
)
from models.sd.sd import apply_modality_dropout
from models.vae.vae import AcousticVAE
from tools.utils import print_log


def _legacy_apply_modality_dropout(
    ct: torch.Tensor,
    mr: torch.Tensor,
    probs: Dict[str, float],
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

    counts = {
        "both": int(mode_both.sum().item()),
        "mri_only": int(mode_mri_only.sum().item()),
        "ct_only": int(mode_ct_only.sum().item()),
        "uncond": int(mode_uncond.sum().item()),
    }
    return ct * ct_mask, mr * mr_mask, counts


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


def _validate_dropout_probs(cfg: DictConfig) -> None:
    total = (
        cfg.dropout_prob.both
        + cfg.dropout_prob.mri_only
        + cfg.dropout_prob.ct_only
        + cfg.dropout_prob.uncond
    )
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"dropout_prob must sum to 1.0, got {total}")


def _resolve_resume_checkpoint(cfg: DictConfig) -> str:
    resume_from = str(cfg.trainer.get("resume_from", "")).strip()
    if not resume_from:
        return ""
    if os.path.isabs(resume_from):
        return resume_from
    return os.path.join(cfg.paths.save_dir, resume_from)


def _linear_warmup_scale(epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return 1.0
    return min(float(epoch - start_epoch + 1) / float(warmup_epochs), 1.0)


def _active_generator_loss_scales(cfg: DictConfig, epoch: int) -> Tuple[float, float]:
    fake_pretrain_epochs = int(cfg.trainer.get("fake_pretrain_epochs", 0))
    warmup_start_epoch = fake_pretrain_epochs + 1
    dm_scale = _linear_warmup_scale(epoch, warmup_start_epoch, int(cfg.trainer.get("dm_warmup_epochs", 0)))
    gan_scale = _linear_warmup_scale(epoch, warmup_start_epoch, int(cfg.trainer.get("gan_warmup_epochs", 0)))
    return dm_scale, gan_scale


def _encode_target_latent(
    vae: AcousticVAE,
    target: torch.Tensor,
    scale_factor: float,
    static_threshold: float | None = None,
) -> torch.Tensor:
    h = vae.encoder(target)
    mu, _ = torch.chunk(h, 2, dim=1)
    latent = mu * scale_factor
    if static_threshold is not None:
        latent = torch.clamp(latent, -float(static_threshold), float(static_threshold))
    return latent


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def _load_fake_model_resume_state(fake_model: torch.nn.Module, state_dict: dict) -> None:
    """
    Resume checkpoints from both the new D-Head version and older fake-model-only
    checkpoints. Older checkpoints simply leave the new D-Head randomly initialized.
    """
    try:
        fake_model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        incompatible = fake_model.load_state_dict(state_dict, strict=False)
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
        if set(incompatible.missing_keys) != allowed_missing or unexpected:
            raise


@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="dmd2")
def train_dmd2(cfg: DictConfig):
    _validate_dropout_probs(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.trainer.use_amp and device.type == "cuda")

    os.makedirs(cfg.paths.save_dir, exist_ok=True)
    os.makedirs(cfg.paths.tensorboard_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.paths.save_dir, "dmd2_config_resolved.yaml"), resolve=True)

    logger = open(os.path.join(cfg.paths.save_dir, "train_dmd2.log"), "a+", encoding="utf-8")
    writer = SummaryWriter(log_dir=cfg.paths.tensorboard_dir)

    train_cache_dir = Path(cfg.data_root) / "sd_persistent_cache_512_train"
    val_cache_dir = Path(cfg.data_root) / "sd_persistent_cache_512_val"
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_train_dataloader(
        root_dir=cfg.data_root,
        split="train",
        cache_dir=train_cache_dir,
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.num_workers,
    )
    val_loader = build_dataloader(
        root_dir=cfg.data_root,
        split="val",
        cache_dir=val_cache_dir,
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.num_workers,
    )

    vae = _load_vae(cfg, device)
    teacher = build_teacher_model(cfg, device)
    generator = build_generator_model(cfg, device)
    fake_model = build_fake_model(cfg, device)
    ema = EMA(generator, decay=cfg.trainer.ema_decay)
    scheduler = _build_scheduler(cfg)

    g_optimizer = AdamW(
        generator.parameters(),
        lr=cfg.trainer.learning_rate,
        betas=(cfg.trainer.g_beta1, cfg.trainer.g_beta2),
        weight_decay=cfg.trainer.weight_decay,
    )
    f_optimizer = AdamW(
        fake_model.parameters(),
        lr=cfg.trainer.fake_learning_rate,
        betas=(cfg.trainer.f_beta1, cfg.trainer.f_beta2),
        weight_decay=cfg.trainer.fake_weight_decay,
    )
    g_scheduler = CosineAnnealingLR(
        g_optimizer,
        T_max=cfg.trainer.epochs,
        eta_min=cfg.trainer.min_learning_rate,
    )
    f_scheduler = CosineAnnealingLR(
        f_optimizer,
        T_max=cfg.trainer.epochs,
        eta_min=cfg.trainer.fake_min_learning_rate,
    )
    g_scaler = GradScaler(enabled=use_amp)
    f_scaler = GradScaler(enabled=use_amp)

    fake_steps_per_g = int(cfg.trainer.get("fake_steps_per_g", 5))
    if fake_steps_per_g < 1:
        raise ValueError(f"trainer.fake_steps_per_g must be >= 1, got {fake_steps_per_g}")

    best_val = float("inf")
    global_step = 0
    start_epoch = 1

    resume_ckpt = _resolve_resume_checkpoint(cfg)
    if resume_ckpt:
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator_state_dict"], strict=True)
        _load_fake_model_resume_state(fake_model, ckpt["fake_model_state_dict"])
        if "ema_state_dict" in ckpt:
            ema.shadow = ckpt["ema_state_dict"]
        if "g_optimizer_state_dict" in ckpt:
            g_optimizer.load_state_dict(ckpt["g_optimizer_state_dict"])
        if "f_optimizer_state_dict" in ckpt:
            f_optimizer.load_state_dict(ckpt["f_optimizer_state_dict"])
        if "g_scheduler_state_dict" in ckpt:
            g_scheduler.load_state_dict(ckpt["g_scheduler_state_dict"])
        if "f_scheduler_state_dict" in ckpt:
            f_scheduler.load_state_dict(ckpt["f_scheduler_state_dict"])
        if "g_scaler_state_dict" in ckpt and use_amp:
            g_scaler.load_state_dict(ckpt["g_scaler_state_dict"])
        if "f_scaler_state_dict" in ckpt and use_amp:
            f_scaler.load_state_dict(ckpt["f_scaler_state_dict"])
        best_val = ckpt.get("best_val_loss", best_val)
        global_step = ckpt.get("global_step", global_step)
        start_epoch = ckpt.get("epoch", 0) + 1
        print_log(
            logger,
            f"resume from {resume_ckpt}, start_epoch={start_epoch}, global_step={global_step}, best={best_val:.6f}",
        )

    if start_epoch > cfg.trainer.epochs:
        print_log(logger, "configured epochs already finished, skip training")
        writer.close()
        logger.close()
        return

    for epoch in range(start_epoch, cfg.trainer.epochs + 1):
        generator.train()
        fake_model.train()
        dm_loss_scale, gan_loss_scale = _active_generator_loss_scales(cfg, epoch)
        mode_counter = {
            "both": 0,
            "mri_only": 0,
            "ct_only": 0,
            "uncond": 0,
            "ct_weakened": 0,
            "mri_weakened": 0,
        }

        train_fake_total = 0.0
        train_g_total = 0.0
        train_teacher = 0.0
        train_recon = 0.0
        train_dm = 0.0
        train_gan = 0.0
        train_fake_score = 0.0
        train_disc = 0.0
        train_real_logit = 0.0
        train_fake_logit = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"DMD2 Train {epoch}/{cfg.trainer.epochs}")
        for batch in pbar:
            condition = batch["condition"].to(device, non_blocking=True)
            ct, mr = torch.chunk(condition, 2, dim=1)
            target = batch["target"].to(device, non_blocking=True)
            slice_idx = batch["slice_idx"].to(device, non_blocking=True)

            with torch.no_grad():
                clean_z0 = _encode_target_latent(
                    vae,
                    target,
                    cfg.loss.latent_scale_factor,
                    cfg.loss.get("static_threshold", None),
                )

            ct_in, mr_in, counts = apply_modality_dropout(
                ct,
                mr,
                probs={
                    "both": cfg.dropout_prob.both,
                    "mri_only": cfg.dropout_prob.mri_only,
                    "ct_only": cfg.dropout_prob.ct_only,
                    "uncond": cfg.dropout_prob.uncond,
                },
                weaken_probs={
                    "ct": float(cfg.dropout_prob.get("weaken_ct", 0.0)),
                    "mri": float(cfg.dropout_prob.get("weaken_mri", 0.0)),
                },
                weaken_range=(
                    float(cfg.dropout_prob.get("weaken_min_keep", 0.35)),
                    float(cfg.dropout_prob.get("weaken_max_keep", 0.80)),
                ),
            )
            for key in mode_counter:
                mode_counter[key] += counts.get(key, 0)

            batch_fake_metrics = {
                "loss_fake": 0.0,
                "loss_fake_score": 0.0,
                "loss_disc": 0.0,
                "real_logit": 0.0,
                "fake_logit": 0.0,
            }
            for _ in range(fake_steps_per_g):
                with torch.inference_mode():
                    _, _, generated_latent, _ = sample_terminal_latent(
                        model=generator,
                        scheduler=scheduler,
                        latent_shape=clean_z0.shape,
                        ct_cond=ct_in,
                        mr_cond=mr_in,
                        slice_idx=slice_idx,
                    )

                f_optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=use_amp):
                    fake_loss, fake_loss_dict = compute_fake_model_loss(
                        fake_model=fake_model,
                        scheduler=scheduler,
                        generated_latent=generated_latent,
                        real_latent=clean_z0,
                        ct_cond=ct_in,
                        mr_cond=mr_in,
                        slice_idx=slice_idx,
                        cfg=cfg,
                    )
                f_scaler.scale(fake_loss).backward()
                f_scaler.unscale_(f_optimizer)
                torch.nn.utils.clip_grad_norm_(fake_model.parameters(), cfg.trainer.fake_gradient_clip)
                f_scaler.step(f_optimizer)
                f_scaler.update()
                for key in batch_fake_metrics:
                    batch_fake_metrics[key] += fake_loss_dict[key]

            for key in batch_fake_metrics:
                batch_fake_metrics[key] /= fake_steps_per_g

            _set_requires_grad(fake_model, False)
            g_optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                g_out = compute_generator_loss(
                    generator=generator,
                    teacher=teacher,
                    fake_model=fake_model,
                    scheduler=scheduler,
                    clean_z0=clean_z0,
                    ct_cond=ct_in,
                    mr_cond=mr_in,
                    slice_idx=slice_idx,
                    cfg=cfg,
                    dm_loss_scale=dm_loss_scale,
                    gan_loss_scale=gan_loss_scale,
                )
            g_scaler.scale(g_out.total).backward()
            g_scaler.unscale_(g_optimizer)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.trainer.gradient_clip)
            g_scaler.step(g_optimizer)
            g_scaler.update()
            _set_requires_grad(fake_model, True)
            ema.update(generator)

            global_step += 1
            train_batches += 1
            train_fake_total += batch_fake_metrics["loss_fake"]
            train_g_total += g_out.loss_dict["loss_total"]
            train_teacher += g_out.loss_dict["loss_teacher"]
            train_recon += g_out.loss_dict["loss_recon"]
            train_dm += g_out.loss_dict["loss_dm"]
            train_gan += g_out.loss_dict["loss_gan"]
            train_fake_score += batch_fake_metrics["loss_fake_score"]
            train_disc += batch_fake_metrics["loss_disc"]
            train_real_logit += batch_fake_metrics["real_logit"]
            train_fake_logit += batch_fake_metrics["fake_logit"]

            if global_step % cfg.trainer.log_interval == 0:
                writer.add_scalar("train/fake_total", batch_fake_metrics["loss_fake"], global_step)
                writer.add_scalar("train/fake_score", batch_fake_metrics["loss_fake_score"], global_step)
                writer.add_scalar("train/fake_disc", batch_fake_metrics["loss_disc"], global_step)
                writer.add_scalar("train/fake_real_logit", batch_fake_metrics["real_logit"], global_step)
                writer.add_scalar("train/fake_fake_logit", batch_fake_metrics["fake_logit"], global_step)
                writer.add_scalar("train/g_total", g_out.loss_dict["loss_total"], global_step)
                writer.add_scalar("train/loss_teacher", g_out.loss_dict["loss_teacher"], global_step)
                writer.add_scalar("train/loss_recon", g_out.loss_dict["loss_recon"], global_step)
                writer.add_scalar("train/loss_dm", g_out.loss_dict["loss_dm"], global_step)
                writer.add_scalar("train/loss_gan", g_out.loss_dict["loss_gan"], global_step)
                writer.add_scalar("train/gan_logit", g_out.loss_dict["gan_logit"], global_step)
                writer.add_scalar("train/dm_loss_scale", g_out.loss_dict["lambda_dm_active"], global_step)
                writer.add_scalar("train/gan_loss_scale", g_out.loss_dict["lambda_gan_active"], global_step)
                writer.add_scalar("train/dm_grad_norm", g_out.loss_dict["dm_grad_norm"], global_step)
                writer.add_scalar("train/dm_grad_raw_norm", g_out.loss_dict["dm_grad_raw_norm"], global_step)
                writer.add_scalar("train/g_lr", g_optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/f_lr", f_optimizer.param_groups[0]["lr"], global_step)

            pbar.set_postfix(
                {
                    "g": f"{g_out.loss_dict['loss_total']:.4f}",
                    "f": f"{batch_fake_metrics['loss_fake']:.4f}",
                }
            )

        g_scheduler.step()
        f_scheduler.step()

        train_g_total_avg = train_g_total / max(train_batches, 1)
        train_fake_total_avg = train_fake_total / max(train_batches, 1)
        train_teacher_avg = train_teacher / max(train_batches, 1)
        train_recon_avg = train_recon / max(train_batches, 1)
        train_dm_avg = train_dm / max(train_batches, 1)
        train_gan_avg = train_gan / max(train_batches, 1)
        train_fake_score_avg = train_fake_score / max(train_batches, 1)
        train_disc_avg = train_disc / max(train_batches, 1)
        train_real_logit_avg = train_real_logit / max(train_batches, 1)
        train_fake_logit_avg = train_fake_logit / max(train_batches, 1)

        ema.store(generator)
        ema.copy_to(generator)
        generator.eval()
        fake_model.eval()

        val_total = 0.0
        val_teacher = 0.0
        val_recon = 0.0
        val_dm = 0.0
        val_gan = 0.0
        val_batches = 0

        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"DMD2 Val {epoch}/{cfg.trainer.epochs}")):
                condition = batch["condition"].to(device, non_blocking=True)
                ct, mr = torch.chunk(condition, 2, dim=1)
                target = batch["target"].to(device, non_blocking=True)
                slice_idx = batch["slice_idx"].to(device, non_blocking=True)

                clean_z0 = _encode_target_latent(
                    vae,
                    target,
                    cfg.loss.latent_scale_factor,
                    cfg.loss.get("static_threshold", None),
                )
                val_seed = None
                if bool(cfg.eval.get("deterministic", True)):
                    val_seed = int(cfg.eval.get("seed", 3407)) + batch_idx * 9973

                with autocast(device_type=device.type, enabled=use_amp):
                    val_out = compute_generator_eval_loss(
                        generator=generator,
                        teacher=teacher,
                        fake_model=fake_model,
                        scheduler=scheduler,
                        clean_z0=clean_z0,
                        ct_cond=ct,
                        mr_cond=mr,
                        slice_idx=slice_idx,
                        cfg=cfg,
                        dm_loss_scale=dm_loss_scale,
                        gan_loss_scale=gan_loss_scale,
                        random_seed_base=val_seed,
                    )
                val_total += float(val_out["total"].item())
                val_teacher += float(val_out["teacher"].item())
                val_recon += float(val_out["recon"].item())
                val_dm += float(val_out["dm"].item())
                val_gan += float(val_out["gan"].item())
                val_batches += 1

        ema.restore(generator)
        generator.train()

        val_total_avg = val_total / max(val_batches, 1)
        val_teacher_avg = val_teacher / max(val_batches, 1)
        val_recon_avg = val_recon / max(val_batches, 1)
        val_dm_avg = val_dm / max(val_batches, 1)
        val_gan_avg = val_gan / max(val_batches, 1)

        writer.add_scalar("epoch/train_fake_total", train_fake_total_avg, epoch)
        writer.add_scalar("epoch/train_fake_score", train_fake_score_avg, epoch)
        writer.add_scalar("epoch/train_disc", train_disc_avg, epoch)
        writer.add_scalar("epoch/train_real_logit", train_real_logit_avg, epoch)
        writer.add_scalar("epoch/train_fake_logit", train_fake_logit_avg, epoch)
        writer.add_scalar("epoch/train_g_total", train_g_total_avg, epoch)
        writer.add_scalar("epoch/train_teacher", train_teacher_avg, epoch)
        writer.add_scalar("epoch/train_recon", train_recon_avg, epoch)
        writer.add_scalar("epoch/train_dm", train_dm_avg, epoch)
        writer.add_scalar("epoch/train_gan", train_gan_avg, epoch)
        writer.add_scalar("epoch/dm_loss_scale", dm_loss_scale, epoch)
        writer.add_scalar("epoch/gan_loss_scale", gan_loss_scale, epoch)
        writer.add_scalar("epoch/val_total", val_total_avg, epoch)
        writer.add_scalar("epoch/val_teacher", val_teacher_avg, epoch)
        writer.add_scalar("epoch/val_recon", val_recon_avg, epoch)
        writer.add_scalar("epoch/val_dm", val_dm_avg, epoch)
        writer.add_scalar("epoch/val_gan", val_gan_avg, epoch)

        mode_total = max(
            mode_counter["both"]
            + mode_counter["mri_only"]
            + mode_counter["ct_only"]
            + mode_counter["uncond"],
            1,
        )
        writer.add_scalar("dropout_prob_observed/both", mode_counter["both"] / mode_total, epoch)
        writer.add_scalar("dropout_prob_observed/mri_only", mode_counter["mri_only"] / mode_total, epoch)
        writer.add_scalar("dropout_prob_observed/ct_only", mode_counter["ct_only"] / mode_total, epoch)
        writer.add_scalar("dropout_prob_observed/uncond", mode_counter["uncond"] / mode_total, epoch)
        writer.add_scalar("dropout_prob_observed/ct_weakened", mode_counter["ct_weakened"] / mode_total, epoch)
        writer.add_scalar("dropout_prob_observed/mri_weakened", mode_counter["mri_weakened"] / mode_total, epoch)
        if getattr(generator, "control_conditioner", None) is not None:
            writer.add_scalar(
                "control/generator_mri_scale",
                generator.control_conditioner.fusion.mri_scale.detach().float().item(),
                epoch,
            )
        if getattr(fake_model, "control_conditioner", None) is not None:
            writer.add_scalar(
                "control/fake_mri_scale",
                fake_model.control_conditioner.fusion.mri_scale.detach().float().item(),
                epoch,
            )

        is_best = val_total_avg < best_val
        if is_best:
            best_val = val_total_avg

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "generator_state_dict": generator.state_dict(),
            "fake_model_state_dict": fake_model.state_dict(),
            "ema_state_dict": ema.shadow,
            "ema_generator_state_dict": ema.as_state_dict(generator),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "f_optimizer_state_dict": f_optimizer.state_dict(),
            "g_scheduler_state_dict": g_scheduler.state_dict(),
            "f_scheduler_state_dict": f_scheduler.state_dict(),
            "g_scaler_state_dict": g_scaler.state_dict(),
            "f_scaler_state_dict": f_scaler.state_dict(),
            "best_val_loss": best_val,
        }

        torch.save(checkpoint, os.path.join(cfg.paths.save_dir, "dmd2_latest.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(cfg.paths.save_dir, "dmd2_best.pth"))
        if epoch % cfg.trainer.save_interval == 0:
            torch.save(checkpoint, os.path.join(cfg.paths.save_dir, f"dmd2_epoch_{epoch}.pth"))

        print_log(
            logger,
            (
                f"[Epoch {epoch}] train_fake={train_fake_total_avg:.6f}, train_g={train_g_total_avg:.6f}, "
                f"teacher={train_teacher_avg:.6f}, recon={train_recon_avg:.6f}, dm={train_dm_avg:.6f}, "
                f"gan={train_gan_avg:.6f}, disc={train_disc_avg:.6f}, "
                f"dm_scale={dm_loss_scale:.3f}, gan_scale={gan_loss_scale:.3f}, "
                f"real_logit={train_real_logit_avg:.6f}, fake_logit={train_fake_logit_avg:.6f}, "
                f"val={val_total_avg:.6f}, "
                f"val_gan={val_gan_avg:.6f}, modes={mode_counter}"
            ),
        )

    writer.close()
    logger.close()
    print("DMD2 training completed.")


if __name__ == "__main__":
    train_dmd2()
