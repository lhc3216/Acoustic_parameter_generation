import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import os
from typing import Any, Dict, Optional
import random
import math
import numpy as np
import hydra
import torch
from diffusers import DDPMScheduler, DDIMScheduler
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm

from data.sd_dataset import build_train_dataloader
from data.test_sd_dataset import build_dataloader
from models.sd.sd import HybridAcousticDiffusion, apply_modality_dropout
from models.sd.sd_loss import DiffusionLoss
from models.vae.vae import AcousticVAE
from monai.utils import set_determinism
from tools.sd_val import validate_a_noise, validate_b_physics
from tools.utils import print_log

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)

def _load_vae(cfg: DictConfig, device: torch.device) -> AcousticVAE:
    vae = AcousticVAE().to(device)
    ckpt = torch.load(cfg.trainer.vae_ckpt, map_location=device, weights_only=False)
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

def _resolve_resume_checkpoint(cfg: DictConfig) -> Optional[Path]:
    resume_raw = cfg.trainer.get("resume_from", None)
    if resume_raw is None:
        return None
    resume_from = str(resume_raw).strip()
    if not resume_from or resume_from.lower() in {"none", "null", "~"}:
        return None
    latest_ckpt = ROOT / cfg.trainer.save_dir / cfg.task / resume_from
    return latest_ckpt if latest_ckpt.exists() else None

def _get_rng_state() -> Dict[str, Any]:
    rng_state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def _set_rng_state(rng_state: Dict[str, Any]) -> None:
    def _to_byte_tensor(state: Any) -> Optional[torch.Tensor]:
        if state is None:
            return None
        if isinstance(state, torch.Tensor):
            return state.detach().cpu().to(dtype=torch.uint8)
        return torch.as_tensor(state, dtype=torch.uint8)

    if not rng_state:
        return
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch_state = _to_byte_tensor(rng_state["torch"])
        if torch_state is not None:
            torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in rng_state:
        cuda_states = [_to_byte_tensor(s) for s in rng_state["cuda"]]
        cuda_states = [s for s in cuda_states if s is not None]
        if cuda_states:
            available_cuda = torch.cuda.device_count()
            if available_cuda <= 0:
                return
            if len(cuda_states) > available_cuda:
                cuda_states = cuda_states[:available_cuda]
            for idx, state in enumerate(cuda_states):
                torch.cuda.set_rng_state(state, device=idx)


def _safe_len(obj) -> Optional[int]:
    try:
        return len(obj)
    except TypeError:
        return None


def _build_checkpoint(
    epoch: int,
    global_step: int,
    best_val_psnr: float,
    best_val_ssim: float,
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    ema_model: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_psnr": best_val_psnr,
        "best_val_ssim": best_val_ssim,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": _get_rng_state(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if ema_model is not None:
        checkpoint["ema_model_state_dict"] = ema_model.state_dict()
    return checkpoint


@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="sd_config")
def train_sd(cfg: DictConfig):
    vae_ckpt_path = Path(str(cfg.trainer.vae_ckpt))
    
    if not vae_ckpt_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt_path}")

    set_seed(seed=42)
    _validate_dropout_probs(cfg)
    grad_accum_steps = int(cfg.trainer.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError(f"trainer.grad_accum_steps must be >= 1, got {grad_accum_steps}")
    save_interval = int(cfg.trainer.get("save_interval", 1))
    if save_interval < 1:
        raise ValueError(f"trainer.save_interval must be >= 1, got {save_interval}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = Path(ROOT) / cfg.trainer.save_dir / cfg.task
    save_dir.mkdir(parents=True, exist_ok=True)
    val_dir = save_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    logger = open(save_dir / "train_sd.log", "a+", encoding="utf-8")
    tb_log_dir = save_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    OmegaConf.save(cfg, save_dir / "sd_config_resolved.yaml", resolve=True)

    train_cache_dir = Path(cfg.data_root) / "sd_persistent_cache_512_train"
    val_cache_dir = Path(cfg.data_root) / "sd_persistent_cache_512_val"
    os.makedirs(train_cache_dir, exist_ok=True)
    os.makedirs(val_cache_dir, exist_ok=True)

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
    train_loader_len = _safe_len(train_loader)
    print_log(logger, f"resume_from: {cfg.trainer.resume_from}")
    print_log(
        logger,
        (
            f"grad_accum_steps={grad_accum_steps}, "
            f"train_loader: {train_loader_len if train_loader_len is not None else 'iterable'}, "
            f"val_loader: {len(val_loader)}"
        ),
    )
    
    vae = _load_vae(cfg, DEVICE)
    model = HybridAcousticDiffusion(
        pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
        latent_channels=cfg.model.latent_channels,
        cond_channels=cfg.model.cond_channels,
        max_z_slices=int(cfg.model.get("max_z_slices", 256)),
    ).to(DEVICE)
    if bool(cfg.model.get("gradient_checkpointing", True)):
        model.unet.enable_gradient_checkpointing()

    ema_decay = 0.9999
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)).to(DEVICE)

    train_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
    )

    val_scheduler = DDIMScheduler(
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

    criterion = DiffusionLoss(
        alphas_cumprod=train_scheduler.alphas_cumprod.float(),
        lambda_noise=cfg.loss.lambda_noise,
        lambda_l1=cfg.loss.lambda_l1,
    ).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
    )
    if train_loader_len is not None:
        updates_per_epoch = max(1, math.ceil(train_loader_len / grad_accum_steps))
    else:
        fallback_steps = int(cfg.trainer.get("steps", 0))
        if fallback_steps <= 0:
            raise ValueError("Iterable train_loader has no length; please set trainer.steps > 0 for scheduler.")
        updates_per_epoch = max(1, math.ceil(fallback_steps / grad_accum_steps))
    total_steps = updates_per_epoch * cfg.trainer.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.trainer.min_learning_rate)
    scaler = GradScaler(enabled=DEVICE.type == "cuda")
    best_val_psnr = float("-inf")
    best_val_ssim = -1.0
    global_step = 0
    start_epoch = 1
    val_psnr = float("-inf")
    val_ssim = -1.0

    resume_requested = str(cfg.trainer.get("resume_from", "")).strip()
    resume_ckpt_path = _resolve_resume_checkpoint(cfg)
    if resume_requested and resume_requested.lower() not in {"none", "null", "~"} and resume_ckpt_path is None:
        print_log(logger, f"resume_from path not found, fallback to fresh training: {resume_requested}")
    
    if resume_ckpt_path:
        print_log(logger, f"resume_ckpt_path: {resume_ckpt_path}")
        resume_ckpt = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
        resume_is_strict = True
        try:
            model.load_state_dict(resume_ckpt["model_state_dict"], strict=True)
        except RuntimeError as e:
            resume_is_strict = False
            incompatible = model.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
            print_log(
                logger,
                (
                    "Partial model resume because the architecture changed. "
                    f"missing_keys={incompatible.missing_keys}, "
                    f"unexpected_keys={incompatible.unexpected_keys}, error={e}"
                ),
            )

        if resume_is_strict and "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        elif not resume_is_strict:
            print_log(logger, "Skip optimizer resume after partial model load.")
        if resume_is_strict and "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if resume_is_strict and "scaler_state_dict" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        if "rng_state" in resume_ckpt:
            try:
                _set_rng_state(resume_ckpt["rng_state"])
            except Exception as e:
                print_log(logger, f"Skip RNG restore due to incompatibility: {e}")
        if "ema_model_state_dict" in resume_ckpt:
            if resume_is_strict:
                ema_model.load_state_dict(resume_ckpt["ema_model_state_dict"], strict=True)
            else:
                ema_model.load_state_dict(resume_ckpt["ema_model_state_dict"], strict=False)
            print_log(logger, "Loaded EMA model weights.")

        if resume_is_strict:
            best_val_psnr = resume_ckpt.get("best_val_psnr", resume_ckpt.get("best_val_loss", best_val_psnr))
            best_val_ssim = resume_ckpt.get("best_val_ssim", best_val_ssim)
            global_step = resume_ckpt.get("global_step", global_step)
            start_epoch = resume_ckpt.get("epoch", 0) + 1
        else:
            best_val_psnr = float("-inf")
            best_val_ssim = -1.0
            global_step = 0
            start_epoch = 1
            print_log(logger, "Treat partial model load as warm-start: reset epoch, global_step, and best metrics.")
        del resume_ckpt
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        print(
            f"Resumed training from {resume_ckpt_path} "
            f"(start_epoch={start_epoch}, global_step={global_step}, best_val_psnr={best_val_psnr:.6f})"
        )

    if start_epoch > cfg.trainer.epochs:
        print(
            f"Start epoch {start_epoch} is greater than configured epochs {cfg.trainer.epochs}. "
            "Nothing to train."
        )
        writer.close()
        logger.close()
        return

    def _handle_post_step(loss_value: float, loss_dict: Dict[str, float]) -> None:
        nonlocal global_step, val_psnr, val_ssim

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        ema_model.update_parameters(model)
        global_step += 1

        if global_step > cfg.trainer.val_start_step:
            if global_step % cfg.trainer.val_a == 0:
                model.eval()
                val_mse = validate_a_noise(
                    val_loader,
                    model,
                    vae,
                    train_scheduler,
                    cfg,
                    DEVICE,
                    use_amp=cfg.trainer.use_amp,
                )
                writer.add_scalar("val/loss_a", val_mse, global_step)
                print_log(logger, f"val_mse: {val_mse:.6f}")
                model.train()
            if global_step % cfg.trainer.val_b == 0:
                model.eval()
                val_psnr, val_ssim = validate_b_physics(
                    val_loader,
                    ema_model,
                    vae,
                    val_scheduler,
                    cfg,
                    DEVICE,
                    use_amp=cfg.trainer.use_amp,
                    save_path=val_dir,
                    steps=global_step,
                    max_save_images=int(cfg.trainer.get("val_b_save_images", 8)),
                )
                writer.add_scalar("val/psnr", val_psnr, global_step)
                writer.add_scalar("val/ssim", val_ssim, global_step)
                print_log(logger, f"val_psnr: {val_psnr:.6f}, val_ssim: {val_ssim:.6f}")
                model.train()
        if global_step % cfg.trainer.log_interval == 0:
            writer.add_scalar("train/loss_total", loss_value, global_step)
            writer.add_scalar("train/loss_noise", loss_dict["loss_noise"], global_step)
            writer.add_scalar("train/loss_l1", loss_dict["loss_l1"], global_step)
            writer.add_scalar("train/avg_time_weight", loss_dict["avg_time_weight"], global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)


    try:
        for epoch in range(start_epoch, cfg.trainer.epochs + 1):
            model.train()
            train_total = 0.0
            train_batches = 0
            mode_counter = {"both": 0, "mri_only": 0, "ct_only": 0, "uncond": 0}
            optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(train_loader, desc=f"Train {epoch}/{cfg.trainer.epochs}", total=train_loader_len)
            accum_counter = 0
            for batch in pbar:
                accum_counter += 1
                should_step = accum_counter == grad_accum_steps
                condition = batch["condition"].to(DEVICE, non_blocking=True)
                ct_in, mr_in = torch.chunk(condition, 2, dim=1)
                target = batch["target"].to(DEVICE, non_blocking=True)
                slice_idx = batch["slice_idx"].to(DEVICE, non_blocking=True)
                with torch.no_grad():
                    h = vae.encoder(target)
                    clean_z0, _ = torch.chunk(h, 2, dim=1)
                    clean_z0 = clean_z0 * cfg.diffusion.latent_scale_factor
                    clean_z0 = torch.clamp(clean_z0, -cfg.diffusion.static_threshold, cfg.diffusion.static_threshold)

                noise = torch.randn_like(clean_z0)
                timesteps = torch.randint(
                    0,
                    cfg.diffusion.num_train_timesteps,
                    (clean_z0.shape[0],),
                    device=DEVICE,
                    dtype=torch.long,
                )
                z_t = train_scheduler.add_noise(clean_z0, noise, timesteps)

                ct_in, mr_in, counts = apply_modality_dropout(
                    ct_in,
                    mr_in,
                    probs={
                        "both": cfg.dropout_prob.both,
                        "mri_only": cfg.dropout_prob.mri_only,
                        "ct_only": cfg.dropout_prob.ct_only,
                        "uncond": cfg.dropout_prob.uncond,
                    },
                )
                for k in mode_counter:
                    mode_counter[k] += counts[k]

                with autocast(device_type=DEVICE.type, enabled=cfg.trainer.use_amp):
                    noise_pred = model(
                        noisy_latent=z_t,
                        timesteps=timesteps,
                        ct_images=ct_in,
                        mri_images=mr_in,
                        slice_idx=slice_idx,
                    )
                    loss, loss_dict = criterion(noise_pred, noise, z_t, timesteps, clean_z0)
                scaler.scale(loss / grad_accum_steps).backward()

                train_total += loss.item()
                train_batches += 1

                if should_step:
                    _handle_post_step(loss.item(), loss_dict)
                    accum_counter = 0

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if accum_counter > 0:
                _handle_post_step(loss.item(), loss_dict)
            train_loss_avg = train_total / max(train_batches, 1)

            checkpoint = _build_checkpoint(
                epoch=epoch,
                global_step=global_step,
                best_val_psnr=best_val_psnr,
                best_val_ssim=best_val_ssim,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema_model=ema_model,
            )

            if val_psnr is not None and val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_val_ssim = val_ssim
                checkpoint["best_val_psnr"] = best_val_psnr
                checkpoint["best_val_ssim"] = best_val_ssim
                torch.save(checkpoint, save_dir / "sd_best.pth")

            if epoch % save_interval == 0:
                torch.save(checkpoint, save_dir / f"sd_epoch_{epoch}.pth")
            torch.save(checkpoint, save_dir / "sd_latest.pth")

        print("SD training completed.")
    except torch.OutOfMemoryError as e:
        print_log(logger, f"CUDA OOM: {e}")
        raise RuntimeError(
            "CUDA OOM during SD training. Please lower trainer.batch_size or raise trainer.grad_accum_steps."
        ) from e
    finally:
        writer.close()
        logger.close()

if __name__ == "__main__":
    train_sd()
