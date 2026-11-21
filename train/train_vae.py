import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import os
import random
import time
from datetime import datetime

import hydra
import numpy as np
import torch
from monai.data import DataLoader, PersistentDataset, list_data_collate
from monai.utils import set_determinism
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.vae_dataset import (
    get_dynamic_aug_transforms,
    get_static_cache_transforms,
    get_vae_data_dicts_all_slices,
)
from models.vae.vae import AcousticVAE
from models.vae.vae_loss import PhysicsInformedVAELoss, get_dynamic_lambdas
from tools.utils import print_log


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dynamic_collate_fn(dynamic_transforms):
    def _dynamic_collate_fn(batch):
        augmented_batch = [dynamic_transforms(item) for item in batch]
        return list_data_collate(augmented_batch)

    return _dynamic_collate_fn


@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="vae_config")
def train_vae(cfg: DictConfig):
    set_seed(seed=42)

    save_dir = Path(ROOT) / cfg.trainer.save_dir / cfg.task
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = open(save_dir / "train_vae.log", "a+", encoding="utf-8")
    writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
    OmegaConf.save(cfg, save_dir / "vae_config_resolved.yaml", resolve=True)

    train_start_timestamp = time.time()
    train_start_dt = datetime.now()
    print_log(logger, f"Training task: {cfg.task}")
    print_log(logger, f"Training start time: {train_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_loss = float("inf")
    start_epoch = 1
    global_step = 0

    train_cache_dir = os.path.join(cfg.data_root, "vae_persistent_cache_512_train")
    val_cache_dir = os.path.join(cfg.data_root, "vae_persistent_cache_512_val")
    os.makedirs(train_cache_dir, exist_ok=True)
    os.makedirs(val_cache_dir, exist_ok=True)

    train_dicts = get_vae_data_dicts_all_slices(cfg.data_root, split="train")
    val_dicts = get_vae_data_dicts_all_slices(cfg.data_root, split="val")
    static_transforms = get_static_cache_transforms()
    train_dynamic_transforms = get_dynamic_aug_transforms(split_name="train")

    train_dataset = PersistentDataset(
        data=train_dicts,
        transform=static_transforms,
        cache_dir=train_cache_dir,
    )
    val_dataset = PersistentDataset(
        data=val_dicts,
        transform=static_transforms,
        cache_dir=val_cache_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=build_dynamic_collate_fn(train_dynamic_transforms),
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    print_log(logger, f"Training samples: {len(train_dataset)} | Training batches: {len(train_loader)}")
    print_log(logger, f"Validation samples: {len(val_dataset)} | Validation batches: {len(val_loader)}")

    model = AcousticVAE().to(device)
    criterion = PhysicsInformedVAELoss(
        lambda_l1=1.0,
        lambda_grad=0.0,
        lambda_imp=0.0,
        lambda_kl=0.0,
        rho_min=cfg.loss.rho_min,
        rho_max=cfg.loss.rho_max,
        c_min=cfg.loss.c_min,
        c_max=cfg.loss.c_max,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.trainer.learning_rate, weight_decay=cfg.trainer.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.trainer.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=device.type == "cuda")

    resume_from = cfg.trainer.get("resume_from", None)
    if resume_from:
        resume_path = Path(ROOT) / cfg.trainer.save_dir / cfg.task / resume_from
        if resume_path.exists():
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            
            if "criterion_lambdas" in checkpoint:
                lam_dict = checkpoint["criterion_lambdas"]
                criterion.update_lambdas(
                    lam_dict["lambda_l1"],
                    lam_dict["lambda_kl"],
                    lam_dict["lambda_grad"],
                    lam_dict["lambda_imp"],
                )
            
            if "python_random_state" in checkpoint:
                random.setstate(checkpoint["python_random_state"])
            if "numpy_random_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_random_state"])
            if "torch_random_state" in checkpoint:
                torch_state = checkpoint["torch_random_state"]
                if isinstance(torch_state, np.ndarray):
                    torch_state = torch.from_numpy(torch_state).byte()
                elif isinstance(torch_state, torch.Tensor):
                    torch_state = torch_state.cpu().byte()
                else:
                    torch_state = torch.tensor(list(torch_state), dtype=torch.uint8)
                torch.set_rng_state(torch_state)
            if device.type == "cuda" and "torch_cuda_random_state_all" in checkpoint:
                cuda_states = checkpoint["torch_cuda_random_state_all"]
                if cuda_states is not None:
                    fixed_states = []
                    for s in cuda_states:
                        if isinstance(s, np.ndarray):
                            s = torch.from_numpy(s).byte()
                        elif isinstance(s, torch.Tensor):
                            s = s.cpu().byte()
                        else:
                            s = torch.tensor(list(s), dtype=torch.uint8)
                        fixed_states.append(s)
                    torch.cuda.set_rng_state_all(fixed_states)
            
            print_log(
                logger, 
                f"Resuming training from {resume_path}\n"
                f"  -> Completed epochs: {checkpoint['epoch']}, resuming at epoch {start_epoch}\n"
                f"  -> Global step: {global_step}\n"
                f"  -> Best validation loss: {best_val_loss:.6f}\n"
                f"  -> Restored learning rate: {optimizer.param_groups[0]['lr']:.8f}"
            )
        else:
            print_log(logger, f"Resume checkpoint not found: {resume_path}; starting training from scratch.")

    print(f"Starting training on device: {device}")
    for epoch in range(start_epoch, cfg.trainer.epochs + 1):
        lam_l1, lam_kl, lam_grad, lam_imp = get_dynamic_lambdas(epoch, cfg.trainer.epochs)
        if cfg.task == "basic_loss+grad":
            lam_imp = 0
        if cfg.task == "basic_loss":
            lam_imp = 0
            lam_grad = 0
        criterion.update_lambdas(lam_l1, lam_kl, lam_grad, lam_imp)
        print_log(
            logger,
            (
                f"epoch {epoch} loss weights: "
                f"lambda_l1={lam_l1:.6f}, lambda_kl={lam_kl:.6f}, "
                f"lambda_grad={lam_grad:.6f}, lambda_imp={lam_imp:.6f}"
            ),
        )

        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        train_grad = 0.0
        train_kl = 0.0
        train_imp = 0.0
        train_samples = 0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{cfg.trainer.epochs}")
        for batch in pbar:
            images = batch["target"].to(device, non_blocking=True)
            batch_size = images.shape[0]
            train_samples += batch_size

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                recon_images, mu, logvar = model(images)
                total_loss, loss_dict = criterion(
                    recon_images.float(),
                    images.float(),
                    mu.float(),
                    logvar.float(),
                )

            scaler.scale(total_loss).backward() 
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.trainer.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(total_loss.item())
            train_loss += loss_value * batch_size
            train_l1 += float(loss_dict["loss_l1"]) * batch_size
            train_grad += float(loss_dict["loss_grad"]) * batch_size
            train_imp += float(loss_dict["loss_imp"]) * batch_size
            train_kl += float(loss_dict["loss_kl"]) * batch_size
            global_step += 1

            writer.add_scalar("step_loss/train", loss_value, global_step)
            writer.add_scalar("step_lr", optimizer.param_groups[0]["lr"], global_step)
            pbar.set_postfix({"loss": f"{loss_value:.6f}"})

        train_loss_avg = train_loss / train_samples
        train_l1_avg = train_l1 / train_samples
        train_grad_avg = train_grad / train_samples
        train_imp_avg = train_imp / train_samples
        train_kl_avg = train_kl / train_samples

        model.eval()
        val_loss = 0.0
        val_l1 = 0.0
        val_grad = 0.0
        val_kl = 0.0
        val_imp = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                images = batch["target"].to(device, non_blocking=True)
                batch_size = images.shape[0]
                val_samples += batch_size

                with autocast(device_type=device.type, enabled=device.type == "cuda"):
                    recon_images, mu, logvar = model(images)
                    total_loss, loss_dict = criterion(
                        recon_images.float(),
                        images.float(),
                        mu.float(),
                        logvar.float(),
                    )

                val_loss += float(total_loss.item()) * batch_size
                val_l1 += float(loss_dict["loss_l1"]) * batch_size
                val_grad += float(loss_dict["loss_grad"]) * batch_size
                val_kl += float(loss_dict["loss_kl"]) * batch_size
                val_imp += float(loss_dict["loss_imp"]) * batch_size

        val_loss_avg = val_loss / val_samples
        val_l1_avg = val_l1 / val_samples
        val_grad_avg = val_grad / val_samples
        val_kl_avg = val_kl / val_samples
        val_imp_avg = val_imp / val_samples

        scheduler.step()

        writer.add_scalar("loss/train_total", train_loss_avg, epoch)
        writer.add_scalar("loss/val_total", val_loss_avg, epoch)
        writer.add_scalar("loss/train_l1", train_l1_avg, epoch)
        writer.add_scalar("loss/train_grad", train_grad_avg, epoch)
        writer.add_scalar("loss/train_imp", train_imp_avg, epoch)
        writer.add_scalar("loss/train_kl", train_kl_avg, epoch)
        writer.add_scalar("loss/val_l1", val_l1_avg, epoch)
        writer.add_scalar("loss/val_grad", val_grad_avg, epoch)
        writer.add_scalar("loss/val_imp", val_imp_avg, epoch)
        writer.add_scalar("loss/val_kl", val_kl_avg, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print_log(
            logger,
            (
                f"Epoch [{epoch}/{cfg.trainer.epochs}] "
                f"train_loss={train_loss_avg:.6f}, val_loss={val_loss_avg:.6f}, "
                f"train_l1={train_l1_avg:.6f}, val_l1={val_l1_avg:.6f}, "
                f"train_grad={train_grad_avg:.6f}, val_grad={val_grad_avg:.6f}, "
                f"train_imp={train_imp_avg:.6f}, val_imp={val_imp_avg:.6f}, "
                f"train_kl={train_kl_avg:.6f}, val_kl={val_kl_avg:.6f}, "
                f"lr={optimizer.param_groups[0]['lr']:.8f}"
            ),
        )

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "criterion_lambdas": {
                "lambda_l1": float(criterion.lambda_l1),
                "lambda_kl": float(criterion.lambda_kl),
                "lambda_grad": float(criterion.lambda_grad),
                "lambda_imp": float(criterion.lambda_imp),
            },
            "best_val_loss": best_val_loss,
            "val_loss": val_loss_avg,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, save_dir / "vae_best.pth")
            print_log(logger, f"Best model updated: epoch={epoch}, val_loss={best_val_loss:.6f}")

        if epoch % cfg.trainer.save_interval == 0:
            torch.save(checkpoint, save_dir / f"vae_epoch_{epoch}.pth")
            print_log(logger, f"Saved checkpoint: {save_dir / f'vae_epoch_{epoch}.pth'}")

        torch.save(checkpoint, save_dir / "vae_last.pth")

    train_end_timestamp = time.time()
    train_end_dt = datetime.now()
    print_log(logger, f"Training end time: {train_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print_log(logger, f"Total training time (seconds): {train_end_timestamp - train_start_timestamp:.2f}")
    writer.close()
    logger.close()
    print("VAE training completed.")


if __name__ == "__main__":
    train_vae()
