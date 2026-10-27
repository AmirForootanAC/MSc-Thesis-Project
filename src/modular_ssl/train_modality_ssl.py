"""
Decoupled Single-Modality Multi-View SSL Trainer.

Usage:
    # 1. Train X-ray SSL branch (Fast: only 2,972 train visits, ~1.89 views/visit)
    python -m src.modular_ssl.train_modality_ssl --modality radiograph

    # 2. Train Intraoral Photo SSL branch (6,126 train visits, multi-view MIL)
    python -m src.modular_ssl.train_modality_ssl --modality photograph
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.ssl_v2.cloud_sync import KagglePersistenceManager
from . import config
from .dataset import SingleModalitySSLDataset, single_modality_ssl_collate
from .models import SingleVisualModalitySSLModel, DecoupledModalitySSLLoss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def build_optimizer(model: SingleVisualModalitySSLModel) -> AdamW:
    vis_params = [
        p for p in model.vision_encoder.parameters() if p.requires_grad
    ]
    txt_params = [
        p for p in model.text_encoder.parameters() if p.requires_grad
    ]
    proj_params = list(model.vision_proj.parameters()) + list(
        model.text_proj.parameters()
    )

    return AdamW(
        [
            {
                "name": "vision",
                "params": vis_params,
                "lr": config.SSL_VISION_LR,
                "base_lr": config.SSL_VISION_LR,
            },
            {
                "name": "text",
                "params": txt_params,
                "lr": config.SSL_TEXT_LR,
                "base_lr": config.SSL_TEXT_LR,
            },
            {
                "name": "proj",
                "params": proj_params,
                "lr": config.SSL_HEAD_LR,
                "base_lr": config.SSL_HEAD_LR,
            },
        ],
        weight_decay=config.SSL_WEIGHT_DECAY,
    )


@torch.no_grad()
def evaluate_ssl(
    runner: nn.Module,
    criterion: DecoupledModalitySSLLoss,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    runner.eval()
    total_loss = 0.0
    steps = 0
    pos_sims, neg_sims = [], []

    for batch in tqdm(loader, desc="[Val SSL]", leave=False):
        imgs = batch["padded_images"].to(device, non_blocking=True)
        vmask = batch["view_mask"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        amask = batch["attention_mask"].to(device, non_blocking=True)
        has_txt = batch["has_text"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
            out = runner(imgs, vmask, ids, amask)
            loss, _ = criterion(out, vmask, has_txt)

        if loss.item() > 0:
            total_loss += loss.item()
            steps += 1

        if has_txt.sum() >= 2:
                    zv = out["z_vis"][has_txt]
                    zt = out["z_txt"][has_txt]
                    sim = zv @ zt.T
                    # Ensure tensors are detached to prevent memory leak
                    pos_sims.append(torch.diag(sim).mean().detach().item())
                    eye = torch.eye(sim.size(0), dtype=torch.bool, device=device)
                    neg_sims.append(sim[~eye].mean().detach().item())

    pos_m = float(np.mean(pos_sims)) if pos_sims else 0.0
    neg_m = float(np.mean(neg_sims)) if neg_sims else 0.0
    return {
        "val_loss": total_loss / max(steps, 1),
        "pos_sim": pos_m,
        "neg_sim": neg_m,
        "gap": pos_m - neg_m,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Decoupled Single-Modality SSL Model."
    )
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["photograph", "radiograph"],
        help="Visual modality to pretrain independently.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of epochs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    modality = args.modality
    total_epochs = args.epochs or (
        config.SSL_EPOCHS_PHOTO
        if modality == "photograph"
        else config.SSL_EPOCHS_RADIO
    )

    out_dir = (
        config.PHOTO_SSL_DIR
        if modality == "photograph"
        else config.RADIO_SSL_DIR
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sync_mgr = KagglePersistenceManager(config.PROJECT_ROOT)
    sync_mgr.results_dir = config.OUTPUT_ROOT
    sync_mgr.restore_all_from_backup()

    print("=" * 75)
    print(f"MODULAR MULTI-VIEW SSL PRETRAINING — {modality.upper()}")
    print("=" * 75)
    print(f"Device      : {device} (GPUs: {config.NUM_GPUS})")
    print(f"Image Root  : {config.IMAGE_ROOT}")
    print(f"Batch Size  : {config.SSL_BATCH_SIZE} | Epochs: {total_epochs}")
    print("=" * 75)

    train_ds = SingleModalitySSLDataset(
        modality=modality, split="train", is_train=True
    )
    val_ds = SingleModalitySSLDataset(
        modality=modality, split="validation", is_train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.SSL_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(config.NUM_WORKERS > 0),
        collate_fn=single_modality_ssl_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.SSL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(config.NUM_WORKERS > 0),
        collate_fn=single_modality_ssl_collate,
    )

    base_model = SingleVisualModalitySSLModel(modality=modality).to(device)
    runner = (
        nn.DataParallel(base_model)
        if config.USE_MULTI_GPU
        else base_model
    )

    criterion = DecoupledModalitySSLLoss().to(device)
    optimizer = build_optimizer(base_model)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - config.SSL_WARMUP_EPOCHS, 1),
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=config.USE_AMP)

    best_path = out_dir / f"best_{modality}_ssl.pt"
    latest_path = out_dir / f"latest_{modality}_ssl.pt"
    history_path = out_dir / f"{modality}_ssl_history.json"

    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    if latest_path.exists():
        print(f"[Resume] Loading checkpoint from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt["best_val_loss"])
        history = ckpt["history"]

    for epoch in range(start_epoch, total_epochs + 1):
        if epoch <= config.SSL_WARMUP_EPOCHS:
            w = epoch / config.SSL_WARMUP_EPOCHS
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * w

        runner.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss_sum = 0.0
        valid_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"[{modality.upper()} SSL] Epoch {epoch:02d}/{total_epochs}",
        )

        for batch in pbar:
            imgs = batch["padded_images"].to(device, non_blocking=True)
            vmask = batch["view_mask"].to(device, non_blocking=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
            amask = batch["attention_mask"].to(device, non_blocking=True)
            has_txt = batch["has_text"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type, enabled=config.USE_AMP
            ):
                out = runner(imgs, vmask, ids, amask)
                loss, loss_dict = criterion(out, vmask, has_txt)

            if not loss_dict:
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            train_loss_sum += loss.item()
            valid_steps += 1
            pbar.set_postfix(loss=f"{train_loss_sum / valid_steps:.4f}")

        if epoch > config.SSL_WARMUP_EPOCHS:
            scheduler.step()

        train_loss = train_loss_sum / max(valid_steps, 1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        val_stats = evaluate_ssl(runner, criterion, val_loader, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_stats,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_stats['val_loss']:.4f} | "
            f"CosSim Gap: +{val_stats['gap']:.4f} "
            f"(Pos: {val_stats['pos_sim']:.3f}, Neg: {val_stats['neg_sim']:.3f})"
        )

        is_best = val_stats["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_stats["val_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "gap": val_stats["gap"],
                    "vision_encoder_state_dict": base_model.vision_encoder.state_dict(),
                    "text_encoder_state_dict": base_model.text_encoder.state_dict(),
                    "model_state_dict": base_model.state_dict(),
                },
                best_path,
            )
            print(f"  -> New best {modality} SSL model saved!")

        torch.save(
            {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "history": history,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            },
            latest_path,
        )
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if is_best:
            sync_mgr.backup_file(best_path)
        if epoch % 3 == 0 or epoch == total_epochs:
            sync_mgr.backup_file(latest_path)
        sync_mgr.backup_file(history_path)
        sync_mgr.push_jsons_to_github(
            f"Modular SSL ({modality}) Epoch {epoch} metrics"
        )


if __name__ == "__main__":
    main()