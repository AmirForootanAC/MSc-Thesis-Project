"""
Multi-GPU Self-Supervised Pretraining (SSL v2) for COde Dataset.

Features:
1. Parallelizes the 3 specialized encoders (ConvNeXt-Tiny, DenseNet121,
   Bio_ClinicalBERT) across both Kaggle Tesla T4 GPUs via nn.DataParallel,
   then gathers embeddings to compute the full-batch contrastive matrix!
2. Evaluates validation SSL loss and cross-modal positive vs. negative
   cosine similarity gap at the end of every epoch.
3. Crash-safe resume via latest_ssl_checkpoint.pt.
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import config
from .dataset import MultiViewSSLDataset, pad_and_collate_multiview
from .encoders import MultiViewVisionEncoder, ClinicalTextEncoder
from .contrastive_v2 import DynamicMultimodalSSLV2Loss


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultimodalEncodersV2(nn.Module):
    """
    Container for the 3 specialized multi-view encoders.
    Wrapped in nn.DataParallel so GPU 0 and GPU 1 each encode half the batch.
    """

    def __init__(self):
        super().__init__()
        self.photo_encoder = MultiViewVisionEncoder(
            backbone_name=config.PHOTO_BACKBONE,
            pretrained=True,
            out_dim=config.SHARED_EMBED_DIM,
            dropout=0.25,
            freeze_early_stages=1,
        )
        self.radio_encoder = MultiViewVisionEncoder(
            backbone_name=config.RADIO_BACKBONE,
            pretrained=True,
            out_dim=config.SHARED_EMBED_DIM,
            dropout=0.25,
            freeze_early_stages=1,
        )
        self.text_encoder = ClinicalTextEncoder(
            model_name=config.TEXT_MODEL_NAME,
            out_dim=config.SHARED_EMBED_DIM,
            freeze_layers=config.FREEZE_TEXT_LAYERS,
            internal_dropout=0.20,
            dropout=0.25,
        )

    def forward(
        self,
        photo_images: torch.Tensor,
        photo_view_mask: torch.Tensor,
        radio_images: torch.Tensor,
        radio_view_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        photo_out = self.photo_encoder(photo_images, photo_view_mask)
        radio_out = self.radio_encoder(radio_images, radio_view_mask)
        text_out = self.text_encoder(input_ids, attention_mask)
        return photo_out, radio_out, text_out


class MultimodalSSLV2System(nn.Module):
    """Combines MultimodalEncodersV2 and DynamicMultimodalSSLV2Loss."""

    def __init__(self):
        super().__init__()
        self.encoders = MultimodalEncodersV2()
        self.criterion = DynamicMultimodalSSLV2Loss(
            embed_dim=config.SHARED_EMBED_DIM,
            proj_dim=config.PROJECTION_DIM,
            temperature=config.SSL_TEMPERATURE,
            soft_alpha=config.SSL_SOFT_TARGET_ALPHA,
            local_view_weight=config.SSL_LOCAL_VIEW_WEIGHT,
            intra_view_weight=config.SSL_INTRA_VIEW_WEIGHT,
        )


def build_ssl_optimizer(system: MultimodalSSLV2System) -> AdamW:
    vision_params = []
    text_params = []
    head_params = []

    for name, param in system.encoders.named_parameters():
        if not param.requires_grad:
            continue
        if "text_encoder.transformer" in name:
            text_params.append(param)
        elif "mil_aggregator" in name or "text_encoder.projector" in name or "token_attn" in name:
            head_params.append(param)
        else:
            vision_params.append(param)

    for param in system.criterion.parameters():
        if param.requires_grad:
            head_params.append(param)

    param_groups = [
        {
            "name": "vision_backbone",
            "params": vision_params,
            "lr": config.SSL_VISION_LR,
            "base_lr": config.SSL_VISION_LR,
        },
        {
            "name": "text_backbone",
            "params": text_params,
            "lr": config.SSL_TEXT_LR,
            "base_lr": config.SSL_TEXT_LR,
        },
        {
            "name": "ssl_heads",
            "params": head_params,
            "lr": config.SSL_HEAD_LR,
            "base_lr": config.SSL_HEAD_LR,
        },
    ]
    return AdamW(param_groups, weight_decay=config.SSL_WEIGHT_DECAY)


@torch.no_grad()
def evaluate_ssl_validation(
    encoder_runner: nn.Module,
    criterion: DynamicMultimodalSSLV2Loss,
    val_loader: DataLoader,
    device: torch.device,
):
    encoder_runner.eval()
    criterion.eval()

    total_loss = 0.0
    valid_batches = 0
    pos_sims = []
    neg_sims = []

    for batch in tqdm(val_loader, desc="[SSL Val]", leave=False):
        photo_images = batch["photo_images"].to(device, non_blocking=True)
        photo_mask = batch["photo_view_mask"].to(device, non_blocking=True)
        radio_images = batch["radio_images"].to(device, non_blocking=True)
        radio_mask = batch["radio_view_mask"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        has_photo = batch["has_photo"].to(device, non_blocking=True)
        has_radio = batch["has_radio"].to(device, non_blocking=True)
        has_text = batch["has_text"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
            photo_out, radio_out, text_out = encoder_runner(
                photo_images,
                photo_mask,
                radio_images,
                radio_mask,
                input_ids,
                attention_mask,
            )
            loss, pairs = criterion(
                photo_out,
                radio_out,
                text_out,
                has_photo,
                has_radio,
                has_text,
                photo_mask,
                radio_mask,
            )

        if pairs:
            total_loss += loss.item()
            valid_batches += 1

        # Compute positive vs negative cosine similarity on photo-text pairs
        mask_pt = has_photo & has_text
        if mask_pt.sum() >= 2:
            zp = criterion.photo_proj(photo_out["visit_embedding"][mask_pt])
            zt = criterion.text_proj(text_out["visit_embedding"][mask_pt])
            sim_mat = zp @ zt.T
            diag = torch.diag(sim_mat)
            pos_sims.append(diag.mean().item())
            off_diag = sim_mat[~torch.eye(sim_mat.size(0), dtype=torch.bool, device=device)]
            neg_sims.append(off_diag.mean().item())

    avg_val_loss = total_loss / max(valid_batches, 1)
    mean_pos = float(np.mean(pos_sims)) if pos_sims else 0.0
    mean_neg = float(np.mean(neg_sims)) if neg_sims else 0.0
    alignment_gap = mean_pos - mean_neg

    return {
        "val_loss": avg_val_loss,
        "pos_cosine_sim": mean_pos,
        "neg_cosine_sim": mean_neg,
        "alignment_gap": alignment_gap,
    }


def main():
    seed_everything(config.SEED)
    device = torch.device(config.DEVICE)
    config.SSL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SSL v2 MULTI-VIEW PRETRAINING (NO AVERAGING | 2x T4 OPTIMIZED)")
    print("=" * 70)
    print(f"Device       : {device} (GPUs available: {config.NUM_GPUS})")
    print(f"Dataset      : {config.DATASET_PATH}")
    print(f"Photo Model  : {config.PHOTO_BACKBONE} (max {config.MAX_PHOTO_VIEWS_TRAIN} views/visit)")
    print(f"Radio Model  : {config.RADIO_BACKBONE} + CLAHE (max {config.MAX_RADIO_VIEWS_TRAIN} views/visit)")
    print(f"Text Model   : {config.TEXT_MODEL_NAME}")
    print(f"Batch Size   : {config.SSL_BATCH_SIZE} (Grad Accum: {config.SSL_GRAD_ACCUM_STEPS} -> Effective: {config.SSL_BATCH_SIZE * config.SSL_GRAD_ACCUM_STEPS})")
    print("=" * 70)

    train_dataset = MultiViewSSLDataset(split="train", is_train=True)
    val_dataset = MultiViewSSLDataset(split="validation", is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.SSL_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=pad_and_collate_multiview,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.SSL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=pad_and_collate_multiview,
    )

    system = MultimodalSSLV2System().to(device)

    if config.USE_MULTI_GPU:
        print(f"[Multi-GPU] Wrapping MultimodalEncodersV2 in DataParallel across {config.NUM_GPUS} GPUs.")
        encoder_runner = nn.DataParallel(system.encoders)
    else:
        encoder_runner = system.encoders

    optimizer = build_ssl_optimizer(system)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(config.SSL_EPOCHS - config.SSL_WARMUP_EPOCHS, 1),
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=config.USE_AMP)

    best_path = config.SSL_OUTPUT_DIR / "best_ssl_model.pt"
    latest_path = config.SSL_OUTPUT_DIR / "latest_ssl_checkpoint.pt"
    history_path = config.SSL_OUTPUT_DIR / "ssl_history.json"

    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    if latest_path.exists():
        print(f"\n[Resume] Restoring SSL pretraining from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        system.load_state_dict(ckpt["system_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        history = ckpt.get("history", [])

    for epoch in range(start_epoch, config.SSL_EPOCHS + 1):
        # Linear Warmup
        if epoch <= config.SSL_WARMUP_EPOCHS:
            warmup_factor = epoch / config.SSL_WARMUP_EPOCHS
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * warmup_factor

        encoder_runner.train()
        system.criterion.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        valid_steps = 0
        pair_sums = {
            "image_text": 0.0,
            "image_radiograph": 0.0,
            "radiograph_text": 0.0,
            "intra_multiview": 0.0,
        }

        pbar = tqdm(train_loader, desc=f"SSL Epoch {epoch:02d}/{config.SSL_EPOCHS}")
        num_batches = len(train_loader)

        for step, batch in enumerate(pbar):
            photo_images = batch["photo_images"].to(device, non_blocking=True)
            photo_mask = batch["photo_view_mask"].to(device, non_blocking=True)
            radio_images = batch["radio_images"].to(device, non_blocking=True)
            radio_mask = batch["radio_view_mask"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            has_photo = batch["has_photo"].to(device, non_blocking=True)
            has_radio = batch["has_radio"].to(device, non_blocking=True)
            has_text = batch["has_text"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
                photo_out, radio_out, text_out = encoder_runner(
                    photo_images,
                    photo_mask,
                    radio_images,
                    radio_mask,
                    input_ids,
                    attention_mask,
                )
                loss, pairs = system.criterion(
                    photo_out,
                    radio_out,
                    text_out,
                    has_photo,
                    has_radio,
                    has_text,
                    photo_mask,
                    radio_mask,
                )
                scaled_loss = loss / config.SSL_GRAD_ACCUM_STEPS

            if not pairs:
                continue

            scaler.scale(scaled_loss).backward()

            if (step + 1) % config.SSL_GRAD_ACCUM_STEPS == 0 or (step + 1) == num_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            valid_steps += 1
            for k, v in pairs.items():
                if k in pair_sums:
                    pair_sums[k] += v.item()

            pbar.set_postfix(
                loss=f"{epoch_loss / max(valid_steps, 1):.4f}",
                pairs=",".join(pairs.keys()),
            )

        if epoch > config.SSL_WARMUP_EPOCHS:
            scheduler.step()

        train_avg_loss = epoch_loss / max(valid_steps, 1)
        val_stats = evaluate_ssl_validation(
            encoder_runner, system.criterion, val_loader, device
        )

        record = {
            "epoch": epoch,
            "train_loss": train_avg_loss,
            "val_loss": val_stats["val_loss"],
            "pos_cosine_sim": val_stats["pos_cosine_sim"],
            "neg_cosine_sim": val_stats["neg_cosine_sim"],
            "alignment_gap": val_stats["alignment_gap"],
            "pair_losses": {
                k: v / max(valid_steps, 1) for k, v in pair_sums.items()
            },
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_avg_loss:.4f} | "
            f"Val Loss: {val_stats['val_loss']:.4f} | "
            f"CosSim Gap: +{val_stats['alignment_gap']:.4f} "
            f"(Pos: {val_stats['pos_cosine_sim']:.3f}, Neg: {val_stats['neg_cosine_sim']:.3f})"
        )

        if val_stats["val_loss"] < best_val_loss:
            best_val_loss = val_stats["val_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "alignment_gap": val_stats["alignment_gap"],
                    "encoders_state_dict": system.encoders.state_dict(),
                    "system_state_dict": system.state_dict(),
                },
                best_path,
            )
            print(f"  -> Saved new best SSL model (Val Loss = {best_val_loss:.4f})")

        torch.save(
            {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "history": history,
                "system_state_dict": system.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            },
            latest_path,
        )
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()