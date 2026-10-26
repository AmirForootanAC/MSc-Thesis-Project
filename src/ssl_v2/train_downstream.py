"""
Unified Training & Evaluation Pipeline for the 4 Scenarios (SSL v2 + Fusion v2).

Scenarios supported via CLI (--scenario):
    - image_only
    - xray_only
    - text_only
    - full_multimodal
    - all (runs all 4 scenarios sequentially and generates a comparison summary)

Features:
    - Loads pretrained SSL v2 weights from best_ssl_model.pt
    - Multi-GPU execution across 2x T4 GPUs via nn.DataParallel + AMP FP16
    - Multi-View Test-Time Augmentation (horizontal flip across all views)
    - Validation-only threshold optimization + crash-safe resume + Cloud/GitHub Sync
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.baseline.final_linear_v4.loss import ClassWeightedBCEWithLogitsLoss
from src.baseline.final_linear_v4.metrics import (
    compute_metrics,
    optimize_thresholds,
)

from . import config
from .cloud_sync import KagglePersistenceManager
from .dataset import MultiViewDownstreamDataset, pad_and_collate_multiview
from .fusion_model import MultimodalScenarioClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_bn_stats(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def compute_pos_weights(train_dataset: MultiViewDownstreamDataset) -> np.ndarray:
    labels = train_dataset.df[config.LABEL_NAMES].to_numpy(dtype=np.float32)
    pos_counts = np.maximum(labels.sum(axis=0), 1.0)
    neg_counts = labels.shape[0] - pos_counts
    raw_weights = neg_counts / pos_counts
    damped = np.clip(np.sqrt(raw_weights), 1.0, config.MAX_POS_WEIGHT)
    return damped.astype(np.float32)


def build_optimizer(model: MultimodalScenarioClassifier) -> AdamW:
    backbone_params = []
    text_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "text_encoder.transformer" in name:
            text_params.append(param)
        elif "spatial_extractor" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if backbone_params:
        groups.append(
            {
                "name": "vision_backbone",
                "params": backbone_params,
                "lr": config.BACKBONE_LR,
                "base_lr": config.BACKBONE_LR,
                "weight_decay": 0.02,
            }
        )
    if text_params:
        groups.append(
            {
                "name": "text_backbone",
                "params": text_params,
                "lr": config.TEXT_LR,
                "base_lr": config.TEXT_LR,
                "weight_decay": config.WEIGHT_DECAY,
            }
        )
    if head_params:
        groups.append(
            {
                "name": "fusion_head",
                "params": head_params,
                "lr": config.FUSION_HEAD_LR,
                "base_lr": config.FUSION_HEAD_LR,
                "weight_decay": config.WEIGHT_DECAY,
            }
        )

    return AdamW(groups, weight_decay=config.WEIGHT_DECAY)


def forward_batch(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    flip_views: bool = False,
):
    kwargs = {}
    if "photo_images" in batch:
        imgs = batch["photo_images"].to(device, non_blocking=True)
        if flip_views:
            imgs = imgs.flip(dims=[-1])
        kwargs["photo_images"] = imgs
        kwargs["photo_view_mask"] = batch["photo_view_mask"].to(
            device, non_blocking=True
        )

    if "radio_images" in batch:
        r_imgs = batch["radio_images"].to(device, non_blocking=True)
        if flip_views:
            r_imgs = r_imgs.flip(dims=[-1])
        kwargs["radio_images"] = r_imgs
        kwargs["radio_view_mask"] = batch["radio_view_mask"].to(
            device, non_blocking=True
        )

    if "input_ids" in batch:
        kwargs["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
        kwargs["attention_mask"] = batch["attention_mask"].to(
            device, non_blocking=True
        )

    return model(**kwargs)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_tta: bool = False,
    desc: str = "[Eval]",
):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    has_visual = any(
        m in ("photograph", "radiograph") for m in loader.dataset.modalities
    )

    for batch in tqdm(loader, desc=desc, leave=False):
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
            logits = forward_batch(model, batch, device, flip_views=False)

            if use_tta and has_visual:
                logits_flipped = forward_batch(
                    model, batch, device, flip_views=True
                )
                probs = 0.5 * (
                    torch.sigmoid(logits) + torch.sigmoid(logits_flipped)
                )
                probs = torch.clamp(probs, 1e-6, 1.0 - 1.0e-6)
                logits = torch.log(probs / (1.0 - probs))

            loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.float().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, logits_np, labels_np


def run_scenario(
    scenario_name: str,
    ssl_checkpoint_path: Path,
    sync_mgr: KagglePersistenceManager,
    force_rerun: bool = False,
) -> dict:
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    modalities = config.SCENARIOS[scenario_name]

    out_dir = config.DOWNSTREAM_OUTPUT_DIR / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)
    final_results_path = out_dir / "final_results.json"
    history_json_path = out_dir / "history.json"
    best_model_path = out_dir / "best_model.pt"
    latest_ckpt_path = out_dir / "latest.pt"

    if final_results_path.exists() and not force_rerun:
        print(
            f"\n[Skip] Scenario '{scenario_name}' already completed at "
            f"{final_results_path}"
        )
        return json.loads(final_results_path.read_text(encoding="utf-8"))

    print("\n" + "=" * 75)
    print(f"RUNNING SCENARIO: {scenario_name.upper()} | Modalities: {modalities}")
    print("=" * 75)

    train_ds = MultiViewDownstreamDataset(
        split="train", modalities=modalities, is_train=True
    )
    val_ds = MultiViewDownstreamDataset(
        split="validation", modalities=modalities, is_train=False
    )
    test_ds = MultiViewDownstreamDataset(
        split="test", modalities=modalities, is_train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.DOWNSTREAM_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=pad_and_collate_multiview,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.DOWNSTREAM_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=pad_and_collate_multiview,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.DOWNSTREAM_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=pad_and_collate_multiview,
    )

    pos_weights = compute_pos_weights(train_ds)
    criterion = ClassWeightedBCEWithLogitsLoss(
        pos_weights=pos_weights,
        label_smoothing=config.LABEL_SMOOTHING,
    ).to(device)

    base_model = MultimodalScenarioClassifier(
        modalities=modalities,
        num_labels=config.NUM_LABELS,
        embed_dim=config.SHARED_EMBED_DIM,
        dropout=config.DROPOUT,
        modality_dropout=config.MODALITY_DROPOUT,
    )
    ssl_loaded = base_model.load_ssl_pretrained_weights(ssl_checkpoint_path)
    base_model = base_model.to(device)

    if config.USE_MULTI_GPU:
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    optimizer = build_optimizer(base_model)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(config.DOWNSTREAM_EPOCHS - config.WARMUP_EPOCHS, 1),
        eta_min=1e-7,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=config.USE_AMP)

    start_epoch = 1
    best_val_f1 = -math.inf
    best_epoch = 0
    patience = 0
    history = []

    if latest_ckpt_path.exists() and not force_rerun:
        print(
            f"[Resume] Restoring scenario '{scenario_name}' from {latest_ckpt_path}"
        )
        ckpt = torch.load(
            latest_ckpt_path, map_location=device, weights_only=False
        )
        base_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_f1 = float(ckpt["best_val_f1"])
        best_epoch = int(ckpt["best_epoch"])
        patience = int(ckpt["patience"])
        history = ckpt["history"]

    t0 = time.time()

    for epoch in range(start_epoch, config.DOWNSTREAM_EPOCHS + 1):
        if epoch <= config.WARMUP_EPOCHS:
            prog = epoch / config.WARMUP_EPOCHS
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * prog

        model.train()
        freeze_bn_stats(model)
        optimizer.zero_grad(set_to_none=True)

        train_loss_sum = 0.0
        train_samples = 0
        num_batches = len(train_loader)

        pbar = tqdm(
            train_loader,
            desc=f"[{scenario_name}] Epoch {epoch:02d}/{config.DOWNSTREAM_EPOCHS}",
            leave=False,
        )

        for step, batch in enumerate(pbar):
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type, enabled=config.USE_AMP
            ):
                logits = forward_batch(model, batch, device, flip_views=False)
                loss = criterion(logits, labels)
                scaled_loss = loss / config.DOWNSTREAM_GRAD_ACCUM_STEPS

            scaler.scale(scaled_loss).backward()

            if (
                (step + 1) % config.DOWNSTREAM_GRAD_ACCUM_STEPS == 0
                or (step + 1) == num_batches
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(), config.GRADIENT_CLIP_NORM
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = labels.size(0)
            train_loss_sum += loss.item() * bs
            train_samples += bs
            pbar.set_postfix(
                loss=f"{train_loss_sum / max(train_samples, 1):.4f}"
            )

        if epoch > config.WARMUP_EPOCHS:
            scheduler.step()

        train_loss = train_loss_sum / max(train_samples, 1)
        val_loss, val_logits, val_labels = evaluate_split(
            model, val_loader, criterion, device, use_tta=False, desc="[Val]"
        )
        val_metrics = compute_metrics(
            val_logits,
            val_labels,
            threshold=config.DEFAULT_THRESHOLD,
            label_names=config.LABEL_NAMES,
        )

        cur_f1 = val_metrics["macro_f1"]
        improved = cur_f1 > best_val_f1

        if improved:
            best_val_f1 = cur_f1
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "val_macro_f1": best_val_f1,
                    "model_state_dict": base_model.state_dict(),
                },
                best_model_path,
            )
        else:
            patience += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(val_loss),
                "validation_macro_f1": float(val_metrics["macro_f1"]),
                "validation_micro_f1": float(val_metrics["micro_f1"]),
                "validation_auroc": float(val_metrics["auroc"]),
                "validation_accuracy": float(val_metrics["accuracy"]),
            }
        )

        torch.save(
            {
                "epoch": epoch,
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
                "patience": patience,
                "history": history,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            },
            latest_ckpt_path,
        )
        history_json_path.write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Macro F1: {cur_f1:.4f} | "
            f"Val AUROC: {val_metrics['auroc']:.4f} | "
            f"Patience: {patience}/{config.EARLY_STOPPING_PATIENCE}"
        )

        # Live Cloud Backup & GitHub Sync after each epoch
        if improved:
            sync_mgr.backup_file(best_model_path)
        if epoch % 2 == 0 or epoch == config.DOWNSTREAM_EPOCHS:
            sync_mgr.backup_file(latest_ckpt_path)
        sync_mgr.backup_file(history_json_path)
        sync_mgr.push_jsons_to_github(
            f"Downstream v2 {scenario_name} epoch {epoch} history"
        )

        if patience >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # Load best checkpoint for final evaluation
    best_ckpt = torch.load(
        best_model_path, map_location=device, weights_only=False
    )
    base_model.load_state_dict(best_ckpt["model_state_dict"])

    _, val_logits_tta, val_labels_tta = evaluate_split(
        model,
        val_loader,
        criterion,
        device,
        use_tta=config.USE_TTA,
        desc="[Val TTA]",
    )
    val_default = compute_metrics(
        val_logits_tta,
        val_labels_tta,
        threshold=config.DEFAULT_THRESHOLD,
        label_names=config.LABEL_NAMES,
    )
    opt_thresholds = optimize_thresholds(
        val_logits_tta,
        val_labels_tta,
        minimum=config.THRESHOLD_MIN,
        maximum=config.THRESHOLD_MAX,
        step=config.THRESHOLD_STEP,
    )
    val_optimized = compute_metrics(
        val_logits_tta,
        val_labels_tta,
        threshold=opt_thresholds,
        label_names=config.LABEL_NAMES,
    )

    _, test_logits_tta, test_labels_tta = evaluate_split(
        model,
        test_loader,
        criterion,
        device,
        use_tta=config.USE_TTA,
        desc="[Test TTA]",
    )
    test_default = compute_metrics(
        test_logits_tta,
        test_labels_tta,
        threshold=config.DEFAULT_THRESHOLD,
        label_names=config.LABEL_NAMES,
    )
    test_optimized = compute_metrics(
        test_logits_tta,
        test_labels_tta,
        threshold=opt_thresholds,
        label_names=config.LABEL_NAMES,
    )

    results = {
        "experiment": "ssl_v2_multiview_four_label_fusion",
        "scenario": scenario_name,
        "modalities": list(modalities),
        "ssl_pretrained_loaded": ssl_loaded,
        "labels": config.LABEL_NAMES,
        "best_epoch": int(best_epoch),
        "best_validation_macro_f1": float(best_val_f1),
        "optimized_thresholds": opt_thresholds.tolist(),
        "validation_default": val_default,
        "validation_optimized": val_optimized,
        "test_default": test_default,
        "test_optimized": test_optimized,
        "elapsed_seconds": float(time.time() - t0),
        "history": history,
    }

    final_results_path.write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # Final backup of completed scenario results
    sync_mgr.backup_file(final_results_path)
    sync_mgr.push_jsons_to_github(
        f"Downstream v2 {scenario_name} final test results"
    )

    print("\n" + "-" * 75)
    print(
        f"FINAL TEST RESULTS — {scenario_name.upper()} (Best Epoch: {best_epoch})"
    )
    print("-" * 75)
    print(f"Test Macro F1 (@0.5) : {test_default['macro_f1']:.4f}")
    print(f"Test Macro F1 (Opt)  : {test_optimized['macro_f1']:.4f}")
    print(f"Test Micro F1 (@0.5) : {test_default['micro_f1']:.4f}")
    print(f"Test Macro AUROC     : {test_default['auroc']:.4f}")
    print(f"Test Accuracy        : {test_default['accuracy']:.4f}")
    print("-" * 75)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SSL v2 Downstream / Multimodal Fusion across 4 scenarios."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=[
            "image_only",
            "xray_only",
            "text_only",
            "full_multimodal",
            "all",
        ],
        help="Which scenario to train and evaluate.",
    )
    parser.add_argument(
        "--ssl-checkpoint",
        type=str,
        default=str(config.SSL_OUTPUT_DIR / "best_ssl_model.pt"),
        help="Path to the pretrained SSL v2 checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if final_results.json exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Restore any existing checkpoints from Cloud / Kaggle Input at session start
    sync_mgr = KagglePersistenceManager(config.PROJECT_ROOT)
    sync_mgr.restore_all_from_backup()

    ssl_ckpt = Path(args.ssl_checkpoint)

    scenarios_to_run = (
        ["image_only", "xray_only", "text_only", "full_multimodal"]
        if args.scenario == "all"
        else [args.scenario]
    )

    summary_table = {}
    for scen in scenarios_to_run:
        res = run_scenario(
            scen, ssl_ckpt, sync_mgr=sync_mgr, force_rerun=args.force
        )
        summary_table[scen] = {
            "test_macro_f1_0.5": res["test_default"]["macro_f1"],
            "test_macro_f1_opt": res["test_optimized"]["macro_f1"],
            "test_micro_f1": res["test_default"]["micro_f1"],
            "test_auroc": res["test_default"]["auroc"],
            "test_accuracy": res["test_default"]["accuracy"],
        }

    if len(summary_table) > 1:
        summary_path = (
            config.DOWNSTREAM_OUTPUT_DIR / "all_4scenarios_summary.json"
        )
        summary_path.write_text(
            json.dumps(summary_table, indent=2), encoding="utf-8"
        )
        sync_mgr.backup_file(summary_path)
        sync_mgr.push_jsons_to_github(
            "Complete 4-scenario SSL v2 benchmark summary"
        )

        print("\n" + "=" * 75)
        print("4-SCENARIO BENCHMARK SUMMARY (TEST SET)")
        print("=" * 75)
        for scen, m in summary_table.items():
            print(
                f"{scen:<18} | Macro F1: {m['test_macro_f1_0.5']:.4f} "
                f"(Opt: {m['test_macro_f1_opt']:.4f}) | "
                f"Micro F1: {m['test_micro_f1']:.4f} | AUROC: {m['test_auroc']:.4f}"
            )
        print("=" * 75)


if __name__ == "__main__":
    main()