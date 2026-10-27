"""
Fast 4-Scenario Late-Fusion Trainer & Evaluator on Multi-View SSL Representations.

Runs:
    - image_only
    - xray_only
    - text_only
    - full_multimodal
    - all (default: runs all 4 scenarios in <2 minutes total!)
"""

import argparse
import json
import math
import random
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.baseline.final_linear_v4.loss import ClassWeightedBCEWithLogitsLoss
from src.baseline.final_linear_v4.metrics import (
    compute_metrics,
    optimize_thresholds,
)
from src.ssl_v2.cloud_sync import KagglePersistenceManager
from . import config
from .dataset import PrecomputedMultiViewDataset
from .models import MultiViewGatedCrossAttentionFusion


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weights(labels: np.ndarray) -> np.ndarray:
    pos = np.maximum(labels.sum(axis=0), 1.0)
    neg = labels.shape[0] - pos
    return np.clip(np.sqrt(neg / pos), 1.0, config.MAX_POS_WEIGHT).astype(
        np.float32
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, total_n = 0.0, 0

    for batch in loader:
        batch_gpu = {k: v.to(device) for k, v in batch.items()}
        labels = batch_gpu["labels"]
        logits = model(batch_gpu)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return (
        total_loss / max(total_n, 1),
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def run_one_scenario(scenario_name: str, sync_mgr: KagglePersistenceManager) -> dict:
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    modalities = config.SCENARIOS[scenario_name]

    out_dir = config.FUSION_SCENARIOS_DIR / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PrecomputedMultiViewDataset("train")
    val_ds = PrecomputedMultiViewDataset("validation")
    test_ds = PrecomputedMultiViewDataset("test")

    train_loader = DataLoader(
        train_ds, batch_size=config.FUSION_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.FUSION_BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.FUSION_BATCH_SIZE, shuffle=False
    )

    pos_weights = compute_pos_weights(train_ds.labels.numpy())
    criterion = ClassWeightedBCEWithLogitsLoss(
        pos_weights=pos_weights,
        label_smoothing=config.LABEL_SMOOTHING,
    ).to(device)

    model = MultiViewGatedCrossAttentionFusion(
        modalities=modalities,
        embed_dim=config.SHARED_EMBED_DIM,
        num_labels=config.NUM_LABELS,
        dropout=config.FUSION_DROPOUT,
        modality_dropout=config.MODALITY_DROPOUT,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.FUSION_LR,
        weight_decay=config.FUSION_WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.FUSION_EPOCHS, eta_min=1e-6
    )

    best_val_f1 = -math.inf
    best_epoch = 0
    patience = 0
    best_path = out_dir / "best_fusion_model.pt"
    history = []
    t0 = time.time()

    for epoch in range(1, config.FUSION_EPOCHS + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0

        for batch in train_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            labels = batch_gpu["labels"]

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_gpu)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = labels.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

        scheduler.step()
        train_loss = train_loss_sum / max(train_n, 1)

        val_loss, val_logits, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        val_metrics = compute_metrics(
            val_logits,
            val_labels,
            threshold=config.DEFAULT_THRESHOLD,
            label_names=config.LABEL_NAMES,
        )

        cur_f1 = val_metrics["macro_f1"]
        if cur_f1 > best_val_f1:
            best_val_f1 = cur_f1
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(val_loss),
                "validation_macro_f1": float(cur_f1),
                "validation_auroc": float(val_metrics["auroc"]),
            }
        )

        if patience >= config.FUSION_PATIENCE:
            break

    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=False)
    )

    _, val_logits, val_labels = evaluate(model, val_loader, criterion, device)
    opt_thresholds = optimize_thresholds(
        val_logits,
        val_labels,
        minimum=config.THRESHOLD_MIN,
        maximum=config.THRESHOLD_MAX,
        step=config.THRESHOLD_STEP,
    )
    val_default = compute_metrics(
        val_logits, val_labels, config.DEFAULT_THRESHOLD, config.LABEL_NAMES
    )
    val_opt = compute_metrics(
        val_logits, val_labels, opt_thresholds, config.LABEL_NAMES
    )

    _, test_logits, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    test_default = compute_metrics(
        test_logits, test_labels, config.DEFAULT_THRESHOLD, config.LABEL_NAMES
    )
    test_opt = compute_metrics(
        test_logits, test_labels, opt_thresholds, config.LABEL_NAMES
    )

    results = {
        "experiment": "modular_multiview_ssl_late_fusion",
        "scenario": scenario_name,
        "modalities": list(modalities),
        "labels": config.LABEL_NAMES,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": float(best_val_f1),
        "optimized_thresholds": opt_thresholds.tolist(),
        "validation_default": val_default,
        "validation_optimized": val_opt,
        "test_default": test_default,
        "test_optimized": test_opt,
        "elapsed_seconds": float(time.time() - t0),
        "history": history,
    }

    res_path = out_dir / "final_results.json"
    res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sync_mgr.backup_file(res_path)

    print(
        f"[{scenario_name:<15}] Best Epoch: {best_epoch:02d} | "
        f"Test Macro F1 (@0.5): {test_default['macro_f1']:.4f} | "
        f"Opt F1: {test_opt['macro_f1']:.4f} | "
        f"Micro F1: {test_default['micro_f1']:.4f} | "
        f"AUROC: {test_default['auroc']:.4f}"
    )
    return results


def main():
    parser = argparse.ArgumentParser()
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
    )
    args = parser.parse_args()

    sync_mgr = KagglePersistenceManager(config.PROJECT_ROOT)
    sync_mgr.results_dir = config.OUTPUT_ROOT
    sync_mgr.restore_all_from_backup()

    scenarios = (
        ["image_only", "xray_only", "text_only", "full_multimodal"]
        if args.scenario == "all"
        else [args.scenario]
    )

    print("=" * 80)
    print("MODULAR SSL v2 — 4-SCENARIO MULTI-VIEW FUSION EVALUATION")
    print("=" * 80)

    summary = {}
    for sc in scenarios:
        res = run_one_scenario(sc, sync_mgr)
        summary[sc] = {
            "macro_f1_0.5": res["test_default"]["macro_f1"],
            "macro_f1_opt": res["test_optimized"]["macro_f1"],
            "micro_f1": res["test_default"]["micro_f1"],
            "auroc": res["test_default"]["auroc"],
            "accuracy": res["test_default"]["accuracy"],
            "per_label_f1": {
                k: v["f1"] for k, v in res["test_default"]["per_label"].items()
            },
        }

    sum_path = config.FUSION_SCENARIOS_DIR / "all_4scenarios_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sync_mgr.backup_file(sum_path)
    sync_mgr.push_jsons_to_github("Complete Modular SSL 4-scenario fusion results")
    print("=" * 80)


if __name__ == "__main__":
    main()