from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 7
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20

INPUT_DIM = 1024
HIDDEN_DIM = 128
NUM_LABELS = 6

CACHE_ROOT = Path(
    "results/baseline/final_strong_fusion_screen/feature_cache"
)

RESULT_ROOT = Path(
    "results/baseline/final_strong_fusion_screen/linear_probe_test"
)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FeatureDataset(Dataset):
    def __init__(self, path: Path):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        self.labels = data["labels"].float()

        self.features = torch.cat(
            [
                data["photograph"].float(),
                data["radiograph"].float(),
                data["text"].float(),
            ],
            dim=1,
        )

        if self.features.shape[1] != INPUT_DIM:
            raise ValueError(
                f"Expected {INPUT_DIM} features, "
                f"got {self.features.shape[1]}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(INPUT_DIM),
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, NUM_LABELS),
        )

    def forward(self, x):
        return self.net(x)


def get_pos_weights(dataset):
    positives = dataset.labels.sum(dim=0)
    negatives = dataset.labels.shape[0] - positives

    return negatives / positives.clamp_min(1.0)


def collect_logits(model, loader, device):
    model.eval()

    logits_all = []
    labels_all = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            logits_all.append(logits.cpu())
            labels_all.append(y.cpu())

    return (
        torch.cat(logits_all),
        torch.cat(labels_all),
    )


def metrics(logits, labels, thresholds=None):
    probabilities = torch.sigmoid(logits).numpy()
    y_true = labels.numpy().astype(int)

    if thresholds is None:
        thresholds = np.full(NUM_LABELS, 0.5)

    y_pred = (
        probabilities >= np.asarray(thresholds)
    ).astype(int)

    result = {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "auroc": float(
            roc_auc_score(
                y_true,
                probabilities,
                average="macro",
            )
        ),
        "per_label_f1": [
            float(x)
            for x in f1_score(
                y_true,
                y_pred,
                average=None,
                zero_division=0,
            )
        ],
    }

    return result


def optimize_thresholds(logits, labels):
    probabilities = torch.sigmoid(logits).numpy()
    y_true = labels.numpy().astype(int)

    thresholds = np.arange(0.05, 0.951, 0.05)

    best_thresholds = []
    best_f1 = []

    for label_idx in range(NUM_LABELS):
        best_score = -1.0
        best_threshold = 0.5

        for threshold in thresholds:
            pred = (
                probabilities[:, label_idx]
                >= threshold
            ).astype(int)

            score = f1_score(
                y_true[:, label_idx],
                pred,
                zero_division=0,
            )

            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        best_thresholds.append(best_threshold)
        best_f1.append(float(best_score))

    return best_thresholds, best_f1


def main():
    seed_everything()

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("FROZEN LINEAR PROBE — FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    train_dataset = FeatureDataset(
        CACHE_ROOT / "train_v1.pt"
    )

    val_dataset = FeatureDataset(
        CACHE_ROOT / "validation_v1.pt"
    )

    test_dataset = FeatureDataset(
        CACHE_ROOT / "test_v1.pt"
    )

    print(
        f"Train: {len(train_dataset)}"
        f" | Val: {len(val_dataset)}"
        f" | Test: {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = LinearProbe().to(device)

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {trainable_params:,}"
    )

    pos_weights = get_pos_weights(
        train_dataset
    ).to(device)

    print(
        "Positive weights:",
        [round(float(x), 4) for x in pos_weights],
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_macro = -1.0
    best_epoch = -1
    patience_counter = 0

    best_state = None
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss = criterion(logits, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += (
                loss.item() * y.shape[0]
            )
            total_samples += y.shape[0]

        train_loss = total_loss / total_samples

        val_logits, val_labels = collect_logits(
            model,
            val_loader,
            device,
        )

        val_metrics = metrics(
            val_logits,
            val_labels,
        )

        val_macro = val_metrics["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_macro,
            "val_micro_f1": val_metrics["micro_f1"],
            "val_auroc": val_metrics["auroc"],
        })

        marker = ""

        if val_macro > best_macro:
            best_macro = val_macro
            best_epoch = epoch
            patience_counter = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            marker = "  <-- BEST"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val_macro_f1={val_macro:.4f} | "
            f"val_micro_f1={val_metrics['micro_f1']:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f}"
            f"{marker}"
        )

        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}."
            )
            break

    # --------------------------------------------------------
    # Restore best validation checkpoint
    # --------------------------------------------------------

    model.load_state_dict(best_state)

    print()
    print("=" * 70)
    print("BEST VALIDATION CHECKPOINT")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best Val Macro F1: {best_macro:.4f}")

    # --------------------------------------------------------
    # Validation: default + threshold optimization
    # --------------------------------------------------------

    val_logits, val_labels = collect_logits(
        model,
        val_loader,
        device,
    )

    val_default = metrics(
        val_logits,
        val_labels,
    )

    thresholds, threshold_scores = (
        optimize_thresholds(
            val_logits,
            val_labels,
        )
    )

    val_optimized = metrics(
        val_logits,
        val_labels,
        thresholds,
    )

    print()
    print("Validation @0.5:")
    print(
        f"Macro F1: {val_default['macro_f1']:.4f}"
    )
    print(
        f"Micro F1: {val_default['micro_f1']:.4f}"
    )
    print(
        f"AUROC:    {val_default['auroc']:.4f}"
    )

    print()
    print("Validation threshold optimization:")
    print(
        f"Macro F1: {val_optimized['macro_f1']:.4f}"
    )
    print(
        "Thresholds:",
        thresholds,
    )

    # --------------------------------------------------------
    # TEST — final untouched evaluation
    # --------------------------------------------------------

    test_logits, test_labels = collect_logits(
        model,
        test_loader,
        device,
    )

    test_default = metrics(
        test_logits,
        test_labels,
    )

    test_thresholded = metrics(
        test_logits,
        test_labels,
        thresholds,
    )

    print()
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    print(
        f"Test Macro F1 @0.5: "
        f"{test_default['macro_f1']:.4f}"
    )

    print(
        f"Test Macro F1 @val thresholds: "
        f"{test_thresholded['macro_f1']:.4f}"
    )

    print(
        f"Test Micro F1: "
        f"{test_thresholded['micro_f1']:.4f}"
    )

    print(
        f"Test AUROC: "
        f"{test_thresholded['auroc']:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_thresholded['accuracy']:.4f}"
    )

    print()
    print("Per-label Test F1 @ validation thresholds:")

    labels = [
        "caries",
        "gingivitis",
        "malocclusion",
        "pulpitis",
        "tooth_loss",
        "tooth_structure_loss",
    ]

    for name, score in zip(
        labels,
        test_thresholded["per_label_f1"],
    ):
        print(
            f"  {name:22s}: {score:.4f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results = {
        "experiment": "frozen_linear_probe_test",
        "seed": SEED,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "architecture": {
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "trainable_parameters": trainable_params,
        },
        "training": {
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "gradient_clip": 1.0,
            "loss": "weighted_BCEWithLogitsLoss",
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro,
        "validation_default": val_default,
        "validation_optimized": val_optimized,
        "validation_thresholds": thresholds,
        "validation_threshold_f1": threshold_scores,
        "test_default": test_default,
        "test_with_validation_thresholds": test_thresholded,
        "history": history,
    }

    with open(
        RESULT_ROOT / "results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        f"Results saved to: "
        f"{RESULT_ROOT / 'results.json'}"
    )

    print()
    print("TEST WAS NOT USED FOR TRAINING.")
    print("TEST WAS NOT USED FOR MODEL SELECTION.")
    print("TEST WAS NOT USED FOR THRESHOLD OPTIMIZATION.")
    print("=" * 70)


if __name__ == "__main__":
    main()
