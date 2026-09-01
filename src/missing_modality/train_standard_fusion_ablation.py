"""
Milestone 8.4.2 — Standard Fusion Ablation Training.

Purpose
-------
Train the standard multimodal fusion baseline used as the
controlled ablation against RobustFusion.

Experimental control
--------------------
The model uses the exact same frozen SSL representations,
population, architecture, optimizer, loss, and model-selection
criterion as RobustFusion.

The ONLY intended differences are:

    Standard Fusion:
        - no modality dropout
        - no modality-presence mask

    Robust Fusion:
        - modality dropout
        - explicit modality-presence mask

Training population
-------------------
Complete-case SSL representations inherited from Milestone 7.

Test data is never used for training or model selection.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.baseline import config
from src.baseline.metrics import compute_metrics

from src.missing_modality.robust_fusion_dataset import (
    RobustFusionRepresentationDataset,
)

from src.missing_modality.robust_fusion_model import (
    RobustFusion,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 64
EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DROPOUT = 0.3

REPRESENTATION_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "fusion"
    / "ssl_representations"
)

RESULT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "milestone8_missing_modality"
    / "06_robustness_ablation"
    / "02_standard_fusion"
)


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0

    for batch in tqdm(
        loader,
        desc="Training",
    ):

        image = batch["image"].to(
            DEVICE,
            non_blocking=True,
        )

        radiograph = batch["radiograph"].to(
            DEVICE,
            non_blocking=True,
        )

        text = batch["text"].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Standard Fusion:
        # all modalities are available.
        #
        # No stochastic modality dropout.
        # No explicit modality-presence mask.
        # ----------------------------------------------------

        modality_mask = torch.ones(
            image.shape[0],
            3,
            device=DEVICE,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            image,
            radiograph,
            text,
            modality_mask,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0

    all_logits = []
    all_labels = []

    for batch in tqdm(
        loader,
        desc="Validation",
    ):

        image = batch["image"].to(
            DEVICE,
            non_blocking=True,
        )

        radiograph = batch["radiograph"].to(
            DEVICE,
            non_blocking=True,
        )

        text = batch["text"].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )

        modality_mask = torch.ones(
            image.shape[0],
            3,
            device=DEVICE,
        )

        logits = model(
            image,
            radiograph,
            text,
            modality_mask,
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += loss.item()

        all_logits.append(
            logits.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

    logits = torch.cat(
        all_logits,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    metrics = compute_metrics(
        logits,
        labels,
        threshold=0.5,
    )

    return (
        total_loss / len(loader),
        metrics,
    )


# ============================================================
# Main
# ============================================================

def main():

    seed_everything(SEED)

    print("=" * 100)
    print(
        "MILESTONE 8.4.2 — STANDARD FUSION ABLATION TRAINING"
    )
    print("=" * 100)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Representation root:",
        REPRESENTATION_ROOT,
    )

    print(
        "Result root:",
        RESULT_ROOT,
    )

    print()
    print(
        "Training strategy:"
    )

    print(
        "  Modality dropout: DISABLED"
    )

    print(
        "  Modality presence mask: DISABLED"
    )

    print(
        "  SSL encoders: FROZEN"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = RobustFusionRepresentationDataset(
        REPRESENTATION_ROOT,
        "train",
    )

    validation_dataset = RobustFusionRepresentationDataset(
        REPRESENTATION_ROOT,
        "validation",
    )

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RobustFusion(
        image_dim=2048,
        radiograph_dim=2048,
        text_dim=768,
        modality_dim=512,
        hidden_dim=512,
        num_labels=config.NUM_LABELS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_macro_f1 = -1.0
    best_epoch = -1

    for epoch in range(EPOCHS):

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        validation_loss, metrics = evaluate(
            model,
            validation_loader,
            criterion,
        )

        print(
            f"Train loss: {train_loss:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Macro F1: "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Micro F1: "
            f"{metrics['micro_f1']:.4f}"
        )

        print(
            f"AUROC: "
            f"{metrics['auroc']:.4f}"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            **metrics,
        }

        history.append(epoch_record)

        # ----------------------------------------------------
        # Model selection
        # ----------------------------------------------------

        if metrics["macro_f1"] > best_macro_f1:

            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch + 1

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "model_name":
                    "standard_ablation",

                "epoch":
                    best_epoch,

                "validation_metrics":
                    metrics,

                "seed":
                    SEED,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "dropout":
                    DROPOUT,

                "modality_dropout":
                    False,

                "modality_presence_mask":
                    False,

            }

            torch.save(
                checkpoint,
                RESULT_ROOT / "best_model.pt",
            )

            print(
                "Saved best model."
            )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    with open(
        RESULT_ROOT / "history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    training_config = {

        "milestone":
            "8.4.2",

        "experiment":
            "Standard Fusion Ablation Training",

        "model":
            "RobustFusion architecture used as "
            "standard-fusion control",

        "seed":
            SEED,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "dropout":
            DROPOUT,

        "modality_dropout":
            False,

        "modality_presence_mask":
            False,

        "device":
            DEVICE,

        "num_labels":
            config.NUM_LABELS,

        "image_dim":
            2048,

        "radiograph_dim":
            2048,

        "text_dim":
            768,

        "modality_dim":
            512,

        "hidden_dim":
            512,

        "representation_root":
            str(REPRESENTATION_ROOT),

        "train_samples":
            len(train_dataset),

        "validation_samples":
            len(validation_dataset),

        "test_used_for_training":
            False,

        "test_used_for_model_selection":
            False,

        "model_selection_metric":
            "validation_macro_f1",

        "ssl_encoders_frozen":
            True,

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            best_macro_f1,
    }

    with open(
        RESULT_ROOT / "config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            training_config,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "MILESTONE 8.4.2 TRAINING COMPLETE"
    )
    print("=" * 100)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation Macro F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Results saved to:\n{RESULT_ROOT}"
    )


if __name__ == "__main__":
    main()