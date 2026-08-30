"""
Training for Milestone 7.3 multimodal fusion.

The SSL representations are frozen because they are loaded
from precomputed representation files.

Supported fusion models:
    simple
    main

Milestone 7.3 currently trains the complete multimodal model:

    image + radiograph + text

Missing-modality experiments belong to Milestone 8.
"""

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

from src.fusion.dataset import (
    FusionRepresentationDataset,
)

from src.fusion.fusion_model import (
    SimpleFusion,
    MainFusion,
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

MODEL_NAME = "main"
# simple
# main

BATCH_SIZE = 64
EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

REPRESENTATION_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "fusion"
    / "ssl_representations"
)

RESULT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "fusion"
    / MODEL_NAME
)


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# Model
# ============================================================

def build_model():

    if MODEL_NAME == "simple":

        return SimpleFusion(
            image_dim=2048,
            radiograph_dim=2048,
            text_dim=768,
            hidden_dim=512,
            num_labels=config.NUM_LABELS,
            dropout=0.3,
        )

    if MODEL_NAME == "main":

        return MainFusion(
            image_dim=2048,
            radiograph_dim=2048,
            text_dim=768,
            modality_dim=512,
            hidden_dim=512,
            num_labels=config.NUM_LABELS,
            dropout=0.3,
        )

    raise ValueError(
        f"Unknown MODEL_NAME: {MODEL_NAME}"
    )


# ============================================================
# One training epoch
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

        radiograph = batch[
            "radiograph"
        ].to(
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

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            image,
            radiograph,
            text,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return (
        total_loss
        /
        len(loader)
    )


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

        radiograph = batch[
            "radiograph"
        ].to(
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

        logits = model(
            image,
            radiograph,
            text,
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

    seed_everything(
        SEED
    )

    print("=" * 60)
    print("MILESTONE 7.3 — MULTIMODAL FUSION")
    print("=" * 60)

    print(
        "Model:",
        MODEL_NAME,
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Representation root:",
        REPRESENTATION_ROOT,
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = FusionRepresentationDataset(
        REPRESENTATION_ROOT,
        "train",
    )

    validation_dataset = FusionRepresentationDataset(
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
        pin_memory=(
            DEVICE == "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model().to(
        DEVICE
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Output directory
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

    for epoch in range(
        EPOCHS
    ):

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
            f"Train loss: "
            f"{train_loss:.4f}"
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

        history.append(
            epoch_record
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if metrics["macro_f1"] > best_macro_f1:

            best_macro_f1 = metrics[
                "macro_f1"
            ]

            best_epoch = epoch + 1

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "model_name":
                    MODEL_NAME,

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
            }

            torch.save(
                checkpoint,
                RESULT_ROOT
                / "best_model.pt",
            )

            print(
                "Saved best model."
            )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with open(
        RESULT_ROOT / "history.json",
        "w",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    training_config = {
        "model": MODEL_NAME,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "device": DEVICE,
        "num_labels": config.NUM_LABELS,
        "image_dim": 2048,
        "radiograph_dim": 2048,
        "text_dim": 768,
        "representation_root":
            str(REPRESENTATION_ROOT),
        "best_epoch": best_epoch,
        "best_validation_macro_f1":
            best_macro_f1,
    }

    with open(
        RESULT_ROOT / "config.json",
        "w",
    ) as f:

        json.dump(
            training_config,
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation Macro F1: "
        f"{best_macro_f1:.4f}"
    )


if __name__ == "__main__":
    main()