"""
Milestone 8.3.2 — Robust Fusion Training.

Purpose
-------
Train a multimodal fusion classifier that is explicitly exposed
to missing-modality conditions during training.

The SSL representations remain frozen.

Training strategy
-----------------
Complete-case SSL representations are loaded from Milestone 7.

During training, modality dropout is applied dynamically.

Possible training states include:

    Image + X-ray + Text
    Image + Text
    Image + X-ray
    Image only

The model receives both:
    1. modality representations
    2. an explicit modality-presence mask

Validation is performed without stochastic modality dropout.

Model selection uses validation Macro F1 only.

The test set is never used during training or model selection.
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

# Probability that a modality is dropped independently.
MODALITY_DROPOUT_PROBABILITY = 0.30

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
    / "04_robust_fusion_training"
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
# Modality dropout
# ============================================================

def sample_modality_mask(
    batch_size,
    device,
    drop_probability,
):
    """
    Generate a modality-presence mask.

    Each modality is independently dropped with the given
    probability.

    The all-missing state is explicitly prevented.

    Mask order:
        image
        radiograph
        text
    """

    mask = (
        torch.rand(
            batch_size,
            3,
            device=device,
        )
        >= drop_probability
    ).float()

    all_missing = (
        mask.sum(dim=1) == 0
    )

    if all_missing.any():

        indices = torch.where(
            all_missing
        )[0]

        selected = torch.randint(
            low=0,
            high=3,
            size=(len(indices),),
            device=device,
        )

        mask[
            indices,
            selected,
        ] = 1.0

    return mask


# ============================================================
# Apply modality mask
# ============================================================

def apply_mask(
    image,
    radiograph,
    text,
    mask,
):

    image = (
        image
        *
        mask[:, 0:1]
    )

    radiograph = (
        radiograph
        *
        mask[:, 1:2]
    )

    text = (
        text
        *
        mask[:, 2:3]
    )

    return (
        image,
        radiograph,
        text,
    )


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

        image = batch[
            "image"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        radiograph = batch[
            "radiograph"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        text = batch[
            "text"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch[
            "labels"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        # ----------------------------------------------
        # Dynamic modality dropout
        # ----------------------------------------------

        modality_mask = sample_modality_mask(
            batch_size=image.shape[0],
            device=DEVICE,
            drop_probability=(
                MODALITY_DROPOUT_PROBABILITY
            ),
        )

        image, radiograph, text = apply_mask(
            image,
            radiograph,
            text,
            modality_mask,
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

    # Validation uses complete multimodal input.
    modality_mask = torch.ones(
        1,
        3,
        device=DEVICE,
    )

    for batch in tqdm(
        loader,
        desc="Validation",
    ):

        image = batch[
            "image"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        radiograph = batch[
            "radiograph"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        text = batch[
            "text"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch[
            "labels"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        batch_mask = modality_mask.expand(
            image.shape[0],
            -1,
        )

        logits = model(
            image,
            radiograph,
            text,
            batch_mask,
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

    print("=" * 100)
    print(
        "MILESTONE 8.3.2 — ROBUST FUSION TRAINING"
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
        "Modality dropout probability:",
        MODALITY_DROPOUT_PROBABILITY,
    )

    print(
        "Result root:",
        RESULT_ROOT,
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

            best_macro_f1 = (
                metrics["macro_f1"]
            )

            best_epoch = epoch + 1

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "model_name":
                    "robust",

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

                "modality_dropout_probability":
                    MODALITY_DROPOUT_PROBABILITY,
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
    # History
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
    # Configuration
    # --------------------------------------------------------

    training_config = {
        "milestone":
            "8.3.2",

        "experiment":
            "Robust Fusion Training",

        "model":
            "RobustFusion",

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

        "modality_dropout_probability":
            MODALITY_DROPOUT_PROBABILITY,

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

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            best_macro_f1,

        "test_used_for_model_selection":
            False,

        "ssl_encoders_frozen":
            True,
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
    print("=" * 100)
    print(
        "MILESTONE 8.3.2 TRAINING COMPLETE"
    )
    print("=" * 100)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        "Best validation Macro F1:",
        f"{best_macro_f1:.4f}",
    )


if __name__ == "__main__":
    main()