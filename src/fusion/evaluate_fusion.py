"""
Test evaluation for Milestone 7.3 multimodal fusion.

The best model selected using validation Macro F1 is evaluated
once on the held-out test split.

This script reports:

    Macro F1
    Micro F1
    AUROC
    Accuracy

The test set is NOT used for model selection.
"""

import json
from pathlib import Path

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

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_NAME = "main"
# simple
# main

BATCH_SIZE = 64

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

REPRESENTATION_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fusion"
    / "ssl_representations"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "fusion"
    / MODEL_NAME
    / "best_model.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fusion"
    / MODEL_NAME
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


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
# Main
# ============================================================

def main():

    print("=" * 60)
    print("MILESTONE 7.3 — TEST EVALUATION")
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
        "Model path:",
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = FusionRepresentationDataset(
        REPRESENTATION_ROOT,
        "test",
    )

    test_loader = DataLoader(
        test_dataset,
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

    model = build_model()

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Best model not found:\n"
            f"{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    # --------------------------------------------------------
    # Support checkpoint format
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_logits = []
    all_labels = []

    with torch.no_grad():

        for batch in tqdm(
            test_loader,
            desc="Test",
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

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = compute_metrics(
        logits,
        labels,
        threshold=0.5,
    )

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"Samples: "
        f"{len(test_dataset)}"
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

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "model": MODEL_NAME,
        "split": "test",
        "samples": len(test_dataset),
        **metrics,
    }

    with open(
        OUTPUT_ROOT
        / "test_metrics.json",
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"Saved test metrics to: "
        f"{OUTPUT_ROOT / 'test_metrics.json'}"
    )


if __name__ == "__main__":
    main()