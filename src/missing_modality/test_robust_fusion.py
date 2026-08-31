"""
Milestone 8.3.3 — Robust Fusion Test.

Purpose
-------
Evaluate the best RobustFusion checkpoint from Milestone 8.3.2
under controlled missing-modality conditions.

The SSL representations remain frozen.

The test population is the exact complete-case test population
used by the previous fusion milestones.

Evaluation scenarios
---------------------
    Complete
    Image missing
    Radiograph missing
    Text missing
    Image + Radiograph missing
    Image + Text missing
    Radiograph + Text missing

For every scenario:
    1. The corresponding modality representation is zeroed.
    2. The explicit modality-presence mask is updated.
    3. Predictions are generated.
    4. Macro F1, Micro F1, AUROC, and Accuracy are computed.

The test set is used only for final evaluation.
No model selection or parameter tuning is performed here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
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

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 64

REPRESENTATION_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "fusion"
    / "ssl_representations"
)

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "milestone8_missing_modality"
    / "04_robust_fusion_training"
    / "best_model.pt"
)

RESULT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "milestone8_missing_modality"
    / "05_robust_fusion_test"
)


# ============================================================
# Test scenarios
# ============================================================

SCENARIOS = {
    "complete": {
        "description": "All modalities available",
        "mask": [1, 1, 1],
    },

    "image_missing": {
        "description": "Image missing",
        "mask": [0, 1, 1],
    },

    "radiograph_missing": {
        "description": "Radiograph missing",
        "mask": [1, 0, 1],
    },

    "text_missing": {
        "description": "Clinical text missing",
        "mask": [1, 1, 0],
    },

    "image_radiograph_missing": {
        "description": "Image and radiograph missing",
        "mask": [0, 0, 1],
    },

    "image_text_missing": {
        "description": "Image and text missing",
        "mask": [0, 1, 0],
    },

    "radiograph_text_missing": {
        "description": "Radiograph and text missing",
        "mask": [1, 0, 0],
    },
}


# ============================================================
# Load model
# ============================================================

def load_model():

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model = RobustFusion(
        image_dim=2048,
        radiograph_dim=2048,
        text_dim=768,
        modality_dim=512,
        hidden_dim=512,
        num_labels=config.NUM_LABELS,
        dropout=0.3,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


# ============================================================
# Apply scenario
# ============================================================

def apply_scenario(
    image,
    radiograph,
    text,
    mask_values,
):
    """
    Apply a deterministic missing-modality scenario.

    Both representations and the explicit modality mask
    are modified.
    """

    batch_size = image.shape[0]

    mask = torch.tensor(
        mask_values,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0).expand(
        batch_size,
        -1,
    )

    image = (
        image
        * mask[:, 0:1]
    )

    radiograph = (
        radiograph
        * mask[:, 1:2]
    )

    text = (
        text
        * mask[:, 2:3]
    )

    return (
        image,
        radiograph,
        text,
        mask,
    )


# ============================================================
# Evaluate one scenario
# ============================================================

@torch.no_grad()
def evaluate_scenario(
    model,
    loader,
    mask_values,
):
    """
    Evaluate one deterministic missing-modality scenario.
    """

    all_logits = []
    all_labels = []

    for batch in tqdm(
        loader,
        desc="Evaluating",
        leave=False,
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

        (
            image,
            radiograph,
            text,
            modality_mask,
        ) = apply_scenario(
            image,
            radiograph,
            text,
            mask_values,
        )

        logits = model(
            image,
            radiograph,
            text,
            modality_mask,
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

    metrics = compute_metrics(
        logits,
        labels,
        threshold=0.5,
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 100)
    print(
        "MILESTONE 8.3.3 — ROBUST FUSION TEST"
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
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Result root:",
        RESULT_ROOT,
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not REPRESENTATION_ROOT.exists():

        raise FileNotFoundError(
            "Representation root not found: "
            f"{REPRESENTATION_ROOT}"
        )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = (
        RobustFusionRepresentationDataset(
            REPRESENTATION_ROOT,
            "test",
        )
    )

    print()
    print(
        f"test: {len(test_dataset)} "
        "SSL representation samples"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

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

    model, checkpoint = load_model()

    print()
    print(
        "Loaded checkpoint:"
    )

    print(
        "  Model:",
        checkpoint.get(
            "model_name",
            "unknown",
        ),
    )

    print(
        "  Epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "  Validation Macro F1:",
        f"{checkpoint.get('validation_metrics', {}).get('macro_f1', float('nan')):.4f}",
    )

    # --------------------------------------------------------
    # Result directory
    # --------------------------------------------------------

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Evaluate scenarios
    # --------------------------------------------------------

    results = []

    print()
    print("-" * 100)

    for scenario_name, scenario in SCENARIOS.items():

        print()
        print(
            f"Scenario: {scenario_name}"
        )

        print(
            f"Description: "
            f"{scenario['description']}"
        )

        print(
            f"Mask: "
            f"{scenario['mask']}"
        )

        metrics = evaluate_scenario(
            model,
            test_loader,
            scenario["mask"],
        )

        result = {
            "scenario": scenario_name,

            "description":
                scenario["description"],

            "image_available":
                bool(scenario["mask"][0]),

            "radiograph_available":
                bool(scenario["mask"][1]),

            "text_available":
                bool(scenario["mask"][2]),

            "samples":
                len(test_dataset),

            "macro_f1":
                metrics["macro_f1"],

            "micro_f1":
                metrics["micro_f1"],

            "auroc":
                metrics["auroc"],

            "accuracy":
                metrics["accuracy"],
        }

        results.append(result)

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

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output = {
        "milestone":
            "8.3.3",

        "experiment":
            "Robust Fusion Test",

        "representation_root":
            str(REPRESENTATION_ROOT),

        "checkpoint":
            str(CHECKPOINT_PATH),

        "checkpoint_epoch":
            checkpoint.get(
                "epoch"
            ),

        "checkpoint_validation_macro_f1":
            checkpoint.get(
                "validation_metrics",
                {},
            ).get(
                "macro_f1"
            ),

        "test_samples":
            len(test_dataset),

        "batch_size":
            BATCH_SIZE,

        "device":
            DEVICE,

        "threshold":
            0.5,

        "model_selection_on_test":
            False,

        "scenarios":
            results,
    }

    with open(
        RESULT_ROOT / "test_results.json",
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    dataframe = pd.DataFrame(
        results
    )

    dataframe.to_csv(
        RESULT_ROOT / "test_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    test_config = {
        "milestone":
            "8.3.3",

        "experiment":
            "Robust Fusion Test",

        "device":
            DEVICE,

        "batch_size":
            BATCH_SIZE,

        "representation_root":
            str(REPRESENTATION_ROOT),

        "checkpoint_path":
            str(CHECKPOINT_PATH),

        "test_population":
            "Complete-case test population from Milestone 7 SSL representations",

        "test_samples":
            len(test_dataset),

        "threshold":
            0.5,

        "model_selection_on_test":
            False,

        "ssl_encoders_frozen":
            True,

        "scenarios":
            SCENARIOS,
    }

    with open(
        RESULT_ROOT / "config.json",
        "w",
    ) as f:

        json.dump(
            test_config,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "MILESTONE 8.3.3 TEST COMPLETE"
    )
    print("=" * 100)

    print()

    print(
        dataframe[
            [
                "scenario",
                "macro_f1",
                "micro_f1",
                "auroc",
                "accuracy",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULT_ROOT
    )


if __name__ == "__main__":
    main()