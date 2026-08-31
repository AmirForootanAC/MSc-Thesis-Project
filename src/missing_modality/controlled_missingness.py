"""
Milestone 8.2 — Controlled Missing-Modality Evaluation.

Purpose
-------
Evaluate the existing Milestone 7.3 SSL Fusion — Main model
under controlled missing-modality conditions.

IMPORTANT
---------
This is an evaluation-only experiment.

The model is:
    - NOT retrained
    - NOT fine-tuned
    - NOT modified

The existing MainFusion checkpoint from Milestone 7.3 is used.

Experimental population
-----------------------
Only the complete-case test population is used.

Therefore every scenario contains exactly the same test visits.
Only the modality representation supplied to the already-trained
fusion model is changed.

Scenarios
---------
A — Complete:
    Image + X-ray + Text

B — X-ray Missing:
    Image + Text

C — Text Missing:
    Image + X-ray

D — Multiple Missing:
    Image only

Missing representations are replaced by zero vectors.

This provides a controlled baseline for measuring the performance
degradation caused by missing modalities before designing a
missing-modality-robust fusion model.

Output
------
results/milestone8_missing_modality/02_controlled_missingness/

    controlled_missingness_results.json
    controlled_missingness_results.csv
"""


from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.baseline import config
from src.baseline.metrics import compute_metrics

from src.fusion.dataset import (
    FusionRepresentationDataset,
)

from src.fusion.fusion_model import (
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

BATCH_SIZE = 64

MODEL_NAME = "main"

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

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
    / "main"
    / "best_model.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "milestone8_missing_modality"
    / "02_controlled_missingness"
)


# ============================================================
# Controlled scenarios
# ============================================================

SCENARIOS = {
    "A_complete": {
        "description": "Image + X-ray + Text",
        "missing": [],
    },

    "B_missing_xray": {
        "description": "Image + Text",
        "missing": ["radiograph"],
    },

    "C_missing_text": {
        "description": "Image + X-ray",
        "missing": ["text"],
    },

    "D_missing_multiple": {
        "description": "Image Only",
        "missing": [
            "radiograph",
            "text",
        ],
    },
}


# ============================================================
# Model
# ============================================================

def build_model():
    """
    Build exactly the Milestone 7.3 MainFusion architecture.
    """

    return MainFusion(
        image_dim=2048,
        radiograph_dim=2048,
        text_dim=768,
        modality_dim=512,
        hidden_dim=512,
        num_labels=config.NUM_LABELS,
        dropout=0.3,
    )


# ============================================================
# Checkpoint
# ============================================================

def load_model():
    """
    Load the existing Milestone 7.3 MainFusion checkpoint.

    No training or fine-tuning is performed.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "MainFusion checkpoint not found:\n"
            f"{MODEL_PATH}"
        )

    model = build_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
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

    return model


# ============================================================
# Dataset
# ============================================================

def load_test_dataset():
    """
    Load the exact complete-case test population used by
    Milestone 7.3 Fusion.

    Expected:
        625 test samples.
    """

    dataset = FusionRepresentationDataset(
        REPRESENTATION_ROOT,
        "test",
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Complete-case test dataset is empty."
        )

    return dataset


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_scenario(
    model,
    dataset,
    missing_modalities,
):
    """
    Evaluate one controlled missing-modality scenario.

    Missing representations are replaced with zero vectors.

    The labels and sample population remain unchanged.
    """

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
    )

    all_logits = []
    all_labels = []

    for batch in tqdm(
        loader,
        desc="Evaluating",
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

        # ----------------------------------------------------
        # Controlled modality removal
        # ----------------------------------------------------

        if "image" in missing_modalities:

            image = torch.zeros_like(
                image
            )

        if "radiograph" in missing_modalities:

            radiograph = torch.zeros_like(
                radiograph
            )

        if "text" in missing_modalities:

            text = torch.zeros_like(
                text
            )

        # ----------------------------------------------------
        # Existing MainFusion model
        # ----------------------------------------------------

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

    metrics = compute_metrics(
        logits,
        labels,
        threshold=0.5,
    )

    return metrics


# ============================================================
# Performance drop
# ============================================================

def calculate_drop(
    complete_metrics,
    scenario_metrics,
):
    """
    Calculate absolute performance degradation relative
    to the complete-case scenario.

    Drop is reported in percentage points.

    Example:
        complete Macro F1 = 0.7676
        missing Macro F1 = 0.7000

        drop = 0.0676
    """

    return {
        "macro_f1_drop":
            complete_metrics["macro_f1"]
            - scenario_metrics["macro_f1"],

        "micro_f1_drop":
            complete_metrics["micro_f1"]
            - scenario_metrics["micro_f1"],

        "auroc_drop":
            complete_metrics["auroc"]
            - scenario_metrics["auroc"],

        "accuracy_drop":
            complete_metrics["accuracy"]
            - scenario_metrics["accuracy"],
    }


# ============================================================
# Validation
# ============================================================

def validate_population(
    dataset,
):
    """
    Verify that the controlled experiment uses exactly
    the complete-case test population used by Milestone 7.3
    SSL Fusion.

    The authoritative Fusion representation dataset contains
    633 complete-case test samples.

    No population is reconstructed or regenerated here.
    """

    expected_test_samples = 633

    if len(dataset) != expected_test_samples:

        raise ValueError(
            "Controlled missingness population mismatch.\n"
            f"Expected Milestone 7.3 Fusion test population: "
            f"{expected_test_samples}\n"
            f"Found: {len(dataset)}"
        )

    checkup_ids = [
        str(value)
        for value in dataset.checkup_ids
    ]

    if len(checkup_ids) != len(
        set(checkup_ids)
    ):

        raise ValueError(
            "Duplicate checkup_id values detected "
            "in the controlled test population."
        )

    print(
        "[INFO] Complete-case test population: PASS"
    )

    print(
        f"[INFO] Test samples: {len(dataset)}"
    )

# ============================================================
# Save results
# ============================================================

def save_results(
    results,
):
    """
    Save JSON and CSV summaries.
    """

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        OUTPUT_ROOT
        / "controlled_missingness_results.json"
    )

    csv_path = (
        OUTPUT_ROOT
        / "controlled_missingness_results.csv"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False,
        )

    rows = []

    for scenario_name, record in (
        results["scenarios"].items()
    ):

        row = {
            "scenario":
                scenario_name,

            "description":
                record["description"],

            "missing_modalities":
                ",".join(
                    record[
                        "missing_modalities"
                    ]
                ),

            "samples":
                record["samples"],

            "macro_f1":
                record["metrics"]["macro_f1"],

            "micro_f1":
                record["metrics"]["micro_f1"],

            "auroc":
                record["metrics"]["auroc"],

            "accuracy":
                record["metrics"]["accuracy"],

            "macro_f1_drop":
                record["performance_drop"][
                    "macro_f1_drop"
                ],

            "micro_f1_drop":
                record["performance_drop"][
                    "micro_f1_drop"
                ],

            "auroc_drop":
                record["performance_drop"][
                    "auroc_drop"
                ],

            "accuracy_drop":
                record["performance_drop"][
                    "accuracy_drop"
                ],
        }

        rows.append(row)

    pd.DataFrame(
        rows
    ).to_csv(
        csv_path,
        index=False,
    )

    print()
    print(
        f"[INFO] JSON saved to: {json_path}"
    )

    print(
        f"[INFO] CSV saved to: {csv_path}"
    )


# ============================================================
# Console report
# ============================================================

def print_report(
    results,
):
    """
    Print compact comparison table.
    """

    print()
    print(
        "=" * 110
    )

    print(
        "MILESTONE 8.2 — CONTROLLED MISSING-MODALITY EVALUATION"
    )

    print(
        "=" * 110
    )

    print()

    print(
        f"{'Scenario':<24}"
        f"{'Samples':>10}"
        f"{'Macro F1':>12}"
        f"{'Micro F1':>12}"
        f"{'AUROC':>12}"
        f"{'Accuracy':>12}"
        f"{'Macro Drop':>14}"
    )

    print(
        "-" * 110
    )

    for scenario_name, record in (
        results["scenarios"].items()
    ):

        metrics = record[
            "metrics"
        ]

        drop = record[
            "performance_drop"
        ]

        print(
            f"{scenario_name:<24}"
            f"{record['samples']:>10}"
            f"{metrics['macro_f1']:>12.4f}"
            f"{metrics['micro_f1']:>12.4f}"
            f"{metrics['auroc']:>12.4f}"
            f"{metrics['accuracy']:>12.4f}"
            f"{drop['macro_f1_drop']:>14.4f}"
        )

    print()

    print(
        "Positive Drop = performance degradation "
        "relative to Complete."
    )

    print(
        "Missing representations are replaced by zero vectors."
    )

    print(
        "=" * 110
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 110
    )

    print(
        "MILESTONE 8.2 — CONTROLLED MISSING-MODALITY EVALUATION"
    )

    print(
        "=" * 110
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Model: SSL Fusion — Main"
    )

    print(
        f"Checkpoint: {MODEL_PATH}"
    )

    print(
        f"Representation root: "
        f"{REPRESENTATION_ROOT}"
    )

    print()

    # --------------------------------------------------------
    # Load complete-case test population
    # --------------------------------------------------------

    dataset = load_test_dataset()

    validate_population(
        dataset
    )

    # --------------------------------------------------------
    # Load existing MainFusion checkpoint
    # --------------------------------------------------------

    model = load_model()

    print(
        "[INFO] Existing MainFusion checkpoint loaded."
    )

    print(
        "[INFO] No training or fine-tuning will be performed."
    )

    # --------------------------------------------------------
    # Evaluate scenarios
    # --------------------------------------------------------

    scenario_results = {}

    complete_metrics = None

    for scenario_name, scenario in (
        SCENARIOS.items()
    ):

        print()
        print(
            "-" * 80
        )

        print(
            f"Scenario: {scenario_name}"
        )

        print(
            f"Input: {scenario['description']}"
        )

        if scenario["missing"]:

            print(
                "Missing: "
                + ", ".join(
                    scenario["missing"]
                )
            )

        else:

            print(
                "Missing: none"
            )

        print(
            "-" * 80
        )

        metrics = evaluate_scenario(
            model,
            dataset,
            scenario["missing"],
        )

        if scenario_name == "A_complete":

            complete_metrics = metrics

        scenario_results[
            scenario_name
        ] = {
            "description":
                scenario["description"],

            "missing_modalities":
                scenario["missing"],

            "samples":
                len(dataset),

            "metrics":
                metrics,
        }

    # --------------------------------------------------------
    # Calculate performance degradation
    # --------------------------------------------------------

    for scenario_name, record in (
        scenario_results.items()
    ):

        record[
            "performance_drop"
        ] = calculate_drop(
            complete_metrics,
            record["metrics"],
        )

    # --------------------------------------------------------
    # Final result object
    # --------------------------------------------------------

    results = {

        "milestone":
            "8.2",

        "experiment":
            "Controlled Missing-Modality Evaluation",

        "purpose":
            (
                "Evaluate the existing Milestone 7.3 "
                "SSL Fusion Main model under controlled "
                "missing-modality conditions before "
                "designing a robust missing-modality model."
            ),

        "model":
            {
                "name":
                    "SSL Fusion — Main",

                "architecture":
                    (
                        "Image 2048->512, "
                        "Radiograph 2048->512, "
                        "Text 768->512, "
                        "Concat 1536->512->6"
                    ),

                "checkpoint":
                    str(MODEL_PATH),

                "training_performed":
                    False,

                "fine_tuning_performed":
                    False,
            },

        "population":
            {
                "dataset":
                    (
                        "Complete-case test population "
                        "from Milestone 7.3"
                    ),

                "representation_root":
                    str(REPRESENTATION_ROOT),

                "samples":
                    len(dataset),

                "same_population_across_scenarios":
                    True,
            },

        "missing_modality_method":
            {
                "method":
                    "zero_representation",

                "description":
                    (
                        "The representation of a missing "
                        "modality is replaced with a zero "
                        "vector of the corresponding "
                        "representation dimension."
                    ),

                "retraining":
                    False,

                "imputation":
                    False,
            },

        "scenarios":
            scenario_results,

        "validation":
            {
                "complete_case_population":
                    "PASS",

                "same_population_across_scenarios":
                    "PASS",

                "patient_level_split":
                    "Inherited from Milestone 7",

                "test_used_for_model_selection":
                    False,

                "model_modified":
                    False,

                "model_retrained":
                    False,
            },
    }

    save_results(
        results
    )

    print_report(
        results
    )


if __name__ == "__main__":
    main()