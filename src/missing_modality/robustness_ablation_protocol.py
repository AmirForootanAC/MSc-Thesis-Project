"""
Milestone 8.4.1 — Robustness Ablation Protocol.

Purpose
-------
Define and validate a controlled ablation study comparing:

    1. Standard multimodal fusion
    2. Robust fusion trained with modality dropout

Both models use:
    - identical frozen SSL representations
    - identical train/validation/test populations
    - identical label space
    - identical test scenarios

The only intended difference is the robustness training strategy.

No test data is used for training or model selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "results"
    / "six_label_patient_level_dataset"
    / "labeled_dataset.csv"
)

REPRESENTATION_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fusion"
    / "ssl_representations"
)

ROBUST_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "milestone8_missing_modality"
    / "04_robust_fusion_training"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "milestone8_missing_modality"
    / "06_robustness_ablation"
    / "01_protocol"
)


# ============================================================
# Expected populations
# ============================================================

EXPECTED_SPLITS = {
    "train": 6129,
    "validation": 1330,
    "test": 1316,
}

EXPECTED_COMPLETE_CASES = {
    "train": 2896,
    "validation": 616,
    "test": 625,
}

EXPECTED_REPRESENTATIONS = {
    "train": 2935,
    "validation": 627,
    "test": 633,
}


# ============================================================
# Test scenarios
# ============================================================

TEST_SCENARIOS = {
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
# Dataset loading
# ============================================================

def load_dataset() -> pd.DataFrame:

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    dataset = pd.read_csv(DATASET_PATH)

    required_columns = {
        "patient_id",
        "checkup_id",
        "split",
        "has_six_label",
        "photographs",
        "radiographs",
        "examination",
    }

    missing = required_columns - set(dataset.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    return dataset


# ============================================================
# Modality presence
# ============================================================

def is_present(value) -> bool:

    if pd.isna(value):
        return False

    value = str(value).strip()

    if not value:
        return False

    return value.lower() not in {
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "[]",
    }


def add_modality_presence(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    result = dataset.copy()

    result["image_present"] = (
        result["photographs"]
        .apply(is_present)
    )

    result["radiograph_present"] = (
        result["radiographs"]
        .apply(is_present)
    )

    result["text_present"] = (
        result["examination"]
        .apply(is_present)
    )

    result["complete_case"] = (
        result["image_present"]
        & result["radiograph_present"]
        & result["text_present"]
    )

    return result


# ============================================================
# Validation
# ============================================================

def validate_dataset(
    dataset: pd.DataFrame,
):

    # --------------------------------------------------------
    # Visit uniqueness
    # --------------------------------------------------------

    duplicates = dataset["checkup_id"].duplicated().sum()

    if duplicates:
        raise ValueError(
            f"Duplicate checkup_id values: {duplicates}"
        )

    # --------------------------------------------------------
    # Patient-level split isolation
    # --------------------------------------------------------

    split_counts = (
        dataset
        .groupby("patient_id")["split"]
        .nunique()
    )

    leakage = split_counts[split_counts > 1]

    if len(leakage):
        raise ValueError(
            "Patient-level split leakage detected: "
            f"{len(leakage)} patients."
        )

    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------

    actual = (
        dataset.groupby("split")
        .size()
        .to_dict()
    )

    if actual != EXPECTED_SPLITS:
        raise ValueError(
            "Split population mismatch.\n"
            f"Expected: {EXPECTED_SPLITS}\n"
            f"Found: {actual}"
        )

    # --------------------------------------------------------
    # Complete cases
    # --------------------------------------------------------

    complete = (
        dataset.groupby("split")["complete_case"]
        .sum()
        .astype(int)
        .to_dict()
    )

    if complete != EXPECTED_COMPLETE_CASES:
        raise ValueError(
            "Complete-case population mismatch.\n"
            f"Expected: {EXPECTED_COMPLETE_CASES}\n"
            f"Found: {complete}"
        )


# ============================================================
# Representation validation
# ============================================================

def validate_representations():

    records = []

    for split, expected in EXPECTED_REPRESENTATIONS.items():

        path = (
            REPRESENTATION_ROOT
            / f"{split}.pt"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing representation file:\n{path}"
            )

        data = __import__("torch").load(
            path,
            map_location="cpu",
        )

        count = len(data["checkup_id"])

        if count != expected:
            raise ValueError(
                f"{split} representation population mismatch. "
                f"Expected {expected}, found {count}."
            )

        records.append(
            {
                "split": split,
                "representation_samples": count,
                "path": str(path),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Build protocol tables
# ============================================================

def build_population_table(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for split in [
        "train",
        "validation",
        "test",
    ]:

        subset = dataset[
            dataset["split"] == split
        ]

        records.append(
            {
                "split": split,
                "visits": len(subset),
                "patients": subset["patient_id"].nunique(),
                "complete_cases": int(
                    subset["complete_case"].sum()
                ),
                "natural_missing_image": int(
                    (~subset["image_present"]).sum()
                ),
                "natural_missing_radiograph": int(
                    (~subset["radiograph_present"]).sum()
                ),
                "natural_missing_text": int(
                    (~subset["text_present"]).sum()
                ),
            }
        )

    return pd.DataFrame(records)


def build_scenario_table() -> pd.DataFrame:

    records = []

    for name, scenario in TEST_SCENARIOS.items():

        records.append(
            {
                "scenario": name,
                "description": scenario["description"],
                "image": scenario["mask"][0],
                "radiograph": scenario["mask"][1],
                "text": scenario["mask"][2],
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Summary
# ============================================================

def build_summary(
    population: pd.DataFrame,
    representations: pd.DataFrame,
) -> dict:

    return {
        "milestone": "8.4",
        "experiment": "Robustness Ablation Study",

        "research_question": (
            "Does explicit modality-dropout training improve "
            "robustness to missing modalities compared with "
            "standard multimodal fusion?"
        ),

        "comparison": {
            "standard_fusion": {
                "modality_dropout": False,
                "modality_presence_mask": False,
                "ssl_encoders_frozen": True,
            },
            "robust_fusion": {
                "modality_dropout": True,
                "modality_presence_mask": True,
                "ssl_encoders_frozen": True,
                "checkpoint": str(ROBUST_CHECKPOINT),
            },
        },

        "shared_data": {
            "dataset": str(DATASET_PATH),
            "representations": str(REPRESENTATION_ROOT),
            "train_population": 2935,
            "validation_population": 627,
            "test_population": 633,
        },

        "evaluation": {
            "test_population": 633,
            "test_scenarios": list(TEST_SCENARIOS.keys()),
            "model_selection_metric": "validation_macro_f1",
            "test_used_for_model_selection": False,
        },

        "split_policy": {
            "patient_level_split_inherited": True,
            "regenerated": False,
            "leakage_check": "PASS",
        },

        "validation": {
            "dataset_population": "PASS",
            "complete_case_population": "PASS",
            "representation_population": "PASS",
            "patient_level_isolation": "PASS",
            "robust_checkpoint_exists": ROBUST_CHECKPOINT.exists(),
        },
    }


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "MILESTONE 8.4.1 — ROBUSTNESS ABLATION PROTOCOL"
    )
    print("=" * 100)

    print("\n[INFO] Loading authoritative dataset...")

    dataset = load_dataset()

    dataset = add_modality_presence(dataset)

    print(
        f"[INFO] Visits: {len(dataset):,}"
    )

    print(
        f"[INFO] Patients: "
        f"{dataset['patient_id'].nunique():,}"
    )

    print("\n[INFO] Validating dataset...")

    validate_dataset(dataset)

    print(
        "[INFO] Dataset validation: PASS"
    )

    print("\n[INFO] Validating SSL representations...")

    representations = validate_representations()

    print(
        representations.to_string(index=False)
    )

    print(
        "\n[INFO] Representation validation: PASS"
    )

    population = build_population_table(dataset)

    scenarios = build_scenario_table()

    summary = build_summary(
        population,
        representations,
    )

    population.to_csv(
        OUTPUT_DIR / "population.csv",
        index=False,
    )

    representations.to_csv(
        OUTPUT_DIR / "representations.csv",
        index=False,
    )

    scenarios.to_csv(
        OUTPUT_DIR / "test_scenarios.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "protocol_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "-" * 100)
    print("Population")
    print("-" * 100)

    print(
        population.to_string(index=False)
    )

    print("\n" + "-" * 100)
    print("Test scenarios")
    print("-" * 100)

    print(
        scenarios.to_string(index=False)
    )

    print("\n" + "=" * 100)
    print(
        "MILESTONE 8.4.1 PROTOCOL VALIDATION COMPLETE"
    )
    print("=" * 100)

    print(
        f"\n[INFO] Results saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()