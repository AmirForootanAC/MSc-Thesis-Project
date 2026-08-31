"""
Milestone 8.3.1 — Robust Fusion Training Protocol

Defines and validates the population protocol for the
missing-modality robust fusion model.

Authoritative dataset:
    results/six_label_patient_level_dataset/labeled_dataset.csv

Important:
    - Patient-level split is inherited and never regenerated.
    - Test population is never used for training or validation.
    - SSL representations are inherited from Milestone 7.
    - No labels or modalities are reconstructed here.
    - This module only defines/validates the experimental protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = Path(
    "results/six_label_patient_level_dataset/labeled_dataset.csv"
)

REPRESENTATION_ROOT = Path(
    "results/fusion/ssl_representations"
)

OUTPUT_DIR = Path(
    "results/milestone8_missing_modality/03_robust_fusion_protocol"
)

PROTOCOL_SUMMARY_PATH = (
    OUTPUT_DIR / "protocol_summary.json"
)

POPULATION_PATH = (
    OUTPUT_DIR / "population_by_split.csv"
)

TRAINING_SCENARIOS_PATH = (
    OUTPUT_DIR / "training_scenarios.csv"
)

VALIDATION_SCENARIOS_PATH = (
    OUTPUT_DIR / "validation_scenarios.csv"
)


# ============================================================
# Modality definitions
# ============================================================

MODALITIES = [
    "image",
    "radiograph",
    "text",
]

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
        "missing": ["radiograph", "text"],
    },
}


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate Milestone 8.3.1 robust-fusion "
            "training and evaluation protocol."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing protocol outputs.",
    )

    return parser.parse_args()


# ============================================================
# Output preparation
# ============================================================

def prepare_output_directory(force: bool):

    if OUTPUT_DIR.exists():

        if not force:
            raise FileExistsError(
                f"{OUTPUT_DIR} already exists. "
                "Use --force to overwrite."
            )

        for path in OUTPUT_DIR.iterdir():

            if path.is_file():
                path.unlink()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Dataset loading
# ============================================================

def load_dataset() -> pd.DataFrame:

    print(
        "[INFO] Loading authoritative six-label dataset..."
    )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataset = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"[INFO] Visits   : {len(dataset):,}"
    )

    print(
        f"[INFO] Patients : "
        f"{dataset['patient_id'].nunique():,}"
    )

    return dataset


# ============================================================
# Dataset validation
# ============================================================

def validate_dataset(
    dataset: pd.DataFrame,
):

    print(
        "[INFO] Validating authoritative dataset..."
    )

    required_columns = {
        "patient_id",
        "checkup_id",
        "split",
        "has_six_label",
        "photographs",
        "radiographs",
        "examination",
    }

    missing = (
        required_columns
        - set(dataset.columns)
    )

    if missing:
        raise ValueError(
            "Missing required dataset columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Visit uniqueness
    # --------------------------------------------------------

    duplicate_visits = (
        dataset["checkup_id"]
        .duplicated()
        .sum()
    )

    if duplicate_visits:
        raise ValueError(
            "Duplicate checkup_id values detected: "
            f"{duplicate_visits:,}"
        )

    # --------------------------------------------------------
    # Patient-level split isolation
    # --------------------------------------------------------

    patient_split_counts = (
        dataset
        .groupby("patient_id")["split"]
        .nunique()
    )

    leakage = patient_split_counts[
        patient_split_counts > 1
    ]

    if len(leakage):
        raise ValueError(
            "Patient-level split leakage detected: "
            f"{len(leakage):,} patients."
        )

    # --------------------------------------------------------
    # Split values
    # --------------------------------------------------------

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        dataset["split"]
        .dropna()
        .unique()
    )

    invalid = (
        actual_splits
        - expected_splits
    )

    if invalid:
        raise ValueError(
            "Invalid split values: "
            f"{sorted(invalid)}"
        )

    print(
        "[INFO] Input validation: PASS"
    )


# ============================================================
# Modality presence
# ============================================================

def is_present(value) -> bool:

    if pd.isna(value):
        return False

    value = str(value).strip()

    if not value:
        return False

    normalized = value.lower()

    if normalized in {
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "[]",
    }:
        return False

    return True


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

    return result


# ============================================================
# Natural modality pattern
# ============================================================

def modality_code(row) -> str:

    return (
        f"{int(row['image_present'])}"
        f"{int(row['radiograph_present'])}"
        f"{int(row['text_present'])}"
    )


def add_modality_patterns(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    result = dataset.copy()

    result["natural_modality_pattern"] = (
        result.apply(
            modality_code,
            axis=1,
        )
    )

    return result


# ============================================================
# Population statistics
# ============================================================

def generate_population_statistics(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = dataset[
            dataset["split"] == split_name
        ]

        total = len(subset)

        complete = (
            subset[
                "natural_modality_pattern"
            ] == "111"
        ).sum()

        missing_xray = (
            subset[
                "radiograph_present"
            ] == False
        ).sum()

        missing_text = (
            subset[
                "text_present"
            ] == False
        ).sum()

        missing_image = (
            subset[
                "image_present"
            ] == False
        ).sum()

        labeled = (
            subset[
                "has_six_label"
            ] == 1
        ).sum()

        records.append(
            {
                "split": split_name,
                "visits": int(total),
                "patients": int(
                    subset["patient_id"]
                    .nunique()
                ),
                "labeled_visits": int(labeled),
                "complete_cases": int(complete),
                "missing_xray": int(missing_xray),
                "missing_text": int(missing_text),
                "missing_image": int(missing_image),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Training protocol
# ============================================================

def build_training_scenarios(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    train = dataset[
        dataset["split"] == "train"
    ]

    records = []

    for scenario_name, config in SCENARIOS.items():

        missing = config["missing"]

        records.append(
            {
                "scenario": scenario_name,
                "description": config["description"],
                "source_split": "train",
                "population_policy": (
                    "Natural samples plus "
                    "controlled missingness generated "
                    "from complete training cases."
                ),
                "missing_modalities": (
                    ",".join(missing)
                    if missing
                    else ""
                ),
                "complete_training_cases": int(
                    (
                        train[
                            "natural_modality_pattern"
                        ] == "111"
                    ).sum()
                ),
                "natural_training_samples": int(
                    len(train)
                ),
                "controlled_generation": (
                    "yes"
                    if missing
                    else "no"
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Validation protocol
# ============================================================

def build_validation_scenarios(
    dataset: pd.DataFrame,
) -> pd.DataFrame:

    validation = dataset[
        dataset["split"] == "validation"
    ]

    records = []

    for scenario_name, config in SCENARIOS.items():

        missing = config["missing"]

        records.append(
            {
                "scenario": scenario_name,
                "description": config["description"],
                "source_split": "validation",
                "population_policy": (
                    "Fixed deterministic controlled "
                    "missingness generated from "
                    "complete validation cases."
                ),
                "missing_modalities": (
                    ",".join(missing)
                    if missing
                    else ""
                ),
                "complete_validation_cases": int(
                    (
                        validation[
                            "natural_modality_pattern"
                        ] == "111"
                    ).sum()
                ),
                "controlled_generation": (
                    "yes"
                    if missing
                    else "no"
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Protocol validation
# ============================================================

def validate_protocol(
    dataset: pd.DataFrame,
    population: pd.DataFrame,
):

    # --------------------------------------------------------
    # Authoritative dataset
    # --------------------------------------------------------

    if DATASET_PATH != Path(
        "results/six_label_patient_level_dataset/labeled_dataset.csv"
    ):
        raise ValueError(
            "Authoritative dataset path changed."
        )

    # --------------------------------------------------------
    # Split populations
    # --------------------------------------------------------

    expected_train = 6129
    expected_validation = 1330
    expected_test = 1316

    actual = dict(
        zip(
            population["split"],
            population["visits"],
        )
    )

    expected = {
        "train": expected_train,
        "validation": expected_validation,
        "test": expected_test,
    }

    if actual != expected:
        raise ValueError(
            "Split population mismatch.\n"
            f"Expected: {expected}\n"
            f"Found: {actual}"
        )

    # --------------------------------------------------------
    # Complete-case populations
    # --------------------------------------------------------

    expected_complete = {
        "train": 2896,
        "validation": 616,
        "test": 625,
    }

    actual_complete = dict(
        zip(
            population["split"],
            population["complete_cases"],
        )
    )

    if actual_complete != expected_complete:
        raise ValueError(
            "Complete-case population mismatch.\n"
            f"Expected: {expected_complete}\n"
            f"Found: {actual_complete}"
        )

    # --------------------------------------------------------
    # Test isolation
    # --------------------------------------------------------

    test = dataset[
        dataset["split"] == "test"
    ]

    if len(test) != expected_test:
        raise ValueError(
            "Test population changed."
        )

    print(
        "[INFO] Protocol population validation: PASS"
    )


# ============================================================
# Summary
# ============================================================

def build_summary(
    dataset: pd.DataFrame,
    population: pd.DataFrame,
) -> dict:

    return {

        "milestone": "8.3.1",

        "experiment":
            "Robust Fusion Dataset and Protocol",

        "purpose": (
            "Define and validate the training and "
            "validation population protocol for "
            "missing-modality robust fusion."
        ),

        "authoritative_dataset":
            str(DATASET_PATH),

        "representation_root":
            str(REPRESENTATION_ROOT),

        "model_basis":
            "Milestone 7.3 MainFusion",

        "labels": [
            "label_caries",
            "label_gingivitis",
            "label_malocclusion",
            "label_pulpitis",
            "label_tooth_loss",
            "label_tooth_structure_loss",
        ],

        "split_policy": {
            "patient_level_split":
                "Inherited from finalized dataset",

            "regeneration":
                False,

            "train_visits":
                int(
                    len(
                        dataset[
                            dataset["split"]
                            == "train"
                        ]
                    )
                ),

            "validation_visits":
                int(
                    len(
                        dataset[
                            dataset["split"]
                            == "validation"
                        ]
                    )
                ),

            "test_visits":
                int(
                    len(
                        dataset[
                            dataset["split"]
                            == "test"
                        ]
                    )
                ),
        },

        "training_policy": {
            "natural_missingness":
                True,

            "controlled_missingness":
                True,

            "source_for_controlled_missingness":
                "Complete training cases",

            "test_used_for_training":
                False,

            "test_used_for_model_selection":
                False,
        },

        "validation_policy": {
            "controlled_missingness":
                True,

            "deterministic":
                True,

            "source":
                "Complete validation cases",

            "test_used":
                False,
        },

        "test_policy": {
            "population":
                "Unmodified held-out test split",

            "training":
                False,

            "model_selection":
                False,
        },

        "missingness_scenarios":
            SCENARIOS,

        "validation": {
            "authoritative_dataset":
                "PASS",

            "patient_level_split":
                "PASS",

            "visit_uniqueness":
                "PASS",

            "population_counts":
                "PASS",

            "complete_case_counts":
                "PASS",

            "test_isolation":
                "PASS",

            "dataset_modified":
                False,
        },
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    prepare_output_directory(
        args.force
    )

    dataset = load_dataset()

    validate_dataset(
        dataset
    )

    dataset = add_modality_presence(
        dataset
    )

    dataset = add_modality_patterns(
        dataset
    )

    population = (
        generate_population_statistics(
            dataset
        )
    )

    validate_protocol(
        dataset,
        population,
    )

    training_scenarios = (
        build_training_scenarios(
            dataset
        )
    )

    validation_scenarios = (
        build_validation_scenarios(
            dataset
        )
    )

    summary = build_summary(
        dataset,
        population,
    )

    population.to_csv(
        POPULATION_PATH,
        index=False,
    )

    training_scenarios.to_csv(
        TRAINING_SCENARIOS_PATH,
        index=False,
    )

    validation_scenarios.to_csv(
        VALIDATION_SCENARIOS_PATH,
        index=False,
    )

    with open(
        PROTOCOL_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 110)
    print(
        "MILESTONE 8.3.1 — ROBUST FUSION PROTOCOL"
    )
    print("=" * 110)

    print()
    print(
        population.to_string(
            index=False
        )
    )

    print()
    print(
        "[INFO] Protocol validation: PASS"
    )

    print(
        f"[INFO] Results saved to: {OUTPUT_DIR}"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()