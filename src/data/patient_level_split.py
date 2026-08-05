"""
Patient-Level Dataset Split Pipeline

Creates a reproducible patient-level train/validation/test split
for the COde dental dataset.

Split policy:
- Split unit: patient_id
- Ratios:
    train: 70%
    validation: 15%
    test: 15%
- Random seed: 42

Stratification:
- Patient-level diagnosis availability
- Patient-level radiograph availability

This module only creates split artifacts.
Leakage validation and split quality auditing are handled separately.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path(
    "data/raw/COde-Dataset/complete_dataset.csv"
)

OUTPUT_DIR = Path(
    "results/patient_level_split"
)

DEFAULT_SEED = 42


# ============================================================
# Utility Functions
# ============================================================

def create_output_directory(force: bool):
    """
    Create output directory.

    Existing outputs require --force.
    """

    if OUTPUT_DIR.exists() and not force:
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Use --force to overwrite."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def load_dataset():
    """
    Load COde dataset.

    Existing dataset split column is ignored.
    """

    print(
        f"[INFO] Loading dataset: {DATA_PATH}"
    )

    df = pd.read_csv(DATA_PATH)

    print(
        f"[INFO] Loaded {len(df):,} rows."
    )

    # Remove original COde split if exists.
    # The thesis split must be generated independently.
    df = df.drop(
        columns=["split"],
        errors="ignore"
    )

    return df


def build_patient_table(df: pd.DataFrame):
    """
    Create patient-level metadata table.

    Features:
    - diagnosis availability
    - radiograph availability
    - number of visits

    Used for stratified patient split.
    """

    print(
        "[INFO] Building patient-level table..."
    )

    patient_table = (
        df.groupby("patient_id")
        .agg(
            diagnosis_available=(
                "diagnosis",
                lambda x: x.notna().any()
            ),
            radiograph_available=(
                "radiographs",
                lambda x: x.notna().any()
            ),
            num_visits=(
                "checkup_id",
                "count"
            )
        )
        .reset_index()
    )

    patient_table["stratify_label"] = (
        patient_table["diagnosis_available"]
        .astype(str)
        + "_"
        + patient_table["radiograph_available"]
        .astype(str)
    )

    print(
        f"[INFO] Patients: {len(patient_table)}"
    )

    print(
        "[INFO] Stratification groups:"
    )

    print(
        patient_table["stratify_label"]
        .value_counts()
    )

    return patient_table


def create_patient_split(
    patient_table: pd.DataFrame,
    seed: int
):
    """
    Generate train/validation/test split
    at patient level.
    """

    print(
        "[INFO] Creating patient-level split..."
    )

    train_patients, temp_patients = train_test_split(
        patient_table,
        test_size=0.30,
        random_state=seed,
        stratify=patient_table["stratify_label"]
    )

    validation_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=seed,
        stratify=temp_patients["stratify_label"]
    )

    train_patients = train_patients.copy()
    validation_patients = validation_patients.copy()
    test_patients = test_patients.copy()

    train_patients["split"] = "train"
    validation_patients["split"] = "validation"
    test_patients["split"] = "test"

    patient_split = pd.concat(
        [
            train_patients,
            validation_patients,
            test_patients
        ],
        ignore_index=True
    )

    return patient_split


def save_outputs(
    df: pd.DataFrame,
    patient_split: pd.DataFrame
):
    """
    Save generated split artifacts.
    """

    print(
        "[INFO] Saving outputs..."
    )

    patient_split_output = (
        patient_split[
            [
                "patient_id",
                "split"
            ]
        ]
        .sort_values("patient_id")
    )

    patient_split_output.to_csv(
        OUTPUT_DIR / "patient_split.csv",
        index=False
    )


    visit_split = df.merge(
        patient_split_output,
        on="patient_id",
        how="left"
    )


    visit_split.to_csv(
        OUTPUT_DIR / "visit_split.csv",
        index=False
    )


    summary = {

        "seed": DEFAULT_SEED,

        "split_ratios": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15
        },

        "patient_counts": (
            patient_split_output["split"]
            .value_counts()
            .to_dict()
        ),

        "visit_counts": (
            visit_split["split"]
            .value_counts()
            .to_dict()
        ),

        "total_patients": int(
            patient_split_output["patient_id"]
            .nunique()
        ),

        "total_visits": int(
            len(visit_split)
        )
    }


    with open(
        OUTPUT_DIR / "split_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create patient-level split for COde dataset."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED
    )

    parser.add_argument(
        "--force",
        action="store_true"
    )

    args = parser.parse_args()


    create_output_directory(
        args.force
    )


    df = load_dataset()


    patient_table = build_patient_table(
        df
    )


    patient_split = create_patient_split(
        patient_table,
        args.seed
    )


    save_outputs(
        df,
        patient_split
    )


    print()
    print("=" * 70)
    print(
        "Patient-level split completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":

    main()