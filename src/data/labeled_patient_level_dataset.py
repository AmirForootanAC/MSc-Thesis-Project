"""
Labeled Patient-Level Dataset Pipeline

Connects reconstructed labels to the finalized patient-level split.

Inputs:
    results/label_reconstruction/reconstructed_dataset.csv
    results/patient_level_split/patient_split.csv

Outputs:
    results/labeled_patient_level_dataset/
        labeled_dataset.csv
        label_split_distribution.csv
        dataset_summary.json

The patient-level split is treated as finalized and is NOT regenerated
or modified by this module.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

RECONSTRUCTED_DATASET_PATH = Path(
    "results/label_reconstruction/reconstructed_dataset.csv"
)

PATIENT_SPLIT_PATH = Path(
    "results/patient_level_split/patient_split.csv"
)

OUTPUT_DIR = Path(
    "results/labeled_patient_level_dataset"
)

LABELED_DATASET_PATH = (
    OUTPUT_DIR / "labeled_dataset.csv"
)

LABEL_DISTRIBUTION_PATH = (
    OUTPUT_DIR / "label_split_distribution.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "dataset_summary.json"
)


# ============================================================
# Selected Labels
# ============================================================

LABEL_COLUMNS = [
    "label_gingivitis",
    "label_class_ii_malocclusion",
    "label_dental_crowding",
    "label_tooth_structure_loss",
    "label_dental_caries",
    "label_convex_profile",
    "label_mandibular_skeletal_asymmetry",
    "label_periodontitis",
    "label_class_iii_malocclusion",
    "label_pulpitis",
    "label_deep_overbite",
    "label_class_i_malocclusion",
    "label_tooth_loss",
]


LABEL_NAMES = {
    "label_gingivitis":
        "Gingivitis",

    "label_class_ii_malocclusion":
        "Class II Malocclusion",

    "label_dental_crowding":
        "Dental Crowding",

    "label_tooth_structure_loss":
        "Tooth Structure Loss",

    "label_dental_caries":
        "Dental Caries",

    "label_convex_profile":
        "Convex Profile",

    "label_mandibular_skeletal_asymmetry":
        "Mandibular Skeletal Asymmetry",

    "label_periodontitis":
        "Periodontitis",

    "label_class_iii_malocclusion":
        "Class III Malocclusion",

    "label_pulpitis":
        "Pulpitis",

    "label_deep_overbite":
        "Deep Overbite",

    "label_class_i_malocclusion":
        "Class I Malocclusion",

    "label_tooth_loss":
        "Tooth Loss",
}


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Attach reconstructed labels to the finalized "
            "patient-level split."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite previous outputs."
    )

    return parser.parse_args()


# ============================================================
# Output Directory
# ============================================================

def prepare_output_directory(force: bool):
    """
    Prepare output directory.

    Existing outputs require --force.
    """

    if OUTPUT_DIR.exists():

        if not force:
            raise FileExistsError(
                f"{OUTPUT_DIR} already exists. "
                "Use --force to overwrite."
            )

        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Load Inputs
# ============================================================

def load_inputs():
    """
    Load reconstructed dataset and finalized patient split.
    """

    print(
        "[INFO] Loading reconstructed dataset..."
    )

    if not RECONSTRUCTED_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Reconstructed dataset not found: "
            f"{RECONSTRUCTED_DATASET_PATH}"
        )

    reconstructed = pd.read_csv(
        RECONSTRUCTED_DATASET_PATH
    )

    print(
        f"[INFO] Reconstructed visits: "
        f"{len(reconstructed):,}"
    )

    print(
        "[INFO] Loading patient-level split..."
    )

    if not PATIENT_SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Patient split not found: "
            f"{PATIENT_SPLIT_PATH}"
        )

    patient_split = pd.read_csv(
        PATIENT_SPLIT_PATH
    )

    print(
        f"[INFO] Split patients: "
        f"{len(patient_split):,}"
    )

    return reconstructed, patient_split


# ============================================================
# Validate Inputs
# ============================================================

def validate_inputs(
    reconstructed: pd.DataFrame,
    patient_split: pd.DataFrame,
):
    """
    Validate required columns and split integrity.
    """

    required_reconstructed = {
        "patient_id",
        "checkup_id",
    }

    missing_reconstructed = (
        required_reconstructed
        - set(reconstructed.columns)
    )

    if missing_reconstructed:
        raise ValueError(
            "Missing columns in reconstructed dataset: "
            f"{sorted(missing_reconstructed)}"
        )

    missing_labels = [
        column
        for column in LABEL_COLUMNS
        if column not in reconstructed.columns
    ]

    if missing_labels:
        raise ValueError(
            "Missing reconstructed label columns: "
            f"{missing_labels}"
        )

    required_split = {
        "patient_id",
        "split",
    }

    missing_split = (
        required_split
        - set(patient_split.columns)
    )

    if missing_split:
        raise ValueError(
            "Missing columns in patient split: "
            f"{sorted(missing_split)}"
        )

    # --------------------------------------------------------
    # Patient uniqueness
    # --------------------------------------------------------

    duplicate_patients = (
        patient_split["patient_id"]
        .duplicated()
        .sum()
    )

    if duplicate_patients != 0:
        raise ValueError(
            "Patient split contains duplicate patient_id values: "
            f"{duplicate_patients}"
        )

    # --------------------------------------------------------
    # Valid split names
    # --------------------------------------------------------

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        patient_split["split"]
        .dropna()
        .unique()
    )

    invalid_splits = (
        actual_splits
        - valid_splits
    )

    if invalid_splits:
        raise ValueError(
            f"Unexpected split values: {sorted(invalid_splits)}"
        )

    print(
        "[INFO] Input validation: PASS"
    )


# ============================================================
# Attach Patient-Level Split
# ============================================================

def attach_patient_split(
    reconstructed: pd.DataFrame,
    patient_split: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the patient-level split assignment to every visit.

    The split is defined exclusively at patient level:
        patient_id -> split

    Therefore every visit belonging to a patient receives
    the same split assignment.

    Validation:
        - Every reconstructed patient must exist in patient_split.
        - Each patient must have exactly one split.
        - Merge must be many-to-one from visits to patients.
        - No split column from the reconstructed dataset is trusted.
    """

    print("[INFO] Attaching patient-level split...")

    # --------------------------------------------------------
    # Keep only the authoritative patient -> split mapping
    # --------------------------------------------------------

    split_map = patient_split[
        ["patient_id", "split"]
    ].copy()

    # --------------------------------------------------------
    # Validate patient-level split mapping
    # --------------------------------------------------------

    if split_map["patient_id"].duplicated().any():
        duplicated_patients = (
            split_map.loc[
                split_map["patient_id"].duplicated(
                    keep=False
                ),
                "patient_id",
            ]
            .nunique()
        )

        raise ValueError(
            "patient_split.csv contains duplicated patient_id "
            f"entries: {duplicated_patients:,}"
        )

    invalid_splits = set(
        split_map["split"].dropna().unique()
    ) - {
        "train",
        "validation",
        "test",
    }

    if invalid_splits:
        raise ValueError(
            "Unexpected split values found: "
            f"{sorted(invalid_splits)}"
        )

    # --------------------------------------------------------
    # Never use a pre-existing split column from reconstructed
    # dataset. The patient-level split is authoritative.
    # --------------------------------------------------------

    reconstructed = reconstructed.copy()

    if "split" in reconstructed.columns:
        reconstructed = reconstructed.drop(
            columns=["split"]
        )

    # --------------------------------------------------------
    # Validate patient coverage
    # --------------------------------------------------------

    reconstructed_patients = set(
        reconstructed["patient_id"].dropna().unique()
    )

    split_patients = set(
        split_map["patient_id"].dropna().unique()
    )

    missing_patients = (
        reconstructed_patients - split_patients
    )

    if missing_patients:
        raise ValueError(
            "Some reconstructed patients are missing from "
            "patient_split.csv: "
            f"{len(missing_patients):,}"
        )

    # --------------------------------------------------------
    # Attach patient-level split
    # --------------------------------------------------------

    result = reconstructed.merge(
        split_map,
        on="patient_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Validate assignment
    # --------------------------------------------------------

    missing_split = result["split"].isna().sum()

    if missing_split > 0:
        raise ValueError(
            "Some visits could not be assigned to a split: "
            f"{missing_split:,}"
        )

    # --------------------------------------------------------
    # Verify patient consistency
    #
    # Every patient must belong to exactly one split.
    # --------------------------------------------------------

    patient_split_counts = (
        result.groupby("patient_id")["split"]
        .nunique()
    )

    inconsistent_patients = (
        patient_split_counts[
            patient_split_counts > 1
        ]
    )

    if len(inconsistent_patients) > 0:
        raise ValueError(
            "Patient-level split inconsistency detected: "
            f"{len(inconsistent_patients):,} patients "
            "appear in multiple splits."
        )

    print("[INFO] Patient-level split attachment: PASS")

    print(
        result["split"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return result

# ============================================================
# Validate Final Split
# ============================================================

def validate_final_split(
    labeled: pd.DataFrame,
):
    """
    Confirm patient-level isolation after merging labels.
    """

    print(
        "[INFO] Validating final patient-level split..."
    )

    patient_split_counts = (
        labeled
        .groupby("patient_id")["split"]
        .nunique()
    )

    leakage_patients = (
        patient_split_counts[
            patient_split_counts > 1
        ]
    )

    if len(leakage_patients) != 0:
        raise ValueError(
            "Patient-level split leakage detected. "
            f"Patients in multiple splits: "
            f"{len(leakage_patients)}"
        )

    # --------------------------------------------------------
    # Check visit uniqueness
    # --------------------------------------------------------

    duplicate_visits = (
        labeled["checkup_id"]
        .duplicated()
        .sum()
    )

    if duplicate_visits != 0:
        raise ValueError(
            "Duplicate checkup_id values detected: "
            f"{duplicate_visits}"
        )

    print(
        "[INFO] Patient isolation: PASS"
    )

    print(
        "[INFO] Visit uniqueness: PASS"
    )


# ============================================================
# Label Split Distribution
# ============================================================

def generate_label_split_distribution(
    labeled: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate visit-level label distribution across splits.
    """

    print(
        "[INFO] Generating label split distribution..."
    )

    records = []

    for rank, column in enumerate(
        LABEL_COLUMNS,
        start=1
    ):

        label = LABEL_NAMES[column]

        total_visits = int(
            labeled[column].sum()
        )

        total_patients = int(
            labeled.loc[
                labeled[column] == 1,
                "patient_id"
            ].nunique()
        )

        split_counts = {}

        for split_name in [
            "train",
            "validation",
            "test",
        ]:

            split_subset = labeled[
                (labeled["split"] == split_name)
                & (labeled[column] == 1)
            ]

            split_counts[
                f"{split_name}_visits"
            ] = int(
                len(split_subset)
            )

        train_visits = split_counts[
            "train_visits"
        ]

        validation_visits = split_counts[
            "validation_visits"
        ]

        test_visits = split_counts[
            "test_visits"
        ]

        records.append(
            {
                "rank": rank,
                "label": label,

                "train_visits":
                    train_visits,

                "validation_visits":
                    validation_visits,

                "test_visits":
                    test_visits,

                "total_visits":
                    total_visits,

                "patients":
                    total_patients,

                "train_pct":
                    (
                        train_visits
                        / total_visits
                        * 100
                        if total_visits > 0
                        else 0.0
                    ),

                "validation_pct":
                    (
                        validation_visits
                        / total_visits
                        * 100
                        if total_visits > 0
                        else 0.0
                    ),

                "test_pct":
                    (
                        test_visits
                        / total_visits
                        * 100
                        if total_visits > 0
                        else 0.0
                    ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Split Statistics
# ============================================================

def generate_split_statistics(
    labeled: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate overall visit and patient counts per split.
    """

    records = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = labeled[
            labeled["split"] == split_name
        ]

        records.append(
            {
                "split": split_name,

                "patients":
                    subset["patient_id"]
                    .nunique(),

                "visits":
                    len(subset),

                "visits_with_label":
                    int(
                        subset[
                            "has_reconstructed_label"
                        ].sum()
                    ),

                "label_coverage_pct":
                    (
                        subset[
                            "has_reconstructed_label"
                        ].mean()
                        * 100
                    ),

                "multi_label_visits":
                    int(
                        (
                            subset[
                                "reconstructed_label_count"
                            ] > 1
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Final Summary
# ============================================================

def generate_summary(
    labeled: pd.DataFrame,
    label_distribution: pd.DataFrame,
    split_statistics: pd.DataFrame,
) -> dict:
    """
    Generate final dataset summary.
    """

    return {

        "dataset": {
            "total_visits": int(
                len(labeled)
            ),

            "total_patients": int(
                labeled["patient_id"]
                .nunique()
            ),
        },

        "split": {
            "patient_counts": (
                labeled
                .groupby("split")["patient_id"]
                .nunique()
                .to_dict()
            ),

            "visit_counts": (
                labeled["split"]
                .value_counts()
                .to_dict()
            ),
        },

        "labels": {
            "number_of_labels": (
                len(LABEL_COLUMNS)
            ),

            "visits_with_at_least_one_label":
                int(
                    labeled[
                        "has_reconstructed_label"
                    ].sum()
                ),

            "label_coverage_percentage":
                float(
                    labeled[
                        "has_reconstructed_label"
                    ].mean()
                    * 100
                ),

            "multi_label_visits":
                int(
                    (
                        labeled[
                            "reconstructed_label_count"
                        ] > 1
                    ).sum()
                ),
        },

        "label_split_distribution": (
            label_distribution
            .to_dict(orient="records")
        ),

        "split_statistics": (
            split_statistics
            .to_dict(orient="records")
        ),

        "validation": {
            "patient_level_split":
                "PASS",

            "visit_uniqueness":
                "PASS",

            "all_visits_assigned":
                "PASS",
        },
    }


# ============================================================
# Save Outputs
# ============================================================

def save_outputs(
    labeled: pd.DataFrame,
    label_distribution: pd.DataFrame,
    summary: dict,
):
    """
    Save final labeled dataset and audit artifacts.
    """

    print(
        "[INFO] Saving outputs..."
    )

    labeled.to_csv(
        LABELED_DATASET_PATH,
        index=False
    )

    label_distribution.to_csv(
        LABEL_DISTRIBUTION_PATH,
        index=False
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# Console Report
# ============================================================

def print_report(
    labeled: pd.DataFrame,
    label_distribution: pd.DataFrame,
):
    """
    Print final dataset report.
    """

    print()
    print("=" * 100)
    print(
        "LABELED PATIENT-LEVEL DATASET"
    )
    print("=" * 100)

    print(
        f"Total visits   : "
        f"{len(labeled):,}"
    )

    print(
        f"Total patients : "
        f"{labeled['patient_id'].nunique():,}"
    )

    print()

    print(
        f"{'Split':<15}"
        f"{'Patients':>12}"
        f"{'Visits':>12}"
        f"{'Labeled Visits':>18}"
        f"{'Coverage':>12}"
    )

    print("-" * 70)

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = labeled[
            labeled["split"] == split_name
        ]

        labeled_visits = int(
            subset[
                "has_reconstructed_label"
            ].sum()
        )

        coverage = (
            labeled_visits
            / len(subset)
            * 100
        )

        print(
            f"{split_name:<15}"
            f"{subset['patient_id'].nunique():>12,}"
            f"{len(subset):>12,}"
            f"{labeled_visits:>18,}"
            f"{coverage:>11.2f}%"
        )

    print()
    print(
        "=" * 100
    )

    print(
        "LABEL DISTRIBUTION"
    )

    print(
        "=" * 100
    )

    print(
        label_distribution.to_string(
            index=False,
            formatters={
                "train_pct":
                    lambda x: f"{x:.1f}%",

                "validation_pct":
                    lambda x: f"{x:.1f}%",

                "test_pct":
                    lambda x: f"{x:.1f}%",
            }
        )
    )

    print()
    print(
        f"[INFO] Final dataset: "
        f"{LABELED_DATASET_PATH}"
    )

    print(
        f"[INFO] Label distribution: "
        f"{LABEL_DISTRIBUTION_PATH}"
    )

    print(
        f"[INFO] Summary: "
        f"{SUMMARY_PATH}"
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    prepare_output_directory(
        args.force
    )

    reconstructed, patient_split = (
        load_inputs()
    )

    validate_inputs(
        reconstructed,
        patient_split
    )

    labeled = attach_patient_split(
        reconstructed,
        patient_split
    )

    validate_final_split(
        labeled
    )

    label_distribution = (
        generate_label_split_distribution(
            labeled
        )
    )

    split_statistics = (
        generate_split_statistics(
            labeled
        )
    )

    summary = generate_summary(
        labeled,
        label_distribution,
        split_statistics
    )

    save_outputs(
        labeled,
        label_distribution,
        summary
    )

    print_report(
        labeled,
        label_distribution
    )

    print()
    print("=" * 100)
    print(
        "Labeled patient-level dataset "
        "completed successfully."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()