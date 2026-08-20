"""
Six-Label Patient-Level Dataset Pipeline

Creates a six-label benchmark dataset from the finalized
13-label patient-level dataset.

Input:
    results/labeled_patient_level_dataset/labeled_dataset.csv

Output:
    results/six_label_patient_level_dataset/
        labeled_dataset.csv
        label_split_distribution.csv
        dataset_summary.json

The existing patient-level split is authoritative and is NOT
regenerated or modified by this module.

Six benchmark labels:
    - Caries
    - Gingivitis
    - Malocclusion
    - Pulpitis
    - Tooth Loss
    - Tooth Structure Loss

Malocclusion is reconstructed as:

    Class I Malocclusion
    OR Class II Malocclusion
    OR Class III Malocclusion
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

INPUT_DATASET_PATH = Path(
    "results/labeled_patient_level_dataset/labeled_dataset.csv"
)

OUTPUT_DIR = Path(
    "results/six_label_patient_level_dataset"
)

OUTPUT_DATASET_PATH = (
    OUTPUT_DIR / "labeled_dataset.csv"
)

LABEL_DISTRIBUTION_PATH = (
    OUTPUT_DIR / "label_split_distribution.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "dataset_summary.json"
)


# ============================================================
# Source Labels
# ============================================================

SOURCE_LABEL_COLUMNS = [
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


# ============================================================
# Final Six Labels
# ============================================================

FINAL_LABEL_COLUMNS = [
    "label_caries",
    "label_gingivitis",
    "label_malocclusion",
    "label_pulpitis",
    "label_tooth_loss",
    "label_tooth_structure_loss",
]


LABEL_NAMES = {
    "label_caries":
        "Caries",

    "label_gingivitis":
        "Gingivitis",

    "label_malocclusion":
        "Malocclusion",

    "label_pulpitis":
        "Pulpitis",

    "label_tooth_loss":
        "Tooth Loss",

    "label_tooth_structure_loss":
        "Tooth Structure Loss",
}


# ============================================================
# Malocclusion Source Labels
# ============================================================

MALOCCLUSION_SOURCE_LABELS = [
    "label_class_i_malocclusion",
    "label_class_ii_malocclusion",
    "label_class_iii_malocclusion",
]


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create the six-label patient-level benchmark "
            "dataset from the finalized 13-label dataset."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite previous outputs.",
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
        exist_ok=True,
    )


# ============================================================
# Load Input
# ============================================================

def load_input_dataset() -> pd.DataFrame:
    """
    Load the finalized 13-label patient-level dataset.
    """

    print(
        "[INFO] Loading finalized 13-label dataset..."
    )

    if not INPUT_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Input dataset not found: "
            f"{INPUT_DATASET_PATH}"
        )

    dataset = pd.read_csv(
        INPUT_DATASET_PATH
    )

    print(
        f"[INFO] Input visits: "
        f"{len(dataset):,}"
    )

    print(
        f"[INFO] Input patients: "
        f"{dataset['patient_id'].nunique():,}"
    )

    return dataset


# ============================================================
# Validate Input
# ============================================================

def validate_input(
    dataset: pd.DataFrame,
):
    """
    Validate the finalized 13-label dataset before conversion.
    """

    required_columns = {
        "patient_id",
        "checkup_id",
        "split",
        "has_reconstructed_label",
        "reconstructed_label_count",
    }

    missing_required = (
        required_columns
        - set(dataset.columns)
    )

    if missing_required:
        raise ValueError(
            "Missing required columns in input dataset: "
            f"{sorted(missing_required)}"
        )

    missing_source_labels = [
        column
        for column in SOURCE_LABEL_COLUMNS
        if column not in dataset.columns
    ]

    if missing_source_labels:
        raise ValueError(
            "Missing source label columns: "
            f"{missing_source_labels}"
        )

    # --------------------------------------------------------
    # Patient uniqueness within split
    # --------------------------------------------------------

    patient_split_counts = (
        dataset
        .groupby("patient_id")["split"]
        .nunique()
    )

    inconsistent_patients = (
        patient_split_counts[
            patient_split_counts > 1
        ]
    )

    if len(inconsistent_patients) != 0:
        raise ValueError(
            "Patient-level split inconsistency detected: "
            f"{len(inconsistent_patients):,} patients "
            "appear in multiple splits."
        )

    # --------------------------------------------------------
    # Visit uniqueness
    # --------------------------------------------------------

    duplicate_visits = (
        dataset["checkup_id"]
        .duplicated()
        .sum()
    )

    if duplicate_visits != 0:
        raise ValueError(
            "Duplicate checkup_id values detected: "
            f"{duplicate_visits:,}"
        )

    # --------------------------------------------------------
    # Split validity
    # --------------------------------------------------------

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        dataset["split"]
        .dropna()
        .unique()
    )

    invalid_splits = (
        actual_splits
        - valid_splits
    )

    if invalid_splits:
        raise ValueError(
            "Unexpected split values: "
            f"{sorted(invalid_splits)}"
        )

    print(
        "[INFO] Input validation: PASS"
    )


# ============================================================
# Build Six Labels
# ============================================================

def build_six_label_dataset(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the finalized 13-label dataset into the
    six-label benchmark dataset.

    Only the label representation changes.

    Patient identities, visits, modalities, text, and split
    assignments are preserved.
    """

    print(
        "[INFO] Building six-label benchmark dataset..."
    )

    result = dataset.copy()

    # --------------------------------------------------------
    # Construct final six labels
    # --------------------------------------------------------

    result["label_caries"] = (
        result["label_dental_caries"]
        .astype(int)
    )

    result["label_gingivitis"] = (
        result["label_gingivitis"]
        .astype(int)
    )

    result["label_malocclusion"] = (
        (
            result["label_class_i_malocclusion"].astype(bool)
            |
            result["label_class_ii_malocclusion"].astype(bool)
            |
            result["label_class_iii_malocclusion"].astype(bool)
        )
        .astype(int)
    )

    result["label_pulpitis"] = (
        result["label_pulpitis"]
        .astype(int)
    )

    result["label_tooth_loss"] = (
        result["label_tooth_loss"]
        .astype(int)
    )

    result["label_tooth_structure_loss"] = (
        result["label_tooth_structure_loss"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Generate new six-label statistics BEFORE removing the
    # original 13-label columns.
    # --------------------------------------------------------

    result["six_label_count"] = (
        result[FINAL_LABEL_COLUMNS]
        .sum(axis=1)
        .astype(int)
    )

    result["has_six_label"] = (
        result["six_label_count"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Remove old 13-label representation.
    #
    # Some final six-label columns share the same names as their
    # original source labels. Therefore, preserve all final labels
    # when dropping the old 13-label representation.
    # --------------------------------------------------------

    columns_to_drop = [
        column
        for column in SOURCE_LABEL_COLUMNS
        if column not in FINAL_LABEL_COLUMNS
    ]

    result = result.drop(
        columns=columns_to_drop,
        errors="raise",
    )

    print(
        "[INFO] Six-label construction: PASS"
    )

    return result


# ============================================================
# Validate Six Labels
# ============================================================

def validate_six_label_dataset(
    original: pd.DataFrame,
    six_label: pd.DataFrame,
):
    """
    Validate that the six-label conversion preserved the
    dataset structure and correctly constructed Malocclusion.
    """

    print(
        "[INFO] Validating six-label dataset..."
    )

    # --------------------------------------------------------
    # Same number of visits
    # --------------------------------------------------------

    if len(original) != len(six_label):
        raise ValueError(
            "Visit count changed during six-label conversion: "
            f"{len(original):,} -> {len(six_label):,}"
        )

    # --------------------------------------------------------
    # Same patient set
    # --------------------------------------------------------

    original_patients = set(
        original["patient_id"]
        .unique()
    )

    final_patients = set(
        six_label["patient_id"]
        .unique()
    )

    if original_patients != final_patients:
        raise ValueError(
            "Patient set changed during six-label conversion."
        )

    # --------------------------------------------------------
    # Same visit set
    # --------------------------------------------------------

    original_visits = set(
        original["checkup_id"]
        .unique()
    )

    final_visits = set(
        six_label["checkup_id"]
        .unique()
    )

    if original_visits != final_visits:
        raise ValueError(
            "Visit set changed during six-label conversion."
        )

    # --------------------------------------------------------
    # Same split assignment
    # --------------------------------------------------------

    original_split = (
        original[
            ["checkup_id", "split"]
        ]
        .sort_values("checkup_id")
        .reset_index(drop=True)
    )

    final_split = (
        six_label[
            ["checkup_id", "split"]
        ]
        .sort_values("checkup_id")
        .reset_index(drop=True)
    )

    if not original_split.equals(final_split):
        raise ValueError(
            "Patient-level split assignments changed."
        )

    # --------------------------------------------------------
    # Required final labels
    # --------------------------------------------------------

    missing_labels = [
        label
        for label in FINAL_LABEL_COLUMNS
        if label not in six_label.columns
    ]

    if missing_labels:
        raise ValueError(
            "Missing final six-label columns: "
            f"{missing_labels}"
        )

    # --------------------------------------------------------
    # Binary validation
    # --------------------------------------------------------

    for label in FINAL_LABEL_COLUMNS:

        unique_values = set(
            six_label[label]
            .dropna()
            .unique()
        )

        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Non-binary values found in {label}: "
                f"{sorted(unique_values)}"
            )

    # --------------------------------------------------------
    # Six-label count validation
    # --------------------------------------------------------

    expected_count = (
        six_label[
            FINAL_LABEL_COLUMNS
        ]
        .sum(axis=1)
        .astype(int)
    )

    actual_count = (
        six_label[
            "six_label_count"
        ]
        .astype(int)
    )

    if not expected_count.equals(actual_count):
        raise ValueError(
            "six_label_count validation failed."
        )

    expected_has_label = (
        expected_count > 0
    ).astype(int)

    actual_has_label = (
        six_label[
            "has_six_label"
        ]
        .astype(int)
    )

    if not expected_has_label.equals(
        actual_has_label
    ):
        raise ValueError(
            "has_six_label validation failed."
        )

    # --------------------------------------------------------
    # Explicit Malocclusion validation
    # --------------------------------------------------------

    expected_malocclusion = (
        (
            original[
                "label_class_i_malocclusion"
            ].astype(bool)
            |
            original[
                "label_class_ii_malocclusion"
            ].astype(bool)
            |
            original[
                "label_class_iii_malocclusion"
            ].astype(bool)
        )
        .astype(int)
    )

    actual_malocclusion = (
        six_label[
            "label_malocclusion"
        ]
        .reset_index(drop=True)
    )

    expected_malocclusion = (
        expected_malocclusion
        .reset_index(drop=True)
    )

    if not expected_malocclusion.equals(
        actual_malocclusion
    ):
        raise ValueError(
            "Malocclusion OR mapping validation failed."
        )

    # --------------------------------------------------------
    # Patient-level split isolation
    # --------------------------------------------------------

    patient_split_counts = (
        six_label
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
            "Patient-level split leakage detected after "
            "six-label conversion: "
            f"{len(leakage_patients):,} patients."
        )

    # --------------------------------------------------------
    # Old 13-label columns must not remain
    # --------------------------------------------------------

    remaining_old_labels = [
        column
        for column in SOURCE_LABEL_COLUMNS
        if column not in MALOCCLUSION_SOURCE_LABELS
        and column in six_label.columns
        and column not in FINAL_LABEL_COLUMNS
    ]

    if remaining_old_labels:
        raise ValueError(
            "Unexpected old label columns remain: "
            f"{remaining_old_labels}"
        )

    print(
        "[INFO] Six-label validation: PASS"
    )


# ============================================================
# Label Split Distribution
# ============================================================

def generate_label_split_distribution(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate visit-level label distribution across splits.
    """

    print(
        "[INFO] Generating six-label split distribution..."
    )

    records = []

    for rank, column in enumerate(
        FINAL_LABEL_COLUMNS,
        start=1,
    ):

        label = LABEL_NAMES[column]

        total_visits = int(
            dataset[column].sum()
        )

        total_patients = int(
            dataset.loc[
                dataset[column] == 1,
                "patient_id",
            ].nunique()
        )

        split_counts = {}

        for split_name in [
            "train",
            "validation",
            "test",
        ]:

            subset = dataset[
                (dataset["split"] == split_name)
                &
                (dataset[column] == 1)
            ]

            split_counts[
                f"{split_name}_visits"
            ] = int(len(subset))

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
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate overall visit and patient statistics per split.
    """

    records = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = dataset[
            dataset["split"] == split_name
        ]

        records.append(
            {
                "split": split_name,

                "patients":
                    subset["patient_id"]
                    .nunique(),

                "visits":
                    len(subset),

                "visits_with_six_label":
                    int(
                        subset[
                            "has_six_label"
                        ].sum()
                    ),

                "label_coverage_pct":
                    (
                        subset[
                            "has_six_label"
                        ].mean()
                        * 100
                    ),

                "multi_label_visits":
                    int(
                        (
                            subset[
                                "six_label_count"
                            ] > 1
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Summary
# ============================================================

def generate_summary(
    dataset: pd.DataFrame,
    label_distribution: pd.DataFrame,
    split_statistics: pd.DataFrame,
) -> dict:
    """
    Generate final six-label dataset summary.
    """

    return {

        "dataset": {
            "source":
                str(INPUT_DATASET_PATH),

            "total_visits":
                int(len(dataset)),

            "total_patients":
                int(
                    dataset["patient_id"]
                    .nunique()
                ),

            "number_of_labels":
                len(FINAL_LABEL_COLUMNS),

            "labels":
                FINAL_LABEL_COLUMNS,
        },

        "malocclusion_mapping": {
            "target":
                "label_malocclusion",

            "sources":
                MALOCCLUSION_SOURCE_LABELS,

            "operation":
                "logical OR",
        },

        "split": {
            "patient_counts": (
                dataset
                .groupby("split")["patient_id"]
                .nunique()
                .to_dict()
            ),

            "visit_counts": (
                dataset["split"]
                .value_counts()
                .to_dict()
            ),
        },

        "labels": {
            "visits_with_at_least_one_label":
                int(
                    dataset[
                        "has_six_label"
                    ].sum()
                ),

            "label_coverage_percentage":
                float(
                    dataset[
                        "has_six_label"
                    ].mean()
                    * 100
                ),

            "multi_label_visits":
                int(
                    (
                        dataset[
                            "six_label_count"
                        ] > 1
                    ).sum()
                ),
        },

        "label_split_distribution":
            label_distribution
            .to_dict(
                orient="records"
            ),

        "split_statistics":
            split_statistics
            .to_dict(
                orient="records"
            ),

        "validation": {
            "patient_level_split":
                "PASS",

            "visit_uniqueness":
                "PASS",

            "patient_set_preserved":
                "PASS",

            "visit_set_preserved":
                "PASS",

            "split_assignments_preserved":
                "PASS",

            "malocclusion_mapping":
                "PASS",
        },
    }


# ============================================================
# Save Outputs
# ============================================================

def save_outputs(
    dataset: pd.DataFrame,
    label_distribution: pd.DataFrame,
    summary: dict,
):
    """
    Save final six-label dataset and summary artifacts.
    """

    print(
        "[INFO] Saving outputs..."
    )

    dataset.to_csv(
        OUTPUT_DATASET_PATH,
        index=False,
    )

    label_distribution.to_csv(
        LABEL_DISTRIBUTION_PATH,
        index=False,
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# Console Report
# ============================================================

def print_report(
    dataset: pd.DataFrame,
    label_distribution: pd.DataFrame,
):
    """
    Print final six-label dataset report.
    """

    print()
    print("=" * 100)
    print(
        "SIX-LABEL PATIENT-LEVEL DATASET"
    )
    print("=" * 100)

    print(
        f"Total visits   : "
        f"{len(dataset):,}"
    )

    print(
        f"Total patients : "
        f"{dataset['patient_id'].nunique():,}"
    )

    print(
        "Labels         : "
        f"{len(FINAL_LABEL_COLUMNS)}"
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

        subset = dataset[
            dataset["split"] == split_name
        ]

        labeled_visits = int(
            subset[
                "has_six_label"
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
    print("=" * 100)
    print(
        "LABEL DISTRIBUTION"
    )
    print("=" * 100)

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
            },
        )
    )

    print()
    print(
        f"[INFO] Final dataset: "
        f"{OUTPUT_DATASET_PATH}"
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

    original = load_input_dataset()

    validate_input(
        original
    )

    six_label = build_six_label_dataset(
        original
    )

    validate_six_label_dataset(
        original,
        six_label,
    )

    label_distribution = (
        generate_label_split_distribution(
            six_label
        )
    )

    split_statistics = (
        generate_split_statistics(
            six_label
        )
    )

    summary = generate_summary(
        six_label,
        label_distribution,
        split_statistics,
    )

    save_outputs(
        six_label,
        label_distribution,
        summary,
    )

    print_report(
        six_label,
        label_distribution,
    )

    print()
    print("=" * 100)
    print(
        "Six-label patient-level dataset "
        "completed successfully."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()