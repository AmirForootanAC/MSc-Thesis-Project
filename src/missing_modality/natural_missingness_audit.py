"""
Milestone 8.1 — Natural Missingness Audit
==========================================

Purpose
-------
Audit naturally occurring modality missingness in the authoritative
six-label patient-level benchmark dataset.

This module DOES NOT:
    - regenerate patient-level splits
    - modify the dataset
    - impute missing modalities
    - remove samples
    - train a model
    - create controlled missingness

It only describes the naturally occurring modality-presence patterns
that will define the population for later Milestone 8 experiments.

Authoritative input
-------------------
results/six_label_patient_level_dataset/labeled_dataset.csv

Outputs
-------
results/milestone8_missing_modality/01_natural_missingness/
    natural_missingness_summary.json
    modality_patterns.csv
    modality_patterns_by_split.csv
    label_distribution_by_pattern.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_DATASET = Path(
    "results/six_label_patient_level_dataset/labeled_dataset.csv"
)

OUTPUT_DIR = Path(
    "results/milestone8_missing_modality"
    "/01_natural_missingness"
)

SUMMARY_PATH = OUTPUT_DIR / "natural_missingness_summary.json"
PATTERNS_PATH = OUTPUT_DIR / "modality_patterns.csv"
PATTERNS_SPLIT_PATH = (
    OUTPUT_DIR / "modality_patterns_by_split.csv"
)
LABEL_DISTRIBUTION_PATH = (
    OUTPUT_DIR / "label_distribution_by_pattern.csv"
)


FINAL_LABEL_COLUMNS = [
    "label_caries",
    "label_gingivitis",
    "label_malocclusion",
    "label_pulpitis",
    "label_tooth_loss",
    "label_tooth_structure_loss",
]


LABEL_NAMES = {
    "label_caries": "Caries",
    "label_gingivitis": "Gingivitis",
    "label_malocclusion": "Malocclusion",
    "label_pulpitis": "Pulpitis",
    "label_tooth_loss": "Tooth Loss",
    "label_tooth_structure_loss": "Tooth Structure Loss",
}


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit natural modality missingness in the "
            "authoritative six-label patient-level dataset."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Milestone 8.1 outputs.",
    )

    return parser.parse_args()


# ============================================================
# Output Directory
# ============================================================

def prepare_output_directory(force: bool):
    """
    Prepare the Milestone 8.1 output directory.
    """

    if OUTPUT_DIR.exists():

        existing_files = list(
            OUTPUT_DIR.iterdir()
        )

        if existing_files and not force:
            raise FileExistsError(
                f"Output directory already contains files: "
                f"{OUTPUT_DIR}. "
                "Use --force to overwrite."
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Dataset Loading
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the authoritative six-label benchmark dataset.
    """

    print(
        "[INFO] Loading authoritative six-label dataset..."
    )

    if not INPUT_DATASET.exists():
        raise FileNotFoundError(
            "Authoritative six-label dataset not found:\n"
            f"{INPUT_DATASET}"
        )

    dataset = pd.read_csv(
        INPUT_DATASET
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
# Input Validation
# ============================================================

def validate_dataset(
    dataset: pd.DataFrame,
):
    """
    Validate that the dataset is suitable for the audit.
    """

    print(
        "[INFO] Validating input dataset..."
    )

    required_columns = {
        "patient_id",
        "checkup_id",
        "split",
        "photographs",
        "radiographs",
        "examination",
        "has_six_label",
        "six_label_count",
        *FINAL_LABEL_COLUMNS,
    }

    missing_columns = sorted(
        required_columns
        - set(dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "Required columns are missing:\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Visit uniqueness
    # --------------------------------------------------------

    duplicate_visits = int(
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
    # Patient-level split isolation
    # --------------------------------------------------------

    patient_split_counts = (
        dataset
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
            "Patient-level split leakage detected: "
            f"{len(leakage_patients):,} patients."
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

    unexpected_splits = (
        actual_splits
        - expected_splits
    )

    if unexpected_splits:
        raise ValueError(
            "Unexpected split values: "
            f"{sorted(unexpected_splits)}"
        )

    # --------------------------------------------------------
    # Six-label validation
    # --------------------------------------------------------

    for column in FINAL_LABEL_COLUMNS:

        values = set(
            dataset[column]
            .dropna()
            .unique()
        )

        if not values.issubset({0, 1}):
            raise ValueError(
                f"Non-binary values found in {column}: "
                f"{sorted(values)}"
            )

    print(
        "[INFO] Input validation: PASS"
    )


# ============================================================
# Modality Presence
# ============================================================

def is_missing_value(value) -> bool:
    """
    Generic missing-value detection.

    Empty strings and common textual missing markers are treated
    as missing.
    """

    if value is None:
        return True

    if pd.isna(value):
        return True

    normalized = str(value).strip().lower()

    return normalized in {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
    }


def has_modality(value) -> bool:
    """
    Determine whether a modality field contains actual data.

    For photographs/radiographs, the dataset stores comma-separated
    file names. For text, a non-empty string indicates presence.
    """

    if is_missing_value(value):
        return False

    text = str(value).strip()

    if not text:
        return False

    return True


def add_modality_presence(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add explicit modality-presence columns.
    """

    result = dataset.copy()

    result["has_image"] = (
        result["photographs"]
        .apply(has_modality)
    )

    result["has_xray"] = (
        result["radiographs"]
        .apply(has_modality)
    )

    result["has_text"] = (
        result["examination"]
        .apply(has_modality)
    )

    return result


# ============================================================
# Pattern Definition
# ============================================================

def build_pattern(row) -> str:
    """
    Build a human-readable modality-presence pattern.

    Format:
        Image + X-ray + Text
        Image + Text
        Image
        Image + X-ray
        etc.
    """

    modalities = []

    if row["has_image"]:
        modalities.append("Image")

    if row["has_xray"]:
        modalities.append("X-ray")

    if row["has_text"]:
        modalities.append("Text")

    if not modalities:
        return "No Modality"

    return " + ".join(modalities)


def build_pattern_code(row) -> str:
    """
    Build a compact binary modality code.

    Order:
        Image / X-ray / Text

    Examples:
        111 -> Image + X-ray + Text
        101 -> Image + Text
        100 -> Image only
        110 -> Image + X-ray
    """

    return (
        f"{int(row['has_image'])}"
        f"{int(row['has_xray'])}"
        f"{int(row['has_text'])}"
    )


def add_patterns(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add modality pattern information.
    """

    result = dataset.copy()

    result["modality_pattern_code"] = (
        result.apply(
            build_pattern_code,
            axis=1,
        )
    )

    result["modality_pattern"] = (
        result.apply(
            build_pattern,
            axis=1,
        )
    )

    result["missing_modality_count"] = (
        (~result["has_image"]).astype(int)
        +
        (~result["has_xray"]).astype(int)
        +
        (~result["has_text"]).astype(int)
    )

    result["available_modality_count"] = (
        result["has_image"].astype(int)
        +
        result["has_xray"].astype(int)
        +
        result["has_text"].astype(int)
    )

    result["is_complete_case"] = (
        result["modality_pattern_code"] == "111"
    ).astype(int)

    return result


# ============================================================
# Overall Pattern Statistics
# ============================================================

def generate_pattern_statistics(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate overall modality-pattern statistics.
    """

    total_visits = len(dataset)

    records = []

    grouped = (
        dataset
        .groupby(
            [
                "modality_pattern_code",
                "modality_pattern",
                "missing_modality_count",
                "available_modality_count",
            ],
            dropna=False,
        )
        .agg(
            visits=("checkup_id", "size"),
            patients=("patient_id", "nunique"),
            labeled_visits=("has_six_label", "sum"),
        )
        .reset_index()
    )

    grouped["percentage"] = (
        grouped["visits"]
        / total_visits
        * 100
    )

    grouped["label_coverage_percentage"] = (
        grouped["labeled_visits"]
        / grouped["visits"]
        * 100
    )

    grouped = grouped.sort_values(
        "modality_pattern_code",
        ascending=False,
    )

    return grouped.reset_index(
        drop=True
    )


# ============================================================
# Pattern Statistics by Split
# ============================================================

def generate_pattern_split_statistics(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate modality-pattern statistics separately for
    train, validation, and test.
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

        total_visits = len(subset)

        grouped = (
            subset
            .groupby(
                [
                    "modality_pattern_code",
                    "modality_pattern",
                ],
                dropna=False,
            )
            .agg(
                visits=("checkup_id", "size"),
                patients=("patient_id", "nunique"),
                labeled_visits=("has_six_label", "sum"),
            )
            .reset_index()
        )

        grouped["split"] = split_name

        grouped["percentage_of_split"] = (
            grouped["visits"]
            / total_visits
            * 100
        )

        grouped["label_coverage_percentage"] = (
            grouped["labeled_visits"]
            / grouped["visits"]
            * 100
        )

        records.append(
            grouped
        )

    result = pd.concat(
        records,
        ignore_index=True,
    )

    return result[
        [
            "split",
            "modality_pattern_code",
            "modality_pattern",
            "visits",
            "patients",
            "labeled_visits",
            "percentage_of_split",
            "label_coverage_percentage",
        ]
    ]


# ============================================================
# Label Distribution by Pattern
# ============================================================

def generate_label_distribution(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate six-label prevalence within each natural
    modality-presence pattern.

    This is descriptive only. No filtering or balancing occurs.
    """

    records = []

    grouped = dataset.groupby(
        [
            "modality_pattern_code",
            "modality_pattern",
        ],
        dropna=False,
    )

    for (
        pattern_code,
        pattern_name,
    ), subset in grouped:

        total_visits = len(subset)

        for label_column in FINAL_LABEL_COLUMNS:

            positive_visits = int(
                subset[label_column].sum()
            )

            prevalence = (
                positive_visits
                / total_visits
                * 100
                if total_visits > 0
                else 0.0
            )

            records.append(
                {
                    "modality_pattern_code":
                        pattern_code,

                    "modality_pattern":
                        pattern_name,

                    "label_column":
                        label_column,

                    "label_name":
                        LABEL_NAMES[label_column],

                    "visits":
                        total_visits,

                    "positive_visits":
                        positive_visits,

                    "prevalence_percentage":
                        prevalence,
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# Scenario Mapping
# ============================================================

def assign_reference_scenario(
    pattern_code: str,
) -> str:
    """
    Map naturally occurring patterns to the predefined
    Milestone 8 reference scenarios.

    This mapping is descriptive only.

    A = complete multimodal
    B = X-ray missing
    C = image only
    D = text missing

    Other patterns remain explicitly marked as OTHER.
    """

    mapping = {
        "111": "Scenario A — Complete",
        "101": "Scenario B — X-ray Missing",
        "100": "Scenario C — Image Only",
        "110": "Scenario D — Text Missing",
    }

    return mapping.get(
        pattern_code,
        "Other Natural Pattern",
    )


# ============================================================
# Summary
# ============================================================

def generate_summary(
    dataset: pd.DataFrame,
    pattern_statistics: pd.DataFrame,
) -> dict:
    """
    Generate JSON summary of the natural missingness audit.
    """

    scenario_statistics = []

    for _, row in pattern_statistics.iterrows():

        scenario = assign_reference_scenario(
            row["modality_pattern_code"]
        )

        scenario_statistics.append(
            {
                "modality_pattern_code":
                    row["modality_pattern_code"],

                "modality_pattern":
                    row["modality_pattern"],

                "reference_scenario":
                    scenario,

                "visits":
                    int(row["visits"]),

                "patients":
                    int(row["patients"]),

                "percentage":
                    float(row["percentage"]),

                "labeled_visits":
                    int(row["labeled_visits"]),

                "label_coverage_percentage":
                    float(
                        row[
                            "label_coverage_percentage"
                        ]
                    ),
            }
        )

    split_statistics = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = dataset[
            dataset["split"] == split_name
        ]

        split_statistics[split_name] = {
            "visits":
                int(len(subset)),

            "patients":
                int(
                    subset["patient_id"]
                    .nunique()
                ),

            "complete_cases":
                int(
                    subset["is_complete_case"]
                    .sum()
                ),

            "missing_xray":
                int(
                    (
                        ~subset["has_xray"]
                    ).sum()
                ),

            "missing_text":
                int(
                    (
                        ~subset["has_text"]
                    ).sum()
                ),

            "missing_image":
                int(
                    (
                        ~subset["has_image"]
                    ).sum()
                ),
        }

    return {
        "milestone": "8.1",
        "experiment": "Natural Missingness Audit",

        "purpose": (
            "Descriptive audit of naturally occurring modality "
            "missingness in the authoritative six-label "
            "patient-level benchmark dataset."
        ),

        "authoritative_dataset":
            str(INPUT_DATASET),

        "dataset": {
            "total_visits":
                int(len(dataset)),

            "total_patients":
                int(
                    dataset["patient_id"]
                    .nunique()
                ),

            "six_labels":
                FINAL_LABEL_COLUMNS,
        },

        "modality_presence": {
            "image_present":
                int(dataset["has_image"].sum()),

            "image_missing":
                int(
                    (~dataset["has_image"]).sum()
                ),

            "xray_present":
                int(dataset["has_xray"].sum()),

            "xray_missing":
                int(
                    (~dataset["has_xray"]).sum()
                ),

            "text_present":
                int(dataset["has_text"].sum()),

            "text_missing":
                int(
                    (~dataset["has_text"]).sum()
                ),
        },

        "pattern_count":
            int(
                dataset[
                    "modality_pattern_code"
                ].nunique()
            ),

        "natural_patterns":
            scenario_statistics,

        "split_statistics":
            split_statistics,

        "validation": {
            "patient_level_split":
                "PASS",

            "visit_uniqueness":
                "PASS",

            "authoritative_six_label_dataset":
                "PASS",

            "dataset_modified":
                False,

            "imputation_performed":
                False,

            "samples_removed":
                False,
        },
    }


# ============================================================
# Save
# ============================================================

def save_outputs(
    pattern_statistics: pd.DataFrame,
    pattern_split_statistics: pd.DataFrame,
    label_distribution: pd.DataFrame,
    summary: dict,
):
    """
    Save all Milestone 8.1 audit artifacts.
    """

    pattern_statistics.to_csv(
        PATTERNS_PATH,
        index=False,
    )

    pattern_split_statistics.to_csv(
        PATTERNS_SPLIT_PATH,
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
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# Console Report
# ============================================================

def print_report(
    dataset: pd.DataFrame,
    pattern_statistics: pd.DataFrame,
):
    """
    Print concise audit report.
    """

    print()
    print("=" * 100)
    print(
        "MILESTONE 8.1 — NATURAL MISSINGNESS AUDIT"
    )
    print("=" * 100)

    print(
        f"Dataset visits   : {len(dataset):,}"
    )

    print(
        f"Dataset patients : "
        f"{dataset['patient_id'].nunique():,}"
    )

    print()

    print(
        f"{'Code':<8}"
        f"{'Pattern':<32}"
        f"{'Visits':>10}"
        f"{'Patients':>12}"
        f"{'Percent':>12}"
        f"{'Labeled':>12}"
    )

    print("-" * 86)

    for _, row in pattern_statistics.iterrows():

        print(
            f"{row['modality_pattern_code']:<8}"
            f"{row['modality_pattern']:<32}"
            f"{int(row['visits']):>10,}"
            f"{int(row['patients']):>12,}"
            f"{row['percentage']:>11.2f}%"
            f"{int(row['labeled_visits']):>12,}"
        )

    print()

    print(
        "[INFO] Complete cases: "
        f"{int(dataset['is_complete_case'].sum()):,}"
    )

    print(
        "[INFO] Missing X-ray: "
        f"{int((~dataset['has_xray']).sum()):,}"
    )

    print(
        "[INFO] Missing Text: "
        f"{int((~dataset['has_text']).sum()):,}"
    )

    print(
        "[INFO] Missing Image: "
        f"{int((~dataset['has_image']).sum()):,}"
    )

    print()
    print(
        f"[INFO] Results saved to: {OUTPUT_DIR}"
    )

    print("=" * 100)


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

    dataset = add_patterns(
        dataset
    )

    pattern_statistics = (
        generate_pattern_statistics(
            dataset
        )
    )

    pattern_split_statistics = (
        generate_pattern_split_statistics(
            dataset
        )
    )

    label_distribution = (
        generate_label_distribution(
            dataset
        )
    )

    summary = generate_summary(
        dataset,
        pattern_statistics,
    )

    save_outputs(
        pattern_statistics,
        pattern_split_statistics,
        label_distribution,
        summary,
    )

    print_report(
        dataset,
        pattern_statistics,
    )


if __name__ == "__main__":
    main()