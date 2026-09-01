"""
Milestone 8.5.1 — Natural Missingness Robustness Protocol.

Purpose
-------
Define the evaluation population and naturally occurring
missing-modality patterns for the full six-label
patient-level dataset.

Unlike Milestones 7 and 8.3/8.4, this protocol does NOT
restrict the population to complete multimodal visits.

Dataset
-------
    results/six_label_patient_level_dataset/labeled_dataset.csv

The authoritative patient-level split is preserved.

Modality availability
---------------------
    Image       -> photographs
    Radiograph  -> radiographs
    Text        -> clinical text fields

For each visit, a deterministic modality-availability pattern
is assigned:

    complete
    image_missing
    radiograph_missing
    text_missing
    image_radiograph_missing
    image_text_missing
    radiograph_text_missing
    all_missing

The protocol reports:
    - sample counts
    - percentages
    - distribution by split
    - overall distribution

No model training is performed in this milestone.
"""

from pathlib import Path

import json

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

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "milestone8_missing_modality"
    / "07_natural_missingness_robust"
    / "01_protocol"
)


TEXT_COLUMNS = [
    "chief_complaint",
    "present_illness",
    "past_medical_record",
    "examination",
]


EXPECTED_SPLITS = [
    "train",
    "validation",
    "test",
]


PATTERN_ORDER = [
    "complete",
    "image_missing",
    "radiograph_missing",
    "text_missing",
    "image_radiograph_missing",
    "image_text_missing",
    "radiograph_text_missing",
    "all_missing",
]


# ============================================================
# Utility
# ============================================================

def has_value(value):
    """
    Return True when a modality field contains a usable value.
    """

    return (
        pd.notna(value)
        and str(value).strip() != ""
    )


def has_text(row):
    """
    Clinical text is considered available when at least one
    of the designated clinical text fields is non-empty.
    """

    for column in TEXT_COLUMNS:

        if has_value(row[column]):
            return True

    return False


def assign_pattern(
    image_available,
    radiograph_available,
    text_available,
):
    """
    Assign a deterministic modality pattern.
    """

    key = (
        int(image_available),
        int(radiograph_available),
        int(text_available),
    )

    patterns = {
        (1, 1, 1): "complete",

        (0, 1, 1): "image_missing",

        (1, 0, 1): "radiograph_missing",

        (1, 1, 0): "text_missing",

        (0, 0, 1): "image_radiograph_missing",

        (0, 1, 0): "image_text_missing",

        (1, 0, 0): "radiograph_text_missing",

        (0, 0, 0): "all_missing",
    }

    return patterns[key]


# ============================================================
# Load dataset
# ============================================================

def load_dataset():

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    required_columns = [
        "checkup_id",
        "patient_id",
        "split",
        "photographs",
        "radiographs",
        *TEXT_COLUMNS,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# Build modality availability
# ============================================================

def build_protocol(df):

    result = df[
        [
            "checkup_id",
            "patient_id",
            "split",
        ]
    ].copy()

    result["image_available"] = (
        df["photographs"]
        .apply(has_value)
    )

    result["radiograph_available"] = (
        df["radiographs"]
        .apply(has_value)
    )

    result["text_available"] = (
        df.apply(
            has_text,
            axis=1,
        )
    )

    result["missing_modality_count"] = (
        3
        - result[
            [
                "image_available",
                "radiograph_available",
                "text_available",
            ]
        ].sum(axis=1)
    )

    result["pattern"] = [
        assign_pattern(
            image_available,
            radiograph_available,
            text_available,
        )
        for image_available,
        radiograph_available,
        text_available
        in zip(
            result["image_available"],
            result["radiograph_available"],
            result["text_available"],
        )
    ]

    return result


# ============================================================
# Distribution
# ============================================================

def build_distribution(
    protocol,
    split=None,
):

    if split is None:
        subset = protocol
        split_name = "all"
    else:
        subset = protocol[
            protocol["split"] == split
        ]
        split_name = split

    total = len(subset)

    rows = []

    for pattern in PATTERN_ORDER:

        count = int(
            (
                subset["pattern"]
                == pattern
            ).sum()
        )

        percentage = (
            100.0 * count / total
            if total > 0
            else 0.0
        )

        mask = {
            "complete": [1, 1, 1],
            "image_missing": [0, 1, 1],
            "radiograph_missing": [1, 0, 1],
            "text_missing": [1, 1, 0],
            "image_radiograph_missing": [0, 0, 1],
            "image_text_missing": [0, 1, 0],
            "radiograph_text_missing": [1, 0, 0],
            "all_missing": [0, 0, 0],
        }[pattern]

        rows.append(
            {
                "split": split_name,
                "pattern": pattern,
                "image_available": bool(mask[0]),
                "radiograph_available": bool(mask[1]),
                "text_available": bool(mask[2]),
                "missing_modality_count": (
                    3 - sum(mask)
                ),
                "samples": count,
                "percentage": percentage,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 100)
    print(
        "MILESTONE 8.5.1 — NATURAL MISSINGNESS ROBUSTNESS PROTOCOL"
    )
    print("=" * 100)

    print()
    print("Dataset:")
    print(DATASET_PATH)

    print()
    print("Result root:")
    print(RESULT_ROOT)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    print()
    print(
        f"Total labeled visits: {len(df)}"
    )

    print(
        f"Patients: {df['patient_id'].nunique()}"
    )

    print(
        f"Splits: {sorted(df['split'].dropna().unique())}"
    )

    # --------------------------------------------------------
    # Build protocol
    # --------------------------------------------------------

    protocol = build_protocol(df)

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if len(protocol) != len(df):

        raise RuntimeError(
            "Protocol population size changed."
        )

    if protocol["checkup_id"].duplicated().any():

        raise RuntimeError(
            "Duplicate checkup_id detected."
        )

    if protocol["patient_id"].isna().any():

        raise RuntimeError(
            "Missing patient_id detected."
        )

    if not set(protocol["split"]).issubset(
        set(EXPECTED_SPLITS)
    ):

        raise RuntimeError(
            "Unexpected split values detected."
        )

    # Every visit must have exactly one pattern.
    pattern_counts = (
        protocol["pattern"]
        .value_counts()
    )

    if pattern_counts.sum() != len(protocol):

        raise RuntimeError(
            "Pattern assignment is incomplete."
        )

    # --------------------------------------------------------
    # Save visit-level population
    # --------------------------------------------------------

    population_path = (
        RESULT_ROOT
        / "population.csv"
    )

    protocol.to_csv(
        population_path,
        index=False,
    )

    # --------------------------------------------------------
    # Overall distribution
    # --------------------------------------------------------

    overall = build_distribution(
        protocol
    )

    overall_path = (
        RESULT_ROOT
        / "pattern_distribution.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    # --------------------------------------------------------
    # Split distribution
    # --------------------------------------------------------

    split_frames = []

    for split in EXPECTED_SPLITS:

        split_frames.append(
            build_distribution(
                protocol,
                split=split,
            )
        )

    by_split = pd.concat(
        split_frames,
        ignore_index=True,
    )

    by_split_path = (
        RESULT_ROOT
        / "pattern_distribution_by_split.csv"
    )

    by_split.to_csv(
        by_split_path,
        index=False,
    )

    # --------------------------------------------------------
    # Modality availability summary
    # --------------------------------------------------------

    modality_summary = []

    total = len(protocol)

    for modality in [
        "image_available",
        "radiograph_available",
        "text_available",
    ]:

        available = int(
            protocol[modality].sum()
        )

        missing = total - available

        modality_summary.append(
            {
                "modality": modality.replace(
                    "_available",
                    "",
                ),
                "available_samples": available,
                "missing_samples": missing,
                "available_percentage": (
                    100.0 * available / total
                    if total > 0
                    else 0.0
                ),
                "missing_percentage": (
                    100.0 * missing / total
                    if total > 0
                    else 0.0
                ),
            }
        )

    modality_summary_df = pd.DataFrame(
        modality_summary
    )

    modality_summary_df.to_csv(
        RESULT_ROOT
        / "modality_availability.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {
        "milestone": "8.5.1",
        "experiment": (
            "Natural Missingness Robustness Protocol"
        ),
        "dataset": str(DATASET_PATH),
        "total_visits": int(len(protocol)),
        "unique_patients": int(
            protocol["patient_id"].nunique()
        ),
        "split_counts": {
            split: int(
                (
                    protocol["split"]
                    == split
                ).sum()
            )
            for split in EXPECTED_SPLITS
        },
        "modality_availability": {
            row["modality"]: {
                "available_samples": int(
                    row["available_samples"]
                ),
                "missing_samples": int(
                    row["missing_samples"]
                ),
                "available_percentage": float(
                    row["available_percentage"]
                ),
                "missing_percentage": float(
                    row["missing_percentage"]
                ),
            }
            for _, row
            in modality_summary_df.iterrows()
        },
        "overall_pattern_distribution": {
            row["pattern"]: {
                "samples": int(
                    row["samples"]
                ),
                "percentage": float(
                    row["percentage"]
                ),
            }
            for _, row
            in overall.iterrows()
        },
        "protocol": {
            "patient_level_split_preserved": True,
            "complete_case_filter": False,
            "natural_missingness": True,
            "model_training": False,
            "test_population_definition": (
                "All six-label patient-level visits "
                "within the authoritative split."
            ),
        },
    }

    with open(
        RESULT_ROOT
        / "protocol_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("OVERALL NATURAL MISSINGNESS DISTRIBUTION")
    print("-" * 100)

    print(
        overall[
            [
                "pattern",
                "samples",
                "percentage",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("-" * 100)
    print("DISTRIBUTION BY SPLIT")
    print("-" * 100)

    print(
        by_split[
            [
                "split",
                "pattern",
                "samples",
                "percentage",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("-" * 100)
    print("MODALITY AVAILABILITY")
    print("-" * 100)

    print(
        modality_summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print(
        "MILESTONE 8.5.1 — PASS"
    )
    print("=" * 100)

    print()
    print("Saved:")
    print(population_path)
    print(overall_path)
    print(by_split_path)
    print(
        RESULT_ROOT
        / "modality_availability.csv"
    )
    print(
        RESULT_ROOT
        / "protocol_summary.json"
    )


if __name__ == "__main__":
    main()