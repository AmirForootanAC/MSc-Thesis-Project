"""
Patient-Level Split Leakage Validation Audit

Validates the integrity of the generated patient-level split.

Checks:
- Patient overlap between splits
- Visit overlap between splits
- Photograph filename overlap
- Radiograph filename overlap
- Patient assignment consistency
- Visit assignment consistency

Outputs:
results/patient_level_split_leakage/

- leakage_summary.json
- assignment_validation.json
- patient_overlap.csv
- visit_overlap.csv
- photograph_overlap.csv
- radiograph_overlap.csv

This module only validates existing splits.
It does not modify dataset splits.
"""

import argparse
import json
from pathlib import Path
from itertools import combinations

import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path(
    "results/patient_level_split"
)

OUTPUT_DIR = Path(
    "results/patient_level_split_leakage"
)


DEFAULT_SPLITS = [
    "train",
    "validation",
    "test"
]


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


def load_split_files():
    """
    Load generated patient-level split files.
    """

    print(
        "[INFO] Loading split files..."
    )

    patient_split_path = (
        INPUT_DIR /
        "patient_split.csv"
    )

    visit_split_path = (
        INPUT_DIR /
        "visit_split.csv"
    )


    patient_split = pd.read_csv(
        patient_split_path
    )

    visit_split = pd.read_csv(
        visit_split_path
    )


    print(
        f"[INFO] Patients loaded: "
        f"{patient_split['patient_id'].nunique()}"
    )

    print(
        f"[INFO] Visits loaded: "
        f"{len(visit_split)}"
    )


    return (
        patient_split,
        visit_split
    )


# ============================================================
# Assignment Validation
# ============================================================

def validate_assignments(
    patient_split: pd.DataFrame,
    visit_split: pd.DataFrame
):
    """
    Verify that every patient and visit
    has exactly one split assignment.
    """

    print(
        "[INFO] Validating split assignments..."
    )


    patient_counts = (
        patient_split
        .groupby("patient_id")
        ["split"]
        .nunique()
    )


    invalid_patients = (
        patient_counts[
            patient_counts != 1
        ]
        .index
        .tolist()
    )


    visit_counts = (
        visit_split
        .groupby("checkup_id")
        ["split"]
        .nunique()
    )


    invalid_visits = (
        visit_counts[
            visit_counts != 1
        ]
        .index
        .tolist()
    )


    result = {

        "total_patients":
            int(
                patient_split["patient_id"]
                .nunique()
            ),

        "total_visits":
            int(
                visit_split["checkup_id"]
                .nunique()
            ),

        "patients_with_multiple_assignments":
            len(invalid_patients),

        "visits_with_multiple_assignments":
            len(invalid_visits),

        "status":
            "PASS"
            if (
                len(invalid_patients) == 0
                and
                len(invalid_visits) == 0
            )
            else
            "FAIL"
    }


    invalid_patient_df = (
        pd.DataFrame(
            {
                "patient_id":
                    invalid_patients
            }
        )
    )


    invalid_visit_df = (
        pd.DataFrame(
            {
                "checkup_id":
                    invalid_visits
            }
        )
    )


    return (
        result,
        invalid_patient_df,
        invalid_visit_df
    )


# ============================================================
# Overlap Validation
# ============================================================

def find_pairwise_overlap(
    df: pd.DataFrame,
    id_column: str
):
    """
    Find overlap between split pairs.
    """

    records = []


    split_groups = {

        split:
            set(
                df.loc[
                    df["split"] == split,
                    id_column
                ]
            )

        for split in DEFAULT_SPLITS

    }


    for split_a, split_b in combinations(
        DEFAULT_SPLITS,
        2
    ):

        overlap = (
            split_groups[split_a]
            &
            split_groups[split_b]
        )


        for item in overlap:

            records.append(
                {
                    "id": item,
                    "split_1": split_a,
                    "split_2": split_b
                }
            )


    return pd.DataFrame(
        records
    )


def validate_patient_overlap(
    patient_split
):
    """
    Validate patient isolation.
    """

    print(
        "[INFO] Checking patient overlap..."
    )

    return find_pairwise_overlap(
        patient_split,
        "patient_id"
    )


def validate_visit_overlap(
    visit_split
):
    """
    Validate visit isolation.
    """

    print(
        "[INFO] Checking visit overlap..."
    )

    return find_pairwise_overlap(
        visit_split,
        "checkup_id"
    )


# ============================================================
# Image Parsing
# ============================================================

def extract_image_names(
    series: pd.Series
):
    """
    Extract image filenames from
    comma-separated image columns.
    """

    images = set()


    for value in series.dropna():

        for image in str(value).split(","):

            image = image.strip()

            if image:
                images.add(image)


    return images

# ============================================================
# Image Leakage Validation
# ============================================================

def validate_image_overlap(
    visit_split: pd.DataFrame,
    column_name: str
):
    """
    Validate image filename overlap
    between dataset splits.
    """

    print(
        f"[INFO] Checking {column_name} overlap..."
    )


    split_images = {}


    for split in DEFAULT_SPLITS:

        split_df = visit_split[
            visit_split["split"] == split
        ]

        split_images[split] = extract_image_names(
            split_df[column_name]
        )


    records = []


    for split_a, split_b in combinations(
        DEFAULT_SPLITS,
        2
    ):

        overlap = (
            split_images[split_a]
            &
            split_images[split_b]
        )


        for image_name in overlap:

            records.append(
                {
                    "image_name": image_name,
                    "split_1": split_a,
                    "split_2": split_b
                }
            )


    return pd.DataFrame(
        records
    )


# ============================================================
# Summary Generation
# ============================================================

def build_summary(
    patient_overlap: pd.DataFrame,
    visit_overlap: pd.DataFrame,
    photograph_overlap: pd.DataFrame,
    radiograph_overlap: pd.DataFrame,
    assignment_validation: dict
):
    """
    Create final leakage summary.
    """

    patient_leakage = len(
        patient_overlap
    )

    visit_leakage = len(
        visit_overlap
    )

    photograph_leakage = len(
        photograph_overlap
    )

    radiograph_leakage = len(
        radiograph_overlap
    )


    assignment_status = (
        assignment_validation["status"]
    )


    leakage_status = (
        "SAFE"
        if (
            patient_leakage == 0
            and
            visit_leakage == 0
            and
            photograph_leakage == 0
            and
            radiograph_leakage == 0
            and
            assignment_status == "PASS"
        )
        else
        "UNSAFE"
    )


    summary = {

        "patient_overlap_count":
            patient_leakage,

        "visit_overlap_count":
            visit_leakage,

        "photograph_overlap_count":
            photograph_leakage,

        "radiograph_overlap_count":
            radiograph_leakage,

        "assignment_validation":
            assignment_status,

        "status":
            leakage_status
    }


    return summary


# ============================================================
# Save Outputs
# ============================================================

def save_outputs(
    summary: dict,
    assignment_validation: dict,
    patient_overlap: pd.DataFrame,
    visit_overlap: pd.DataFrame,
    photograph_overlap: pd.DataFrame,
    radiograph_overlap: pd.DataFrame
):
    """
    Save audit outputs.
    """

    print(
        "[INFO] Saving leakage reports..."
    )


    with open(
        OUTPUT_DIR / "leakage_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )


    with open(
        OUTPUT_DIR / "assignment_validation.json",
        "w"
    ) as f:

        json.dump(
            assignment_validation,
            f,
            indent=4
        )


    patient_overlap.to_csv(
        OUTPUT_DIR / "patient_overlap.csv",
        index=False
    )


    visit_overlap.to_csv(
        OUTPUT_DIR / "visit_overlap.csv",
        index=False
    )


    photograph_overlap.to_csv(
        OUTPUT_DIR / "photograph_overlap.csv",
        index=False
    )


    radiograph_overlap.to_csv(
        OUTPUT_DIR / "radiograph_overlap.csv",
        index=False
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Validate patient-level split leakage."
        )
    )


    parser.add_argument(
        "--force",
        action="store_true"
    )


    args = parser.parse_args()


    create_output_directory(
        args.force
    )


    patient_split, visit_split = (
        load_split_files()
    )


    (
        assignment_validation,
        invalid_patients,
        invalid_visits
    ) = validate_assignments(
        patient_split,
        visit_split
    )


    patient_overlap = (
        validate_patient_overlap(
            patient_split
        )
    )


    visit_overlap = (
        validate_visit_overlap(
            visit_split
        )
    )


    photograph_overlap = (
        validate_image_overlap(
            visit_split,
            "photographs"
        )
    )


    radiograph_overlap = (
        validate_image_overlap(
            visit_split,
            "radiographs"
        )
    )


    summary = build_summary(
        patient_overlap,
        visit_overlap,
        photograph_overlap,
        radiograph_overlap,
        assignment_validation
    )


    save_outputs(
        summary,
        assignment_validation,
        patient_overlap,
        visit_overlap,
        photograph_overlap,
        radiograph_overlap
    )


    print()
    print("=" * 70)
    print(
        "Patient-level leakage validation completed."
    )
    print(
        f"Status: {summary['status']}"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()