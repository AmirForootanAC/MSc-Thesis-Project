"""
Split Quality Audit

This audit evaluates the quality of the generated patient-level split.

Checks:
    - Split statistics
    - Patient statistics
    - Modality distribution
    - Missing modality distribution
    - Diagnosis availability
    - Longitudinal missing modality
    - Duplicate image isolation
    - Hash reuse statistics

Outputs:
    results/split_quality_audit/
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

DATASET_PATH = Path(
    "data/raw/COde-Dataset/complete_dataset.csv"
)

PATIENT_SPLIT_PATH = Path(
    "results/patient_level_split/patient_split.csv"
)

VISIT_SPLIT_PATH = Path(
    "results/patient_level_split/visit_split.csv"
)

IMAGE_HASH_CACHE_PATH = Path(
    "results/patient_image_duplication_audit/image_hash_cache.csv"
)

PATIENT_DUPLICATION_PATH = Path(
    "results/patient_image_duplication_audit/patient_image_duplication.csv"
)

OUTPUT_DIR = Path(
    "results/split_quality_audit"
)


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Split Quality Audit"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite previous outputs."
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def prepare_output_directory(force: bool):

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


def count_images(series: pd.Series) -> int:
    """
    Count image filenames inside
    comma-separated image columns.
    """

    total = 0

    for value in series.dropna():

        value = str(value).strip()

        if value == "":
            continue

        total += len(
            [
                x.strip()
                for x in value.split(",")
                if x.strip()
            ]
        )

    return total


def image_list(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if value == "":
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def has_photo(row):

    return (
        pd.notna(row["photographs"])
        and
        str(row["photographs"]).strip() != ""
    )


def has_radiograph(row):

    return (
        pd.notna(row["radiographs"])
        and
        str(row["radiographs"]).strip() != ""
    )


def has_clinical(row):

    return (
        pd.notna(row["patient_record"])
        and
        str(row["patient_record"]).strip() != ""
    )


# ============================================================
# Load Data
# ============================================================

def load_data():

    print(
        "[INFO] Loading dataset..."
    )

    dataset = pd.read_csv(
        DATASET_PATH
    )
    
    # Remove original dataset split column
    # because a new patient-level split is used
    if "split" in dataset.columns:
        dataset = dataset.drop(
            columns=["split"]
        )

    patient_split = pd.read_csv(
        PATIENT_SPLIT_PATH
    )

    visit_split = pd.read_csv(
        VISIT_SPLIT_PATH
    )

    hash_cache = pd.read_csv(
        IMAGE_HASH_CACHE_PATH
    )

    duplication = pd.read_csv(
        PATIENT_DUPLICATION_PATH
    )

    print(
        f"[INFO] Visits: {len(dataset):,}"
    )

    print(
        f"[INFO] Patients: "
        f"{dataset['patient_id'].nunique():,}"
    )

    print(
        f"[INFO] Image hashes: "
        f"{len(hash_cache):,}"
    )

    print(
        f"[INFO] Duplicate pairs: "
        f"{len(duplication):,}"
    )

    return (
        dataset,
        patient_split,
        visit_split,
        hash_cache,
        duplication
    )


# ============================================================
# Split Statistics
# ============================================================

def generate_split_statistics(
    dataset: pd.DataFrame,
    visit_split: pd.DataFrame
):

    print(
        "[INFO] Generating split statistics..."
    )

    merged = dataset.merge(
        visit_split[
            [
                "checkup_id",
                "patient_id",
                "split"
            ]
        ],
        on=[
            "checkup_id",
            "patient_id"
        ],
        how="left"
    )


    records = []


    for split_name in sorted(
        merged["split"].dropna().unique()
    ):

        subset = merged[
            merged["split"] == split_name
        ]


        records.append(
            {
                "split": split_name,

                "patients":
                    subset["patient_id"].nunique(),

                "visits":
                    len(subset),

                "photographs":
                    count_images(
                        subset["photographs"]
                    ),

                "radiographs":
                    count_images(
                        subset["radiographs"]
                    ),

                "clinical_records":
                    subset["patient_record"]
                    .notna()
                    .sum()
            }
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# Patient Statistics
# ============================================================

def generate_patient_statistics(
    visit_split: pd.DataFrame
):

    print(
        "[INFO] Generating patient statistics..."
    )


    visits_per_patient = (
        visit_split
        .groupby(
            [
                "patient_id",
                "split"
            ]
        )
        .size()
        .reset_index(
            name="num_visits"
        )
    )


    records = []


    for split_name in sorted(
        visits_per_patient["split"].unique()
    ):

        subset = visits_per_patient[
            visits_per_patient["split"]
            ==
            split_name
        ]


        records.append(
            {
                "split": split_name,

                "patients":
                    len(subset),

                "single_visit_patients":
                    (
                        subset["num_visits"]
                        ==
                        1
                    )
                    .sum(),

                "multi_visit_patients":
                    (
                        subset["num_visits"]
                        >
                        1
                    )
                    .sum(),

                "mean_visits_per_patient":
                    subset["num_visits"]
                    .mean(),

                "median_visits_per_patient":
                    subset["num_visits"]
                    .median(),

                "max_visits_per_patient":
                    subset["num_visits"]
                    .max()
            }
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# Modality Distribution
# ============================================================

def generate_modality_distribution(
    dataset: pd.DataFrame,
    visit_split: pd.DataFrame
):

    print(
        "[INFO] Generating modality distribution..."
    )


    merged = dataset.merge(
        visit_split[
            [
                "checkup_id",
                "patient_id",
                "split"
            ]
        ],
        on=[
            "checkup_id",
            "patient_id"
        ],
        how="left"
    )


    merged["has_photo"] = (
        merged.apply(
            has_photo,
            axis=1
        )
    )


    merged["has_radiograph"] = (
        merged.apply(
            has_radiograph,
            axis=1
        )
    )


    merged["has_clinical"] = (
        merged.apply(
            has_clinical,
            axis=1
        )
    )


    merged["is_multimodal"] = (
        merged["has_photo"]
        &
        merged["has_radiograph"]
        &
        merged["has_clinical"]
    )


    records = []


    for split_name in sorted(
        merged["split"].dropna().unique()
    ):

        subset = merged[
            merged["split"]
            ==
            split_name
        ]


        total = len(subset)


        records.append(
            {
                "split": split_name,

                "visits":
                    total,

                "photo_available":
                    subset["has_photo"]
                    .sum(),

                "photo_percentage":
                    (
                        subset["has_photo"]
                        .sum()
                        /
                        total
                        *
                        100
                    ),

                "radiograph_available":
                    subset["has_radiograph"]
                    .sum(),

                "radiograph_percentage":
                    (
                        subset["has_radiograph"]
                        .sum()
                        /
                        total
                        *
                        100
                    ),

                "clinical_available":
                    subset["has_clinical"]
                    .sum(),

                "clinical_percentage":
                    (
                        subset["has_clinical"]
                        .sum()
                        /
                        total
                        *
                        100
                    ),

                "multimodal_available":
                    subset["is_multimodal"]
                    .sum(),

                "multimodal_percentage":
                    (
                        subset["is_multimodal"]
                        .sum()
                        /
                        total
                        *
                        100
                    )
            }
        )


    return pd.DataFrame(
        records
    )

# ============================================================
# Missing Modality Distribution
# ============================================================

def generate_missing_modality_distribution(
    dataset: pd.DataFrame,
    visit_split: pd.DataFrame
):

    print(
        "[INFO] Generating missing modality distribution..."
    )


    merged = dataset.merge(
        visit_split[
            [
                "checkup_id",
                "patient_id",
                "split"
            ]
        ],
        on=[
            "checkup_id",
            "patient_id"
        ],
        how="left"
    )


    merged["has_photo"] = (
        merged.apply(
            has_photo,
            axis=1
        )
    )

    merged["has_radiograph"] = (
        merged.apply(
            has_radiograph,
            axis=1
        )
    )

    merged["has_clinical"] = (
        merged.apply(
            has_clinical,
            axis=1
        )
    )


    def modality_state(row):

        if (
            row["has_photo"]
            and row["has_radiograph"]
            and row["has_clinical"]
        ):
            return "full_multimodal"

        if (
            row["has_clinical"]
            and row["has_photo"]
            and not row["has_radiograph"]
        ):
            return "missing_radiograph"

        if (
            row["has_clinical"]
            and row["has_radiograph"]
            and not row["has_photo"]
        ):
            return "missing_photo"

        if (
            row["has_photo"]
            and not row["has_clinical"]
        ):
            return "photo_only"

        if (
            row["has_radiograph"]
            and not row["has_clinical"]
        ):
            return "radiograph_only"

        if row["has_clinical"]:
            return "clinical_only"

        return "no_modality"


    merged["modality_state"] = (
        merged.apply(
            modality_state,
            axis=1
        )
    )


    result = (
        merged
        .groupby(
            [
                "split",
                "modality_state"
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )


    result["percentage"] = (
        result.groupby("split")["count"]
        .transform(
            lambda x:
            x / x.sum() * 100
        )
    )


    return result


# ============================================================
# Longitudinal Missing Modality Analysis
# ============================================================

def generate_longitudinal_modality_summary(
    dataset: pd.DataFrame,
    visit_split: pd.DataFrame
):

    print(
        "[INFO] Generating longitudinal modality summary..."
    )


    merged = dataset.merge(
        visit_split[
            [
                "checkup_id",
                "patient_id",
                "split"
            ]
        ],
        on=[
            "checkup_id",
            "patient_id"
        ],
        how="left"
    )


    merged["has_radiograph"] = (
        merged.apply(
            has_radiograph,
            axis=1
        )
    )


    patient_summary = (
        merged
        .groupby("patient_id")
        .agg(
            total_visits=(
                "checkup_id",
                "count"
            ),

            visits_with_radiograph=(
                "has_radiograph",
                "sum"
            )
        )
        .reset_index()
    )


    patient_summary[
        "has_longitudinal_radiograph_completion"
    ] = (
        (
            patient_summary[
                "visits_with_radiograph"
            ]
            > 0
        )
        &
        (
            patient_summary[
                "visits_with_radiograph"
            ]
            <
            patient_summary[
                "total_visits"
            ]
        )
    )


    patients_with_completion = (
        patient_summary[
            "has_longitudinal_radiograph_completion"
        ]
        .sum()
    )


    merged = merged.merge(
        patient_summary[
            [
                "patient_id",
                "has_longitudinal_radiograph_completion"
            ]
        ],
        on="patient_id",
        how="left"
    )


    visits_without_radiograph = (
        (
            ~merged["has_radiograph"]
        )
        &
        merged[
            "has_longitudinal_radiograph_completion"
        ]
    ).sum()


    result = pd.DataFrame(
        [
            {
                "patients_with_longitudinal_radiograph_completion":
                    patients_with_completion,

                "visits_without_radiograph_but_available_elsewhere":
                    visits_without_radiograph,

                "total_patients":
                    merged["patient_id"]
                    .nunique(),

                "total_visits":
                    len(merged)
            }
        ]
    )


    return result


# ============================================================
# Diagnosis Availability Distribution
# ============================================================

def generate_diagnosis_distribution(
    dataset: pd.DataFrame,
    visit_split: pd.DataFrame
):

    print(
        "[INFO] Generating diagnosis distribution..."
    )


    merged = dataset.merge(
        visit_split[
            [
                "checkup_id",
                "patient_id",
                "split"
            ]
        ],
        on=[
            "checkup_id",
            "patient_id"
        ],
        how="left"
    )


    merged["diagnosis_available"] = (
        merged["diagnosis"]
        .notna()
    )


    result = (
        merged
        .groupby(
            [
                "split",
                "diagnosis_available"
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )


    result["percentage"] = (
        result.groupby("split")["count"]
        .transform(
            lambda x:
            x / x.sum() * 100
        )
    )


    return result

# ============================================================
# Duplicate Hash Isolation
# ============================================================

def generate_duplicate_hash_audit(
    duplication: pd.DataFrame,
    visit_split: pd.DataFrame
):

    print(
        "[INFO] Checking duplicate hash isolation..."
    )


    visit_mapping = (
        visit_split[
            [
                "checkup_id",
                "split"
            ]
        ]
        .rename(
            columns={
                "checkup_id": "visit"
            }
        )
    )


    duplication = duplication.copy()


    duplication = (
        duplication
        .merge(
            visit_mapping,
            left_on="visit_1",
            right_on="visit",
            how="left"
        )
        .rename(
            columns={
                "split": "split_1"
            }
        )
        .drop(
            columns=["visit"]
        )
    )


    duplication = (
        duplication
        .merge(
            visit_mapping,
            left_on="visit_2",
            right_on="visit",
            how="left"
        )
        .rename(
            columns={
                "split": "split_2"
            }
        )
        .drop(
            columns=["visit"]
        )
    )


    duplication["cross_split"] = (
        duplication["split_1"]
        !=
        duplication["split_2"]
    )


    leakage = duplication[
        duplication["cross_split"]
    ]


    summary = pd.DataFrame(
        [
            {
                "total_duplicate_pairs":
                    len(duplication),

                "cross_split_duplicate_pairs":
                    len(leakage),

                "status":
                    (
                        "SAFE"
                        if len(leakage) == 0
                        else "UNSAFE"
                    )
            }
        ]
    )


    return (
        summary,
        leakage,
        duplication
    )


# ============================================================
# Hash Reuse Statistics
# ============================================================

def generate_hash_reuse_statistics(
    hash_cache: pd.DataFrame
):

    print(
        "[INFO] Generating hash reuse statistics..."
    )


    hash_usage = (
        hash_cache
        .groupby(
            "sha256"
        )
        .size()
        .reset_index(
            name="reuse_count"
        )
    )


    summary = pd.DataFrame(
        [
            {
                "total_images":
                    len(hash_cache),

                "unique_hashes":
                    hash_usage["sha256"]
                    .nunique(),

                "duplicated_hashes":
                    (
                        hash_usage["reuse_count"]
                        >
                        1
                    )
                    .sum(),

                "maximum_reuse_count":
                    hash_usage["reuse_count"]
                    .max()
            }
        ]
    )


    return (
        summary,
        hash_usage
    )


# ============================================================
# Save Outputs
# ============================================================

def save_outputs(
    outputs: dict
):

    print(
        "[INFO] Saving audit outputs..."
    )


    for filename, data in outputs.items():

        path = (
            OUTPUT_DIR
            /
            filename
        )


        if isinstance(
            data,
            pd.DataFrame
        ):

            data.to_csv(
                path,
                index=False
            )


        elif isinstance(
            data,
            dict
        ):

            with open(
                path,
                "w"
            ) as f:

                json.dump(
                    data,
                    f,
                    indent=4
                )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()


    prepare_output_directory(
        args.force
    )


    (
        dataset,
        patient_split,
        visit_split,
        hash_cache,
        duplication
    ) = load_data()


    split_statistics = (
        generate_split_statistics(
            dataset,
            visit_split
        )
    )


    patient_statistics = (
        generate_patient_statistics(
            visit_split
        )
    )


    modality_distribution = (
        generate_modality_distribution(
            dataset,
            visit_split
        )
    )


    missing_modality_distribution = (
        generate_missing_modality_distribution(
            dataset,
            visit_split
        )
    )


    longitudinal_summary = (
        generate_longitudinal_modality_summary(
            dataset,
            visit_split
        )
    )


    diagnosis_distribution = (
        generate_diagnosis_distribution(
            dataset,
            visit_split
        )
    )


    (
        duplicate_summary,
        duplicate_leakage,
        duplicate_details
    ) = generate_duplicate_hash_audit(
        duplication,
        visit_split
    )


    (
        hash_summary,
        hash_usage
    ) = generate_hash_reuse_statistics(
        hash_cache
    )


    final_summary = {

        "split_quality_status":
            (
                "PASS"
                if duplicate_summary[
                    "cross_split_duplicate_pairs"
                ]
                .iloc[0]
                ==
                0
                else
                "FAIL"
            ),

        "duplicate_hash_leakage":
            int(
                duplicate_summary[
                    "cross_split_duplicate_pairs"
                ]
                .iloc[0]
            ),

        "total_patients":
            int(
                dataset["patient_id"]
                .nunique()
            ),

        "total_visits":
            int(
                len(dataset)
            )
    }


    outputs = {

        "split_statistics.csv":
            split_statistics,

        "patient_statistics.csv":
            patient_statistics,

        "modality_distribution.csv":
            modality_distribution,

        "missing_modality_distribution.csv":
            missing_modality_distribution,

        "longitudinal_modality_summary.csv":
            longitudinal_summary,

        "diagnosis_distribution.csv":
            diagnosis_distribution,

        "duplicate_hash_summary.csv":
            duplicate_summary,

        "duplicate_hash_leakage.csv":
            duplicate_leakage,

        "duplicate_visit_split_consistency.csv":
            duplicate_details,

        "hash_reuse_statistics.csv":
            hash_usage,

        "hash_summary.csv":
            hash_summary,

        "audit_summary.json":
            final_summary
    }


    save_outputs(
        outputs
    )


    print()
    print("=" * 70)
    print(
        "Split Quality Audit completed successfully."
    )
    print(
        f"Status: {final_summary['split_quality_status']}"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()