"""
Multimodal Dataset Audit for the COde Dataset.

This module performs a first-level audit of the COde dataset by analyzing:

1. Patient-level structure
2. Visit-level structure
3. Clinical text availability
4. Photograph availability
5. Radiograph availability
6. Basic multimodal missingness

The module does not infer image types such as OPG, CBCT, or
periapical radiographs. Such analysis is intentionally deferred
to later audit stages.

All outputs are saved to the results directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "COde-Dataset"
    / "complete_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "multimodal_dataset_audit"
)


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Load the COde dataset CSV file.

    Parameters
    ----------
    dataset_path : Path
        Path to complete_dataset.csv.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}"
        )

    return pd.read_csv(dataset_path)


# =============================================================================
# Helper Functions
# =============================================================================

def is_non_empty(value) -> bool:
    """
    Check whether a dataset cell contains meaningful content.

    Empty strings, NaN values, and common null-like strings
    are considered missing.
    """

    if pd.isna(value):
        return False

    value = str(value).strip()

    if value == "":
        return False

    if value.lower() in {"nan", "none", "null", "[]"}:
        return False

    return True


def count_images(value) -> int:
    """
    Count image filenames stored in a dataset cell.

    The COde dataset stores multiple image filenames as
    comma-separated values.
    """

    if not is_non_empty(value):
        return 0

    filenames = [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]

    return len(filenames)


# =============================================================================
# Modality Availability
# =============================================================================

def build_modality_availability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build visit-level modality availability information.

    The analysis distinguishes between:

    - Clinical text
    - Photographs
    - Radiographs

    No assumptions are made about the semantic type of images.
    """

    clinical_columns = [
        "patient_record",
        "chief_complaint",
        "present_illness",
        "past_medical_record",
        "examination",
        "radiographs_examination",
        "diagnosis",
        "treatment_plan",
        "treatment_recommendations",
        "management",
        "medical_instructions",
        "remarks",
        "anomalies",
    ]

    available_clinical_columns = [
        column
        for column in clinical_columns
        if column in df.columns
    ]

    result = df[
        [
            "id",
            "patient_id",
            "checkup_id",
            "photographs",
            "radiographs",
        ]
    ].copy()

    result["clinical_text_available"] = df[
        available_clinical_columns
    ].apply(
        lambda row: any(
            is_non_empty(value)
            for value in row
        ),
        axis=1,
    )

    result["photographs_available"] = result[
        "photographs"
    ].apply(is_non_empty)

    result["radiographs_available"] = result[
        "radiographs"
    ].apply(is_non_empty)

    result["photograph_count"] = result[
        "photographs"
    ].apply(count_images)

    result["radiograph_count"] = result[
        "radiographs"
    ].apply(count_images)

    return result


# =============================================================================
# Visit-Level Summary
# =============================================================================

def build_visit_summary(
    modality_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a visit-level multimodal summary.

    Each row represents one patient visit.
    """

    summary = modality_df[
        [
            "patient_id",
            "checkup_id",
            "clinical_text_available",
            "photographs_available",
            "radiographs_available",
            "photograph_count",
            "radiograph_count",
        ]
    ].copy()

    summary["num_available_modalities"] = (
        summary[
            [
                "clinical_text_available",
                "photographs_available",
                "radiographs_available",
            ]
        ]
        .sum(axis=1)
    )

    summary["all_modalities_available"] = (
        summary["num_available_modalities"] == 3
    )

    return summary


# =============================================================================
# Patient-Level Summary
# =============================================================================

def build_patient_summary(
    visit_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a patient-level multimodal summary.

    The summary captures modality availability across
    all visits belonging to each patient.
    """

    summary = (
        visit_summary
        .groupby("patient_id")
        .agg(
            num_visits=("checkup_id", "nunique"),
            visits_with_clinical_text=(
                "clinical_text_available",
                "sum",
            ),
            visits_with_photographs=(
                "photographs_available",
                "sum",
            ),
            visits_with_radiographs=(
                "radiographs_available",
                "sum",
            ),
            total_photographs=(
                "photograph_count",
                "sum",
            ),
            total_radiographs=(
                "radiograph_count",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "patient_has_any_radiograph"
    ] = summary[
        "visits_with_radiographs"
    ] > 0

    summary[
        "patient_has_any_photograph"
    ] = summary[
        "visits_with_photographs"
    ] > 0

    return summary


# =============================================================================
# Clinical Evidence Extraction
# =============================================================================

def build_clinical_modality_evidence(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract basic clinical evidence related to imaging.

    This stage only identifies whether clinical text contains
    references to radiographic or photographic concepts.

    It does not attempt to classify image types.
    """

    clinical_columns = [
        "chief_complaint",
        "present_illness",
        "examination",
        "radiographs_examination",
        "diagnosis",
        "treatment_plan",
        "treatment_recommendations",
        "management",
        "medical_instructions",
        "remarks",
        "anomalies",
        "chief_complaint_cn",
        "present_illness_cn",
        "examination_cn",
        "radiographs_examination_cn",
        "diagnosis_cn",
        "treatment_plan_cn",
        "treatment_recommendations_cn",
        "management_cn",
        "medical_instructions_cn",
        "remarks_cn",
        "anomalies_cn",
    ]

    available_columns = [
        column
        for column in clinical_columns
        if column in df.columns
    ]

    result = df[
        [
            "id",
            "patient_id",
            "checkup_id",
        ]
    ].copy()

    combined_text = (
        df[available_columns]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.lower()
    )

    radiograph_keywords = [
        "radiograph",
        "x-ray",
        "xray",
        "cbct",
        "panoramic",
        "opg",
        "periapical",
        "bitewing",
        "cephalometric",
        "x光",
        "拍片",
        "全景片",
        "根尖片",
        "cbct",
    ]

    photograph_keywords = [
        "photograph",
        "photo",
        "clinical photo",
        "intraoral photo",
        "oral photograph",
        "口腔照片",
        "照片",
    ]

    result["clinical_radiograph_evidence"] = (
        combined_text.apply(
            lambda text: any(
                keyword in text
                for keyword in radiograph_keywords
            )
        )
    )

    result["clinical_photograph_evidence"] = (
        combined_text.apply(
            lambda text: any(
                keyword in text
                for keyword in photograph_keywords
            )
        )
    )

    result["clinical_text_length"] = (
        combined_text.str.len()
    )

    return result


# =============================================================================
# Audit Summary
# =============================================================================

def build_audit_summary(
    df: pd.DataFrame,
    modality_df: pd.DataFrame,
    visit_summary: pd.DataFrame,
    patient_summary: pd.DataFrame,
) -> dict:
    """
    Build a JSON-serializable summary of the multimodal audit.
    """

    summary = {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "num_unique_patients": int(
            df["patient_id"].nunique()
        ),
        "num_unique_visits": int(
            df["checkup_id"].nunique()
        ),
        "visits_with_clinical_text": int(
            modality_df[
                "clinical_text_available"
            ].sum()
        ),
        "visits_with_photographs": int(
            modality_df[
                "photographs_available"
            ].sum()
        ),
        "visits_with_radiographs": int(
            modality_df[
                "radiographs_available"
            ].sum()
        ),
        "visits_with_all_modalities": int(
            visit_summary[
                "all_modalities_available"
            ].sum()
        ),
        "patients_with_any_photograph": int(
            patient_summary[
                "patient_has_any_photograph"
            ].sum()
        ),
        "patients_with_any_radiograph": int(
            patient_summary[
                "patient_has_any_radiograph"
            ].sum()
        ),
    }

    return summary


# =============================================================================
# Main Audit Pipeline
# =============================================================================

def run_multimodal_audit(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict:
    """
    Run the complete first-level multimodal dataset audit.

    The pipeline is idempotent by default.

    If the main output already exists, the audit will not be
    recomputed unless force=True is explicitly provided.
    """

    summary_path = output_dir / "audit_summary.json"

    if summary_path.exists() and not force:
        print(
            "Multimodal audit already completed."
        )
        print(
            f"Existing results found at: {output_dir}"
        )
        print(
            "Use force=True only if the audit "
            "needs to be recomputed."
        )

        with open(
            summary_path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading COde dataset...")

    df = load_dataset(dataset_path)

    print(
        f"Dataset loaded: {len(df):,} rows"
    )

    print(
        "Analyzing modality availability..."
    )

    modality_df = (
        build_modality_availability(df)
    )

    print(
        "Building visit-level summary..."
    )

    visit_summary = build_visit_summary(
        modality_df
    )

    print(
        "Building patient-level summary..."
    )

    patient_summary = build_patient_summary(
        visit_summary
    )

    print(
        "Extracting clinical modality evidence..."
    )

    clinical_evidence = (
        build_clinical_modality_evidence(df)
    )

    print("Saving audit results...")

    modality_df.to_csv(
        output_dir
        / "modality_availability.csv",
        index=False,
    )

    visit_summary.to_csv(
        output_dir
        / "visit_modality_summary.csv",
        index=False,
    )

    patient_summary.to_csv(
        output_dir
        / "patient_modality_summary.csv",
        index=False,
    )

    clinical_evidence.to_csv(
        output_dir
        / "clinical_modality_evidence.csv",
        index=False,
    )

    summary = build_audit_summary(
        df=df,
        modality_df=modality_df,
        visit_summary=visit_summary,
        patient_summary=patient_summary,
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "Multimodal audit completed successfully."
    )

    print(
        f"Results saved to: {output_dir}"
    )

    return summary


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    run_multimodal_audit()