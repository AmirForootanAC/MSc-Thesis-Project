from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ============================================================
# Constants
# ============================================================

IMAGE_MODALITIES = ("photographs", "radiographs")

CLINICAL_TEXT_COLUMNS = [
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

CLINICAL_EVIDENCE_KEYWORDS = {
    "photographs": [
        "photograph",
        "photographs",
        "photo",
        "photos",
        "intraoral photo",
        "intraoral photograph",
        "oral photograph",
        "clinical photograph",
        "clinical photo",
        "picture",
        "image",
        "images",
        "oral image",
    ],
    "radiographs": [
        "radiograph",
        "radiographs",
        "x-ray",
        "x ray",
        "xray",
        "panoramic",
        "opg",
        "cbct",
        "cone beam",
        "periapical",
        "bitewing",
        "cephalometric",
        "dental x-ray",
        "dental radiograph",
    ],
}

RADIOGRAPH_TYPE_PATTERNS = {
    "cbct": [
        r"\bcbct\b",
        r"\bcone[- ]?beam computed tomography\b",
        r"\bcone[- ]?beam ct\b",
    ],
    "panoramic_opg": [
        r"\bpanoramic\b",
        r"\bopg\b",
        r"\borthopantomogram\b",
    ],
    "periapical": [
        r"\bperiapical\b",
        r"\bperiapical x[- ]?ray\b",
        r"\bperiapical radiograph\b",
    ],
    "bitewing": [
        r"\bbitewing\b",
        r"\bbite[- ]?wing\b",
    ],
    "cephalometric": [
        r"\bcephalometric\b",
        r"\bcephalogram\b",
    ],
    "xray_unspecified": [
        r"\bx[- ]?ray\b",
        r"\bxray\b",
        r"\bradiograph\b",
        r"\bradiographs\b",
    ],
}

IMAGE_FILENAME_PATTERN = re.compile(
    r"^(?P<patient_id>\d+)-(?P<visit_number>\d+)-(?P<image_index>\d+)\.(?P<extension>[A-Za-z0-9]+)$"
)


# ============================================================
# Configuration
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deep patient trace and multimodal dataset audit."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/audit.yaml",
        help="Path to audit configuration file.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even when complete outputs already exist.",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}"
        )

    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    return config


def get_config_path(
    config: dict[str, Any],
    section: str,
    key: str,
    default: str,
) -> Path:
    value = (
        config
        .get(section, {})
        .get(key, default)
    )

    return Path(value).expanduser()


# ============================================================
# Dataset Loading
# ============================================================

def load_dataset(dataset_path: Path) -> pd.DataFrame:
    print(f"[INFO] Loading dataset: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    df = pd.read_csv(
        dataset_path,
        low_memory=False,
    )

    print(f"[INFO] Loaded {len(df):,} rows.")
    print(f"[INFO] Found {len(df.columns)} columns.")

    required_columns = {
        "id",
        "checkup_id",
        "patient_id",
        "checkup_time",
        "photographs",
        "radiographs",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df


# ============================================================
# Utility Functions
# ============================================================

def normalize_patient_id(value: Any) -> str:
    if pd.isna(value):
        return ""

    try:
        return f"{int(float(value)):04d}"
    except (ValueError, TypeError):
        return str(value).strip()


def normalize_visit_id(value: Any) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def split_image_filenames(value: Any) -> list[str]:
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    filenames = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    return filenames


def parse_image_filename(filename: str) -> dict[str, Any]:
    match = IMAGE_FILENAME_PATTERN.match(filename)

    if not match:
        return {
            "filename": filename,
            "filename_parse_success": False,
            "parsed_patient_id": None,
            "parsed_visit_number": None,
            "parsed_image_index": None,
            "extension": None,
        }

    groups = match.groupdict()

    return {
        "filename": filename,
        "filename_parse_success": True,
        "parsed_patient_id": groups["patient_id"],
        "parsed_visit_number": int(groups["visit_number"]),
        "parsed_image_index": int(groups["image_index"]),
        "extension": groups["extension"].lower(),
    }


def combine_clinical_text(row: pd.Series) -> str:
    parts = []

    for column in CLINICAL_TEXT_COLUMNS:
        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            parts.append(value)

    return " ".join(parts)


def find_keyword_evidence(
    text: str,
    keywords: list[str],
) -> list[str]:
    text_lower = text.lower()

    matches = []

    for keyword in keywords:
        if keyword.lower() in text_lower:
            matches.append(keyword)

    return sorted(set(matches))


def detect_radiograph_types(
    text: str,
) -> list[str]:
    text_lower = text.lower()

    detected_types = []

    for radiograph_type, patterns in RADIOGRAPH_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                detected_types.append(radiograph_type)
                break

    return sorted(set(detected_types))


# ============================================================
# Clinical Evidence
# ============================================================

def build_clinical_evidence(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    radiograph_rows = []
    photograph_rows = []

    for _, row in df.iterrows():
        clinical_text = combine_clinical_text(row)

        patient_id = normalize_patient_id(row["patient_id"])
        checkup_id = normalize_visit_id(row["checkup_id"])

        radiograph_keywords = find_keyword_evidence(
            clinical_text,
            CLINICAL_EVIDENCE_KEYWORDS["radiographs"],
        )

        photograph_keywords = find_keyword_evidence(
            clinical_text,
            CLINICAL_EVIDENCE_KEYWORDS["photographs"],
        )

        radiograph_types = detect_radiograph_types(
            clinical_text
        )

        has_radiographs = len(
            split_image_filenames(row["radiographs"])
        ) > 0

        has_photographs = len(
            split_image_filenames(row["photographs"])
        ) > 0

        radiograph_rows.append(
            {
                "id": row["id"],
                "patient_id": patient_id,
                "checkup_id": checkup_id,
                "has_radiograph_files": has_radiographs,
                "clinical_radiograph_evidence": bool(
                    radiograph_keywords
                ),
                "radiograph_evidence_keywords": "|".join(
                    radiograph_keywords
                ),
                "radiograph_type_weak_evidence": "|".join(
                    radiograph_types
                ),
                "radiograph_type_evidence_count": len(
                    radiograph_types
                ),
                "radiograph_missing_but_clinical_evidence": (
                    not has_radiographs
                    and bool(radiograph_keywords)
                ),
                "radiograph_present_without_clinical_evidence": (
                    has_radiographs
                    and not bool(radiograph_keywords)
                ),
            }
        )

        photograph_rows.append(
            {
                "id": row["id"],
                "patient_id": patient_id,
                "checkup_id": checkup_id,
                "has_photograph_files": has_photographs,
                "clinical_photograph_evidence": bool(
                    photograph_keywords
                ),
                "photograph_evidence_keywords": "|".join(
                    photograph_keywords
                ),
                "photograph_missing_but_clinical_evidence": (
                    not has_photographs
                    and bool(photograph_keywords)
                ),
                "photograph_present_without_clinical_evidence": (
                    has_photographs
                    and not bool(photograph_keywords)
                ),
            }
        )

    return (
        pd.DataFrame(radiograph_rows),
        pd.DataFrame(photograph_rows),
    )


# ============================================================
# Patient Summary
# ============================================================

def build_patient_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    work_df = df.copy()

    work_df["patient_id_normalized"] = work_df[
        "patient_id"
    ].apply(normalize_patient_id)

    work_df["has_photographs"] = work_df[
        "photographs"
    ].apply(
        lambda value: len(split_image_filenames(value)) > 0
    )

    work_df["has_radiographs"] = work_df[
        "radiographs"
    ].apply(
        lambda value: len(split_image_filenames(value)) > 0
    )

    summary = (
        work_df
        .groupby("patient_id_normalized", sort=True)
        .agg(
            num_visits=("checkup_id", "nunique"),
            num_rows=("id", "count"),
            visits_with_photographs=(
                "has_photographs",
                "sum",
            ),
            visits_with_radiographs=(
                "has_radiographs",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["has_any_photograph"] = (
        summary["visits_with_photographs"] > 0
    )

    summary["has_any_radiograph"] = (
        summary["visits_with_radiographs"] > 0
    )

    summary["has_both_modalities"] = (
        summary["has_any_photograph"]
        & summary["has_any_radiograph"]
    )

    return summary


# ============================================================
# Visit Summary
# ============================================================

def build_visit_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        photographs = split_image_filenames(
            row["photographs"]
        )

        radiographs = split_image_filenames(
            row["radiographs"]
        )

        clinical_text = combine_clinical_text(row)

        rows.append(
            {
                "id": row["id"],
                "patient_id": normalize_patient_id(
                    row["patient_id"]
                ),
                "checkup_id": normalize_visit_id(
                    row["checkup_id"]
                ),
                "checkup_time": row["checkup_time"],
                "dataset_split": (
                    row["split"]
                    if "split" in row.index
                    else None
                ),
                "has_clinical_text": bool(
                    clinical_text.strip()
                ),
                "has_photographs": bool(photographs),
                "has_radiographs": bool(radiographs),
                "num_photographs": len(photographs),
                "num_radiographs": len(radiographs),
                "has_all_modalities": (
                    bool(clinical_text.strip())
                    and bool(photographs)
                    and bool(radiographs)
                ),
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["patient_id", "checkup_id"],
        kind="stable",
    ).reset_index(drop=True)


# ============================================================
# Image Trace
# ============================================================

def build_image_trace(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        patient_id = normalize_patient_id(
            row["patient_id"]
        )

        checkup_id = normalize_visit_id(
            row["checkup_id"]
        )

        for modality in IMAGE_MODALITIES:
            filenames = split_image_filenames(
                row[modality]
            )

            for filename in filenames:
                parsed = parse_image_filename(
                    filename
                )

                rows.append(
                    {
                        "id": row["id"],
                        "patient_id": patient_id,
                        "checkup_id": checkup_id,
                        "modality": modality,
                        "filename": filename,
                        "filename_parse_success": parsed[
                            "filename_parse_success"
                        ],
                        "parsed_patient_id": parsed[
                            "parsed_patient_id"
                        ],
                        "parsed_visit_number": parsed[
                            "parsed_visit_number"
                        ],
                        "parsed_image_index": parsed[
                            "parsed_image_index"
                        ],
                        "extension": parsed[
                            "extension"
                        ],
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "patient_id",
                "checkup_id",
                "modality",
                "filename",
                "filename_parse_success",
                "parsed_patient_id",
                "parsed_visit_number",
                "parsed_image_index",
                "extension",
            ]
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        [
            "patient_id",
            "checkup_id",
            "modality",
            "filename",
        ],
        kind="stable",
    ).reset_index(drop=True)


# ============================================================
# Image Repetition Summary
# ============================================================

def build_image_repetition_summary(
    image_trace: pd.DataFrame,
) -> pd.DataFrame:
    if image_trace.empty:
        return pd.DataFrame()

    grouped = (
        image_trace
        .groupby(
            ["modality", "filename"],
            sort=True,
        )
        .agg(
            num_occurrences=("filename", "size"),
            num_unique_visits=("checkup_id", "nunique"),
            num_unique_patients=("patient_id", "nunique"),
            patient_ids=(
                "patient_id",
                lambda values: "|".join(
                    sorted(set(values))
                ),
            ),
            checkup_ids=(
                "checkup_id",
                lambda values: "|".join(
                    sorted(set(values))
                ),
            ),
        )
        .reset_index()
    )

    grouped["repeated_within_same_visit"] = (
        grouped["num_occurrences"] > grouped["num_unique_visits"]
    )

    grouped["repeated_across_visits"] = (
        grouped["num_unique_visits"] > 1
    )

    grouped["repeated_across_patients"] = (
        grouped["num_unique_patients"] > 1
    )

    grouped["is_repeated"] = (
        grouped["num_occurrences"] > 1
    )

    return grouped.sort_values(
        [
            "modality",
            "filename",
        ],
        kind="stable",
    ).reset_index(drop=True)


# ============================================================
# Visit-Level Image Repetition
# ============================================================

def build_visit_image_repetition(
    image_trace: pd.DataFrame,
) -> pd.DataFrame:
    if image_trace.empty:
        return pd.DataFrame()

    first_occurrence = (
        image_trace
        .groupby(
            ["modality", "filename"],
            sort=True,
        )
        .agg(
            first_patient_id=("patient_id", "first"),
            first_checkup_id=("checkup_id", "first"),
        )
        .reset_index()
    )

    trace = image_trace.merge(
        first_occurrence,
        on=["modality", "filename"],
        how="left",
    )

    trace["is_first_filename_occurrence"] = (
        trace["checkup_id"]
        == trace["first_checkup_id"]
    )

    trace["is_repeated_filename"] = (
        ~trace["is_first_filename_occurrence"]
    )

    summary = (
        trace
        .groupby(
            [
                "patient_id",
                "checkup_id",
                "modality",
            ],
            sort=True,
        )
        .agg(
            total_images=("filename", "size"),
            new_images=(
                "is_first_filename_occurrence",
                "sum",
            ),
            repeated_images=(
                "is_repeated_filename",
                "sum",
            ),
            unique_images=(
                "filename",
                "nunique",
            ),
        )
        .reset_index()
    )

    summary["repetition_rate"] = (
        summary["repeated_images"]
        / summary["total_images"]
    )

    return summary.sort_values(
        [
            "patient_id",
            "checkup_id",
            "modality",
        ],
        kind="stable",
    ).reset_index(drop=True)


# ============================================================
# Patient Modality Trace
# ============================================================

def build_patient_modality_trace(
    visit_summary: pd.DataFrame,
) -> pd.DataFrame:
    if visit_summary.empty:
        return pd.DataFrame()

    rows = []

    for patient_id, group in visit_summary.groupby(
        "patient_id",
        sort=True,
    ):
        group = group.sort_values(
            "checkup_id",
            kind="stable",
        ).reset_index(drop=True)

        previous_photographs = False
        previous_radiographs = False

        for visit_index, row in group.iterrows():
            has_photographs = bool(
                row["has_photographs"]
            )

            has_radiographs = bool(
                row["has_radiographs"]
            )

            rows.append(
                {
                    "patient_id": patient_id,
                    "checkup_id": row["checkup_id"],
                    "visit_order": visit_index + 1,
                    "num_visits_for_patient": len(group),
                    "has_clinical_text": row[
                        "has_clinical_text"
                    ],
                    "has_photographs": has_photographs,
                    "has_radiographs": has_radiographs,
                    "photograph_status": (
                        "present"
                        if has_photographs
                        else "missing"
                    ),
                    "radiograph_status": (
                        "present"
                        if has_radiographs
                        else "missing"
                    ),
                    "photograph_appeared_for_first_time": (
                        has_photographs
                        and not previous_photographs
                    ),
                    "radiograph_appeared_for_first_time": (
                        has_radiographs
                        and not previous_radiographs
                    ),
                    "photograph_disappeared_after_previous_visit": (
                        previous_photographs
                        and not has_photographs
                    ),
                    "radiograph_disappeared_after_previous_visit": (
                        previous_radiographs
                        and not has_radiographs
                    ),
                }
            )

            previous_photographs = has_photographs
            previous_radiographs = has_radiographs

    return pd.DataFrame(rows)


# ============================================================
# Image Availability
# ============================================================

def build_image_availability(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        photographs = split_image_filenames(
            row["photographs"]
        )

        radiographs = split_image_filenames(
            row["radiographs"]
        )

        rows.append(
            {
                "id": row["id"],
                "patient_id": normalize_patient_id(
                    row["patient_id"]
                ),
                "checkup_id": normalize_visit_id(
                    row["checkup_id"]
                ),
                "num_photographs": len(
                    photographs
                ),
                "num_radiographs": len(
                    radiographs
                ),
                "has_photographs": bool(
                    photographs
                ),
                "has_radiographs": bool(
                    radiographs
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "patient_id",
            "checkup_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


# ============================================================
# Clinical Missingness
# ============================================================

def build_clinical_missingness(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        result = {
            "id": row["id"],
            "patient_id": normalize_patient_id(
                row["patient_id"]
            ),
            "checkup_id": normalize_visit_id(
                row["checkup_id"]
            ),
        }

        for column in CLINICAL_TEXT_COLUMNS:
            if column not in df.columns:
                continue

            value = row[column]

            result[f"{column}_missing"] = (
                pd.isna(value)
                or not str(value).strip()
            )

        rows.append(result)

    return pd.DataFrame(rows).sort_values(
        [
            "patient_id",
            "checkup_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


# ============================================================
# Audit Summary
# ============================================================

def build_audit_summary(
    df: pd.DataFrame,
    patient_summary: pd.DataFrame,
    visit_summary: pd.DataFrame,
    image_trace: pd.DataFrame,
    image_repetition_summary: pd.DataFrame,
    radiograph_evidence: pd.DataFrame,
    photograph_evidence: pd.DataFrame,
) -> dict[str, Any]:
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
            visit_summary["has_clinical_text"].sum()
        ),
        "visits_with_photographs": int(
            visit_summary["has_photographs"].sum()
        ),
        "visits_with_radiographs": int(
            visit_summary["has_radiographs"].sum()
        ),
        "visits_with_all_modalities": int(
            visit_summary["has_all_modalities"].sum()
        ),
        "patients_with_any_photograph": int(
            patient_summary[
                "has_any_photograph"
            ].sum()
        ),
        "patients_with_any_radiograph": int(
            patient_summary[
                "has_any_radiograph"
            ].sum()
        ),
        "patients_with_both_modalities": int(
            patient_summary[
                "has_both_modalities"
            ].sum()
        ),
        "total_image_trace_rows": int(
            len(image_trace)
        ),
        "unique_image_filenames": int(
            image_trace["filename"].nunique()
            if not image_trace.empty
            else 0
        ),
        "repeated_image_filenames": int(
            image_repetition_summary[
                "is_repeated"
            ].sum()
            if not image_repetition_summary.empty
            else 0
        ),
        "filenames_repeated_across_visits": int(
            image_repetition_summary[
                "repeated_across_visits"
            ].sum()
            if not image_repetition_summary.empty
            else 0
        ),
        "filenames_repeated_across_patients": int(
            image_repetition_summary[
                "repeated_across_patients"
            ].sum()
            if not image_repetition_summary.empty
            else 0
        ),
        "visits_with_radiograph_clinical_evidence_but_no_images": int(
            radiograph_evidence[
                "radiograph_missing_but_clinical_evidence"
            ].sum()
        ),
        "visits_with_radiographs_but_no_clinical_evidence": int(
            radiograph_evidence[
                "radiograph_present_without_clinical_evidence"
            ].sum()
        ),
        "visits_with_photograph_clinical_evidence_but_no_images": int(
            photograph_evidence[
                "photograph_missing_but_clinical_evidence"
            ].sum()
        ),
        "visits_with_photographs_but_no_clinical_evidence": int(
            photograph_evidence[
                "photograph_present_without_clinical_evidence"
            ].sum()
        ),
    }

    return summary


# ============================================================
# Output Handling
# ============================================================

EXPECTED_OUTPUTS = [
    "audit_summary.json",
    "patient_summary.csv",
    "visit_summary.csv",
    "image_availability.csv",
    "clinical_missingness.csv",
    "clinical_radiograph_evidence.csv",
    "clinical_photograph_evidence.csv",
    "image_trace.csv",
    "image_repetition_summary.csv",
    "visit_image_repetition.csv",
    "patient_modality_trace.csv",
]


def outputs_are_complete(
    output_dir: Path,
) -> bool:
    return all(
        (output_dir / filename).exists()
        for filename in EXPECTED_OUTPUTS
    )


def save_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    dataframe.to_csv(
        output_path,
        index=False,
    )


def save_outputs(
    output_dir: Path,
    audit_summary: dict[str, Any],
    patient_summary: pd.DataFrame,
    visit_summary: pd.DataFrame,
    image_availability: pd.DataFrame,
    clinical_missingness: pd.DataFrame,
    radiograph_evidence: pd.DataFrame,
    photograph_evidence: pd.DataFrame,
    image_trace: pd.DataFrame,
    image_repetition_summary: pd.DataFrame,
    visit_image_repetition: pd.DataFrame,
    patient_modality_trace: pd.DataFrame,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_dir / "audit_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit_summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    save_dataframe(
        patient_summary,
        output_dir / "patient_summary.csv",
    )

    save_dataframe(
        visit_summary,
        output_dir / "visit_summary.csv",
    )

    save_dataframe(
        image_availability,
        output_dir / "image_availability.csv",
    )

    save_dataframe(
        clinical_missingness,
        output_dir / "clinical_missingness.csv",
    )

    save_dataframe(
        radiograph_evidence,
        output_dir
        / "clinical_radiograph_evidence.csv",
    )

    save_dataframe(
        photograph_evidence,
        output_dir
        / "clinical_photograph_evidence.csv",
    )

    save_dataframe(
        image_trace,
        output_dir / "image_trace.csv",
    )

    save_dataframe(
        image_repetition_summary,
        output_dir
        / "image_repetition_summary.csv",
    )

    save_dataframe(
        visit_image_repetition,
        output_dir
        / "visit_image_repetition.csv",
    )

    save_dataframe(
        patient_modality_trace,
        output_dir
        / "patient_modality_trace.csv",
    )


# ============================================================
# Main Pipeline
# ============================================================

def main() -> None:
    args = parse_args()

    config = load_config(
        args.config
    )

    dataset_path = get_config_path(
        config,
        "data",
        "dataset_path",
        "data/raw/COde-Dataset/complete_dataset.csv",
    )

    output_dir = get_config_path(
        config,
        "output",
        "patient_trace_audit_dir",
        "results/patient_trace_audit",
    )

    if (
        not args.force
        and outputs_are_complete(output_dir)
    ):
        print(
            "[INFO] Patient Trace Audit outputs "
            "already exist and appear complete."
        )
        print(
            "[INFO] Use --force to regenerate them."
        )
        return

    df = load_dataset(
        dataset_path
    )

    patient_summary = build_patient_summary(
        df
    )

    visit_summary = build_visit_summary(
        df
    )

    image_availability = build_image_availability(
        df
    )

    clinical_missingness = (
        build_clinical_missingness(df)
    )

    (
        radiograph_evidence,
        photograph_evidence,
    ) = build_clinical_evidence(
        df
    )

    image_trace = build_image_trace(
        df
    )

    image_repetition_summary = (
        build_image_repetition_summary(
            image_trace
        )
    )

    visit_image_repetition = (
        build_visit_image_repetition(
            image_trace
        )
    )

    patient_modality_trace = (
        build_patient_modality_trace(
            visit_summary
        )
    )

    audit_summary = build_audit_summary(
        df=df,
        patient_summary=patient_summary,
        visit_summary=visit_summary,
        image_trace=image_trace,
        image_repetition_summary=(
            image_repetition_summary
        ),
        radiograph_evidence=(
            radiograph_evidence
        ),
        photograph_evidence=(
            photograph_evidence
        ),
    )

    save_outputs(
        output_dir=output_dir,
        audit_summary=audit_summary,
        patient_summary=patient_summary,
        visit_summary=visit_summary,
        image_availability=image_availability,
        clinical_missingness=(
            clinical_missingness
        ),
        radiograph_evidence=(
            radiograph_evidence
        ),
        photograph_evidence=(
            photograph_evidence
        ),
        image_trace=image_trace,
        image_repetition_summary=(
            image_repetition_summary
        ),
        visit_image_repetition=(
            visit_image_repetition
        ),
        patient_modality_trace=(
            patient_modality_trace
        ),
    )

    print()
    print("=" * 60)
    print(
        "Patient Trace Audit completed successfully."
    )
    print("=" * 60)
    print(
        f"Results saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()