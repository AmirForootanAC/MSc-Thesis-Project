"""
Four-Label Patient-Level Dataset Pipeline (SSL v2 & Multimodal Fusion v2)

Creates the canonical four-label benchmark dataset from the
already text-sanitized six-label patient-level dataset by removing
the two noisy/low-retention prosthetic follow-up labels:
    - label_tooth_loss (excluded)
    - label_tooth_structure_loss (excluded)

Retained Core Clinical Labels (4 Domains):
    1. label_caries       (Restorative / Cariology)
    2. label_gingivitis   (Periodontology)
    3. label_malocclusion (Orthodontics)
    4. label_pulpitis     (Endodontics)

Input:
    results/six_label_patient_level_dataset/labeled_dataset.csv

Output:
    results/four_label_patient_level_dataset/
        labeled_dataset.csv
        label_split_distribution.csv
        dataset_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_DATASET_PATH = Path(
    "results/six_label_patient_level_dataset/labeled_dataset.csv"
)

OUTPUT_DIR = Path(
    "results/four_label_patient_level_dataset"
)

OUTPUT_DATASET_PATH = OUTPUT_DIR / "labeled_dataset.csv"
LABEL_DISTRIBUTION_PATH = OUTPUT_DIR / "label_split_distribution.csv"
SUMMARY_PATH = OUTPUT_DIR / "dataset_summary.json"

TEXT_COLUMNS = [
    "chief_complaint",
    "present_illness",
    "past_medical_record",
    "examination",
]

FINAL_LABEL_COLUMNS = [
    "label_caries",
    "label_gingivitis",
    "label_malocclusion",
    "label_pulpitis",
]

DROPPED_LABEL_COLUMNS = [
    "label_tooth_loss",
    "label_tooth_structure_loss",
]

LABEL_NAMES = {
    "label_caries": "Caries",
    "label_gingivitis": "Gingivitis",
    "label_malocclusion": "Malocclusion",
    "label_pulpitis": "Pulpitis",
}

EXPECTED_TOTAL_VISITS = {
    "train": 6129,
    "validation": 1330,
    "test": 1316,
}

EXPECTED_COMPLETE_CASE_VISITS = {
    "train": 2935,
    "validation": 627,
    "test": 633,
}


def has_value(value) -> bool:
    return pd.notna(value) and str(value).strip() != ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the canonical four-label patient-level dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing four-label dataset outputs.",
    )
    return parser.parse_args()


def build_four_label_dataset() -> None:
    args = parse_args()

    if OUTPUT_DIR.exists() and not args.force:
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. Pass --force to overwrite."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Canonical six-label dataset not found at: {INPUT_DATASET_PATH}"
        )

    df = pd.read_csv(INPUT_DATASET_PATH)

    required_cols = [
        "checkup_id",
        "patient_id",
        "split",
        "photographs",
        "radiographs",
        *TEXT_COLUMNS,
        *FINAL_LABEL_COLUMNS,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop excluded noisy labels and old six-label counters if present
    cols_to_remove = [
        c
        for c in [*DROPPED_LABEL_COLUMNS, "six_label_count", "has_six_label"]
        if c in df.columns
    ]
    df = df.drop(columns=cols_to_remove)

    for col in FINAL_LABEL_COLUMNS:
        df[col] = df[col].astype(int)

    df["four_label_count"] = df[FINAL_LABEL_COLUMNS].sum(axis=1).astype(int)
    df["has_four_label"] = (df["four_label_count"] > 0).astype(int)

    # Validate patient-level split isolation
    patient_splits = df.groupby("patient_id")["split"].nunique()
    if (patient_splits > 1).any():
        raise ValueError("Patient-level split leakage detected!")

    # Validate split counts (all visits and complete-case visits)
    is_complete = (
        df["photographs"].apply(has_value)
        & df["radiographs"].apply(has_value)
        & df[TEXT_COLUMNS].notna().any(axis=1)
    )

    for split_name, exp_all in EXPECTED_TOTAL_VISITS.items():
        actual_all = int((df["split"] == split_name).sum())
        if actual_all != exp_all:
            raise RuntimeError(
                f"Split {split_name} total visit mismatch: "
                f"expected {exp_all}, got {actual_all}"
            )

    for split_name, exp_cc in EXPECTED_COMPLETE_CASE_VISITS.items():
        actual_cc = int(((df["split"] == split_name) & is_complete).sum())
        if actual_cc != exp_cc:
            raise RuntimeError(
                f"Split {split_name} complete-case mismatch: "
                f"expected {exp_cc}, got {actual_cc}"
            )

    # Distribution table
    dist_records = []
    for rank, col in enumerate(FINAL_LABEL_COLUMNS, start=1):
        tot_v = int(df[col].sum())
        cc_v = int(df.loc[is_complete, col].sum())
        tot_p = int(df.loc[df[col] == 1, "patient_id"].nunique())

        tr_v = int(df.loc[df["split"] == "train", col].sum())
        va_v = int(df.loc[df["split"] == "validation", col].sum())
        te_v = int(df.loc[df["split"] == "test", col].sum())

        cc_tr = int(df.loc[(df["split"] == "train") & is_complete, col].sum())
        cc_va = int(
            df.loc[(df["split"] == "validation") & is_complete, col].sum()
        )
        cc_te = int(df.loc[(df["split"] == "test") & is_complete, col].sum())

        dist_records.append(
            {
                "rank": rank,
                "column": col,
                "label": LABEL_NAMES[col],
                "all_train_visits": tr_v,
                "all_val_visits": va_v,
                "all_test_visits": te_v,
                "all_total_visits": tot_v,
                "complete_case_train": cc_tr,
                "complete_case_val": cc_va,
                "complete_case_test": cc_te,
                "complete_case_total": cc_v,
                "total_patients": tot_p,
            }
        )

    dist_df = pd.DataFrame(dist_records)

    summary = {
        "dataset": {
            "source": str(INPUT_DATASET_PATH),
            "total_visits": int(len(df)),
            "total_complete_case_visits": int(is_complete.sum()),
            "total_patients": int(df["patient_id"].nunique()),
            "number_of_labels": len(FINAL_LABEL_COLUMNS),
            "labels": FINAL_LABEL_COLUMNS,
            "dropped_labels": DROPPED_LABEL_COLUMNS,
        },
        "expected_total_visits": EXPECTED_TOTAL_VISITS,
        "expected_complete_case_visits": EXPECTED_COMPLETE_CASE_VISITS,
        "label_distribution": dist_records,
    }

    df.to_csv(OUTPUT_DATASET_PATH, index=False)
    dist_df.to_csv(LABEL_DISTRIBUTION_PATH, index=False)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 75)
    print("FOUR-LABEL CANONICAL DATASET CREATED SUCCESSFULLY")
    print("=" * 75)
    print(f"Saved dataset : {OUTPUT_DATASET_PATH}")
    print(f"Total visits  : {len(df):,} | Complete-case visits: {int(is_complete.sum()):,}")
    print("\nLabel Distribution (All vs Complete-Case):")
    print(dist_df.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    build_four_label_dataset()