from pathlib import Path
import json

import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = Path(
    "results/six_label_patient_level_dataset/labeled_dataset.csv"
)

OUTPUT_DIR = Path("results/ssl_dataset_audit")


# SSL text definition:
# Same clinical text fields used by the text-only baseline.
TEXT_COLUMNS = [
    "chief_complaint",
    "present_illness",
    "past_medical_record",
    "examination",
]


# ============================================================
# Helpers
# ============================================================

def has_value(value) -> bool:
    """Return True when a dataframe cell contains usable content."""
    if pd.isna(value):
        return False

    if isinstance(value, str):
        return bool(value.strip())

    return True


def has_text(row) -> bool:
    """Check whether at least one approved clinical text field exists."""
    return any(has_value(row[col]) for col in TEXT_COLUMNS)


def has_images(value) -> bool:
    """Check whether photograph path/list information exists."""
    return has_value(value)


def has_radiographs(value) -> bool:
    """Check whether radiograph path/list information exists."""
    return has_value(value)


# ============================================================
# Main Audit
# ============================================================

def main():
    print("=" * 70)
    print("SSL Dataset Preparation & Pair Availability Audit")
    print("=" * 70)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)

    print(f"\nDataset: {DATASET_PATH}")
    print(f"Rows: {len(df):,}")

    required_columns = [
        "checkup_id",
        "patient_id",
        "photographs",
        "radiographs",
        "split",
        *TEXT_COLUMNS,
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Modality availability
    # --------------------------------------------------------

    df["has_image"] = df["photographs"].apply(has_images)
    df["has_radiograph"] = df["radiographs"].apply(has_radiographs)
    df["has_text"] = df.apply(has_text, axis=1)

    # --------------------------------------------------------
    # Availability pattern
    # --------------------------------------------------------

    def modality_pattern(row):
        image = row["has_image"]
        xray = row["has_radiograph"]
        text = row["has_text"]

        if image and xray and text:
            return "image+xray+text"
        if image and xray:
            return "image+xray"
        if image and text:
            return "image+text"
        if xray and text:
            return "xray+text"
        if image:
            return "image_only"
        if xray:
            return "xray_only"
        if text:
            return "text_only"
        return "none"

    df["modality_pattern"] = df.apply(modality_pattern, axis=1)

    # --------------------------------------------------------
    # Pair / triplet availability
    # --------------------------------------------------------

    df["has_image_text_pair"] = (
        df["has_image"] & df["has_text"]
    )

    df["has_image_xray_pair"] = (
        df["has_image"] & df["has_radiograph"]
    )

    df["has_xray_text_pair"] = (
        df["has_radiograph"] & df["has_text"]
    )

    df["has_complete_triplet"] = (
        df["has_image"]
        & df["has_radiograph"]
        & df["has_text"]
    )

    # --------------------------------------------------------
    # Basic counts
    # --------------------------------------------------------

    total = len(df)

    modality_counts = {
        "total_visits": total,
        "image_available": int(df["has_image"].sum()),
        "radiograph_available": int(df["has_radiograph"].sum()),
        "text_available": int(df["has_text"].sum()),
        "image_text_pairs": int(df["has_image_text_pair"].sum()),
        "image_radiograph_pairs": int(df["has_image_xray_pair"].sum()),
        "radiograph_text_pairs": int(df["has_xray_text_pair"].sum()),
        "complete_triplets": int(df["has_complete_triplet"].sum()),
        "no_modality": int((df["modality_pattern"] == "none").sum()),
    }

    # --------------------------------------------------------
    # Pair percentages
    # --------------------------------------------------------

    percentages = {
        key: round(value / total * 100, 2)
        for key, value in modality_counts.items()
        if key != "total_visits"
    }

    # --------------------------------------------------------
    # Modality pattern distribution
    # --------------------------------------------------------

    pattern_distribution = (
        df.groupby(["modality_pattern", "split"])
        .size()
        .reset_index(name="visits")
        .sort_values(["modality_pattern", "split"])
    )

    pattern_distribution.to_csv(
        OUTPUT_DIR / "modality_pattern_distribution.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Pair distribution by split
    # --------------------------------------------------------

    pair_records = []

    pair_definitions = {
        "image_text": "has_image_text_pair",
        "image_radiograph": "has_image_xray_pair",
        "radiograph_text": "has_xray_text_pair",
        "complete_triplet": "has_complete_triplet",
    }

    for split, split_df in df.groupby("split"):
        for pair_name, column in pair_definitions.items():
            count = int(split_df[column].sum())

            pair_records.append(
                {
                    "split": split,
                    "pair_type": pair_name,
                    "visits": count,
                    "percentage_of_split": round(
                        count / len(split_df) * 100,
                        2,
                    ),
                }
            )

    pair_distribution = pd.DataFrame(pair_records)

    pair_distribution.to_csv(
        OUTPUT_DIR / "pair_availability_by_split.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Modality availability by split
    # --------------------------------------------------------

    split_records = []

    for split, split_df in df.groupby("split"):
        split_records.append(
            {
                "split": split,
                "visits": len(split_df),
                "image_available": int(split_df["has_image"].sum()),
                "radiograph_available": int(
                    split_df["has_radiograph"].sum()
                ),
                "text_available": int(split_df["has_text"].sum()),
                "image_text_pairs": int(
                    split_df["has_image_text_pair"].sum()
                ),
                "image_radiograph_pairs": int(
                    split_df["has_image_xray_pair"].sum()
                ),
                "radiograph_text_pairs": int(
                    split_df["has_xray_text_pair"].sum()
                ),
                "complete_triplets": int(
                    split_df["has_complete_triplet"].sum()
                ),
            }
        )

    split_distribution = pd.DataFrame(split_records)

    split_distribution.to_csv(
        OUTPUT_DIR / "split_availability.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Patient-level validation
    # --------------------------------------------------------

    patient_split_counts = (
        df.groupby("patient_id")["split"]
        .nunique()
    )

    patients_in_multiple_splits = int(
        (patient_split_counts > 1).sum()
    )

    # --------------------------------------------------------
    # Audit summary
    # --------------------------------------------------------

    summary = {
        "dataset": str(DATASET_PATH),
        "total_visits": total,
        "total_patients": int(df["patient_id"].nunique()),
        "split_counts": {
            str(k): int(v)
            for k, v in df["split"].value_counts().to_dict().items()
        },
        "modality_counts": modality_counts,
        "percentages_of_all_visits": percentages,
        "modality_pattern_counts": {
            str(k): int(v)
            for k, v in df["modality_pattern"]
            .value_counts()
            .to_dict()
            .items()
        },
        "patient_level_split_validation": {
            "patients_in_multiple_splits": patients_in_multiple_splits,
            "status": (
                "PASS"
                if patients_in_multiple_splits == 0
                else "FAIL"
            ),
        },
        "text_fields_used_for_ssl": TEXT_COLUMNS,
        "note": (
            "Pair availability is defined at visit level. "
            "A pair means both modalities are available for "
            "the same visit."
        ),
    }

    with open(
        OUTPUT_DIR / "audit_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n--- Overall Modality Availability ---")

    for key, value in modality_counts.items():
        if key == "total_visits":
            print(f"{key:25s}: {value:,}")
        else:
            pct = value / total * 100
            print(
                f"{key:25s}: {value:,} ({pct:.2f}%)"
            )

    print("\n--- Modality Patterns ---")

    for pattern, count in (
        df["modality_pattern"]
        .value_counts()
        .items()
    ):
        print(
            f"{pattern:25s}: "
            f"{count:,} ({count / total * 100:.2f}%)"
        )

    print("\n--- Patient Split Validation ---")
    print(
        f"Patients in multiple splits: "
        f"{patients_in_multiple_splits}"
    )

    print(
        "Status:",
        "PASS" if patients_in_multiple_splits == 0 else "FAIL",
    )

    print("\n--- Output ---")
    print(OUTPUT_DIR / "audit_summary.json")
    print(OUTPUT_DIR / "modality_pattern_distribution.csv")
    print(OUTPUT_DIR / "pair_availability_by_split.csv")
    print(OUTPUT_DIR / "split_availability.csv")

    print("\nAudit completed successfully.")


if __name__ == "__main__":
    main()