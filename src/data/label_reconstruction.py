"""
Label Reconstruction Pipeline

Reconstructs selected diagnostic labels from the COde dataset
anomalies_en field and attaches them to the original dataset.

Selected labels:
1. Gingivitis
2. Class II Malocclusion
3. Dental Crowding
4. Tooth Structure Loss
5. Dental Caries
6. Convex Profile
7. Mandibular Skeletal Asymmetry
8. Periodontitis
9. Class III Malocclusion
10. Pulpitis
11. Deep Overbite
12. Class I Malocclusion
13. Tooth Loss

Important:
- Labels are reconstructed from anomalies_en only.
- Original dataset columns are preserved.
- A visit may contain multiple reconstructed labels.
- No artificial single-label assignment is performed.

Outputs:
    results/label_reconstruction/
        reconstructed_dataset.csv
        label_summary.csv
        reconstruction_summary.json
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

OUTPUT_DIR = Path(
    "results/label_reconstruction"
)

OUTPUT_DATASET_PATH = (
    OUTPUT_DIR / "reconstructed_dataset.csv"
)

LABEL_SUMMARY_PATH = (
    OUTPUT_DIR / "label_summary.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "reconstruction_summary.json"
)

# ============================================================
# Selected Labels
# ============================================================

LABEL_PATTERNS = {
    "Gingivitis": r"\bgingivitis\b",

    "Class II Malocclusion":
        r"\bclass\s*ii\s+malocclusion\b",

    "Dental Crowding":
        r"\bdental\s+crowding\b",

    "Tooth Structure Loss":
        r"\btooth\s+structure\s+loss\b",

    "Dental Caries":
        r"\bdental\s+caries\b",

    "Convex Profile":
        r"\bconvex\s+profile\b",

    "Mandibular Skeletal Asymmetry":
        r"\bmandibular\s+skeletal\s+asymmetry\b",

    "Periodontitis":
        r"\bperiodontitis\b",

    "Class III Malocclusion":
        r"\bclass\s*iii\s+malocclusion\b",

    "Pulpitis":
        r"\bpulpitis\b",

    "Deep Overbite":
        r"\bdeep\s+overbite\b",

    "Class I Malocclusion":
        r"\bclass\s*i\s+malocclusion\b",

    "Tooth Loss":
        r"\btooth\s+loss\b",
}


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct selected labels from COde "
            "anomalies_en and attach them to the dataset."
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
# Load Dataset
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the complete COde dataset.
    """

    print(
        f"[INFO] Loading dataset: {DATASET_PATH}"
    )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    required_columns = {
        "patient_id",
        "checkup_id",
        "anomalies_en",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print(
        f"[INFO] Loaded {len(df):,} visits."
    )

    print(
        f"[INFO] Patients: "
        f"{df['patient_id'].nunique():,}"
    )

    return df


# ============================================================
# Label Reconstruction
# ============================================================

def reconstruct_labels(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Reconstruct selected labels from anomalies_en.

    Each selected label receives its own binary column.

    A separate reconstructed_labels column stores all
    matched labels for each visit.
    """

    print(
        "[INFO] Reconstructing selected labels..."
    )

    result = df.copy()

    anomalies = (
        result["anomalies_en"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    matched_columns = []

    for label, pattern in LABEL_PATTERNS.items():

        column_name = (
            "label_"
            + label.lower()
            .replace(" ", "_")
        )

        result[column_name] = (
            anomalies
            .str.contains(
                pattern,
                case=False,
                regex=True,
                na=False
            )
            .astype("int8")
        )

        matched_columns.append(
            (label, column_name)
        )

    # --------------------------------------------------------
    # Number of reconstructed labels per visit
    # --------------------------------------------------------

    result["reconstructed_label_count"] = (
        result[
            [
                column
                for _, column in matched_columns
            ]
        ]
        .sum(axis=1)
        .astype("int8")
    )

    # --------------------------------------------------------
    # Combined label representation
    # --------------------------------------------------------

    def combine_labels(row):

        labels = [
            label
            for label, column in matched_columns
            if row[column] == 1
        ]

        return "|".join(labels)

    result["reconstructed_labels"] = (
        result.apply(
            combine_labels,
            axis=1
        )
    )

    result["has_reconstructed_label"] = (
        result["reconstructed_label_count"] > 0
    ).astype("int8")

    return result


# ============================================================
# Label Summary
# ============================================================

def generate_label_summary(
    reconstructed_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate visit-level and patient-level statistics
    for reconstructed labels.
    """

    records = []

    for rank, label in enumerate(
        LABEL_PATTERNS.keys(),
        start=1
    ):

        column_name = (
            "label_"
            + label.lower()
            .replace(" ", "_")
        )

        subset = reconstructed_df[
            reconstructed_df[column_name] == 1
        ]

        records.append(
            {
                "rank": rank,
                "label": label,
                "visits": len(subset),
                "patients": (
                    subset["patient_id"]
                    .nunique()
                ),
                "visit_percentage": (
                    len(subset)
                    / len(reconstructed_df)
                    * 100
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Reconstruction Summary
# ============================================================

def generate_summary(
    original_df: pd.DataFrame,
    reconstructed_df: pd.DataFrame,
    label_summary: pd.DataFrame,
) -> dict:
    """
    Generate global reconstruction summary.
    """

    anomalies = (
        original_df["anomalies_en"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    summary = {

        "dataset": {
            "total_visits": int(
                len(original_df)
            ),
            "total_patients": int(
                original_df["patient_id"]
                .nunique()
            ),
            "visits_with_anomalies_en": int(
                (anomalies != "").sum()
            ),
            "visits_without_anomalies_en": int(
                (anomalies == "").sum()
            ),
        },

        "reconstruction": {
            "number_of_selected_labels": (
                len(LABEL_PATTERNS)
            ),
            "selected_labels": list(
                LABEL_PATTERNS.keys()
            ),
            "visits_with_at_least_one_selected_label": int(
                reconstructed_df[
                    "has_reconstructed_label"
                ].sum()
            ),
            "visits_without_selected_label": int(
                (
                    reconstructed_df[
                        "has_reconstructed_label"
                    ] == 0
                ).sum()
            ),
            "coverage_percentage": float(
                reconstructed_df[
                    "has_reconstructed_label"
                ].mean()
                * 100
            ),
        },

        "multi_label": {
            "visits_with_multiple_selected_labels": int(
                (
                    reconstructed_df[
                        "reconstructed_label_count"
                    ] > 1
                ).sum()
            ),
            "maximum_labels_per_visit": int(
                reconstructed_df[
                    "reconstructed_label_count"
                ].max()
            ),
        },

        "label_statistics": (
            label_summary
            .to_dict(orient="records")
        ),
    }

    return summary


# ============================================================
# Save Outputs
# ============================================================

def save_outputs(
    reconstructed_df: pd.DataFrame,
    label_summary: pd.DataFrame,
    summary: dict,
):
    """
    Save reconstruction outputs.
    """

    print(
        "[INFO] Saving reconstruction outputs..."
    )

    reconstructed_df.to_csv(
        OUTPUT_DATASET_PATH,
        index=False
    )

    label_summary.to_csv(
        LABEL_SUMMARY_PATH,
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
    label_summary: pd.DataFrame,
    summary: dict,
):
    """
    Print concise reconstruction report.
    """

    print()
    print("=" * 80)
    print(
        "LABEL RECONSTRUCTION SUMMARY"
    )
    print("=" * 80)

    print(
        f"Total visits: "
        f"{summary['dataset']['total_visits']:,}"
    )

    print(
        f"Total patients: "
        f"{summary['dataset']['total_patients']:,}"
    )

    print(
        f"Selected labels: "
        f"{summary['reconstruction']['number_of_selected_labels']}"
    )

    print(
        f"Visits with >=1 selected label: "
        f"{summary['reconstruction']['visits_with_at_least_one_selected_label']:,}"
    )

    print(
        f"Coverage: "
        f"{summary['reconstruction']['coverage_percentage']:.2f}%"
    )

    print()
    print(
        f"{'Rank':<6}"
        f"{'Label':<35}"
        f"{'Visits':>10}"
        f"{'Patients':>12}"
        f"{'Visit %':>10}"
    )

    print("-" * 80)

    for _, row in label_summary.iterrows():

        print(
            f"{int(row['rank']):<6}"
            f"{row['label']:<35}"
            f"{int(row['visits']):>10,}"
            f"{int(row['patients']):>12,}"
            f"{row['visit_percentage']:>9.2f}%"
        )

    print()
    print(
        f"Multi-label visits: "
        f"{summary['multi_label']['visits_with_multiple_selected_labels']:,}"
    )

    print(
        f"Maximum labels per visit: "
        f"{summary['multi_label']['maximum_labels_per_visit']}"
    )

    print()
    print(
        f"[INFO] Dataset saved to: "
        f"{OUTPUT_DATASET_PATH}"
    )

    print(
        f"[INFO] Label summary saved to: "
        f"{LABEL_SUMMARY_PATH}"
    )

    print(
        f"[INFO] Summary saved to: "
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

    df = load_dataset()

    reconstructed_df = reconstruct_labels(
        df
    )

    label_summary = generate_label_summary(
        reconstructed_df
    )

    summary = generate_summary(
        df,
        reconstructed_df,
        label_summary
    )

    save_outputs(
        reconstructed_df,
        label_summary,
        summary
    )

    print_report(
        label_summary,
        summary
    )

    print()
    print("=" * 80)
    print(
        "Label reconstruction completed successfully."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()