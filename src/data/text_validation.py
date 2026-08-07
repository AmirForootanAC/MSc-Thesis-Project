"""
File: src/data/text_validation.py

Description
-----------
Validate clinical text fields in the COde dataset.

This module evaluates:

- Missing values
- Empty strings
- Character length statistics

Outputs are written to:

results/text_validation/

    field_statistics.csv
    missing_text.csv
    text_length_statistics.csv
    text_validation_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.text_loader import ClinicalTextLoader


# ============================================================
# Configuration
# ============================================================

CSV_PATH = Path(
    "data/raw/COde-Dataset/complete_dataset.csv"
)

OUTPUT_DIR = Path(
    "results/text_validation"
)


# ============================================================
# Utilities
# ============================================================

def ensure_output_directory(force: bool = False) -> None:
    """
    Create output directory.

    Parameters
    ----------
    force : bool
        Currently kept for consistency with other audit scripts.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def compute_statistics(series: pd.Series) -> dict:
    """
    Compute character-length statistics for non-empty texts.

    Parameters
    ----------
    series : pandas.Series

    Returns
    -------
    dict
    """

    lengths = (
        series.astype(str)
        .str.len()
    )

    lengths = lengths[lengths > 0]

    if len(lengths) == 0:
        return {
            "min_length": 0,
            "max_length": 0,
            "mean_length": 0.0,
            "median_length": 0.0,
        }

    return {
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.mean()),
        "median_length": float(lengths.median()),
    }


# ============================================================
# Validation
# ============================================================

def run_validation(df: pd.DataFrame):
    """
    Validate all supported clinical text fields.
    """

    loader = ClinicalTextLoader()

    statistics = []
    missing_rows = []

    total_rows = len(df)

    for field in loader.TEXT_FIELDS:

        print(f"[INFO] Validating: {field}")

        normalized = (
            df[field]
            .apply(loader.normalize_text)
        )

        missing_mask = normalized == ""

        num_missing = int(missing_mask.sum())
        num_available = int(total_rows - num_missing)

        stats = compute_statistics(
            normalized
        )

        statistics.append(
            {
                "field": field,
                "total_rows": total_rows,
                "available": num_available,
                "missing": num_missing,
                "missing_percent": (
                    num_missing / total_rows * 100
                ),
                **stats,
            }
        )

        if num_missing > 0:

            tmp = df.loc[
                missing_mask,
                [
                    "patient_id",
                    "checkup_id",
                ],
            ].copy()

            tmp["field"] = field

            missing_rows.append(tmp)

    statistics_df = pd.DataFrame(
        statistics
    )

    if missing_rows:
        missing_df = pd.concat(
            missing_rows,
            ignore_index=True,
        )
    else:
        missing_df = pd.DataFrame(
            columns=[
                "patient_id",
                "checkup_id",
                "field",
            ]
        )

    return (
        statistics_df,
        missing_df,
    )


# ============================================================
# Save Results
# ============================================================

def save_results(
    statistics_df,
    missing_df,
    summary,
):
    """
    Save CSV outputs.
    """

    statistics_df.to_csv(
        OUTPUT_DIR / "field_statistics.csv",
        index=False,
    )

    statistics_df[
        [
            "field",
            "min_length",
            "mean_length",
            "median_length",
            "max_length",
        ]
    ].to_csv(
        OUTPUT_DIR / "text_length_statistics.csv",
        index=False,
    )

    missing_df.to_csv(
        OUTPUT_DIR / "missing_text.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "text_validation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    with open(
        OUTPUT_DIR / "text_validation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

# ============================================================
# CLI
# ============================================================

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Validate clinical text fields in the COde dataset."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite previous validation results.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    """
    Main entry point.
    """

    args = parse_args()

    ensure_output_directory(
        force=args.force,
    )

    print(f"[INFO] Loading dataset: {CSV_PATH}")

    df = pd.read_csv(
        CSV_PATH,
    )

    print(f"[INFO] Loaded {len(df)} rows.")

    statistics_df, missing_df = run_validation(
        df
    )

    total_missing = int(
        statistics_df["missing"].sum()
    )

    total_cells = (
        len(df)
        * len(statistics_df)
    )

    overall_missing_rate = (
        total_missing
        / total_cells
        * 100
    )

    summary = {
        "total_rows": len(df),
        "validated_fields": len(statistics_df),
        "fields": statistics_df["field"].tolist(),
        "total_missing_values": total_missing,
        "overall_missing_rate": round(
            overall_missing_rate,
            2,
        ),
    }

    save_results(
        statistics_df,
        missing_df,
        summary,
    )

    print()
    print("=" * 60)
    print("Clinical text validation completed.")
    print("=" * 60)

    print(
        json.dumps(
            summary,
            indent=4,
        )
    )

    print()
    print(
        f"Results saved to: {OUTPUT_DIR}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()