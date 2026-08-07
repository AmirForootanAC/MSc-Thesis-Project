"""
Missing modality validation for COde dataset.

This module analyzes natural modality missingness
without removing samples or performing imputation.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.sample import MultimodalSample
from src.data.missing_modality import generate_missing_flags


CSV_PATH = (
    "data/raw/COde-Dataset/complete_dataset.csv"
)

OUTPUT_DIR = Path(
    "results/missing_modality"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results",
    )

    return parser.parse_args()


def ensure_output_directory():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def parse_images(value):
    """
    Convert comma separated image names
    into list representation.
    """

    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


def build_sample(row):
    """
    Build minimal multimodal sample.
    """

    return MultimodalSample(
        patient_id=str(row["patient_id"]),
        visit_id=str(row["checkup_id"]),

        photographs=parse_images(
            row["photographs"]
        ),

        radiographs=parse_images(
            row["radiographs"]
        ),

        clinical_text={
            "examination": str(
                row["examination"]
            )
            if not pd.isna(row["examination"])
            else ""
        },
    )


def run_validation(df):

    counters = {
        "photographs_missing": 0,
        "radiographs_missing": 0,
        "clinical_text_missing": 0,
    }

    for _, row in df.iterrows():

        sample = build_sample(row)

        flags = generate_missing_flags(
            sample
        )

        for key, value in flags.items():

            if value:
                counters[key] += 1

    return counters


def save_results(summary):

    summary_path = (
        OUTPUT_DIR
        /
        "missing_modality_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )


def main():

    parse_args()

    ensure_output_directory()

    print(
        f"[INFO] Loading dataset: {CSV_PATH}"
    )

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"[INFO] Loaded {len(df)} rows."
    )

    counters = run_validation(
        df
    )

    summary = {
        "total_samples": len(df),
        **counters,
    }


    for key in counters:

        summary[
            key + "_rate"
        ] = round(
            counters[key]
            /
            len(df)
            *
            100,
            2,
        )


    save_results(
        summary
    )

    print()
    print("=" * 60)
    print(
        "Missing modality validation completed."
    )
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


if __name__ == "__main__":
    main()