# ============================================================
# Episode Relation Review Sampling
# ============================================================
#
# Purpose:
# Create a stratified manual-review sample from the episode
# relation characterization output.
#
# This script does NOT modify the original characterization.
# It only creates a compact CSV for sanity checking the
# episode classification heuristic.
#
# Sampling groups:
# - possible_same_episode: 30
# - insufficient_evidence: 20
# - possible_different_episode: 20
# - likely_same_episode: 10
# - likely_different_episode: 10
#
# The sampling is stratified by episode relation and uses a
# fixed random seed for reproducibility.
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path(
    "results/patient_treatment_characterization/"
    "duplicated_visit_treatment.csv"
)

OUTPUT_PATH = Path(
    "results/patient_treatment_characterization/"
    "episode_relation_review_sample.csv"
)

RANDOM_STATE = 42


SAMPLE_SIZES = {
    "possible_same_episode": 30,
    "insufficient_evidence": 20,
    "possible_different_episode": 20,
    "likely_same_episode": 10,
    "likely_different_episode": 10,
}


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create a stratified manual-review sample "
            "for episode relation characterization."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Input duplicated visit characterization CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output review sample CSV.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        "[INFO] Loading episode characterization data..."
    )

    if not args.input.exists():

        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    df = pd.read_csv(
        args.input,
        low_memory=False,
    )

    print(
        f"[INFO] Loaded {len(df):,} rows."
    )

    # --------------------------------------------------------
    # Columns for manual review
    # --------------------------------------------------------

    review_columns = [
        "patient_id",
        "modality",
        "visit_1",
        "visit_2",
        "visit_1_number",
        "visit_2_number",
        "visit_number_gap",
        "visit_relationship",
        "visit_1_datetime",
        "visit_2_datetime",
        "temporal_gap_days_calculated",
        "temporal_gap_category",

        "diagnosis_1",
        "diagnosis_2",
        "diagnosis_relation",

        "treatment_context_1",
        "treatment_context_2",
        "treatment_relation",

        "clinical_context_1",
        "clinical_context_2",
        "clinical_context_relation",

        "teeth_mentioned_1",
        "teeth_mentioned_2",
        "shared_teeth",
        "num_shared_teeth",
        "tooth_overlap_relation",

        "treatment_stages_1",
        "treatment_stages_2",
        "treatment_stage_relation",
        "treatment_progression",

        "episode_relation",
    ]

    missing_columns = [
        column
        for column in review_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # Stratified sampling
    # --------------------------------------------------------

    sampled_groups = []

    print()
    print(
        "[INFO] Creating stratified review sample..."
    )

    for relation, sample_size in SAMPLE_SIZES.items():

        group = df[
            df["episode_relation"]
            == relation
        ].copy()

        available = len(group)

        actual_sample_size = min(
            sample_size,
            available,
        )

        if actual_sample_size == 0:

            print(
                f"[WARNING] No rows found for: "
                f"{relation}"
            )

            continue

        sample = group.sample(
            n=actual_sample_size,
            random_state=RANDOM_STATE,
        ).copy()

        # Add explicit review order
        sample["review_group"] = relation

        sampled_groups.append(
            sample
        )

        print(
            f"[INFO] {relation}: "
            f"requested={sample_size}, "
            f"available={available}, "
            f"sampled={actual_sample_size}"
        )

    # --------------------------------------------------------
    # Combine samples
    # --------------------------------------------------------

    if not sampled_groups:

        raise RuntimeError(
            "No samples could be created."
        )

    review_df = pd.concat(
        sampled_groups,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Sort groups for easier manual review
    # --------------------------------------------------------

    relation_order = {
        "likely_same_episode": 1,
        "possible_same_episode": 2,
        "insufficient_evidence": 3,
        "possible_different_episode": 4,
        "likely_different_episode": 5,
    }

    review_df["_relation_order"] = (
        review_df["episode_relation"]
        .map(relation_order)
        .fillna(999)
    )

    review_df = (
        review_df
        .sort_values(
            [
                "_relation_order",
                "patient_id",
                "visit_1",
                "visit_2",
            ]
        )
        .drop(
            columns=[
                "_relation_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Add review ID
    # --------------------------------------------------------

    review_df.insert(
        0,
        "review_id",
        range(
            1,
            len(review_df) + 1,
        ),
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_df[
        [
            "review_id",
            "review_group",
        ]
        + review_columns
    ].to_csv(
        args.output,
        index=False,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "Episode Relation Review Sample "
        "created successfully."
    )

    print(
        "=" * 60
    )

    print(
        f"Total review samples: "
        f"{len(review_df):,}"
    )

    print()
    print(
        "Samples by episode relation:"
    )

    print(
        review_df[
            "episode_relation"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Samples by tooth overlap:"
    )

    print(
        review_df[
            "tooth_overlap_relation"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        args.output
    )


if __name__ == "__main__":
    main()