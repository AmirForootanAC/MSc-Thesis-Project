from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


# ============================================================
# Configuration
# ============================================================

DEFAULT_CONFIG_PATH = Path("configs/audit.yaml")


# ============================================================
# Utility Functions
# ============================================================

def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def ensure_output_dir(
    output_dir: Path,
) -> None:
    """Create output directory if needed."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_json(
    data: dict,
    output_path: Path,
) -> None:
    """Save deterministic JSON output."""

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            sort_keys=True,
        )


# ============================================================
# Data Loading
# ============================================================

def load_duplication_data(
    duplication_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load patient-level duplication audit outputs."""

    duplication_path = (
        duplication_dir
        / "patient_image_duplication.csv"
    )

    patient_summary_path = (
        duplication_dir
        / "patient_duplication_summary.csv"
    )

    if not duplication_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {duplication_path}"
        )

    if not patient_summary_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {patient_summary_path}"
        )

    duplication_df = pd.read_csv(
        duplication_path,
        dtype={
            "patient_id": str,
            "modality": str,
            "sha256": str,
            "visit_1": str,
            "visit_2": str,
            "filename_1": str,
            "filename_2": str,
        },
    )

    patient_summary_df = pd.read_csv(
        patient_summary_path,
        dtype={
            "patient_id": str,
        },
    )

    duplication_df["patient_id"] = (
        duplication_df["patient_id"]
        .str.zfill(4)
    )

    patient_summary_df["patient_id"] = (
        patient_summary_df["patient_id"]
        .str.zfill(4)
    )

    return (
        duplication_df,
        patient_summary_df,
    )


def load_visit_metadata(
    metadata_path: Path,
) -> pd.DataFrame:
    """Load visit-level metadata."""

    required_columns = [
        "checkup_id",
        "patient_id",
        "checkup_time",
    ]

    df = pd.read_csv(
        metadata_path,
        usecols=required_columns,
        dtype={
            "checkup_id": str,
            "patient_id": str,
            "checkup_time": str,
        },
    )

    df["patient_id"] = (
        df["patient_id"]
        .str.zfill(4)
    )

    df["visit_number"] = (
        df["checkup_id"]
        .str.split("-")
        .str[-1]
        .astype(int)
    )

    df["checkup_datetime"] = pd.to_datetime(
        df["checkup_time"],
        errors="coerce",
    )

    return df


# ============================================================
# Duplicated Image Summary
# ============================================================

def build_duplicated_image_summary(
    duplication_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per unique duplicated image.

    Duplication is defined strictly within the same patient.
    """

    records = []

    group_columns = [
        "patient_id",
        "modality",
        "sha256",
    ]

    for (
        patient_id,
        modality,
        sha256,
    ), group in duplication_df.groupby(
        group_columns,
        sort=True,
    ):
        visits = sorted(
            set(
                group["visit_1"].tolist()
                + group["visit_2"].tolist()
            )
        )

        filenames = sorted(
            set(
                group["filename_1"].tolist()
                + group["filename_2"].tolist()
            )
        )

        visit_numbers = sorted(
            int(
                visit.split("-")[-1]
            )
            for visit in visits
        )

        records.append(
            {
                "patient_id": patient_id,
                "modality": modality,
                "sha256": sha256,
                "num_unique_visits": len(visits),
                "visits_with_image": "|".join(
                    visits
                ),
                "filenames_observed": "|".join(
                    filenames
                ),
                "min_visit_number": min(
                    visit_numbers
                ),
                "max_visit_number": max(
                    visit_numbers
                ),
                "visit_span": (
                    max(visit_numbers)
                    - min(visit_numbers)
                ),
                "reused_across_3plus_visits": (
                    len(visits) >= 3
                ),
            }
        )

    return pd.DataFrame(
        records
    ).sort_values(
        [
            "patient_id",
            "modality",
            "min_visit_number",
            "sha256",
        ]
    ).reset_index(
        drop=True
    )


# ============================================================
# Visit Pair Reuse
# ============================================================

def build_visit_pair_reuse(
    duplication_df: pd.DataFrame,
    visit_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per patient/modality/visit-pair.

    Only intra-patient visit pairs are included.
    """

    pair_df = (
        duplication_df
        .groupby(
            [
                "patient_id",
                "modality",
                "visit_1",
                "visit_2",
            ],
            as_index=False,
        )
        .agg(
            num_shared_images=(
                "sha256",
                "nunique",
            )
        )
    )

    pair_df["visit_1_number"] = (
        pair_df["visit_1"]
        .str.split("-")
        .str[-1]
        .astype(int)
    )

    pair_df["visit_2_number"] = (
        pair_df["visit_2"]
        .str.split("-")
        .str[-1]
        .astype(int)
    )

    pair_df["visit_number_gap"] = (
        pair_df["visit_2_number"]
        - pair_df["visit_1_number"]
    )

    pair_df["visit_relationship"] = (
        pair_df["visit_number_gap"]
        .apply(
            lambda value: (
                "consecutive"
                if value == 1
                else "non_consecutive"
            )
        )
    )

    metadata = visit_metadata[
        [
            "checkup_id",
            "checkup_datetime",
        ]
    ].copy()

    metadata_1 = metadata.rename(
        columns={
            "checkup_id": "visit_1",
            "checkup_datetime": "visit_1_datetime",
        }
    )

    metadata_2 = metadata.rename(
        columns={
            "checkup_id": "visit_2",
            "checkup_datetime": "visit_2_datetime",
        }
    )

    pair_df = pair_df.merge(
        metadata_1,
        on="visit_1",
        how="left",
    )

    pair_df = pair_df.merge(
        metadata_2,
        on="visit_2",
        how="left",
    )

    pair_df["temporal_gap_days"] = (
        pair_df[
            "visit_2_datetime"
        ]
        - pair_df[
            "visit_1_datetime"
        ]
    ).dt.days.abs()

    return pair_df.sort_values(
        [
            "patient_id",
            "visit_1_number",
            "visit_2_number",
            "modality",
        ]
    ).reset_index(
        drop=True
    )


# ============================================================
# Patient Longitudinal Reuse
# ============================================================

def build_patient_longitudinal_reuse(
    duplicated_image_summary: pd.DataFrame,
    visit_pair_reuse: pd.DataFrame,
) -> pd.DataFrame:
    """Build patient-level longitudinal reuse summary."""

    patient_ids = sorted(
        set(
            duplicated_image_summary[
                "patient_id"
            ].tolist()
            + visit_pair_reuse[
                "patient_id"
            ].tolist()
        )
    )

    rows = []

    for patient_id in patient_ids:

        patient_images = (
            duplicated_image_summary[
                duplicated_image_summary[
                    "patient_id"
                ]
                == patient_id
            ]
        )

        patient_pairs = (
            visit_pair_reuse[
                visit_pair_reuse[
                    "patient_id"
                ]
                == patient_id
            ]
        )

        rows.append(
            {
                "patient_id": patient_id,

                "num_duplicated_images": int(
                    patient_images.shape[0]
                ),

                "num_reused_visit_pairs": int(
                    patient_pairs[
                        [
                            "visit_1",
                            "visit_2",
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),

                "num_consecutive_reused_pairs": int(
                    (
                        patient_pairs[
                            "visit_relationship"
                        ]
                        == "consecutive"
                    ).sum()
                ),

                "num_non_consecutive_reused_pairs": int(
                    (
                        patient_pairs[
                            "visit_relationship"
                        ]
                        == "non_consecutive"
                    ).sum()
                ),

                "num_images_reused_across_3plus_visits": int(
                    patient_images[
                        "reused_across_3plus_visits"
                    ].sum()
                ),

                "has_consecutive_reuse": bool(
                    (
                        patient_pairs[
                            "visit_relationship"
                        ]
                        == "consecutive"
                    ).any()
                ),

                "has_non_consecutive_reuse": bool(
                    (
                        patient_pairs[
                            "visit_relationship"
                        ]
                        == "non_consecutive"
                    ).any()
                ),

                "has_3plus_visit_reuse": bool(
                    patient_images[
                        "reused_across_3plus_visits"
                    ].any()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Patient Modality Reuse Patterns
# ============================================================

def build_patient_reuse_patterns(
    duplication_df: pd.DataFrame,
    patient_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize duplication patterns by patient and modality."""

    rows = []

    for patient_id, group in duplication_df.groupby(
        "patient_id",
        sort=True,
    ):

        row = {
            "patient_id": patient_id,
        }

        for modality in [
            "photographs",
            "radiographs",
        ]:

            modality_group = group[
                group[
                    "modality"
                ]
                == modality
            ]

            duplicated_images = (
                modality_group[
                    "sha256"
                ]
                .nunique()
            )

            reused_visit_pairs = (
                modality_group[
                    [
                        "visit_1",
                        "visit_2",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            )

            visits_involved = len(
                set(
                    modality_group[
                        "visit_1"
                    ].tolist()
                    + modality_group[
                        "visit_2"
                    ].tolist()
                )
            )

            row[
                f"{modality}_duplicated_images"
            ] = int(
                duplicated_images
            )

            row[
                f"{modality}_reused_visit_pairs"
            ] = int(
                reused_visit_pairs
            )

            row[
                f"{modality}_visits_involved"
            ] = int(
                visits_involved
            )

        rows.append(
            row
        )

    pattern_df = pd.DataFrame(
        rows
    )

    pattern_df = pattern_df.merge(
        patient_summary_df[
            [
                "patient_id",
                "num_visits",
            ]
        ],
        on="patient_id",
        how="left",
    )

    pattern_df[
        "has_both_modality_reuse"
    ] = (
        (
            pattern_df[
                "photographs_duplicated_images"
            ]
            > 0
        )
        & (
            pattern_df[
                "radiographs_duplicated_images"
            ]
            > 0
        )
    )

    return pattern_df.sort_values(
        "patient_id"
    ).reset_index(
        drop=True
    )


# ============================================================
# Audit Summary
# ============================================================

def build_audit_summary(
    duplicated_image_summary: pd.DataFrame,
    visit_pair_reuse: pd.DataFrame,
    patient_longitudinal_reuse: pd.DataFrame,
) -> dict:
    """Build characterization audit summary."""

    return {
        "num_unique_duplicated_images": int(
            duplicated_image_summary.shape[0]
        ),

        "num_patients_with_any_duplication": int(
            (
                patient_longitudinal_reuse[
                    "num_duplicated_images"
                ]
                > 0
            ).sum()
        ),

        "num_reused_visit_pairs": int(
            visit_pair_reuse[
                [
                    "patient_id",
                    "visit_1",
                    "visit_2",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),

        "num_consecutive_reused_visit_pairs": int(
            (
                visit_pair_reuse[
                    "visit_relationship"
                ]
                == "consecutive"
            ).sum()
        ),

        "num_non_consecutive_reused_visit_pairs": int(
            (
                visit_pair_reuse[
                    "visit_relationship"
                ]
                == "non_consecutive"
            ).sum()
        ),

        "num_images_reused_across_3plus_visits": int(
            duplicated_image_summary[
                "reused_across_3plus_visits"
            ].sum()
        ),

        "num_patients_with_3plus_visit_reuse": int(
            patient_longitudinal_reuse[
                "has_3plus_visit_reuse"
            ].sum()
        ),

        "num_patients_with_photograph_reuse": int(
            (
                duplicated_image_summary[
                    duplicated_image_summary[
                        "modality"
                    ]
                    == "photographs"
                ]
                [
                    "patient_id"
                ]
                .nunique()
            )
        ),

        "num_patients_with_radiograph_reuse": int(
            (
                duplicated_image_summary[
                    duplicated_image_summary[
                        "modality"
                    ]
                    == "radiographs"
                ]
                [
                    "patient_id"
                ]
                .nunique()
            )
        ),
    }


# ============================================================
# Main Pipeline
# ============================================================

def main(
    config_path: Path,
    force: bool,
) -> None:
    """Run cross-visit duplication characterization."""

    config = load_config(
        config_path
    )

    metadata_path = Path(
        config[
            "paths"
        ][
            "metadata_file"
        ]
    )

    duplication_dir = Path(
        config[
            "output"
        ][
            "patient_image_duplication_dir"
        ]
    )

    output_dir = Path(
        config[
            "output"
        ][
            "patient_cross_visit_characterization_dir"
        ]
    )

    ensure_output_dir(
        output_dir
    )

    expected_outputs = [
        output_dir
        / "audit_summary.json",

        output_dir
        / "duplicated_image_summary.csv",

        output_dir
        / "visit_pair_reuse.csv",

        output_dir
        / "patient_longitudinal_reuse.csv",

        output_dir
        / "patient_reuse_patterns.csv",
    ]

    if (
        not force
        and all(
            path.exists()
            for path in expected_outputs
        )
    ):
        print(
            "[INFO] Cross-visit characterization "
            "outputs already exist."
        )

        print(
            "[INFO] Use --force to regenerate."
        )

        return

    print(
        "[INFO] Loading duplication audit..."
    )

    (
        duplication_df,
        patient_summary_df,
    ) = load_duplication_data(
        duplication_dir
    )

    print(
        f"[INFO] Loaded "
        f"{len(duplication_df):,} "
        "duplicate image-pair rows."
    )

    print(
        "[INFO] Loading visit metadata..."
    )

    visit_metadata = load_visit_metadata(
        metadata_path
    )

    print(
        "[INFO] Building duplicated image summary..."
    )

    duplicated_image_summary = (
        build_duplicated_image_summary(
            duplication_df
        )
    )

    print(
        "[INFO] Building visit-pair reuse analysis..."
    )

    visit_pair_reuse = (
        build_visit_pair_reuse(
            duplication_df,
            visit_metadata,
        )
    )

    print(
        "[INFO] Building patient longitudinal reuse..."
    )

    patient_longitudinal_reuse = (
        build_patient_longitudinal_reuse(
            duplicated_image_summary,
            visit_pair_reuse,
        )
    )

    print(
        "[INFO] Building patient modality patterns..."
    )

    patient_reuse_patterns = (
        build_patient_reuse_patterns(
            duplication_df,
            patient_summary_df,
        )
    )

    audit_summary = build_audit_summary(
        duplicated_image_summary,
        visit_pair_reuse,
        patient_longitudinal_reuse,
    )

    duplicated_image_summary.to_csv(
        output_dir
        / "duplicated_image_summary.csv",
        index=False,
    )

    visit_pair_reuse.to_csv(
        output_dir
        / "visit_pair_reuse.csv",
        index=False,
    )

    patient_longitudinal_reuse.to_csv(
        output_dir
        / "patient_longitudinal_reuse.csv",
        index=False,
    )

    patient_reuse_patterns.to_csv(
        output_dir
        / "patient_reuse_patterns.csv",
        index=False,
    )

    save_json(
        audit_summary,
        output_dir
        / "audit_summary.json",
    )

    print()
    print("=" * 60)
    print(
        "Patient Cross-Visit Characterization "
        "completed successfully."
    )
    print("=" * 60)

    print(
        f"Unique duplicated images: "
        f"{audit_summary['num_unique_duplicated_images']:,}"
    )

    print(
        f"Patients with duplication: "
        f"{audit_summary['num_patients_with_any_duplication']:,}"
    )

    print(
        f"Reused visit pairs: "
        f"{audit_summary['num_reused_visit_pairs']:,}"
    )

    print(
        f"Consecutive reused pairs: "
        f"{audit_summary['num_consecutive_reused_visit_pairs']:,}"
    )

    print(
        f"Non-consecutive reused pairs: "
        f"{audit_summary['num_non_consecutive_reused_visit_pairs']:,}"
    )

    print(
        f"Images reused across 3+ visits: "
        f"{audit_summary['num_images_reused_across_3plus_visits']:,}"
    )

    print(
        f"Results saved to: "
        f"{output_dir}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Characterize intra-patient "
            "cross-visit image duplication."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force regeneration of "
            "existing outputs."
        ),
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        force=args.force,
    )