from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ============================================================
# Configuration
# ============================================================

IMAGE_MODALITIES = (
    "photographs",
    "radiographs",
)

MODALITY_DIRECTORY_NAMES = {
    "photographs": "Photographs",
    "radiographs": "Radiographs",
}

EXPECTED_OUTPUTS = [
    "audit_summary.json",
    "patient_image_duplication.csv",
    "patient_duplication_summary.csv",
    "image_hash_cache.csv",
]


# ============================================================
# Argument Parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact cross-visit image duplication "
            "within the same patient."
        )
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
        help="Force regeneration of all outputs.",
    )

    return parser.parse_args()


# ============================================================
# Configuration Loading
# ============================================================

def load_config(
    config_path: str,
) -> dict[str, Any]:
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}"
        )

    with config_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration file must contain a YAML mapping."
        )

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

def load_dataset(
    dataset_path: Path,
) -> pd.DataFrame:
    print(
        f"[INFO] Loading dataset: {dataset_path}"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    df = pd.read_csv(
        dataset_path,
        low_memory=False,
    )

    required_columns = {
        "id",
        "checkup_id",
        "patient_id",
        "photographs",
        "radiographs",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print(
        f"[INFO] Loaded {len(df):,} rows."
    )

    return df


# ============================================================
# Utility Functions
# ============================================================

def normalize_patient_id(
    value: Any,
) -> str:
    if pd.isna(value):
        return ""

    try:
        return f"{int(float(value)):04d}"
    except (
        ValueError,
        TypeError,
    ):
        return str(value).strip()


def normalize_visit_id(
    value: Any,
) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def split_image_filenames(
    value: Any,
) -> list[str]:
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        filename.strip()
        for filename in value.split(",")
        if filename.strip()
    ]


# ============================================================
# Image Inventory
# ============================================================

def build_image_inventory(
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
                rows.append(
                    {
                        "id": row["id"],
                        "patient_id": patient_id,
                        "checkup_id": checkup_id,
                        "modality": modality,
                        "filename": filename,
                    }
                )

    inventory = pd.DataFrame(rows)

    if inventory.empty:
        return inventory

    return inventory.sort_values(
        [
            "patient_id",
            "modality",
            "checkup_id",
            "filename",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


# ============================================================
# File Resolution
# ============================================================

def resolve_image_path(
    image_root: Path,
    modality: str,
    filename: str,
) -> Path:
    directory_name = (
        MODALITY_DIRECTORY_NAMES[
            modality
        ]
    )

    return (
        image_root
        / directory_name
        / filename
    )


# ============================================================
# SHA-256 Hashing
# ============================================================

def calculate_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    sha256 = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# Hash Cache
# ============================================================

def load_hash_cache(
    cache_path: Path,
) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(
            columns=[
                "modality",
                "filename",
                "relative_path",
                "file_size",
                "modified_time",
                "sha256",
            ]
        )

    cache = pd.read_csv(
        cache_path
    )

    return cache


def get_cached_hash(
    cache: pd.DataFrame,
    modality: str,
    filename: str,
    file_path: Path,
) -> str | None:
    if cache.empty:
        return None

    matches = cache[
        (cache["modality"] == modality)
        & (cache["filename"] == filename)
    ]

    if matches.empty:
        return None

    record = matches.iloc[0]

    try:
        current_size = file_path.stat().st_size
        current_modified_time = (
            file_path.stat().st_mtime
        )
    except FileNotFoundError:
        return None

    if (
        int(record["file_size"])
        == current_size
        and abs(
            float(
                record[
                    "modified_time"
                ]
            )
            - current_modified_time
        )
        < 1e-6
    ):
        return str(
            record["sha256"]
        )

    return None


def build_hash_cache(
    inventory: pd.DataFrame,
    image_root: Path,
    existing_cache: pd.DataFrame,
) -> pd.DataFrame:
    if inventory.empty:
        return existing_cache

    rows = []

    unique_images = (
        inventory[
            [
                "modality",
                "filename",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "modality",
                "filename",
            ],
            kind="stable",
        )
    )

    total = len(
        unique_images
    )

    print(
        f"[INFO] Processing "
        f"{total:,} unique image files."
    )

    for index, image in enumerate(
        unique_images.itertuples(
            index=False
        ),
        start=1,
    ):
        modality = image.modality
        filename = image.filename

        file_path = resolve_image_path(
            image_root=image_root,
            modality=modality,
            filename=filename,
        )

        if not file_path.exists():
            print(
                "[WARNING] Missing image file: "
                f"{file_path}"
            )

            rows.append(
                {
                    "modality": modality,
                    "filename": filename,
                    "relative_path": str(
                        file_path.relative_to(
                            image_root
                        )
                    ),
                    "file_size": None,
                    "modified_time": None,
                    "sha256": None,
                }
            )

            continue

        cached_hash = get_cached_hash(
            cache=existing_cache,
            modality=modality,
            filename=filename,
            file_path=file_path,
        )

        if cached_hash is not None:
            file_stat = (
                file_path.stat()
            )

            rows.append(
                {
                    "modality": modality,
                    "filename": filename,
                    "relative_path": str(
                        file_path.relative_to(
                            image_root
                        )
                    ),
                    "file_size": file_stat.st_size,
                    "modified_time": file_stat.st_mtime,
                    "sha256": cached_hash,
                }
            )

        else:
            print(
                f"[INFO] Hashing "
                f"{index:,}/{total:,}: "
                f"{modality}/{filename}"
            )

            file_stat = (
                file_path.stat()
            )

            image_hash = (
                calculate_sha256(
                    file_path
                )
            )

            rows.append(
                {
                    "modality": modality,
                    "filename": filename,
                    "relative_path": str(
                        file_path.relative_to(
                            image_root
                        )
                    ),
                    "file_size": file_stat.st_size,
                    "modified_time": file_stat.st_mtime,
                    "sha256": image_hash,
                }
            )

    cache = pd.DataFrame(
        rows
    )

    return cache.sort_values(
        [
            "modality",
            "filename",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


# ============================================================
# Intra-Patient Cross-Visit Duplication
# ============================================================

def find_cross_visit_duplicates(
    inventory: pd.DataFrame,
    hash_cache: pd.DataFrame,
) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()

    inventory = inventory.merge(
        hash_cache[
            [
                "modality",
                "filename",
                "sha256",
            ]
        ],
        on=[
            "modality",
            "filename",
        ],
        how="left",
    )

    inventory = inventory[
        inventory["sha256"].notna()
    ].copy()

    if inventory.empty:
        return pd.DataFrame()

    duplicate_groups = []

    grouped = inventory.groupby(
        [
            "patient_id",
            "modality",
            "sha256",
        ],
        sort=True,
    )

    for (
        patient_id,
        modality,
        image_hash,
    ), group in grouped:

        unique_visits = (
            group[
                "checkup_id"
            ]
            .drop_duplicates()
            .tolist()
        )

        if len(
            unique_visits
        ) < 2:
            continue

        group = group.sort_values(
            [
                "checkup_id",
                "filename",
            ],
            kind="stable",
        )

        records = group.to_dict(
            orient="records"
        )

        for first_index in range(
            len(records)
        ):
            for second_index in range(
                first_index + 1,
                len(records),
            ):
                first = records[
                    first_index
                ]

                second = records[
                    second_index
                ]

                if (
                    first[
                        "checkup_id"
                    ]
                    == second[
                        "checkup_id"
                    ]
                ):
                    continue

                duplicate_groups.append(
                    {
                        "patient_id": patient_id,
                        "modality": modality,
                        "sha256": image_hash,
                        "visit_1": min(
                            first[
                                "checkup_id"
                            ],
                            second[
                                "checkup_id"
                            ],
                        ),
                        "visit_2": max(
                            first[
                                "checkup_id"
                            ],
                            second[
                                "checkup_id"
                            ],
                        ),
                        "filename_1": (
                            first[
                                "filename"
                            ]
                            if first[
                                "checkup_id"
                            ]
                            < second[
                                "checkup_id"
                            ]
                            else second[
                                "filename"
                            ]
                        ),
                        "filename_2": (
                            second[
                                "filename"
                            ]
                            if first[
                                "checkup_id"
                            ]
                            < second[
                                "checkup_id"
                            ]
                            else first[
                                "filename"
                            ]
                        ),
                    }
                )

    if not duplicate_groups:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "modality",
                "sha256",
                "visit_1",
                "visit_2",
                "filename_1",
                "filename_2",
            ]
        )

    return pd.DataFrame(
        duplicate_groups
    ).sort_values(
        [
            "patient_id",
            "modality",
            "visit_1",
            "visit_2",
            "filename_1",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


# ============================================================
# Patient-Level Duplication Summary
# ============================================================

def build_patient_duplication_summary(
    df: pd.DataFrame,
    duplicates: pd.DataFrame,
) -> pd.DataFrame:
    patients = (
        df["patient_id"]
        .apply(
            normalize_patient_id
        )
        .drop_duplicates()
        .sort_values()
    )

    visit_counts = (
        df.assign(
            patient_id_normalized=(
                df["patient_id"]
                .apply(
                    normalize_patient_id
                )
            )
        )
        .groupby(
            "patient_id_normalized"
        )["checkup_id"]
        .nunique()
    )

    summary = pd.DataFrame(
        {
            "patient_id": patients,
        }
    )

    summary[
        "num_visits"
    ] = summary[
        "patient_id"
    ].map(
        visit_counts
    ).fillna(0).astype(int)

    if duplicates.empty:
        summary[
            "photograph_cross_visit_duplicates"
        ] = 0

        summary[
            "radiograph_cross_visit_duplicates"
        ] = 0

        summary[
            "has_photograph_duplication"
        ] = False

        summary[
            "has_radiograph_duplication"
        ] = False

        return summary

    duplicate_counts = (
        duplicates
        .groupby(
            [
                "patient_id",
                "modality",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    summary[
        "photograph_cross_visit_duplicates"
    ] = (
        summary[
            "patient_id"
        ]
        .map(
            duplicate_counts.get(
                "photographs",
                pd.Series(
                    dtype=int
                ),
            )
        )
        .fillna(0)
        .astype(int)
    )

    summary[
        "radiograph_cross_visit_duplicates"
    ] = (
        summary[
            "patient_id"
        ]
        .map(
            duplicate_counts.get(
                "radiographs",
                pd.Series(
                    dtype=int
                ),
            )
        )
        .fillna(0)
        .astype(int)
    )

    summary[
        "has_photograph_duplication"
    ] = (
        summary[
            "photograph_cross_visit_duplicates"
        ]
        > 0
    )

    summary[
        "has_radiograph_duplication"
    ] = (
        summary[
            "radiograph_cross_visit_duplicates"
        ]
        > 0
    )

    return summary.sort_values(
        "patient_id",
        kind="stable",
    ).reset_index(
        drop=True
    )


# ============================================================
# Audit Summary
# ============================================================

def build_audit_summary(
    inventory: pd.DataFrame,
    hash_cache: pd.DataFrame,
    duplicates: pd.DataFrame,
    patient_summary: pd.DataFrame,
) -> dict[str, Any]:
    summary = {
        "num_patients": int(
            patient_summary[
                "patient_id"
            ].nunique()
        ),
        "num_image_references": int(
            len(inventory)
        ),
        "num_unique_image_files": int(
            inventory[
                [
                    "modality",
                    "filename",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "num_hashed_files": int(
            hash_cache[
                "sha256"
            ]
            .notna()
            .sum()
        ),
        "num_cross_visit_duplicate_pairs": int(
            len(duplicates)
        ),
        "patients_with_any_cross_visit_duplication": int(
            (
                patient_summary[
                    [
                        "has_photograph_duplication",
                        "has_radiograph_duplication",
                    ]
                ]
                .any(axis=1)
            ).sum()
        ),
        "patients_with_photograph_duplication": int(
            patient_summary[
                "has_photograph_duplication"
            ].sum()
        ),
        "patients_with_radiograph_duplication": int(
            patient_summary[
                "has_radiograph_duplication"
            ].sum()
        ),
    }

    return summary


# ============================================================
# Output Handling
# ============================================================

def outputs_are_complete(
    output_dir: Path,
) -> bool:
    return all(
        (
            output_dir
            / filename
        ).exists()
        for filename in EXPECTED_OUTPUTS
    )


def save_outputs(
    output_dir: Path,
    audit_summary: dict[str, Any],
    duplicates: pd.DataFrame,
    patient_summary: pd.DataFrame,
    hash_cache: pd.DataFrame,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_dir
        / "audit_summary.json"
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

    duplicates.to_csv(
        output_dir
        / "patient_image_duplication.csv",
        index=False,
    )

    patient_summary.to_csv(
        output_dir
        / "patient_duplication_summary.csv",
        index=False,
    )

    hash_cache.to_csv(
        output_dir
        / "image_hash_cache.csv",
        index=False,
    )


# ============================================================
# Main
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

    image_root = get_config_path(
        config,
        "data",
        "image_root",
        "data/raw/COde-Dataset/Images",
    )

    output_dir = get_config_path(
        config,
        "output",
        "patient_image_duplication_dir",
        "results/patient_image_duplication_audit",
    )

    if (
        not args.force
        and outputs_are_complete(
            output_dir
        )
    ):
        print(
            "[INFO] Patient image duplication "
            "audit outputs already exist."
        )

        print(
            "[INFO] Use --force to regenerate."
        )

        return

    if not image_root.exists():
        raise FileNotFoundError(
            f"Image root not found: {image_root}"
        )

    df = load_dataset(
        dataset_path
    )

    print(
        "[INFO] Building image inventory..."
    )

    inventory = build_image_inventory(
        df
    )

    cache_path = (
        output_dir
        / "image_hash_cache.csv"
    )

    existing_cache = load_hash_cache(
        cache_path
    )

    print(
        "[INFO] Building image hash cache..."
    )

    hash_cache = build_hash_cache(
        inventory=inventory,
        image_root=image_root,
        existing_cache=existing_cache,
    )

    print(
        "[INFO] Searching for "
        "intra-patient cross-visit duplicates..."
    )

    duplicates = (
        find_cross_visit_duplicates(
            inventory=inventory,
            hash_cache=hash_cache,
        )
    )

    patient_summary = (
        build_patient_duplication_summary(
            df=df,
            duplicates=duplicates,
        )
    )

    audit_summary = build_audit_summary(
        inventory=inventory,
        hash_cache=hash_cache,
        duplicates=duplicates,
        patient_summary=patient_summary,
    )

    save_outputs(
        output_dir=output_dir,
        audit_summary=audit_summary,
        duplicates=duplicates,
        patient_summary=patient_summary,
        hash_cache=hash_cache,
    )

    print()
    print(
        "=" * 60
    )

    print(
        "Patient Image Duplication Audit "
        "completed successfully."
    )

    print(
        "=" * 60
    )

    print(
        f"Results saved to: {output_dir}"
    )

    print(
        f"Cross-visit duplicate pairs: "
        f"{audit_summary['num_cross_visit_duplicate_pairs']:,}"
    )

    print(
        f"Patients with any duplication: "
        f"{audit_summary['patients_with_any_cross_visit_duplication']:,}"
    )


if __name__ == "__main__":
    main()