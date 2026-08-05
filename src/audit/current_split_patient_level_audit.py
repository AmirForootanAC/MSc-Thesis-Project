"""
Current Split vs Required Patient-Level Split Audit.

This audit evaluates the structure and safety of the current COde
dataset splits before constructing a new patient-level split.

The audit analyzes:

1. train.json
2. train-cls.json
3. train-diagnostic.json
4. test_cls.json
5. test_diagnostic.json

The audit extracts patient IDs, visit IDs, and image paths from
the JSON files and checks:

- Number of samples per split
- Number of unique patients per split
- Number of unique visits per split
- Number of unique images per split
- Patient overlap between all split pairs
- Patient leakage between current train and test groups
- Visit overlap between current train and test groups
- Image overlap between current train and test groups
- Overlap between classification and diagnostic task splits
- Whether the current split satisfies patient-level isolation

Important distinction:

- Overlap between train-cls and train-diagnostic is not considered
  train-test leakage because both belong to the training side.
- Overlap between train and test at the patient level is considered
  critical leakage.
- Visit-level and image-level overlap are reported separately.

The final output provides a structured comparison between:

Current Split
and
Required Patient-Level Split

No dataset split is modified by this script.
"""


from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "COde-Dataset"
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "current_split_patient_level_audit"
)


SPLIT_FILES = {
    "train": "train.json",
    "train_cls": "train-cls.json",
    "train_diagnostic": "train-diagnostic.json",
    "test_cls": "test_cls.json",
    "test_diagnostic": "test_diagnostic.json",
}


TRAIN_SPLITS = {
    "train",
    "train_cls",
    "train_diagnostic",
}


TEST_SPLITS = {
    "test_cls",
    "test_diagnostic",
}


# =============================================================================
# JSON Loading
# =============================================================================

def load_split_json(
    path: Path,
) -> list[dict]:
    """
    Load one dataset split JSON file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            f"Expected a list in {path}, "
            f"got {type(data).__name__}."
        )

    return data


# =============================================================================
# Identifier Extraction
# =============================================================================

def extract_image_identifiers(
    image_paths: list[str],
) -> tuple[set[str], set[str], set[str]]:
    """
    Extract patient IDs, visit IDs, and image paths
    from image paths.

    Expected path structure:

    Images/Photographs/3168-001-04.jpg
    Images/Radiographs/3168-001-02.jpg

    The identifier format is assumed to be:

    patient_id-visit_number-image_number

    Example:

    3168-001-04.jpg

    Patient ID:
        3168

    Visit ID:
        3168-001

    The full image path is retained as the image identifier.
    """

    patient_ids = set()
    visit_ids = set()
    normalized_images = set()

    for image_path in image_paths:

        if not isinstance(
            image_path,
            str,
        ):
            continue

        image_path = image_path.strip()

        if not image_path:
            continue

        normalized_path = image_path.replace(
            "\\",
            "/",
        )

        normalized_images.add(
            normalized_path
        )

        filename = Path(
            normalized_path
        ).name

        stem = Path(
            filename
        ).stem

        parts = stem.split("-")

        if len(parts) < 3:
            continue

        patient_id = parts[0]

        visit_number = parts[1]

        patient_ids.add(
            patient_id.zfill(4)
        )

        visit_ids.add(
            f"{patient_id.zfill(4)}-{visit_number}"
        )

    return (
        patient_ids,
        visit_ids,
        normalized_images,
    )


def extract_sample_record(
    sample: dict,
) -> dict:
    """
    Extract patient, visit, and image identifiers
    from one JSON sample.
    """

    image_paths = sample.get(
        "images",
        [],
    )

    if not isinstance(
        image_paths,
        list,
    ):
        image_paths = []

    (
        patient_ids,
        visit_ids,
        image_paths,
    ) = extract_image_identifiers(
        image_paths
    )

    return {
        "patient_ids": patient_ids,
        "visit_ids": visit_ids,
        "image_paths": image_paths,
    }


# =============================================================================
# Split-Level Analysis
# =============================================================================

def build_split_inventory(
    split_data: dict[str, list[dict]],
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, set[str]]],
]:
    """
    Build sample-level inventory and split-level identifier sets.
    """

    records = []

    split_identifiers = {}

    for split_name, samples in split_data.items():

        split_patient_ids = set()
        split_visit_ids = set()
        split_image_paths = set()

        for sample_index, sample in enumerate(
            samples
        ):

            extracted = extract_sample_record(
                sample
            )

            patient_ids = extracted[
                "patient_ids"
            ]

            visit_ids = extracted[
                "visit_ids"
            ]

            image_paths = extracted[
                "image_paths"
            ]

            split_patient_ids.update(
                patient_ids
            )

            split_visit_ids.update(
                visit_ids
            )

            split_image_paths.update(
                image_paths
            )

            records.append(
                {
                    "split": split_name,
                    "sample_index": sample_index,
                    "num_patients_in_sample": len(
                        patient_ids
                    ),
                    "num_visits_in_sample": len(
                        visit_ids
                    ),
                    "num_images_in_sample": len(
                        image_paths
                    ),
                    "patient_ids": "|".join(
                        sorted(
                            patient_ids
                        )
                    ),
                    "visit_ids": "|".join(
                        sorted(
                            visit_ids
                        )
                    ),
                }
            )

        split_identifiers[
            split_name
        ] = {
            "patients": split_patient_ids,
            "visits": split_visit_ids,
            "images": split_image_paths,
        }

    inventory_df = pd.DataFrame(
        records
    )

    return (
        inventory_df,
        split_identifiers,
    )


# =============================================================================
# Split Summary
# =============================================================================

def build_split_summary(
    split_data: dict[str, list[dict]],
    split_identifiers: dict[
        str,
        dict[str, set[str]],
    ],
) -> pd.DataFrame:
    """
    Build one summary row per split.
    """

    rows = []

    for split_name in SPLIT_FILES:

        identifiers = split_identifiers[
            split_name
        ]

        rows.append(
            {
                "split": split_name,
                "source_file": SPLIT_FILES[
                    split_name
                ],
                "num_samples": len(
                    split_data[
                        split_name
                    ]
                ),
                "num_unique_patients": len(
                    identifiers[
                        "patients"
                    ]
                ),
                "num_unique_visits": len(
                    identifiers[
                        "visits"
                    ]
                ),
                "num_unique_images": len(
                    identifiers[
                        "images"
                    ]
                ),
                "split_role": (
                    "train"
                    if split_name
                    in TRAIN_SPLITS
                    else "test"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Pairwise Overlap Analysis
# =============================================================================

def build_pairwise_overlap(
    split_identifiers: dict[
        str,
        dict[str, set[str]],
    ],
) -> pd.DataFrame:
    """
    Compute pairwise overlap between all split pairs.
    """

    rows = []

    split_names = list(
        SPLIT_FILES.keys()
    )

    for split_a, split_b in combinations(
        split_names,
        2,
    ):

        patients_a = split_identifiers[
            split_a
        ][
            "patients"
        ]

        patients_b = split_identifiers[
            split_b
        ][
            "patients"
        ]

        visits_a = split_identifiers[
            split_a
        ][
            "visits"
        ]

        visits_b = split_identifiers[
            split_b
        ][
            "visits"
        ]

        images_a = split_identifiers[
            split_a
        ][
            "images"
        ]

        images_b = split_identifiers[
            split_b
        ][
            "images"
        ]

        shared_patients = (
            patients_a
            & patients_b
        )

        shared_visits = (
            visits_a
            & visits_b
        )

        shared_images = (
            images_a
            & images_b
        )

        is_train_test_pair = (
            (
                split_a
                in TRAIN_SPLITS
            )
            and (
                split_b
                in TEST_SPLITS
            )
        ) or (
            (
                split_b
                in TRAIN_SPLITS
            )
            and (
                split_a
                in TEST_SPLITS
            )
        )

        rows.append(
            {
                "split_a": split_a,
                "split_b": split_b,
                "split_a_role": (
                    "train"
                    if split_a
                    in TRAIN_SPLITS
                    else "test"
                ),
                "split_b_role": (
                    "train"
                    if split_b
                    in TRAIN_SPLITS
                    else "test"
                ),
                "is_train_test_pair": (
                    is_train_test_pair
                ),
                "num_shared_patients": len(
                    shared_patients
                ),
                "num_shared_visits": len(
                    shared_visits
                ),
                "num_shared_images": len(
                    shared_images
                ),
                "shared_patients": "|".join(
                    sorted(
                        shared_patients
                    )
                ),
                "shared_visits": "|".join(
                    sorted(
                        shared_visits
                    )
                ),
                "shared_images": "|".join(
                    sorted(
                        shared_images
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Train/Test Leakage Analysis
# =============================================================================

def build_train_test_leakage_summary(
    split_identifiers: dict[
        str,
        dict[str, set[str]],
    ],
) -> dict:
    """
    Analyze patient, visit, and image leakage
    between the current training and test groups.

    A patient is considered leaked if the same patient
    appears in at least one train split and at least
    one test split.

    The same logic is applied to visits and images.
    """

    train_patients = set().union(
        *[
            split_identifiers[
                split_name
            ][
                "patients"
            ]
            for split_name in TRAIN_SPLITS
        ]
    )

    test_patients = set().union(
        *[
            split_identifiers[
                split_name
            ][
                "patients"
            ]
            for split_name in TEST_SPLITS
        ]
    )

    train_visits = set().union(
        *[
            split_identifiers[
                split_name
            ][
                "visits"
            ]
            for split_name in TRAIN_SPLITS
        ]
    )

    test_visits = set().union(
        *[
            split_identifiers[
                split_name
            ][
                "visits"
            ]
            for split_name in TEST_SPLITS
        ]
    )

    train_images = set().union(
        *[
            split_identifiers[
                split_name
            ][
                "images"
            ]
            for split_name in TRAIN_SPLITS
        ]
    )

    test_images = set().union(
        *[
            split_identifiers[
                split_name
            ][
                "images"
            ]
            for split_name in TEST_SPLITS
        ]
    )

    shared_patients = (
        train_patients
        & test_patients
    )

    shared_visits = (
        train_visits
        & test_visits
    )

    shared_images = (
        train_images
        & test_images
    )

    return {
        "num_train_patients": len(
            train_patients
        ),
        "num_test_patients": len(
            test_patients
        ),
        "num_train_visits": len(
            train_visits
        ),
        "num_test_visits": len(
            test_visits
        ),
        "num_train_images": len(
            train_images
        ),
        "num_test_images": len(
            test_images
        ),
        "num_shared_train_test_patients": len(
            shared_patients
        ),
        "num_shared_train_test_visits": len(
            shared_visits
        ),
        "num_shared_train_test_images": len(
            shared_images
        ),
        "patient_level_leakage_detected": bool(
            shared_patients
        ),
        "visit_level_overlap_detected": bool(
            shared_visits
        ),
        "image_level_overlap_detected": bool(
            shared_images
        ),
        "shared_train_test_patients": sorted(
            shared_patients
        ),
        "shared_train_test_visits": sorted(
            shared_visits
        ),
        "shared_train_test_images": sorted(
            shared_images
        ),
    }


# =============================================================================
# Required Patient-Level Split Decision
# =============================================================================

def build_split_decision(
    leakage_summary: dict,
) -> dict:
    """
    Build the final decision comparing the current split
    against the required patient-level split.
    """

    patient_leakage = leakage_summary[
        "patient_level_leakage_detected"
    ]

    if patient_leakage:

        current_split_status = (
            "UNSAFE_FOR_PATIENT_LEVEL_EVALUATION"
        )

        decision = (
            "The current split cannot be used "
            "as a patient-level train/test split."
        )

    else:

        current_split_status = (
            "NO_PATIENT_LEVEL_TRAIN_TEST_LEAKAGE_DETECTED"
        )

        decision = (
            "No patient-level train/test leakage "
            "was detected in the current split."
        )

    return {
        "current_split_status": (
            current_split_status
        ),
        "required_split_unit": (
            "patient_id"
        ),
        "required_split_constraint": (
            "Each patient_id must belong exclusively "
            "to one of train, validation, or test."
        ),
        "current_split_decision": decision,
        "patient_level_split_required": True,
        "recommendation": (
            "Construct a new patient-level split "
            "before final model training and evaluation."
        ),
    }


# =============================================================================
# Final Audit Summary
# =============================================================================

def build_final_audit_summary(
    split_summary: pd.DataFrame,
    pairwise_overlap: pd.DataFrame,
    leakage_summary: dict,
    split_decision: dict,
) -> dict:
    """
    Build the final Current Split vs Required Patient-Level Split report.
    """

    train_test_pairs = pairwise_overlap[
        pairwise_overlap[
            "is_train_test_pair"
        ]
    ]

    return {
        "audit_name": (
            "Current Split vs Required "
            "Patient-Level Split"
        ),
        "audit_scope": {
            "split_files": SPLIT_FILES,
            "train_splits": sorted(
                TRAIN_SPLITS
            ),
            "test_splits": sorted(
                TEST_SPLITS
            ),
        },
        "split_summary": (
            split_summary
            .to_dict(
                orient="records"
            )
        ),
        "train_test_pairwise_overlap": (
            train_test_pairs
            .to_dict(
                orient="records"
            )
        ),
        "train_test_leakage": leakage_summary,
        "decision": split_decision,
    }


# =============================================================================
# Main Audit Pipeline
# =============================================================================

def run_audit(
    dataset_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> dict:
    """
    Run the complete current split audit.
    """

    summary_path = (
        output_dir
        / "current_split_vs_required_patient_level_split.json"
    )

    if (
        summary_path.exists()
        and not force
    ):

        print(
            "[INFO] Current split audit "
            "already completed."
        )

        print(
            f"[INFO] Existing results found at: "
            f"{output_dir}"
        )

        print(
            "[INFO] Use --force to regenerate."
        )

        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(
                file
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load split files
    # -------------------------------------------------------------------------

    print(
        "[INFO] Loading current dataset splits..."
    )

    split_data = {}

    for split_name, filename in SPLIT_FILES.items():

        path = (
            dataset_dir
            / filename
        )

        print(
            f"[INFO] Loading "
            f"{filename}..."
        )

        split_data[
            split_name
        ] = load_split_json(
            path
        )

        print(
            f"[INFO] {split_name}: "
            f"{len(split_data[split_name]):,} samples"
        )

    # -------------------------------------------------------------------------
    # Build inventory
    # -------------------------------------------------------------------------

    print()
    print(
        "[INFO] Extracting patient, visit, "
        "and image identifiers..."
    )

    (
        inventory_df,
        split_identifiers,
    ) = build_split_inventory(
        split_data
    )

    # -------------------------------------------------------------------------
    # Split summary
    # -------------------------------------------------------------------------

    print(
        "[INFO] Building split summary..."
    )

    split_summary = build_split_summary(
        split_data,
        split_identifiers,
    )

    # -------------------------------------------------------------------------
    # Pairwise overlap
    # -------------------------------------------------------------------------

    print(
        "[INFO] Computing pairwise overlap..."
    )

    pairwise_overlap = build_pairwise_overlap(
        split_identifiers
    )

    # -------------------------------------------------------------------------
    # Train/test leakage
    # -------------------------------------------------------------------------

    print(
        "[INFO] Analyzing train/test leakage..."
    )

    leakage_summary = (
        build_train_test_leakage_summary(
            split_identifiers
        )
    )

    # -------------------------------------------------------------------------
    # Split decision
    # -------------------------------------------------------------------------

    print(
        "[INFO] Building patient-level "
        "split decision..."
    )

    split_decision = build_split_decision(
        leakage_summary
    )

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    final_summary = build_final_audit_summary(
        split_summary=split_summary,
        pairwise_overlap=pairwise_overlap,
        leakage_summary=leakage_summary,
        split_decision=split_decision,
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    inventory_df.to_csv(
        output_dir
        / "split_sample_inventory.csv",
        index=False,
    )

    split_summary.to_csv(
        output_dir
        / "split_summary.csv",
        index=False,
    )

    pairwise_overlap.to_csv(
        output_dir
        / "pairwise_split_overlap.csv",
        index=False,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_summary,
            file,
            indent=4,
            ensure_ascii=False,
            sort_keys=True,
        )

    # -------------------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "Current Split vs Required "
        "Patient-Level Split Audit"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "Split Summary:"
    )

    print(
        split_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Train/Test Leakage:"
    )

    print(
        f"Shared patients: "
        f"{leakage_summary['num_shared_train_test_patients']:,}"
    )

    print(
        f"Shared visits: "
        f"{leakage_summary['num_shared_train_test_visits']:,}"
    )

    print(
        f"Shared images: "
        f"{leakage_summary['num_shared_train_test_images']:,}"
    )

    print()
    print(
        "Patient-Level Leakage Detected: "
        f"{leakage_summary['patient_level_leakage_detected']}"
    )

    print()
    print(
        "Decision:"
    )

    print(
        split_decision[
            "current_split_status"
        ]
    )

    print()
    print(
        "Recommendation:"
    )

    print(
        split_decision[
            "recommendation"
        ]
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        output_dir
    )

    print(
        "=" * 70
    )

    return final_summary


# =============================================================================
# Script Entry Point
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Audit the current COde dataset "
            "splits for patient-level leakage."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=(
            "Directory containing the current "
            "dataset split JSON files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where audit outputs "
            "will be saved."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force regeneration of existing "
            "audit outputs."
        ),
    )

    args = parser.parse_args()

    run_audit(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()