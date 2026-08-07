"""
Image validation utilities for COde dataset.

This module validates image references from the dataset and checks:

- File existence
- Image readability
- Image dimensions
- Corrupted files

Supported modalities:
- Photographs
- Radiographs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

DEFAULT_DATASET_PATH = Path(
    "data/raw/COde-Dataset/complete_dataset.csv"
)

DEFAULT_IMAGE_ROOT = Path(
    "data/raw/COde-Dataset/Images"
)

DEFAULT_OUTPUT_DIR = Path(
    "results/image_validation"
)


# ---------------------------------------------------------
# Image Validation
# ---------------------------------------------------------


def validate_image(
    image_path: Path,
) -> dict:
    """
    Validate a single image file.

    Checks:
    - existence
    - readability
    - dimensions
    """

    result = {
        "image_path": str(image_path),
        "exists": False,
        "readable": False,
        "width": None,
        "height": None,
        "error": None,
    }

    if not image_path.exists():
        result["error"] = "file_not_found"
        return result

    result["exists"] = True

    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            result["width"] = img.width
            result["height"] = img.height

        result["readable"] = True

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------
# Dataset Extraction
# ---------------------------------------------------------


def extract_image_records(
    df: pd.DataFrame,
    modality: str,
    image_root: Path,
) -> list[dict]:
    """
    Extract image references from dataset.

    Each CSV cell may contain:
    - single filename
    - multiple comma-separated filenames
    """

    records = []

    column = modality.lower()

    modality_dir = image_root / (
        "Photographs"
        if modality == "Photographs"
        else "Radiographs"
    )

    for index, value in df[column].items():

        if pd.isna(value):
            continue

        filenames = [
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        ]

        for filename in filenames:

            image_path = modality_dir / filename

            records.append(
                {
                    "row_index": index,
                    "modality": modality,
                    "filename": filename,
                    "image_path": image_path,
                }
            )

    return records


# ---------------------------------------------------------
# Main Validation Pipeline
# ---------------------------------------------------------


def run_validation(
    dataset_path: Path,
    image_root: Path,
    output_dir: Path,
    force: bool = False,
):
    """
    Execute image validation pipeline.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir /
        "image_validation_summary.json"
    )

    if summary_path.exists() and not force:
        print(
            "[INFO] Validation already exists."
        )
        print(
            "[INFO] Use --force to regenerate."
        )
        return

    print(
        f"[INFO] Loading dataset: {dataset_path}"
    )

    df = pd.read_csv(dataset_path)

    print(
        f"[INFO] Loaded {len(df)} rows."
    )

    all_results = []

    for modality in [
        "Photographs",
        "Radiographs",
    ]:

        print(
            f"[INFO] Validating {modality}"
        )

        records = extract_image_records(
            df=df,
            modality=modality,
            image_root=image_root,
        )

        for record in records:

            result = validate_image(
                record["image_path"]
            )

            result.update(
                {
                    "row_index": record["row_index"],
                    "modality": record["modality"],
                    "filename": record["filename"],
                }
            )

            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    photograph_df = results_df[
        results_df["modality"] == "Photographs"
    ]

    radiograph_df = results_df[
        results_df["modality"] == "Radiographs"
    ]

    corrupted_df = results_df[
        results_df["readable"] == False
    ]

    photograph_df.to_csv(
        output_dir /
        "photograph_validation.csv",
        index=False,
    )

    radiograph_df.to_csv(
        output_dir /
        "radiograph_validation.csv",
        index=False,
    )

    corrupted_df.to_csv(
        output_dir /
        "corrupted_images.csv",
        index=False,
    )

    summary = {
        "total_images_checked": len(results_df),
        "photographs_checked": len(photograph_df),
        "radiographs_checked": len(radiograph_df),
        "missing_files": int(
            (~results_df["exists"]).sum()
        ),
        "unreadable_files": int(
            (~results_df["readable"]).sum()
        ),
    }

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

    print(
        "\n"
        "=" * 60
    )

    print(
        "Image validation completed."
    )

    print(
        json.dumps(
            summary,
            indent=4,
        )
    )

    print(
        f"Results saved to: {output_dir}"
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(
        description="Validate COde dataset images."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing outputs.",
    )

    args = parser.parse_args()

    run_validation(
        dataset_path=DEFAULT_DATASET_PATH,
        image_root=DEFAULT_IMAGE_ROOT,
        output_dir=DEFAULT_OUTPUT_DIR,
        force=args.force,
    )


if __name__ == "__main__":
    main()