"""
Dataset validation for COde PyTorch Dataset.

This module validates the integrity of the multimodal
dataset layer before model training.
"""

import argparse
import json
from pathlib import Path

from src.data.code_dataset import COdeDataset


CSV_PATH = (
    "data/raw/COde-Dataset/complete_dataset.csv"
)

OUTPUT_DIR = Path(
    "results/dataset_validation"
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def ensure_output():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def validate_dataset(dataset):

    errors = []

    statistics = {

        "total_samples": len(dataset),

        "samples_with_photographs": 0,

        "samples_with_radiographs": 0,

        "samples_with_clinical_text": 0,

    }


    for idx in range(len(dataset)):

        try:

            sample = dataset[idx]


            if sample.photographs:
                statistics[
                    "samples_with_photographs"
                ] += 1


            if sample.radiographs:
                statistics[
                    "samples_with_radiographs"
                ] += 1


            if sample.clinical_text:
                statistics[
                    "samples_with_clinical_text"
                ] += 1



            if (
                sample.radiographs
                and sample.missing_flags[
                    "radiographs_missing"
                ]
            ):
                errors.append(
                    f"Invalid radiograph flag at {idx}"
                )


            if (
                not sample.radiographs
                and not sample.missing_flags[
                    "radiographs_missing"
                ]
            ):
                errors.append(
                    f"Missing radiograph flag at {idx}"
                )


        except Exception as e:

            errors.append(
                {
                    "index": idx,
                    "error": str(e),
                }
            )


    return statistics, errors



def save_results(
    summary,
    errors,
):

    with open(
        OUTPUT_DIR /
        "dataset_validation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "summary": summary,
                "errors": errors,
            },
            f,
            indent=4,
        )



def main():

    parse_args()

    ensure_output()


    print(
        f"[INFO] Loading dataset: {CSV_PATH}"
    )


    dataset = COdeDataset(
        CSV_PATH
    )


    print(
        f"[INFO] Dataset size: {len(dataset)}"
    )


    summary, errors = validate_dataset(
        dataset
    )


    save_results(
        summary,
        errors,
    )


    print()
    print("=" * 60)
    print(
        "Dataset validation completed."
    )
    print("=" * 60)


    print(
        json.dumps(
            {
                "summary": summary,
                "num_errors": len(errors),
            },
            indent=4,
        )
    )


    print()

    print(
        f"Results saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()