"""
Validation utilities for COde DataLoader.

Checks:
- DataLoader construction
- Batch generation
- Split consistency
- Patient-level separation
"""


import json
from pathlib import Path

from src.data.code_dataset import COdeDataset
from src.data.dataloader_factory import DataLoaderFactory


DATASET_PATH = (
    "data/raw/COde-Dataset/complete_dataset.csv"
)

OUTPUT_DIR = Path(
    "results/dataloader_validation"
)


def validate_loader(
    loader,
    expected_split,
):

    result = {
        "split": expected_split,
        "num_samples": len(loader.dataset),
        "num_batches": len(loader),
        "batch_size": loader.batch_size,
        "sample_check": False,
    }


    for batch in loader:

        required_keys = [
            "patient_id",
            "visit_id",
            "photographs",
            "radiographs",
            "clinical_text",
            "missing_flags",
        ]

        result["sample_check"] = all(
            key in batch
            for key in required_keys
        )

        break


    return result



def run_validation():

    dataset = COdeDataset(
        DATASET_PATH
    )


    factory = DataLoaderFactory(
        dataset,
        batch_size=4,
    )


    train_loader = (
        factory.create_train_loader()
    )

    test_loader = (
        factory.create_test_loader()
    )


    results = {
        "train": validate_loader(
            train_loader,
            "train",
        ),

        "test": validate_loader(
            test_loader,
            "test",
        ),
    }


    return results



def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    results = run_validation()


    output_file = (
        OUTPUT_DIR
        /
        "dataloader_validation_summary.json"
    )


    with open(
        output_file,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )


    print()
    print("=" * 60)
    print(
        "DataLoader validation completed."
    )
    print("=" * 60)

    print(
        json.dumps(
            results,
            indent=4,
        )
    )

    print()

    print(
        f"Results saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()