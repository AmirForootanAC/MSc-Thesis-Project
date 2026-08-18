"""
Custom collation for baseline COde samples.
"""

import torch


def baseline_collate(batch):
    """
    Collate baseline samples into a batch.

    Images and radiographs remain as variable-length lists.
    Labels are stacked into a batch tensor.
    """

    return {
        "checkup_id": [
            sample["checkup_id"]
            for sample in batch
        ],

        "patient_id": [
            sample["patient_id"]
            for sample in batch
        ],

        "images": [
            sample["images"]
            for sample in batch
        ],

        "radiographs": [
            sample["radiographs"]
            for sample in batch
        ],

        "text": [
            sample["text"]
            for sample in batch
        ],

        "labels": torch.stack(
            [
                sample["labels"]
                for sample in batch
            ]
        ),
    }