"""
Collate function for dynamic multimodal SSL batches.

Missing modalities are preserved.
No modality is artificially padded or replaced.
"""

import torch


def ssl_collate(batch):
    """
    Collate dynamic multimodal SSL samples.

    Each modality remains optional.

    Returns:
        checkup_id
        patient_id
        images
        radiographs
        text
        has_image
        has_radiograph
        has_text
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

        "has_image": torch.tensor(
            [
                sample["has_image"]
                for sample in batch
            ],
            dtype=torch.bool,
        ),

        "has_radiograph": torch.tensor(
            [
                sample["has_radiograph"]
                for sample in batch
            ],
            dtype=torch.bool,
        ),

        "has_text": torch.tensor(
            [
                sample["has_text"]
                for sample in batch
            ],
            dtype=torch.bool,
        ),
    }