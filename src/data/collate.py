"""
Custom collation for multimodal COde samples.
"""

from typing import List

from src.data.sample import MultimodalSample



def multimodal_collate(
    batch: List[MultimodalSample],
):
    """
    Collate multimodal samples into a batch.

    Images remain as variable-length lists.
    """

    return {
        "patient_id": [
            sample.patient_id
            for sample in batch
        ],

        "visit_id": [
            sample.visit_id
            for sample in batch
        ],

        "photographs": [
            sample.photographs
            for sample in batch
        ],

        "radiographs": [
            sample.radiographs
            for sample in batch
        ],

        "clinical_text": [
            sample.clinical_text
            for sample in batch
        ],

        "metadata": [
            sample.metadata
            for sample in batch
        ],

        "labels": [
            sample.labels
            for sample in batch
        ],

        "missing_flags": [
            sample.missing_flags
            for sample in batch
        ],
    }