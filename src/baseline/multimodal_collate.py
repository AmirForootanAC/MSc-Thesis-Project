import torch


def multimodal_collate(batch):
    """
    Collate function for full multimodal baseline.

    Handles:
    - variable length photographs
    - variable length radiographs
    - tokenized clinical text
    - multi-label targets
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


        "input_ids": torch.stack(
            [
                sample["input_ids"]
                for sample in batch
            ]
        ),


        "attention_mask": torch.stack(
            [
                sample["attention_mask"]
                for sample in batch
            ]
        ),


        "labels": torch.stack(
            [
                sample["labels"]
                for sample in batch
            ]
        ),
    }