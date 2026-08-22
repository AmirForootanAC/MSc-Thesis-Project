import torch


def text_collate(batch):
    return {
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

        "text": [
            sample["text"]
            for sample in batch
        ],
    }