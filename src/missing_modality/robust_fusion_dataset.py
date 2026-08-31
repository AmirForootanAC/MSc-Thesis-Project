"""
Dataset utilities for Milestone 8.3.2.

The dataset loads the frozen SSL representations produced during
Milestone 7.

Training uses complete-case representations and applies controlled
modality dropout dynamically.

No test representations are modified.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class RobustFusionRepresentationDataset(Dataset):
    """
    Dataset for robust fusion training/evaluation.

    Expected files:

        results/fusion/ssl_representations/train.pt
        results/fusion/ssl_representations/validation.pt
        results/fusion/ssl_representations/test.pt

    Each file contains:

        checkup_id
        patient_id
        image
        radiograph
        text
        labels
    """

    REQUIRED_KEYS = {
        "checkup_id",
        "patient_id",
        "image",
        "radiograph",
        "text",
        "labels",
    }

    EXPECTED_DIMS = {
        "image": 2048,
        "radiograph": 2048,
        "text": 768,
    }

    SPLITS = {
        "train",
        "validation",
        "test",
    }

    def __init__(
        self,
        representation_root,
        split,
    ):
        if split not in self.SPLITS:
            raise ValueError(
                f"Unknown split: {split}. "
                f"Expected one of {sorted(self.SPLITS)}."
            )

        self.representation_root = Path(
            representation_root
        )

        self.split = split

        path = (
            self.representation_root
            / f"{split}.pt"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Representation file not found:\n{path}"
            )

        data = torch.load(
            path,
            map_location="cpu",
        )

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected dictionary in {path}."
            )

        missing = (
            self.REQUIRED_KEYS
            - set(data.keys())
        )

        if missing:
            raise ValueError(
                f"Missing keys in {path}: "
                f"{sorted(missing)}"
            )

        self.checkup_ids = list(
            data["checkup_id"]
        )

        self.patient_ids = list(
            data["patient_id"]
        )

        self.image = data["image"].float()
        self.radiograph = data[
            "radiograph"
        ].float()
        self.text = data["text"].float()
        self.labels = data["labels"].float()

        self._validate()

        print(
            f"{split}: "
            f"{len(self)} SSL representation samples"
        )

    def _validate(self):

        n = len(self.checkup_ids)

        if len(self.patient_ids) != n:
            raise ValueError(
                "patient_id count mismatch."
            )

        for name, tensor in [
            ("image", self.image),
            ("radiograph", self.radiograph),
            ("text", self.text),
            ("labels", self.labels),
        ]:
            if tensor.ndim != 2:
                raise ValueError(
                    f"{name} representation must be 2D."
                )

            if tensor.shape[0] != n:
                raise ValueError(
                    f"{name} sample count mismatch."
                )

        for name, dim in self.EXPECTED_DIMS.items():

            tensor = getattr(
                self,
                name,
            )

            if tensor.shape[1] != dim:
                raise ValueError(
                    f"Unexpected {name} dimension: "
                    f"{tensor.shape[1]}"
                )

    def __len__(self):
        return len(self.checkup_ids)

    def __getitem__(self, index):

        return {
            "checkup_id":
                self.checkup_ids[index],

            "patient_id":
                self.patient_ids[index],

            "image":
                self.image[index],

            "radiograph":
                self.radiograph[index],

            "text":
                self.text[index],

            "labels":
                self.labels[index],
        }