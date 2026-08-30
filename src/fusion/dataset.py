"""
Datasets for Milestone 7 multimodal fusion.

FusionDataset
--------------
Raw complete-case dataset protocol.

FusionRepresentationDataset
----------------------------
Loads frozen SSL representations extracted by
representation_extractor.py.

All fusion experiments use:
    photograph representation : 2048
    radiograph representation : 2048
    text representation       : 768
    labels                     : 6

Missing-modality experiments are intentionally NOT handled here.
They belong to Milestone 8.
"""

from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset

from src.baseline import config


# ============================================================
# Raw complete-case dataset
# ============================================================

class FusionDataset(Dataset):
    """
    Complete-case multimodal dataset for Milestone 7.

    Each sample contains:
        photograph
        radiograph
        clinical text
        six labels
    """

    TEXT_COLUMNS = [
        "chief_complaint",
        "present_illness",
        "past_medical_record",
        "examination",
    ]

    SPLITS = {
        "train",
        "validation",
        "test",
    }

    def __init__(
        self,
        csv_path=None,
        split="train",
    ):

        if split not in self.SPLITS:
            raise ValueError(
                f"Unknown split: {split}. "
                f"Expected one of {sorted(self.SPLITS)}."
            )

        if csv_path is None:
            csv_path = config.DATASET_PATH

        self.csv_path = Path(csv_path)
        self.split = split

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.csv_path}"
            )

        self.df = pd.read_csv(
            self.csv_path
        )

        self._validate_columns()

        # ----------------------------------------------------
        # Authoritative patient-level split
        # ----------------------------------------------------

        self.df = (
            self.df[
                self.df["split"] == split
            ]
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Complete multimodal cases only
        # ----------------------------------------------------

        self.df = self.df[
            self.df["photographs"].apply(
                self.has_value
            )
            &
            self.df["radiographs"].apply(
                self.has_value
            )
            &
            self.df[self.TEXT_COLUMNS]
            .notna()
            .any(axis=1)
        ].reset_index(drop=True)

        print(
            f"{split}: "
            f"{len(self.df)} complete multimodal samples"
        )

    # ========================================================
    # Validation
    # ========================================================

    def _validate_columns(self):

        required = [
            "split",
            "checkup_id",
            "patient_id",
            "photographs",
            "radiographs",
            *self.TEXT_COLUMNS,
            *config.LABEL_NAMES,
        ]

        missing = [
            column
            for column in required
            if column not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def has_value(value):

        return (
            pd.notna(value)
            and str(value).strip() != ""
        )

    @staticmethod
    def parse_images(value):

        if pd.isna(value):
            return []

        return [
            filename.strip()
            for filename in str(value).split(",")
            if filename.strip()
        ]

    @classmethod
    def build_text(cls, row):

        parts = []

        for column in cls.TEXT_COLUMNS:

            value = row[column]

            if pd.notna(value):

                value = str(value).strip()

                if value:
                    parts.append(value)

        return " ".join(parts)

    @staticmethod
    def load_labels(row):

        labels = (
            row[config.LABEL_NAMES]
            .astype(float)
            .values
        )

        return torch.tensor(
            labels,
            dtype=torch.float32,
        )

    # ========================================================
    # Dataset
    # ========================================================

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        return {
            "checkup_id": str(
                row["checkup_id"]
            ),

            "patient_id": str(
                row["patient_id"]
            ),

            "images": self.parse_images(
                row["photographs"]
            ),

            "radiographs": self.parse_images(
                row["radiographs"]
            ),

            "text": self.build_text(
                row
            ),

            "labels": self.load_labels(
                row
            ),
        }


# ============================================================
# SSL representation dataset
# ============================================================

class FusionRepresentationDataset(Dataset):
    """
    Dataset for frozen SSL representations.

    Expected file:

        results/fusion/ssl_representations/{split}.pt

    Each file must contain:

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

        self.path = (
            self.representation_root
            / f"{split}.pt"
        )

        if not self.path.exists():
            raise FileNotFoundError(
                f"Representation file not found:\n"
                f"{self.path}"
            )

        data = torch.load(
            self.path,
            map_location="cpu",
        )

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected dictionary in {self.path}, "
                f"got {type(data).__name__}."
            )

        missing = (
            self.REQUIRED_KEYS
            -
            set(data.keys())
        )

        if missing:
            raise ValueError(
                f"Missing representation keys in "
                f"{self.path}: {sorted(missing)}"
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

    # ========================================================
    # Validation
    # ========================================================

    def _validate(self):

        n = len(self.checkup_ids)

        if len(self.patient_ids) != n:
            raise ValueError(
                "patient_id count does not match "
                "checkup_id count."
            )

        if self.image.ndim != 2:
            raise ValueError(
                f"Image representation must be 2D, "
                f"got {tuple(self.image.shape)}"
            )

        if self.radiograph.ndim != 2:
            raise ValueError(
                "Radiograph representation must be 2D, "
                f"got {tuple(self.radiograph.shape)}"
            )

        if self.text.ndim != 2:
            raise ValueError(
                f"Text representation must be 2D, "
                f"got {tuple(self.text.shape)}"
            )

        if self.labels.ndim != 2:
            raise ValueError(
                f"Labels must be 2D, "
                f"got {tuple(self.labels.shape)}"
            )

        if self.image.shape[0] != n:
            raise ValueError(
                "Image representation count mismatch."
            )

        if self.radiograph.shape[0] != n:
            raise ValueError(
                "Radiograph representation count mismatch."
            )

        if self.text.shape[0] != n:
            raise ValueError(
                "Text representation count mismatch."
            )

        if self.labels.shape[0] != n:
            raise ValueError(
                "Label count mismatch."
            )

        if (
            self.image.shape[1]
            != self.EXPECTED_DIMS["image"]
        ):
            raise ValueError(
                "Unexpected image representation "
                f"dimension: {self.image.shape[1]}"
            )

        if (
            self.radiograph.shape[1]
            != self.EXPECTED_DIMS["radiograph"]
        ):
            raise ValueError(
                "Unexpected radiograph representation "
                f"dimension: {self.radiograph.shape[1]}"
            )

        if (
            self.text.shape[1]
            != self.EXPECTED_DIMS["text"]
        ):
            raise ValueError(
                "Unexpected text representation "
                f"dimension: {self.text.shape[1]}"
            )

        if (
            self.labels.shape[1]
            != config.NUM_LABELS
        ):
            raise ValueError(
                "Unexpected number of labels: "
                f"{self.labels.shape[1]}"
            )

    # ========================================================
    # Dataset
    # ========================================================

    def __len__(self):
        return len(self.checkup_ids)

    def __getitem__(self, index):

        return {
            "checkup_id": self.checkup_ids[index],
            "patient_id": self.patient_ids[index],

            "image": self.image[index],

            "radiograph": self.radiograph[index],

            "text": self.text[index],

            "labels": self.labels[index],
        }