"""
Dynamic multimodal dataset for self-supervised learning on COde.

Unlike the supervised multimodal baseline, this dataset does NOT
require complete multimodal visits.

A visit may contain:
    - photographs
    - radiographs
    - clinical text

Missing modalities are preserved and represented by empty lists / flags.

Text policy follows the supervised baseline:
    - chief_complaint
    - present_illness
    - past_medical_record
    - examination

Excluded:
    - anomalies_en
    - diagnosis
    - treatment_plan
    - treatment_recommendations
    - management
    - other post-diagnostic fields
"""

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class COdeSSLDataset(Dataset):
    """
    Dynamic multimodal dataset for self-supervised pretraining.

    Unit of sample:
        One dental visit.

    Important:
        No complete-case filtering is applied.
    """

    TEXT_COLUMNS = [
        "chief_complaint",
        "present_illness",
        "past_medical_record",
        "examination",
    ]

    def __init__(
        self,
        csv_path,
        split="train",
    ):
        self.csv_path = Path(csv_path)
        self.split = split

        self.df = pd.read_csv(
            self.csv_path
        )

        # --------------------------------------------------
        # Validate required columns
        # --------------------------------------------------

        required_columns = [
            "checkup_id",
            "patient_id",
            "photographs",
            "radiographs",
            *self.TEXT_COLUMNS,
            "split",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # --------------------------------------------------
        # Patient-level authoritative split
        # --------------------------------------------------

        self.df = self.df[
            self.df["split"] == split
        ].reset_index(drop=True)

        print(
            f"{split}: {len(self.df)} SSL visits"
        )

    def __len__(self):
        return len(self.df)

    @staticmethod
    def has_value(value):
        """
        Return True when a dataframe cell contains
        meaningful content.
        """

        return (
            pd.notna(value)
            and str(value).strip() != ""
        )

    @staticmethod
    def parse_images(value):
        """
        Parse comma-separated image filenames.

        Missing modality -> empty list.
        """

        if pd.isna(value):
            return []

        return [
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        ]

    def build_text(self, row):
        """
        Build safe clinical text representation.

        Only the same four clinical fields used by the
        supervised text baseline are included.
        """

        parts = []

        for column in self.TEXT_COLUMNS:

            value = row[column]

            if pd.notna(value):

                value = str(value).strip()

                if value:
                    parts.append(value)

        return " ".join(parts)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        images = self.parse_images(
            row["photographs"]
        )

        radiographs = self.parse_images(
            row["radiographs"]
        )

        text = self.build_text(
            row
        )

        return {
            "checkup_id": str(
                row["checkup_id"]
            ),

            "patient_id": str(
                row["patient_id"]
            ),

            "images": images,

            "radiographs": radiographs,

            "text": text,

            "has_image": len(images) > 0,

            "has_radiograph": len(radiographs) > 0,

            "has_text": bool(text),
        }