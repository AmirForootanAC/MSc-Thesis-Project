import pandas as pd
import torch

from torch.utils.data import Dataset

from src.baseline import config


class SSLDownstreamDataset(Dataset):
    """
    Dataset for SSL downstream classification.

    Supports:
        - image
        - radiograph
        - text

    For single-modality evaluation/training, samples without
    the requested modality are excluded.

    Labels:
        Same six labels used by the supervised baselines.
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
        modality=None,
    ):

        self.df = pd.read_csv(csv_path)

        # --------------------------------------------------
        # Validate columns
        # --------------------------------------------------

        required = [
            "split",
            "photographs",
            "radiographs",
            "checkup_id",
            "patient_id",
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

        # --------------------------------------------------
        # Patient-level split
        # --------------------------------------------------

        self.df = (
            self.df[
                self.df["split"] == split
            ]
            .reset_index(drop=True)
        )

        # --------------------------------------------------
        # Modality filtering
        # --------------------------------------------------

        if modality is not None:

            if modality not in [
                "image",
                "radiograph",
                "text",
            ]:
                raise ValueError(
                    f"Unknown modality: {modality}"
                )

            if modality == "image":

                self.df = self.df[
                    self.df["photographs"]
                    .notna()
                    &
                    (
                        self.df["photographs"]
                        .astype(str)
                        .str.strip()
                        != ""
                    )
                ]

            elif modality == "radiograph":

                self.df = self.df[
                    self.df["radiographs"]
                    .notna()
                    &
                    (
                        self.df["radiographs"]
                        .astype(str)
                        .str.strip()
                        != ""
                    )
                ]

            elif modality == "text":

                self.df = self.df[
                    self.df.apply(
                        lambda row: bool(
                            self.build_text(row)
                        ),
                        axis=1,
                    )
                ]

            self.df = (
                self.df
                .reset_index(drop=True)
            )

        # --------------------------------------------------
        # Report
        # --------------------------------------------------

        print(
            f"{split}: {len(self.df)} SSL downstream samples"
        )

    # ======================================================
    # Helpers
    # ======================================================

    @staticmethod
    def parse_images(value):

        if pd.isna(value):
            return []

        return [
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        ]

    @classmethod
    def build_text(
        cls,
        row,
    ):

        parts = []

        for column in cls.TEXT_COLUMNS:

            value = row[column]

            if pd.notna(value):

                value = str(value).strip()

                if value:
                    parts.append(value)

        return " ".join(parts)

    @staticmethod
    def load_labels(
        row,
    ):

        labels = (
            row[config.LABEL_NAMES]
            .astype(float)
            .values
        )

        return torch.tensor(
            labels,
            dtype=torch.float32,
        )

    # ======================================================
    # Dataset
    # ======================================================

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        idx,
    ):

        row = self.df.iloc[idx]

        return {

            "checkup_id":
                str(row["checkup_id"]),

            "patient_id":
                str(row["patient_id"]),

            "images":
                self.parse_images(
                    row["photographs"]
                ),

            "radiographs":
                self.parse_images(
                    row["radiographs"]
                ),

            "text":
                self.build_text(row),

            "labels":
                self.load_labels(row),
        }