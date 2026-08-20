from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import Dataset

from src.baseline import config
from src.baseline.image_loader import COdeImageLoader


class COdeBaselineDataset(Dataset):
    """
    Patient-level baseline dataset for COde classification.

    Supports:
    - photograph
    - radiograph
    - text
    - multi-label targets
    - multi-image loading

    Labels are defined centrally in src.baseline.config.
    """

    def __init__(
        self,
        csv_path,
        split="train",
        image_root=None,
        transform=None,
        require_modality=None,
    ):

        self.csv_path = Path(csv_path)
        self.split = split
        self.image_root = (
            Path(image_root)
            if image_root
            else None
        )
        self.transform = transform
        self.require_modality = require_modality

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        self.df = pd.read_csv(
            self.csv_path
        )

        # ----------------------------------------------------
        # Validate labels
        # ----------------------------------------------------

        missing_labels = [
            column
            for column in config.LABEL_NAMES
            if column not in self.df.columns
        ]

        if missing_labels:
            raise ValueError(
                "Missing label columns in dataset: "
                f"{missing_labels}"
            )

        # ----------------------------------------------------
        # Validate label count
        # ----------------------------------------------------

        if len(config.LABEL_NAMES) != config.NUM_LABELS:
            raise ValueError(
                "NUM_LABELS does not match LABEL_NAMES: "
                f"{config.NUM_LABELS} vs "
                f"{len(config.LABEL_NAMES)}"
            )

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        self.df = self.df[
            self.df["split"] == split
        ].reset_index(drop=True)

        # ----------------------------------------------------
        # Required modality filtering
        # ----------------------------------------------------

        if self.require_modality:

            if self.require_modality == "radiograph":

                self.df = self.df[
                    self.df["radiographs"].notna()
                    &
                    (
                        self.df["radiographs"]
                        .str.strip()
                        != ""
                    )
                ]

            elif self.require_modality == "photograph":

                self.df = self.df[
                    self.df["photographs"].notna()
                    &
                    (
                        self.df["photographs"]
                        .str.strip()
                        != ""
                    )
                ]

            else:

                raise ValueError(
                    "Unknown required modality: "
                    f"{self.require_modality}"
                )

            self.df = (
                self.df
                .reset_index(drop=True)
            )

        # ----------------------------------------------------
        # Image loader
        # ----------------------------------------------------

        self.image_loader = COdeImageLoader(
            image_root=self.image_root,
            transform=self.transform,
        )

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        print(
            f"{split}: {len(self.df)} samples"
        )

        if self.require_modality:

            print(
                f"Required modality: "
                f"{self.require_modality}"
            )

    def __len__(self):

        return len(self.df)

    def _parse_images(
        self,
        value,
    ):
        """
        Parse comma-separated image filenames.
        """

        if pd.isna(value):
            return []

        return [
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        ]

    def _load_images(
        self,
        filenames,
        modality,
    ):
        """
        Load all images from a modality.

        Returns:
            list[torch.Tensor]
        """

        images = []

        for filename in filenames:

            image = self.image_loader.load(
                filename=filename,
                modality=modality,
            )

            images.append(image)

        return images

    def _load_labels(
        self,
        row,
    ):
        """
        Load multi-label target vector.
        """

        labels = (
            row[config.LABEL_NAMES]
            .astype(float)
            .values
        )

        return torch.tensor(
            labels,
            dtype=torch.float32,
        )

    def __getitem__(
        self,
        idx,
    ):

        row = self.df.iloc[idx]

        sample = {

            "checkup_id":
                row["checkup_id"],

            "patient_id":
                row["patient_id"],

            "images":
                self._load_images(
                    self._parse_images(
                        row["photographs"]
                    ),
                    modality="photograph",
                ),

            "radiographs":
                self._load_images(
                    self._parse_images(
                        row["radiographs"]
                    ),
                    modality="radiograph",
                ),

            "text":
                str(
                    row["anomalies_en"]
                )
                if not pd.isna(
                    row["anomalies_en"]
                )
                else "",

            "labels":
                self._load_labels(row),
        }

        return sample