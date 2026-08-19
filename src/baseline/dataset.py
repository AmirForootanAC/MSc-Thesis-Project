from pathlib import Path
import pandas as pd
import torch

from torch.utils.data import Dataset

from src.baseline.image_loader import COdeImageLoader


class COdeBaselineDataset(Dataset):
    """
    Patient-level baseline dataset for COde classification.

    Supports:
    - image
    - radiograph
    - text
    - multi-label targets
    - multi-image loading
    """

    LABEL_COLUMNS = [
        "label_gingivitis",
        "label_class_ii_malocclusion",
        "label_dental_crowding",
        "label_tooth_structure_loss",
        "label_dental_caries",
        "label_convex_profile",
        "label_mandibular_skeletal_asymmetry",
        "label_periodontitis",
        "label_class_iii_malocclusion",
        "label_pulpitis",
        "label_deep_overbite",
        "label_class_i_malocclusion",
        "label_tooth_loss",
    ]

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
        self.image_root = Path(image_root) if image_root else None
        self.transform = transform
        self.require_modality = require_modality

        self.df = pd.read_csv(self.csv_path)

        self.df = self.df[
            self.df["split"] == split
        ].reset_index(drop=True)


        if self.require_modality:

            if self.require_modality == "radiograph":

                self.df = self.df[
                    self.df["radiographs"].notna()
                    &
                    (self.df["radiographs"].str.strip() != "")
                ]

            elif self.require_modality == "photograph":

                self.df = self.df[
                    self.df["photographs"].notna()
                    &
                    (self.df["photographs"].str.strip() != "")
                ]

            else:

                raise ValueError(
                    f"Unknown required modality: {self.require_modality}"
                )


            self.df = self.df.reset_index(drop=True)


        self.image_loader = COdeImageLoader(
            image_root=self.image_root,
            transform=self.transform,
        )


        print(
            f"{split}: {len(self.df)} samples"
        )


        if self.require_modality:

            print(
                f"Required modality: {self.require_modality}"
            )


    def __len__(self):
        return len(self.df)


    def _parse_images(self, value):
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
        - list of image tensors
        """

        images = []

        for filename in filenames:

            image = self.image_loader.load(
                filename=filename,
                modality=modality,
            )

            images.append(image)

        return images


    def _load_labels(self, row):

        labels = (
            row[self.LABEL_COLUMNS]
            .astype(float)
            .values
        )

        return torch.tensor(
            labels,
            dtype=torch.float32
        )


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        sample = {

            "checkup_id": row["checkup_id"],

            "patient_id": row["patient_id"],

            "images": self._load_images(
                self._parse_images(
                    row["photographs"]
                ),
                modality="photograph",
            ),

            "radiographs": self._load_images(
                self._parse_images(
                    row["radiographs"]
                ),
                modality="radiograph",
            ),

            "text": str(
                row["anomalies_en"]
            )
            if not pd.isna(row["anomalies_en"])
            else "",

            "labels": self._load_labels(row)

        }

        return sample