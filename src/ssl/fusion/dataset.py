import pandas as pd
import torch

from torch.utils.data import Dataset
from src.ssl.fusion.preprocess import FusionPreprocessor

from src.baseline import config


class MultimodalFusionDataset(Dataset):
    """
    Dataset for multimodal fusion stage.

    Uses only complete multimodal samples:
        Image + Radiograph + Text

    This stage does NOT handle missing modalities.
    Missing-modality robustness will be evaluated separately
    in Milestone 8.

    Labels:
        Same six labels used in previous experiments.
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

        self.df = pd.read_csv(csv_path)

        required = [
            "split",
            "photographs",
            "radiographs",
            "checkup_id",
            "patient_id",
            *self.TEXT_COLUMNS,
            "label_gingivitis",
            "label_class_ii_malocclusion",
            "label_class_iii_malocclusion",
            "label_class_i_malocclusion",
            "label_dental_caries",
            "label_pulpitis",
            "label_tooth_loss",
            "label_tooth_structure_loss",
        ]

        missing = [
            c for c in required
            if c not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # patient-level split
        self.df = (
            self.df[
                self.df["split"] == split
            ]
            .reset_index(drop=True)
        )

        self.preprocessor = FusionPreprocessor()

        # --------------------------------------------------
        # Fusion requires all modalities
        # --------------------------------------------------

        self.df = self.df[
            self.has_images()
            &
            self.has_radiographs()
            &
            self.has_text()
        ].reset_index(drop=True)

        print(
            f"{split}: {len(self.df)} fusion samples"
        )


    # ======================================================
    # Modality checks
    # ======================================================

    def has_images(self):

        return (
            self.df["photographs"]
            .notna()
            &
            (
                self.df["photographs"]
                .astype(str)
                .str.strip()
                != ""
            )
        )


    def has_radiographs(self):

        return (
            self.df["radiographs"]
            .notna()
            &
            (
                self.df["radiographs"]
                .astype(str)
                .str.strip()
                != ""
            )
        )


    def has_text(self):

        def check(row):

            return bool(
                self.build_text(row)
            )

        return self.df.apply(
            check,
            axis=1,
        )


    # ======================================================
    # Helpers
    # ======================================================

    @staticmethod
    def parse_paths(value):

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

        for col in cls.TEXT_COLUMNS:

            value = row[col]

            if pd.notna(value):

                value = str(value).strip()

                if value:
                    parts.append(value)

        return " ".join(parts)

    def load_fusion_labels(self, row):

        malocclusion = max(
            row["label_class_i_malocclusion"],
            row["label_class_ii_malocclusion"],
            row["label_class_iii_malocclusion"],
        )

        labels = [
            row["label_dental_caries"],
            row["label_gingivitis"],
            malocclusion,
            row["label_pulpitis"],
            row["label_tooth_loss"],
            row["label_tooth_structure_loss"],
        ]

        return torch.tensor(
            labels,
            dtype=torch.float32,
        )

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


    # ======================================================
    # Dataset API
    # ======================================================

    def __len__(self):

        return len(self.df)


    def __getitem__(
        self,
        idx,
    ):

        row = self.df.iloc[idx]


        image_paths = self.parse_paths(
            row["photographs"]
        )

        radiograph_paths = self.parse_paths(
            row["radiographs"]
        )

        text = self.build_text(row)


        image_tensor = (
            self.preprocessor
            .process_images(
                image_paths,
                modality="image",
            )
        )

        radiograph_tensor = (
            self.preprocessor
            .process_images(
                radiograph_paths,
                modality="radiograph",
            )
        )


        text_tokens = (
            self.preprocessor
            .process_text(text)
        )


        return {

            "checkup_id":
                str(row["checkup_id"]),


            "patient_id":
                str(row["patient_id"]),


            "images":
                image_tensor,


            "radiographs":
                radiograph_tensor,


            "input_ids":
                text_tokens["input_ids"],


            "attention_mask":
                text_tokens["attention_mask"],


            "labels":
                self.load_fusion_labels(row),
        }