from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.baseline import config
from src.baseline.image_loader import COdeImageLoader
from src.baseline.transforms import get_image_transform


class COdeMultimodalDataset(Dataset):
    """
    Full multimodal dataset for COde six-label baseline.

    Modalities:
        - photographs
        - radiographs
        - clinical text

    Uses complete-case visits only:
        photograph + radiograph + clinical text

    Text:
        clinical history + examination

    Excludes:
        anomalies_en
        diagnosis
        treatment_plan
        management
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
        image_root=None,
        transform=None,
        tokenizer_name="distilbert-base-uncased",
        max_length=256,
    ):

        self.csv_path = Path(csv_path)

        self.split = split

        self.max_length = max_length


        self.df = pd.read_csv(
            self.csv_path
        )


        # -----------------------------
        # Validate labels
        # -----------------------------

        missing_labels = [
            c
            for c in config.LABEL_NAMES
            if c not in self.df.columns
        ]

        if missing_labels:
            raise ValueError(
                f"Missing labels: {missing_labels}"
            )


        # -----------------------------
        # Split
        # -----------------------------

        self.df = self.df[
            self.df["split"] == split
        ].reset_index(drop=True)



        # -----------------------------
        # Complete multimodal filtering
        # -----------------------------

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



        # -----------------------------
        # Image transform
        # -----------------------------

        if transform is None:
            transform = get_image_transform()


        self.image_loader = COdeImageLoader(
            image_root=Path(image_root),
            transform=transform,
        )



        # -----------------------------
        # Tokenizer
        # -----------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )



        print(
            f"{split}: {len(self.df)} multimodal samples"
        )



    def __len__(self):

        return len(self.df)



    @staticmethod
    def has_value(x):

        return (
            pd.notna(x)
            and str(x).strip() != ""
        )



    def parse_images(
        self,
        value,
    ):

        if pd.isna(value):

            return []

        return [
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        ]



    def load_images(
        self,
        filenames,
        modality,
    ):

        images = []

        for filename in filenames:

            image = self.image_loader.load(
                filename=filename,
                modality=modality,
            )

            images.append(image)


        return images



    def build_text(
        self,
        row,
    ):

        parts = []

        for col in self.TEXT_COLUMNS:

            value = row[col]

            if pd.notna(value):

                value = str(value).strip()

                if value:

                    parts.append(value)


        return " ".join(parts)



    def encode_text(
        self,
        text,
    ):

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


        return {
            "input_ids":
                encoded["input_ids"].squeeze(0),

            "attention_mask":
                encoded["attention_mask"].squeeze(0),
        }



    def load_labels(
        self,
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



    def __getitem__(
        self,
        idx,
    ):

        row = self.df.iloc[idx]


        text = self.build_text(
            row
        )


        text_tokens = self.encode_text(
            text
        )


        return {

            "checkup_id":
                row["checkup_id"],


            "patient_id":
                row["patient_id"],


            "images":
                self.load_images(
                    self.parse_images(
                        row["photographs"]
                    ),
                    modality="photograph",
                ),


            "radiographs":
                self.load_images(
                    self.parse_images(
                        row["radiographs"]
                    ),
                    modality="radiograph",
                ),


            "input_ids":
                text_tokens["input_ids"],


            "attention_mask":
                text_tokens["attention_mask"],


            "labels":
                self.load_labels(
                    row
                ),
        }