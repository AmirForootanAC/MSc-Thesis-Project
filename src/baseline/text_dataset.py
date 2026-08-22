from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from src.baseline import config


class COdeTextDataset(Dataset):
    """
    Text-only dataset for COde six-label classification.

    Input:
        Clinical history + examination

    Excludes:
        diagnosis
        treatment_plan
        management
        radiographic findings
        demographics
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
        tokenizer_name="distilbert-base-uncased",
        max_length=256,
    ):

        self.csv_path = Path(csv_path)

        self.split = split

        self.max_length = max_length

        self.df = pd.read_csv(
            self.csv_path
        )

        self.df = self.df[
            self.df["split"] == split
        ].reset_index(drop=True)


        missing = [
            c for c in self.TEXT_COLUMNS
            if c not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing text columns: {missing}"
            )


        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )


        print(
            f"{split}: {len(self.df)} samples"
        )


    def __len__(self):
        return len(self.df)


    def build_text(self, row):

        parts = []

        for col in self.TEXT_COLUMNS:

            value = row[col]

            if pd.notna(value):

                value = str(value).strip()

                if value:
                    parts.append(value)


        return " ".join(parts)



    def __getitem__(self, idx):

        row = self.df.iloc[idx]


        text = self.build_text(
            row
        )


        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


        labels = (
            row[config.LABEL_NAMES]
            .astype(float)
            .values
        )


        return {

            "input_ids":
                encoded["input_ids"].squeeze(0),

            "attention_mask":
                encoded["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.float32,
                ),

            "text":
                text,
        }