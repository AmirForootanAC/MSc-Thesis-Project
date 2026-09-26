"""Authoritative complete-case dataset for all strong V2 scenarios."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.baseline.image_loader import COdeImageLoader

from . import config


class CompleteCaseDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split,
        image_root,
        modalities,
        tokenizer_name=None,
        max_length=None,
        transform=None,
        photograph_transform=None,
        radiograph_transform=None,
    ):
        self.modalities = tuple(modalities)

        self.max_length = (
            max_length
            or config.TEXT_MAX_LENGTH
        )

        df = pd.read_csv(
            Path(csv_path)
        )

        required = [
            "split",
            "photographs",
            "radiographs",
            *config.LABEL_NAMES,
            *config.TEXT_COLUMNS,
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        df = df[
            df["split"] == split
        ].copy()

        # IMPORTANT:
        # Every scenario uses exactly the same
        # complete-case cohort.
        df = df[
            df["photographs"].apply(
                self.has_value
            )
            & df["radiographs"].apply(
                self.has_value
            )
            & df[
                config.TEXT_COLUMNS
            ].notna().any(axis=1)
        ].reset_index(
            drop=True
        )

        expected = config.EXPECTED_COUNTS.get(
            split
        )

        if (
            expected is not None
            and len(df) != expected
        ):
            raise RuntimeError(
                f"Complete-case mismatch for "
                f"{split}: expected {expected}, "
                f"found {len(df)}"
            )

        self.df = df

        self.photograph_loader = None
        self.radiograph_loader = None

        if "photograph" in self.modalities:
            self.photograph_loader = (
                COdeImageLoader(
                    Path(image_root),
                    photograph_transform
                    or transform,
                )
            )

        if "radiograph" in self.modalities:
            self.radiograph_loader = (
                COdeImageLoader(
                    Path(image_root),
                    radiograph_transform
                    or transform,
                )
            )

        self.tokenizer = None

        if "text" in self.modalities:
            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    tokenizer_name
                    or config.TEXT_MODEL_NAME
                )
            )

        print(
            f"{split}: {len(self.df)} "
            f"complete-case samples | "
            f"{self.modalities}"
        )

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
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        ]

    def build_text(self, row):
        parts = []

        for column in config.TEXT_COLUMNS:
            if (
                pd.notna(row[column])
                and str(row[column]).strip()
            ):
                parts.append(
                    str(row[column]).strip()
                )

        return " ".join(parts)

    def load_images(
        self,
        value,
        modality,
    ):
        if modality == "photograph":
            loader = self.photograph_loader
        elif modality == "radiograph":
            loader = self.radiograph_loader
        else:
            raise ValueError(
                f"Unknown modality: {modality}"
            )

        return [
            loader.load(
                filename,
                modality,
            )
            for filename in self.parse_images(
                value
            )
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sample = {
            "checkup_id": row["checkup_id"],
            "patient_id": row["patient_id"],
            "labels": torch.tensor(
                row[
                    config.LABEL_NAMES
                ].astype(float).values,
                dtype=torch.float32,
            ),
        }

        if "photograph" in self.modalities:
            sample["images"] = self.load_images(
                row["photographs"],
                "photograph",
            )

        if "radiograph" in self.modalities:
            sample["radiographs"] = self.load_images(
                row["radiographs"],
                "radiograph",
            )

        if "text" in self.modalities:
            encoded = self.tokenizer(
                self.build_text(row),
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            sample["input_ids"] = (
                encoded["input_ids"]
                .squeeze(0)
            )

            sample["attention_mask"] = (
                encoded["attention_mask"]
                .squeeze(0)
            )

        return sample