"""Authoritative complete-case dataset shared by every scenario."""
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from . import config
from src.baseline.image_loader import COdeImageLoader
from src.baseline.transforms import get_image_transform

class CompleteCaseDataset(Dataset):
    def __init__(self, csv_path, split, image_root, modalities, tokenizer_name=None, max_length=None, transform=None):
        self.modalities = tuple(modalities)
        self.max_length = max_length or config.TEXT_MAX_LENGTH
        df = pd.read_csv(Path(csv_path))
        required = ["split", "photographs", "radiographs", *config.LABEL_NAMES, *config.TEXT_COLUMNS]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = df[df["split"] == split].copy()
        df = df[
            df["photographs"].apply(self.has_value)
            & df["radiographs"].apply(self.has_value)
            & df[config.TEXT_COLUMNS].notna().any(axis=1)
        ].reset_index(drop=True)
        expected = config.EXPECTED_COUNTS.get(split)
        if expected is not None and len(df) != expected:
            raise RuntimeError(f"Complete-case mismatch for {split}: expected {expected}, found {len(df)}")
        self.df = df
        self.image_loader = None
        if "photograph" in self.modalities or "radiograph" in self.modalities:
            self.image_loader = COdeImageLoader(Path(image_root), transform or get_image_transform())
        self.tokenizer = None
        if "text" in self.modalities:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or config.TEXT_MODEL_NAME)
        print(f"{split}: {len(self.df)} complete-case samples | {self.modalities}")

    @staticmethod
    def has_value(x):
        return pd.notna(x) and str(x).strip() != ""

    @staticmethod
    def parse_images(value):
        if pd.isna(value):
            return []
        return [x.strip() for x in str(value).split(",") if x.strip()]

    def build_text(self, row):
        return " ".join(str(row[c]).strip() for c in config.TEXT_COLUMNS if pd.notna(row[c]) and str(row[c]).strip())

    def load_images(self, value, modality):
        return [self.image_loader.load(f, modality) for f in self.parse_images(value)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {
            "checkup_id": row["checkup_id"],
            "patient_id": row["patient_id"],
            "labels": torch.tensor(row[config.LABEL_NAMES].astype(float).values, dtype=torch.float32),
        }
        if "photograph" in self.modalities:
            sample["images"] = self.load_images(row["photographs"], "photograph")
        if "radiograph" in self.modalities:
            sample["radiographs"] = self.load_images(row["radiographs"], "radiograph")
        if "text" in self.modalities:
            encoded = self.tokenizer(self.build_text(row), padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
            sample["input_ids"] = encoded["input_ids"].squeeze(0)
            sample["attention_mask"] = encoded["attention_mask"].squeeze(0)
        return sample
