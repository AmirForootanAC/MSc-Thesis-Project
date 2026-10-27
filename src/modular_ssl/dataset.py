"""
Datasets for Decoupled Single-Modality SSL and Precomputed Multi-View Fusion.
"""

from pathlib import Path
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.baseline.image_loader import COdeImageLoader
from src.ssl_v2.transforms import (
    get_ssl_train_transform,
    get_eval_transform,
)
from . import config


def has_value(val) -> bool:
    return pd.notna(val) and str(val).strip() != ""


def parse_filenames(val) -> list[str]:
    if pd.isna(val):
        return []
    return [x.strip() for x in str(val).split(",") if x.strip()]


def build_text(row) -> str:
    parts = []
    for col in config.TEXT_COLUMNS:
        if col in row and pd.notna(row[col]):
            s = str(row[col]).strip()
            if s:
                parts.append(s)
    return " ".join(parts)


class SingleModalitySSLDataset(Dataset):
    """
    Loads ONLY visits that possess the target visual modality ('photograph' or 'radiograph').
    Eliminates unnecessary disk I/O for the inactive visual modality!
    """

    def __init__(
        self,
        modality: str = "photograph",
        split: str = "train",
        is_train: bool = True,
        csv_path=None,
        image_root=None,
    ):
        if modality not in ("photograph", "radiograph"):
            raise ValueError(f"Unknown modality: {modality}")

        self.modality = modality
        self.col_name = "photographs" if modality == "photograph" else "radiographs"
        self.split = split
        self.is_train = is_train

        df = pd.read_csv(Path(csv_path or config.DATASET_PATH))
        df = df[df["split"] == split].copy()
        df = df[df[self.col_name].apply(has_value)].reset_index(drop=True)
        self.df = df

        self.max_views = (
            (
                config.MAX_PHOTO_VIEWS_TRAIN
                if modality == "photograph"
                else config.MAX_RADIO_VIEWS_TRAIN
            )
            if is_train
            else (
                config.MAX_PHOTO_VIEWS_EXTRACT
                if modality == "photograph"
                else config.MAX_RADIO_VIEWS_EXTRACT
            )
        )

        # Enable CLAHE during radiograph evaluation/extraction or radiograph SSL
        use_clahe = modality == "radiograph"
        tf = (
            get_ssl_train_transform(config.IMAGE_SIZE, modality, use_clahe=use_clahe)
            if is_train
            else get_eval_transform(config.IMAGE_SIZE, modality, use_clahe=use_clahe)
        )

        self.loader = COdeImageLoader(Path(image_root or config.IMAGE_ROOT), tf)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

        print(
            f"[Modular SSL Dataset | {modality.upper()}] "
            f"{split}: {len(self.df):,} valid visits loaded."
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        fnames = parse_filenames(row[self.col_name])

        if len(fnames) > self.max_views:
            selected = (
                random.sample(fnames, self.max_views)
                if self.is_train
                else fnames[: self.max_views]
            )
        else:
            selected = fnames

        views = []
        for fn in selected:
            try:
                views.append(self.loader.load(fn, self.modality))
            except Exception:
                continue

        if not views:
            views = [
                torch.zeros(
                    (3, config.IMAGE_SIZE, config.IMAGE_SIZE),
                    dtype=torch.float32,
                )
            ]

        text = build_text(row)
        encoded = self.tokenizer(
            text if text else "none",
            padding="max_length",
            truncation=True,
            max_length=config.TEXT_MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "checkup_id": str(row["checkup_id"]),
            "views": views,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "has_text": bool(text),
        }


def single_modality_ssl_collate(batch: list[dict]) -> dict:
    B = len(batch)
    max_v = max(len(item["views"]) for item in batch)
    img_sz = config.IMAGE_SIZE

    padded = torch.zeros((B, max_v, 3, img_sz, img_sz), dtype=torch.float32)
    mask = torch.zeros((B, max_v), dtype=torch.bool)

    for i, item in enumerate(batch):
        for j, v in enumerate(item["views"]):
            padded[i, j] = v
            mask[i, j] = True

    return {
        "checkup_id": [item["checkup_id"] for item in batch],
        "padded_images": padded,
        "view_mask": mask,
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "attention_mask": torch.stack(
            [item["attention_mask"] for item in batch], dim=0
        ),
        "has_text": torch.tensor(
            [item["has_text"] for item in batch], dtype=torch.bool
        ),
    }


class PrecomputedMultiViewDataset(Dataset):
    """
    Loads pre-extracted multi-view SSL representations from .pt files
    for instant (<2 minutes) 4-scenario Fusion training and evaluation.
    """

    def __init__(self, split: str, rep_dir: Path | None = None):
        rep_dir = Path(rep_dir or config.REPRESENTATION_DIR)
        path = rep_dir / f"{split}_multiview_rep.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Precomputed representation file not found: {path}\n"
                "Run `python -m src.modular_ssl.extract_representations` first."
            )

        data = torch.load(path, map_location="cpu", weights_only=False)
        self.checkup_ids = data["checkup_id"]
        self.patient_ids = data["patient_id"]
        self.photo_visit = data["photo_visit"].float()
        self.photo_views = data["photo_views"].float()
        self.photo_mask = data["photo_mask"].bool()
        self.radio_visit = data["radio_visit"].float()
        self.radio_views = data["radio_views"].float()
        self.radio_mask = data["radio_mask"].bool()
        self.text_visit = data["text_visit"].float()
        self.text_raw = data["text_raw"].float()
        self.labels = data["labels"].float()

        expected = config.EXPECTED_COMPLETE_CASE_COUNTS.get(split)
        if expected is not None and len(self.checkup_ids) != expected:
            raise RuntimeError(
                f"Representation count mismatch for {split}: "
                f"expected {expected}, got {len(self.checkup_ids)}"
            )

    def __len__(self) -> int:
        return len(self.checkup_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "photo_visit": self.photo_visit[idx],
            "photo_views": self.photo_views[idx],
            "photo_mask": self.photo_mask[idx],
            "radio_visit": self.radio_visit[idx],
            "radio_views": self.radio_views[idx],
            "radio_mask": self.radio_mask[idx],
            "text_visit": self.text_visit[idx],
            "text_raw": self.text_raw[idx],
            "labels": self.labels[idx],
        }