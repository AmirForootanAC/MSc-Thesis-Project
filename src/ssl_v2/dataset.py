"""
Multi-View Datasets and Collate Functions for SSL v2 & 4-Scenario Fusion.

Zero pixel-space or dataset-level averaging:
Each visit returns its list of individual view tensors, which are collated
into a padded 5D tensor [B, V_max, 3, H, W] + boolean view_mask [B, V_max]
for seamless multi-GPU execution across 2x T4 GPUs.
"""

from pathlib import Path
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.baseline.image_loader import COdeImageLoader
from . import config
from .transforms import (
    get_ssl_train_transform,
    get_downstream_train_transform,
    get_eval_transform,
)


def _has_value(value) -> bool:
    return pd.notna(value) and str(value).strip() != ""


def _parse_filenames(value) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _build_clinical_text(row) -> str:
    parts = []
    for col in config.TEXT_COLUMNS:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val:
                parts.append(val)
    return " ".join(parts)


# ============================================================
# 1. Dynamic Multi-View SSL Pretraining Dataset (All Visits)
# ============================================================

class MultiViewSSLDataset(Dataset):
    """
    Loads all visits in a split (train=6,129 | val=1,330 | test=1,316)
    without complete-case filtering. Preserves all views up to max_views.
    """

    def __init__(
        self,
        csv_path=None,
        split: str = "train",
        image_root=None,
        is_train: bool = True,
    ):
        self.csv_path = Path(csv_path or config.DATASET_PATH)
        self.split = split
        self.image_root = Path(image_root or config.IMAGE_ROOT)
        self.is_train = is_train

        df = pd.read_csv(self.csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        self.max_photo_views = (
            config.MAX_PHOTO_VIEWS_TRAIN
            if is_train
            else config.MAX_PHOTO_VIEWS_EVAL
        )
        self.max_radio_views = (
            config.MAX_RADIO_VIEWS_TRAIN
            if is_train
            else config.MAX_RADIO_VIEWS_EVAL
        )

        photo_tf = (
            get_ssl_train_transform(config.IMAGE_SIZE, "photograph", False)
            if is_train
            else get_eval_transform(config.IMAGE_SIZE, "photograph", False)
        )
        radio_tf = (
            get_ssl_train_transform(
                config.IMAGE_SIZE, "radiograph", config.USE_CLAHE_FOR_XRAY
            )
            if is_train
            else get_eval_transform(
                config.IMAGE_SIZE, "radiograph", config.USE_CLAHE_FOR_XRAY
            )
        )

        self.photo_loader = COdeImageLoader(self.image_root, photo_tf)
        self.radio_loader = COdeImageLoader(self.image_root, radio_tf)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

        print(f"[SSL v2 Dataset] {split}: {len(self.df):,} visits loaded.")

    def __len__(self) -> int:
        return len(self.df)

    def _load_view_list(
        self,
        filenames: list[str],
        modality: str,
        loader: COdeImageLoader,
        max_views: int,
    ) -> list[torch.Tensor]:
        if not filenames:
            return []

        selected = list(filenames)
        if len(selected) > max_views:
            if self.is_train:
                selected = random.sample(selected, max_views)
            else:
                selected = selected[:max_views]

        tensors = []
        for fname in selected:
            try:
                tensors.append(loader.load(fname, modality))
            except Exception:
                continue
        return tensors

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        photo_files = _parse_filenames(row["photographs"])
        radio_files = _parse_filenames(row["radiographs"])
        text = _build_clinical_text(row)

        photo_views = self._load_view_list(
            photo_files, "photograph", self.photo_loader, self.max_photo_views
        )
        radio_views = self._load_view_list(
            radio_files, "radiograph", self.radio_loader, self.max_radio_views
        )

        encoded = self.tokenizer(
            text if text else "none",
            padding="max_length",
            truncation=True,
            max_length=config.TEXT_MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "checkup_id": str(row["checkup_id"]),
            "patient_id": str(row["patient_id"]),
            "photo_views": photo_views,
            "radio_views": radio_views,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "has_photo": len(photo_views) > 0,
            "has_radio": len(radio_views) > 0,
            "has_text": bool(text),
        }


# ============================================================
# 2. Complete-Case Downstream Dataset (4 Scenarios, 4 Labels)
# ============================================================

class MultiViewDownstreamDataset(Dataset):
    """
    Authoritative complete-case dataset (train=2,935 | val=627 | test=633)
    supporting all 4 scenarios:
        - image_only      ("photograph",)
        - xray_only       ("radiograph",)
        - text_only       ("text",)
        - full_multimodal ("photograph", "radiograph", "text")
    """

    def __init__(
        self,
        csv_path=None,
        split: str = "train",
        image_root=None,
        modalities: tuple = ("photograph", "radiograph", "text"),
        is_train: bool = True,
    ):
        self.csv_path = Path(csv_path or config.DATASET_PATH)
        self.split = split
        self.image_root = Path(image_root or config.IMAGE_ROOT)
        self.modalities = tuple(modalities)
        self.is_train = is_train

        df = pd.read_csv(self.csv_path)

        required = [
            "split",
            "checkup_id",
            "patient_id",
            "photographs",
            "radiographs",
            *config.TEXT_COLUMNS,
            *config.LABEL_NAMES,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")

        df = df[df["split"] == split].copy()
        df = df[
            df["photographs"].apply(_has_value)
            & df["radiographs"].apply(_has_value)
            & df[config.TEXT_COLUMNS].notna().any(axis=1)
        ].reset_index(drop=True)

        expected = config.EXPECTED_COMPLETE_CASE_COUNTS.get(split)
        if expected is not None and len(df) != expected:
            raise RuntimeError(
                f"Complete-case count mismatch on {split}: "
                f"expected {expected}, got {len(df)}"
            )

        self.df = df

        self.max_photo_views = (
            config.MAX_PHOTO_VIEWS_TRAIN
            if is_train
            else config.MAX_PHOTO_VIEWS_EVAL
        )
        self.max_radio_views = (
            config.MAX_RADIO_VIEWS_TRAIN
            if is_train
            else config.MAX_RADIO_VIEWS_EVAL
        )

        self.photo_loader = None
        self.radio_loader = None
        self.tokenizer = None

        if "photograph" in self.modalities:
            p_tf = (
                get_downstream_train_transform(
                    config.IMAGE_SIZE, "photograph", False
                )
                if is_train
                else get_eval_transform(config.IMAGE_SIZE, "photograph", False)
            )
            self.photo_loader = COdeImageLoader(self.image_root, p_tf)

        if "radiograph" in self.modalities:
            r_tf = (
                get_downstream_train_transform(
                    config.IMAGE_SIZE, "radiograph", config.USE_CLAHE_FOR_XRAY
                )
                if is_train
                else get_eval_transform(
                    config.IMAGE_SIZE, "radiograph", config.USE_CLAHE_FOR_XRAY
                )
            )
            self.radio_loader = COdeImageLoader(self.image_root, r_tf)

        if "text" in self.modalities:
            self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

        print(
            f"[Downstream v2] {split}: {len(self.df):,} complete-case visits | "
            f"Modalities: {self.modalities}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_views(
        self,
        filenames: list[str],
        modality: str,
        loader: COdeImageLoader,
        max_views: int,
    ) -> list[torch.Tensor]:
        selected = list(filenames)
        if len(selected) > max_views:
            if self.is_train:
                selected = random.sample(selected, max_views)
            else:
                selected = selected[:max_views]

        views = []
        for fname in selected:
            views.append(loader.load(fname, modality))
        if not views:
            raise RuntimeError(f"Zero {modality} views loaded in complete-case visit.")
        return views

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        item = {
            "checkup_id": str(row["checkup_id"]),
            "patient_id": str(row["patient_id"]),
            "labels": torch.tensor(
                row[config.LABEL_NAMES].astype(float).values,
                dtype=torch.float32,
            ),
        }

        if "photograph" in self.modalities:
            item["photo_views"] = self._load_views(
                _parse_filenames(row["photographs"]),
                "photograph",
                self.photo_loader,
                self.max_photo_views,
            )

        if "radiograph" in self.modalities:
            item["radio_views"] = self._load_views(
                _parse_filenames(row["radiographs"]),
                "radiograph",
                self.radio_loader,
                self.max_radio_views,
            )

        if "text" in self.modalities:
            encoded = self.tokenizer(
                _build_clinical_text(row),
                padding="max_length",
                truncation=True,
                max_length=config.TEXT_MAX_LENGTH,
                return_tensors="pt",
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)

        return item


# ============================================================
# 3. Dynamic 5D Multi-View Collate Function (DataParallel Ready)
# ============================================================

def _pack_view_lists(
    view_lists: list[list[torch.Tensor]],
    image_size: int = config.IMAGE_SIZE,
):
    """
    Packs a batch of variable-length view lists into:
        padded_tensor : FloatTensor [B, V_max, 3, H, W]
        view_mask     : BoolTensor  [B, V_max]
    """
    B = len(view_lists)
    max_v = max((len(v) for v in view_lists), default=1)
    max_v = max(max_v, 1)

    padded = torch.zeros((B, max_v, 3, image_size, image_size), dtype=torch.float32)
    mask = torch.zeros((B, max_v), dtype=torch.bool)

    for i, views in enumerate(view_lists):
        for j, img_tensor in enumerate(views):
            padded[i, j] = img_tensor
            mask[i, j] = True

    return padded, mask


def pad_and_collate_multiview(batch: list[dict]) -> dict:
    """
    Universal collate function for both MultiViewSSLDataset and
    MultiViewDownstreamDataset.
    """
    out = {
        "checkup_id": [b["checkup_id"] for b in batch],
        "patient_id": [b["patient_id"] for b in batch],
    }

    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch], dim=0)

    if "photo_views" in batch[0]:
        photo_images, photo_mask = _pack_view_lists(
            [b["photo_views"] for b in batch]
        )
        out["photo_images"] = photo_images
        out["photo_view_mask"] = photo_mask

    if "radio_views" in batch[0]:
        radio_images, radio_mask = _pack_view_lists(
            [b["radio_views"] for b in batch]
        )
        out["radio_images"] = radio_images
        out["radio_view_mask"] = radio_mask

    if "input_ids" in batch[0]:
        out["input_ids"] = torch.stack([b["input_ids"] for b in batch], dim=0)
        out["attention_mask"] = torch.stack(
            [b["attention_mask"] for b in batch], dim=0
        )

    if "has_photo" in batch[0]:
        out["has_photo"] = torch.tensor(
            [b["has_photo"] for b in batch], dtype=torch.bool
        )
        out["has_radio"] = torch.tensor(
            [b["has_radio"] for b in batch], dtype=torch.bool
        )
        out["has_text"] = torch.tensor(
            [b["has_text"] for b in batch], dtype=torch.bool
        )

    return out