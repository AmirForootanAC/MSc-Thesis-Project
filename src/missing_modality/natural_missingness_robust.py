"""
Milestone 8.5.2 — Natural Missingness Robust Fusion.

Purpose
-------
Evaluate robust multimodal fusion under naturally occurring
missing-modality patterns in the COde dataset.

This milestone is intentionally different from Milestone 8.3.2.

Milestone 8.3.2
---------------
Controlled modality dropout applied to complete-case SSL
representations during training.

Milestone 8.5.2
---------------
Naturally occurring missing modalities are preserved from the
original six-label patient-level dataset.

Pipeline
--------
1. Load the authoritative six-label patient-level dataset.
2. Load the frozen SSL checkpoint from Milestone 6.
3. Extract modality-specific SSL representations for the entire
   natural-missingness population.
4. Represent missing modalities with zero vectors.
5. Store an explicit modality-presence mask.
6. Train RobustFusion using the natural training population.
7. Select the best checkpoint using validation Macro F1.
8. Evaluate the untouched test population.
9. Report performance:
       - overall
       - by natural missingness pattern
       - by number of missing modalities
       - by split

Important
---------
- Dataset:
      results/six_label_patient_level_dataset/labeled_dataset.csv

- SSL checkpoint:
      results/ssl_pretraining/multimodal_dynamic/best_ssl_model.pt

- Patient-level split is authoritative.

- SSL encoders remain frozen.

- No test samples are used for model selection.

- Missing modalities are NOT synthetically generated.

- A missing modality is represented by a zero vector and an
  explicit availability mask.

- Complete cases are retained as the "complete" natural pattern.

- All-missing samples are excluded from model training/evaluation
  because there is no usable modality representation. They are
  still reported in the population audit.

Output
------
results/milestone8_missing_modality/
    08_natural_missingness_robust/
        01_representations/
            train.pt
            validation.pt
            test.pt

        02_training/
            best_model.pt
            history.json
            config.json

        03_test/
            overall_results.csv
            pattern_results.csv
            missing_count_results.csv
            test_results.json
            config.json

        04_summary/
            protocol_summary.json
"""


from pathlib import Path

import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import argparse

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import AutoTokenizer

from src.baseline import config
from src.baseline.metrics import compute_metrics
from src.baseline.image_loader import COdeImageLoader
from src.baseline.transforms import get_image_transform

from src.fusion.representation_extractor import (
    load_ssl_model,
    extract_image_representations,
    extract_text_representations,
)

from src.missing_modality.robust_fusion_model import (
    RobustFusion,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

DATASET_PATH = (
    PROJECT_ROOT
    / "results"
    / "six_label_patient_level_dataset"
    / "labeled_dataset.csv"
)


# ------------------------------------------------------------
# SSL checkpoint
# ------------------------------------------------------------

SSL_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "ssl_pretraining"
    / "multimodal_dynamic"
    / "best_ssl_model.pt"
)


# ------------------------------------------------------------
# Image root
# ------------------------------------------------------------

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "COde-Dataset"
    / "Images"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "milestone8_missing_modality"
    / "08_natural_missingness_robust"
)


REPRESENTATION_ROOT = (
    RESULT_ROOT
    / "01_representations"
)

TRAINING_ROOT = (
    RESULT_ROOT
    / "02_training"
)

TEST_ROOT = (
    RESULT_ROOT
    / "03_test"
)

SUMMARY_ROOT = (
    RESULT_ROOT
    / "04_summary"
)


# ------------------------------------------------------------
# Text
# ------------------------------------------------------------

TEXT_COLUMNS = [
    "chief_complaint",
    "present_illness",
    "past_medical_record",
    "examination",
]

TEXT_MODEL_NAME = (
    "distilbert-base-uncased"
)

TEXT_MAX_LENGTH = 256


# ------------------------------------------------------------
# Representation dimensions
# ------------------------------------------------------------

IMAGE_DIM = 2048
RADIOGRAPH_DIM = 2048
TEXT_DIM = 768


# ------------------------------------------------------------
# Fusion model
# ------------------------------------------------------------

MODALITY_DIM = 512
HIDDEN_DIM = 512
DROPOUT = 0.3


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

BATCH_SIZE = 32
EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


# ------------------------------------------------------------
# Representation extraction
# ------------------------------------------------------------

EXTRACTION_BATCH_SIZE = 2


EXPECTED_SPLITS = [
    "train",
    "validation",
    "test",
]


PATTERN_ORDER = [
    "complete",
    "image_missing",
    "radiograph_missing",
    "text_missing",
    "image_radiograph_missing",
    "image_text_missing",
    "radiograph_text_missing",
    "all_missing",
]


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# Modality utilities
# ============================================================

def has_value(value):

    return (
        pd.notna(value)
        and str(value).strip() != ""
    )


def has_text(row):

    for column in TEXT_COLUMNS:

        if has_value(row[column]):

            return True

    return False


def parse_images(value):
    """
    Parse COde dataset image entries.
map_location="cpu"
    The dataset may store multiple image filenames using
    comma or semicolon separators. Normalize both formats
    into a clean list of filenames.
    """

    if not has_value(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    normalized = text.replace(";", ",")

    parts = [
        item.strip()
        for item in normalized.split(",")
        if item.strip()
    ]

    return parts


def build_text(row):

    pieces = []

    for column in TEXT_COLUMNS:

        value = row[column]

        if has_value(value):

            pieces.append(
                str(value).strip()
            )

    return " ".join(pieces)


def assign_pattern(
    image_available,
    radiograph_available,
    text_available,
):

    key = (
        int(image_available),
        int(radiograph_available),
        int(text_available),
    )

    patterns = {

        (1, 1, 1):
            "complete",

        (0, 1, 1):
            "image_missing",

        (1, 0, 1):
            "radiograph_missing",

        (1, 1, 0):
            "text_missing",

        (0, 0, 1):
            "image_radiograph_missing",

        (0, 1, 0):
            "image_text_missing",

        (1, 0, 0):
            "radiograph_text_missing",

        (0, 0, 0):
            "all_missing",
    }

    return patterns[key]


# ============================================================
# Natural missingness dataframe
# ============================================================

def load_dataset():

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    required_columns = [
        "checkup_id",
        "patient_id",
        "split",
        "photographs",
        "radiographs",
        *TEXT_COLUMNS,
        *config.LABEL_NAMES,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def prepare_dataframe(df):

    result = df.copy()

    result["image_available"] = (
        result["photographs"]
        .apply(has_value)
    )

    result["radiograph_available"] = (
        result["radiographs"]
        .apply(has_value)
    )

    result["text_available"] = (
        result.apply(
            has_text,
            axis=1,
        )
    )

    result["missing_modality_count"] = (
        3
        -
        result[
            [
                "image_available",
                "radiograph_available",
                "text_available",
            ]
        ].sum(axis=1)
    )

    result["pattern"] = [
        assign_pattern(
            image_available,
            radiograph_available,
            text_available,
        )
        for image_available,
        radiograph_available,
        text_available
        in zip(
            result["image_available"],
            result["radiograph_available"],
            result["text_available"],
        )
    ]

    return result


# ============================================================
# Dataset for representation extraction
# ============================================================

class NaturalRepresentationDataset(Dataset):
    """
    Raw natural-missingness dataset used to extract frozen SSL
    representations.

    Missing modalities are NOT imputed.

    Availability is preserved explicitly.
    """

    def __init__(
        self,
        dataframe,
        split,
    ):

        self.df = (
            dataframe[
                dataframe["split"] == split
            ]
            .reset_index(drop=True)
        )

        print(
            f"{split}: "
            f"{len(self.df)} natural-missingness samples"
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        return {
            "checkup_id":
                row["checkup_id"],

            "patient_id":
                row["patient_id"],

            "images":
                parse_images(
                    row["photographs"]
                ),

            "radiographs":
                parse_images(
                    row["radiographs"]
                ),

            "text":
                build_text(row),

            "image_available":
                float(
                    row["image_available"]
                ),

            "radiograph_available":
                float(
                    row["radiograph_available"]
                ),

            "text_available":
                float(
                    row["text_available"]
                ),

            "pattern":
                row["pattern"],

            "missing_modality_count":
                int(
                    row[
                        "missing_modality_count"
                    ]
                ),

            "labels":
                torch.tensor(
                    [
                        float(row[label])
                        for label in config.LABEL_NAMES
                    ],
                    dtype=torch.float32,
                ),
        }


def natural_collate(batch):

    return {

        "checkup_id": [
            item["checkup_id"]
            for item in batch
        ],

        "patient_id": [
            item["patient_id"]
            for item in batch
        ],

        "images": [
            item["images"]
            for item in batch
        ],

        "radiographs": [
            item["radiographs"]
            for item in batch
        ],

        "text": [
            item["text"]
            for item in batch
        ],

        "image_available":
            torch.tensor(
                [
                    item["image_available"]
                    for item in batch
                ],
                dtype=torch.float32,
            ),

        "radiograph_available":
            torch.tensor(
                [
                    item["radiograph_available"]
                    for item in batch
                ],
                dtype=torch.float32,
            ),

        "text_available":
            torch.tensor(
                [
                    item["text_available"]
                    for item in batch
                ],
                dtype=torch.float32,
            ),

        "pattern": [
            item["pattern"]
            for item in batch
        ],

        "missing_modality_count":
            torch.tensor(
                [
                    item[
                        "missing_modality_count"
                    ]
                    for item in batch
                ],
                dtype=torch.long,
            ),

        "labels":
            torch.stack(
                [
                    item["labels"]
                    for item in batch
                ],
                dim=0,
            ),
    }


# ============================================================
# Representation extraction helpers
# ============================================================

def extract_single_split(
    model,
    dataframe,
    split,
):

    dataset = NaturalRepresentationDataset(
        dataframe,
        split,
    )

    loader = DataLoader(
        dataset,
        batch_size=EXTRACTION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=natural_collate,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL_NAME
    )

    transform = get_image_transform()

    image_loader = COdeImageLoader(
        IMAGE_ROOT
    )

    image_features = []
    radiograph_features = []
    text_features = []
    labels = []

    checkup_ids = []
    patient_ids = []

    image_available_all = []
    radiograph_available_all = []
    text_available_all = []

    patterns = []
    missing_counts = []

    for batch_index, batch in enumerate(
        tqdm(
            loader,
            desc=f"Extract {split}",
        )
    ):

        batch_size = len(
            batch["checkup_id"]
        )

        # ----------------------------------------------------
        # Availability mask
        # ----------------------------------------------------

        modality_mask = torch.stack(
            [
                batch["image_available"],
                batch["radiograph_available"],
                batch["text_available"],
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image_repr = torch.zeros(
            batch_size,
            IMAGE_DIM,
            dtype=torch.float32,
        )

        image_indices = [
            i
            for i, files in enumerate(
                batch["images"]
            )
            if len(files) > 0
        ]

        if image_indices:

            selected_files = [
                batch["images"][i]
                for i in image_indices
            ]

            selected_repr = (
                extract_image_representations(
                    file_lists=selected_files,
                    encoder=model.encoders.image_encoder,
                    image_loader=image_loader,
                    transform=transform,
                    device=DEVICE,
                    modality="photograph",
                )
            )

            image_repr[
                image_indices
            ] = selected_repr.cpu()

        # ----------------------------------------------------
        # Radiograph
        # ----------------------------------------------------

        radiograph_repr = torch.zeros(
            batch_size,
            RADIOGRAPH_DIM,
            dtype=torch.float32,
        )

        radiograph_indices = [
            i
            for i, files in enumerate(
                batch["radiographs"]
            )
            if len(files) > 0
        ]

        if radiograph_indices:

            selected_files = [
                batch["radiographs"][i]
                for i in radiograph_indices
            ]

            selected_repr = (
                extract_image_representations(
                    file_lists=selected_files,
                    encoder=model.encoders.radiograph_encoder,
                    image_loader=image_loader,
                    transform=transform,
                    device=DEVICE,
                    modality="radiograph",
                )
            )

            radiograph_repr[
                radiograph_indices
            ] = selected_repr.cpu()

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        text_repr = torch.zeros(
            batch_size,
            TEXT_DIM,
            dtype=torch.float32,
        )

        text_indices = [
            i
            for i, text in enumerate(
                batch["text"]
            )
            if has_value(text)
        ]

        if text_indices:

            selected_texts = [
                batch["text"][i]
                for i in text_indices
            ]

            selected_repr = (
                extract_text_representations(
                    texts=selected_texts,
                    model=model,
                    tokenizer=tokenizer,
                    device=DEVICE,
                )
            )

            text_repr[
                text_indices
            ] = selected_repr.cpu()

        # ----------------------------------------------------
        # Append
        # ----------------------------------------------------

        image_features.append(
            image_repr
        )

        radiograph_features.append(
            radiograph_repr
        )

        text_features.append(
            text_repr
        )

        labels.append(
            batch["labels"]
        )

        checkup_ids.extend(
            batch["checkup_id"]
        )

        patient_ids.extend(
            batch["patient_id"]
        )

        image_available_all.append(
            batch["image_available"]
        )

        radiograph_available_all.append(
            batch["radiograph_available"]
        )

        text_available_all.append(
            batch["text_available"]
        )

        patterns.extend(
            batch["pattern"]
        )

        missing_counts.append(
            batch["missing_modality_count"]
        )

    # --------------------------------------------------------
    # Concatenate
    # --------------------------------------------------------

    image_features = torch.cat(
        image_features,
        dim=0,
    )

    radiograph_features = torch.cat(
        radiograph_features,
        dim=0,
    )

    text_features = torch.cat(
        text_features,
        dim=0,
    )

    labels = torch.cat(
        labels,
        dim=0,
    )

    image_available = torch.cat(
        image_available_all,
        dim=0,
    )

    radiograph_available = torch.cat(
        radiograph_available_all,
        dim=0,
    )

    text_available = torch.cat(
        text_available_all,
        dim=0,
    )

    missing_modality_count = torch.cat(
        missing_counts,
        dim=0,
    )

    modality_mask = torch.stack(
        [
            image_available,
            radiograph_available,
            text_available,
        ],
        dim=1,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    n = len(dataset)

    assert image_features.shape == (
        n,
        IMAGE_DIM,
    )

    assert radiograph_features.shape == (
        n,
        RADIOGRAPH_DIM,
    )

    assert text_features.shape == (
        n,
        TEXT_DIM,
    )

    assert labels.shape == (
        n,
        config.NUM_LABELS,
    )

    assert modality_mask.shape == (
        n,
        3,
    )

    assert len(checkup_ids) == n

    assert len(patient_ids) == n

    assert len(patterns) == n

    assert missing_modality_count.shape == (
        n,
    )

    # --------------------------------------------------------
    # Build artifact
    # --------------------------------------------------------

    representations = {

        "checkup_id":
            checkup_ids,

        "patient_id":
            patient_ids,

        "image":
            image_features,

        "radiograph":
            radiograph_features,

        "text":
            text_features,

        "labels":
            labels,

        "modality_mask":
            modality_mask,

        "pattern":
            patterns,

        "missing_modality_count":
            missing_modality_count,
    }

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    complete_count = sum(
        pattern == "complete"
        for pattern in patterns
    )

    all_missing_count = sum(
        pattern == "all_missing"
        for pattern in patterns
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"{split.upper()} NATURAL REPRESENTATION CHECK"
    )

    print(
        "=" * 80
    )

    print(
        "Samples:",
        n,
    )

    print(
        "Complete:",
        complete_count,
    )

    print(
        "All missing:",
        all_missing_count,
    )

    print(
        "Image:",
        tuple(image_features.shape),
    )

    print(
        "Radiograph:",
        tuple(radiograph_features.shape),
    )

    print(
        "Text:",
        tuple(text_features.shape),
    )

    print(
        "Mask:",
        tuple(modality_mask.shape),
    )

    print(
        "Labels:",
        tuple(labels.shape),
    )

    print(
        "PASS"
    )

    print(
        "=" * 80
    )

    return representations


def save_representation(
    split,
    representations,
):

    REPRESENTATION_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPRESENTATION_ROOT
        / f"{split}.pt"
    )

    torch.save(
        representations,
        output_path,
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Representation extraction pipeline
# ============================================================

def extract_all_representations(
    dataframe,
):

    if not SSL_CHECKPOINT.exists():

        raise FileNotFoundError(
            "SSL checkpoint not found:\n"
            f"{SSL_CHECKPOINT}"
        )

    print()
    print("=" * 100)
    print(
        "MILESTONE 8.5.2 — SSL REPRESENTATION EXTRACTION"
    )
    print("=" * 100)

    print(
        "Dataset:",
        DATASET_PATH,
    )

    print(
        "SSL checkpoint:",
        SSL_CHECKPOINT,
    )

    print(
        "Device:",
        DEVICE,
    )

    model = load_ssl_model(
        SSL_CHECKPOINT,
        DEVICE,
    )

    print(
        "\nSSL checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # Explicitly freeze SSL model
    # --------------------------------------------------------

    model.eval()

    for parameter in model.parameters():

        parameter.requires_grad = False

    # --------------------------------------------------------
    # Extract each split
    # --------------------------------------------------------

    for split in EXPECTED_SPLITS:

        representations = (
            extract_single_split(
                model=model,
                dataframe=dataframe,
                split=split,
            )
        )

        save_representation(
            split,
            representations,
        )

    print()
    print("=" * 100)
    print(
        "MILESTONE 8.5.2 — REPRESENTATION EXTRACTION COMPLETE"
    )
    print("=" * 100)


# ============================================================
# Representation dataset for fusion training
# ============================================================

class NaturalFusionRepresentationDataset(
    Dataset
):
    """
    Loads natural-missingness SSL representations.

    Samples with all three modalities missing are excluded from
    model training/evaluation because no predictive modality is
    available.

    Their existence remains documented in the population audit.
    """

    REQUIRED_KEYS = {
        "checkup_id",
        "patient_id",
        "image",
        "radiograph",
        "text",
        "labels",
        "modality_mask",
        "pattern",
        "missing_modality_count",
    }

    def __init__(
        self,
        representation_root,
        split,
        exclude_all_missing=True,
    ):

        path = (
            Path(representation_root)
            / f"{split}.pt"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Representation file not found:\n"
                f"{path}"
            )

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(data, dict):

            raise ValueError(
                f"Expected dictionary in {path}."
            )

        missing = (
            self.REQUIRED_KEYS
            -
            set(data.keys())
        )

        if missing:

            raise ValueError(
                f"Missing keys in {path}: "
                f"{sorted(missing)}"
            )

        self.checkup_ids = list(
            data["checkup_id"]
        )

        self.patient_ids = list(
            data["patient_id"]
        )

        self.image = (
            data["image"]
            .float()
        )

        self.radiograph = (
            data["radiograph"]
            .float()
        )

        self.text = (
            data["text"]
            .float()
        )

        self.labels = (
            data["labels"]
            .float()
        )

        self.modality_mask = (
            data["modality_mask"]
            .float()
        )

        self.patterns = list(
            data["pattern"]
        )

        self.missing_modality_count = (
            data[
                "missing_modality_count"
            ]
            .long()
        )

        # ----------------------------------------------------
        # Filter all-missing samples
        # ----------------------------------------------------

        if exclude_all_missing:

            keep = (
                self.modality_mask
                .sum(dim=1)
                > 0
            )

            self.checkup_ids = [
                value
                for value, flag
                in zip(
                    self.checkup_ids,
                    keep.tolist(),
                )
                if flag
            ]

            self.patient_ids = [
                value
                for value, flag
                in zip(
                    self.patient_ids,
                    keep.tolist(),
                )
                if flag
            ]

            self.image = (
                self.image[keep]
            )

            self.radiograph = (
                self.radiograph[keep]
            )

            self.text = (
                self.text[keep]
            )

            self.labels = (
                self.labels[keep]
            )

            self.modality_mask = (
                self.modality_mask[keep]
            )

            self.patterns = [
                value
                for value, flag
                in zip(
                    self.patterns,
                    keep.tolist(),
                )
                if flag
            ]

            self.missing_modality_count = (
                self.missing_modality_count[
                    keep
                ]
            )

        self._validate()

        print(
            f"{split}: "
            f"{len(self)} natural fusion samples"
        )

    def _validate(self):

        n = len(
            self.checkup_ids
        )

        if len(
            self.patient_ids
        ) != n:

            raise ValueError(
                "patient_id count mismatch."
            )

        tensors = [
            (
                "image",
                self.image,
                IMAGE_DIM,
            ),
            (
                "radiograph",
                self.radiograph,
                RADIOGRAPH_DIM,
            ),
            (
                "text",
                self.text,
                TEXT_DIM,
            ),
            (
                "labels",
                self.labels,
                config.NUM_LABELS,
            ),
            (
                "modality_mask",
                self.modality_mask,
                3,
            ),
        ]

        for name, tensor, dimension in tensors:

            if tensor.ndim != 2:

                raise ValueError(
                    f"{name} must be 2D."
                )

            if tensor.shape[0] != n:

                raise ValueError(
                    f"{name} sample count mismatch."
                )

            if tensor.shape[1] != dimension:

                raise ValueError(
                    f"Unexpected {name} dimension: "
                    f"{tensor.shape[1]}"
                )

        if (
            self.missing_modality_count
            .shape[0]
            != n
        ):

            raise ValueError(
                "Missing-modality count mismatch."
            )

        if len(
            self.patterns
        ) != n:

            raise ValueError(
                "Pattern count mismatch."
            )

    def __len__(self):

        return len(
            self.checkup_ids
        )

    def __getitem__(
        self,
        index,
    ):

        return {

            "checkup_id":
                self.checkup_ids[index],

            "patient_id":
                self.patient_ids[index],

            "image":
                self.image[index],

            "radiograph":
                self.radiograph[index],

            "text":
                self.text[index],

            "labels":
                self.labels[index],

            "modality_mask":
                self.modality_mask[index],

            "pattern":
                self.patterns[index],

            "missing_modality_count":
                self.missing_modality_count[
                    index
                ],
        }


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0

    for batch in tqdm(
        loader,
        desc="Training",
    ):

        image = batch[
            "image"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        radiograph = batch[
            "radiograph"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        text = batch[
            "text"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        modality_mask = batch[
            "modality_mask"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch[
            "labels"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            image,
            radiograph,
            text,
            modality_mask,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return (
        total_loss
        /
        max(len(loader), 1)
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_dataset(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0

    all_logits = []
    all_labels = []

    all_patterns = []
    all_missing_counts = []

    all_checkup_ids = []

    for batch in tqdm(
        loader,
        desc="Evaluation",
    ):

        image = batch[
            "image"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        radiograph = batch[
            "radiograph"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        text = batch[
            "text"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        modality_mask = batch[
            "modality_mask"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch[
            "labels"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(
            image,
            radiograph,
            text,
            modality_mask,
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += loss.item()

        all_logits.append(
            logits.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

        all_patterns.extend(
            batch["pattern"]
        )

        all_missing_counts.append(
            batch[
                "missing_modality_count"
            ].cpu()
        )

        all_checkup_ids.extend(
            batch["checkup_id"]
        )

    logits = torch.cat(
        all_logits,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    missing_counts = torch.cat(
        all_missing_counts,
        dim=0,
    )

    metrics = compute_metrics(
        logits,
        labels,
        threshold=0.5,
    )

    return {

        "loss":
            total_loss
            /
            max(len(loader), 1),

        "metrics":
            metrics,

        "logits":
            logits,

        "labels":
            labels,

        "patterns":
            all_patterns,

        "missing_modality_count":
            missing_counts,

        "checkup_id":
            all_checkup_ids,
    }


# ============================================================
# Train robust fusion
# ============================================================

def train_robust_fusion():

    print()
    print("=" * 100)
    print(
        "MILESTONE 8.5.2 — NATURAL MISSINGNESS ROBUST FUSION TRAINING"
    )
    print("=" * 100)

    train_dataset = (
        NaturalFusionRepresentationDataset(
            REPRESENTATION_ROOT,
            "train",
            exclude_all_missing=True,
        )
    )

    validation_dataset = (
        NaturalFusionRepresentationDataset(
            REPRESENTATION_ROOT,
            "validation",
            exclude_all_missing=True,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
    )

    model = RobustFusion(
        image_dim=IMAGE_DIM,
        radiograph_dim=RADIOGRAPH_DIM,
        text_dim=TEXT_DIM,
        modality_dim=MODALITY_DIM,
        hidden_dim=HIDDEN_DIM,
        num_labels=config.NUM_LABELS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    TRAINING_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_macro_f1 = -1.0
    best_epoch = -1

    for epoch in range(
        EPOCHS
    ):

        print()
        print(
            "=" * 80
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            "=" * 80
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        validation_result = (
            evaluate_dataset(
                model,
                validation_loader,
                criterion,
            )
        )

        validation_loss = (
            validation_result["loss"]
        )

        metrics = (
            validation_result["metrics"]
        )

        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Macro F1: "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Micro F1: "
            f"{metrics['micro_f1']:.4f}"
        )

        print(
            f"AUROC: "
            f"{metrics['auroc']:.4f}"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        record = {

            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "validation_loss":
                validation_loss,

            **metrics,
        }

        history.append(
            record
        )

        if (
            metrics["macro_f1"]
            >
            best_macro_f1
        ):

            best_macro_f1 = (
                metrics["macro_f1"]
            )

            best_epoch = (
                epoch + 1
            )

            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "model_name":
                    "natural_missingness_robust",

                "milestone":
                    "8.5.2",

                "epoch":
                    best_epoch,

                "validation_metrics":
                    metrics,

                "seed":
                    SEED,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "dataset":
                    str(DATASET_PATH),

                "ssl_checkpoint":
                    str(SSL_CHECKPOINT),

                "ssl_encoders_frozen":
                    True,

                "natural_missingness":
                    True,

                "controlled_dropout":
                    False,
            }

            torch.save(
                checkpoint,
                TRAINING_ROOT
                / "best_model.pt",
            )

            print(
                "Saved best model."
            )

    with open(
        TRAINING_ROOT
        / "history.json",
        "w",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    training_config = {

        "milestone":
            "8.5.2",

        "experiment":
            "Natural Missingness Robust Fusion",

        "model":
            "RobustFusion",

        "seed":
            SEED,

        "device":
            DEVICE,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "dropout":
            DROPOUT,

        "image_dim":
            IMAGE_DIM,

        "radiograph_dim":
            RADIOGRAPH_DIM,

        "text_dim":
            TEXT_DIM,

        "modality_dim":
            MODALITY_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "num_labels":
            config.NUM_LABELS,

        "dataset":
            str(DATASET_PATH),

        "ssl_checkpoint":
            str(SSL_CHECKPOINT),

        "representation_root":
            str(REPRESENTATION_ROOT),

        "ssl_encoders_frozen":
            True,

        "natural_missingness":
            True,

        "controlled_dropout":
            False,

        "all_missing_excluded":
            True,

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            best_macro_f1,

        "test_used_for_model_selection":
            False,
    }

    with open(
        TRAINING_ROOT
        / "config.json",
        "w",
    ) as f:

        json.dump(
            training_config,
            f,
            indent=2,
        )

    print()
    print("=" * 100)

    print(
        "MILESTONE 8.5.2 TRAINING COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation Macro F1:",
        f"{best_macro_f1:.4f}",
    )


# ============================================================
# Pattern-wise evaluation
# ============================================================

def evaluate_by_pattern(
    result,
):

    logits = result["logits"]

    labels = result["labels"]

    patterns = result["patterns"]

    missing_counts = (
        result["missing_modality_count"]
        .tolist()
    )

    rows = []

    for pattern in PATTERN_ORDER:

        indices = [
            index
            for index, value
            in enumerate(patterns)
            if value == pattern
        ]

        if not indices:

            rows.append({

                "pattern":
                    pattern,

                "samples":
                    0,

                "macro_f1":
                    np.nan,

                "micro_f1":
                    np.nan,

                "auroc":
                    np.nan,

                "accuracy":
                    np.nan,
            })

            continue

        pattern_logits = (
            logits[indices]
        )

        pattern_labels = (
            labels[indices]
        )

        # A pattern may be too small or contain a label with no
        # positive/negative variation. compute_metrics is still
        # the authoritative metric implementation used elsewhere.
        try:

            metrics = compute_metrics(
                pattern_logits,
                pattern_labels,
                threshold=0.5,
            )

            rows.append({

                "pattern":
                    pattern,

                "samples":
                    len(indices),

                "macro_f1":
                    metrics["macro_f1"],

                "micro_f1":
                    metrics["micro_f1"],

                "auroc":
                    metrics["auroc"],

                "accuracy":
                    metrics["accuracy"],
            })

        except Exception as error:

            print()
            print(
                f"WARNING: metric calculation failed "
                f"for pattern={pattern}: {error}"
            )

            rows.append({

                "pattern":
                    pattern,

                "samples":
                    len(indices),

                "macro_f1":
                    np.nan,

                "micro_f1":
                    np.nan,

                "auroc":
                    np.nan,

                "accuracy":
                    np.nan,
            })

    return pd.DataFrame(
        rows
    )


def evaluate_by_missing_count(
    result,
):

    logits = result["logits"]

    labels = result["labels"]

    missing_counts = (
        result[
            "missing_modality_count"
        ]
        .tolist()
    )

    rows = []

    for count in [
        0,
        1,
        2,
    ]:

        indices = [
            index
            for index, value
            in enumerate(
                missing_counts
            )
            if value == count
        ]

        if not indices:

            rows.append({

                "missing_modality_count":
                    count,

                "samples":
                    0,

                "macro_f1":
                    np.nan,

                "micro_f1":
                    np.nan,

                "auroc":
                    np.nan,

                "accuracy":
                    np.nan,
            })

            continue

        try:

            metrics = compute_metrics(
                logits[indices],
                labels[indices],
                threshold=0.5,
            )

            rows.append({

                "missing_modality_count":
                    count,

                "samples":
                    len(indices),

                "macro_f1":
                    metrics["macro_f1"],

                "micro_f1":
                    metrics["micro_f1"],

                "auroc":
                    metrics["auroc"],

                "accuracy":
                    metrics["accuracy"],
            })

        except Exception as error:

            print(
                f"WARNING: metric calculation failed "
                f"for missing count {count}: {error}"
            )

            rows.append({

                "missing_modality_count":
                    count,

                "samples":
                    len(indices),

                "macro_f1":
                    np.nan,

                "micro_f1":
                    np.nan,

                "auroc":
                    np.nan,

                "accuracy":
                    np.nan,
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Final test evaluation
# ============================================================

def test_model():

    print()
    print("=" * 100)
    print(
        "MILESTONE 8.5.2 — NATURAL MISSINGNESS TEST"
    )
    print("=" * 100)

    checkpoint_path = (
        TRAINING_ROOT
        / "best_model.pt"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Best model not found:\n"
            f"{checkpoint_path}"
        )

    test_dataset = (
        NaturalFusionRepresentationDataset(
            REPRESENTATION_ROOT,
            "test",
            exclude_all_missing=True,
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
    )

    model = RobustFusion(
        image_dim=IMAGE_DIM,
        radiograph_dim=RADIOGRAPH_DIM,
        text_dim=TEXT_DIM,
        modality_dim=MODALITY_DIM,
        hidden_dim=HIDDEN_DIM,
        num_labels=config.NUM_LABELS,
        dropout=DROPOUT,
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    result = evaluate_dataset(
        model,
        test_loader,
        criterion,
    )

    overall_metrics = result[
        "metrics"
    ]

    print()
    print("=" * 80)

    print(
        "OVERALL NATURAL MISSINGNESS TEST"
    )

    print("=" * 80)

    print(
        "Samples:",
        len(test_dataset),
    )

    print(
        "Loss:",
        f"{result['loss']:.4f}",
    )

    print(
        "Macro F1:",
        f"{overall_metrics['macro_f1']:.4f}",
    )

    print(
        "Micro F1:",
        f"{overall_metrics['micro_f1']:.4f}",
    )

    print(
        "AUROC:",
        f"{overall_metrics['auroc']:.4f}",
    )

    print(
        "Accuracy:",
        f"{overall_metrics['accuracy']:.4f}",
    )

    # --------------------------------------------------------
    # Pattern results
    # --------------------------------------------------------

    pattern_df = evaluate_by_pattern(
        result
    )

    print()
    print(
        "=" * 80
    )

    print(
        "TEST RESULTS BY NATURAL MISSINGNESS PATTERN"
    )

    print(
        "=" * 80
    )

    print(
        pattern_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Missing-count results
    # --------------------------------------------------------

    missing_count_df = (
        evaluate_by_missing_count(
            result
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "TEST RESULTS BY NUMBER OF MISSING MODALITIES"
    )

    print(
        "=" * 80
    )

    print(
        missing_count_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    TEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_df = pd.DataFrame(
        [
            {
                "scenario":
                    "natural_missingness_all_patterns",

                "description":
                    "Natural test population excluding all-missing visits",

                "samples":
                    len(test_dataset),

                "macro_f1":
                    overall_metrics[
                        "macro_f1"
                    ],

                "micro_f1":
                    overall_metrics[
                        "micro_f1"
                    ],

                "auroc":
                    overall_metrics[
                        "auroc"
                    ],

                "accuracy":
                    overall_metrics[
                        "accuracy"
                    ],
            }
        ]
    )

    overall_df.to_csv(
        TEST_ROOT
        / "overall_results.csv",
        index=False,
    )

    pattern_df.to_csv(
        TEST_ROOT
        / "pattern_results.csv",
        index=False,
    )

    missing_count_df.to_csv(
        TEST_ROOT
        / "missing_count_results.csv",
        index=False,
    )

    test_summary = {

        "milestone":
            "8.5.2",

        "dataset":
            str(DATASET_PATH),

        "ssl_checkpoint":
            str(SSL_CHECKPOINT),

        "best_model":
            str(checkpoint_path),

        "test_samples":
            len(test_dataset),

        "all_missing_excluded":
            True,

        "metrics":
            overall_metrics,

        "best_validation_macro_f1":
            checkpoint[
                "validation_metrics"
            ][
                "macro_f1"
            ],

        "best_epoch":
            checkpoint[
                "epoch"
            ],
    }

    with open(
        TEST_ROOT
        / "test_results.json",
        "w",
    ) as f:

        json.dump(
            test_summary,
            f,
            indent=2,
        )

    test_config = {

        "milestone":
            "8.5.2",

        "dataset":
            str(DATASET_PATH),

        "ssl_checkpoint":
            str(SSL_CHECKPOINT),

        "representation_root":
            str(REPRESENTATION_ROOT),

        "training_checkpoint":
            str(checkpoint_path),

        "device":
            DEVICE,

        "seed":
            SEED,

        "batch_size":
            BATCH_SIZE,

        "threshold":
            0.5,

        "natural_missingness":
            True,

        "controlled_dropout":
            False,

        "ssl_encoders_frozen":
            True,

        "test_used_for_model_selection":
            False,
    }

    with open(
        TEST_ROOT
        / "config.json",
        "w",
    ) as f:

        json.dump(
            test_config,
            f,
            indent=2,
        )

    print()
    print(
        "=" * 100
    )

    print(
        "MILESTONE 8.5.2 TEST COMPLETE"
    )

    print(
        "=" * 100
    )


# ============================================================
# Population summary
# ============================================================

def save_population_summary(
    dataframe,
):

    SUMMARY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_summary = (
        dataframe.groupby(
            [
                "split",
                "pattern",
            ]
        )
        .size()
        .reset_index(
            name="samples"
        )
    )

    overall_summary = (
        dataframe.groupby(
            "pattern"
        )
        .size()
        .reset_index(
            name="samples"
        )
    )

    overall_summary[
        "percentage"
    ] = (
        overall_summary["samples"]
        /
        len(dataframe)
        *
        100.0
    )

    missing_count_summary = (
        dataframe.groupby(
            "missing_modality_count"
        )
        .size()
        .reset_index(
            name="samples"
        )
    )

    missing_count_summary[
        "percentage"
    ] = (
        missing_count_summary["samples"]
        /
        len(dataframe)
        *
        100.0
    )

    population_summary = {

        "milestone":
            "8.5.2",

        "dataset":
            str(DATASET_PATH),

        "total_rows":
            int(len(dataframe)),

        "patients":
            int(
                dataframe[
                    "patient_id"
                ]
                .nunique()
            ),

        "checkups":
            int(
                dataframe[
                    "checkup_id"
                ]
                .nunique()
            ),

        "image_available":
            int(
                dataframe[
                    "image_available"
                ]
                .sum()
            ),

        "radiograph_available":
            int(
                dataframe[
                    "radiograph_available"
                ]
                .sum()
            ),

        "text_available":
            int(
                dataframe[
                    "text_available"
                ]
                .sum()
            ),

        "complete_cases":
            int(
                (
                    dataframe["pattern"]
                    ==
                    "complete"
                ).sum()
            ),

        "all_missing":
            int(
                (
                    dataframe["pattern"]
                    ==
                    "all_missing"
                ).sum()
            ),

        "split_counts":
            {
                str(split):
                    int(
                        (
                            dataframe[
                                "split"
                            ]
                            ==
                            split
                        ).sum()
                    )
                for split in EXPECTED_SPLITS
            },

        "pattern_counts":
            {
                str(pattern):
                    int(
                        (
                            dataframe[
                                "pattern"
                            ]
                            ==
                            pattern
                        ).sum()
                    )
                for pattern in PATTERN_ORDER
            },
    }

    split_summary.to_csv(
        SUMMARY_ROOT
        / "pattern_distribution_by_split.csv",
        index=False,
    )

    overall_summary.to_csv(
        SUMMARY_ROOT
        / "pattern_distribution.csv",
        index=False,
    )

    missing_count_summary.to_csv(
        SUMMARY_ROOT
        / "missing_count_distribution.csv",
        index=False,
    )

    with open(
        SUMMARY_ROOT
        / "protocol_summary.json",
        "w",
    ) as f:

        json.dump(
            population_summary,
            f,
            indent=2,
        )

    print()
    print(
        "=" * 100
    )

    print(
        "NATURAL MISSINGNESS POPULATION SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        "Total rows:",
        len(dataframe),
    )

    print(
        "Complete cases:",
        population_summary[
            "complete_cases"
        ],
    )

    print(
        "All-missing:",
        population_summary[
            "all_missing"
        ],
    )

    print()
    print(
        overall_summary.to_string(
            index=False
        )
    )

    print(
        "=" * 100
    )


# ============================================================
# Main
# ============================================================

def main():

    seed_everything(
        SEED
    )

    parser = argparse.ArgumentParser(
        description=(
            "Milestone 8.5.2 — "
            "Natural Missingness Robustness"
        )
    )

    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help=(
            "Skip SSL representation extraction "
            "and use existing representation files."
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 110)

    print(
        "MILESTONE 8.5.2 — NATURAL MISSINGNESS ROBUSTNESS"
    )

    print(
        "=" * 110
    )

    print(
        "Dataset:",
        DATASET_PATH,
    )

    print(
        "SSL checkpoint:",
        SSL_CHECKPOINT,
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Result root:",
        RESULT_ROOT,
    )

    # --------------------------------------------------------
    # Load and prepare natural population
    # --------------------------------------------------------

    dataframe = load_dataset()

    dataframe = prepare_dataframe(
        dataframe
    )

    # --------------------------------------------------------
    # Validate authoritative split
    # --------------------------------------------------------

    observed_splits = set(
        dataframe["split"]
        .dropna()
        .unique()
    )

    expected_splits = set(
        EXPECTED_SPLITS
    )

    if observed_splits != expected_splits:

        raise ValueError(
            "Unexpected split values. "
            f"Observed={sorted(observed_splits)}, "
            f"Expected={sorted(expected_splits)}"
        )

    # --------------------------------------------------------
    # Save protocol summary
    # --------------------------------------------------------

    save_population_summary(
        dataframe
    )

    # --------------------------------------------------------
    # SSL representation extraction
    # --------------------------------------------------------

    representation_files = [
        REPRESENTATION_ROOT / f"{split}.pt"
        for split in EXPECTED_SPLITS
    ]

    representations_exist = all(
        path.exists()
        for path in representation_files
    )

    if args.skip_extraction:

        if not representations_exist:

            missing_files = [
                str(path)
                for path in representation_files
                if not path.exists()
            ]

            raise FileNotFoundError(
                "Cannot skip representation extraction. "
                "The following representation files are missing:\n"
                + "\n".join(missing_files)
            )

        print()
        print("=" * 100)

        print(
            "MILESTONE 8.5.2 — "
            "USING EXISTING SSL REPRESENTATIONS"
        )

        print("=" * 100)

        for path in representation_files:

            print(
                "Using:",
                path,
            )

        print(
            "Representation extraction skipped."
        )

        print(
            "PASS"
        )

        print(
            "=" * 100
        )

    else:

        extract_all_representations(
            dataframe
        )

    # --------------------------------------------------------
    # Train robust fusion
    # --------------------------------------------------------

    train_robust_fusion()

    # --------------------------------------------------------
    # Final untouched test evaluation
    # --------------------------------------------------------

    test_model()

    print()
    print("=" * 110)

    print(
        "MILESTONE 8.5.2 — PASS"
    )

    print("=" * 110)

if __name__ == "__main__":

    main()