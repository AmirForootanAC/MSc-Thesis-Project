"""
SSL representation extraction for Milestone 7.

Uses the best SSL checkpoint from Milestone 6 and extracts
modality-specific encoder representations for complete
multimodal COde visits.

Representations:
    Photograph  -> 2048
    Radiograph  -> 2048
    Text        -> 768

Important:
    - Uses the same six-label dataset as the supervised baseline.
    - Uses the same patient-level splits.
    - Uses complete multimodal visits only.
    - Missing-modality experiments are NOT part of this module.
    - Projection heads are NOT used.
"""

from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.ssl.model import MultimodalSSLModel

from src.baseline import config
from src.baseline.image_loader import COdeImageLoader
from src.baseline.transforms import get_image_transform


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "results"
    / "six_label_patient_level_dataset"
    / "labeled_dataset.csv"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "COde-Dataset"
    / "Images"
)

SSL_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "ssl_pretraining"
    / "multimodal_dynamic"
    / "best_ssl_model.pt"
)

REPRESENTATION_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fusion"
    / "ssl_representations"
)

TEXT_MODEL_NAME = "distilbert-base-uncased"
TEXT_MAX_LENGTH = 256

BATCH_SIZE = 2

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Dataset
# ============================================================

class CompleteMultimodalRepresentationDataset(Dataset):
    """
    Complete-case dataset for SSL representation extraction.

    Each sample must contain:

        photograph
        radiograph
        clinical text
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
        split,
    ):

        self.df = pd.read_csv(csv_path)

        required_columns = [
            "split",
            "checkup_id",
            "patient_id",
            "photographs",
            "radiographs",
            *self.TEXT_COLUMNS,
            *config.LABEL_NAMES,
        ]

        missing = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # ----------------------------------------------------
        # Patient-level authoritative split
        # ----------------------------------------------------

        self.df = self.df[
            self.df["split"] == split
        ].reset_index(drop=True)

        # ----------------------------------------------------
        # Complete multimodal filtering
        # ----------------------------------------------------

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

        print(
            f"{split}: "
            f"{len(self.df)} complete multimodal samples"
        )

    def __len__(self):
        return len(self.df)

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

    @classmethod
    def build_text(cls, row):

        parts = []

        for column in cls.TEXT_COLUMNS:

            value = row[column]

            if pd.notna(value):

                value = str(value).strip()

                if value:
                    parts.append(value)

        return " ".join(parts)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        return {
            "checkup_id": str(
                row["checkup_id"]
            ),

            "patient_id": str(
                row["patient_id"]
            ),

            "images": self.parse_images(
                row["photographs"]
            ),

            "radiographs": self.parse_images(
                row["radiographs"]
            ),

            "text": self.build_text(
                row
            ),

            "labels": torch.tensor(
                row[config.LABEL_NAMES]
                .astype(float)
                .values,
                dtype=torch.float32,
            ),
        }


# ============================================================
# Collate
# ============================================================

def representation_collate(batch):

    return {
        "checkup_id": [
            sample["checkup_id"]
            for sample in batch
        ],

        "patient_id": [
            sample["patient_id"]
            for sample in batch
        ],

        "images": [
            sample["images"]
            for sample in batch
        ],

        "radiographs": [
            sample["radiographs"]
            for sample in batch
        ],

        "text": [
            sample["text"]
            for sample in batch
        ],

        "labels": torch.stack(
            [
                sample["labels"]
                for sample in batch
            ]
        ),
    }


# ============================================================
# Image Representation
# ============================================================

def extract_image_representations(
    file_lists,
    encoder,
    image_loader,
    transform,
    device,
    modality,
):

    batch_features = []

    for files in file_lists:

        sample_features = []

        for filename in files:

            try:

                image = image_loader.load(
                    filename,
                    modality=modality,
                )

                image = transform(image)

                image = (
                    image
                    .unsqueeze(0)
                    .to(device)
                )

                feature = encoder(
                    image
                )

                feature = feature.squeeze(0)

                sample_features.append(
                    feature
                )

            except Exception as exc:

                print(
                    f"Warning: failed to load "
                    f"{modality} '{filename}': {exc}"
                )

                continue

        if not sample_features:

            raise RuntimeError(
                f"No valid {modality} images found "
                "for a complete multimodal sample."
            )

        feature = torch.stack(
            sample_features
        ).mean(dim=0)

        batch_features.append(
            feature
        )

    return torch.stack(
        batch_features
    )


# ============================================================
# Text Representation
# ============================================================

def extract_text_representations(
    texts,
    model,
    tokenizer,
    device,
):
    """
    Extract text representations through the SSL model's
    encoder interface.

    Output:
        [batch_size, 768]
    """

    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=TEXT_MAX_LENGTH,
        return_tensors="pt",
    )

    input_ids = tokens[
        "input_ids"
    ].to(device)

    attention_mask = tokens[
        "attention_mask"
    ].to(device)

    return model.encoders.encode_text(
        input_ids,
        attention_mask,
    )


# ============================================================
# Checkpoint Loading
# ============================================================

def load_ssl_model(
    checkpoint_path,
    device,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"SSL checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    model = MultimodalSSLModel()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# Save
# ============================================================

def save_representations(
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
        f"Saved representations: {output_path}"
    )


# ============================================================
# Split Extraction
# ============================================================

@torch.no_grad()
def extract_split(
    model,
    split,
):

    dataset = CompleteMultimodalRepresentationDataset(
        csv_path=DATASET_PATH,
        split=split,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=representation_collate,
        num_workers=0,
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

    for batch_index, batch in enumerate(loader):

        print(
            f"{split}: "
            f"batch {batch_index + 1}/{len(loader)}"
        )

        # ----------------------------------------------------
        # Photograph
        # ----------------------------------------------------

        image_repr = extract_image_representations(
            file_lists=batch["images"],
            encoder=model.encoders.image_encoder,
            image_loader=image_loader,
            transform=transform,
            device=DEVICE,
            modality="photograph",
        )

        # ----------------------------------------------------
        # Radiograph
        # ----------------------------------------------------

        radiograph_repr = extract_image_representations(
            file_lists=batch["radiographs"],
            encoder=model.encoders.radiograph_encoder,
            image_loader=image_loader,
            transform=transform,
            device=DEVICE,
            modality="radiograph",
        )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        text_repr = extract_text_representations(
            texts=batch["text"],
            model=model,
            tokenizer=tokenizer,
            device=DEVICE,
        )

        image_features.append(
            image_repr.cpu()
        )

        radiograph_features.append(
            radiograph_repr.cpu()
        )

        text_features.append(
            text_repr.cpu()
        )

        labels.append(
            batch["labels"].cpu()
        )

        checkup_ids.extend(
            batch["checkup_id"]
        )

        patient_ids.extend(
            batch["patient_id"]
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

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert image_features.shape == (
        len(dataset),
        2048,
    )

    assert radiograph_features.shape == (
        len(dataset),
        2048,
    )

    assert text_features.shape == (
        len(dataset),
        768,
    )

    assert labels.shape == (
        len(dataset),
        config.NUM_LABELS,
    )

    assert len(checkup_ids) == len(dataset)

    assert len(patient_ids) == len(dataset)

    # --------------------------------------------------------
    # Build representation artifact
    # --------------------------------------------------------

    representations = {
        "checkup_id": checkup_ids,
        "patient_id": patient_ids,
        "image": image_features,
        "radiograph": radiograph_features,
        "text": text_features,
        "labels": labels,
    }

    print()
    print("=" * 60)
    print(f"{split.upper()} REPRESENTATION CHECK")
    print("=" * 60)

    print(
        "Samples:",
        len(dataset),
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
        "Labels:",
        tuple(labels.shape),
    )

    print("PASS")
    print("=" * 60)

    save_representations(
        split,
        representations,
    )

    return representations


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("MILESTONE 7.2 — SSL REPRESENTATION EXTRACTION")
    print("=" * 60)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "SSL checkpoint:",
        SSL_CHECKPOINT,
    )

    print(
        "Dataset:",
        DATASET_PATH,
    )

    print(
        "Output:",
        REPRESENTATION_ROOT,
    )

    model = load_ssl_model(
        SSL_CHECKPOINT,
        DEVICE,
    )

    print(
        "\nSSL checkpoint loaded successfully."
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        extract_split(
            model=model,
            split=split,
        )

    print()
    print("=" * 60)
    print("MILESTONE 7.2 — PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()