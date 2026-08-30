"""
Diagnostic experiment — End-to-End SSL Fusion Fine-Tuning.

Purpose
-------
Test whether the current SSL Fusion performance is limited by the
use of frozen/precomputed SSL representations.

This experiment keeps the existing Milestone 7 Fusion architecture
but fine-tunes the SSL encoders jointly with the Fusion network.

Protocol
--------
    Photograph + Radiograph + Clinical Text
                    |
             SSL encoders
                    |
              MainFusion
                    |
                6 labels

Unlike the original Fusion pipeline:

    representation_extractor.py
            -> .pt files
            -> frozen representations
            -> Fusion MLP

this experiment keeps the SSL encoders inside the computational
graph and optimizes them end-to-end.

Important
---------
- Uses the SAME complete-case population as Fusion.
- Uses the authoritative patient-level split.
- Uses six labels.
- Uses the existing best SSL checkpoint as initialization.
- Does NOT use SSL projection heads.
- Does NOT modify the existing Fusion experiment.
- This is a diagnostic experiment.
- Missing-modality experiments are NOT part of this script.

Resume
------
Two checkpoints are maintained:

    best_model.pt
        -> Best validation Macro F1 model.

    latest_checkpoint.pt
        -> Most recent training state, including optimizer state.

Training always resumes from latest_checkpoint.pt when available,
while best_model.pt remains the authoritative best model.

This prevents resuming from an older best epoch and accidentally
discarding later training progress.

Expected complete-case population:
    4,195 visits

Expected split:
    train + validation + test = 4,195

Output:
    results/fusion/main_finetune/
"""


import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import AutoTokenizer

from src.baseline.collate import baseline_collate
from src.baseline.image_loader import COdeImageLoader
from src.baseline.transforms import get_image_transform
from src.baseline.metrics import compute_metrics

from src.fusion.dataset import FusionDataset
from src.fusion.fusion_model import MainFusion

from src.ssl.model import MultimodalSSLModel


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

CSV_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "six_label_patient_level_dataset"
    / "labeled_dataset.csv"
)

EXPECTED_TOTAL_SAMPLES = 4195


# ------------------------------------------------------------
# SSL checkpoint
# ------------------------------------------------------------

SSL_WEIGHT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "ssl_pretraining"
    / "multimodal_dynamic"
    / "best_ssl_model.pt"
)


# ------------------------------------------------------------
# Image root
# ------------------------------------------------------------

IMAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "COde-Dataset"
    / "Images"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

RESULT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "fusion"
    / "main_finetune"
)


# ------------------------------------------------------------
# Text
# ------------------------------------------------------------

TEXT_MODEL_NAME = (
    "distilbert-base-uncased"
)

TEXT_MAX_LENGTH = 256


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

BATCH_SIZE = 8

GRADIENT_ACCUMULATION_STEPS = 8

EPOCHS = 70

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 1e-4

PATIENCE = 20


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

MODEL_NAME = "main"

NUM_LABELS = 6

IMAGE_DIM = 2048

RADIOGRAPH_DIM = 2048

TEXT_DIM = 768

MODALITY_DIM = 512

HIDDEN_DIM = 512

DROPOUT = 0.3


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
# Complete-case verification
# ============================================================

def verify_complete_case_population():

    """
    Verify that FusionDataset produces exactly the expected
    complete-case population.

    FusionDataset itself performs:
        photograph available
        AND radiograph available
        AND clinical text available
    """

    if not CSV_SOURCE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{CSV_SOURCE}"
        )

    print()
    print("=" * 60)
    print("VERIFYING COMPLETE-CASE POPULATION")
    print("=" * 60)

    df = pd.read_csv(
        CSV_SOURCE
    )

    print(
        "Source rows:",
        len(df),
    )

    required = [
        "split",
        "photographs",
        "radiographs",
        "chief_complaint",
        "present_illness",
        "past_medical_record",
        "examination",
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

    def has_value(value):

        return (
            pd.notna(value)
            and str(value).strip() != ""
        )

    image_available = (
        df["photographs"]
        .apply(has_value)
    )

    radiograph_available = (
        df["radiographs"]
        .apply(has_value)
    )

    text_available = (
        df[
            [
                "chief_complaint",
                "present_illness",
                "past_medical_record",
                "examination",
            ]
        ]
        .notna()
        .any(axis=1)
    )

    complete_case = (
        image_available
        &
        radiograph_available
        &
        text_available
    )

    count = int(
        complete_case.sum()
    )

    print(
        "Complete-case visits:",
        count,
    )

    print(
        "Expected:",
        EXPECTED_TOTAL_SAMPLES,
    )

    if count != EXPECTED_TOTAL_SAMPLES:

        raise RuntimeError(
            "Complete-case population mismatch.\n"
            f"Expected {EXPECTED_TOTAL_SAMPLES}, "
            f"found {count}."
        )

    split_counts = (
        df.loc[
            complete_case,
            "split",
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Complete-case split distribution:"
    )

    for split, count in split_counts.items():

        print(
            f"  {split}: {count}"
        )

    print()
    print(
        "PASS — complete-case population verified."
    )

    print("=" * 60)


# ============================================================
# Model
# ============================================================

class EndToEndSSLFusion(nn.Module):

    """
    End-to-end multimodal model.

    SSL encoders:
        image       -> 2048
        radiograph  -> 2048
        text        -> 768

    Fusion:
        MainFusion

    SSL projection heads are intentionally excluded.
    """

    def __init__(
        self,
        ssl_model,
        fusion_model,
    ):

        super().__init__()

        self.ssl_model = ssl_model

        self.fusion_model = fusion_model

    def forward(
        self,
        image,
        radiograph,
        input_ids,
        attention_mask,
    ):

        image_features = (
            self.ssl_model
            .encoders
            .encode_image(
                image
            )
        )

        radiograph_features = (
            self.ssl_model
            .encoders
            .encode_radiograph(
                radiograph
            )
        )

        text_features = (
            self.ssl_model
            .encoders
            .encode_text(
                input_ids,
                attention_mask,
            )
        )

        logits = self.fusion_model(
            image_features,
            radiograph_features,
            text_features,
        )

        return logits


# ============================================================
# Load SSL model
# ============================================================

def load_ssl_model():

    if not SSL_WEIGHT.exists():

        raise FileNotFoundError(
            f"SSL checkpoint not found:\n"
            f"{SSL_WEIGHT}"
        )

    print()
    print(
        "Loading SSL checkpoint:"
    )

    print(
        SSL_WEIGHT
    )

    ssl_model = MultimodalSSLModel(
        text_model_name=TEXT_MODEL_NAME
    )

    checkpoint = torch.load(
        SSL_WEIGHT,
        map_location="cpu",
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    ssl_model.load_state_dict(
        state_dict
    )

    ssl_model = ssl_model.to(
        DEVICE
    )

    return ssl_model


# ============================================================
# Build model
# ============================================================

def build_model():

    ssl_model = load_ssl_model()

    # --------------------------------------------------------
    # SSL encoders are intentionally trainable.
    # --------------------------------------------------------

    for parameter in (
        ssl_model
        .encoders
        .parameters()
    ):

        parameter.requires_grad = True

    # --------------------------------------------------------
    # SSL projection heads are NOT used.
    # --------------------------------------------------------

    for parameter in (
        ssl_model
        .projectors
        .parameters()
    ):

        parameter.requires_grad = False

    # --------------------------------------------------------
    # Existing MainFusion architecture
    # --------------------------------------------------------

    fusion_model = MainFusion(
        image_dim=IMAGE_DIM,
        radiograph_dim=RADIOGRAPH_DIM,
        text_dim=TEXT_DIM,
        modality_dim=MODALITY_DIM,
        hidden_dim=HIDDEN_DIM,
        num_labels=NUM_LABELS,
        dropout=DROPOUT,
    )

    model = EndToEndSSLFusion(
        ssl_model=ssl_model,
        fusion_model=fusion_model,
    )

    model = model.to(
        DEVICE
    )

    return model


# ============================================================
# Image loading
# ============================================================

def load_image_batch(
    file_lists,
    modality,
    image_loader,
    transform,
):

    """
    Load multiple images per visit.

    Each visit is represented by the mean of its valid image
    tensors before entering the SSL encoder.
    """

    visit_images = []

    for files in file_lists:

        images = []

        for filename in files:

            try:

                image = image_loader.load(
                    filename,
                    modality=modality,
                )

                image = transform(
                    image
                )

                images.append(
                    image
                )

            except Exception as exc:

                print(
                    f"Warning: failed to load "
                    f"{modality} '{filename}': "
                    f"{exc}"
                )

                continue

        if not images:

            raise RuntimeError(
                f"No valid {modality} image found "
                "for a complete-case sample."
            )

        visit_images.append(
            torch.stack(
                images
            ).mean(
                dim=0
            )
        )

    return torch.stack(
        visit_images
    ).to(
        DEVICE,
        non_blocking=True,
    )


# ============================================================
# Prepare batch
# ============================================================

def prepare_batch(
    batch,
    image_loader,
    transform,
    tokenizer,
):

    # --------------------------------------------------------
    # Photograph
    # --------------------------------------------------------

    image = load_image_batch(
        batch["images"],
        modality="photograph",
        image_loader=image_loader,
        transform=transform,
    )

    # --------------------------------------------------------
    # Radiograph
    # --------------------------------------------------------

    radiograph = load_image_batch(
        batch["radiographs"],
        modality="radiograph",
        image_loader=image_loader,
        transform=transform,
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    tokens = tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=TEXT_MAX_LENGTH,
        return_tensors="pt",
    )

    input_ids = tokens[
        "input_ids"
    ].to(
        DEVICE,
        non_blocking=True,
    )

    attention_mask = tokens[
        "attention_mask"
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

    return (
        image,
        radiograph,
        input_ids,
        attention_mask,
        labels,
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    image_loader,
    transform,
    tokenizer,
    desc="Validation",
):

    model.eval()

    total_loss = 0.0

    all_logits = []

    all_labels = []

    for batch in tqdm(
        loader,
        desc=desc,
        leave=False,
    ):

        (
            image,
            radiograph,
            input_ids,
            attention_mask,
            labels,
        ) = prepare_batch(
            batch,
            image_loader,
            transform,
            tokenizer,
        )

        logits = model(
            image,
            radiograph,
            input_ids,
            attention_mask,
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

    logits = torch.cat(
        all_logits,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    metrics = compute_metrics(
        logits,
        labels,
        threshold=0.5,
    )

    return (
        total_loss / len(loader),
        metrics,
    )


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    image_loader,
    transform,
    tokenizer,
):

    model.train()

    total_loss = 0.0

    optimizer.zero_grad(
        set_to_none=True
    )

    progress = tqdm(
        loader,
        desc="Training",
    )

    for batch_index, batch in enumerate(
        progress
    ):

        (
            image,
            radiograph,
            input_ids,
            attention_mask,
            labels,
        ) = prepare_batch(
            batch,
            image_loader,
            transform,
            tokenizer,
        )

        logits = model(
            image,
            radiograph,
            input_ids,
            attention_mask,
        )

        loss = criterion(
            logits,
            labels,
        )

        # ----------------------------------------------------
        # Gradient accumulation
        # ----------------------------------------------------

        scaled_loss = (
            loss
            /
            GRADIENT_ACCUMULATION_STEPS
        )

        scaled_loss.backward()

        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

        is_accumulation_boundary = (
            (batch_index + 1)
            % GRADIENT_ACCUMULATION_STEPS
            == 0
        )

        is_last_batch = (
            batch_index + 1
            == len(loader)
        )

        if (
            is_accumulation_boundary
            or is_last_batch
        ):

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return (
        total_loss
        /
        len(loader)
    )


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    validation_metrics,
    best_macro_f1,
    best_auroc,
):

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "epoch":
            epoch,

        "model":
            MODEL_NAME,

        "experiment":
            "end_to_end_ssl_fusion_finetune",

        "validation_metrics":
            validation_metrics,

        "best_macro_f1":
            best_macro_f1,

        "best_auroc":
            best_auroc,

        "seed":
            SEED,

        "batch_size":
            BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# Main
# ============================================================

def main():

    seed_everything(
        SEED
    )

    print("=" * 60)
    print(
        "DIAGNOSTIC — END-TO-END SSL FUSION FINE-TUNING"
    )
    print("=" * 60)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Dataset:",
        CSV_SOURCE,
    )

    print(
        "SSL checkpoint:",
        SSL_WEIGHT,
    )

    print(
        "Output:",
        RESULT_ROOT,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Gradient accumulation:",
        GRADIENT_ACCUMULATION_STEPS,
    )

    print(
        "Effective batch size:",
        BATCH_SIZE
        *
        GRADIENT_ACCUMULATION_STEPS,
    )

    print(
        "Total epochs:",
        EPOCHS,
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Verify dataset
    # --------------------------------------------------------

    verify_complete_case_population()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = FusionDataset(
        csv_path=CSV_SOURCE,
        split="train",
    )

    validation_dataset = FusionDataset(
        csv_path=CSV_SOURCE,
        split="validation",
    )

    test_dataset = FusionDataset(
        csv_path=CSV_SOURCE,
        split="test",
    )

    train_count = len(
        train_dataset
    )

    validation_count = len(
        validation_dataset
    )

    test_count = len(
        test_dataset
    )

    total_count = (
        train_count
        +
        validation_count
        +
        test_count
    )

    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Train      : {train_count}"
    )

    print(
        f"Validation : {validation_count}"
    )

    print(
        f"Test       : {test_count}"
    )

    print(
        f"Total      : {total_count}"
    )

    if total_count != EXPECTED_TOTAL_SAMPLES:

        raise RuntimeError(
            "Dataset total mismatch.\n"
            f"Expected {EXPECTED_TOTAL_SAMPLES}, "
            f"found {total_count}."
        )

    print(
        "PASS"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
        collate_fn=baseline_collate,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
        collate_fn=baseline_collate,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
        collate_fn=baseline_collate,
    )

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    image_loader = COdeImageLoader(
        IMAGE_ROOT
    )

    transform = get_image_transform()

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL_NAME
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Building end-to-end model..."
    )

    model = build_model()

    # --------------------------------------------------------
    # Trainable parameters
    # --------------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        RESULT_ROOT
        / "best_model.pt"
    )

    latest_path = (
        RESULT_ROOT
        / "latest_checkpoint.pt"
    )

    history_path = (
        RESULT_ROOT
        / "history.json"
    )

    # ========================================================
    # Resume state
    # ========================================================

    history = []

    best_macro_f1 = -float("inf")

    best_auroc = -float("inf")

    best_epoch = -1

    patience_counter = 0

    start_epoch = 0

    # --------------------------------------------------------
    # Load history first
    # --------------------------------------------------------

    if history_path.exists():

        with open(
            history_path,
            "r",
        ) as f:

            history = json.load(
                f
            )

        history = sorted(
            history,
            key=lambda record:
                record["epoch"],
        )

    # ========================================================
    # Resume from latest checkpoint
    # ========================================================

    if latest_path.exists():

        print()
        print("=" * 60)
        print("RESUME MODE")
        print("=" * 60)

        print(
            "Resume checkpoint:",
            latest_path,
        )

        checkpoint = torch.load(
            latest_path,
            map_location=DEVICE,
        )

        # ----------------------------------------------------
        # Load latest model
        # ----------------------------------------------------

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        # ----------------------------------------------------
        # Load optimizer
        # ----------------------------------------------------

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        previous_epoch = int(
            checkpoint["epoch"]
        )

        start_epoch = previous_epoch

        print(
            f"Loaded latest model from epoch: "
            f"{previous_epoch}"
        )

        print(
            "Optimizer state restored."
        )

        # ----------------------------------------------------
        # Restore best state from history
        #
        # IMPORTANT:
        # Best state is determined from the full history,
        # not from the latest checkpoint's validation metrics.
        # ----------------------------------------------------

        if history:

            best_record = max(
                history,
                key=lambda record:
                    record["macro_f1"],
            )

            best_epoch = int(
                best_record["epoch"]
            )

            best_macro_f1 = float(
                best_record["macro_f1"]
            )

            best_auroc = float(
                best_record["auroc"]
            )

            epochs_since_best = (
                previous_epoch
                -
                best_epoch
            )

            patience_counter = max(
                0,
                epochs_since_best,
            )

        else:

            best_epoch = int(
                checkpoint.get(
                    "epoch",
                    previous_epoch,
                )
            )

            best_macro_f1 = float(
                checkpoint.get(
                    "best_macro_f1",
                    checkpoint[
                        "validation_metrics"
                    ]["macro_f1"],
                )
            )

            best_auroc = float(
                checkpoint.get(
                    "best_auroc",
                    checkpoint[
                        "validation_metrics"
                    ]["auroc"],
                )
            )

            patience_counter = 0

        print(
            f"Loaded history: "
            f"{len(history)} epochs"
        )

        print()
        print(
            f"Previous best epoch: "
            f"{best_epoch}"
        )

        print(
            f"Previous best validation "
            f"Macro F1: {best_macro_f1:.4f}"
        )

        print(
            f"Previous best validation "
            f"AUROC: {best_auroc:.4f}"
        )

        print(
            f"Epochs since best: "
            f"{patience_counter}"
        )

        print("=" * 60)

    # ========================================================
    # Backward-compatible fallback
    #
    # If latest_checkpoint.pt does not exist but best_model.pt
    # does, resume from best_model.pt.
    # ========================================================

    elif best_path.exists():

        print()
        print("=" * 60)
        print("LEGACY RESUME MODE")
        print("=" * 60)

        print(
            "No latest checkpoint found."
        )

        print(
            "Falling back to:",
            best_path,
        )

        checkpoint = torch.load(
            best_path,
            map_location=DEVICE,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer_state_loaded = False

        if (
            isinstance(checkpoint, dict)
            and "optimizer_state_dict" in checkpoint
        ):

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

            optimizer_state_loaded = True

        previous_epoch = int(
            checkpoint.get(
                "epoch",
                0,
            )
        )

        start_epoch = previous_epoch

        if history:

            best_record = max(
                history,
                key=lambda record:
                    record["macro_f1"],
            )

            best_epoch = int(
                best_record["epoch"]
            )

            best_macro_f1 = float(
                best_record["macro_f1"]
            )

            best_auroc = float(
                best_record["auroc"]
            )

            patience_counter = max(
                0,
                previous_epoch
                -
                best_epoch,
            )

        else:

            best_epoch = previous_epoch

            best_macro_f1 = float(
                checkpoint.get(
                    "best_macro_f1",
                    checkpoint[
                        "validation_metrics"
                    ]["macro_f1"],
                )
            )

            best_auroc = float(
                checkpoint.get(
                    "best_auroc",
                    checkpoint[
                        "validation_metrics"
                    ]["auroc"],
                )
            )

            patience_counter = 0

        print(
            f"Loaded model from epoch: "
            f"{previous_epoch}"
        )

        print(
            f"Previous best epoch: "
            f"{best_epoch}"
        )

        print(
            f"Previous best validation "
            f"Macro F1: {best_macro_f1:.4f}"
        )

        print(
            f"Previous best validation "
            f"AUROC: {best_auroc:.4f}"
        )

        print(
            f"Epochs since best: "
            f"{patience_counter}"
        )

        if optimizer_state_loaded:

            print(
                "Optimizer state restored."
            )

        else:

            print(
                "Optimizer state not available."
            )

            print(
                "Model weights were restored, "
                "but optimizer starts fresh."
            )

        print("=" * 60)

    else:

        print()
        print(
            "No previous checkpoint found."
        )

        print(
            "Starting from Epoch 1."
        )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        start_epoch,
        EPOCHS,
    ):

        current_epoch = epoch + 1

        print()
        print("=" * 60)
        print(
            f"Epoch {current_epoch}/{EPOCHS}"
        )
        print("=" * 60)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            image_loader=image_loader,
            transform=transform,
            tokenizer=tokenizer,
        )

        validation_loss, validation_metrics = (
            evaluate(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                image_loader=image_loader,
                transform=transform,
                tokenizer=tokenizer,
                desc="Validation",
            )
        )

        print()
        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Macro F1: "
            f"{validation_metrics['macro_f1']:.4f}"
        )

        print(
            f"Validation Micro F1: "
            f"{validation_metrics['micro_f1']:.4f}"
        )

        print(
            f"Validation AUROC: "
            f"{validation_metrics['auroc']:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        epoch_record = {
            "epoch": current_epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            **validation_metrics,
        }

        history.append(
            epoch_record
        )

        # ----------------------------------------------------
        # Best model selection
        #
        # Validation Macro F1 remains the primary criterion.
        # ----------------------------------------------------

        current_macro_f1 = (
            validation_metrics["macro_f1"]
        )

        is_new_best = (
            current_macro_f1
            >
            best_macro_f1
        )

        if is_new_best:

            best_macro_f1 = (
                current_macro_f1
            )

            best_auroc = (
                validation_metrics["auroc"]
            )

            best_epoch = (
                current_epoch
            )

            patience_counter = 0

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=best_epoch,
                validation_metrics=validation_metrics,
                best_macro_f1=best_macro_f1,
                best_auroc=best_auroc,
            )

            print()
            print(
                "-> New best checkpoint."
            )

        else:

            patience_counter += 1

            print()
            print(
                f"-> No Macro F1 improvement "
                f"({patience_counter}/{PATIENCE})"
            )

        # ----------------------------------------------------
        # ALWAYS save latest training state.
        #
        # This is the critical Resume fix.
        # ----------------------------------------------------

        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            epoch=current_epoch,
            validation_metrics=validation_metrics,
            best_macro_f1=best_macro_f1,
            best_auroc=best_auroc,
        )

        # ----------------------------------------------------
        # Save history after every epoch
        # ----------------------------------------------------

        with open(
            history_path,
            "w",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            patience_counter
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping."
            )

            break

    # ========================================================
    # Load best model
    # ========================================================

    print()
    print("=" * 60)
    print("LOADING BEST MODEL")
    print("=" * 60)

    if not best_path.exists():

        raise RuntimeError(
            "Best model checkpoint does not exist."
        )

    checkpoint = torch.load(
        best_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if (
        isinstance(checkpoint, dict)
        and "epoch" in checkpoint
    ):

        best_epoch = int(
            checkpoint["epoch"]
        )

    if (
        isinstance(checkpoint, dict)
        and "best_macro_f1" in checkpoint
    ):

        best_macro_f1 = float(
            checkpoint["best_macro_f1"]
        )

    if (
        isinstance(checkpoint, dict)
        and "best_auroc" in checkpoint
    ):

        best_auroc = float(
            checkpoint["best_auroc"]
        )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation AUROC: "
        f"{best_auroc:.4f}"
    )

    print(
        f"Best validation Macro F1: "
        f"{best_macro_f1:.4f}"
    )

    # ========================================================
    # Final test
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL TEST — END-TO-END SSL FUSION")
    print("=" * 60)

    test_loss, test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        image_loader=image_loader,
        transform=transform,
        tokenizer=tokenizer,
        desc="Test",
    )

    print()
    print(
        f"Test loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Micro F1: "
        f"{test_metrics['micro_f1']:.4f}"
    )

    print(
        f"AUROC: "
        f"{test_metrics['auroc']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        RESULT_ROOT
        / "final_model.pt",
    )

    # --------------------------------------------------------
    # Save test metrics
    # --------------------------------------------------------

    with open(
        RESULT_ROOT
        / "test_metrics.json",
        "w",
    ) as f:

        json.dump(
            {
                "experiment":
                    "end_to_end_ssl_fusion_finetune",

                "model":
                    MODEL_NAME,

                "split":
                    "test",

                "samples":
                    test_count,

                "macro_f1":
                    test_metrics["macro_f1"],

                "micro_f1":
                    test_metrics["micro_f1"],

                "auroc":
                    test_metrics["auroc"],

                "accuracy":
                    test_metrics["accuracy"],

                "test_loss":
                    test_loss,

                "best_epoch":
                    best_epoch,

                "best_validation_auroc":
                    best_auroc,

                "best_validation_macro_f1":
                    best_macro_f1,

                "seed":
                    SEED,

                "batch_size":
                    BATCH_SIZE,

                "gradient_accumulation_steps":
                    GRADIENT_ACCUMULATION_STEPS,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "ssl_checkpoint":
                    str(SSL_WEIGHT),

                "dataset":
                    str(CSV_SOURCE),

                "complete_case_population":
                    EXPECTED_TOTAL_SAMPLES,
            },
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    with open(
        RESULT_ROOT
        / "config.json",
        "w",
    ) as f:

        json.dump(
            {
                "experiment":
                    "end_to_end_ssl_fusion_finetune",

                "model":
                    MODEL_NAME,

                "device":
                    DEVICE,

                "seed":
                    SEED,

                "dataset":
                    str(CSV_SOURCE),

                "expected_complete_case_samples":
                    EXPECTED_TOTAL_SAMPLES,

                "train_samples":
                    train_count,

                "validation_samples":
                    validation_count,

                "test_samples":
                    test_count,

                "ssl_checkpoint":
                    str(SSL_WEIGHT),

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
                    NUM_LABELS,

                "dropout":
                    DROPOUT,

                "batch_size":
                    BATCH_SIZE,

                "gradient_accumulation_steps":
                    GRADIENT_ACCUMULATION_STEPS,

                "effective_batch_size":
                    BATCH_SIZE
                    *
                    GRADIENT_ACCUMULATION_STEPS,

                "epochs":
                    EPOCHS,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "patience":
                    PATIENCE,

                "fine_tune_ssl_encoders":
                    True,

                "ssl_projection_heads_used":
                    False,

                "missing_modality":
                    False,

                "best_epoch":
                    best_epoch,

                "best_validation_auroc":
                    best_auroc,

                "best_validation_macro_f1":
                    best_macro_f1,
            },
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("END-TO-END SSL FUSION — COMPLETE")
    print("=" * 60)

    print()
    print(
        "Test samples:",
        test_count,
    )

    print(
        f"Macro F1 : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Micro F1 : "
        f"{test_metrics['micro_f1']:.4f}"
    )

    print(
        f"AUROC    : "
        f"{test_metrics['auroc']:.4f}"
    )

    print(
        f"Accuracy : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULT_ROOT
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()