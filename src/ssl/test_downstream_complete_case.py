"""
TEST — SSL downstream on the complete multimodal population.

Purpose
-------
Diagnostic experiment for Milestone 7.

This script evaluates the EXISTING SSL checkpoint using the exact
complete-case population used by Fusion:

    Photograph + Radiograph + Clinical Text

The original SSL downstream protocol is preserved:

    - same SSL checkpoint
    - same downstream classifier
    - same fine-tuning mode
    - same optimizer
    - same learning rate
    - same epochs
    - same early stopping
    - same six labels

The only intentional difference is the dataset population.

Instead of allowing each modality to use its own available samples,
all three modalities are evaluated on the SAME complete-case visits.

Expected total population:
    4,195 visits

Important:
    During fine-tuning, ONLY the encoder corresponding to the
    evaluated modality is trainable.

    Example:
        image experiment
            -> image encoder trainable
            -> radiograph encoder frozen
            -> text encoder frozen

This prevents unused modality encoders from being unnecessarily
included in the trainable parameter set.

This is a diagnostic/test script only.
Do NOT use this file as the production downstream pipeline.
"""

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.ssl.model import MultimodalSSLModel
from src.ssl.downstream_dataset import SSLDownstreamDataset
from src.ssl.tokenizer import ClinicalTokenizer

from src.baseline.collate import baseline_collate
from src.baseline.image_loader import COdeImageLoader
from src.baseline.transforms import get_image_transform
from src.baseline.metrics import compute_metrics
from src.baseline import config


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------
# IMPORTANT:
# This is the SAME dataset used by Fusion.
# ------------------------------------------------------------

CSV_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "six_label_patient_level_dataset"
    / "labeled_dataset.csv"
)


# ------------------------------------------------------------
# Temporary complete-case dataset
# ------------------------------------------------------------

COMPLETE_CASE_CSV = (
    PROJECT_ROOT
    / "results"
    / "ssl_pretraining"
    / "test_complete_case_4195"
    / "complete_case_dataset.csv"
)


SSL_WEIGHT = (
    PROJECT_ROOT
    / "results"
    / "ssl_pretraining"
    / "multimodal_dynamic"
    / "best_ssl_model.pt"
)


IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "COde-Dataset"
    / "Images"
)


RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ssl_pretraining"
    / "test_complete_case_4195"
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ------------------------------------------------------------
# Same downstream settings as src/ssl/downstream.py
# ------------------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 5

MODE = "fine_tune"


# ------------------------------------------------------------
# Modalities
# ------------------------------------------------------------

MODALITIES = [
    "image",
    "radiograph",
    "text",
]


NUM_LABELS = 6


# ============================================================
# Complete-case dataset creation
# ============================================================

TEXT_COLUMNS = [
    "chief_complaint",
    "present_illness",
    "past_medical_record",
    "examination",
]


def has_value(value):

    return (
        pd.notna(value)
        and str(value).strip() != ""
    )


def build_complete_case_dataset():

    print()
    print("=" * 60)
    print("BUILDING COMPLETE-CASE DATASET")
    print("=" * 60)

    print(
        "Source:",
        CSV_SOURCE,
    )

    if not CSV_SOURCE.exists():

        raise FileNotFoundError(
            f"Source dataset not found:\n"
            f"{CSV_SOURCE}"
        )

    df = pd.read_csv(
        CSV_SOURCE
    )

    print(
        "Original rows:",
        len(df),
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "split",
        "checkup_id",
        "patient_id",
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

    # --------------------------------------------------------
    # Same complete-case protocol as Fusion
    # --------------------------------------------------------

    image_available = (
        df["photographs"]
        .apply(has_value)
    )

    radiograph_available = (
        df["radiographs"]
        .apply(has_value)
    )

    text_available = (
        df[TEXT_COLUMNS]
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

    df_complete = (
        df[
            complete_case
        ]
        .reset_index(drop=True)
    )

    print()
    print(
        "Complete-case rows:",
        len(df_complete),
    )

    print(
        "Expected:",
        4195,
    )

    if len(df_complete) != 4195:

        raise RuntimeError(
            "Complete-case count is not 4,195.\n"
            f"Found: {len(df_complete)}\n"
            "STOPPING to avoid running the experiment "
            "on the wrong population."
        )

    # --------------------------------------------------------
    # Split statistics
    # --------------------------------------------------------

    print()
    print("Complete-case split distribution:")

    split_counts = (
        df_complete["split"]
        .value_counts()
        .sort_index()
    )

    for split, count in split_counts.items():

        print(
            f"  {split}: {count}"
        )

    # --------------------------------------------------------
    # Verify every sample has all modalities
    # --------------------------------------------------------

    assert (
        df_complete["photographs"]
        .apply(has_value)
        .all()
    )

    assert (
        df_complete["radiographs"]
        .apply(has_value)
        .all()
    )

    assert (
        df_complete[TEXT_COLUMNS]
        .notna()
        .any(axis=1)
        .all()
    )

    # --------------------------------------------------------
    # Save temporary dataset
    # --------------------------------------------------------

    COMPLETE_CASE_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_complete.to_csv(
        COMPLETE_CASE_CSV,
        index=False,
    )

    print()
    print(
        "Saved complete-case dataset:"
    )

    print(
        COMPLETE_CASE_CSV
    )

    print("=" * 60)

    return df_complete


# ============================================================
# Classifier
# ============================================================

class SSLDownstreamClassifier(nn.Module):

    def __init__(
        self,
        ssl_model,
        modality,
    ):

        super().__init__()

        self.ssl_model = ssl_model
        self.modality = modality

        if modality in [
            "image",
            "radiograph",
        ]:

            input_dim = 2048

        elif modality == "text":

            input_dim = 768

        else:

            raise ValueError(
                f"Unknown modality: {modality}"
            )

        self.classifier = nn.Sequential(

            nn.Linear(
                input_dim,
                512,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                512,
                NUM_LABELS,
            ),
        )

    def forward(
        self,
        x,
    ):

        if self.modality == "image":

            features = (
                self.ssl_model
                .encoders
                .encode_image(x)
            )

        elif self.modality == "radiograph":

            features = (
                self.ssl_model
                .encoders
                .encode_radiograph(x)
            )

        elif self.modality == "text":

            input_ids, attention_mask = x

            features = (
                self.ssl_model
                .encoders
                .encode_text(
                    input_ids,
                    attention_mask,
                )
            )

        else:

            raise ValueError(
                f"Unknown modality: {self.modality}"
            )

        return self.classifier(
            features
        )


# ============================================================
# Configure trainable parameters
# ============================================================

def configure_trainable_parameters(
    model,
    modality,
):

    # --------------------------------------------------------
    # First freeze EVERYTHING in the SSL model.
    # --------------------------------------------------------

    for parameter in model.ssl_model.parameters():

        parameter.requires_grad = False

    # --------------------------------------------------------
    # Unfreeze ONLY the encoder corresponding to the
    # modality currently being evaluated.
    # --------------------------------------------------------

    if modality == "image":

        for parameter in (
            model.ssl_model
            .encoders
            .image_encoder
            .parameters()
        ):

            parameter.requires_grad = True

    elif modality == "radiograph":

        for parameter in (
            model.ssl_model
            .encoders
            .radiograph_encoder
            .parameters()
        ):

            parameter.requires_grad = True

    elif modality == "text":

        for parameter in (
            model.ssl_model
            .encoders
            .text_encoder
            .parameters()
        ):

            parameter.requires_grad = True

    else:

        raise ValueError(
            f"Unknown modality: {modality}"
        )

    # --------------------------------------------------------
    # Classifier is always trainable.
    # --------------------------------------------------------

    for parameter in model.classifier.parameters():

        parameter.requires_grad = True

    # --------------------------------------------------------
    # Collect trainable parameters.
    # --------------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    return trainable_parameters


# ============================================================
# Image loading
# ============================================================

def load_images(
    files,
    modality,
    loader,
    transform,
):

    outputs = []

    for sample in files:

        images = []

        for filename in sample:

            try:

                image = loader.load(
                    filename,
                    modality=modality,
                )

                images.append(
                    transform(image)
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
                f"No valid {modality} image "
                "found in a complete-case sample."
            )

        outputs.append(
            torch.stack(
                images
            ).mean(dim=0)
        )

    return torch.stack(
        outputs
    ).to(DEVICE)


# ============================================================
# Prepare input
# ============================================================

def prepare_input(
    batch,
    modality,
    image_loader,
    transform,
    tokenizer,
):

    if modality == "image":

        return load_images(
            batch["images"],
            "photograph",
            image_loader,
            transform,
        )

    if modality == "radiograph":

        return load_images(
            batch["radiographs"],
            "radiograph",
            image_loader,
            transform,
        )

    if modality == "text":

        tokens = tokenizer(
            batch["text"]
        )

        return (
            tokens["input_ids"].to(
                DEVICE
            ),
            tokens["attention_mask"].to(
                DEVICE
            ),
        )

    raise ValueError(
        f"Unknown modality: {modality}"
    )


# ============================================================
# Build loader
# ============================================================

def build_loader(
    split,
    modality,
    shuffle,
):

    dataset = SSLDownstreamDataset(
        str(COMPLETE_CASE_CSV),
        split=split,
        modality=modality,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(
            DEVICE == "cuda"
        ),
        collate_fn=baseline_collate,
    )

    return dataset, loader


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
    modality,
    image_loader,
    transform,
    tokenizer,
):

    model.eval()

    logits_all = []
    labels_all = []

    with torch.no_grad():

        for batch in tqdm(
            loader,
            desc="Evaluation",
            leave=False,
        ):

            x = prepare_input(
                batch,
                modality,
                image_loader,
                transform,
                tokenizer,
            )

            labels = batch[
                "labels"
            ].to(DEVICE)

            logits = model(x)

            logits_all.append(
                logits.cpu()
            )

            labels_all.append(
                labels.cpu()
            )

    logits = torch.cat(
        logits_all
    )

    labels = torch.cat(
        labels_all
    )

    return compute_metrics(
        logits,
        labels,
        threshold=0.5,
    )


# ============================================================
# Run one modality
# ============================================================

def run_modality(
    modality,
    image_loader,
    transform,
    tokenizer,
):

    print()
    print("=" * 60)
    print(
        f"COMPLETE-CASE SSL DOWNSTREAM — "
        f"{modality.upper()}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_dataset, train_loader = build_loader(
        "train",
        modality,
        shuffle=True,
    )

    validation_dataset, validation_loader = build_loader(
        "validation",
        modality,
        shuffle=False,
    )

    test_dataset, test_loader = build_loader(
        "test",
        modality,
        shuffle=False,
    )

    print()
    print(
        f"Train      : {len(train_dataset)}"
    )

    print(
        f"Validation : {len(validation_dataset)}"
    )

    print(
        f"Test       : {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    total = (
        len(train_dataset)
        +
        len(validation_dataset)
        +
        len(test_dataset)
    )

    print(
        f"Total      : {total}"
    )

    if total != 4195:

        raise RuntimeError(
            f"Expected 4,195 samples across splits, "
            f"found {total}."
        )

    # --------------------------------------------------------
    # SSL model
    # --------------------------------------------------------

    ssl_model = MultimodalSSLModel(
        text_model_name="distilbert-base-uncased"
    )

    checkpoint = torch.load(
        SSL_WEIGHT,
        map_location="cpu",
    )

    ssl_model.load_state_dict(
        checkpoint
    )

    model = SSLDownstreamClassifier(
        ssl_model,
        modality,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Configure fine-tuning
    # --------------------------------------------------------

    if MODE == "fine_tune":

        trainable_parameters = (
            configure_trainable_parameters(
                model=model,
                modality=modality,
            )
        )

    elif MODE == "linear_probe":

        for parameter in (
            model.ssl_model.parameters()
        ):

            parameter.requires_grad = False

        for parameter in (
            model.classifier.parameters()
        ):

            parameter.requires_grad = True

        trainable_parameters = (
            model.classifier.parameters()
        )

    else:

        raise ValueError(
            f"Unknown MODE: {MODE}"
        )

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_auroc = -float("inf")
    best_epoch = -1
    patience_counter = 0

    modality_output = (
        RESULT_ROOT
        / modality
        / MODE
    )

    modality_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        modality_output
        / "best_model.pt"
    )

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        batches = 0

        progress = tqdm(
            train_loader,
            desc=(
                f"{modality} "
                f"Epoch {epoch + 1}/{EPOCHS}"
            ),
        )

        for batch in progress:

            optimizer.zero_grad(
                set_to_none=True
            )

            x = prepare_input(
                batch,
                modality,
                image_loader,
                transform,
                tokenizer,
            )

            labels = batch[
                "labels"
            ].to(DEVICE)

            logits = model(x)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            batches += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            total_loss / batches
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_metrics = evaluate(
            model,
            validation_loader,
            modality,
            image_loader,
            transform,
            tokenizer,
        )

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation": validation_metrics,
        }

        history.append(
            record
        )

        print()
        print(
            f"Epoch {epoch + 1}: "
            f"loss={train_loss:.4f} "
            f"val_AUROC="
            f"{validation_metrics['auroc']:.4f} "
            f"val_macro_F1="
            f"{validation_metrics['macro_f1']:.4f} "
            f"val_micro_F1="
            f"{validation_metrics['micro_f1']:.4f}"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            validation_metrics["auroc"]
            >
            best_auroc
        ):

            best_auroc = (
                validation_metrics["auroc"]
            )

            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                model.state_dict(),
                best_path,
            )

            print(
                "  -> New best checkpoint"
            )

        else:

            patience_counter += 1

            print(
                f"  -> No improvement "
                f"({patience_counter}/{PATIENCE})"
            )

        if patience_counter >= PATIENCE:

            print(
                "Early stopping."
            )

            break

    # --------------------------------------------------------
    # Load best
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            best_path,
            map_location=DEVICE,
        )
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        f"FINAL TEST — {modality.upper()}"
    )
    print("=" * 60)

    test_metrics = evaluate(
        model,
        test_loader,
        modality,
        image_loader,
        transform,
        tokenizer,
    )

    print(
        test_metrics
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        modality_output
        / "final_model.pt",
    )

    result = {
        "experiment": (
            "complete_case_4195"
        ),
        "modality": modality,
        "mode": MODE,
        "dataset": str(
            COMPLETE_CASE_CSV
        ),
        "total_samples": 4195,
        "train_samples": len(
            train_dataset
        ),
        "validation_samples": len(
            validation_dataset
        ),
        "test_samples": len(
            test_dataset
        ),
        "best_epoch": best_epoch,
        "best_validation_auroc": best_auroc,
        "test_metrics": test_metrics,
        "history": history,
    }

    with open(
        modality_output
        / "history.json",
        "w",
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
        )

    with open(
        modality_output
        / "test_metrics.json",
        "w",
    ) as f:

        json.dump(
            test_metrics,
            f,
            indent=4,
        )

    print()
    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation AUROC: "
        f"{best_auroc:.4f}"
    )

    print(
        f"Saved to: {modality_output}"
    )

    return test_metrics


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print(
        "TEST — SSL DOWNSTREAM "
        "ON FUSION COMPLETE-CASE DATASET"
    )
    print("=" * 60)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Source dataset:",
        CSV_SOURCE,
    )

    print(
        "SSL checkpoint:",
        SSL_WEIGHT,
    )

    print(
        "Expected complete-case visits:",
        4195,
    )

    # --------------------------------------------------------
    # Build exactly the population used by Fusion
    # --------------------------------------------------------

    build_complete_case_dataset()

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    image_loader = COdeImageLoader(
        IMAGE_ROOT
    )

    transform = get_image_transform()

    tokenizer = ClinicalTokenizer()

    # --------------------------------------------------------
    # Run all three modalities
    # --------------------------------------------------------

    all_results = {}

    for modality in MODALITIES:

        all_results[modality] = run_modality(
            modality=modality,
            image_loader=image_loader,
            transform=transform,
            tokenizer=tokenizer,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("COMPLETE-CASE SSL DOWNSTREAM SUMMARY")
    print("=" * 60)

    for modality, metrics in all_results.items():

        print()
        print(
            modality.upper()
        )

        print(
            f"Macro F1 : "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Micro F1 : "
            f"{metrics['micro_f1']:.4f}"
        )

        print(
            f"AUROC    : "
            f"{metrics['auroc']:.4f}"
        )

        print(
            f"Accuracy : "
            f"{metrics['accuracy']:.4f}"
        )

    print()
    print("=" * 60)
    print("COMPLETE-CASE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()