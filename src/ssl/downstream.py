"""
SSL downstream classification.

Modes:
    - linear_probe
    - fine_tune

Modalities:
    - image
    - radiograph
    - text

Protocol:

    Train
       |
       v
    Validation
       |
       | select best checkpoint
       v
    Test
"""

import json
from pathlib import Path

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
# CONFIG
# ============================================================

MODALITY = "radiograph"
# image
# radiograph
# text

MODE = "fine_tune"
# linear_probe
# fine_tune


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


BATCH_SIZE = 16

EPOCHS = 20

# Fine-tuning should use a smaller LR than classifier-only training.
LR = 1e-5

WEIGHT_DECAY = 1e-4

PATIENCE = 5


CSV = (
    "results/"
    "six_label_patient_level_dataset/"
    "labeled_dataset.csv"
)


SSL_WEIGHT = (
    "results/"
    "ssl_pretraining/"
    "multimodal_dynamic/"
    "best_ssl_model.pt"
)


OUT = (
    Path("results")
    / "ssl_pretraining"
    / "downstream"
    / MODALITY
    / MODE
)


OUT.mkdir(
    parents=True,
    exist_ok=True,
)


NUM_LABELS = 6


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

        return self.classifier(features)


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

        imgs = []

        for filename in sample:

            try:

                img = loader.load(
                    filename,
                    modality=modality,
                )

                imgs.append(
                    transform(img)
                )

            except Exception:

                continue

        if imgs:

            outputs.append(
                torch.stack(imgs).mean(0)
            )

        else:

            # Should almost never happen because the dataset
            # already guarantees modality availability.
            outputs.append(
                torch.zeros(
                    3,
                    224,
                    224,
                )
            )

    return torch.stack(
        outputs
    ).to(DEVICE)


# ============================================================
# Prepare input
# ============================================================

def prepare_input(
    batch,
    image_loader,
    transform,
    tokenizer,
):

    if MODALITY == "image":

        return load_images(
            batch["images"],
            "photograph",
            image_loader,
            transform,
        )

    if MODALITY == "radiograph":

        return load_images(
            batch["radiographs"],
            "radiograph",
            image_loader,
            transform,
        )

    if MODALITY == "text":

        tokens = tokenizer(
            batch["text"]
        )

        return (
            tokens["input_ids"].to(DEVICE),
            tokens["attention_mask"].to(DEVICE),
        )

    raise ValueError(
        f"Unknown modality: {MODALITY}"
    )


# ============================================================
# Build loader
# ============================================================

def build_loader(
    split,
    shuffle,
):

    dataset = SSLDownstreamDataset(
        CSV,
        split=split,
        modality=MODALITY,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        collate_fn=baseline_collate,
    )

    return dataset, loader


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
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
                image_loader,
                transform,
                tokenizer,
            )

            labels = batch["labels"].to(
                DEVICE
            )

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

    metrics = compute_metrics(
        logits,
        labels,
        threshold=0.5,
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("SSL DOWNSTREAM")
    print("=" * 60)

    print(
        f"Modality : {MODALITY}"
    )

    print(
        f"Mode     : {MODE}"
    )

    print(
        f"Device   : {DEVICE}"
    )

    print(
        f"Labels   : {config.LABEL_NAMES}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_dataset, train_loader = build_loader(
        "train",
        shuffle=True,
    )

    val_dataset, val_loader = build_loader(
        "validation",
        shuffle=False,
    )

    test_dataset, test_loader = build_loader(
        "test",
        shuffle=False,
    )

    print()
    print(
        f"Train      : {len(train_dataset)}"
    )

    print(
        f"Validation : {len(val_dataset)}"
    )

    print(
        f"Test       : {len(test_dataset)}"
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
        MODALITY,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Freeze / fine-tune
    # --------------------------------------------------------

    if MODE == "linear_probe":

        for parameter in model.ssl_model.parameters():

            parameter.requires_grad = False

        trainable_parameters = (
            model.classifier.parameters()
        )

    elif MODE == "fine_tune":

        # Enable encoder fine-tuning.
        for parameter in model.ssl_model.encoders.parameters():

            parameter.requires_grad = True

        # Projection heads are not used by downstream
        # classification, so keep them frozen.
        for parameter in model.ssl_model.projectors.parameters():

            parameter.requires_grad = False

        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

    else:

        raise ValueError(
            f"Unknown MODE: {MODE}"
        )

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    print(
        f"Trainable parameters: {trainable_count:,}"
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
    # Utilities
    # --------------------------------------------------------

    image_loader = COdeImageLoader(
        "data/raw/COde-Dataset/Images"
    )

    transform = get_image_transform()

    tokenizer = ClinicalTokenizer()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_auroc = -float("inf")
    best_epoch = -1
    patience_counter = 0

    best_path = (
        OUT / "best_model.pt"
    )

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
        )

        for batch in progress:

            optimizer.zero_grad(
                set_to_none=True
            )

            x = prepare_input(
                batch,
                image_loader,
                transform,
                tokenizer,
            )

            labels = batch["labels"].to(
                DEVICE
            )

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

        val_metrics = evaluate(
            model,
            val_loader,
            image_loader,
            transform,
            tokenizer,
        )

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation": val_metrics,
        }

        history.append(
            record
        )

        print()
        print(
            f"Epoch {epoch + 1}: "
            f"loss={train_loss:.4f} "
            f"val_AUROC={val_metrics['auroc']:.4f} "
            f"val_macro_F1={val_metrics['macro_f1']:.4f} "
            f"val_micro_F1={val_metrics['micro_f1']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f}"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_metrics["auroc"] > best_auroc:

            best_auroc = (
                val_metrics["auroc"]
            )

            best_epoch = epoch + 1

            patience_counter = 0

            torch.save(
                model.state_dict(),
                best_path,
            )

            print(
                f"  -> New best checkpoint "
                f"(AUROC={best_auroc:.4f})"
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
    # Load best checkpoint
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
    print("FINAL TEST")
    print("=" * 60)

    test_metrics = evaluate(
        model,
        test_loader,
        image_loader,
        transform,
        tokenizer,
    )

    print(
        test_metrics
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        OUT / "final_model.pt",
    )

    with open(
        OUT / "history.json",
        "w",
    ) as f:

        json.dump(
            {
                "modality": MODALITY,
                "mode": MODE,
                "best_epoch": best_epoch,
                "best_validation_auroc": best_auroc,
                "test_metrics": test_metrics,
                "history": history,
            },
            f,
            indent=4,
        )

    with open(
        OUT / "test_metrics.json",
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
        f"Saved to: {OUT}"
    )


if __name__ == "__main__":
    main()