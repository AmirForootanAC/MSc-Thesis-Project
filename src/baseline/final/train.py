"""Single entry point for corrected complete-case supervised baselines."""

import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config

from src.baseline.final.collate import complete_case_collate
from src.baseline.final.dataset import CompleteCaseDataset
from src.baseline.final.metrics import (
    compute_metrics,
    optimize_thresholds,
)
from src.baseline.final.model import FinalSupervisedBaseline


def seed_everything(seed):
    """Set all relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_lists(items, device):
    """Move nested image tensors to the target device."""
    return [
        [x.to(device) for x in sample]
        for sample in items
    ]


def forward(model, batch, device):
    """Run a forward pass using only the modalities in the batch."""
    kwargs = {}

    if "images" in batch:
        kwargs["images"] = move_lists(
            batch["images"],
            device,
        )

    if "radiographs" in batch:
        kwargs["radiographs"] = move_lists(
            batch["radiographs"],
            device,
        )

    if "input_ids" in batch:
        kwargs["input_ids"] = batch["input_ids"].to(device)
        kwargs["attention_mask"] = batch[
            "attention_mask"
        ].to(device)

    return model(**kwargs)


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    """Run one training epoch."""
    model.train()

    total_loss = 0.0

    for batch in tqdm(
        loader,
        desc="Training",
        leave=False,
    ):
        labels = batch["labels"].to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = forward(
            model,
            batch,
            device,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def collect(
    model,
    loader,
    criterion,
    device,
    desc,
):
    """
    Collect loss, logits, and labels without gradient computation.

    Used for validation and final test evaluation.
    """
    model.eval()

    total_loss = 0.0
    logits = []
    labels = []

    for batch in tqdm(
        loader,
        desc=desc,
        leave=False,
    ):
        y = batch["labels"].to(device)

        z = forward(
            model,
            batch,
            device,
        )

        loss = criterion(
            z,
            y,
        )

        total_loss += loss.item()

        logits.append(
            z.cpu()
        )
        labels.append(
            y.cpu()
        )

    logits = torch.cat(
        logits,
        dim=0,
    ).numpy()

    labels = torch.cat(
        labels,
        dim=0,
    ).numpy()

    return (
        total_loss / len(loader),
        logits,
        labels,
    )


def make_loader(dataset, shuffle):
    """Create a DataLoader using the final baseline protocol."""
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=complete_case_collate,
    )


def parameter_counts(model):
    """Return total, trainable, and frozen parameter counts."""
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen = total - trainable

    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(frozen),
    }


def config_snapshot():
    """
    Store the relevant experimental configuration inside final_results.json.

    This makes each result self-contained for later analysis and writing.
    """
    return {
        "batch_size": config.BATCH_SIZE,
        "num_epochs": config.NUM_EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "early_stopping_patience": (
            config.EARLY_STOPPING_PATIENCE
        ),
        "default_threshold": config.DEFAULT_THRESHOLD,
        "threshold_min": config.THRESHOLD_MIN,
        "threshold_max": config.THRESHOLD_MAX,
        "threshold_step": config.THRESHOLD_STEP,
        "seed": config.SEED,
        "device": config.DEVICE,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.PIN_MEMORY,
        "text_model_name": config.TEXT_MODEL_NAME,
        "text_max_length": config.TEXT_MAX_LENGTH,
        "text_columns": list(config.TEXT_COLUMNS),
        "image_size": config.IMAGE_SIZE,
        "pretrained_image_encoder": (
            config.PRETRAINED_IMAGE_ENCODER
        ),
        "freeze_image_encoder": (
            config.FREEZE_IMAGE_ENCODER
        ),
        "hidden_dim": 256,
        "loss": "BCEWithLogitsLoss",
    }


def main():
    """Run one complete supervised baseline scenario."""
    seed_everything(config.SEED)

    if (
        config.DEVICE == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is unavailable."
        )

    device = torch.device(
        config.DEVICE
    )

    scenario = config.ACTIVE_SCENARIO
    modalities = config.ACTIVE_MODALITIES

    print(f"\nScenario: {scenario}")
    print(f"Modalities: {modalities}")
    print(f"Device: {device}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Weight decay: {config.WEIGHT_DECAY}")
    print(
        "Pretrained image encoder: "
        f"{config.PRETRAINED_IMAGE_ENCODER}"
    )
    print(
        "Freeze image encoder: "
        f"{config.FREEZE_IMAGE_ENCODER}"
    )

    print("\nCheckpoint selection:")
    print(
        "Validation Macro F1 @ threshold 0.5"
    )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    train_ds = CompleteCaseDataset(
        config.DATASET_PATH,
        "train",
        config.IMAGE_ROOT,
        modalities,
    )

    val_ds = CompleteCaseDataset(
        config.DATASET_PATH,
        "validation",
        config.IMAGE_ROOT,
        modalities,
    )

    test_ds = CompleteCaseDataset(
        config.DATASET_PATH,
        "test",
        config.IMAGE_ROOT,
        modalities,
    )

    print(
        f"\nTrain samples: {len(train_ds)}"
    )
    print(
        f"Validation samples: {len(val_ds)}"
    )
    print(
        f"Test samples: {len(test_ds)}"
    )

    if len(train_ds) != config.EXPECTED_COUNTS["train"]:
        raise RuntimeError(
            "Unexpected train sample count: "
            f"{len(train_ds)} != "
            f"{config.EXPECTED_COUNTS['train']}"
        )

    if len(val_ds) != config.EXPECTED_COUNTS["validation"]:
        raise RuntimeError(
            "Unexpected validation sample count: "
            f"{len(val_ds)} != "
            f"{config.EXPECTED_COUNTS['validation']}"
        )

    if len(test_ds) != config.EXPECTED_COUNTS["test"]:
        raise RuntimeError(
            "Unexpected test sample count: "
            f"{len(test_ds)} != "
            f"{config.EXPECTED_COUNTS['test']}"
        )

    train_dl = make_loader(
        train_ds,
        shuffle=True,
    )

    val_dl = make_loader(
        val_ds,
        shuffle=False,
    )

    test_dl = make_loader(
        test_ds,
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model = FinalSupervisedBaseline(
        modalities,
        config.TEXT_MODEL_NAME,
        config.NUM_LABELS,
        config.PRETRAINED_IMAGE_ENCODER,
        config.FREEZE_IMAGE_ENCODER,
        hidden_dim=256,
    ).to(device)

    params = parameter_counts(model)

    print("\nParameter counts:")
    print(
        f"Total:     {params['total']:,}"
    )
    print(
        f"Trainable: {params['trainable']:,}"
    )
    print(
        f"Frozen:    {params['frozen']:,}"
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------

    out = (
        config.RESULT_ROOT
        / scenario
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = (
        out / "best_model.pt"
    )

    history_path = (
        out / "history.json"
    )

    results_path = (
        out / "final_results.json"
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    history = []

    best_val_macro_f1 = -1.0
    best_val_metrics = None
    best_epoch = None
    patience = 0

    for epoch in range(
        config.NUM_EPOCHS
    ):
        epoch_number = epoch + 1

        train_loss = train_epoch(
            model,
            train_dl,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_logits, val_labels = collect(
            model,
            val_dl,
            criterion,
            device,
            "Validation",
        )

        val_metrics = compute_metrics(
            val_logits,
            val_labels,
            threshold=config.DEFAULT_THRESHOLD,
            label_names=config.LABEL_NAMES,
        )

        print(
            f"Epoch {epoch_number}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro F1: "
            f"{val_metrics['macro_f1']:.4f} | "
            f"Val Micro F1: "
            f"{val_metrics['micro_f1']:.4f} | "
            f"Val AUROC: "
            f"{val_metrics['auroc']:.4f} | "
            f"Val Accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )

        record = {
            "epoch": epoch_number,
            "train_loss": float(train_loss),
            "validation_loss": float(val_loss),
            "validation_metrics": val_metrics,
        }

        history.append(record)

        current_macro_f1 = (
            val_metrics["macro_f1"]
        )

        if current_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = (
                current_macro_f1
            )

            best_val_metrics = val_metrics
            best_epoch = epoch_number
            patience = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch_number,
                    "best_val_macro_f1": (
                        best_val_macro_f1
                    ),
                    "best_val_metrics": (
                        best_val_metrics
                    ),
                    "scenario": scenario,
                    "modalities": list(
                        modalities
                    ),
                    "seed": config.SEED,
                    "hidden_dim": 256,
                    "config": config_snapshot(),
                },
                checkpoint,
            )

            print(
                "✓ Saved best checkpoint "
                f"(Val Macro F1: "
                f"{best_val_macro_f1:.4f})"
            )

        else:
            patience += 1

            print(
                "No improvement "
                f"({patience}/"
                f"{config.EARLY_STOPPING_PATIENCE})"
            )

        # Save history after every epoch so that the latest
        # completed epoch is preserved even if execution stops.
        history_path.write_text(
            json.dumps(
                history,
                indent=2,
            )
        )

        if (
            patience
            >= config.EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping triggered."
            )
            break

    # ------------------------------------------------------------------
    # Load best checkpoint
    # ------------------------------------------------------------------

    state = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        state["model_state"]
    )

    print("\n" + "=" * 60)
    print("Best checkpoint loaded")
    print(
        f"Checkpoint epoch: "
        f"{state['epoch']}"
    )
    print(
        "Best validation Macro F1 @ 0.5: "
        f"{state['best_val_macro_f1']:.4f}"
    )
    print("=" * 60)

    # ------------------------------------------------------------------
    # Final validation and test evaluation
    #
    # Test is evaluated ONLY after checkpoint selection.
    # ------------------------------------------------------------------

    _, val_logits, val_labels = collect(
        model,
        val_dl,
        criterion,
        device,
        "Final validation",
    )

    _, test_logits, test_labels = collect(
        model,
        test_dl,
        criterion,
        device,
        "Test",
    )

    # Thresholds are learned from validation only.
    thresholds = optimize_thresholds(
        val_logits,
        val_labels,
        config.THRESHOLD_MIN,
        config.THRESHOLD_MAX,
        config.THRESHOLD_STEP,
    )

    final_validation_default = compute_metrics(
        val_logits,
        val_labels,
        threshold=config.DEFAULT_THRESHOLD,
        label_names=config.LABEL_NAMES,
    )

    test_default = compute_metrics(
        test_logits,
        test_labels,
        threshold=config.DEFAULT_THRESHOLD,
        label_names=config.LABEL_NAMES,
    )

    test_optimized = compute_metrics(
        test_logits,
        test_labels,
        threshold=thresholds,
        label_names=config.LABEL_NAMES,
    )

    # ------------------------------------------------------------------
    # Final result artifact
    # ------------------------------------------------------------------

    result = {
        "scenario": scenario,
        "modalities": list(modalities),
        "labels": list(config.LABEL_NAMES),
        "seed": config.SEED,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "test_samples": len(test_ds),
        "sample_counts_expected": dict(
            config.EXPECTED_COUNTS
        ),
        "loss": "BCEWithLogitsLoss",
        "checkpoint_selection": (
            "validation_macro_f1_at_threshold_0.5"
        ),
        "checkpoint_epoch": int(
            state["epoch"]
        ),
        "best_validation_macro_f1": float(
            state["best_val_macro_f1"]
        ),
        "best_validation_metrics": (
            state["best_val_metrics"]
        ),
        "final_validation_default": (
            final_validation_default
        ),
        "validation_optimized_thresholds": (
            thresholds.tolist()
        ),
        "test_default": test_default,
        "test_optimized": test_optimized,
        "parameter_counts": params,
        "config": config_snapshot(),
    }

    results_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    # ------------------------------------------------------------------
    # Final console summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print("\nThreshold = 0.5")
    print(
        f"Macro F1: "
        f"{test_default['macro_f1']:.4f}"
    )
    print(
        f"Micro F1: "
        f"{test_default['micro_f1']:.4f}"
    )
    print(
        f"AUROC:    "
        f"{test_default['auroc']:.4f}"
    )
    print(
        f"Accuracy: "
        f"{test_default['accuracy']:.4f}"
    )

    print("\nValidation-optimized thresholds")
    print(
        thresholds.tolist()
    )
    print(
        f"Macro F1: "
        f"{test_optimized['macro_f1']:.4f}"
    )
    print(
        f"Micro F1: "
        f"{test_optimized['micro_f1']:.4f}"
    )
    print(
        f"AUROC:    "
        f"{test_optimized['auroc']:.4f}"
    )
    print(
        f"Accuracy: "
        f"{test_optimized['accuracy']:.4f}"
    )

    print(
        f"\nSaved history: "
        f"{history_path}"
    )
    print(
        f"Saved results: "
        f"{results_path}"
    )
    print(
        f"Saved checkpoint: "
        f"{checkpoint}"
    )


if __name__ == "__main__":
    main()