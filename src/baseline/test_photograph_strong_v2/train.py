"""Strong photograph-only ResNet50 experiment - version 2."""

import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.baseline.final.dataset import CompleteCaseDataset
from src.baseline.final.metrics import (
    compute_metrics,
    optimize_thresholds,
    per_label_metrics,
)

from . import config
from .loss import ClassWeightedBCEWithLogitsLoss
from .model import MaxStrengthPhotographResNet50
from .transforms import (
    get_eval_transform,
    get_train_transform,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def amp_enabled(device):
    return device.type == "cuda"


def create_grad_scaler(device):
    if not amp_enabled(device):
        return torch.amp.GradScaler(
            "cpu",
            enabled=False,
        )

    return torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )


def build_datasets():
    train_dataset = CompleteCaseDataset(
        csv_path=config.DATASET_PATH,
        image_root=config.IMAGE_ROOT,
        split="train",
        modalities=config.MODALITIES,
        transform=get_train_transform(
            image_size=config.IMAGE_SIZE
        ),
    )

    validation_dataset = CompleteCaseDataset(
        csv_path=config.DATASET_PATH,
        image_root=config.IMAGE_ROOT,
        split="validation",
        modalities=config.MODALITIES,
        transform=get_eval_transform(
            image_size=config.IMAGE_SIZE
        ),
    )

    test_dataset = CompleteCaseDataset(
        csv_path=config.DATASET_PATH,
        image_root=config.IMAGE_ROOT,
        split="test",
        modalities=config.MODALITIES,
        transform=get_eval_transform(
            image_size=config.IMAGE_SIZE
        ),
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def photograph_collate_fn(batch):
    return {
        "photographs": [
            sample["images"]
            for sample in batch
        ],
        "labels": torch.stack(
            [
                sample["labels"]
                for sample in batch
            ]
        ),
        "patient_ids": [
            sample.get("patient_id")
            for sample in batch
        ],
        "checkup_ids": [
            sample.get("checkup_id")
            for sample in batch
        ],
    }


def move_photographs_to_device(
    photographs,
    device,
):
    moved = []

    for image_list in photographs:

        if len(image_list) == 0:
            raise RuntimeError(
                "Encountered a sample with zero photographs."
            )

        moved_images = [
            image.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            for image in image_list
        ]

        moved.append(
            moved_images
        )

    return moved


def horizontally_flip_photographs(
    photographs,
):
    flipped = []

    for image_list in photographs:

        flipped_list = [
            image.flip(
                dims=[2]
            )
            for image in image_list
        ]

        flipped.append(
            flipped_list
        )

    return flipped


def get_training_labels(dataset):
    if hasattr(dataset, "df"):
        return dataset.df[
            config.LABEL_NAMES
        ].to_numpy(
            dtype=np.float32
        )

    labels = []

    for index in range(len(dataset)):
        sample = dataset[index]

        labels.append(
            sample["labels"].cpu().numpy()
        )

    return np.asarray(
        labels,
        dtype=np.float32,
    )


def compute_pos_weights(labels):
    labels = np.asarray(
        labels,
        dtype=np.float32,
    )

    positive_counts = labels.sum(
        axis=0
    )

    negative_counts = (
        labels.shape[0]
        - positive_counts
    )

    positive_counts = np.maximum(
        positive_counts,
        1.0,
    )

    weights = (
        negative_counts
        / positive_counts
    )

    return weights.astype(
        np.float32
    )


def build_criterion(
    train_labels,
    device,
):
    pos_weights = compute_pos_weights(
        train_labels
    )

    print("\nClass positive weights:")

    for name, weight in zip(
        config.LABEL_NAMES,
        pos_weights,
    ):
        print(
            f"  {name:<28} {weight:.4f}"
        )

    criterion = ClassWeightedBCEWithLogitsLoss(
        pos_weights=pos_weights
    )

    return criterion.to(device)


def build_model():
    return MaxStrengthPhotographResNet50(
        num_labels=config.NUM_LABELS,
        pretrained=config.PRETRAINED,
        dropout=config.DROPOUT,
    )


def freeze_batchnorm_statistics(model):
    for module in model.modules():

        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
            ),
        ):
            module.eval()

            if module.weight is not None:
                module.weight.requires_grad = True

            if module.bias is not None:
                module.bias.requires_grad = True


def build_optimizer(model):
    backbone_parameters = []
    head_parameters = []

    backbone_modules = [
        model.layer0,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    ]

    head_modules = [
        model.cbam,
        model.classifier,
    ]

    for module in backbone_modules:

        backbone_parameters.extend(
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    for module in head_modules:

        head_parameters.extend(
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    return AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": config.BACKBONE_LR,
            },
            {
                "params": head_parameters,
                "lr": config.CLASSIFIER_LR,
            },
        ],
        weight_decay=config.WEIGHT_DECAY,
    )


def set_warmup_learning_rate(
    optimizer,
    epoch_index,
):
    if config.WARMUP_EPOCHS <= 0:
        return

    progress = (
        epoch_index + 1
    ) / config.WARMUP_EPOCHS

    progress = min(
        progress,
        1.0,
    )

    optimizer.param_groups[0]["lr"] = (
        config.BACKBONE_LR
        * progress
    )

    optimizer.param_groups[1]["lr"] = (
        config.CLASSIFIER_LR
        * progress
    )


def current_learning_rates(
    optimizer
):
    return [
        group["lr"]
        for group in optimizer.param_groups
    ]


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
):
    model.train()

    freeze_batchnorm_statistics(model)

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0
    sample_count = 0

    all_logits = []
    all_labels = []

    num_batches = len(loader)

    progress_bar = tqdm(
        loader,
        total=num_batches,
        desc=f"Epoch {epoch:02d} [Train]",
        leave=True,
        dynamic_ncols=True,
    )

    for batch_index, batch in enumerate(
        progress_bar
    ):

        labels = batch["labels"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        photographs = (
            move_photographs_to_device(
                batch["photographs"],
                device,
            )
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled(device),
        ):

            logits = model(
                photographs
            )

            if logits.ndim != 2:
                raise RuntimeError(
                    "Expected model output with "
                    f"shape [batch, labels], got "
                    f"{tuple(logits.shape)}"
                )

            if logits.shape != labels.shape:
                raise RuntimeError(
                    "Logit/label shape mismatch: "
                    f"logits={tuple(logits.shape)}, "
                    f"labels={tuple(labels.shape)}"
                )

            loss = criterion(
                logits,
                labels,
            )

            scaled_loss = (
                loss
                / config.GRADIENT_ACCUMULATION_STEPS
            )

        scaler.scale(
            scaled_loss
        ).backward()

        should_step = (
            (
                batch_index + 1
            )
            % config.GRADIENT_ACCUMULATION_STEPS
            == 0
            or
            (
                batch_index + 1
            )
            == num_batches
        )

        if should_step:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.GRADIENT_CLIP_NORM,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        batch_size = labels.shape[0]

        total_loss += (
            loss.detach().item()
            * batch_size
        )

        sample_count += batch_size

        all_logits.append(
            logits.detach()
            .float()
            .cpu()
            .numpy()
        )

        all_labels.append(
            labels.detach()
            .cpu()
            .numpy()
        )

        running_loss = (
            total_loss
            / max(
                sample_count,
                1,
            )
        )

        progress_bar.set_postfix(
            loss=f"{running_loss:.4f}"
        )

    logits_np = np.concatenate(
        all_logits,
        axis=0,
    )

    labels_np = np.concatenate(
        all_labels,
        axis=0,
    )

    metrics = compute_metrics(
        logits_np,
        labels_np,
        threshold=config.DEFAULT_THRESHOLD,
        label_names=config.LABEL_NAMES,
    )

    average_loss = (
        total_loss
        / max(
            sample_count,
            1,
        )
    )

    return (
        average_loss,
        metrics,
        logits_np,
        labels_np,
    )


def probabilities_to_logits(
    probabilities
):
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    probabilities = np.clip(
        probabilities,
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(
        probabilities
        / (
            1.0
            - probabilities
        )
    )


@torch.no_grad()
def predict(
    model,
    loader,
    device,
    use_tta=False,
    description="[Eval]",
):
    model.eval()

    all_labels = []
    all_logits = []

    progress_bar = tqdm(
        loader,
        total=len(loader),
        desc=description,
        leave=True,
        dynamic_ncols=True,
    )

    for batch in progress_bar:

        labels = batch["labels"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        photographs = (
            move_photographs_to_device(
                batch["photographs"],
                device,
            )
        )

        logits = model(
            photographs
        )

        if logits.shape != labels.shape:
            raise RuntimeError(
                "Logit/label shape mismatch during "
                "evaluation: "
                f"logits={tuple(logits.shape)}, "
                f"labels={tuple(labels.shape)}"
            )

        if not use_tta:

            final_logits = logits

        else:

            flipped_photographs = (
                horizontally_flip_photographs(
                    photographs
                )
            )

            flipped_logits = model(
                flipped_photographs
            )

            original_probabilities = (
                torch.sigmoid(
                    logits
                )
            )

            flipped_probabilities = (
                torch.sigmoid(
                    flipped_logits
                )
            )

            mean_probabilities = (
                original_probabilities
                + flipped_probabilities
            ) / 2.0

            final_logits = torch.from_numpy(
                probabilities_to_logits(
                    mean_probabilities
                    .detach()
                    .cpu()
                    .numpy()
                )
            ).to(
                device=device,
                dtype=torch.float32,
            )

        all_logits.append(
            final_logits.detach()
            .float()
            .cpu()
            .numpy()
        )

        all_labels.append(
            labels.detach()
            .cpu()
            .numpy()
        )

    logits_np = np.concatenate(
        all_logits,
        axis=0,
    )

    labels_np = np.concatenate(
        all_labels,
        axis=0,
    )

    return (
        logits_np,
        labels_np,
    )


def reset_output_directory():

    if config.RESULT_ROOT.exists():
        shutil.rmtree(
            config.RESULT_ROOT
        )

    (
        config.RESULT_ROOT
        / "checkpoints"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


def checkpoint_path(
    epoch
):
    return (
        config.RESULT_ROOT
        / "checkpoints"
        / f"epoch_{epoch:03d}.pt"
    )


def save_checkpoint(
    path,
    model,
    epoch,
    val_macro_f1,
):
    torch.save(
        {
            "epoch": int(epoch),
            "val_macro_f1": float(
                val_macro_f1
            ),
            "model_state_dict":
                model.state_dict(),
        },
        path,
    )


def update_top_k(
    top_k,
    path,
    score,
):
    record = {
        "path": str(path),
        "epoch": int(
            path.stem.split("_")[-1]
        ),
        "score": float(score),
    }

    top_k.append(
        record
    )

    top_k.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    while (
        len(top_k)
        > config.TOP_K_CHECKPOINTS
    ):

        removed = top_k.pop()

        removed_path = Path(
            removed["path"]
        )

        if removed_path.exists():
            removed_path.unlink()

    return top_k


def load_checkpoint(
    model,
    path,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    return checkpoint


def main():

    set_seed(
        config.SEED
    )

    device = get_device()

    print("=" * 70)
    print(
        "STRONG PHOTOGRAPH-ONLY EXPERIMENT v2"
    )
    print("=" * 70)

    print(
        f"Dataset: {config.DATASET_PATH}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Image size: {config.IMAGE_SIZE}"
    )

    print(
        f"Batch size: {config.BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{config.GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Backbone LR: "
        f"{config.BACKBONE_LR}"
    )

    print(
        f"Classifier LR: "
        f"{config.CLASSIFIER_LR}"
    )

    print(
        f"Weight decay: "
        f"{config.WEIGHT_DECAY}"
    )

    print(
        "Loss: "
        "Class-weighted BCEWithLogitsLoss"
    )

    print(
        "Weighted sampler: False"
    )

    print(
        "CBAM: True"
    )

    print(
        "RGB augmentation: "
        "brightness=0.12, "
        "contrast=0.12, "
        "saturation=0.12, "
        "hue=0.03"
    )

    print(
        f"TTA: {config.USE_TTA}"
    )

    print(
        f"Top-K checkpoints: "
        f"{config.TOP_K_CHECKPOINTS}"
    )

    print(
        "BatchNorm statistics frozen: True"
    )

    print("=" * 70)

    reset_output_directory()

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = build_datasets()

    expected = config.EXPECTED_COUNTS

    if len(train_dataset) != expected["train"]:
        raise RuntimeError(
            "Unexpected train count: "
            f"expected={expected['train']}, "
            f"found={len(train_dataset)}"
        )

    if (
        len(validation_dataset)
        != expected["validation"]
    ):
        raise RuntimeError(
            "Unexpected validation count: "
            f"expected={expected['validation']}, "
            f"found={len(validation_dataset)}"
        )

    if len(test_dataset) != expected["test"]:
        raise RuntimeError(
            "Unexpected test count: "
            f"expected={expected['test']}, "
            f"found={len(test_dataset)}"
        )

    print("\nDataset counts:")
    print(
        f"Train:      {len(train_dataset)}"
    )
    print(
        f"Validation: {len(validation_dataset)}"
    )
    print(
        f"Test:       {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=photograph_collate_fn,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=photograph_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=photograph_collate_fn,
    )

    train_labels = get_training_labels(
        train_dataset
    )

    criterion = build_criterion(
        train_labels,
        device,
    )

    model = build_model().to(
        device
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = (
        total_parameters
        - trainable_parameters
    )

    print("\nModel parameters:")

    print(
        f"Total:     "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Frozen:    "
        f"{frozen_parameters:,}"
    )

    optimizer = build_optimizer(
        model
    )

    cosine_epochs = max(
        config.NUM_EPOCHS
        - config.WARMUP_EPOCHS,
        1,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=1e-7,
    )

    scaler = create_grad_scaler(
        device
    )

    best_val_macro_f1 = -math.inf
    best_epoch = None
    patience_counter = 0
    top_k = []
    history = []

    start_time = time.time()

    for epoch_index in range(
        config.NUM_EPOCHS
    ):

        epoch = (
            epoch_index
            + 1
        )

        if (
            epoch
            <= config.WARMUP_EPOCHS
        ):

            set_warmup_learning_rate(
                optimizer,
                epoch_index,
            )

        (
            train_loss,
            train_metrics,
            _,
            _,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )

        model.eval()

        validation_loss = 0.0
        validation_samples = 0
        validation_logits = []
        validation_labels = []

        progress_bar = tqdm(
            validation_loader,
            total=len(
                validation_loader
            ),
            desc=f"Epoch {epoch:02d} [Val]",
            leave=True,
            dynamic_ncols=True,
        )

        with torch.no_grad():

            for batch in progress_bar:

                labels = batch[
                    "labels"
                ].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                photographs = (
                    move_photographs_to_device(
                        batch["photographs"],
                        device,
                    )
                )

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=amp_enabled(
                        device
                    ),
                ):

                    logits = model(
                        photographs
                    )

                    if logits.shape != labels.shape:
                        raise RuntimeError(
                            "Logit/label shape mismatch "
                            "during validation: "
                            f"logits={tuple(logits.shape)}, "
                            f"labels={tuple(labels.shape)}"
                        )

                    loss = criterion(
                        logits,
                        labels,
                    )

                batch_size = labels.shape[0]

                validation_loss += (
                    loss.detach().item()
                    * batch_size
                )

                validation_samples += (
                    batch_size
                )

                validation_logits.append(
                    logits.detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                validation_labels.append(
                    labels.detach()
                    .cpu()
                    .numpy()
                )

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        validation_logits_np = (
            np.concatenate(
                validation_logits,
                axis=0,
            )
        )

        validation_labels_np = (
            np.concatenate(
                validation_labels,
                axis=0,
            )
        )

        validation_loss /= max(
            validation_samples,
            1,
        )

        validation_metrics = compute_metrics(
            validation_logits_np,
            validation_labels_np,
            threshold=config.DEFAULT_THRESHOLD,
            label_names=config.LABEL_NAMES,
        )

        current_lrs = current_learning_rates(
            optimizer
        )

        if epoch > config.WARMUP_EPOCHS:
            scheduler.step()

        val_macro_f1 = float(
            validation_metrics[
                "macro_f1"
            ]
        )

        checkpoint_file = checkpoint_path(
            epoch
        )

        save_checkpoint(
            checkpoint_file,
            model,
            epoch,
            val_macro_f1,
        )

        top_k = update_top_k(
            top_k,
            checkpoint_file,
            val_macro_f1,
        )

        if (
            val_macro_f1
            > best_val_macro_f1
        ):

            best_val_macro_f1 = (
                val_macro_f1
            )

            best_epoch = epoch

            patience_counter = 0

            print(
                f"  -> New best validation "
                f"Macro F1: "
                f"{best_val_macro_f1:.4f}"
            )

        else:

            patience_counter += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_macro_f1": float(
                    train_metrics["macro_f1"]
                ),
                "train_micro_f1": float(
                    train_metrics["micro_f1"]
                ),
                "train_auroc": float(
                    train_metrics["auroc"]
                ),
                "val_loss": float(validation_loss),
                "val_macro_f1": float(
                    validation_metrics["macro_f1"]
                ),
                "val_micro_f1": float(
                    validation_metrics["micro_f1"]
                ),
                "val_auroc": float(
                    validation_metrics["auroc"]
                ),
                "val_accuracy": float(
                    validation_metrics["accuracy"]
                ),
                "backbone_lr": float(
                    current_lrs[0]
                ),
                "classifier_lr": float(
                    current_lrs[1]
                ),
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Macro F1 "
            f"{train_metrics['macro_f1']:.4f} | "
            f"Val Loss {validation_loss:.4f} | "
            f"Val Macro F1 "
            f"{validation_metrics['macro_f1']:.4f} | "
            f"Val AUROC "
            f"{validation_metrics['auroc']:.4f}"
        )

        print(
            f"  LR: "
            f"backbone={current_lrs[0]:.2e} "
            f"classifier={current_lrs[1]:.2e} | "
            f"patience="
            f"{patience_counter}/"
            f"{config.EARLY_STOPPING_PATIENCE}"
        )

        if (
            patience_counter
            >= config.EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    if best_epoch is None:
        raise RuntimeError(
            "Training finished without "
            "producing a best checkpoint."
        )

    best_checkpoint = None

    for item in top_k:

        if (
            item["epoch"]
            == best_epoch
        ):

            best_checkpoint = Path(
                item["path"]
            )

            break

    if best_checkpoint is None:
        raise RuntimeError(
            "Best checkpoint was not found "
            "inside the retained top-K checkpoints."
        )

    checkpoint = load_checkpoint(
        model,
        best_checkpoint,
        device,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BEST CHECKPOINT"
    )

    print(
        "=" * 70
    )

    print(
        f"Epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation Macro F1: "
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    validation_logits, validation_labels = predict(
        model,
        validation_loader,
        device,
        use_tta=config.USE_TTA,
        description="[Validation TTA]",
    )

    validation_default = compute_metrics(
        validation_logits,
        validation_labels,
        threshold=config.DEFAULT_THRESHOLD,
        label_names=config.LABEL_NAMES,
    )

    optimized_thresholds = optimize_thresholds(
        validation_logits,
        validation_labels,
        minimum=config.THRESHOLD_MIN,
        maximum=config.THRESHOLD_MAX,
        step=config.THRESHOLD_STEP,
    )

    validation_optimized = compute_metrics(
        validation_logits,
        validation_labels,
        threshold=optimized_thresholds,
        label_names=config.LABEL_NAMES,
    )

    test_logits, test_labels = predict(
        model,
        test_loader,
        device,
        use_tta=config.USE_TTA,
        description="[Test TTA]",
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
        threshold=optimized_thresholds,
        label_names=config.LABEL_NAMES,
    )

    test_per_label = per_label_metrics(
        test_logits,
        test_labels,
        threshold=config.DEFAULT_THRESHOLD,
        label_names=config.LABEL_NAMES,
    )

    elapsed_seconds = (
        time.time()
        - start_time
    )

    results = {
        "experiment":
            "photograph_only_max_strength_v2",

        "dataset":
            str(config.DATASET_PATH),

        "modalities":
            list(config.MODALITIES),

        "image_size":
            config.IMAGE_SIZE,

        "batch_size":
            config.BATCH_SIZE,

        "gradient_accumulation_steps":
            config.GRADIENT_ACCUMULATION_STEPS,

        "backbone_lr":
            config.BACKBONE_LR,

        "classifier_lr":
            config.CLASSIFIER_LR,

        "weight_decay":
            config.WEIGHT_DECAY,

        "loss":
            "class_weighted_bce_with_logits",

        "weighted_sampler":
            False,

        "cbam":
            True,

        "rgb_augmentation": {
            "brightness": 0.12,
            "contrast": 0.12,
            "saturation": 0.12,
            "hue": 0.03,
        },

        "batchnorm_statistics_frozen":
            True,

        "tta":
            config.USE_TTA,

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            float(best_val_macro_f1),

        "test_default":
            test_default,

        "validation_default":
            validation_default,

        "validation_optimized":
            validation_optimized,

        "test_optimized":
            test_optimized,

        "optimized_thresholds":
            np.asarray(
                optimized_thresholds
            ).tolist(),

        "test_per_label":
            test_per_label,

        "top_k_checkpoints":
            top_k,

        "history":
            history,

        "elapsed_seconds":
            float(elapsed_seconds),
    }

    result_path = (
        config.RESULT_ROOT
        / "final_results.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Macro F1 @ 0.5: "
        f"{test_default['macro_f1']:.4f}"
    )

    print(
        f"Micro F1 @ 0.5: "
        f"{test_default['micro_f1']:.4f}"
    )

    print(
        f"AUROC: "
        f"{test_default['auroc']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{test_default['accuracy']:.4f}"
    )

    print(
        f"\nOptimized Macro F1: "
        f"{test_optimized['macro_f1']:.4f}"
    )

    print(
        "Optimized thresholds: "
        f"{np.asarray(optimized_thresholds).tolist()}"
    )

    print(
        f"\nResults saved to:\n"
        f"{result_path}"
    )

    print(
        f"\nElapsed time: "
        f"{elapsed_seconds / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()