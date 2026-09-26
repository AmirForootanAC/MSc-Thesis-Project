"""Single entry point for all strong V2 complete-case baselines.

Supports crash-safe resume from the last completed epoch.
"""

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

from . import config
from .collate import complete_case_collate
from .dataset import CompleteCaseDataset
from .loss import ClassWeightedBCEWithLogitsLoss
from .metrics import (
    compute_metrics,
    optimize_thresholds,
)
from .model import FinalLinearV3
from .transforms import (
    get_eval_transform,
    get_train_transform,
)


# ================================================================
# Reproducibility
# ================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    torch_state = state["torch"]
    if not isinstance(torch_state, torch.ByteTensor):
        torch_state = torch.as_tensor(
            torch_state,
            dtype=torch.uint8,
            device="cpu",
        )

    torch.set_rng_state(torch_state.cpu())

    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = []

        for cuda_state in state["cuda"]:
            if not isinstance(cuda_state, torch.ByteTensor):
                cuda_state = torch.as_tensor(
                    cuda_state,
                    dtype=torch.uint8,
                    device="cpu",
                )

            cuda_states.append(cuda_state.cpu())

        torch.cuda.set_rng_state_all(cuda_states)


# ================================================================
# Device / AMP
# ================================================================

def get_device():
    if (
        config.DEVICE == "cuda"
        and torch.cuda.is_available()
    ):
        return torch.device("cuda")

    return torch.device("cpu")


def amp_enabled(device):
    return device.type == "cuda"


def create_grad_scaler(device):
    return torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled(device),
    )


# ================================================================
# Batch helpers
# ================================================================

def move_image_lists(
    image_lists,
    device,
):
    moved = []

    for image_list in image_lists:
        if len(image_list) == 0:
            raise RuntimeError(
                "Encountered a sample with zero images."
            )

        moved.append(
            [
                image.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                for image in image_list
            ]
        )

    return moved


def horizontally_flip_image_lists(
    image_lists,
):
    flipped = []

    for image_list in image_lists:
        flipped.append(
            [
                image.flip(dims=[2])
                for image in image_list
            ]
        )

    return flipped


def move_batch_to_device(
    batch,
    device,
):
    result = {
        "labels": batch["labels"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
    }

    if "images" in batch:
        result["images"] = move_image_lists(
            batch["images"],
            device,
        )

    if "radiographs" in batch:
        result["radiographs"] = move_image_lists(
            batch["radiographs"],
            device,
        )

    if "input_ids" in batch:
        result["input_ids"] = batch[
            "input_ids"
        ].to(
            device=device,
            non_blocking=True,
        )

        result["attention_mask"] = batch[
            "attention_mask"
        ].to(
            device=device,
            non_blocking=True,
        )

    return result


def model_forward(
    model,
    batch,
):
    kwargs = {}

    if "images" in batch:
        kwargs["images"] = batch["images"]

    if "radiographs" in batch:
        kwargs["radiographs"] = batch["radiographs"]

    if "input_ids" in batch:
        kwargs["input_ids"] = batch["input_ids"]
        kwargs["attention_mask"] = batch[
            "attention_mask"
        ]

    return model(**kwargs)


# ================================================================
# BatchNorm handling
# ================================================================

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


# ================================================================
# Dataset
# ================================================================

def build_datasets():
    modalities = config.ACTIVE_MODALITIES

    photograph_train_transform = None
    photograph_eval_transform = None

    radiograph_train_transform = None
    radiograph_eval_transform = None

    if "photograph" in modalities:
        photograph_train_transform = (
            get_train_transform(
                image_size=config.IMAGE_SIZE,
                modality="photograph",
            )
        )

        photograph_eval_transform = (
            get_eval_transform(
                image_size=config.IMAGE_SIZE,
            )
        )

    if "radiograph" in modalities:
        radiograph_train_transform = (
            get_train_transform(
                image_size=config.IMAGE_SIZE,
                modality="radiograph",
            )
        )

        radiograph_eval_transform = (
            get_eval_transform(
                image_size=config.IMAGE_SIZE,
            )
        )

    train_dataset = CompleteCaseDataset(
        csv_path=config.DATASET_PATH,
        split="train",
        image_root=config.IMAGE_ROOT,
        modalities=modalities,
        tokenizer_name=config.TEXT_MODEL_NAME,
        max_length=config.TEXT_MAX_LENGTH,
        photograph_transform=(
            photograph_train_transform
        ),
        radiograph_transform=(
            radiograph_train_transform
        ),
    )

    validation_dataset = CompleteCaseDataset(
        csv_path=config.DATASET_PATH,
        split="validation",
        image_root=config.IMAGE_ROOT,
        modalities=modalities,
        tokenizer_name=config.TEXT_MODEL_NAME,
        max_length=config.TEXT_MAX_LENGTH,
        photograph_transform=(
            photograph_eval_transform
        ),
        radiograph_transform=(
            radiograph_eval_transform
        ),
    )

    test_dataset = CompleteCaseDataset(
        csv_path=config.DATASET_PATH,
        split="test",
        image_root=config.IMAGE_ROOT,
        modalities=modalities,
        tokenizer_name=config.TEXT_MODEL_NAME,
        max_length=config.TEXT_MAX_LENGTH,
        photograph_transform=(
            photograph_eval_transform
        ),
        radiograph_transform=(
            radiograph_eval_transform
        ),
    )

    expected = config.EXPECTED_COUNTS

    actual = {
        "train": len(train_dataset),
        "validation": len(validation_dataset),
        "test": len(test_dataset),
    }

    for split in expected:
        if actual[split] != expected[split]:
            raise RuntimeError(
                f"Complete-case mismatch for {split}: "
                f"expected={expected[split]}, "
                f"found={actual[split]}"
            )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


# ================================================================
# DataLoader
# ================================================================

def make_loader(
    dataset,
    shuffle,
):
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=complete_case_collate,
    )


# ================================================================
# Class-weighted loss
# ================================================================

def get_training_labels(dataset):
    return dataset.df[
        config.LABEL_NAMES
    ].to_numpy(
        dtype=np.float32
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

    return (
        negative_counts
        / positive_counts
    ).astype(
        np.float32
    )


def build_criterion(
    train_dataset,
    device,
):
    labels = get_training_labels(
        train_dataset
    )

    pos_weights = compute_pos_weights(
        labels
    )

    print("\nClass positive weights:")

    for name, weight in zip(
        config.LABEL_NAMES,
        pos_weights,
    ):
        print(
            f"  {name:<30} {weight:.4f}"
        )

    return ClassWeightedBCEWithLogitsLoss(
        pos_weights=pos_weights
    ).to(device)


# ================================================================
# Model
# ================================================================

def build_model():
    return FinalLinearV3(
        modalities=config.ACTIVE_MODALITIES,
        text_model_name=config.TEXT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
        pretrained_image=(
            config.PRETRAINED_IMAGE_ENCODER
        ),
        freeze_image_encoder=(
            config.FREEZE_IMAGE_ENCODER
        ),
    )


def parameter_counts(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
    }


# ================================================================
# Optimizer
# ================================================================

def collect_parameters(
    module,
):
    return [
        p
        for p in module.parameters()
        if p.requires_grad
    ]


def build_optimizer(model):
    backbone_parameters = []
    text_parameters = []
    head_parameters = []

    image_names = [
        "photograph_aggregator",
        "radiograph_aggregator",
    ]

    for name in image_names:
        if hasattr(model, name):
            module = getattr(
                model,
                name,
            )

            backbone_modules = [
                module.layer0,
                module.layer1,
                module.layer2,
                module.layer3,
                module.layer4,
            ]

            head_modules = [
                module.cbam,
                module.projection,
            ]

            for submodule in backbone_modules:
                backbone_parameters.extend(
                    collect_parameters(
                        submodule
                    )
                )

            for submodule in head_modules:
                head_parameters.extend(
                    collect_parameters(
                        submodule
                    )
                )

    if hasattr(
        model,
        "text_encoder",
    ):
        text_parameters.extend(
            collect_parameters(
                model.text_encoder
            )
        )

    head_parameters.extend(
        collect_parameters(
            model.classifier
        )
    )

    parameter_groups = []

    if backbone_parameters:
        parameter_groups.append(
            {
                "name": "backbone",
                "params": backbone_parameters,
                "lr": config.BACKBONE_LR,
                "base_lr": config.BACKBONE_LR,
            }
        )

    if text_parameters:
        parameter_groups.append(
            {
                "name": "text",
                "params": text_parameters,
                "lr": config.TEXT_LR,
                "base_lr": config.TEXT_LR,
            }
        )

    if head_parameters:
        parameter_groups.append(
            {
                "name": "head",
                "params": head_parameters,
                "lr": config.CLASSIFIER_LR,
                "base_lr": config.CLASSIFIER_LR,
            }
        )

    if not parameter_groups:
        raise RuntimeError(
            "No trainable parameters found."
        )

    return AdamW(
        parameter_groups,
        weight_decay=config.WEIGHT_DECAY,
    )


def set_warmup_learning_rate(
    optimizer,
    epoch_index,
):
    if config.WARMUP_EPOCHS <= 0:
        return

    progress = min(
        (
            epoch_index + 1
        )
        / config.WARMUP_EPOCHS,
        1.0,
    )

    for group in optimizer.param_groups:
        group["lr"] = (
            group["base_lr"]
            * progress
        )


def current_learning_rates(
    optimizer,
):
    return {
        group["name"]: float(
            group["lr"]
        )
        for group in optimizer.param_groups
    }


# ================================================================
# Training
# ================================================================

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

    progress_bar = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch:02d} [Train]",
        leave=False,
        dynamic_ncols=True,
    )

    num_batches = len(loader)

    for batch_index, raw_batch in enumerate(
        progress_bar
    ):
        batch = move_batch_to_device(
            raw_batch,
            device,
        )

        labels = batch["labels"]

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled(device),
        ):
            logits = model_forward(
                model,
                batch,
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

        progress_bar.set_postfix(
            loss=(
                total_loss
                / max(sample_count, 1)
            )
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

    return (
        total_loss
        / max(sample_count, 1),
        metrics,
        logits_np,
        labels_np,
    )


# ================================================================
# Evaluation / TTA
# ================================================================

def probabilities_to_logits(
    probabilities,
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

    all_logits = []
    all_labels = []

    progress_bar = tqdm(
        loader,
        total=len(loader),
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )

    for raw_batch in progress_bar:
        batch = move_batch_to_device(
            raw_batch,
            device,
        )

        labels = batch["labels"]

        logits = model_forward(
            model,
            batch,
        )

        if logits.shape != labels.shape:
            raise RuntimeError(
                "Logit/label shape mismatch "
                "during evaluation: "
                f"logits={tuple(logits.shape)}, "
                f"labels={tuple(labels.shape)}"
            )

        has_images = (
            "images" in batch
            or "radiographs" in batch
        )

        if not use_tta or not has_images:
            final_logits = logits

        else:
            flipped_batch = dict(batch)

            if "images" in batch:
                flipped_batch["images"] = (
                    horizontally_flip_image_lists(
                        batch["images"]
                    )
                )

            if "radiographs" in batch:
                flipped_batch["radiographs"] = (
                    horizontally_flip_image_lists(
                        batch["radiographs"]
                    )
                )

            flipped_logits = model_forward(
                model,
                flipped_batch,
            )

            original_probabilities = (
                torch.sigmoid(logits)
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

    return (
        np.concatenate(
            all_logits,
            axis=0,
        ),
        np.concatenate(
            all_labels,
            axis=0,
        ),
    )


# ================================================================
# Checkpoints
# ================================================================

def scenario_output_root():
    return (
        config.RESULT_ROOT
        / config.ACTIVE_SCENARIO
    )


def prepare_scenario_output_directory():
    """
    Prepare the scenario directory.

    If no previous run exists, create it.

    If an interrupted run exists, preserve it so that training
    can resume.

    If a completed run exists, refuse to overwrite it. This prevents
    accidental destruction of a completed experiment.
    """
    out = scenario_output_root()

    if not out.exists():
        (
            out / "checkpoints"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        return out

    final_results = (
        out / "final_results.json"
    )

    latest_checkpoint = (
        out
        / "checkpoints"
        / "latest.pt"
    )

    if final_results.exists():
        raise RuntimeError(
            "\nA completed experiment already exists:\n"
            f"  {out}\n\n"
            "The existing experiment was NOT deleted.\n"
            "If you intentionally want to rerun this scenario, "
            "delete this scenario directory first."
        )

    (
        out / "checkpoints"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    if latest_checkpoint.exists():
        print(
            "\n[INFO] Existing resumable checkpoint found:"
        )
        print(
            f"       {latest_checkpoint}"
        )
    else:
        print(
            "\n[INFO] Existing scenario directory found, "
            "but no resumable checkpoint exists."
        )

    return out


def checkpoint_path(
    out,
    epoch,
):
    return (
        out
        / "checkpoints"
        / f"epoch_{epoch:03d}.pt"
    )


def latest_checkpoint_path(out):
    return (
        out
        / "checkpoints"
        / "latest.pt"
    )


def atomic_torch_save(
    state,
    path,
):
    path = Path(path)

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        state,
        temporary_path,
    )

    temporary_path.replace(path)


def save_epoch_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    val_macro_f1,
):
    state = {
        "checkpoint_type": "epoch",
        "epoch": int(epoch),
        "val_macro_f1": float(
            val_macro_f1
        ),
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "scheduler_state_dict":
            (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
        "scaler_state_dict":
            scaler.state_dict(),
        "rng_state":
            get_rng_state(),
    }

    atomic_torch_save(
        state,
        path,
    )


def save_latest_checkpoint(
    out,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_macro_f1,
    best_epoch,
    patience_counter,
    history,
    top_k,
):
    state = {
        "checkpoint_type": "latest",
        "epoch": int(epoch),
        "best_val_macro_f1": float(
            best_val_macro_f1
        ),
        "best_epoch": (
            int(best_epoch)
            if best_epoch is not None
            else None
        ),
        "patience_counter": int(
            patience_counter
        ),
        "history": history,
        "top_k": top_k,
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "scheduler_state_dict":
            (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
        "scaler_state_dict":
            scaler.state_dict(),
        "rng_state":
            get_rng_state(),
    }

    path = latest_checkpoint_path(
        out
    )

    atomic_torch_save(
        state,
        path,
    )


def update_top_k(
    top_k,
    path,
    score,
):
    top_k.append(
        {
            "path": str(path),
            "epoch": int(
                path.stem.split("_")[-1]
            ),
            "score": float(score),
        }
    )

    top_k.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    while len(top_k) > config.TOP_K_CHECKPOINTS:
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
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    return checkpoint


def restore_training_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    checkpoint,
    device,
):
    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scheduler_state = checkpoint.get(
        "scheduler_state_dict"
    )

    if (
        scheduler is not None
        and scheduler_state is not None
    ):
        scheduler.load_state_dict(
            scheduler_state
        )

    scaler_state = checkpoint.get(
        "scaler_state_dict"
    )

    if scaler_state is not None:
        scaler.load_state_dict(
            scaler_state
        )

    restore_rng_state(
        checkpoint.get("rng_state")
    )

    return checkpoint


# ================================================================
# Config snapshot
# ================================================================

def config_snapshot():
    return {
        "active_scenario":
            config.ACTIVE_SCENARIO,

        "active_modalities":
            list(config.ACTIVE_MODALITIES),

        "dataset":
            str(config.DATASET_PATH),

        "expected_counts":
            dict(config.EXPECTED_COUNTS),

        "image_size":
            config.IMAGE_SIZE,

        "pretrained_image_encoder":
            config.PRETRAINED_IMAGE_ENCODER,

        "freeze_image_encoder":
            config.FREEZE_IMAGE_ENCODER,

        "batch_size":
            config.BATCH_SIZE,

        "gradient_accumulation_steps":
            config.GRADIENT_ACCUMULATION_STEPS,

        "backbone_lr":
            config.BACKBONE_LR,

        "classifier_lr":
            config.CLASSIFIER_LR,

        "text_lr":
            config.TEXT_LR,

        "weight_decay":
            config.WEIGHT_DECAY,

        "warmup_epochs":
            config.WARMUP_EPOCHS,

        "dropout":
            config.DROPOUT,

        "gradient_clip_norm":
            config.GRADIENT_CLIP_NORM,

        "top_k_checkpoints":
            config.TOP_K_CHECKPOINTS,

        "tta":
            config.USE_TTA,

        "num_epochs":
            config.NUM_EPOCHS,

        "early_stopping_patience":
            config.EARLY_STOPPING_PATIENCE,

        "default_threshold":
            config.DEFAULT_THRESHOLD,

        "threshold_min":
            config.THRESHOLD_MIN,

        "threshold_max":
            config.THRESHOLD_MAX,

        "threshold_step":
            config.THRESHOLD_STEP,

        "seed":
            config.SEED,

        "device":
            config.DEVICE,

        "num_workers":
            config.NUM_WORKERS,

        "pin_memory":
            config.PIN_MEMORY,

        "text_model_name":
            config.TEXT_MODEL_NAME,

        "text_max_length":
            config.TEXT_MAX_LENGTH,

        "text_columns":
            list(config.TEXT_COLUMNS),

        "loss":
            "class_weighted_bce_with_logits",

        "weighted_sampler":
            False,

        "cbam":
            True,

        "batchnorm_statistics_frozen":
            True,

        "image_view_aggregation":
            "mean",

        "resume_support":
            True,

        "resume_checkpoint":
            "latest.pt",
    }


# ================================================================
# Main
# ================================================================

def main():
    set_seed(config.SEED)

    device = get_device()

    scenario = config.ACTIVE_SCENARIO
    modalities = config.ACTIVE_MODALITIES

    print("=" * 70)
    print(
        "STRONG SUPERVISED MULTIMODAL BASELINE V3"
    )
    print("=" * 70)

    print(
        f"Scenario:          {scenario}"
    )

    print(
        f"Modalities:        {modalities}"
    )

    print(
        f"Dataset:           {config.DATASET_PATH}"
    )

    print(
        f"Device:            {device}"
    )

    print(
        f"Image size:        {config.IMAGE_SIZE}"
    )

    print(
        f"Batch size:        {config.BATCH_SIZE}"
    )

    print(
        f"Backbone LR:       {config.BACKBONE_LR}"
    )

    print(
        f"Classifier LR:     {config.CLASSIFIER_LR}"
    )

    print(
        f"Text LR:           {config.TEXT_LR}"
    )

    print(
        f"Weight decay:      {config.WEIGHT_DECAY}"
    )

    print(
        "Loss:              Class-weighted BCE"
    )

    print(
        "Image encoder:     ResNet50 + CBAM"
    )

    print(
        "Image fine-tuning: Full"
    )

    print(
        "View aggregation:  Mean"
    )

    print(
        f"TTA:               {config.USE_TTA}"
    )

    print(
        f"Top-K:             {config.TOP_K_CHECKPOINTS}"
    )

    print(
        "BatchNorm stats:   Frozen"
    )

    print(
        "Checkpoint metric: Validation Macro F1 @ 0.5"
    )

    print(
        "Resume:            Enabled"
    )

    print("=" * 70)

    out = prepare_scenario_output_directory()

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = build_datasets()

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

    train_loader = make_loader(
        train_dataset,
        shuffle=True,
    )

    validation_loader = make_loader(
        validation_dataset,
        shuffle=False,
    )

    test_loader = make_loader(
        test_dataset,
        shuffle=False,
    )

    criterion = build_criterion(
        train_dataset,
        device,
    )

    model = build_model().to(device)

    params = parameter_counts(
        model
    )

    print("\nModel parameters:")
    print(
        f"Total:     {params['total']:,}"
    )
    print(
        f"Trainable: {params['trainable']:,}"
    )
    print(
        f"Frozen:    {params['frozen']:,}"
    )

    optimizer = build_optimizer(
        model
    )

    scheduler = None

    if (
        config.NUM_EPOCHS
        > config.WARMUP_EPOCHS
    ):
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(
                config.NUM_EPOCHS
                - config.WARMUP_EPOCHS,
                1,
            ),
            eta_min=1e-7,
        )

    scaler = create_grad_scaler(
        device
    )

    history = []
    top_k = []

    best_val_macro_f1 = -math.inf
    best_epoch = None
    patience_counter = 0

    start_epoch = 1

    # ============================================================
    # Resume
    # ============================================================

    latest_path = latest_checkpoint_path(
        out
    )

    if latest_path.exists():
        print("\n" + "=" * 70)
        print("RESUMING INTERRUPTED EXPERIMENT")
        print("=" * 70)

        checkpoint = torch.load(
            latest_path,
            map_location=device,
            weights_only=False,
        )

        completed_epoch = int(
            checkpoint["epoch"]
        )

        if completed_epoch >= config.NUM_EPOCHS:
            raise RuntimeError(
                "The latest checkpoint already contains "
                "the configured final epoch, but final_results.json "
                "does not exist. Inspect the experiment directory "
                "before continuing."
            )

        restore_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            checkpoint=checkpoint,
            device=device,
        )

        history = checkpoint.get(
            "history",
            [],
        )

        top_k = checkpoint.get(
            "top_k",
            [],
        )

        best_val_macro_f1 = float(
            checkpoint.get(
                "best_val_macro_f1",
                -math.inf,
            )
        )

        best_epoch = checkpoint.get(
            "best_epoch"
        )

        if best_epoch is not None:
            best_epoch = int(
                best_epoch
            )

        patience_counter = int(
            checkpoint.get(
                "patience_counter",
                0,
            )
        )

        start_epoch = (
            completed_epoch + 1
        )

        print(
            f"Completed epoch:       "
            f"{completed_epoch}"
        )

        print(
            f"Resuming from epoch:   "
            f"{start_epoch}"
        )

        print(
            f"Best validation F1:    "
            f"{best_val_macro_f1:.4f}"
        )

        print(
            f"Best epoch:            "
            f"{best_epoch}"
        )

        print(
            f"Patience:              "
            f"{patience_counter}/"
            f"{config.EARLY_STOPPING_PATIENCE}"
        )

        print("=" * 70)

    else:
        print(
            "\n[INFO] No previous checkpoint found."
        )

    start_time = time.time()

    # ============================================================
    # Training loop
    # ============================================================

    for epoch in range(
        start_epoch,
        config.NUM_EPOCHS + 1,
    ):
        epoch_index = epoch - 1

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
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
        )

        # IMPORTANT:
        # Checkpoint selection validation must NOT use TTA.
        validation_logits, validation_labels = predict(
            model,
            validation_loader,
            device,
            use_tta=False,
            description=(
                f"Epoch {epoch:02d} [Val]"
            ),
        )

        validation_metrics = compute_metrics(
            validation_logits,
            validation_labels,
            threshold=config.DEFAULT_THRESHOLD,
            label_names=config.LABEL_NAMES,
        )

        current_lrs = (
            current_learning_rates(
                optimizer
            )
        )

        current_macro_f1 = (
            validation_metrics[
                "macro_f1"
            ]
        )

        improved = (
            current_macro_f1
            > best_val_macro_f1
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    train_loss
                ),
                "train_macro_f1": float(
                    train_metrics["macro_f1"]
                ),
                "train_micro_f1": float(
                    train_metrics["micro_f1"]
                ),
                "train_auroc": float(
                    train_metrics["auroc"]
                ),
                "train_accuracy": float(
                    train_metrics["accuracy"]
                ),
                "validation_macro_f1": float(
                    validation_metrics[
                        "macro_f1"
                    ]
                ),
                "validation_micro_f1": float(
                    validation_metrics[
                        "micro_f1"
                    ]
                ),
                "validation_auroc": float(
                    validation_metrics[
                        "auroc"
                    ]
                ),
                "validation_accuracy": float(
                    validation_metrics[
                        "accuracy"
                    ]
                ),
                "learning_rates": current_lrs,
            }
        )

        if improved:
            best_val_macro_f1 = (
                current_macro_f1
            )

            best_epoch = epoch

            patience_counter = 0

        else:
            patience_counter += 1

        # --------------------------------------------------------
        # Save epoch checkpoint.
        #
        # This is the historical checkpoint used for top-K.
        # --------------------------------------------------------

        epoch_path = checkpoint_path(
            out,
            epoch,
        )

        # Scheduler is advanced BEFORE the latest checkpoint is
        # written so that resuming starts with the correct optimizer
        # and scheduler state for the next epoch.
        if (
            epoch
            > config.WARMUP_EPOCHS
            and scheduler is not None
        ):
            scheduler.step()

        save_epoch_checkpoint(
            path=epoch_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            val_macro_f1=current_macro_f1,
        )

        top_k = update_top_k(
            top_k,
            epoch_path,
            current_macro_f1,
        )

        # --------------------------------------------------------
        # Save latest resume checkpoint.
        #
        # This is deliberately separate from top-K checkpoints.
        # It is NEVER deleted by top-K maintenance.
        # --------------------------------------------------------

        save_latest_checkpoint(
            out=out,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_macro_f1=best_val_macro_f1,
            best_epoch=best_epoch,
            patience_counter=patience_counter,
            history=history,
            top_k=top_k,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train F1 {train_metrics['macro_f1']:.4f} | "
            f"Val F1 {validation_metrics['macro_f1']:.4f} | "
            f"Val AUROC {validation_metrics['auroc']:.4f} | "
            f"LR {current_lrs} | "
            f"Patience "
            f"{patience_counter}/"
            f"{config.EARLY_STOPPING_PATIENCE}"
        )

        history_path = (
            out / "history.json"
        )

        history_path.write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )

        if (
            patience_counter
            >= config.EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping triggered."
            )
            break

    # ============================================================
    # Best checkpoint
    # ============================================================

    if best_epoch is None:
        raise RuntimeError(
            "Training finished without "
            "a best checkpoint."
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
            "Best checkpoint was not retained "
            "inside top-K checkpoints."
        )

    checkpoint = load_checkpoint(
        model,
        best_checkpoint,
        device,
    )

    print("\n" + "=" * 70)
    print("BEST CHECKPOINT")
    print("=" * 70)

    print(
        f"Epoch:              {checkpoint['epoch']}"
    )

    print(
        f"Validation Macro F1: "
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    # ============================================================
    # Final validation
    # ============================================================

    validation_logits, validation_labels = predict(
        model,
        validation_loader,
        device,
        use_tta=config.USE_TTA,
        description="[Final Validation TTA]",
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

    # ============================================================
    # Final test
    # ============================================================

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

    elapsed_seconds = (
        time.time()
        - start_time
    )

    # ============================================================
    # Final result
    # ============================================================

    results = {
        "experiment":
            "strong_supervised_multimodal_linear_v3",

        "scenario":
            scenario,

        "modalities":
            list(modalities),

        "dataset":
            str(config.DATASET_PATH),

        "labels":
            list(config.LABEL_NAMES),

        "seed":
            config.SEED,

        "train_samples":
            len(train_dataset),

        "validation_samples":
            len(validation_dataset),

        "test_samples":
            len(test_dataset),

        "sample_counts_expected":
            dict(config.EXPECTED_COUNTS),

        "checkpoint_selection":
            "validation_macro_f1_at_threshold_0.5_without_tta",

        "best_epoch":
            int(best_epoch),

        "best_validation_macro_f1":
            float(best_val_macro_f1),

        "validation_default":
            validation_default,

        "validation_optimized":
            validation_optimized,

        "optimized_thresholds":
            np.asarray(
                optimized_thresholds
            ).tolist(),

        "test_default":
            test_default,

        "test_optimized":
            test_optimized,

        "parameter_counts":
            params,

        "top_k_checkpoints":
            top_k,

        "history":
            history,

        "elapsed_seconds":
            float(elapsed_seconds),

        "config":
            config_snapshot(),
    }

    results_path = (
        out / "final_results.json"
    )

    results_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Macro F1 @ 0.5: "
        f"{test_default['macro_f1']:.4f}"
    )

    print(
        f"Micro F1 @ 0.5: "
        f"{test_default['micro_f1']:.4f}"
    )

    print(
        f"AUROC:           "
        f"{test_default['auroc']:.4f}"
    )

    print(
        f"Accuracy:        "
        f"{test_default['accuracy']:.4f}"
    )

    print(
        "\nValidation-derived thresholds:"
    )

    print(
        np.asarray(
            optimized_thresholds
        ).tolist()
    )

    print(
        f"Optimized Macro F1: "
        f"{test_optimized['macro_f1']:.4f}"
    )

    print(
        f"\nResults: "
        f"{results_path}"
    )

    print(
        f"Elapsed: "
        f"{elapsed_seconds / 60:.1f} minutes"
    )

    print(
        "\n[INFO] Training completed successfully."
    )


if __name__ == "__main__":
    main()