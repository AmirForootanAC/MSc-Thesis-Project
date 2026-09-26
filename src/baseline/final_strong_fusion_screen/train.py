"""Fast frozen-feature screening for multimodal fusion."""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.baseline.final_strong_v2.collate import (
    complete_case_collate,
)
from src.baseline.final_strong_v2.dataset import (
    CompleteCaseDataset,
)
from src.baseline.final_strong_v2.metrics import (
    compute_metrics,
)

from src.baseline.final_strong_v2.model import (
    FinalStrongV2,
)

from src.baseline.final_strong_v2.transforms import (
    get_eval_transform,
)

from . import config
from .model import (
    FullTextAnchoredFusion,
    TextAnchoredFusion,
    TextOnlyFusion,
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


# ================================================================
# Device
# ================================================================

def get_device():
    if (
        config.DEVICE == "cuda"
        and torch.cuda.is_available()
    ):
        return torch.device("cuda")

    return torch.device("cpu")


# ================================================================
# Batch handling
# ================================================================

def move_image_lists(
    image_lists,
    device,
):
    result = []

    for image_list in image_lists:
        result.append(
            [
                image.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                for image in image_list
            ]
        )

    return result


def move_batch(
    batch,
    device,
):
    result = {
        "labels": batch["labels"].to(
            device,
            dtype=torch.float32,
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
        ].to(device)

        result["attention_mask"] = batch[
            "attention_mask"
        ].to(device)

    return result


# ================================================================
# Dataset
# ================================================================

def build_dataset(
    split,
    modalities,
):
    transform = get_eval_transform(
        image_size=config.IMAGE_SIZE
    )

    dataset = CompleteCaseDataset(
        csv_path=config.DATASET_PATH,
        split=split,
        image_root=config.IMAGE_ROOT,
        modalities=modalities,
        tokenizer_name=config.TEXT_MODEL_NAME,
        max_length=config.TEXT_MAX_LENGTH,
        photograph_transform=transform,
        radiograph_transform=transform,
    )

    expected = config.EXPECTED_COUNTS[
        split
    ]

    if len(dataset) != expected:
        raise RuntimeError(
            f"{split} count mismatch: "
            f"expected {expected}, "
            f"found {len(dataset)}"
        )

    return dataset


def make_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=complete_case_collate,
    )


# ================================================================
# Load source model
# ================================================================

def build_source_model(
    device,
):
    model = FinalStrongV2(
        modalities=(
            "photograph",
            "radiograph",
            "text",
        ),
        text_model_name=config.TEXT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
        pretrained_image=True,
        freeze_image_encoder=False,
    ).to(device)

    checkpoint = torch.load(
        config.SOURCE_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    print(
        f"\n[INFO] Loaded source checkpoint:"
    )
    print(
        f"       {config.SOURCE_CHECKPOINT}"
    )

    print(
        f"       Epoch: "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"       Val Macro F1: "
        f"{checkpoint.get('val_macro_f1'):.4f}"
    )

    return model


# ================================================================
# Feature extraction
# ================================================================

@torch.no_grad()
def extract_features(
    model,
    loader,
    device,
    modalities,
):
    text_features = []
    photograph_features = []
    radiograph_features = []
    labels = []

    progress = tqdm(
        loader,
        desc="Extracting frozen features",
        dynamic_ncols=True,
    )

    for raw_batch in progress:
        batch = move_batch(
            raw_batch,
            device,
        )

        if "text" in modalities:
            outputs = model.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch[
                    "attention_mask"
                ],
            )

            text_features.append(
                outputs.last_hidden_state[
                    :,
                    0,
                ]
                .float()
                .cpu()
            )

        if "photograph" in modalities:
            photograph = (
                model.photograph_aggregator(
                    batch["images"]
                )
            )

            photograph_features.append(
                photograph.float().cpu()
            )

        if "radiograph" in modalities:
            radiograph = (
                model.radiograph_aggregator(
                    batch["radiographs"]
                )
            )

            radiograph_features.append(
                radiograph.float().cpu()
            )

        labels.append(
            batch["labels"]
            .float()
            .cpu()
        )

    result = {
        "labels": torch.cat(
            labels,
            dim=0,
        )
    }

    if text_features:
        result["text"] = torch.cat(
            text_features,
            dim=0,
        )

    if photograph_features:
        result["photograph"] = torch.cat(
            photograph_features,
            dim=0,
        )

    if radiograph_features:
        result["radiograph"] = torch.cat(
            radiograph_features,
            dim=0,
        )

    return result


# ================================================================
# Cache
# ================================================================

def cache_path(
    split,
):
    config.CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        config.CACHE_ROOT
        / (
            f"{split}_"
            f"{config.CACHE_VERSION}.pt"
        )
    )


def load_or_create_cache(
    model,
    split,
    modalities,
    device,
    loader,
):
    path = cache_path(split)

    if path.exists():
        print(
            f"\n[INFO] Loading cached "
            f"{split} features:"
        )
        print(
            f"       {path}"
        )

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    print(
        f"\n[INFO] Creating "
        f"{split} feature cache..."
    )

    features = extract_features(
        model=model,
        loader=loader,
        device=device,
        modalities=modalities,
    )

    torch.save(
        features,
        path,
    )

    print(
        f"[INFO] Cache saved:"
    )
    print(
        f"       {path}"
    )

    return features


# ================================================================
# Screening dataset
# ================================================================

def make_screening_train_subset(
    features,
):
    n = features["labels"].shape[0]

    if (
        config.SCREEN_TRAIN_SAMPLES
        >= n
    ):
        return features

    generator = torch.Generator()
    generator.manual_seed(
        config.SEED
    )

    indices = torch.randperm(
        n,
        generator=generator,
    )[
        :config.SCREEN_TRAIN_SAMPLES
    ]

    result = {}

    for key, value in features.items():
        result[key] = value[
            indices
        ]

    return result


def build_tensor_loader(
    features,
    modalities,
    shuffle,
):
    tensors = []

    if "text" in modalities:
        tensors.append(
            features["text"]
        )

    if "photograph" in modalities:
        tensors.append(
            features["photograph"]
        )

    if "radiograph" in modalities:
        tensors.append(
            features["radiograph"]
        )

    tensors.append(
        features["labels"]
    )

    dataset = TensorDataset(
        *tensors
    )

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=config.PIN_MEMORY,
    )


# ================================================================
# Model factory
# ================================================================

def build_fusion_model(
    scenario,
):
    if scenario == "text_only":
        return TextOnlyFusion(
            text_dim=768,
            num_labels=config.NUM_LABELS,
        )

    if scenario == "image_text":
        return TextAnchoredFusion(
            text_dim=768,
            image_dim=128,
            num_labels=config.NUM_LABELS,
        )

    if scenario == "full_multimodal":
        return FullTextAnchoredFusion(
            text_dim=768,
            image_dim=128,
            num_labels=config.NUM_LABELS,
        )

    raise ValueError(
        f"Unknown screening scenario: "
        f"{scenario}"
    )


# ================================================================
# Forward
# ================================================================

def fusion_forward(
    model,
    batch,
    scenario,
):
    if scenario == "text_only":
        text, labels = batch

        logits = model(
            text
        )

        return logits, labels

    if scenario == "image_text":
        text, photograph, labels = batch

        logits = model(
            text=text,
            image=photograph,
        )

        return logits, labels

    if scenario == "full_multimodal":
        (
            text,
            photograph,
            radiograph,
            labels,
        ) = batch

        logits = model(
            text=text,
            photograph=photograph,
            radiograph=radiograph,
        )

        return logits, labels

    raise ValueError(
        f"Unknown scenario: {scenario}"
    )


# ================================================================
# Loss
# ================================================================

def compute_pos_weights(
    labels,
):
    labels = labels.numpy()

    positives = labels.sum(
        axis=0
    )

    negatives = (
        labels.shape[0]
        - positives
    )

    positives = np.maximum(
        positives,
        1.0,
    )

    return torch.tensor(
        negatives / positives,
        dtype=torch.float32,
    )


def build_loss(
    train_features,
    device,
):
    pos_weights = compute_pos_weights(
        train_features["labels"]
    ).to(device)

    return torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )


# ================================================================
# One epoch
# ================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scenario,
):
    model.train()

    all_logits = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = tuple(
            tensor.to(
                device,
                non_blocking=True,
            )
            for tensor in batch
        )

        logits, labels = fusion_forward(
            model,
            batch,
            scenario,
        )

        loss = criterion(
            logits,
            labels,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        batch_size = labels.shape[0]

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        all_logits.append(
            logits.detach()
            .cpu()
            .numpy()
        )

        all_labels.append(
            labels.detach()
            .cpu()
            .numpy()
        )

    logits = np.concatenate(
        all_logits,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    metrics = compute_metrics(
        logits,
        labels,
        threshold=0.5,
        label_names=config.LABEL_NAMES,
    )

    return (
        total_loss
        / total_samples,
        metrics,
    )


# ================================================================
# Evaluation
# ================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    scenario,
):
    model.eval()

    all_logits = []
    all_labels = []

    for batch in loader:
        batch = tuple(
            tensor.to(
                device,
                non_blocking=True,
            )
            for tensor in batch
        )

        logits, labels = fusion_forward(
            model,
            batch,
            scenario,
        )

        all_logits.append(
            logits.cpu().numpy()
        )

        all_labels.append(
            labels.cpu().numpy()
        )

    logits = np.concatenate(
        all_logits,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    metrics = compute_metrics(
        logits,
        labels,
        threshold=0.5,
        label_names=config.LABEL_NAMES,
    )

    return metrics


# ================================================================
# Single screening run
# ================================================================

def run_scenario(
    scenario,
    train_features,
    validation_features,
    device,
):
    print("\n")
    print("=" * 70)
    print(
        f"SCREENING: {scenario}"
    )
    print("=" * 70)

    train_features = (
        make_screening_train_subset(
            train_features
        )
    )

    train_loader = build_tensor_loader(
        train_features,
        config.SCENARIOS[scenario],
        shuffle=True,
    )

    validation_loader = build_tensor_loader(
        validation_features,
        config.SCENARIOS[scenario],
        shuffle=False,
    )

    model = build_fusion_model(
        scenario
    ).to(device)

    criterion = build_loss(
        train_features,
        device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_score = -math.inf
    best_epoch = None
    patience = 0

    history = []

    for epoch in range(
        1,
        config.NUM_EPOCHS + 1,
    ):
        start = time.time()

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scenario,
            )
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            scenario,
        )

        score = (
            validation_metrics[
                "macro_f1"
            ]
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience = 0
        else:
            patience += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    train_loss
                ),
                "train_macro_f1": float(
                    train_metrics[
                        "macro_f1"
                    ]
                ),
                "validation_macro_f1": float(
                    validation_metrics[
                        "macro_f1"
                    ]
                ),
                "validation_auroc": float(
                    validation_metrics[
                        "auroc"
                    ]
                ),
                "seconds": float(
                    time.time() - start
                ),
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train F1 "
            f"{train_metrics['macro_f1']:.4f} | "
            f"Val F1 "
            f"{validation_metrics['macro_f1']:.4f} | "
            f"Val AUROC "
            f"{validation_metrics['auroc']:.4f} | "
            f"Patience "
            f"{patience}/"
            f"{config.EARLY_STOPPING_PATIENCE}"
        )

        if (
            patience
            >= config.EARLY_STOPPING_PATIENCE
        ):
            break

    result = {
        "scenario": scenario,
        "train_samples": int(
            len(
                train_features["labels"]
            )
        ),
        "validation_samples": int(
            len(
                validation_features["labels"]
            )
        ),
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_macro_f1": float(
            best_score
        ),
        "history": history,
    }

    return result


# ================================================================
# Main
# ================================================================

def main():
    set_seed(config.SEED)

    device = get_device()

    print("=" * 70)
    print(
        "FAST MULTIMODAL FUSION SCREENING"
    )
    print("=" * 70)

    print(
        f"Device:             {device}"
    )

    print(
        f"Source checkpoint:  "
        f"{config.SOURCE_CHECKPOINT}"
    )

    print(
        f"Train screening:    "
        f"{config.SCREEN_TRAIN_SAMPLES}"
    )

    print(
        f"Validation:         "
        f"{config.EXPECTED_COUNTS['validation']}"
    )

    print(
        f"Epochs:             "
        f"{config.NUM_EPOCHS}"
    )

    print(
        f"Batch size:         "
        f"{config.BATCH_SIZE}"
    )

    print(
        "Encoders:            FROZEN"
    )

    print(
        "Fusion:              TRAINED ONLY"
    )

    print("=" * 70)

    config.RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Build source model
    # ------------------------------------------------------------

    source_model = build_source_model(
        device
    )

    # ------------------------------------------------------------
    # Build complete-case datasets
    # ------------------------------------------------------------

    full_modalities = (
        "photograph",
        "radiograph",
        "text",
    )

    train_dataset = build_dataset(
        "train",
        full_modalities,
    )

    validation_dataset = build_dataset(
        "validation",
        full_modalities,
    )

    train_loader = make_loader(
        train_dataset
    )

    validation_loader = make_loader(
        validation_dataset
    )

    # ------------------------------------------------------------
    # Cache source representations once
    # ------------------------------------------------------------

    train_features = load_or_create_cache(
        model=source_model,
        split="train",
        modalities=full_modalities,
        device=device,
        loader=train_loader,
    )

    validation_features = (
        load_or_create_cache(
            model=source_model,
            split="validation",
            modalities=full_modalities,
            device=device,
            loader=validation_loader,
        )
    )

    del source_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nCached feature shapes:")

    for key, value in train_features.items():
        print(
            f"  {key:<15} "
            f"{tuple(value.shape)}"
        )

    # ------------------------------------------------------------
    # Run all screening scenarios
    # ------------------------------------------------------------

    results = []

    for scenario in config.ACTIVE_SCENARIOS:
        result = run_scenario(
            scenario=scenario,
            train_features=train_features,
            validation_features=(
                validation_features
            ),
            device=device,
        )

        results.append(
            result
        )

    # ------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------

    summary = {
        "experiment":
            "fast_multimodal_fusion_screening",

        "source_checkpoint":
            str(config.SOURCE_CHECKPOINT),

        "source_checkpoint_version":
            "strong_v2",

        "train_samples":
            config.SCREEN_TRAIN_SAMPLES,

        "validation_samples":
            config.EXPECTED_COUNTS[
                "validation"
            ],

        "epochs":
            config.NUM_EPOCHS,

        "seed":
            config.SEED,

        "encoders_frozen":
            True,

        "test_used":
            False,

        "results":
            results,
    }

    output_path = (
        config.RESULT_ROOT
        / "screening_results.json"
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n")
    print("=" * 70)
    print(
        "SCREENING SUMMARY"
    )
    print("=" * 70)

    for result in results:
        print(
            f"{result['scenario']:<20} "
            f"Best Val Macro F1: "
            f"{result['best_validation_macro_f1']:.4f} "
            f"(epoch "
            f"{result['best_epoch']})"
        )

    print("=" * 70)

    print(
        f"\nResults saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()