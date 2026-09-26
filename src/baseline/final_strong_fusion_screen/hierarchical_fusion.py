"""
Hierarchical multimodal fusion on frozen V2 representations.

Stage 1:
    photograph + radiograph -> visual representation

Stage 2:
    visual representation + text -> multimodal representation

Stage 3:
    six independent label-specific prediction heads

Protocol:
    train       -> model fitting
    validation  -> model selection + threshold selection
    test        -> one final evaluation with frozen validation thresholds

No test information is used for training or model selection.
"""

import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Device
# =============================================================================

def get_device():
    if (
        config.DEVICE == "cuda"
        and torch.cuda.is_available()
    ):
        return torch.device("cuda")

    return torch.device("cpu")


# =============================================================================
# Feature cache
# =============================================================================

def load_cache(split):

    path = (
        config.CACHE_ROOT
        / f"{split}_{config.CACHE_VERSION}.pt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature cache:\n{path}"
        )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = (
        "text",
        "photograph",
        "radiograph",
        "labels",
    )

    for key in required:
        if key not in data:
            raise RuntimeError(
                f"{split} cache missing key: {key}"
            )

    return data


# =============================================================================
# DataLoader
# =============================================================================

def make_loader(features, shuffle):

    dataset = TensorDataset(
        features["text"].float(),
        features["photograph"].float(),
        features["radiograph"].float(),
        features["labels"].float(),
    )

    return DataLoader(
        dataset,
        batch_size=64,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=config.PIN_MEMORY,
    )


# =============================================================================
# Hierarchical Fusion Model
# =============================================================================

class HierarchicalFusion(nn.Module):
    """
    Hierarchical visual-first multimodal fusion.

    Level 1:
        photograph -> projection
        radiograph -> projection
        photograph/radiograph interaction
        -> visual representation

    Level 2:
        text -> projection
        visual representation + text
        text/visual interaction
        -> multimodal representation

    Level 3:
        six independent label heads
    """

    def __init__(
        self,
        text_dim=768,
        image_dim=128,
        visual_dim=128,
        multimodal_dim=256,
        num_labels=6,
        dropout=0.20,
    ):
        super().__init__()

        self.num_labels = num_labels

        # ==================================================================
        # LEVEL 1 — VISUAL ENCODING
        # ==================================================================

        self.photo_projection = nn.Sequential(
            nn.Linear(
                image_dim,
                visual_dim,
            ),
            nn.LayerNorm(visual_dim),
            nn.GELU(),
        )

        self.xray_projection = nn.Sequential(
            nn.Linear(
                image_dim,
                visual_dim,
            ),
            nn.LayerNorm(visual_dim),
            nn.GELU(),
        )

        # Explicit visual interaction.
        self.visual_interaction = nn.Sequential(
            nn.Linear(
                visual_dim,
                visual_dim,
            ),
            nn.LayerNorm(visual_dim),
            nn.GELU(),
        )

        self.visual_fusion = nn.Sequential(
            nn.Linear(
                visual_dim * 3,
                visual_dim * 2,
            ),
            nn.LayerNorm(visual_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                visual_dim * 2,
                visual_dim,
            ),
            nn.LayerNorm(visual_dim),
            nn.GELU(),
        )

        # ==================================================================
        # LEVEL 2 — TEXT + VISUAL
        # ==================================================================

        self.text_projection = nn.Sequential(
            nn.Linear(
                text_dim,
                multimodal_dim,
            ),
            nn.LayerNorm(multimodal_dim),
            nn.GELU(),
        )

        self.visual_to_multimodal = nn.Sequential(
            nn.Linear(
                visual_dim,
                multimodal_dim,
            ),
            nn.LayerNorm(multimodal_dim),
            nn.GELU(),
        )

        self.cross_modal_interaction = nn.Sequential(
            nn.Linear(
                multimodal_dim,
                multimodal_dim,
            ),
            nn.LayerNorm(multimodal_dim),
            nn.GELU(),
        )

        # Concatenate:
        #   text
        #   visual
        #   text * visual
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(
                multimodal_dim * 3,
                512,
            ),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                512,
                multimodal_dim,
            ),
            nn.LayerNorm(multimodal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ==================================================================
        # LEVEL 3 — LABEL-SPECIFIC HEADS
        # ==================================================================

        self.label_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        multimodal_dim,
                        96,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(
                        96,
                        1,
                    ),
                )
                for _ in range(num_labels)
            ]
        )

    def forward(
        self,
        text,
        photograph,
        radiograph,
    ):

        # ==============================================================
        # LEVEL 1 — VISUAL FUSION
        # ==============================================================

        photo = self.photo_projection(
            photograph
        )

        xray = self.xray_projection(
            radiograph
        )

        # Multiplicative visual interaction.
        visual_product = photo * xray

        visual_product = self.visual_interaction(
            visual_product
        )

        visual_input = torch.cat(
            [
                photo,
                xray,
                visual_product,
            ],
            dim=1,
        )

        visual = self.visual_fusion(
            visual_input
        )

        # ==============================================================
        # LEVEL 2 — TEXT + VISUAL
        # ==============================================================

        text_h = self.text_projection(
            text
        )

        visual_h = self.visual_to_multimodal(
            visual
        )

        # Explicit text/visual interaction.
        cross_modal = (
            text_h * visual_h
        )

        cross_modal = (
            self.cross_modal_interaction(
                cross_modal
            )
        )

        multimodal_input = torch.cat(
            [
                text_h,
                visual_h,
                cross_modal,
            ],
            dim=1,
        )

        multimodal = self.multimodal_fusion(
            multimodal_input
        )

        # ==============================================================
        # LEVEL 3 — SIX LABEL HEADS
        # ==============================================================

        logits = torch.cat(
            [
                head(multimodal)
                for head in self.label_heads
            ],
            dim=1,
        )

        return logits


# =============================================================================
# Class weights
# =============================================================================

def compute_pos_weights(labels):

    positives = labels.sum(dim=0)

    negatives = (
        labels.shape[0]
        - positives
    )

    positives = torch.clamp(
        positives,
        min=1.0,
    )

    return negatives / positives


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    all_logits = []
    all_labels = []

    for (
        text,
        photograph,
        radiograph,
        labels,
    ) in loader:

        text = text.to(
            device,
            non_blocking=True,
        )

        photograph = photograph.to(
            device,
            non_blocking=True,
        )

        radiograph = radiograph.to(
            device,
            non_blocking=True,
        )

        logits = model(
            text=text,
            photograph=photograph,
            radiograph=radiograph,
        )

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

    probabilities = torch.sigmoid(
        logits
    ).numpy()

    return (
        labels.numpy(),
        probabilities,
    )


# =============================================================================
# Metrics
# =============================================================================

def calculate_metrics(
    y_true,
    probabilities,
    thresholds,
):

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
    )

    thresholds = np.asarray(
        thresholds,
        dtype=np.float32,
    )

    predictions = (
        probabilities
        >= thresholds[None, :]
    ).astype(np.float32)

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y_true,
        predictions,
        average="micro",
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    try:
        auroc = roc_auc_score(
            y_true,
            probabilities,
            average="macro",
        )
    except ValueError:
        auroc = float("nan")

    per_label_f1 = f1_score(
        y_true,
        predictions,
        average=None,
        zero_division=0,
    )

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "auroc": float(auroc),
        "accuracy": float(accuracy),
        "per_label_f1": [
            float(x)
            for x in per_label_f1
        ],
    }


# =============================================================================
# Threshold optimization
# =============================================================================

def optimize_thresholds(
    y_true,
    probabilities,
):

    thresholds = np.arange(
        0.05,
        1.00,
        0.05,
    )

    selected = []

    for label_idx in range(
        y_true.shape[1]
    ):

        best_f1 = -1.0
        best_threshold = 0.50

        for threshold in thresholds:

            predictions = (
                probabilities[:, label_idx]
                >= threshold
            ).astype(np.float32)

            f1 = __import__(
                "sklearn.metrics",
                fromlist=["f1_score"],
            ).f1_score(
                y_true[:, label_idx],
                predictions,
                zero_division=0,
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        selected.append(
            float(best_threshold)
        )

    return selected


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 80)
    print(
        "HIERARCHICAL FUSION — "
        "FROZEN V2 FEATURES"
    )
    print("=" * 80)

    set_seed(config.SEED)

    device = get_device()

    print(
        f"Device: {device}"
    )

    # ------------------------------------------------------------------
    # Load caches
    # ------------------------------------------------------------------

    train_features = load_cache(
        "train"
    )

    val_features = load_cache(
        "validation"
    )

    test_features = load_cache(
        "test"
    )

    print()
    print("Feature shapes:")

    for name, features in [
        ("train", train_features),
        ("validation", val_features),
        ("test", test_features),
    ]:

        print(
            f"{name:12s} | "
            f"text={tuple(features['text'].shape)} | "
            f"photo={tuple(features['photograph'].shape)} | "
            f"xray={tuple(features['radiograph'].shape)} | "
            f"labels={tuple(features['labels'].shape)}"
        )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    train_loader = make_loader(
        train_features,
        shuffle=True,
    )

    val_loader = make_loader(
        val_features,
        shuffle=False,
    )

    test_loader = make_loader(
        test_features,
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model = HierarchicalFusion(
        text_dim=train_features[
            "text"
        ].shape[1],
        image_dim=train_features[
            "photograph"
        ].shape[1],
        num_labels=train_features[
            "labels"
        ].shape[1],
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    pos_weights = compute_pos_weights(
        train_features["labels"]
    ).to(device)

    print()
    print(
        "Positive weights:"
    )

    for idx, weight in enumerate(
        pos_weights.tolist()
    ):

        print(
            f"label_{idx}: "
            f"{weight:.4f}"
        )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weights
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    max_epochs = 60
    patience = 10

    best_macro_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    print()
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    for epoch in range(
        1,
        max_epochs + 1,
    ):

        model.train()

        train_loss = 0.0
        train_count = 0

        for (
            text,
            photograph,
            radiograph,
            labels,
        ) in train_loader:

            text = text.to(
                device,
                non_blocking=True,
            )

            photograph = photograph.to(
                device,
                non_blocking=True,
            )

            radiograph = radiograph.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                text=text,
                photograph=photograph,
                radiograph=radiograph,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = labels.shape[0]

            train_loss += (
                loss.item()
                * batch_size
            )

            train_count += batch_size

        train_loss /= train_count

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        y_val, p_val = evaluate(
            model,
            val_loader,
            device,
        )

        val_metrics = calculate_metrics(
            y_val,
            p_val,
            [0.5] * y_val.shape[1],
        )

        val_macro_f1 = (
            val_metrics["macro_f1"]
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Macro F1 "
            f"{val_macro_f1:.4f} | "
            f"Val Micro F1 "
            f"{val_metrics['micro_f1']:.4f} | "
            f"Val AUROC "
            f"{val_metrics['auroc']:.4f}"
        )

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = (
                val_macro_f1
            )

            best_epoch = epoch

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= patience
            ):

                print(
                    f"Early stopping at "
                    f"epoch {epoch}."
                )

                break

    # ------------------------------------------------------------------
    # Restore best
    # ------------------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    print()
    print("=" * 80)
    print("BEST MODEL")
    print("=" * 80)

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Validation Macro F1: "
        f"{best_macro_f1:.4f}"
    )

    # ------------------------------------------------------------------
    # Validation threshold selection
    # ------------------------------------------------------------------

    y_val, p_val = evaluate(
        model,
        val_loader,
        device,
    )

    default_val = calculate_metrics(
        y_val,
        p_val,
        [0.5] * y_val.shape[1],
    )

    thresholds = optimize_thresholds(
        y_val,
        p_val,
    )

    optimized_val = calculate_metrics(
        y_val,
        p_val,
        thresholds,
    )

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    print(
        f"Default Macro F1:   "
        f"{default_val['macro_f1']:.4f}"
    )

    print(
        f"Optimized Macro F1: "
        f"{optimized_val['macro_f1']:.4f}"
    )

    print(
        f"Micro F1:            "
        f"{optimized_val['micro_f1']:.4f}"
    )

    print(
        f"AUROC:               "
        f"{optimized_val['auroc']:.4f}"
    )

    print(
        "Thresholds: "
        f"{thresholds}"
    )

    # ------------------------------------------------------------------
    # FINAL TEST
    # ------------------------------------------------------------------

    y_test, p_test = evaluate(
        model,
        test_loader,
        device,
    )

    default_test = calculate_metrics(
        y_test,
        p_test,
        [0.5] * y_test.shape[1],
    )

    final_test = calculate_metrics(
        y_test,
        p_test,
        thresholds,
    )

    print()
    print("=" * 80)
    print(
        "FINAL TEST — "
        "VALIDATION THRESHOLDS FROZEN"
    )
    print("=" * 80)

    print(
        f"Default Macro F1: "
        f"{default_test['macro_f1']:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{final_test['macro_f1']:.4f}"
    )

    print(
        f"Micro F1:          "
        f"{final_test['micro_f1']:.4f}"
    )

    print(
        f"AUROC:             "
        f"{final_test['auroc']:.4f}"
    )

    print(
        f"Accuracy:          "
        f"{final_test['accuracy']:.4f}"
    )

    print(
        "Thresholds used: "
        f"{thresholds}"
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    output_dir = (
        config.RESULT_ROOT
        / "hierarchical_fusion"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "model": (
            "hierarchical_visual_text_fusion"
        ),
        "source_checkpoint": str(
            config.SOURCE_CHECKPOINT
        ),
        "best_epoch": best_epoch,
        "validation_default": default_val,
        "validation_threshold_optimized": (
            optimized_val
        ),
        "validation_thresholds": thresholds,
        "test_default": default_test,
        "test_with_validation_thresholds": (
            final_test
        ),
        "protocol": (
            "Frozen V2 representations. "
            "Train-only fitting. "
            "Validation-only model and "
            "threshold selection. "
            "Test evaluated once with "
            "frozen validation thresholds."
        ),
    }

    result_path = (
        output_dir
        / "results.json"
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

    print()
    print(
        f"Results saved to:\n"
        f"{result_path}"
    )

    print()
    print("=" * 80)
    print(
        "TEST WAS NOT USED FOR TRAINING "
        "OR MODEL SELECTION."
    )
    print(
        "NO TEST THRESHOLD OPTIMIZATION."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()