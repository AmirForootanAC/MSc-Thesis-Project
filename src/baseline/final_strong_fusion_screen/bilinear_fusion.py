"""
Low-rank bilinear multimodal fusion on frozen V2 representations.

Modalities:
    text       : 768
    photograph : 128
    radiograph : 128

Fusion:
    - modality projections
    - pairwise multiplicative interactions
    - triple interaction
    - original projected features
    - six independent label-specific heads

Protocol:
    - train: fit
    - validation: model selection + threshold selection
    - test: one final evaluation using frozen validation thresholds
"""

import json
import random
from pathlib import Path

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
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =============================================================================
# Cache
# =============================================================================

def load_cache(split):
    path = config.CACHE_ROOT / f"{split}_{config.CACHE_VERSION}.pt"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature cache:\n{path}"
        )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = [
        "text",
        "photograph",
        "radiograph",
        "labels",
    ]

    for key in required:
        if key not in data:
            raise RuntimeError(
                f"{split} cache missing key: {key}"
            )

    return data


# =============================================================================
# Dataset
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
# Model
# =============================================================================

class LowRankBilinearFusion(nn.Module):
    """
    Low-rank multiplicative fusion.

    Each modality is projected into a shared latent space.

    Pairwise interactions:
        text * photograph
        text * radiograph
        photograph * radiograph

    Triple interaction:
        text * photograph * radiograph

    The original modality representations are retained as well.

    Six independent label heads operate on the fused representation.
    """

    def __init__(
        self,
        text_dim=768,
        image_dim=128,
        hidden_dim=128,
        interaction_dim=64,
        num_labels=6,
        dropout=0.20,
    ):
        super().__init__()

        self.num_labels = num_labels

        # ------------------------------------------------------------------
        # Modality projections
        # ------------------------------------------------------------------

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.photo_projection = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.xray_projection = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # Low-rank interaction projections
        # ------------------------------------------------------------------

        self.text_interaction = nn.Linear(
            hidden_dim,
            interaction_dim,
            bias=False,
        )

        self.photo_interaction = nn.Linear(
            hidden_dim,
            interaction_dim,
            bias=False,
        )

        self.xray_interaction = nn.Linear(
            hidden_dim,
            interaction_dim,
            bias=False,
        )

        # ------------------------------------------------------------------
        # Fusion dimension
        #
        # 3 original projected modalities
        # 3 pairwise interactions
        # 1 triple interaction
        # ------------------------------------------------------------------

        fusion_dim = (
            hidden_dim * 3
            + interaction_dim * 4
        )

        self.fusion_norm = nn.LayerNorm(fusion_dim)

        self.fusion_projection = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # Six independent label heads
        # ------------------------------------------------------------------

        self.label_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(256, 64),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
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
        text_h = self.text_projection(text)
        photo_h = self.photo_projection(photograph)
        xray_h = self.xray_projection(radiograph)

        # Low-rank latent representations
        text_i = self.text_interaction(text_h)
        photo_i = self.photo_interaction(photo_h)
        xray_i = self.xray_interaction(xray_h)

        # Pairwise multiplicative interactions
        text_photo = text_i * photo_i
        text_xray = text_i * xray_i
        photo_xray = photo_i * xray_i

        # Triple interaction
        text_photo_xray = (
            text_i
            * photo_i
            * xray_i
        )

        fused = torch.cat(
            [
                text_h,
                photo_h,
                xray_h,
                text_photo,
                text_xray,
                photo_xray,
                text_photo_xray,
            ],
            dim=1,
        )

        fused = self.fusion_norm(fused)
        fused = self.fusion_projection(fused)

        logits = torch.cat(
            [
                head(fused)
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
    negatives = labels.shape[0] - positives

    positives = torch.clamp(
        positives,
        min=1.0,
    )

    return negatives / positives


# =============================================================================
# Threshold optimization
# =============================================================================

def optimize_thresholds(y_true, probabilities):
    thresholds = np.arange(
        0.05,
        1.00,
        0.05,
    )

    best_thresholds = []

    for label_idx in range(y_true.shape[1]):

        best_f1 = -1.0
        best_threshold = 0.50

        for threshold in thresholds:

            predictions = (
                probabilities[:, label_idx]
                >= threshold
            ).astype(np.float32)

            tp = np.sum(
                (predictions == 1)
                & (y_true[:, label_idx] == 1)
            )

            fp = np.sum(
                (predictions == 1)
                & (y_true[:, label_idx] == 0)
            )

            fn = np.sum(
                (predictions == 0)
                & (y_true[:, label_idx] == 1)
            )

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0.0
            )

            f1 = (
                2 * precision * recall
                / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        best_thresholds.append(
            float(best_threshold)
        )

    return best_thresholds


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

    y_true = labels.numpy()

    return y_true, probabilities


# =============================================================================
# Metrics
# =============================================================================

def metrics_with_thresholds(
    y_true,
    probabilities,
    thresholds,
):
    thresholds = np.asarray(
        thresholds,
        dtype=np.float32,
    )

    predictions = (
        probabilities >= thresholds[None, :]
    ).astype(np.float32)

    # Compute metrics directly here.
    # This avoids passing arguments in the wrong
    # order to the V2 compute_metrics implementation.

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
    )

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
            float(value)
            for value in per_label_f1
        ],
    }


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 80)
    print("LOW-RANK BILINEAR FUSION — FROZEN V2 FEATURES")
    print("=" * 80)

    set_seed(config.SEED)

    device = get_device()

    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load features
    # ------------------------------------------------------------------

    train_features = load_cache("train")
    val_features = load_cache("validation")
    test_features = load_cache("test")

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

    model = LowRankBilinearFusion(
        text_dim=train_features["text"].shape[1],
        image_dim=train_features["photograph"].shape[1],
        hidden_dim=128,
        interaction_dim=64,
        num_labels=train_features["labels"].shape[1],
        dropout=0.20,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(f"Model parameters: {parameter_count:,}")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    pos_weights = compute_pos_weights(
        train_features["labels"]
    ).to(device)

    print()
    print("Positive weights:")

    for idx, weight in enumerate(
        pos_weights.tolist()
    ):
        print(
            f"label_{idx}: {weight:.4f}"
        )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
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

    for epoch in range(1, max_epochs + 1):

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

        val_metrics = metrics_with_thresholds(
            y_val,
            p_val,
            [0.5] * y_val.shape[1],
        )

        val_macro_f1 = float(
            val_metrics["macro_f1"]
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Macro F1 {val_macro_f1:.4f} | "
            f"Val Micro F1 {val_metrics['micro_f1']:.4f} | "
            f"Val AUROC {val_metrics['auroc']:.4f}"
        )

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}."
                )
                break

    # ------------------------------------------------------------------
    # Restore best model
    # ------------------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    print()
    print("=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(f"Best epoch: {best_epoch}")
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

    default_val_metrics = metrics_with_thresholds(
        y_val,
        p_val,
        [0.5] * y_val.shape[1],
    )

    thresholds = optimize_thresholds(
        y_val,
        p_val,
    )

    optimized_val_metrics = metrics_with_thresholds(
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
        f"{default_val_metrics['macro_f1']:.4f}"
    )

    print(
        f"Optimized Macro F1: "
        f"{optimized_val_metrics['macro_f1']:.4f}"
    )

    print(
        f"Micro F1:            "
        f"{optimized_val_metrics['micro_f1']:.4f}"
    )

    print(
        f"AUROC:               "
        f"{optimized_val_metrics['auroc']:.4f}"
    )

    print(
        "Thresholds: "
        + str(thresholds)
    )

    # ------------------------------------------------------------------
    # FINAL TEST
    #
    # Thresholds are frozen from validation.
    # No test optimization.
    # ------------------------------------------------------------------

    y_test, p_test = evaluate(
        model,
        test_loader,
        device,
    )

    test_default_metrics = metrics_with_thresholds(
        y_test,
        p_test,
        [0.5] * y_test.shape[1],
    )

    test_metrics = metrics_with_thresholds(
        y_test,
        p_test,
        thresholds,
    )

    print()
    print("=" * 80)
    print("FINAL TEST — VALIDATION THRESHOLDS FROZEN")
    print("=" * 80)

    print(
        f"Default Macro F1: "
        f"{test_default_metrics['macro_f1']:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Micro F1:          "
        f"{test_metrics['micro_f1']:.4f}"
    )

    print(
        f"AUROC:             "
        f"{test_metrics['auroc']:.4f}"
    )

    print(
        f"Accuracy:          "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        "Thresholds used: "
        + str(thresholds)
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    output_dir = (
        config.RESULT_ROOT
        / "bilinear_fusion"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "model": "low_rank_bilinear_fusion",
        "source_checkpoint": str(
            config.SOURCE_CHECKPOINT
        ),
        "best_epoch": best_epoch,
        "validation_default": default_val_metrics,
        "validation_threshold_optimized": optimized_val_metrics,
        "validation_thresholds": thresholds,
        "test_default": test_default_metrics,
        "test_with_validation_thresholds": test_metrics,
        "test_protocol": (
            "Test used only once after model "
            "selection and validation threshold freezing."
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
        f"Results saved to:\n{result_path}"
    )

    print()
    print("=" * 80)
    print("TEST WAS NOT USED FOR TRAINING OR MODEL SELECTION.")
    print("NO TEST THRESHOLD OPTIMIZATION.")
    print("=" * 80)


if __name__ == "__main__":
    main()