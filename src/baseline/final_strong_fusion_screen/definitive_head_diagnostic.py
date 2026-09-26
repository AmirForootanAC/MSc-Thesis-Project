"""Definitive frozen-feature head diagnostic.

Compares three classifiers on the exact same frozen V2 epoch-9 features:

A) Strict linear probe:
   1024 -> 6

B) Nonlinear MLP probe:
   LayerNorm(1024) -> 128 -> GELU -> Dropout -> 6

C) Original V2 fusion classifier:
   1024 -> 256 -> ReLU -> Dropout -> 6

C is NOT retrained. Its classifier weights are taken directly from
the V2 epoch-009 checkpoint.

Train/validation are used for fitting A and B and selecting thresholds.
The test set is evaluated only after all validation decisions are fixed.
"""

from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path("results/baseline/final_strong_fusion_screen")

FEATURE_CACHE = ROOT / "feature_cache"

TRAIN_FEATURES = FEATURE_CACHE / "train_v1.pt"
VAL_FEATURES = FEATURE_CACHE / "validation_v1.pt"
TEST_FEATURES = FEATURE_CACHE / "test_v1.pt"

V2_CHECKPOINT = Path(
    "results/baseline/final_strong_v2/"
    "full_multimodal/checkpoints/epoch_009.pt"
)

OUTPUT_DIR = ROOT / "definitive_head_diagnostic"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_JSON = OUTPUT_DIR / "results.json"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

NUM_LABELS = 6
INPUT_DIM = 1024

BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 7

LR = 1e-3
WEIGHT_DECAY = 1e-4

SEED = 42

LABEL_NAMES = [
    "caries",
    "gingivitis",
    "malocclusion",
    "pulpitis",
    "tooth_loss",
    "tooth_structure_loss",
]


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------

def load_feature_file(path):
    data = torch.load(
        path,
        map_location="cpu",
    )

    print(f"\nLoaded: {path}")

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected dict in {path}, "
            f"got {type(data)}"
        )

    print(
        "Keys:",
        list(data.keys()),
    )

    labels = data["labels"].float()

    text = data["text"].float()
    photograph = data["photograph"].float()
    radiograph = data["radiograph"].float()

    # IMPORTANT:
    # This is exactly the order used by FinalStrongV2:
    #
    # photograph -> radiograph -> text
    #
    features = torch.cat(
        [
            photograph,
            radiograph,
            text,
        ],
        dim=1,
    )

    if features.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Expected {INPUT_DIM} features, "
            f"found {features.shape[1]}"
        )

    return {
        "features": features,
        "labels": labels,
    }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def safe_auc(y_true, y_score):
    try:
        return float(
            roc_auc_score(
                y_true,
                y_score,
                average="macro",
            )
        )
    except ValueError:
        return float("nan")


def calculate_metrics(
    y_true,
    probabilities,
    thresholds=None,
):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if thresholds is None:
        thresholds = np.full(
            NUM_LABELS,
            0.5,
            dtype=np.float32,
        )

    thresholds = np.asarray(
        thresholds,
        dtype=np.float32,
    )

    predictions = (
        probabilities >= thresholds
    ).astype(np.int64)

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

    auc = safe_auc(
        y_true,
        probabilities,
    )

    per_label = {}

    for i, name in enumerate(
        LABEL_NAMES
    ):
        label_f1 = f1_score(
            y_true[:, i],
            predictions[:, i],
            zero_division=0,
        )

        label_auc = safe_auc(
            y_true[:, i:i + 1],
            probabilities[:, i:i + 1],
        )

        true_prevalence = float(
            y_true[:, i].mean()
        )

        predicted_prevalence = float(
            predictions[:, i].mean()
        )

        per_label[name] = {
            "f1": float(label_f1),
            "auroc": float(label_auc),
            "true_prevalence": true_prevalence,
            "predicted_prevalence": predicted_prevalence,
        }

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "accuracy": float(accuracy),
        "auroc": float(auc),
        "thresholds": thresholds.tolist(),
        "per_label": per_label,
    }


# ---------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------

def optimize_thresholds(
    y_true,
    probabilities,
):
    best_thresholds = []
    best_scores = []

    grid = np.arange(
        0.05,
        1.00,
        0.05,
    )

    for label_idx in range(NUM_LABELS):
        best_f1 = -1.0
        best_threshold = 0.5

        for threshold in grid:
            predictions = (
                probabilities[:, label_idx]
                >= threshold
            ).astype(np.int64)

            score = f1_score(
                y_true[:, label_idx],
                predictions,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(
                    threshold
                )

        best_thresholds.append(
            best_threshold
        )

        best_scores.append(
            best_f1
        )

    return (
        np.asarray(
            best_thresholds,
            dtype=np.float32,
        ),
        best_scores,
    )


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class StrictLinearProbe(nn.Module):
    def __init__(
        self,
        input_dim=INPUT_DIM,
        num_labels=NUM_LABELS,
    ):
        super().__init__()

        self.classifier = nn.Linear(
            input_dim,
            num_labels,
        )

    def forward(self, x):
        return self.classifier(x)


class MLPProbe(nn.Module):
    def __init__(
        self,
        input_dim=INPUT_DIM,
        num_labels=NUM_LABELS,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),

            nn.Linear(
                input_dim,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                128,
                num_labels,
            ),
        )

    def forward(self, x):
        return self.classifier(x)


class OriginalV2Classifier(nn.Module):
    """Exact multimodal classifier from FinalStrongV2."""

    def __init__(
        self,
        input_dim=INPUT_DIM,
        num_labels=NUM_LABELS,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                input_dim,
                256,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                256,
                num_labels,
            ),
        )

    def forward(self, x):
        return self.classifier(x)


# ---------------------------------------------------------------------
# Checkpoint handling
# ---------------------------------------------------------------------

def extract_model_state_dict(checkpoint):
    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Checkpoint is not a dictionary."
        )

    candidates = [
        "model_state_dict",
        "state_dict",
        "model",
    ]

    for key in candidates:
        value = checkpoint.get(key)

        if isinstance(value, dict):
            return value

    # Some checkpoints may directly be state_dicts.
    tensor_values = [
        value
        for value in checkpoint.values()
        if torch.is_tensor(value)
    ]

    if tensor_values:
        return checkpoint

    raise RuntimeError(
        "Could not find model state_dict "
        "inside checkpoint."
    )


def extract_classifier_state_dict(
    state_dict,
):
    classifier_state = {}

    for key, value in state_dict.items():

        clean_key = key

        prefixes = [
            "module.",
            "model.",
        ]

        changed = True

        while changed:
            changed = False

            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[
                        len(prefix):
                    ]
                    changed = True

        if clean_key.startswith(
            "classifier."
        ):
            classifier_state[
                clean_key
            ] = value

    if not classifier_state:
        print(
            "\nCheckpoint keys:"
        )

        for key in state_dict.keys():
            print(
                " ",
                key,
            )

        raise RuntimeError(
            "Could not find classifier.* "
            "keys in V2 checkpoint."
        )

    return classifier_state


def load_original_v2_classifier(
    device,
):
    print(
        "\nLoading original V2 classifier:"
    )
    print(
        V2_CHECKPOINT
    )

    checkpoint = torch.load(
        V2_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_model_state_dict(
        checkpoint
    )

    classifier_state = (
        extract_classifier_state_dict(
            state_dict
        )
    )

    model = OriginalV2Classifier(
        input_dim=INPUT_DIM,
        num_labels=NUM_LABELS,
    )

    missing, unexpected = (
        model.load_state_dict(
            classifier_state,
            strict=False,
        )
    )

    if missing:
        raise RuntimeError(
            f"Missing V2 classifier keys: "
            f"{missing}"
        )

    if unexpected:
        print(
            "Unexpected classifier keys:",
            unexpected,
        )

    model.to(device)
    model.eval()

    print(
        "Original V2 classifier loaded."
    )

    return model


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def compute_pos_weights(
    labels,
):
    positives = labels.sum(
        dim=0
    )

    negatives = (
        labels.shape[0]
        - positives
    )

    weights = (
        negatives
        / positives.clamp_min(1.0)
    )

    return weights


def train_probe(
    model,
    train_x,
    train_y,
    val_x,
    val_y,
    device,
    model_name,
):
    print(
        f"\n{'=' * 70}"
    )
    print(
        f"TRAINING: {model_name}"
    )
    print(
        f"{'=' * 70}"
    )

    model = model.to(device)

    train_dataset = TensorDataset(
        train_x,
        train_y,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    pos_weights = compute_pos_weights(
        train_y
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(
                device,
                non_blocking=True,
            )

            batch_y = batch_y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_x
            )

            loss = criterion(
                logits,
                batch_y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            train_losses.append(
                float(loss.item())
            )

        # Validation
        model.eval()

        with torch.no_grad():
            val_logits = []

            for start in range(
                0,
                len(val_x),
                BATCH_SIZE,
            ):
                batch_x = val_x[
                    start:
                    start + BATCH_SIZE
                ].to(device)

                logits = model(
                    batch_x
                )

                val_logits.append(
                    logits.cpu()
                )

            val_logits = torch.cat(
                val_logits,
                dim=0,
            )

        val_probabilities = (
            torch.sigmoid(
                val_logits
            ).numpy()
        )

        val_metrics = calculate_metrics(
            val_y.numpy(),
            val_probabilities,
        )

        val_f1 = val_metrics[
            "macro_f1"
        ]

        mean_loss = float(
            np.mean(train_losses)
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {mean_loss:.4f} | "
            f"val Macro F1 "
            f"{val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

        else:
            patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):
            print(
                f"Early stopping at epoch "
                f"{epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No best model state saved."
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    with torch.no_grad():
        val_logits = []

        for start in range(
            0,
            len(val_x),
            BATCH_SIZE,
        ):
            batch_x = val_x[
                start:
                start + BATCH_SIZE
            ].to(device)

            val_logits.append(
                model(
                    batch_x
                ).cpu()
            )

        val_logits = torch.cat(
            val_logits,
            dim=0,
        )

    val_probabilities = (
        torch.sigmoid(
            val_logits
        ).numpy()
    )

    default_metrics = calculate_metrics(
        val_y.numpy(),
        val_probabilities,
    )

    thresholds, _ = optimize_thresholds(
        val_y.numpy(),
        val_probabilities,
    )

    optimized_metrics = calculate_metrics(
        val_y.numpy(),
        val_probabilities,
        thresholds,
    )

    print(
        "\nBest epoch:",
        best_epoch,
    )

    print(
        "Validation default Macro F1:",
        f"{default_metrics['macro_f1']:.4f}",
    )

    print(
        "Validation optimized Macro F1:",
        f"{optimized_metrics['macro_f1']:.4f}",
    )

    print(
        "Validation thresholds:",
        thresholds.tolist(),
    )

    return {
        "model": model,
        "best_epoch": best_epoch,
        "val_default": default_metrics,
        "val_optimized": optimized_metrics,
        "thresholds": thresholds,
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def predict(
    model,
    x,
    device,
):
    model.eval()

    outputs = []

    with torch.no_grad():
        for start in range(
            0,
            len(x),
            BATCH_SIZE,
        ):
            batch_x = x[
                start:
                start + BATCH_SIZE
            ].to(device)

            outputs.append(
                model(
                    batch_x
                ).cpu()
            )

    logits = torch.cat(
        outputs,
        dim=0,
    )

    return torch.sigmoid(
        logits
    ).numpy()


def evaluate_model(
    model,
    x,
    y,
    device,
    thresholds=None,
):
    probabilities = predict(
        model,
        x,
        device,
    )

    return calculate_metrics(
        y.numpy(),
        probabilities,
        thresholds,
    )


def print_summary(
    name,
    metrics,
):
    print(
        f"\n{name}"
    )
    print(
        "-" * len(name)
    )

    print(
        "Macro F1 :",
        f"{metrics['macro_f1']:.4f}",
    )

    print(
        "Micro F1 :",
        f"{metrics['micro_f1']:.4f}",
    )

    print(
        "AUROC    :",
        f"{metrics['auroc']:.4f}",
    )

    print(
        "Accuracy :",
        f"{metrics['accuracy']:.4f}",
    )

    print(
        "Thresholds:",
        metrics["thresholds"],
    )

    for label in LABEL_NAMES:
        item = metrics[
            "per_label"
        ][label]

        print(
            f"  {label:24s} "
            f"F1={item['f1']:.4f} "
            f"AUC={item['auroc']:.4f} "
            f"true={item['true_prevalence']:.3f} "
            f"pred={item['predicted_prevalence']:.3f}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # -------------------------------------------------------------
    # Load exact frozen features
    # -------------------------------------------------------------

    train = load_feature_file(
        TRAIN_FEATURES
    )

    val = load_feature_file(
        VAL_FEATURES
    )

    test = load_feature_file(
        TEST_FEATURES
    )

    train_x = train["features"]
    train_y = train["labels"]

    val_x = val["features"]
    val_y = val["labels"]

    test_x = test["features"]
    test_y = test["labels"]

    print(
        "\nFeature shapes:"
    )

    print(
        "Train:",
        tuple(train_x.shape),
        tuple(train_y.shape),
    )

    print(
        "Val:",
        tuple(val_x.shape),
        tuple(val_y.shape),
    )

    print(
        "Test:",
        tuple(test_x.shape),
        tuple(test_y.shape),
    )

    # -------------------------------------------------------------
    # A. Strict linear probe
    # -------------------------------------------------------------

    linear_model = StrictLinearProbe()

    linear_result = train_probe(
        linear_model,
        train_x,
        train_y,
        val_x,
        val_y,
        device,
        "A. Strict Linear Probe",
    )

    # -------------------------------------------------------------
    # B. Nonlinear MLP probe
    # -------------------------------------------------------------

    set_seed(SEED)

    mlp_model = MLPProbe()

    mlp_result = train_probe(
        mlp_model,
        train_x,
        train_y,
        val_x,
        val_y,
        device,
        "B. MLP Probe",
    )

    # -------------------------------------------------------------
    # C. Original V2 classifier
    # -------------------------------------------------------------

    v2_model = load_original_v2_classifier(
        device
    )

    v2_val_default = evaluate_model(
        v2_model,
        val_x,
        val_y,
        device,
    )

    # V2 checkpoint itself is not re-trained.
    # Threshold optimization is performed on validation only
    # for diagnostic purposes.
    v2_val_thresholds, _ = (
        optimize_thresholds(
            val_y.numpy(),
            predict(
                v2_model,
                val_x,
                device,
            ),
        )
    )

    v2_val_optimized = evaluate_model(
        v2_model,
        val_x,
        val_y,
        device,
        v2_val_thresholds,
    )

    # -------------------------------------------------------------
    # Test evaluation
    # -------------------------------------------------------------
    #
    # IMPORTANT:
    # Test is evaluated only after validation thresholds
    # are fixed.
    #

    linear_test_default = evaluate_model(
        linear_result["model"],
        test_x,
        test_y,
        device,
    )

    linear_test_thresholded = (
        evaluate_model(
            linear_result["model"],
            test_x,
            test_y,
            device,
            linear_result["thresholds"],
        )
    )

    mlp_test_default = evaluate_model(
        mlp_result["model"],
        test_x,
        test_y,
        device,
    )

    mlp_test_thresholded = (
        evaluate_model(
            mlp_result["model"],
            test_x,
            test_y,
            device,
            mlp_result["thresholds"],
        )
    )

    v2_test_default = evaluate_model(
        v2_model,
        test_x,
        test_y,
        device,
    )

    v2_test_thresholded = (
        evaluate_model(
            v2_model,
            test_x,
            test_y,
            device,
            v2_val_thresholds,
        )
    )

    # -------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "VALIDATION COMPARISON"
    )

    print(
        "=" * 80
    )

    print_summary(
        "A. Strict Linear Probe — validation @ 0.5",
        linear_result["val_default"],
    )

    print_summary(
        "A. Strict Linear Probe — validation optimized",
        linear_result["val_optimized"],
    )

    print_summary(
        "B. MLP Probe — validation @ 0.5",
        mlp_result["val_default"],
    )

    print_summary(
        "B. MLP Probe — validation optimized",
        mlp_result["val_optimized"],
    )

    print_summary(
        "C. Original V2 Classifier — validation @ 0.5",
        v2_val_default,
    )

    print_summary(
        "C. Original V2 Classifier — validation optimized",
        v2_val_optimized,
    )

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "TEST COMPARISON"
    )

    print(
        "=" * 80
    )

    print_summary(
        "A. Strict Linear Probe — test @ 0.5",
        linear_test_default,
    )

    print_summary(
        "A. Strict Linear Probe — test using VAL thresholds",
        linear_test_thresholded,
    )

    print_summary(
        "B. MLP Probe — test @ 0.5",
        mlp_test_default,
    )

    print_summary(
        "B. MLP Probe — test using VAL thresholds",
        mlp_test_thresholded,
    )

    print_summary(
        "C. Original V2 Classifier — test @ 0.5",
        v2_test_default,
    )

    print_summary(
        "C. Original V2 Classifier — test using VAL thresholds",
        v2_test_thresholded,
    )

    # -------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------

    results = {
        "experiment": (
            "definitive_head_diagnostic"
        ),

        "seed": SEED,

        "feature_source": str(
            V2_CHECKPOINT
        ),

        "feature_dimensions": {
            "photograph": 128,
            "radiograph": 128,
            "text": 768,
            "total": 1024,
        },

        "train_samples": int(
            len(train_x)
        ),

        "validation_samples": int(
            len(val_x)
        ),

        "test_samples": int(
            len(test_x)
        ),

        "models": {
            "strict_linear_probe": {
                "architecture":
                    "1024 -> 6",
                "best_epoch":
                    linear_result["best_epoch"],
                "validation_default":
                    linear_result["val_default"],
                "validation_optimized":
                    linear_result["val_optimized"],
                "test_default":
                    linear_test_default,
                "test_validation_thresholds":
                    linear_test_thresholded,
            },

            "mlp_probe": {
                "architecture":
                    "LayerNorm(1024) -> "
                    "Linear(128) -> GELU -> "
                    "Dropout(0.20) -> 6",
                "best_epoch":
                    mlp_result["best_epoch"],
                "validation_default":
                    mlp_result["val_default"],
                "validation_optimized":
                    mlp_result["val_optimized"],
                "test_default":
                    mlp_test_default,
                "test_validation_thresholds":
                    mlp_test_thresholded,
            },

            "original_v2_classifier": {
                "architecture":
                    "1024 -> 256 -> 6",
                "checkpoint":
                    str(V2_CHECKPOINT),
                "validation_default":
                    v2_val_default,
                "validation_optimized":
                    v2_val_optimized,
                "test_default":
                    v2_test_default,
                "test_validation_thresholds":
                    v2_test_thresholded,
            },
        },
    }

    with open(
        RESULTS_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        "\nResults saved to:"
    )
    print(
        RESULTS_JSON
    )

    print(
        "\nTEST SET WAS NEVER USED FOR "
        "TRAINING OR THRESHOLD OPTIMIZATION."
    )


if __name__ == "__main__":
    main()