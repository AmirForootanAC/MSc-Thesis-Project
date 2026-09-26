"""Evaluate class-wise logistic regression on the held-out test set.

Protocol
--------
- Fit models on TRAIN only.
- Select thresholds on VALIDATION only.
- Freeze thresholds.
- Evaluate TEST once.
- No test threshold optimization.
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


ROOT = Path(
    "results/baseline/final_strong_fusion_screen/feature_cache"
)

TRAIN_PATH = ROOT / "train_v1.pt"
VAL_PATH = ROOT / "validation_v1.pt"
TEST_PATH = ROOT / "test_v1.pt"

NUM_LABELS = 6

LABEL_NAMES = [
    "caries",
    "gingivitis",
    "malocclusion",
    "pulpitis",
    "tooth_loss",
    "tooth_structure_loss",
]

# Validation-derived threshold search.
THRESHOLDS = np.arange(
    0.05,
    1.00,
    0.05,
)

C_VALUES = [
    10.0,
    0.01,
]


def load_features(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def make_X(features):
    return torch.cat(
        [
            features["text"],
            features["photograph"],
            features["radiograph"],
        ],
        dim=1,
    ).numpy().astype(np.float64)


def make_y(features):
    return (
        features["labels"]
        .numpy()
        .astype(np.int64)
    )


def compute_pos_weights(y):
    positives = y.sum(axis=0)
    negatives = y.shape[0] - positives

    positives = np.maximum(
        positives,
        1.0,
    )

    return negatives / positives


def fit_classwise_models(
    X_train,
    y_train,
    C,
):
    models = []

    pos_weights = compute_pos_weights(
        y_train
    )

    for label_idx in range(NUM_LABELS):
        model = LogisticRegression(
            C=C,
            class_weight=None,
            max_iter=5000,
            solver="lbfgs",
            random_state=42,
        )

        # Equivalent class balancing through
        # per-sample weights, matching the earlier
        # class-wise weighted setup.
        sample_weights = np.where(
            y_train[:, label_idx] == 1,
            pos_weights[label_idx],
            1.0,
        )

        model.fit(
            X_train,
            y_train[:, label_idx],
            sample_weight=sample_weights,
        )

        models.append(model)

    return models


def predict_probabilities(
    models,
    X,
):
    probabilities = np.zeros(
        (X.shape[0], NUM_LABELS),
        dtype=np.float64,
    )

    for label_idx, model in enumerate(models):
        probabilities[:, label_idx] = (
            model.predict_proba(X)[:, 1]
        )

    return probabilities


def evaluate_at_thresholds(
    y_true,
    probabilities,
    thresholds,
):
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
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "per_label_f1": per_label_f1,
    }


def optimize_validation_thresholds(
    y_val,
    val_probabilities,
):
    thresholds = np.full(
        NUM_LABELS,
        0.5,
        dtype=np.float64,
    )

    for label_idx in range(NUM_LABELS):
        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in THRESHOLDS:
            predictions = (
                val_probabilities[:, label_idx]
                >= threshold
            ).astype(np.int64)

            score = f1_score(
                y_val[:, label_idx],
                predictions,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        thresholds[label_idx] = (
            best_threshold
        )

    return thresholds


def main():
    print("=" * 80)
    print("CLASS-WISE LOGISTIC REGRESSION — HELD-OUT TEST")
    print("=" * 80)

    print("\nLoading feature caches...")

    train = load_features(TRAIN_PATH)
    val = load_features(VAL_PATH)
    test = load_features(TEST_PATH)

    X_train = make_X(train)
    y_train = make_y(train)

    X_val = make_X(val)
    y_val = make_y(val)

    X_test = make_X(test)
    y_test = make_y(test)

    print(
        f"Train X: {X_train.shape}, y: {y_train.shape}"
    )
    print(
        f"Val   X: {X_val.shape}, y: {y_val.shape}"
    )
    print(
        f"Test  X: {X_test.shape}, y: {y_test.shape}"
    )

    if X_test.shape[0] != 633:
        raise RuntimeError(
            f"Expected 633 test samples, "
            f"found {X_test.shape[0]}"
        )

    print(
        "\nIMPORTANT:"
        "\n  Models fit on TRAIN only."
        "\n  Thresholds selected on VALIDATION only."
        "\n  TEST is used only for final evaluation."
    )

    for C in C_VALUES:
        print("\n" + "-" * 80)
        print(f"C = {C}")
        print("-" * 80)

        # --------------------------------------------------------
        # Fit on TRAIN ONLY
        # --------------------------------------------------------

        models = fit_classwise_models(
            X_train,
            y_train,
            C,
        )

        # --------------------------------------------------------
        # Validation probabilities
        # --------------------------------------------------------

        val_probabilities = (
            predict_probabilities(
                models,
                X_val,
            )
        )

        # --------------------------------------------------------
        # Default threshold = 0.5
        # --------------------------------------------------------

        default_thresholds = np.full(
            NUM_LABELS,
            0.5,
            dtype=np.float64,
        )

        val_default = evaluate_at_thresholds(
            y_val,
            val_probabilities,
            default_thresholds,
        )

        # --------------------------------------------------------
        # Optimize thresholds on VALIDATION ONLY
        # --------------------------------------------------------

        optimized_thresholds = (
            optimize_validation_thresholds(
                y_val,
                val_probabilities,
            )
        )

        val_optimized = evaluate_at_thresholds(
            y_val,
            val_probabilities,
            optimized_thresholds,
        )

        print(
            "\nValidation @0.5:"
        )
        print(
            f"  Macro F1: "
            f"{val_default['macro_f1']:.4f}"
        )
        print(
            f"  Micro F1: "
            f"{val_default['micro_f1']:.4f}"
        )
        print(
            f"  AUROC:    "
            f"{val_default['auroc']:.4f}"
        )

        print(
            "\nValidation optimized:"
        )
        print(
            f"  Macro F1: "
            f"{val_optimized['macro_f1']:.4f}"
        )
        print(
            f"  Micro F1: "
            f"{val_optimized['micro_f1']:.4f}"
        )
        print(
            f"  AUROC:    "
            f"{val_optimized['auroc']:.4f}"
        )
        print(
            "  Thresholds: "
            f"{optimized_thresholds.tolist()}"
        )

        # --------------------------------------------------------
        # TEST PROBABILITIES
        # --------------------------------------------------------

        test_probabilities = (
            predict_probabilities(
                models,
                X_test,
            )
        )

        # --------------------------------------------------------
        # TEST @ 0.5
        # --------------------------------------------------------

        test_default = evaluate_at_thresholds(
            y_test,
            test_probabilities,
            default_thresholds,
        )

        # --------------------------------------------------------
        # TEST using FROZEN validation thresholds
        # --------------------------------------------------------

        test_frozen = evaluate_at_thresholds(
            y_test,
            test_probabilities,
            optimized_thresholds,
        )

        # --------------------------------------------------------
        # Report
        # --------------------------------------------------------

        print("\nTEST @0.5:")
        print(
            f"  Macro F1: "
            f"{test_default['macro_f1']:.4f}"
        )
        print(
            f"  Micro F1: "
            f"{test_default['micro_f1']:.4f}"
        )
        print(
            f"  Accuracy: "
            f"{test_default['accuracy']:.4f}"
        )
        print(
            f"  AUROC:    "
            f"{test_default['auroc']:.4f}"
        )

        print(
            "\nTEST @ VALIDATION-FROZEN THRESHOLDS:"
        )
        print(
            f"  Macro F1: "
            f"{test_frozen['macro_f1']:.4f}"
        )
        print(
            f"  Micro F1: "
            f"{test_frozen['micro_f1']:.4f}"
        )
        print(
            f"  Accuracy: "
            f"{test_frozen['accuracy']:.4f}"
        )
        print(
            f"  AUROC:    "
            f"{test_frozen['auroc']:.4f}"
        )

        print("\nPer-label F1:")
        for label_idx, label_name in enumerate(
            LABEL_NAMES
        ):
            print(
                f"  {label_name:<22} "
                f"{test_frozen['per_label_f1'][label_idx]:.4f}"
            )

        print(
            "\nThresholds used on TEST:"
        )
        print(
            f"  {optimized_thresholds.tolist()}"
        )

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 80)
    print(
        "No threshold optimization was performed on TEST."
    )
    print(
        "TEST predictions were generated only after "
        "training and validation threshold selection."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
