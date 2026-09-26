# src/baseline/final_strong_fusion_screen/classwise_linear.py

from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


ROOT = Path("results/baseline/final_strong_fusion_screen/feature_cache")

TRAIN_PATH = ROOT / "train_v1.pt"
VAL_PATH = ROOT / "validation_v1.pt"

LABELS = [
    "caries",
    "gingivitis",
    "malocclusion",
    "pulpitis",
    "tooth_loss",
    "tooth_structure_loss",
]

POS_WEIGHTS = np.array(
    [
        8.0031,
        6.2291,
        1.0172,
        21.2348,
        39.2055,
        27.7745,
    ],
    dtype=np.float64,
)

C_VALUES = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


def load_features(path):
    data = torch.load(path, map_location="cpu", weights_only=False)

    text = data["text"].numpy().astype(np.float64)
    photograph = data["photograph"].numpy().astype(np.float64)
    radiograph = data["radiograph"].numpy().astype(np.float64)
    labels = data["labels"].numpy().astype(np.int64)

    x = np.concatenate(
        [text, photograph, radiograph],
        axis=1,
    )

    return x, labels


def evaluate(y_true, probabilities, thresholds=None):
    if thresholds is None:
        thresholds = np.full(len(LABELS), 0.5)

    predictions = probabilities >= thresholds[None, :]

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

    auroc = roc_auc_score(
        y_true,
        probabilities,
        average="macro",
    )

    per_label = [
        f1_score(
            y_true[:, i],
            predictions[:, i],
            zero_division=0,
        )
        for i in range(len(LABELS))
    ]

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "accuracy": accuracy,
        "auroc": auroc,
        "per_label": per_label,
    }


def optimize_thresholds(y_true, probabilities):
    thresholds = np.arange(0.05, 0.951, 0.05)

    best_thresholds = []
    best_scores = []

    for i in range(len(LABELS)):
        best_t = 0.5
        best_f1 = -1.0

        for t in thresholds:
            pred = probabilities[:, i] >= t

            score = f1_score(
                y_true[:, i],
                pred,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_t = float(t)

        best_thresholds.append(best_t)
        best_scores.append(best_f1)

    return np.array(best_thresholds), best_scores


def train_models(x_train, y_train, c_value):
    models = []

    for i, label in enumerate(LABELS):
        model = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=3000,
            random_state=42,
        )

        sample_weight = np.where(
            y_train[:, i] == 1,
            POS_WEIGHTS[i],
            1.0,
        )

        model.fit(
            x_train,
            y_train[:, i],
            sample_weight=sample_weight,
        )

        models.append(model)

    return models


def predict(models, x):
    probabilities = np.column_stack(
        [
            model.predict_proba(x)[:, 1]
            for model in models
        ]
    )

    return probabilities


def main():
    print("=" * 80)
    print("CLASS-WISE LOGISTIC REGRESSION C-SWEEP")
    print("=" * 80)

    x_train, y_train = load_features(TRAIN_PATH)
    x_val, y_val = load_features(VAL_PATH)

    print(f"\nTrain X: {x_train.shape}, y: {y_train.shape}")
    print(f"Val   X: {x_val.shape}, y: {y_val.shape}")

    results = []

    for c_value in C_VALUES:
        print("\n" + "-" * 80)
        print(f"C = {c_value}")
        print("-" * 80)

        models = train_models(
            x_train,
            y_train,
            c_value,
        )

        probabilities = predict(
            models,
            x_val,
        )

        default_metrics = evaluate(
            y_val,
            probabilities,
        )

        thresholds, _ = optimize_thresholds(
            y_val,
            probabilities,
        )

        optimized_metrics = evaluate(
            y_val,
            probabilities,
            thresholds,
        )

        results.append(
            {
                "C": c_value,
                "default": default_metrics,
                "optimized": optimized_metrics,
                "thresholds": thresholds,
            }
        )

        print(
            f"@0.5      Macro F1: {default_metrics['macro_f1']:.4f} | "
            f"Micro F1: {default_metrics['micro_f1']:.4f} | "
            f"AUROC: {default_metrics['auroc']:.4f}"
        )

        print(
            f"Optimized  Macro F1: {optimized_metrics['macro_f1']:.4f} | "
            f"Micro F1: {optimized_metrics['micro_f1']:.4f} | "
            f"AUROC: {optimized_metrics['auroc']:.4f}"
        )

        print(
            "Thresholds:",
            np.round(thresholds, 2).tolist(),
        )

        print(
            "Per-label F1 @0.5:",
            " | ".join(
                f"{LABELS[i]}={default_metrics['per_label'][i]:.4f}"
                for i in range(len(LABELS))
            ),
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        "\n"
        f"{'C':>8} | "
        f"{'Default F1':>12} | "
        f"{'Optimized F1':>14} | "
        f"{'Micro F1':>10} | "
        f"{'AUROC':>8}"
    )
    print("-" * 70)

    for result in results:
        d = result["default"]
        o = result["optimized"]

        print(
            f"{result['C']:>8g} | "
            f"{d['macro_f1']:>12.4f} | "
            f"{o['macro_f1']:>14.4f} | "
            f"{o['micro_f1']:>10.4f} | "
            f"{o['auroc']:>8.4f}"
        )

    best_default = max(
        results,
        key=lambda r: r["default"]["macro_f1"],
    )

    best_optimized = max(
        results,
        key=lambda r: r["optimized"]["macro_f1"],
    )

    print("\nBest default-threshold model:")
    print(
        f"C = {best_default['C']} | "
        f"Macro F1 = {best_default['default']['macro_f1']:.4f}"
    )

    print("\nBest validation-threshold model:")
    print(
        f"C = {best_optimized['C']} | "
        f"Macro F1 = {best_optimized['optimized']['macro_f1']:.4f}"
    )
    print(
        "Thresholds:",
        np.round(best_optimized["thresholds"], 2).tolist(),
    )

    print("\nReference:")
    print("Frozen linear probe: 0.8271")
    print("V2 end-to-end full multimodal: 0.8137")

    print("\nNo test data was loaded.")
    print("Thresholds are validation-only.")


if __name__ == "__main__":
    main()