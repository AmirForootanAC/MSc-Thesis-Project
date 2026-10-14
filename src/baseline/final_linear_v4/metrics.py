"""Common metrics and validation-only threshold optimization."""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sigmoid(x):
    x = np.clip(np.asarray(x), -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def macro_auroc(labels, probs):
    try:
        if any(
            len(np.unique(labels[:, j])) < 2
            for j in range(labels.shape[1])
        ):
            return 0.0

        return float(
            roc_auc_score(
                labels,
                probs,
                average="macro",
            )
        )
    except ValueError:
        return 0.0


def per_label_metrics(
    logits,
    labels,
    threshold=0.5,
    label_names=None,
):
    probs = sigmoid(logits)

    thresholds = np.asarray(threshold)

    if thresholds.ndim == 0:
        thresholds = np.full(
            labels.shape[1],
            float(thresholds),
        )

    if len(thresholds) != labels.shape[1]:
        raise ValueError(
            "Number of thresholds must match number of labels."
        )

    if label_names is None:
        label_names = [
            f"label_{i}"
            for i in range(labels.shape[1])
        ]

    if len(label_names) != labels.shape[1]:
        raise ValueError(
            "Number of label names must match number of labels."
        )

    result = {}

    for j, name in enumerate(label_names):
        y_true = labels[:, j]
        y_prob = probs[:, j]
        y_pred = (
            y_prob >= thresholds[j]
        ).astype(int)

        if len(np.unique(y_true)) < 2:
            auroc = 0.0
        else:
            try:
                auroc = float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                    )
                )
            except ValueError:
                auroc = 0.0

        result[name] = {
            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "auroc": auroc,
        }

    return result


def compute_metrics(
    logits,
    labels,
    threshold=0.5,
    label_names=None,
):
    probs = sigmoid(logits)

    pred = (
        probs >= np.asarray(threshold)
    ).astype(int)

    result = {
        "macro_f1": float(
            f1_score(
                labels,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                labels,
                pred,
                average="micro",
                zero_division=0,
            )
        ),
        "auroc": macro_auroc(
            labels,
            probs,
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                pred,
            )
        ),
    }

    if label_names is not None:
        result["per_label"] = per_label_metrics(
            logits=logits,
            labels=labels,
            threshold=threshold,
            label_names=label_names,
        )

    return result


def optimize_thresholds(
    logits,
    labels,
    minimum=0.30,
    maximum=0.70,
    step=0.05,
):
    """
    Optimize one threshold per label using validation data only.
    Requires a meaningful improvement (>0.01) over default 0.5 to prevent
    threshold overfitting on rare labels.
    """
    probs = sigmoid(logits)

    grid = np.arange(
        minimum,
        maximum + step / 2,
        step,
    )

    result = []

    for j in range(labels.shape[1]):
        default_f1 = f1_score(
            labels[:, j],
            (probs[:, j] >= 0.5).astype(int),
            zero_division=0,
        )
        best_t = 0.5
        best_f1 = default_f1 + 0.01

        for t in grid:
            score = f1_score(
                labels[:, j],
                (probs[:, j] >= t).astype(int),
                zero_division=0,
            )

            if (
                score > best_f1 + 1e-4
                or (
                    abs(score - best_f1) <= 1e-4
                    and abs(t - 0.5) < abs(best_t - 0.5)
                )
            ):
                best_f1 = score
                best_t = float(t)

        result.append(best_t)

    return np.asarray(result)