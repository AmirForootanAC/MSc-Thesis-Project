"""
Evaluation metrics for COde baseline classification.
"""

import numpy as np

from scipy.special import expit

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
)


def compute_metrics(
    logits,
    labels,
    threshold=0.5,
):
    """
    Compute multi-label classification metrics.

    Supports:
    - single global threshold
    - label-wise thresholds
    """

    probabilities = expit(
        logits
    )


    threshold = np.asarray(
        threshold
    )


    predictions = (
        probabilities >= threshold
    ).astype(int)


    metrics = {}


    metrics["macro_f1"] = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )


    metrics["micro_f1"] = f1_score(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )


    metrics["accuracy"] = accuracy_score(
        labels,
        predictions,
    )


    try:

        metrics["auroc"] = roc_auc_score(
            labels,
            probabilities,
            average="macro",
        )

    except ValueError:

        metrics["auroc"] = 0.0


    return metrics