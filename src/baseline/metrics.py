import numpy as np
import torch

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

    Inputs:
        logits:
            torch.Tensor [N, C]

        labels:
            torch.Tensor [N, C]

    Returns:
        dict
    """


    # -----------------------------
    # Convert tensors to numpy
    # -----------------------------

    if torch.is_tensor(logits):

        logits = logits.detach().cpu().numpy()


    if torch.is_tensor(labels):

        labels = labels.detach().cpu().numpy()



    # -----------------------------
    # Probabilities
    # -----------------------------

    probabilities = 1 / (
        1 + np.exp(-logits)
    )



    # -----------------------------
    # Threshold handling
    # -----------------------------

    if np.isscalar(threshold):

        threshold_array = threshold

    else:

        threshold_array = np.asarray(
            threshold
        )



    predictions = (
        probabilities >= threshold_array
    ).astype(int)



    # -----------------------------
    # Metrics
    # -----------------------------

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )


    micro_f1 = f1_score(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )


    accuracy = accuracy_score(
        labels,
        predictions,
    )


    try:

        auroc = roc_auc_score(
            labels,
            probabilities,
            average="macro",
        )

    except ValueError:

        auroc = 0.0



    return {

        "macro_f1": float(macro_f1),

        "micro_f1": float(micro_f1),

        "auroc": float(auroc),

        "accuracy": float(accuracy),

    }