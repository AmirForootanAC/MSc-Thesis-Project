"""
Evaluate Text-only baseline using validation-optimized thresholds.
"""

import json

import torch
from torch.utils.data import DataLoader

from src.baseline import config

from src.baseline.text_dataset import (
    COdeTextDataset,
)

from src.baseline.text_collate import (
    text_collate,
)

from src.baseline.text_model import (
    TextOnlyBaseline,
)

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
)


def build_model(device):

    model = TextOnlyBaseline(
        model_name=config.TEXT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
    )

    checkpoint = torch.load(
        config.RESULT_ROOT
        /
        config.TEXT_EXPERIMENT_NAME
        /
        "best_model.pt",
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model = model.to(device)

    model.eval()

    return model



def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )


    dataset = COdeTextDataset(
        csv_path=config.DATASET_PATH,
        split=config.TEST_SPLIT,
    )


    loader = DataLoader(
        dataset,
        batch_size=config.TEXT_BATCH_SIZE,
        shuffle=False,
        collate_fn=text_collate,
    )


    model = build_model(
        device
    )


    logits_all = []
    labels_all = []


    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(
                device
            )

            attention_mask = batch["attention_mask"].to(
                device
            )

            labels = batch["labels"].to(
                device
            )


            logits = model(
                input_ids,
                attention_mask,
            )


            logits_all.append(
                logits.cpu()
            )

            labels_all.append(
                labels.cpu()
            )


    logits = torch.cat(
        logits_all
    )


    labels = torch.cat(
        labels_all
    )


    probabilities = torch.sigmoid(
        logits
    )


    # -------------------------------
    # Load optimized thresholds
    # -------------------------------

    with open(
        config.RESULT_ROOT
        /
        config.TEXT_EXPERIMENT_NAME
        /
        "thresholds.json"
    ) as f:

        threshold_data = json.load(f)


    thresholds = [
        threshold_data["thresholds"][label]
        for label in config.LABEL_NAMES
    ]


    thresholds = torch.tensor(
        thresholds
    )


    predictions = (
        probabilities
        >= thresholds
    )


    metrics = {

        "macro_f1":
            f1_score(
                labels.numpy(),
                predictions.numpy(),
                average="macro",
                zero_division=0,
            ),

        "micro_f1":
            f1_score(
                labels.numpy(),
                predictions.numpy(),
                average="micro",
                zero_division=0,
            ),

        "accuracy":
            accuracy_score(
                labels.numpy(),
                predictions.numpy(),
            ),

        "auroc":
            roc_auc_score(
                labels.numpy(),
                probabilities.numpy(),
                average="macro",
            ),
    }

    print(metrics)


    output = (
        config.RESULT_ROOT
        /
        config.TEXT_EXPERIMENT_NAME
        /
        "test_metrics_optimized.json"
    )


    with open(
        output,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )


    print(
        "Saved:",
        output
    )


if __name__ == "__main__":
    main()