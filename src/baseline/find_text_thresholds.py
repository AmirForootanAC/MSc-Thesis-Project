"""
Find optimal thresholds for text-only baseline.
"""

import json
import numpy as np
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

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

    model.to(device)

    model.eval()

    return model



def collect_predictions(
    model,
    loader,
    device,
):

    logits_all = []
    labels_all = []


    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].to(device)


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


    logits = torch.cat(logits_all)

    labels = torch.cat(labels_all)


    probs = torch.sigmoid(
        logits
    )


    return (
        probs.numpy(),
        labels.numpy()
    )



def find_best_threshold(
    probabilities,
    labels,
):

    thresholds = {}
    scores = {}


    for i in range(labels.shape[1]):

        best_t = 0.5
        best_f1 = 0


        for t in np.arange(
            0.05,
            0.95,
            0.05,
        ):

            pred = (
                probabilities[:,i] >= t
            )


            f1 = f1_score(
                labels[:,i],
                pred,
                zero_division=0,
            )


            if f1 > best_f1:

                best_f1 = f1
                best_t = t


        thresholds[
            config.LABEL_NAMES[i]
        ] = float(best_t)


        scores[
            config.LABEL_NAMES[i]
        ] = float(best_f1)


    return thresholds, scores



def main():

    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        device
    )


    dataset = COdeTextDataset(
        csv_path=config.DATASET_PATH,
        split=config.VALID_SPLIT,
    )


    loader = DataLoader(
        dataset,
        batch_size=config.TEXT_BATCH_SIZE,
        shuffle=False,
        collate_fn=text_collate,
    )


    model = build_model(device)


    probs, labels = collect_predictions(
        model,
        loader,
        device,
    )


    thresholds, scores = find_best_threshold(
        probs,
        labels,
    )


    out = (
        config.RESULT_ROOT
        /
        config.TEXT_EXPERIMENT_NAME
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        out / "thresholds.json",
        "w",
    ) as f:

        json.dump(
            {
                "thresholds": thresholds,
                "label_f1": scores,
            },
            f,
            indent=4,
        )


    print("Saved thresholds")

    for k,v in thresholds.items():

        print(
            k,
            v,
            scores[k]
        )


if __name__=="__main__":
    main()