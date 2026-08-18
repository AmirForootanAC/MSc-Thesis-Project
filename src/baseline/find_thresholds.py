"""
Find optimal classification thresholds
for each label using validation split.
"""

import json

import numpy as np
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import f1_score

from src.baseline import config

from src.baseline.transforms import get_image_transform

from src.baseline.dataset import (
    COdeBaselineDataset,
)

from src.baseline.collate import (
    baseline_collate,
)

from src.baseline.encoder import (
    ResNet50Encoder,
)

from src.baseline.aggregator import (
    MeanImageAggregator,
)

from src.baseline.model import (
    ImageOnlyBaseline,
)

from src.baseline.utils import (
    move_image_batch_to_device,
)



def build_model(device):

    encoder = ResNet50Encoder(
        pretrained=config.PRETRAINED,
        freeze=config.FREEZE_ENCODER,
    )


    aggregator = MeanImageAggregator(
        encoder
    )


    model = ImageOnlyBaseline(
        aggregator,
        num_labels=config.NUM_LABELS,
    )


    model.load_state_dict(
        torch.load(
            config.RESULT_ROOT
            /
            config.EXPERIMENT_NAME
            /
            "best_model.pt",
            map_location=device,
        )
    )


    model = model.to(
        device
    )


    model.eval()


    return model



def collect_predictions(
    model,
    loader,
    device,
):

    all_logits = []
    all_labels = []


    with torch.no_grad():

        for batch in loader:

            images = move_image_batch_to_device(
                batch["images"],
                device,
            )


            labels = batch["labels"].to(
                device
            )


            logits = model(
                images
            )


            all_logits.append(
                logits.cpu()
            )


            all_labels.append(
                labels.cpu()
            )


    logits = torch.cat(
        all_logits
    )


    labels = torch.cat(
        all_labels
    )


    probabilities = torch.sigmoid(
        logits
    )


    return (
        probabilities.numpy(),
        labels.numpy(),
    )



def find_best_threshold(
    probabilities,
    labels,
):

    thresholds = {}

    scores = {}


    for label_idx in range(
        labels.shape[1]
    ):

        best_threshold = 0.5
        best_f1 = 0.0


        for threshold in np.arange(
            0.05,
            0.95,
            0.05,
        ):

            predictions = (
                probabilities[:, label_idx]
                >= threshold
            )


            score = f1_score(
                labels[:, label_idx],
                predictions,
                zero_division=0,
            )


            if score > best_f1:

                best_f1 = score
                best_threshold = threshold


        thresholds[
            COdeBaselineDataset.LABEL_COLUMNS[label_idx]
        ] = float(
            best_threshold
        )


        scores[
            COdeBaselineDataset.LABEL_COLUMNS[label_idx]
        ] = float(
            best_f1
        )


    return thresholds, scores



def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"Device: {device}"
    )


    dataset = COdeBaselineDataset(
        csv_path=config.DATASET_PATH,
        split=config.VALID_SPLIT,
        image_root=config.IMAGE_ROOT,
        transform=get_image_transform(),
    )


    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=baseline_collate,
    )


    model = build_model(
        device
    )


    probabilities, labels = collect_predictions(
        model,
        loader,
        device,
    )


    thresholds, scores = find_best_threshold(
        probabilities,
        labels,
    )


    output_dir = (
        config.RESULT_ROOT
        /
        config.EXPERIMENT_NAME
    )


    with open(
        output_dir
        /
        "thresholds.json",
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


    print(
        "Best thresholds:"
    )


    for label, threshold in thresholds.items():

        print(
            label,
            ":",
            threshold,
            "F1:",
            scores[label],
        )


    print(
        "Saved:",
        output_dir
        /
        "thresholds.json",
    )



if __name__ == "__main__":

    main()