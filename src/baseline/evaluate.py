"""
Evaluation pipeline for COde baseline experiments.
"""

import json

import torch

from torch.utils.data import DataLoader

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

from src.baseline.metrics import (
    compute_metrics,
)

from src.baseline.utils import (
    move_image_batch_to_device,
)



def collect_predictions(
    model,
    loader,
    device,
):

    model.eval()

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



def load_thresholds(
    path,
):

    with open(
        path,
        "r",
    ) as f:

        data = json.load(f)


    thresholds_dict = data[
        "thresholds"
    ]


    thresholds = [
        thresholds_dict[name]
        for name in config.LABEL_NAMES
    ]


    return thresholds



def apply_thresholds(
    probabilities,
    thresholds,
):

    threshold_tensor = torch.tensor(
        thresholds
    ).numpy()


    predictions = (
        probabilities
        >= threshold_tensor
    )


    return predictions.astype(int)



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


    encoder = ResNet50Encoder(
        pretrained=False,
        freeze=config.FREEZE_ENCODER,
    )


    aggregator = MeanImageAggregator(
        encoder
    )


    model = ImageOnlyBaseline(
        aggregator,
        num_labels=config.NUM_LABELS,
    )


    checkpoint = (
        config.RESULT_ROOT
        /
        config.EXPERIMENT_NAME
        /
        "best_model.pt"
    )


    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
        )
    )


    model = model.to(
        device
    )


    probabilities, labels = collect_predictions(
        model,
        loader,
        device,
    )


    print(
        "\nDefault threshold = 0.5"
    )


    default_predictions = (
        probabilities >= 0.5
    ).astype(int)


    default_metrics = compute_metrics(
        probabilities,
        labels,
        threshold=0.5,
    )


    print(
        default_metrics
    )


    threshold_path = (
        config.RESULT_ROOT
        /
        config.EXPERIMENT_NAME
        /
        "thresholds.json"
    )


    thresholds = load_thresholds(
        threshold_path
    )


    print(
        "\nOptimized thresholds"
    )


    optimized_predictions = apply_thresholds(
        probabilities,
        thresholds,
    )


    optimized_metrics = compute_metrics(
        probabilities,
        labels,
        threshold=thresholds,
    )

    print(
        optimized_metrics
    )


    output = {

        "default_threshold_0.5": default_metrics,

        "optimized_thresholds": optimized_metrics,

        "thresholds": {
            name: threshold
            for name, threshold in zip(
                config.LABEL_NAMES,
                thresholds,
            )
        },

    }


    output_path = (
        config.RESULT_ROOT
        /
        config.EXPERIMENT_NAME
        /
        "evaluation.json"
    )


    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )


    print(
        f"\nSaved: {output_path}"
    )



if __name__ == "__main__":

    main()