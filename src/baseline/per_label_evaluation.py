"""
Per-label evaluation for COde image-only baseline.
"""

import json

import numpy as np
import torch

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader

from src.baseline import config

from src.baseline.transforms import (
    get_image_transform,
)

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

from src.baseline.utils import (
    move_image_batch_to_device,
    get_modality_batch,
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
                get_modality_batch(
                    batch,
                    config.MODALITY,
                ),
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



def compute_per_label_metrics(
    probabilities,
    labels,
):

    results = {}


    predictions = (
        probabilities >= 0.5
    ).astype(int)


    for idx, name in enumerate(
        config.LABEL_NAMES
    ):

        label_true = labels[:, idx]

        label_pred = predictions[:, idx]

        label_prob = probabilities[:, idx]


        f1 = f1_score(
            label_true,
            label_pred,
            zero_division=0,
        )


        try:

            auroc = roc_auc_score(
                label_true,
                label_prob,
            )

        except ValueError:

            auroc = 0.0


        support = int(
            label_true.sum()
        )


        results[name] = {

            "f1": float(f1),

            "auroc": float(auroc),

            "support": support,

        }


    return results



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


    results = compute_per_label_metrics(
        probabilities,
        labels,
    )


    output_path = (
        config.RESULT_ROOT
        /
        config.EXPERIMENT_NAME
        /
        "per_label_metrics.json"
    )


    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )


    print(
        "\nPer-label metrics:"
    )


    for name, value in results.items():

        print(
            f"{name}: "
            f"F1={value['f1']:.4f}, "
            f"AUROC={value['auroc']:.4f}, "
            f"support={value['support']}"
        )


    print(
        f"\nSaved: {output_path}"
    )



if __name__ == "__main__":

    main()