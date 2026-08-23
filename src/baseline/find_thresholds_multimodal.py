import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.baseline import config

from src.baseline.multimodal_dataset import COdeMultimodalDataset
from src.baseline.multimodal_collate import multimodal_collate
from src.baseline.multimodal_model import FullMultimodalBaseline


def move_images_to_device(batch_images, device):

    return [
        [
            img.to(device)
            for img in sample_images
        ]
        for sample_images in batch_images
    ]



@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device,
):

    model.eval()

    all_logits = []
    all_labels = []


    for batch in tqdm(loader):

        images = move_images_to_device(
            batch["images"],
            device,
        )


        radiographs = move_images_to_device(
            batch["radiographs"],
            device,
        )


        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)


        logits = model(
            images,
            radiographs,
            input_ids,
            attention_mask,
        )


        all_logits.append(
            logits.cpu()
        )

        all_labels.append(
            labels.cpu()
        )


    logits = torch.cat(
        all_logits,
        dim=0,
    )


    labels = torch.cat(
        all_labels,
        dim=0,
    )


    probabilities = torch.sigmoid(
        logits
    )


    return (
        probabilities.numpy(),
        labels.numpy(),
    )



def f1_score_binary(
    y_true,
    y_pred,
):

    tp = np.sum(
        (y_true == 1)
        &
        (y_pred == 1)
    )

    fp = np.sum(
        (y_true == 0)
        &
        (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1)
        &
        (y_pred == 0)
    )


    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )


    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )


    if precision + recall == 0:

        return 0


    return (
        2
        *
        precision
        *
        recall
        /
        (precision + recall)
    )



def find_best_thresholds(
    probabilities,
    labels,
):

    thresholds = {}


    for idx, label_name in enumerate(
        config.LABEL_NAMES
    ):

        best_threshold = 0.5

        best_f1 = 0


        y_true = labels[:, idx]

        y_prob = probabilities[:, idx]


        for threshold in np.arange(
            0.05,
            0.96,
            0.05,
        ):

            y_pred = (
                y_prob >= threshold
            ).astype(int)


            score = f1_score_binary(
                y_true,
                y_pred,
            )


            if score > best_f1:

                best_f1 = score

                best_threshold = float(
                    threshold
                )


        thresholds[label_name] = {
            "threshold": best_threshold,
            "f1": best_f1,
        }


        print(
            label_name,
            "threshold:",
            best_threshold,
            "F1:",
            round(best_f1,4),
        )


    return thresholds



def main():

    device = torch.device(
        config.DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        device
    )


    val_dataset = COdeMultimodalDataset(
        csv_path=config.DATASET_PATH,
        split=config.VALID_SPLIT,
        image_root=config.IMAGE_ROOT,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=multimodal_collate,
    )


    model = FullMultimodalBaseline(
        text_model_name=config.TEXT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
        pretrained=True,
        freeze_image_encoder=True,
    )


    checkpoint = (
        config.RESULT_ROOT
        /
        "full_multimodal_6label"
        /
        "best_model.pt"
    )


    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
        )
    )


    model = model.to(device)


    probabilities, labels = collect_predictions(
        model,
        val_loader,
        device,
    )


    thresholds = find_best_thresholds(
        probabilities,
        labels,
    )


    output_path = (
        config.RESULT_ROOT
        /
        "full_multimodal_6label"
        /
        "thresholds.json"
    )


    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            thresholds,
            f,
            indent=2,
        )


    print(
        "\nSaved:",
        output_path,
    )



if __name__ == "__main__":

    main()