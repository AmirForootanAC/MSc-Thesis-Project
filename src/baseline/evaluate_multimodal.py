import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.baseline import config

from src.baseline.multimodal_dataset import COdeMultimodalDataset
from src.baseline.multimodal_collate import multimodal_collate
from src.baseline.multimodal_model import FullMultimodalBaseline
from src.baseline.metrics import compute_metrics



def move_images_to_device(
    batch_images,
    device,
):

    return [
        [
            img.to(device)
            for img in sample_images
        ]
        for sample_images in batch_images
    ]



@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    thresholds=None,
):

    model.eval()

    total_loss = 0

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


        loss = criterion(
            logits,
            labels,
        )


        total_loss += loss.item()


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


    # -----------------------------
    # Default threshold evaluation
    # -----------------------------

    default_metrics = compute_metrics(
        logits,
        labels,
        threshold=0.5,
    )


    # -----------------------------
    # Optimized threshold evaluation
    # -----------------------------

    optimized_metrics = None


    if thresholds is not None:

        threshold_values = np.array(
            [
                thresholds[label]["threshold"]
                for label in config.LABEL_NAMES
            ]
        )


        optimized_metrics = compute_metrics(
            logits,
            labels,
            threshold=threshold_values,
        )



    return (
        total_loss / len(loader),
        default_metrics,
        optimized_metrics,
    )



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



    test_dataset = COdeMultimodalDataset(
        csv_path=config.DATASET_PATH,
        split=config.TEST_SPLIT,
        image_root=config.IMAGE_ROOT,
    )



    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=multimodal_collate,
    )


    print(
        "Test samples:",
        len(test_dataset)
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



    criterion = nn.BCEWithLogitsLoss()



    threshold_path = (
        config.RESULT_ROOT
        /
        "full_multimodal_6label"
        /
        "thresholds.json"
    )


    with open(
        threshold_path,
        "r",
    ) as f:

        thresholds = json.load(f)



    loss, default_metrics, optimized_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        thresholds,
    )



    print("\nTest Results")
    print("----------------")


    print(
        "Loss:",
        loss,
    )


    print(
        "\nDefault threshold (0.5)"
    )

    for key, value in default_metrics.items():

        print(
            f"{key}: {value:.4f}"
        )



    print(
        "\nOptimized thresholds"
    )

    for key, value in optimized_metrics.items():

        print(
            f"{key}: {value:.4f}"
        )



    output = {

        "loss": loss,

        "default_threshold": default_metrics,

        "optimized_thresholds": optimized_metrics,

        "threshold_values": thresholds,
    }



    output_path = (
        config.RESULT_ROOT
        /
        "full_multimodal_6label"
        /
        "test_metrics_thresholded.json"
    )



    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )


    print(
        "\nSaved:",
        output_path,
    )



if __name__ == "__main__":

    main()