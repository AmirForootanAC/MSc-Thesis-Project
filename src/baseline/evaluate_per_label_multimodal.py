import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.baseline import config

from src.baseline.multimodal_dataset import COdeMultimodalDataset
from src.baseline.multimodal_collate import multimodal_collate
from src.baseline.multimodal_model import FullMultimodalBaseline



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
    ).numpy()


    return (
        probabilities,
        labels.numpy(),
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
        test_loader,
        device,
    )



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



    threshold_values = np.array(
        [
            thresholds[label]["threshold"]
            for label in config.LABEL_NAMES
        ]
    )



    predictions = (
        probabilities
        >= threshold_values
    ).astype(int)



    results = {}



    for idx, label_name in enumerate(
        config.LABEL_NAMES
    ):

        y_true = labels[:, idx]

        y_pred = predictions[:, idx]

        y_prob = probabilities[:, idx]


        results[label_name] = {

            "support": int(
                y_true.sum()
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


            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),


            "auroc": float(
                roc_auc_score(
                    y_true,
                    y_prob,
                )
            ),
        }



        print(
            "\n",
            label_name
        )


        for key, value in results[label_name].items():

            print(
                f"{key}: {value:.4f}"
                if isinstance(value, float)
                else f"{key}: {value}"
            )



    output_path = (
        config.RESULT_ROOT
        /
        "full_multimodal_6label"
        /
        "per_label_metrics_thresholded.json"
    )



    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )


    print(
        "\nSaved:",
        output_path,
    )



if __name__ == "__main__":

    main()