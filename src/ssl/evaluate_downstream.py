"""
Evaluation of SSL downstream classifiers.

Evaluate:
    image
    radiograph
    text
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.ssl.model import MultimodalSSLModel
from src.ssl.downstream_dataset import SSLDownstreamDataset
from src.ssl.downstream import SSLDownstreamClassifier

from src.baseline.collate import baseline_collate
from src.ssl.tokenizer import ClinicalTokenizer

from src.baseline.image_loader import COdeImageLoader
from src.baseline.transforms import get_image_transform

from src.baseline.metrics import compute_metrics


# =========================
# CONFIG
# =========================

MODALITY = "text"
# image
# radiograph
# text


SPLIT = "test"
# validation
# test


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


BATCH_SIZE = 16


CSV = (
    "results/"
    "six_label_patient_level_dataset/"
    "labeled_dataset.csv"
)


MODEL_PATH = Path(
    f"results/ssl_pretraining/downstream/{MODALITY}/model.pt"
)


OUT = Path(
    "results"
) / "ssl_pretraining" / "evaluation" / MODALITY

OUT.mkdir(
    parents=True,
    exist_ok=True
)


NUM_LABELS = 6



# =========================
# Image loader
# =========================


def load_images(
    files,
    modality,
    loader,
    transform,
):

    outputs=[]


    for sample in files:

        imgs=[]

        for f in sample:

            try:

                img = loader.load(
                    f,
                    modality=modality
                )

                imgs.append(
                    transform(img)
                )

            except Exception:

                continue


        if imgs:

            outputs.append(
                torch.stack(imgs).mean(0)
            )

        else:

            outputs.append(
                torch.zeros(
                    3,
                    224,
                    224
                )
            )


    return torch.stack(outputs).to(
        DEVICE
    )



# =========================
# Main
# =========================


def main():


    dataset = SSLDownstreamDataset(
        CSV,
        split=SPLIT
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=baseline_collate
    )



    ssl_model = MultimodalSSLModel()

    model = SSLDownstreamClassifier(
        ssl_model,
        MODALITY
    )



    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )


    model.load_state_dict(
        checkpoint
    )


    model.to(
        DEVICE
    )

    model.eval()



    image_loader = COdeImageLoader(
        "data/raw/COde-Dataset/Images"
    )


    transform = get_image_transform()

    tokenizer = ClinicalTokenizer()



    logits_all=[]

    labels_all=[]



    with torch.no_grad():


        for batch in tqdm(
            loader,
            desc="Evaluation"
        ):


            if MODALITY=="image":

                x = load_images(
                    batch["images"],
                    "photograph",
                    image_loader,
                    transform
                )


            elif MODALITY=="radiograph":

                x = load_images(
                    batch["radiographs"],
                    "radiograph",
                    image_loader,
                    transform
                )


            else:

                tokens = tokenizer(
                    batch["text"]
                )


                x = (
                    tokens["input_ids"].to(DEVICE),
                    tokens["attention_mask"].to(DEVICE)
                )



            labels = batch["labels"].to(
                DEVICE
            )


            logits = model(
                x
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



    metrics = compute_metrics(
        logits,
        labels
    )



    print(metrics)



    with open(
        OUT /
        f"{SPLIT}_metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )



if __name__=="__main__":
    main()