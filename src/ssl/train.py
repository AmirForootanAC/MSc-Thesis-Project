"""
Dynamic multimodal SSL pretraining.

Objective:
- Image <-> Text
- Image <-> Radiograph
- Radiograph <-> Text

Missing modalities are handled dynamically using masks.
"""

import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler


from src.ssl.dataset import COdeSSLDataset
from src.ssl.collate import ssl_collate
from src.ssl.model import MultimodalSSLModel
from src.ssl.tokenizer import ClinicalTokenizer


from src.baseline.image_loader import COdeImageLoader
from src.baseline.transforms import get_image_transform



# =========================
# Configuration
# =========================

SEED = 42

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4


CSV = (
    "results/"
    "labeled_patient_level_dataset/"
    "labeled_dataset.csv"
)


OUT = Path(
    "results/"
    "ssl_pretraining/"
    "multimodal_dynamic"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)



# =========================
# Reproducibility
# =========================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)



# =========================
# Image loading
# =========================

def load_modality_batch(
    batch_files,
    modality,
    loader,
    transform,
):

    outputs = []
    mask = []


    for files in batch_files:

        imgs = []


        for f in files:

            try:

                img = loader.load(
                    f,
                    modality=modality
                )


                img = transform(
                    img
                )


                imgs.append(
                    img
                )


            except Exception:

                continue



        if imgs:

            outputs.append(
                torch.stack(imgs).mean(0)
            )

            mask.append(True)



        else:

            outputs.append(
                torch.zeros(
                    3,
                    224,
                    224
                )
            )

            mask.append(False)



    return (
        torch.stack(outputs).to(DEVICE),
        torch.tensor(
            mask,
            device=DEVICE
        )
    )



# =========================
# Main training
# =========================

def main():

    seed_everything(SEED)


    print(
        f"Device: {DEVICE}"
    )


    dataset = COdeSSLDataset(
        CSV,
        split="train"
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=ssl_collate,
        num_workers=2,
        pin_memory=True,
    )


    print(
        f"SSL visits: {len(dataset)}"
    )


    model = MultimodalSSLModel().to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )


    tokenizer = ClinicalTokenizer()


    transform = get_image_transform()


    image_loader = COdeImageLoader(
        "data/raw/COde-Dataset/Images"
    )


    scaler = GradScaler(
        "cuda"
    )



    history = {

        "loss": [],

        "image_text_loss": [],

        "image_radiograph_loss": [],

        "radiograph_text_loss": [],

        "valid_batches": [],

        "skipped_batches": []

    }



    best_loss = float(
        "inf"
    )

    best_epoch = -1



    for epoch in range(EPOCHS):


        model.train()


        total_loss = 0.0

        valid_batches = 0

        skipped_batches = 0


        pair_loss_sum = {

            "image_text": 0.0,

            "image_radiograph": 0.0,

            "radiograph_text": 0.0,

        }



        bar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )



        for batch in bar:


            optimizer.zero_grad()



            images, image_mask = load_modality_batch(
                batch["images"],
                "photograph",
                image_loader,
                transform
            )


            radiographs, radiograph_mask = load_modality_batch(
                batch["radiographs"],
                "radiograph",
                image_loader,
                transform
            )



            tokens = tokenizer(
                batch["text"]
            )


            input_ids = tokens["input_ids"].to(
                DEVICE
            )

            attention = tokens["attention_mask"].to(
                DEVICE
            )



            with autocast(
                "cuda",
                enabled=(DEVICE == "cuda")
            ):


                image_z = model.project_image(
                    images
                )


                radiograph_z = model.project_radiograph(
                    radiographs
                )


                text_z = model.project_text(
                    input_ids,
                    attention
                )



                loss, pairs = model(

                    image_embeddings=image_z,

                    radiograph_embeddings=radiograph_z,

                    text_embeddings=text_z,


                    pair_mask={

                        "image_text":

                        image_mask
                        &
                        batch["has_text"].to(DEVICE),


                        "image_radiograph":

                        image_mask
                        &
                        radiograph_mask,


                        "radiograph_text":

                        radiograph_mask
                        &
                        batch["has_text"].to(DEVICE),

                    }
                )



            # No valid pair in this batch

            if len(pairs) == 0:

                skipped_batches += 1

                continue



            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()



            total_loss += loss.item()

            valid_batches += 1



            for name, value in pairs.items():

                if name in pair_loss_sum:

                    pair_loss_sum[name] += value.item()



            bar.set_postfix(

                loss=f"{loss.item():.4f}",

                pairs=",".join(
                    pairs.keys()
                )

            )



        # =========================
        # Epoch statistics
        # =========================


        avg_loss = (
            total_loss / valid_batches
            if valid_batches > 0
            else 0
        )


        history["loss"].append(
            avg_loss
        )


        history["valid_batches"].append(
            valid_batches
        )


        history["skipped_batches"].append(
            skipped_batches
        )


        for name in pair_loss_sum:

            history[f"{name}_loss"].append(

                pair_loss_sum[name] / valid_batches
                if valid_batches > 0
                else 0

            )



        print(
            f"\nEpoch {epoch+1}: {avg_loss:.4f}"
        )


        print(
            "Pairs:",
            {
                k:
                history[f"{k}_loss"][-1]

                for k in pair_loss_sum
            }
        )



        # Save best

        if avg_loss < best_loss:

            best_loss = avg_loss

            best_epoch = epoch + 1


            torch.save(

                model.state_dict(),

                OUT /
                "best_ssl_model.pt"

            )



        # Save last checkpoint

        torch.save(

            model.state_dict(),

            OUT /
            "last_ssl_model.pt"

        )



    # =========================
    # Save history
    # =========================


    history["best_epoch"] = best_epoch

    history["best_loss"] = best_loss



    with open(
        OUT /
        "history.json",
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )



    config = {

        "timestamp":
        datetime.now().isoformat(),

        "dataset":
        "COde",

        "split":
        "train",

        "batch_size":
        BATCH_SIZE,

        "epochs":
        EPOCHS,

        "learning_rate":
        LR,

        "seed":
        SEED,

        "objective":
        "Dynamic multimodal contrastive learning",

        "pairs":

        [
            "image_text",
            "image_radiograph",
            "radiograph_text"
        ],

        "device":
        DEVICE

    }



    with open(
        OUT /
        "config.json",
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )



    print(
        "\nSSL pretraining finished."
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best loss: {best_loss:.4f}"
    )



if __name__ == "__main__":

    main()