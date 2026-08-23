import json

import torch
import torch.nn as nn

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



def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0


    for batch in tqdm(loader):

        optimizer.zero_grad()


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


        loss.backward()

        optimizer.step()


        total_loss += loss.item()



    return total_loss / len(loader)





@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
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



    total_loss /= len(loader)



    all_logits = torch.cat(
        all_logits,
        dim=0,
    )


    all_labels = torch.cat(
        all_labels,
        dim=0,
    )



    metrics = compute_metrics(
        all_logits.numpy(),
        all_labels.numpy(),
        threshold=0.5,
    )



    return (
        total_loss,
        metrics,
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



    train_dataset = COdeMultimodalDataset(
        csv_path=config.DATASET_PATH,
        split=config.TRAIN_SPLIT,
        image_root=config.IMAGE_ROOT,
    )



    val_dataset = COdeMultimodalDataset(
        csv_path=config.DATASET_PATH,
        split=config.VALID_SPLIT,
        image_root=config.IMAGE_ROOT,
    )



    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=multimodal_collate,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=multimodal_collate,
    )



    print(
        "Train samples:",
        len(train_dataset)
    )


    print(
        "Validation samples:",
        len(val_dataset)
    )



    model = FullMultimodalBaseline(
        text_model_name=config.TEXT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
        pretrained=True,
        freeze_image_encoder=True,
    )


    model = model.to(device)



    criterion = nn.BCEWithLogitsLoss()



    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=2e-5,
        weight_decay=1e-4,
    )



    result_dir = (
        config.RESULT_ROOT
        /
        "full_multimodal_6label"
    )


    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )



    history = []


    best_macro_f1 = -1



    for epoch in range(
        config.TEXT_NUM_EPOCHS
    ):


        print(
            f"\nEpoch {epoch+1}"
        )



        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )



        val_loss, metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )



        print(
            f"""
Train loss: {train_loss:.4f}

Validation:
Loss:
{val_loss:.4f}

Macro F1:
{metrics["macro_f1"]:.4f}

Micro F1:
{metrics["micro_f1"]:.4f}

AUROC:
{metrics["auroc"]:.4f}

Accuracy:
{metrics["accuracy"]:.4f}
"""
        )



        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **metrics,
            }
        )



        if metrics["macro_f1"] > best_macro_f1:

            best_macro_f1 = metrics["macro_f1"]


            torch.save(
                model.state_dict(),
                result_dir
                /
                "best_model.pt",
            )


            print(
                "Saved best model"
            )



    with open(
        result_dir / "history.json",
        "w",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )



if __name__ == "__main__":

    main()