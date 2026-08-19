"""
Training pipeline for COde baseline experiments.
"""

import json

import torch
import torch.nn as nn

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

from src.baseline.metrics import (
    compute_metrics,
)

from src.baseline.utils import (
    move_image_batch_to_device,
    get_modality_batch,
)



def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0


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


        optimizer.zero_grad()


        logits = model(
            images
        )


        loss = criterion(
            logits,
            labels,
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()


    return total_loss / len(loader)



def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

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
        all_logits
    ).numpy()


    labels = torch.cat(
        all_labels
    ).numpy()


    metrics = compute_metrics(
        logits,
        labels,
    )


    metrics["loss"] = (
        total_loss / len(loader)
    )


    return metrics



def main():


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"Device: {device}"
    )


    transform = get_image_transform()


    train_dataset = COdeBaselineDataset(
        csv_path=config.DATASET_PATH,
        split=config.TRAIN_SPLIT,
        image_root=config.IMAGE_ROOT,
        transform=transform,
        require_modality=config.REQUIRE_MODALITY,
    )


    val_dataset = COdeBaselineDataset(
        csv_path=config.DATASET_PATH,
        split=config.VALID_SPLIT,
        image_root=config.IMAGE_ROOT,
        transform=transform,
        require_modality=config.REQUIRE_MODALITY,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=baseline_collate,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=baseline_collate,
    )


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


    model = model.to(
        device
    )


    criterion = nn.BCEWithLogitsLoss()


    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )


    best_f1 = -1

    patience_counter = 0


    output_dir = (
        config.RESULT_ROOT
        /
        config.EXPERIMENT_NAME
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    history = []


    for epoch in range(
        config.NUM_EPOCHS
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )


        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )


        print(
            f"""
Epoch {epoch+1}/{config.NUM_EPOCHS}

Train Loss:
{train_loss:.4f}

Validation:
{val_metrics}
"""
        )


        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **val_metrics,
            }
        )


        if val_metrics["macro_f1"] > best_f1:

            best_f1 = val_metrics["macro_f1"]

            patience_counter = 0


            torch.save(
                model.state_dict(),
                output_dir
                /
                "best_model.pt",
            )


        else:

            patience_counter += 1


        if (
            patience_counter
            >= config.EARLY_STOPPING_PATIENCE
        ):

            print(
                "Early stopping."
            )

            break


    with open(
        output_dir
        /
        "history.json",
        "w",
    ) as f:

        json.dump(
            history,
            f,
            indent=4,
        )


    print(
        "Training finished."
    )



if __name__ == "__main__":

    main()