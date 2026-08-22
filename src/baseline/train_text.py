import json

import torch
import torch.nn as nn

from tqdm import tqdm

from torch.utils.data import DataLoader

from src.baseline import config

from src.baseline.text_dataset import (
    COdeTextDataset,
)

from src.baseline.text_model import (
    TextOnlyBaseline,
)

from src.baseline.metrics import (
    compute_metrics,
)



def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0


    for batch in tqdm(
        loader,
        desc="Training",
        leave=False,
    ):

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)


        optimizer.zero_grad()


        logits = model(
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


    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].to(device)


            logits = model(
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


    train_dataset = COdeTextDataset(
        csv_path=config.DATASET_PATH,
        split=config.TRAIN_SPLIT,
        tokenizer_name=config.TEXT_MODEL_NAME,
        max_length=config.TEXT_MAX_LENGTH,
    )


    val_dataset = COdeTextDataset(
        csv_path=config.DATASET_PATH,
        split=config.VALID_SPLIT,
        tokenizer_name=config.TEXT_MODEL_NAME,
        max_length=config.TEXT_MAX_LENGTH,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TEXT_BATCH_SIZE,
        shuffle=True,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TEXT_BATCH_SIZE,
        shuffle=False,
    )


    model = TextOnlyBaseline(
        model_name=config.TEXT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
    )


    model.to(device)


    criterion = nn.BCEWithLogitsLoss()


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.TEXT_LEARNING_RATE,
        weight_decay=config.TEXT_WEIGHT_DECAY,
    )


    output_dir = (
        config.RESULT_ROOT
        /
        config.TEXT_EXPERIMENT_NAME
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    best_f1 = -1

    history = []


    for epoch in range(
        config.TEXT_NUM_EPOCHS
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
Epoch {epoch+1}/{config.TEXT_NUM_EPOCHS}

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


            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_f1": best_f1,
                    "epoch": epoch + 1,
                },
                output_dir
                /
                "best_model.pt",
            )


    with open(
        output_dir / "history.json",
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