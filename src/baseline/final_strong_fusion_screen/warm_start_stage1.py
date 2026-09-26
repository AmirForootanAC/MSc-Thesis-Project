from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_ROOT = Path(
    "results/baseline/final_strong_fusion_screen/feature_cache"
)

TRAIN_FEATURES = FEATURE_ROOT / "train_v1.pt"
VAL_FEATURES = FEATURE_ROOT / "validation_v1.pt"

LABEL_NAMES = [
    "caries",
    "gingivitis",
    "malocclusion",
    "pulpitis",
    "tooth_loss",
    "tooth_structure_loss",
]

BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
SEED = 42

RESULT_ROOT = Path(
    "results/baseline/final_strong_fusion_screen/warm_start_stage1"
)


class V2FusionClassifier(nn.Module):
    """
    Exact multimodal classifier used by FinalStrongV2.

    Input:
        text       = 768
        photograph = 128
        radiograph = 128

    Total:
        1024

    Architecture:
        1024 -> 256 -> 6
    """

    def __init__(self, num_labels=6):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                1024,
                256,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                256,
                num_labels,
            ),
        )

    def forward(self, x):
        return self.classifier(x)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(path):
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    text = data["text"].float()
    photograph = data["photograph"].float()
    radiograph = data["radiograph"].float()
    labels = data["labels"].float()

    features = torch.cat(
        [
            text,
            photograph,
            radiograph,
        ],
        dim=1,
    )

    return features, labels


def compute_pos_weights(labels):
    positive_counts = labels.sum(
        dim=0
    )

    negative_counts = (
        labels.shape[0]
        - positive_counts
    )

    positive_counts = torch.clamp(
        positive_counts,
        min=1.0,
    )

    return (
        negative_counts
        / positive_counts
    )


def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(
                device
            )

            labels = labels.to(
                device
            )

            logits = model(
                features
            )

            loss = criterion(
                logits,
                labels,
            )

            batch_size = labels.shape[0]

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    logits = torch.cat(
        all_logits,
        dim=0,
    ).numpy()

    labels = torch.cat(
        all_labels,
        dim=0,
    ).numpy()

    probabilities = 1.0 / (
        1.0 + np.exp(
            -np.clip(
                logits,
                -50,
                50,
            )
        )
    )

    predictions = (
        probabilities >= 0.5
    )

    per_label_f1 = []

    for i in range(
        len(LABEL_NAMES)
    ):
        y_true = labels[:, i]
        y_pred = predictions[:, i]

        tp = np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )

        fp = np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )

        fn = np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )

        denominator = (
            2 * tp
            + fp
            + fn
        )

        if denominator == 0:
            f1 = 0.0
        else:
            f1 = (
                2 * tp
                / denominator
            )

        per_label_f1.append(
            float(f1)
        )

    macro_f1 = float(
        np.mean(
            per_label_f1
        )
    )

    tp_micro = np.sum(
        (labels == 1)
        & (predictions == 1)
    )

    fp_micro = np.sum(
        (labels == 0)
        & (predictions == 1)
    )

    fn_micro = np.sum(
        (labels == 1)
        & (predictions == 0)
    )

    denominator = (
        2 * tp_micro
        + fp_micro
        + fn_micro
    )

    micro_f1 = (
        float(
            2 * tp_micro
            / denominator
        )
        if denominator > 0
        else 0.0
    )

    accuracy = float(
        np.mean(
            np.all(
                predictions
                == labels,
                axis=1,
            )
        )
    )

    return {
        "loss": (
            total_loss
            / max(total_samples, 1)
        ),
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "accuracy": accuracy,
        "per_label_f1": per_label_f1,
    }


def main():
    set_seed(SEED)

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("V2 FUSION CLASSIFIER WARM-START — STAGE 1")
    print("=" * 80)

    print(
        "\nThis stage trains ONLY the V2 fusion classifier "
        "on frozen cached representations."
    )

    train_x, train_y = load_features(
        TRAIN_FEATURES
    )

    val_x, val_y = load_features(
        VAL_FEATURES
    )

    print(
        f"\nTrain features: {tuple(train_x.shape)}"
    )

    print(
        f"Train labels:   {tuple(train_y.shape)}"
    )

    print(
        f"Val features:   {tuple(val_x.shape)}"
    )

    print(
        f"Val labels:     {tuple(val_y.shape)}"
    )

    if train_x.shape[1] != 1024:
        raise RuntimeError(
            "Expected 1024-dimensional "
            f"fusion features, got {train_x.shape[1]}"
        )

    if train_y.shape[1] != 6:
        raise RuntimeError(
            "Expected 6 labels."
        )

    train_dataset = TensorDataset(
        train_x,
        train_y,
    )

    val_dataset = TensorDataset(
        val_x,
        val_y,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    pos_weights = compute_pos_weights(
        train_y
    )

    print("\nClass positive weights:")

    for name, weight in zip(
        LABEL_NAMES,
        pos_weights,
    ):
        print(
            f"  {name:<30} "
            f"{weight.item():.4f}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    model = V2FusionClassifier(
        num_labels=6
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights.to(
            device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_macro_f1 = -1.0
    best_epoch = None
    patience_counter = 0

    best_state = None

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for features, labels in train_loader:
            features = features.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                features
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            batch_size = labels.shape[0]

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

        train_loss = (
            total_loss
            / max(total_samples, 1)
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        current_f1 = (
            val_metrics["macro_f1"]
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val Macro F1 {current_f1:.4f} | "
            f"Val Micro F1 {val_metrics['micro_f1']:.4f}"
        )

        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            best_epoch = epoch
            patience_counter = 0

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

        else:
            patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):
            print(
                "\nEarly stopping."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No best model was produced."
        )

    model.load_state_dict(
        best_state
    )

    final_metrics = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    checkpoint_path = (
        RESULT_ROOT
        / "stage1_classifier.pt"
    )

    torch.save(
        {
            "stage": 1,
            "architecture":
                "V2 1024-256-6 fusion classifier",
            "feature_source":
                "epoch_009 V2 frozen representations",
            "best_epoch":
                best_epoch,
            "best_validation_macro_f1":
                best_macro_f1,
            "model_state_dict":
                model.state_dict(),
            "pos_weights":
                pos_weights,
        },
        checkpoint_path,
    )

    print("\n" + "=" * 80)
    print("STAGE 1 RESULT")
    print("=" * 80)

    print(
        f"Best epoch:       {best_epoch}"
    )

    print(
        f"Validation Macro F1: "
        f"{final_metrics['macro_f1']:.4f}"
    )

    print(
        f"Validation Micro F1: "
        f"{final_metrics['micro_f1']:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{final_metrics['accuracy']:.4f}"
    )

    print("\nPer-label F1:")

    for name, score in zip(
        LABEL_NAMES,
        final_metrics["per_label_f1"],
    ):
        print(
            f"  {name:<30} "
            f"{score:.4f}"
        )

    print("\nReferences:")
    print(
        "V2 end-to-end epoch 9:       0.8137"
    )
    print(
        "Frozen linear probe:          0.8271"
    )
    print(
        "Classwise Logistic Regression "
        "@ best C:                    0.8248"
    )

    print(
        f"\nCheckpoint saved to:\n"
        f"  {checkpoint_path}"
    )

    print(
        "\nTest set was NOT loaded."
    )


if __name__ == "__main__":
    main()