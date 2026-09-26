from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Config
# ============================================================

RESULT_ROOT = Path(
    "results/baseline/final_strong_fusion_screen/transformer_fusion"
)
CACHE_ROOT = Path(
    "results/baseline/final_strong_fusion_screen/feature_cache"
)

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 10

LR = 1e-3
WEIGHT_DECAY = 1e-4

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.20

NUM_LABELS = 6

MODALITIES = ("photograph", "radiograph", "text")
INPUT_DIMS = {
    "photograph": 128,
    "radiograph": 128,
    "text": 768,
}


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset
# ============================================================

class FeatureDataset(Dataset):
    def __init__(self, path: Path):
        data = torch.load(path, map_location="cpu", weights_only=False)

        self.labels = data["labels"].float()

        self.features = {
            modality: data[modality].float()
            for modality in MODALITIES
        }

        n = len(self.labels)

        for modality in MODALITIES:
            if len(self.features[modality]) != n:
                raise ValueError(
                    f"{modality} feature count mismatch: "
                    f"{len(self.features[modality])} != {n}"
                )

        if self.labels.ndim != 2 or self.labels.shape[1] != NUM_LABELS:
            raise ValueError(
                f"Expected labels shape [N, {NUM_LABELS}], "
                f"got {tuple(self.labels.shape)}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "photograph": self.features["photograph"][idx],
            "radiograph": self.features["radiograph"][idx],
            "text": self.features["text"][idx],
            "labels": self.labels[idx],
        }


# ============================================================
# Model
# ============================================================

class MultimodalTransformer(nn.Module):
    """
    Frozen modality embeddings -> modality-specific projection
    -> 3 modality tokens + CLS token
    -> small Transformer encoder
    -> CLS classifier.
    """

    def __init__(self):
        super().__init__()

        self.projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.LayerNorm(INPUT_DIMS[modality]),
                nn.Linear(INPUT_DIMS[modality], D_MODEL),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            )
            for modality in MODALITIES
        })

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, D_MODEL)
        )

        self.modality_embedding = nn.Parameter(
            torch.zeros(1, len(MODALITIES), D_MODEL)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS,
        )

        self.norm = nn.LayerNorm(D_MODEL)

        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, 96),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(96, NUM_LABELS),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)

    def forward(self, photograph, radiograph, text):
        tokens = []

        for modality, x in (
            ("photograph", photograph),
            ("radiograph", radiograph),
            ("text", text),
        ):
            tokens.append(
                self.projections[modality](x)
            )

        # [B, 3, D]
        tokens = torch.stack(tokens, dim=1)

        # Add modality identity.
        tokens = tokens + self.modality_embedding

        batch_size = tokens.shape[0]

        cls = self.cls_token.expand(
            batch_size, -1, -1
        )

        # [B, 4, D]
        x = torch.cat([cls, tokens], dim=1)

        x = self.transformer(x)

        cls_output = self.norm(x[:, 0])

        return self.classifier(cls_output)


# ============================================================
# Metrics
# ============================================================

def evaluate_logits(logits, labels, thresholds=None):
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)

    if thresholds is None:
        thresholds = np.full(NUM_LABELS, 0.5)

    predictions = (
        probabilities >= np.asarray(thresholds)
    ).astype(int)

    macro_f1 = f1_score(
        labels_np,
        predictions,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        labels_np,
        predictions,
        average="micro",
        zero_division=0,
    )

    accuracy = accuracy_score(
        labels_np,
        predictions,
    )

    try:
        auroc = roc_auc_score(
            labels_np,
            probabilities,
            average="macro",
        )
    except ValueError:
        auroc = float("nan")

    per_label_f1 = f1_score(
        labels_np,
        predictions,
        average=None,
        zero_division=0,
    ).tolist()

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "per_label_f1": [float(x) for x in per_label_f1],
    }


def optimize_thresholds(logits, labels):
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)

    thresholds = np.arange(0.05, 0.951, 0.05)

    best_thresholds = []
    best_scores = []

    for label_idx in range(NUM_LABELS):
        best_score = -1.0
        best_threshold = 0.5

        y_true = labels_np[:, label_idx]
        y_prob = probabilities[:, label_idx]

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            score = f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        best_thresholds.append(best_threshold)
        best_scores.append(best_score)

    return best_thresholds, best_scores


# ============================================================
# Loss
# ============================================================

def get_pos_weights(dataset):
    labels = dataset.labels

    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives

    weights = negatives / positives.clamp_min(1.0)

    return weights


# ============================================================
# Training / inference
# ============================================================

def collect_logits(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            photograph = batch["photograph"].to(device)
            radiograph = batch["radiograph"].to(device)
            text = batch["text"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                photograph,
                radiograph,
                text,
            )

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        photograph = batch["photograph"].to(device)
        radiograph = batch["radiograph"].to(device)
        text = batch["text"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            photograph,
            radiograph,
            text,
        )

        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# Main
# ============================================================

def main():
    seed_everything()

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("MULTIMODAL TRANSFORMER FUSION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Frozen feature cache: {CACHE_ROOT}")
    print()

    train_path = CACHE_ROOT / "train_v1.pt"
    val_path = CACHE_ROOT / "validation_v1.pt"
    test_path = CACHE_ROOT / "test_v1.pt"

    train_dataset = FeatureDataset(train_path)
    val_dataset = FeatureDataset(val_path)
    test_dataset = FeatureDataset(test_path)

    print(
        f"Train: {len(train_dataset)}"
        f" | Val: {len(val_dataset)}"
        f" | Test: {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = MultimodalTransformer().to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {params:,}")

    pos_weights = get_pos_weights(
        train_dataset
    ).to(device)

    print(
        "Positive weights:",
        [round(float(x), 4) for x in pos_weights],
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    best_checkpoint = (
        RESULT_ROOT / "best_model.pt"
    )

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_logits, val_labels = collect_logits(
            model,
            val_loader,
            device,
        )

        val_metrics = evaluate_logits(
            val_logits,
            val_labels,
        )

        val_macro = val_metrics["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{
                f"val_{k}": v
                for k, v in val_metrics.items()
                if k != "per_label_f1"
            },
        })

        marker = ""

        if val_macro > best_macro_f1:
            best_macro_f1 = val_macro
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_macro,
                },
                best_checkpoint,
            )

            marker = "  <-- BEST"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val_macro_f1={val_macro:.4f} | "
            f"val_micro_f1={val_metrics['micro_f1']:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f}"
            f"{marker}"
        )

        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}."
            )
            break

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print()
    print("=" * 70)
    print("BEST VALIDATION MODEL")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best Val Macro F1: {best_macro_f1:.4f}")

    # --------------------------------------------------------
    # Validation thresholds
    # --------------------------------------------------------

    val_logits, val_labels = collect_logits(
        model,
        val_loader,
        device,
    )

    val_default = evaluate_logits(
        val_logits,
        val_labels,
    )

    thresholds, threshold_f1 = optimize_thresholds(
        val_logits,
        val_labels,
    )

    val_optimized = evaluate_logits(
        val_logits,
        val_labels,
        thresholds,
    )

    print()
    print("Validation default:")
    print(
        f"Macro F1: {val_default['macro_f1']:.4f}"
    )
    print(
        f"Micro F1: {val_default['micro_f1']:.4f}"
    )
    print(
        f"AUROC:    {val_default['auroc']:.4f}"
    )

    print()
    print("Validation optimized thresholds:")
    print(
        f"Macro F1: {val_optimized['macro_f1']:.4f}"
    )
    print(
        "Thresholds:",
        thresholds,
    )

    # --------------------------------------------------------
    # Test — single final evaluation
    # --------------------------------------------------------

    test_logits, test_labels = collect_logits(
        model,
        test_loader,
        device,
    )

    test_default = evaluate_logits(
        test_logits,
        test_labels,
    )

    test_thresholded = evaluate_logits(
        test_logits,
        test_labels,
        thresholds,
    )

    print()
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    print(
        f"Test Macro F1 @0.5: "
        f"{test_default['macro_f1']:.4f}"
    )

    print(
        f"Test Macro F1 @val thresholds: "
        f"{test_thresholded['macro_f1']:.4f}"
    )

    print(
        f"Test Micro F1: "
        f"{test_thresholded['micro_f1']:.4f}"
    )

    print(
        f"Test AUROC: "
        f"{test_thresholded['auroc']:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_thresholded['accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "experiment": "multimodal_transformer_fusion",
        "seed": SEED,
        "device": str(device),
        "feature_source": str(
            CACHE_ROOT / "train_v1.pt"
        ),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "modalities": list(MODALITIES),
        "architecture": {
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
            "trainable_parameters": params,
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "validation_default": val_default,
        "validation_optimized": val_optimized,
        "validation_thresholds": thresholds,
        "validation_threshold_f1": threshold_f1,
        "test_default": test_default,
        "test_with_validation_thresholds": test_thresholded,
        "history": history,
    }

    with open(
        RESULT_ROOT / "results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        f"Results saved to: "
        f"{RESULT_ROOT / 'results.json'}"
    )

    print()
    print("TEST SET WAS USED ONLY FOR FINAL EVALUATION.")
    print("NO TEST TRAINING.")
    print("NO TEST THRESHOLD OPTIMIZATION.")
    print("=" * 70)


if __name__ == "__main__":
    main()
