from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# CONFIG
# ============================================================================

ROOT = Path("results/baseline/final_strong_fusion_screen")
CACHE_DIR = ROOT / "feature_cache"
OUTPUT_PATH = ROOT / "fusion_diagnostic_v2_results.json"

TRAIN_CACHE = CACHE_DIR / "train_v1.pt"
VAL_CACHE = CACHE_DIR / "validation_v1.pt"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 7
LR = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN = 256


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# CACHE LOADING
# ============================================================================

def load_cache(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Feature cache not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict cache, got {type(data)}")

    required = ["text", "photograph", "radiograph", "labels"]

    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"{path} is missing keys: {missing}. "
            f"Available keys: {list(data.keys())}"
        )

    return {
        "text": data["text"].float(),
        "photograph": data["photograph"].float(),
        "radiograph": data["radiograph"].float(),
        "labels": data["labels"].float(),
    }


# ============================================================================
# METRICS
# ============================================================================

def multilabel_macro_f1(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = (np.asarray(y_prob) >= threshold).astype(np.int64)

    scores = []

    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))

        denom = 2 * tp + fp + fn

        if denom == 0:
            f1 = 0.0
        else:
            f1 = (2 * tp) / denom

        scores.append(f1)

    return float(np.mean(scores))


def multilabel_micro_f1(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = (np.asarray(y_prob) >= threshold).astype(np.int64)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    denom = 2 * tp + fp + fn

    if denom == 0:
        return 0.0

    return float((2 * tp) / denom)


def multilabel_auroc(y_true, y_prob):
    from sklearn.metrics import roc_auc_score

    values = []

    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_prob[:, j]

        if len(np.unique(yt)) < 2:
            continue

        values.append(roc_auc_score(yt, yp))

    return float(np.mean(values))


# ============================================================================
# LOSS
# ============================================================================

def compute_pos_weights(labels):
    labels = labels.float()

    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives

    weights = negatives / positives.clamp_min(1.0)

    return weights


# ============================================================================
# NORMALIZATION
# ============================================================================

class FeatureNormalizer(nn.Module):
    """
    Per-modality LayerNorm.

    Unlike train-set standardization, this does not require storing
    dataset statistics.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class Standardize(nn.Module):
    """
    Fixed train-set standardization:
        x' = (x - mean) / std

    Statistics are calculated from TRAIN ONLY.
    """

    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(1e-6))

    def forward(self, x):
        return (x - self.mean) / self.std


# ============================================================================
# SHARED BUILDING BLOCKS
# ============================================================================

class Projection(nn.Module):
    def __init__(self, input_dim, output_dim=HIDDEN):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden=HIDDEN):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 6),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# TEST 1
# PER-MODALITY NORMALIZATION + LEARNABLE SCALE
# ============================================================================

class NormalizedFusion(nn.Module):
    """
    Each modality is independently normalized and projected.

    A learnable positive scale controls how much each modality contributes.
    """

    def __init__(self):
        super().__init__()

        self.text = Projection(768)
        self.photo = Projection(128)
        self.xray = Projection(128)

        self.log_scales = nn.Parameter(torch.zeros(3))

        self.head = MLPHead(HIDDEN)

    def forward(self, text, photo, xray):
        t = self.text(text)
        p = self.photo(photo)
        x = self.xray(xray)

        scales = torch.exp(self.log_scales).clamp(0.05, 20.0)

        fused = (
            scales[0] * t
            + scales[1] * p
            + scales[2] * x
        )

        return self.head(fused)


# ============================================================================
# TEST 2
# TRAIN-SET STANDARDIZATION + PROJECT-CONCAT
# ============================================================================

class StandardizedFusion(nn.Module):
    """
    Train-set standardized raw features followed by independent projections
    and concatenation.

    Standardization statistics are fixed and calculated from TRAIN ONLY.
    """

    def __init__(self, text_mean, text_std, photo_mean, photo_std,
                 xray_mean, xray_std):
        super().__init__()

        self.text_norm = Standardize(text_mean, text_std)
        self.photo_norm = Standardize(photo_mean, photo_std)
        self.xray_norm = Standardize(xray_mean, xray_std)

        self.text = nn.Sequential(
            nn.Linear(768, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
        )

        self.photo = nn.Sequential(
            nn.Linear(128, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
        )

        self.xray = nn.Sequential(
            nn.Linear(128, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN * 3),
            nn.Linear(HIDDEN * 3, HIDDEN),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(HIDDEN, 6),
        )

    def forward(self, text, photo, xray):
        t = self.text(self.text_norm(text))
        p = self.photo(self.photo_norm(photo))
        x = self.xray(self.xray_norm(xray))

        fused = torch.cat([t, p, x], dim=1)

        return self.head(fused)


# ============================================================================
# TEST 3
# MODALITY ADAPTERS + TRUE LABEL-SPECIFIC FUSION
# ============================================================================

class LabelSpecificFusion(nn.Module):
    """
    Each modality first receives a small task-specific adapter.

    Then, for EACH LABEL independently, a separate modality weighting is
    predicted.

    Output representation:
        [B, 6, HIDDEN]

    There is NO averaging over labels.

    Each label gets its own fused representation before its classifier.
    """

    def __init__(self):
        super().__init__()

        self.text_adapter = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(HIDDEN, HIDDEN),
        )

        self.photo_adapter = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(HIDDEN, HIDDEN),
        )

        self.xray_adapter = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(HIDDEN, HIDDEN),
        )

        # Residual scaling keeps the adapter close to identity-like
        # behavior initially.
        self.text_res_scale = nn.Parameter(torch.tensor(0.1))
        self.photo_res_scale = nn.Parameter(torch.tensor(0.1))
        self.xray_res_scale = nn.Parameter(torch.tensor(0.1))

        # For each label:
        #   3 modality logits
        #
        # Output shape:
        #   [B, 6, 3]
        self.gate = nn.Sequential(
            nn.Linear(HIDDEN * 3, HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(HIDDEN, 18),
        )

        # Label-specific refinement.
        #
        # One independent small MLP per label.
        self.label_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(HIDDEN),
                    nn.Linear(HIDDEN, 128),
                    nn.GELU(),
                    nn.Dropout(0.15),
                    nn.Linear(128, 1),
                )
                for _ in range(6)
            ]
        )

    def forward(self, text, photo, xray):

        t0 = self.text_adapter(text)
        p0 = self.photo_adapter(photo)
        x0 = self.xray_adapter(xray)

        # Residual adapter behavior.
        #
        # Normalize original projected representations first so their
        # contribution scales are controlled.
        t = t0 + self.text_res_scale * t0
        p = p0 + self.photo_res_scale * p0
        x = x0 + self.xray_res_scale * x0

        modality_cat = torch.cat([t, p, x], dim=1)

        gate_logits = self.gate(modality_cat)
        gate_logits = gate_logits.view(-1, 6, 3)

        gates = torch.softmax(gate_logits, dim=-1)

        modalities = torch.stack(
            [t, p, x],
            dim=2,
        )

        # [B, 6, 3] x [B, H, 3]
        # -> [B, 6, H]
        fused = torch.bmm(
            gates,
            modalities.transpose(1, 2),
        )

        outputs = []

        for label_idx, head in enumerate(self.label_heads):
            label_feature = fused[:, label_idx, :]
            outputs.append(head(label_feature))

        return torch.cat(outputs, dim=1)


# ============================================================================
# DATASET
# ============================================================================

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.text = data["text"]
        self.photo = data["photograph"]
        self.xray = data["radiograph"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.text[idx],
            self.photo[idx],
            self.xray[idx],
            self.labels[idx],
        )


# ============================================================================
# TRAIN / EVAL
# ============================================================================

def evaluate(model, loader):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for text, photo, xray, labels in loader:
            text = text.to(DEVICE, non_blocking=True)
            photo = photo.to(DEVICE, non_blocking=True)
            xray = xray.to(DEVICE, non_blocking=True)

            logits = model(text, photo, xray)

            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(labels)

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    macro_f1 = multilabel_macro_f1(labels, probs)
    micro_f1 = multilabel_micro_f1(labels, probs)
    auroc = multilabel_auroc(labels, probs)

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "auroc": auroc,
    }


def train_model(name, model, train_loader, val_loader, pos_weights):

    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights.to(DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    history = []

    print()
    print("=" * 72)
    print(f"TEST: {name}")
    print("=" * 72)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        running_loss = 0.0
        samples = 0

        for text, photo, xray, labels in train_loader:

            text = text.to(DEVICE, non_blocking=True)
            photo = photo.to(DEVICE, non_blocking=True)
            xray = xray.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(text, photo, xray)

            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss in {name} at epoch {epoch}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += loss.item() * len(labels)
            samples += len(labels)

        metrics = evaluate(model, val_loader)

        epoch_loss = running_loss / max(samples, 1)

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                **metrics,
            }
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            best_metrics = metrics.copy()

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:02d} | "
            f"Loss {epoch_loss:.4f} | "
            f"Val F1 {metrics['macro_f1']:.4f} | "
            f"Micro {metrics['micro_f1']:.4f} | "
            f"AUROC {metrics['auroc']:.4f} | "
            f"Patience {patience_counter}/{PATIENCE}"
        )

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)

    print()
    print(f"BEST: Macro F1 = {best_metrics['macro_f1']:.4f}")
    print(f"Micro F1       = {best_metrics['micro_f1']:.4f}")
    print(f"AUROC          = {best_metrics['auroc']:.4f}")
    print(f"Epoch          = {best_epoch}")

    return {
        "name": name,
        "best_epoch": best_epoch,
        "best_macro_f1": best_metrics["macro_f1"],
        "best_micro_f1": best_metrics["micro_f1"],
        "best_auroc": best_metrics["auroc"],
        "history": history,
    }


# ============================================================================
# TRAIN-ONLY STANDARDIZATION STATISTICS
# ============================================================================

def compute_statistics(train_data):

    stats = {}

    for name in ["text", "photograph", "radiograph"]:

        x = train_data[name].float()

        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False)

        stats[name] = {
            "mean": mean,
            "std": std,
        }

    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():

    seed_everything(SEED)

    print("=" * 72)
    print("FUSION DIAGNOSTIC V2")
    print("=" * 72)
    print(f"Device:          {DEVICE}")
    print(f"Train cache:     {TRAIN_CACHE}")
    print(f"Validation:      {VAL_CACHE}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Max epochs:      {MAX_EPOCHS}")
    print(f"Patience:        {PATIENCE}")
    print()

    train_data = load_cache(TRAIN_CACHE)
    val_data = load_cache(VAL_CACHE)

    print("Feature shapes:")
    print(f"  text:        {tuple(train_data['text'].shape)}")
    print(f"  photograph:  {tuple(train_data['photograph'].shape)}")
    print(f"  radiograph:  {tuple(train_data['radiograph'].shape)}")
    print(f"  labels:      {tuple(train_data['labels'].shape)}")

    if train_data["text"].shape[0] != 2935:
        raise ValueError("Unexpected train size")

    if val_data["text"].shape[0] != 627:
        raise ValueError("Unexpected validation size")

    # ------------------------------------------------------------------------
    # DATASETS
    # ------------------------------------------------------------------------

    train_dataset = FeatureDataset(train_data)
    val_dataset = FeatureDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    pos_weights = compute_pos_weights(train_data["labels"])

    print()
    print("Positive class weights:")
    print(pos_weights.numpy())

    # ------------------------------------------------------------------------
    # TRAIN-ONLY STANDARDIZATION
    # ------------------------------------------------------------------------

    stats = compute_statistics(train_data)

    # ------------------------------------------------------------------------
    # MODELS
    # ------------------------------------------------------------------------

    models = [
        (
            "normalized_scale",
            NormalizedFusion(),
        ),
        (
            "train_standardized",
            StandardizedFusion(
                stats["text"]["mean"],
                stats["text"]["std"],
                stats["photograph"]["mean"],
                stats["photograph"]["std"],
                stats["radiograph"]["mean"],
                stats["radiograph"]["std"],
            ),
        ),
        (
            "adapter_label_specific",
            LabelSpecificFusion(),
        ),
    ]

    results = []

    for name, model in models:

        seed_everything(SEED)

        result = train_model(
            name=name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            pos_weights=pos_weights,
        )

        results.append(result)

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    reference = 0.8271
    v2 = 0.8137

    print()
    print("=" * 72)
    print("DIAGNOSTIC V2 SUMMARY")
    print("=" * 72)

    print(
        f"{'Method':28s} "
        f"{'Macro F1':>10s} "
        f"{'Δ vs 0.8271':>13s} "
        f"{'Δ vs V2':>10s} "
        f"{'AUROC':>10s}"
    )

    print("-" * 72)

    for r in results:

        delta_reference = r["best_macro_f1"] - reference
        delta_v2 = r["best_macro_f1"] - v2

        print(
            f"{r['name']:28s} "
            f"{r['best_macro_f1']:10.4f} "
            f"{delta_reference:+13.4f} "
            f"{delta_v2:+10.4f} "
            f"{r['best_auroc']:10.4f}"
        )

    print("-" * 72)
    print(f"{'linear_probe_reference':28s} {reference:10.4f}")
    print(f"{'V2_end_to_end':28s} {v2:10.4f}")
    print("=" * 72)

    # ------------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------------

    serializable = {
        "config": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "hidden": HIDDEN,
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "feature_source": "frozen V2 representations",
            "reference_linear_probe_macro_f1": reference,
            "v2_macro_f1": v2,
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(serializable, f, indent=2)

    print()
    print(f"Results saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()