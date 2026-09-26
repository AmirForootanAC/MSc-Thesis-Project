from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

CACHE_DIR = (
    ROOT
    / "results"
    / "baseline"
    / "final_strong_fusion_screen"
    / "feature_cache"
)

TRAIN_CACHE = CACHE_DIR / "train_v1.pt"
VAL_CACHE = CACHE_DIR / "validation_v1.pt"

RESULT_DIR = (
    ROOT
    / "results"
    / "baseline"
    / "final_strong_fusion_screen"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 30
PATIENCE = 7
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

LABELS = [
    "caries",
    "gingivitis",
    "malocclusion",
    "pulpitis",
    "tooth_loss",
    "tooth_structure_loss",
]

SCENARIOS = {
    "text": ["text"],
    "photograph": ["photograph"],
    "radiograph": ["radiograph"],
    "text_photograph": ["text", "photograph"],
    "text_radiograph": ["text", "radiograph"],
    "photograph_radiograph": ["photograph", "radiograph"],
    "full": ["text", "photograph", "radiograph"],
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL
# ============================================================

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# DATA
# ============================================================

def load_cache(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    required = {
        "labels",
        "text",
        "photograph",
        "radiograph",
    }

    missing = required - set(data.keys())

    if missing:
        raise RuntimeError(
            f"{path} is missing required keys: {sorted(missing)}"
        )

    return data


def build_features(data, modalities):
    parts = [
        data[m].float()
        for m in modalities
    ]

    return torch.cat(parts, dim=1)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, logits):
    probs = torch.sigmoid(logits).numpy()
    y = y_true.numpy()

    preds = (probs >= 0.5).astype(np.int32)

    macro_f1 = f1_score(
        y,
        preds,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y,
        preds,
        average="micro",
        zero_division=0,
    )

    per_label_f1 = f1_score(
        y,
        preds,
        average=None,
        zero_division=0,
    )

    per_label_auc = []

    for i in range(6):
        try:
            auc = roc_auc_score(y[:, i], probs[:, i])
        except ValueError:
            auc = float("nan")

        per_label_auc.append(auc)

    try:
        macro_auc = roc_auc_score(
            y,
            probs,
            average="macro",
        )
    except ValueError:
        macro_auc = float("nan")

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_auroc": float(macro_auc),
        "per_label_f1": {
            LABELS[i]: float(per_label_f1[i])
            for i in range(6)
        },
        "per_label_auroc": {
            LABELS[i]: float(per_label_auc[i])
            for i in range(6)
        },
    }


# ============================================================
# POS WEIGHTS
# ============================================================

def make_pos_weight(y):
    positives = y.sum(dim=0)
    negatives = y.shape[0] - positives

    weights = negatives / positives.clamp_min(1)

    return weights.float()


# ============================================================
# TRAIN PROBE
# ============================================================

def train_probe(
    x_train,
    y_train,
    x_val,
    y_val,
):
    model = LinearProbe(x_train.shape[1]).to(DEVICE)

    pos_weight = make_pos_weight(y_train).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    train_dataset = torch.utils.data.TensorDataset(
        x_train,
        y_train,
    )

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    best_f1 = -1.0
    best_state = None
    patience = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)

            loss = criterion(logits, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

        model.eval()

        with torch.no_grad():
            val_logits = []

            for start in range(
                0,
                len(x_val),
                BATCH_SIZE,
            ):
                xb = x_val[
                    start:start + BATCH_SIZE
                ].to(DEVICE)

                val_logits.append(
                    model(xb).cpu()
                )

            val_logits = torch.cat(
                val_logits,
                dim=0,
            )

        metrics = compute_metrics(
            y_val,
            val_logits,
        )

        val_f1 = metrics["macro_f1"]

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1

        if patience >= PATIENCE:
            break

    model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        train_logits = []

        for start in range(
            0,
            len(x_train),
            BATCH_SIZE,
        ):
            xb = x_train[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            train_logits.append(
                model(xb).cpu()
            )

        train_logits = torch.cat(
            train_logits,
            dim=0,
        )

        val_logits = []

        for start in range(
            0,
            len(x_val),
            BATCH_SIZE,
        ):
            xb = x_val[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            val_logits.append(
                model(xb).cpu()
            )

        val_logits = torch.cat(
            val_logits,
            dim=0,
        )

    return (
        compute_metrics(y_train, train_logits),
        compute_metrics(y_val, val_logits),
    )


# ============================================================
# REPRESENTATION DIAGNOSTICS
# ============================================================


def representation_statistics(data):
    print()
    print("=" * 70)
    print("REPRESENTATION STATISTICS")
    print("=" * 70)

    modalities = [
        "text",
        "photograph",
        "radiograph",
    ]

    for modality in modalities:
        x = data[modality].float()

        norms = torch.linalg.vector_norm(
            x,
            dim=1,
        )

        print(
            f"{modality:12s} | "
            f"dim={x.shape[1]:4d} | "
            f"mean_norm={norms.mean():.4f} | "
            f"std_norm={norms.std():.4f} | "
            f"mean_abs={x.abs().mean():.4f} | "
            f"std={x.std():.4f}"
        )

    # --------------------------------------------------------
    # Label-wise linear signal from each representation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SINGLE-MODALITY REPRESENTATION SIGNAL")
    print("=" * 70)

    print(
        "The linear probes below are the meaningful comparison "
        "because modality dimensions differ."
    )

    # No pairwise cosine here.
    # Text=768, photograph=128, radiograph=128.


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(SEED)

    print("=" * 70)
    print("FAST MULTIMODAL REPRESENTATION DIAGNOSTIC")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Train cache: {TRAIN_CACHE}")
    print(f"Val cache:   {VAL_CACHE}")
    print()

    train = load_cache(TRAIN_CACHE)
    val = load_cache(VAL_CACHE)

    print(
        f"Train samples: {len(train['labels'])}"
    )
    print(
        f"Validation samples: {len(val['labels'])}"
    )

    representation_statistics(train)

    y_train = train["labels"].float()
    y_val = val["labels"].float()

    all_results = {}

    # --------------------------------------------------------
    # Probe every modality combination
    # --------------------------------------------------------

    for scenario, modalities in SCENARIOS.items():

        print()
        print("=" * 70)
        print(f"LINEAR PROBE: {scenario}")
        print(
            f"Modalities: {', '.join(modalities)}"
        )
        print("=" * 70)

        x_train = build_features(
            train,
            modalities,
        )

        x_val = build_features(
            val,
            modalities,
        )

        print(
            f"Feature dimension: {x_train.shape[1]}"
        )

        train_metrics, val_metrics = train_probe(
            x_train,
            y_train,
            x_val,
            y_val,
        )

        all_results[scenario] = {
            "modalities": modalities,
            "feature_dim": int(x_train.shape[1]),
            "train": train_metrics,
            "validation": val_metrics,
        }

        print(
            f"Best Val Macro F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val Micro F1: "
            f"{val_metrics['micro_f1']:.4f}"
        )

        print(
            f"Val AUROC: "
            f"{val_metrics['macro_auroc']:.4f}"
        )

        print()
        print("Per-label Val F1:")

        for label in LABELS:
            print(
                f"  {label:22s} "
                f"{val_metrics['per_label_f1'][label]:.4f}"
            )

    # --------------------------------------------------------
    # Incremental modality gains
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INCREMENTAL MODALITY GAINS")
    print("=" * 70)

    pairs = [
        ("text", "text_photograph"),
        ("text", "text_radiograph"),
        ("photograph", "photograph_radiograph"),
        ("text_photograph", "full"),
        ("text_radiograph", "full"),
    ]

    for base, extended in pairs:

        base_f1 = all_results[base][
            "validation"
        ]["macro_f1"]

        extended_f1 = all_results[extended][
            "validation"
        ]["macro_f1"]

        delta = extended_f1 - base_f1

        print(
            f"{base:24s} → {extended:24s} "
            f"{base_f1:.4f} → {extended_f1:.4f} "
            f"Δ {delta:+.4f}"
        )

    # --------------------------------------------------------
    # X-ray label-level gain
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("X-RAY LABEL-LEVEL CONTRIBUTION")
    print("=" * 70)

    text_f1 = all_results["text"][
        "validation"
    ]["per_label_f1"]

    text_xray_f1 = all_results["text_radiograph"][
        "validation"
    ]["per_label_f1"]

    text_photo_f1 = all_results["text_photograph"][
        "validation"
    ]["per_label_f1"]

    full_f1 = all_results["full"][
        "validation"
    ]["per_label_f1"]

    print(
        f"{'Label':24s} "
        f"{'Text':>8s} "
        f"{'T+X':>8s} "
        f"{'T+P':>8s} "
        f"{'Full':>8s} "
        f"{'X gain':>9s}"
    )

    print("-" * 70)

    for label in LABELS:

        t = text_f1[label]
        tx = text_xray_f1[label]
        tp = text_photo_f1[label]
        f = full_f1[label]

        print(
            f"{label:24s} "
            f"{t:8.4f} "
            f"{tx:8.4f} "
            f"{tp:8.4f} "
            f"{f:8.4f} "
            f"{tx - t:+9.4f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        RESULT_DIR
        / "representation_diagnostic.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(
        f"Saved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()