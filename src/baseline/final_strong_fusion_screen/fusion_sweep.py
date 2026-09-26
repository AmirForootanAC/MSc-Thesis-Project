from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

OUTPUT_PATH = (
    ROOT
    / "results"
    / "baseline"
    / "final_strong_fusion_screen"
    / "fusion_sweep_results.json"
)

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 40
PATIENCE = 7
BATCH_SIZE = 64

LR = 1e-3
WEIGHT_DECAY = 1e-4

D_MODEL = 256
D_HIDDEN = 256
D_LABEL = 128

DROPOUT = 0.20

LABELS = [
    "caries",
    "gingivitis",
    "malocclusion",
    "pulpitis",
    "tooth_loss",
    "tooth_structure_loss",
]


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
# DATA
# ============================================================

def load_cache(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "labels",
        "text",
        "photograph",
        "radiograph",
    }

    missing = required - set(data.keys())

    if missing:
        raise RuntimeError(
            f"Missing keys in {path}: {sorted(missing)}"
        )

    return {
        key: data[key].float()
        for key in required
    }


# ============================================================
# METRICS
# ============================================================

def metrics_from_logits(
    y_true: torch.Tensor,
    logits: torch.Tensor,
):
    y = y_true.numpy()
    probs = torch.sigmoid(logits).numpy()

    preds = (
        probs >= 0.5
    ).astype(np.int32)

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

    try:
        auroc = roc_auc_score(
            y,
            probs,
            average="macro",
        )
    except ValueError:
        auroc = float("nan")

    per_label = f1_score(
        y,
        preds,
        average=None,
        zero_division=0,
    )

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "auroc": float(auroc),
        "per_label_f1": {
            LABELS[i]: float(per_label[i])
            for i in range(6)
        },
    }


# ============================================================
# COMMON BUILDING BLOCKS
# ============================================================

class Projection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = D_MODEL,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ResidualMLP(nn.Module):
    def __init__(
        self,
        dim: int = D_MODEL,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, D_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_HIDDEN, dim),
            nn.Dropout(DROPOUT),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(
            x + self.net(x)
        )


class Classifier(nn.Module):
    def __init__(
        self,
        input_dim: int = D_MODEL,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(
                input_dim,
                D_HIDDEN,
            ),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(
                D_HIDDEN,
                6,
            ),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 1. CONCAT MLP
# ============================================================

class ConcatMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 6),
        )

    def forward(
        self,
        text,
        photograph,
        radiograph,
    ):
        x = torch.cat(
            [
                text,
                photograph,
                radiograph,
            ],
            dim=1,
        )

        return self.net(x)


# ============================================================
# 2. PROJECT → CONCAT
# ============================================================

class ProjectConcat(nn.Module):
    def __init__(self):
        super().__init__()

        self.text = Projection(768)
        self.photo = Projection(128)
        self.xray = Projection(128)

        self.fusion = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 6),
        )

    def forward(
        self,
        text,
        photograph,
        radiograph,
    ):
        x = torch.cat(
            [
                self.text(text),
                self.photo(photograph),
                self.xray(radiograph),
            ],
            dim=1,
        )

        return self.fusion(x)


# ============================================================
# 3. ADDITIVE FUSION
# ============================================================

class AdditiveFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.text = Projection(768)
        self.photo = Projection(128)
        self.xray = Projection(128)

        self.fusion = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_HIDDEN, 6),
        )

    def forward(
        self,
        text,
        photograph,
        radiograph,
    ):
        x = (
            self.text(text)
            + self.photo(photograph)
            + self.xray(radiograph)
        )

        return self.fusion(x)


# ============================================================
# 4. LEARNABLE GATED FUSION
# ============================================================

class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.text = Projection(768)
        self.photo = Projection(128)
        self.xray = Projection(128)

        self.gates = nn.Parameter(
            torch.zeros(3)
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 6),
        )

    def forward(
        self,
        text,
        photograph,
        radiograph,
    ):
        t = self.text(text)
        p = self.photo(photograph)
        x = self.xray(radiograph)

        gates = torch.sigmoid(
            self.gates
        )

        fused = (
            gates[0] * t
            + gates[1] * p
            + gates[2] * x
        )

        return self.fusion(fused)


# ============================================================
# 5. SOFTMAX MODALITY MIXTURE
# ============================================================

class SoftmaxFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.text = Projection(768)
        self.photo = Projection(128)
        self.xray = Projection(128)

        self.logits = nn.Parameter(
            torch.zeros(3)
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 6),
        )

    def forward(
        self,
        text,
        photograph,
        radiograph,
    ):
        t = self.text(text)
        p = self.photo(photograph)
        x = self.xray(radiograph)

        weights = torch.softmax(
            self.logits,
            dim=0,
        )

        fused = (
            weights[0] * t
            + weights[1] * p
            + weights[2] * x
        )

        return self.fusion(fused)


# ============================================================
# 6. GATED CONCATENATION
# ============================================================

class GatedConcat(nn.Module):
    def __init__(self):
        super().__init__()

        self.text = Projection(768)
        self.photo = Projection(128)
        self.xray = Projection(128)

        self.gate_text = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.Sigmoid(),
        )

        self.gate_photo = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.Sigmoid(),
        )

        self.gate_xray = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 6),
        )

    def forward(
        self,
        text,
        photograph,
        radiograph,
    ):
        t = self.text(text)
        p = self.photo(photograph)
        x = self.xray(radiograph)

        t = t * self.gate_text(t)
        p = p * self.gate_photo(p)
        x = x * self.gate_xray(x)

        fused = torch.cat(
            [t, p, x],
            dim=1,
        )

        return self.fusion(fused)


# ============================================================
# 7. ATTENTION FUSION
# ============================================================

class AttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.text = Projection(768)
        self.photo = Projection(128)
        self.xray = Projection(128)

        self.attention = nn.MultiheadAttention(
            embed_dim=D_MODEL,
            num_heads=4,
            dropout=DROPOUT,
            batch_first=True,
        )

        self.residual = ResidualMLP(
            D_MODEL
        )

        self.classifier = Classifier(
            D_MODEL
        )

    def forward(
        self,
        text,
        photograph,
        radiograph,
    ):
        tokens = torch.stack(
            [
                self.text(text),
                self.photo(photograph),
                self.xray(radiograph),
            ],
            dim=1,
        )

        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )

        fused = attended.mean(
            dim=1
        )

        fused = self.residual(
            fused
        )

        return self.classifier(
            fused
        )


# ============================================================
# 8. TEXT-ANCHORED RESIDUAL FUSION
# ============================================================

class TextAnchoredFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.text = Projection(768)
        self.photo = Projection(128)
        self.xray = Projection(128)

        self.photo_gate = nn.Sequential(
            nn.Linear(
                D_MODEL * 2,
                D_MODEL,
            ),
            nn.GELU(),
            nn.Linear(
                D_MODEL,
                D_MODEL,
            ),
            nn.Sigmoid(),
        )

        self.xray_gate = nn.Sequential(
            nn.Linear(
                D_MODEL * 2,
                D_MODEL,
            ),
            nn.GELU(),
            nn.Linear(
                D_MODEL,
                D_MODEL,
            ),
            nn.Sigmoid(),
        )

        self.residual = ResidualMLP(
            D_MODEL
        )

        self.classifier = Classifier(
            D_MODEL
        )

    def forward(
        self,
        text,
        photograph,
        radiograph,
    ):
        t = self.text(text)
        p = self.photo(photograph)
        x = self.xray(radiograph)

        p_gate = self.photo_gate(
            torch.cat([t, p], dim=1)
        )

        x_gate = self.xray_gate(
            torch.cat([t, x], dim=1)
        )

        fused = (
            t
            + p_gate * p
            + x_gate * x
        )

        fused = self.residual(
            fused
        )

        return self.classifier(
            fused
        )


# ============================================================
# 9. BILINEAR-STYLE INTERACTION
# ============================================================

class InteractionFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.text = Projection(768)
        self.photo = Projection(128)
        self.xray = Projection(128)

        self.text_photo = nn.Sequential(
            nn.Linear(
                D_MODEL,
                D_MODEL,
            ),
            nn.GELU(),
        )

        self.text_xray = nn.Sequential(
            nn.Linear(
                D_MODEL,
                D_MODEL,
            ),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(
                D_MODEL * 3
            ),
            nn.Linear(
                D_MODEL * 3,
                512,
            ),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(
                512,
                256,
            ),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(
                256,
                6,
            ),
        )

    def forward(
        self,
        text,
        photograph,
        radiograph,
    ):
        t = self.text(text)
        p = self.photo(photograph)
        x = self.xray(radiograph)

        tp = self.text_photo(
            t * p
        )

        tx = self.text_xray(
            t * x
        )

        fused = torch.cat(
            [
                t,
                tp,
                tx,
            ],
            dim=1,
        )

        return self.fusion(
            fused
        )


# ============================================================
# 10. LABEL-AWARE MODALITY FUSION
# ============================================================

class LabelAwareFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.text = Projection(768)
        self.photo = Projection(128)
        self.xray = Projection(128)

        self.gate = nn.Sequential(
            nn.Linear(
                D_MODEL,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                18,
            ),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(
                D_MODEL,
                128,
            ),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        self.classifier = nn.Linear(
            128,
            6,
        )

    def forward(
        self,
        text,
        photograph,
        radiograph,
    ):
        t = self.text(text)
        p = self.photo(photograph)
        x = self.xray(radiograph)

        base = (
            t + p + x
        )

        gate_logits = self.gate(
            base
        ).view(
            -1,
            6,
            3,
        )

        weights = torch.softmax(
            gate_logits,
            dim=-1,
        )

        tokens = torch.stack(
            [t, p, x],
            dim=1,
        )

        # [B, 6, 3] × [B, 3, 256]
        fused = torch.bmm(
            weights,
            tokens,
        )

        # One representation per label.
        fused = fused.mean(
            dim=1
        )

        fused = self.head(
            fused
        )

        return self.classifier(
            fused
        )


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "concat_mlp": ConcatMLP,
    "project_concat": ProjectConcat,
    "additive": AdditiveFusion,
    "gated": GatedFusion,
    "softmax": SoftmaxFusion,
    "gated_concat": GatedConcat,
    "attention": AttentionFusion,
    "text_anchored": TextAnchoredFusion,
    "interaction": InteractionFusion,
    "label_aware": LabelAwareFusion,
}


# ============================================================
# TRAIN / EVAL
# ============================================================

@torch.no_grad()
def predict(
    model,
    text,
    photograph,
    radiograph,
):
    model.eval()

    outputs = []

    for start in range(
        0,
        len(text),
        BATCH_SIZE,
    ):
        end = start + BATCH_SIZE

        t = text[start:end].to(
            DEVICE
        )

        p = photograph[start:end].to(
            DEVICE
        )

        x = radiograph[start:end].to(
            DEVICE
        )

        logits = model(
            t,
            p,
            x,
        )

        outputs.append(
            logits.cpu()
        )

    return torch.cat(
        outputs,
        dim=0,
    )


def train_model(
    model,
    train,
    val,
):
    text_train = train["text"]
    photo_train = train["photograph"]
    xray_train = train["radiograph"]
    y_train = train["labels"]

    text_val = val["text"]
    photo_val = val["photograph"]
    xray_val = val["radiograph"]
    y_val = val["labels"]

    positives = y_train.sum(
        dim=0
    )

    negatives = (
        len(y_train)
        - positives
    )

    pos_weight = (
        negatives
        / positives.clamp_min(1)
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    indices = torch.arange(
        len(y_train)
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        permutation = indices[
            torch.randperm(
                len(indices)
            )
        ]

        for start in range(
            0,
            len(permutation),
            BATCH_SIZE,
        ):
            batch_idx = permutation[
                start:start + BATCH_SIZE
            ]

            t = text_train[
                batch_idx
            ].to(DEVICE)

            p = photo_train[
                batch_idx
            ].to(DEVICE)

            x = xray_train[
                batch_idx
            ].to(DEVICE)

            y = y_train[
                batch_idx
            ].to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                t,
                p,
                x,
            )

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

        val_logits = predict(
            model,
            text_val,
            photo_val,
            xray_val,
        )

        val_metrics = metrics_from_logits(
            y_val,
            val_logits,
        )

        val_f1 = val_metrics[
            "macro_f1"
        ]

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience = 0

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict().items()
            }
        else:
            patience += 1

        print(
            f"    Epoch {epoch:02d} | "
            f"Val F1 {val_f1:.4f} | "
            f"AUROC {val_metrics['auroc']:.4f} | "
            f"Patience {patience}/{PATIENCE}"
        )

        if patience >= PATIENCE:
            break

    model.load_state_dict(
        best_state
    )

    final_logits = predict(
        model,
        text_val,
        photo_val,
        xray_val,
    )

    final_metrics = metrics_from_logits(
        y_val,
        final_logits,
    )

    return (
        best_epoch,
        final_metrics,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(SEED)

    print("=" * 72)
    print("MULTIMODAL FUSION ARCHITECTURE SWEEP")
    print("=" * 72)

    print(
        f"Device:          {DEVICE}"
    )
    print(
        f"Train samples:   2935"
    )
    print(
        f"Validation:      627"
    )
    print(
        f"Batch size:      {BATCH_SIZE}"
    )
    print(
        f"Max epochs:      {EPOCHS}"
    )
    print(
        f"Patience:        {PATIENCE}"
    )
    print(
        f"Feature source:  frozen V2 representations"
    )

    train = load_cache(
        TRAIN_CACHE
    )

    val = load_cache(
        VAL_CACHE
    )

    y_train = train["labels"]
    y_val = val["labels"]

    print()
    print(
        "Feature shapes:"
    )

    print(
        f"  text:        {tuple(train['text'].shape)}"
    )
    print(
        f"  photograph:  {tuple(train['photograph'].shape)}"
    )
    print(
        f"  radiograph:  {tuple(train['radiograph'].shape)}"
    )

    results = {}

    for name, model_cls in MODELS.items():

        print()
        print("=" * 72)
        print(
            f"FUSION: {name}"
        )
        print("=" * 72)

        seed_everything(
            SEED
        )

        model = model_cls().to(
            DEVICE
        )

        parameter_count = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(
            f"Trainable parameters: "
            f"{parameter_count:,}"
        )

        best_epoch, metrics = train_model(
            model,
            train,
            val,
        )

        results[name] = {
            "best_epoch": best_epoch,
            "trainable_parameters": parameter_count,
            "validation": metrics,
        }

        print()
        print(
            f"BEST: "
            f"Macro F1 = "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"AUROC = "
            f"{metrics['auroc']:.4f}"
        )

    # --------------------------------------------------------
    # Ranking for investigation only
    # --------------------------------------------------------

    ordered = sorted(
        results.items(),
        key=lambda item:
            item[1]["validation"]["macro_f1"],
        reverse=True,
    )

    print()
    print("=" * 72)
    print("SWEEP SUMMARY")
    print("=" * 72)

    print(
        f"{'Fusion':24s} "
        f"{'Macro F1':>10s} "
        f"{'AUROC':>10s} "
        f"{'Epoch':>8s}"
    )

    print("-" * 72)

    for name, result in ordered:

        metrics = result[
            "validation"
        ]

        print(
            f"{name:24s} "
            f"{metrics['macro_f1']:10.4f} "
            f"{metrics['auroc']:10.4f} "
            f"{result['best_epoch']:8d}"
        )

    # --------------------------------------------------------
    # Compare with diagnostic baseline
    # --------------------------------------------------------

    baseline = 0.8271

    print()
    print("=" * 72)
    print("IMPROVEMENT OVER 0.8271 LINEAR-PROBE REFERENCE")
    print("=" * 72)

    for name, result in ordered:

        f1 = result[
            "validation"
        ]["macro_f1"]

        print(
            f"{name:24s} "
            f"{f1:.4f} "
            f"Δ {f1 - baseline:+.4f}"
        )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"Results saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()