import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.baseline.final_linear_v3 import config, train

ROOT = Path("results/audit/text_only_loss_decomposition")
BASE = Path("results/baseline/final_linear_v3/text_only")
LABELS = list(config.LABEL_NAMES)

ROOT.mkdir(parents=True, exist_ok=True)

# Use the original six-label text-only model.
config.ACTIVE_SCENARIO = "text_only"
config.ACTIVE_MODALITIES = config.SCENARIOS["text_only"]

train.set_seed(config.SEED)
device = train.get_device()

# Find the best checkpoint recorded in the original results.
result_file = BASE / "final_results.json"
if not result_file.exists():
    raise FileNotFoundError(
        f"Original text_only results not found: {result_file}"
    )

results = json.loads(result_file.read_text(encoding="utf-8"))
best_epoch = int(results["best_epoch"])

checkpoint = BASE / "checkpoints" / f"epoch_{best_epoch:03d}.pt"

if not checkpoint.exists():
    for item in results.get("top_k_checkpoints", []):
        if int(item["epoch"]) == best_epoch:
            candidate = Path(item["path"])
            if candidate.exists():
                checkpoint = candidate
                break

if not checkpoint.exists():
    raise FileNotFoundError(
        f"Best checkpoint for epoch {best_epoch} not found. "
        f"Expected: {checkpoint}"
    )

print("=" * 70)
print("SIX-LABEL TEXT-ONLY LOSS DECOMPOSITION")
print("=" * 70)
print("Checkpoint:", checkpoint)
print("Best epoch:", best_epoch)
print("Device:", device)

train_ds, val_ds, _ = train.build_datasets()

model = train.build_model().to(device)
train.load_checkpoint(model, checkpoint, device)
model.eval()

train_loader = train.make_loader(train_ds, shuffle=False)
val_loader = train.make_loader(val_ds, shuffle=False)

# Calculate weights exclusively from the training split.
training_labels = train.get_training_labels(train_ds)
pos_weights = train.compute_pos_weights(training_labels)

weight_tensor = torch.as_tensor(
    pos_weights, dtype=torch.float32
)

def analyze_split(split_name, loader):
    logits, labels = train.predict(
        model=model,
        loader=loader,
        device=device,
        use_tta=False,
        description=f"[{split_name} eval]",
    )

    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.float32)

    # Exactly the same weighted BCE formulation as the baseline,
    # but without reduction, so every sample/label can be inspected.
    loss_matrix = F.binary_cross_entropy_with_logits(
        logits_t,
        labels_t,
        pos_weight=weight_tensor,
        reduction="none",
    ).numpy()

    probabilities = torch.sigmoid(logits_t).numpy()
    predictions = (probabilities >= 0.5).astype(int)
    actual = labels.astype(int)

    sample_losses = loss_matrix.mean(axis=1)
    label_losses = loss_matrix.mean(axis=0)

    metrics = train.compute_metrics(
        logits, labels,
        threshold=0.5,
        label_names=LABELS,
    )

    # Contribution of the highest-loss samples.
    sorted_losses = np.sort(sample_losses)[::-1]
    total_loss = float(sorted_losses.sum())

    concentration = {}
    for percent in (1, 5, 10, 20):
        n = max(1, int(np.ceil(len(sorted_losses) * percent / 100)))
        concentration[f"top_{percent}_percent"] = float(
            sorted_losses[:n].sum() / max(total_loss, 1e-12)
        )

    label_rows = []
    for j, name in enumerate(LABELS):
        y = actual[:, j]
        pred = predictions[:, j]
        prob = probabilities[:, j]

        fp = (y == 0) & (pred == 1)
        fn = (y == 1) & (pred == 0)

        confident_fp = fp & (prob >= 0.90)
        confident_fn = fn & (prob <= 0.10)

        label_rows.append({
            "split": split_name,
            "label": name,
            "positive_count": int(y.sum()),
            "negative_count": int((1 - y).sum()),
            "pos_weight": float(pos_weights[j]),
            "weighted_bce": float(label_losses[j]),
            "loss_share": float(
                loss_matrix[:, j].sum() /
                max(loss_matrix.sum(), 1e-12)
            ),
            "f1": float(metrics["per_label"][name]["f1"]),
            "false_positives": int(fp.sum()),
            "false_negatives": int(fn.sum()),
            "confident_fp_p_ge_090": int(confident_fp.sum()),
            "confident_fn_p_le_010": int(confident_fn.sum()),
            "confident_error_loss_share": float(
                loss_matrix[confident_fp | confident_fn, j].sum()
                / max(loss_matrix[:, j].sum(), 1e-12)
            ),
        })

    sample_rows = []
    for i in range(len(labels)):
        row = {
            "split": split_name,
            "sample_index": i,
            "mean_weighted_bce": float(sample_losses[i]),
            "incorrect_labels": int(
                (predictions[i] != actual[i]).sum()
            ),
        }

        for j, name in enumerate(LABELS):
            short = name.replace("label_", "")
            row[f"{short}_true"] = int(actual[i, j])
            row[f"{short}_prob"] = float(probabilities[i, j])
            row[f"{short}_loss"] = float(loss_matrix[i, j])

        sample_rows.append(row)

    pd.DataFrame(sample_rows).sort_values(
        "mean_weighted_bce", ascending=False
    ).to_csv(ROOT / f"{split_name}_sample_losses.csv", index=False)

    return {
        "split": split_name,
        "samples": int(len(labels)),
        "weighted_bce": float(loss_matrix.mean()),
        "macro_f1": float(metrics["macro_f1"]),
        "micro_f1": float(metrics["micro_f1"]),
        "auroc": float(metrics["auroc"]),
        "loss_concentration": concentration,
    }, label_rows


train_summary, train_rows = analyze_split("train", train_loader)
val_summary, val_rows = analyze_split("validation", val_loader)

summary = {
    "checkpoint": str(checkpoint),
    "best_epoch": best_epoch,
    "train": train_summary,
    "validation": val_summary,
    "validation_minus_train_loss": (
        val_summary["weighted_bce"] - train_summary["weighted_bce"]
    ),
}

(ROOT / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

df = pd.DataFrame(train_rows + val_rows)
df.to_csv(ROOT / "per_label_loss.csv", index=False)

comparison = df.pivot(
    index="label", columns="split", values="weighted_bce"
)
comparison["val_minus_train"] = (
    comparison["validation"] - comparison["train"]
)
comparison = comparison.sort_values(
    "val_minus_train", ascending=False
)
comparison.to_csv(ROOT / "per_label_comparison.csv")

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

for name, data in [
    ("TRAIN", train_summary),
    ("VALIDATION", val_summary),
]:
    print(
        f"{name}: "
        f"Loss={data['weighted_bce']:.5f} | "
        f"Macro-F1={data['macro_f1']:.5f} | "
        f"AUROC={data['auroc']:.5f}"
    )

print(
    "Validation - Train Loss:",
    round(summary["validation_minus_train_loss"], 5)
)

print("\nLOSS CONCENTRATION")
for key, value in val_summary["loss_concentration"].items():
    print(f"{key}: {value:.2%} of total validation loss")

print("\nPER-LABEL LOSS COMPARISON")
print(comparison.round(5).to_string())

print("\nVALIDATION ERROR CONFIDENCE")
val_df = df[df["split"] == "validation"]
print(val_df[[
    "label",
    "false_positives",
    "false_negatives",
    "confident_fp_p_ge_090",
    "confident_fn_p_le_010",
    "confident_error_loss_share",
]].round(4).to_string(index=False))

print("\nSaved to:", ROOT.resolve())
print("No training performed. No original files modified.")
