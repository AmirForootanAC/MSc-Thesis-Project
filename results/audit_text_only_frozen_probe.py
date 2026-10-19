import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.baseline.final_linear_v3 import config, train
from src.baseline.final_linear_v3.dataset import CompleteCaseDataset

ROOT = Path("results/audit/text_only_frozen_probe")
if ROOT.exists():
    raise FileExistsError(
        f"{ROOT} already exists. Existing audit output was not changed."
    )
ROOT.mkdir(parents=True)

SEED = 42
PROBE_DROPOUT = 0.30
PROBE_LR = 1e-3
PROBE_WEIGHT_DECAY = 1e-2
PROBE_EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

config.ACTIVE_SCENARIO = "text_only"
config.ACTIVE_MODALITIES = config.SCENARIOS["text_only"]

device = train.get_device()
print(f"Device: {device}")
print("Experiment: frozen DistilBERT + dropout + linear classifier")
print("Evaluation: train and validation only; test is not evaluated.")

train_ds = CompleteCaseDataset(
    csv_path=config.DATASET_PATH,
    split="train",
    image_root=config.IMAGE_ROOT,
    modalities=("text",),
    tokenizer_name=config.TEXT_MODEL_NAME,
    max_length=config.TEXT_MAX_LENGTH,
)
val_ds = CompleteCaseDataset(
    csv_path=config.DATASET_PATH,
    split="validation",
    image_root=config.IMAGE_ROOT,
    modalities=("text",),
    tokenizer_name=config.TEXT_MODEL_NAME,
    max_length=config.TEXT_MAX_LENGTH,
)

train_loader = train.make_loader(train_ds, shuffle=False)
val_loader = train.make_loader(val_ds, shuffle=False)

model = train.build_model().to(device)
encoder = model.text_encoder
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

@torch.no_grad()
def extract_features(loader, name):
    features, labels = [], []
    for raw in loader:
        batch = train.move_batch_to_device(raw, device)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=train.amp_enabled(device),
        ):
            output = encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        features.append(
            output.last_hidden_state[:, 0].float().cpu()
        )
        labels.append(batch["labels"].float().cpu())
    x = torch.cat(features)
    y = torch.cat(labels)
    print(f"{name}: features={tuple(x.shape)}, labels={tuple(y.shape)}")
    return x, y

x_train, y_train = extract_features(train_loader, "Train")
x_val, y_val = extract_features(val_loader, "Validation")

del model, encoder
if torch.cuda.is_available():
    torch.cuda.empty_cache()

pos_weights = torch.tensor(
    train.compute_pos_weights(
        train.get_training_labels(train_ds)
    ),
    dtype=torch.float32,
    device=device,
)

head = nn.Linear(768, config.NUM_LABELS).to(device)
optimizer = torch.optim.AdamW(
    head.parameters(),
    lr=PROBE_LR,
    weight_decay=PROBE_WEIGHT_DECAY,
)

probe_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)

def evaluate(x, y):
    head.eval()
    with torch.no_grad():
        logits = head(x.to(device))
        targets = y.to(device)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weights
        ).item()
        logits_np = logits.float().cpu().numpy()
        labels_np = y.numpy()
    metrics = train.compute_metrics(
        logits_np,
        labels_np,
        threshold=0.5,
        label_names=config.LABEL_NAMES,
    )
    return loss, metrics

history = []
best_score = -1.0
best_epoch = None
bad_epochs = 0

for epoch in range(1, PROBE_EPOCHS + 1):
    head.train()
    for xb, yb in probe_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = head(F.dropout(xb, p=PROBE_DROPOUT, training=True))
        loss = F.binary_cross_entropy_with_logits(
            logits, yb, pos_weight=pos_weights
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

    # Both losses are measured after the epoch using the same fixed head.
    train_loss, train_metrics = evaluate(x_train, y_train)
    val_loss, val_metrics = evaluate(x_val, y_val)

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "loss_gap": val_loss - train_loss,
        "train_macro_f1": train_metrics["macro_f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_auroc": val_metrics["auroc"],
    }
    history.append(row)
    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"gap={val_loss-train_loss:+.4f} | "
        f"train_F1={train_metrics['macro_f1']:.4f} | "
        f"val_F1={val_metrics['macro_f1']:.4f} | "
        f"val_AUROC={val_metrics['auroc']:.4f}"
    )

    if val_metrics["macro_f1"] > best_score:
        best_score = val_metrics["macro_f1"]
        best_epoch = epoch
        bad_epochs = 0
        torch.save(
            {
                "epoch": epoch,
                "head_state_dict": head.state_dict(),
                "val_macro_f1": best_score,
                "probe_dropout": PROBE_DROPOUT,
                "seed": SEED,
            },
            ROOT / "best_probe.pt",
        )
    else:
        bad_epochs += 1

    (ROOT / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    if bad_epochs >= PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

summary = {
    "experiment": "frozen_distilbert_dropout_linear_probe",
    "train_samples": len(train_ds),
    "validation_samples": len(val_ds),
    "test_evaluated": False,
    "dropout": PROBE_DROPOUT,
    "learning_rate": PROBE_LR,
    "weight_decay": PROBE_WEIGHT_DECAY,
    "best_epoch": best_epoch,
    "best_val_macro_f1": best_score,
    "history_file": str(ROOT / "history.json"),
    "checkpoint_file": str(ROOT / "best_probe.pt"),
}
(ROOT / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print("\nSummary:")
print(json.dumps(summary, indent=2))
print(f"\nAll probe outputs saved to: {ROOT}")
