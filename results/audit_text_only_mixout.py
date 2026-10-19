import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.final_linear_v3 import config, train
from src.baseline.final_linear_v3.dataset import CompleteCaseDataset


# Completely isolated experiment: never overwrite previous results.
OUT = PROJECT_ROOT / "results/audit/text_only_mixout_p90"
if OUT.exists():
    raise FileExistsError(
        f"{OUT} already exists. Nothing was overwritten."
    )
OUT.mkdir(parents=True)

SEED = 42
MIXOUT_P = 0.90
MAX_EPOCHS = 15
PATIENCE = 4

config.ACTIVE_SCENARIO = "text_only"
config.ACTIVE_MODALITIES = config.SCENARIOS["text_only"]

train.set_seed(SEED)
device = train.get_device()

print(f"Device: {device}")
print("Experiment: trainable DistilBERT with Mixout")
print(f"Mixout probability: {MIXOUT_P}")
print(f"Max epochs: {MAX_EPOCHS}; patience: {PATIENCE}")
print("Evaluation: train + validation only; test is NOT loaded.")

train_ds = CompleteCaseDataset(
    csv_path=config.DATASET_PATH,
    split="train",
    image_root=config.IMAGE_ROOT,
    modalities=("text",),
    tokenizer_name=config.TEXT_MODEL_NAME,
    max_length=config.TEXT_MAX_LENGTH,
    photograph_transform=None,
    radiograph_transform=None,
)
val_ds = CompleteCaseDataset(
    csv_path=config.DATASET_PATH,
    split="validation",
    image_root=config.IMAGE_ROOT,
    modalities=("text",),
    tokenizer_name=config.TEXT_MODEL_NAME,
    max_length=config.TEXT_MAX_LENGTH,
    photograph_transform=None,
    radiograph_transform=None,
)

if len(train_ds) != 2935 or len(val_ds) != 627:
    raise RuntimeError(
        f"Unexpected split sizes: train={len(train_ds)}, "
        f"validation={len(val_ds)}"
    )

train_loader = train.make_loader(train_ds, shuffle=True)
train_eval_loader = train.make_loader(train_ds, shuffle=False)
val_loader = train.make_loader(val_ds, shuffle=False)


class MixoutLinear(nn.Linear):
    """Linear layer regularized toward its pretrained parameters."""

    def __init__(self, source, probability):
        super().__init__(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
        )
        self.probability = probability

        with torch.no_grad():
            self.weight.copy_(source.weight)
            if source.bias is not None:
                self.bias.copy_(source.bias)

        self.register_buffer(
            "target_weight", source.weight.detach().clone()
        )
        if source.bias is not None:
            self.register_buffer(
                "target_bias", source.bias.detach().clone()
            )
        else:
            self.register_buffer("target_bias", None)

    def forward(self, inputs):
        if not self.training or self.probability == 0:
            return F.linear(inputs, self.weight, self.bias)

        keep_probability = 1.0 - self.probability
        mask_w = torch.empty_like(self.weight).bernoulli_(
            keep_probability
        )
        mixed_weight = self.target_weight + (
            mask_w * (self.weight - self.target_weight)
            / keep_probability
        )

        mixed_bias = self.bias
        if self.bias is not None:
            mask_b = torch.empty_like(self.bias).bernoulli_(
                keep_probability
            )
            mixed_bias = self.target_bias + (
                mask_b * (self.bias - self.target_bias)
                / keep_probability
            )

        return F.linear(inputs, mixed_weight, mixed_bias)


def apply_mixout(module, probability):
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                MixoutLinear(child, probability),
            )
            replaced += 1
        else:
            replaced += apply_mixout(child, probability)
    return replaced


model = train.build_model()
replaced = apply_mixout(model.text_encoder, MIXOUT_P)
model = model.to(device)

print(f"Replaced pretrained Linear layers: {replaced}")
print(f"Trainable parameters: {train.parameter_counts(model)['trainable']:,}")

criterion = train.build_criterion(train_ds, device)
optimizer = train.build_optimizer(model)
scaler = train.create_grad_scaler(device)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    count = 0
    logits_all = []
    labels_all = []

    for raw_batch in loader:
        batch = train.move_batch_to_device(raw_batch, device)
        labels = batch["labels"]

        with torch.amp.autocast(
            device_type=device.type,
            enabled=train.amp_enabled(device),
        ):
            logits = train.model_forward(model, batch)
            loss = criterion(logits, labels)

        n = labels.shape[0]
        total_loss += loss.float().item() * n
        count += n
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(labels.cpu().numpy())

    logits_np = np.concatenate(logits_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)

    metrics = train.compute_metrics(
        logits_np,
        labels_np,
        threshold=0.5,
        label_names=config.LABEL_NAMES,
    )
    return total_loss / count, metrics


history = []
best_f1 = -1.0
best_epoch = 0
bad_epochs = 0

for epoch in range(1, MAX_EPOCHS + 1):
    # The regular training-mode loss is intentionally not used for
    # train-vs-validation comparison because dropout/Mixout is active.
    train.train_one_epoch(
        model, train_loader, criterion, optimizer,
        scaler, device, epoch
    )

    # Both losses are measured in eval mode with identical settings.
    train_loss, train_metrics = evaluate(train_eval_loader)
    val_loss, val_metrics = evaluate(val_loader)

    record = {
        "epoch": epoch,
        "train_loss_eval_mode": float(train_loss),
        "val_loss_eval_mode": float(val_loss),
        "loss_gap": float(val_loss - train_loss),
        "train_macro_f1": float(train_metrics["macro_f1"]),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "val_micro_f1": float(val_metrics["micro_f1"]),
        "val_auroc": float(val_metrics["auroc"]),
        "val_per_label": val_metrics.get("per_label", {}),
    }
    history.append(record)

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss_eval={train_loss:.4f} | "
        f"val_loss_eval={val_loss:.4f} | "
        f"gap={val_loss - train_loss:+.4f} | "
        f"train_F1={train_metrics['macro_f1']:.4f} | "
        f"val_F1={val_metrics['macro_f1']:.4f} | "
        f"val_AUROC={val_metrics['auroc']:.4f}",
        flush=True,
    )

    with open(OUT / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if val_metrics["macro_f1"] > best_f1:
        best_f1 = float(val_metrics["macro_f1"])
        best_epoch = epoch
        bad_epochs = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": best_f1,
                "mixout_probability": MIXOUT_P,
                "seed": SEED,
            },
            OUT / "best.pt",
        )
    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

summary = {
    "experiment": "text_only_distilbert_mixout",
    "mixout_probability": MIXOUT_P,
    "seed": SEED,
    "train_samples": len(train_ds),
    "validation_samples": len(val_ds),
    "test_evaluated": False,
    "mixout_linear_layers": replaced,
    "best_epoch": best_epoch,
    "best_val_macro_f1": best_f1,
    "output_dir": str(OUT.relative_to(PROJECT_ROOT)),
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSUMMARY")
print(json.dumps(summary, indent=2))
print(f"\nAll outputs saved to: {OUT}")
