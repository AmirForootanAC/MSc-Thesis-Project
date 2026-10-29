import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, f1_score
from tqdm.auto import tqdm

from . import config
from .dataset import SingleImageLabeledRadiographDataset
from .model import BackboneWrapper


class SingleXrayClassifier(nn.Module):
    """مدل طبقه‌بندی مستقیم روی تک‌تصویر رادیوگرافی"""
    def __init__(self, arch="convnext_tiny", num_classes=4):
        super().__init__()
        self.backbone = BackboneWrapper(arch=arch, pretrained=True)
        in_dim = self.backbone.out_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(0.3),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


def compute_optimal_f1(y_true: np.ndarray, y_probs: np.ndarray) -> tuple[float, float]:
    """محاسبه F1 هم با آستانه ۰.۵ و هم با بهترین آستانه اختصاصی هر کلاس (Per-Class Threshold Optimization)"""
    f1_at_05 = f1_score(y_true, (y_probs >= 0.5).astype(int), average="macro", zero_division=0)
    
    opt_preds = np.zeros_like(y_probs, dtype=int)
    for c in range(y_true.shape[1]):
        best_c_f1 = -1.0
        best_th = 0.5
        for th in np.arange(0.10, 0.90, 0.05):
            score = f1_score(y_true[:, c], (y_probs[:, c] >= th).astype(int), zero_division=0)
            if score > best_c_f1:
                best_c_f1 = score
                best_th = th
        opt_preds[:, c] = (y_probs[:, c] >= best_th).astype(int)
        
    opt_macro_f1 = f1_score(y_true, opt_preds, average="macro", zero_division=0)
    return f1_at_05, opt_macro_f1


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(imgs)
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    auc = roc_auc_score(all_labels, all_probs, average="macro")
    f1_05, opt_f1 = compute_optimal_f1(all_labels, all_probs)
    return auc, f1_05, opt_f1


def main():
    device = torch.device(config.DEVICE)
    print(f"🚀 Starting Single-Image X-Ray Fine-Tuning & Evaluation on {device}...")

    train_ds = SingleImageLabeledRadiographDataset(split="train", is_train=True)
    val_ds = SingleImageLabeledRadiographDataset(split="validation", is_train=False)
    test_ds = SingleImageLabeledRadiographDataset(split="test", is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    model = SingleXrayClassifier(arch=config.BACKBONE, num_classes=len(config.LABEL_NAMES)).to(device)

    # لود کردن وزن‌های پیش‌آموزش‌دیده با DINO
    dino_ckpt_path = config.OUTPUT_DIR / "best_dino_xray.pt"
    if dino_ckpt_path.exists():
        ckpt = torch.load(dino_ckpt_path, map_location=device, weights_only=False)
        model.backbone.load_state_dict(ckpt["teacher_backbone"])
        print(f"✅ Loaded DINO Teacher Backbone from {dino_ckpt_path}")
    else:
        print("⚠️ DINO checkpoint not found! Training from ImageNet initialization.")

    # محاسبه وزن کلاس‌های نامتوازن (Pos Weight) برای جلوگیری از پیش‌بینی صفر و بالا بردن F1
    train_labels = np.array([s[1] for s in train_ds.samples])
    pos_counts = train_labels.sum(axis=0) + 1e-5
    neg_counts = len(train_labels) - pos_counts + 1e-5
    pos_weight = torch.tensor(np.clip(neg_counts / pos_counts, 1.0, 10.0), dtype=torch.float32, device=device)
    print(f"⚖️ Class Pos-Weights: {pos_weight.cpu().numpy().round(2)}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 5e-4},
    ], weight_decay=1e-2)

    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))
    best_val_opt_f1 = 0.0
    epochs = 20

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [Train]", leave=False):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        val_auc, val_f1_05, val_opt_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | F1(@0.5): {val_f1_05:.4f} | Opt F1: {val_opt_f1:.4f}")

        if val_opt_f1 > best_val_opt_f1:
            best_val_opt_f1 = val_opt_f1
            torch.save(model.state_dict(), config.OUTPUT_DIR / "best_finetuned_xray_classifier.pt")

    # ارزیابی نهایی روی داده Test
    print("\n" + "="*60)
    model.load_state_dict(torch.load(config.OUTPUT_DIR / "best_finetuned_xray_classifier.pt", map_location=device, weights_only=False))
    test_auc, test_f1_05, test_opt_f1 = evaluate(model, test_loader, device)
    print(f"🏆 FINAL SINGLE-IMAGE X-RAY TEST RESULTS:")
    print(f"   Test AUROC  : {test_auc:.4f}")
    print(f"   Test F1@0.5 : {test_f1_05:.4f}")
    print(f"   Test Opt F1 : {test_opt_f1:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()