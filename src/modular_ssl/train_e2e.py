import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

# ایمپورت از کدهای خودتان
from src.modular_ssl import config
from src.modular_ssl.extract_representations import CompleteCaseExtractionDataset
from src.modular_ssl.models import SingleVisualModalitySSLModel

# ==============================================================================
# ماژول فیوژن برای یادگیری End-to-End
# ==============================================================================
class MultimodalFusionHead(nn.Module):
    def __init__(self, feat_dim=256, num_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, p_feat, r_feat, t_feat):
        combined = torch.cat([p_feat, r_feat, t_feat], dim=1)
        return self.fc(combined)

# ==============================================================================
# تابع اصلی
# ==============================================================================
def main():
    device = torch.device(config.DEVICE)
    print(f"🚀 Starting End-to-End Modular Fine-Tuning on {device}...")

    # لود دیتاست‌ها
    train_ds = CompleteCaseExtractionDataset(split="train")
    val_ds = CompleteCaseExtractionDataset(split="validation")
    
    # بچ‌سایز کوچک برای جلوگیری از OOM گرافیک
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    # لود شبکه‌ها
    photo_ssl = SingleVisualModalitySSLModel("photograph").to(device)
    radio_ssl = SingleVisualModalitySSLModel("radiograph").to(device)
    fusion_head = MultimodalFusionHead(feat_dim=256, num_classes=4).to(device)

    # لود وزن‌های Latest / Best Gap
    photo_ckpt = config.PHOTO_SSL_DIR / "best_photograph_ssl.pt"
    radio_ckpt = config.RADIO_SSL_DIR / "best_radiograph_ssl.pt"
    
    if photo_ckpt.exists():
        photo_ssl.load_state_dict(torch.load(photo_ckpt, map_location=device)["model_state_dict"])
        print("✅ Photograph SSL checkpoint loaded.")
    if radio_ckpt.exists():
        radio_ssl.load_state_dict(torch.load(radio_ckpt, map_location=device)["model_state_dict"])
        print("✅ Radiograph SSL checkpoint loaded.")

    # ⚠️ باز کردن قفل آموزش (Unfreeze) برای تصاویر
    for param in photo_ssl.vision_encoder.parameters():
        param.requires_grad = True
    for param in radio_ssl.vision_encoder.parameters():
        param.requires_grad = True
        
    # فریز کردن متن برای جلوگیری از Text Dominance
    for param in photo_ssl.text_encoder.parameters():
        param.requires_grad = False
    for param in radio_ssl.text_encoder.parameters():
        param.requires_grad = False

    # نرخ یادگیری متفاوت
    optimizer = AdamW([
        {'params': photo_ssl.vision_encoder.parameters(), 'lr': 1e-5},
        {'params': radio_ssl.vision_encoder.parameters(), 'lr': 1e-5},
        {'params': fusion_head.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device='cuda', enabled=config.USE_AMP)
    
    epochs = 15
    accumulation_steps = 4 
    best_val_auc = 0.0

    print("\n" + "="*50)
    print("🔥 BEGIN TRAINING LOOP 🔥")
    print("="*50 + "\n")

    for epoch in range(1, epochs + 1):
        photo_ssl.train()
        radio_ssl.train()
        fusion_head.train()
        
        # لایه‌های Batchnorm متن را روی Eval می‌گذاریم که آپدیت نشوند
        photo_ssl.text_encoder.eval()
        radio_ssl.text_encoder.eval()
        
        train_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for step, batch in enumerate(pbar):
            p_imgs, p_mask = batch["photo_images"].to(device), batch["photo_mask"].to(device)
            r_imgs, r_mask = batch["radio_images"].to(device), batch["radio_mask"].to(device)
            ids, amask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
                p_out = photo_ssl.vision_encoder(p_imgs, p_mask)["visit_embedding"]
                r_out = radio_ssl.vision_encoder(r_imgs, r_mask)["visit_embedding"]
                
                with torch.no_grad():
                    t_out_p = photo_ssl.text_encoder(ids, amask)["visit_embedding"]
                    t_out_r = radio_ssl.text_encoder(ids, amask)["visit_embedding"]
                    t_out = 0.5 * (t_out_p + t_out_r)
                
                logits = fusion_head(p_out, r_out, t_out)
                loss = criterion(logits, labels) / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss += (loss.item() * accumulation_steps)
            pbar.set_postfix(loss=train_loss/(step+1))
            
        # ==================== Validation ====================
        photo_ssl.eval()
        radio_ssl.eval()
        fusion_head.eval()
        
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Valid]"):
                p_imgs, p_mask = batch["photo_images"].to(device), batch["photo_mask"].to(device)
                r_imgs, r_mask = batch["radio_images"].to(device), batch["radio_mask"].to(device)
                ids, amask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
                
                with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
                    p_out = photo_ssl.vision_encoder(p_imgs, p_mask)["visit_embedding"]
                    r_out = radio_ssl.vision_encoder(r_imgs, r_mask)["visit_embedding"]
                    t_out = 0.5 * (photo_ssl.text_encoder(ids, amask)["visit_embedding"] + radio_ssl.text_encoder(ids, amask)["visit_embedding"])
                    
                    logits = fusion_head(p_out, r_out, t_out)
                    probs = torch.sigmoid(logits)
                    
                val_preds.append(probs.cpu().numpy())
                val_labels.append(batch["labels"].numpy())
                
        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        
        val_auc = roc_auc_score(val_labels, val_preds, average='macro')
        val_f1 = f1_score(val_labels, (val_preds > 0.5).astype(int), average='macro')
        
        print(f"👉 Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            out_dir = config.OUTPUT_ROOT / "best_e2e_finetuned"
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(fusion_head.state_dict(), out_dir / "fusion_head.pt")
            torch.save(photo_ssl.vision_encoder.state_dict(), out_dir / "photo_encoder.pt")
            torch.save(radio_ssl.vision_encoder.state_dict(), out_dir / "radio_encoder.pt")
            print(f"🌟 New best End-to-End model saved (AUC: {val_auc:.4f})")

if __name__ == "__main__":
    main()