import math
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from . import config
from .dataset import UnlabeledRadiographDataset
from .model import build_dino_models
from .loss import DINOLoss


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0):
    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = torch.linspace(0, base_value, warmup_iters) if warmup_epochs > 0 else torch.tensor([])
    iters = torch.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + torch.cos(math.pi * iters / len(iters)))
    return torch.cat((warmup_schedule, schedule)).numpy()


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.DEVICE)
    print(f"🚀 Starting Pure DINO Pretraining on Radiographs | Device: {device}")

    dataset = UnlabeledRadiographDataset(split="train", enhance=True)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )

    student, teacher = build_dino_models(arch=config.BACKBONE, out_dim=config.OUT_DIM)
    student, teacher = student.to(device), teacher.to(device)

    dino_loss = DINOLoss(
        out_dim=config.OUT_DIM,
        ncrops=config.LOCAL_CROPS_NUMBER + 2,
        warmup_teacher_temp=config.WARMUP_TEACHER_TEMP,
        teacher_temp=config.TEACHER_TEMP,
        warmup_teacher_temp_epochs=config.WARMUP_TEACHER_TEMP_EPOCHS,
        nepochs=config.EPOCHS,
        student_temp=config.STUDENT_TEMP,
        center_momentum=config.CENTER_MOMENTUM,
    ).to(device)

    optimizer = AdamW(student.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    # زمان‌بندی‌های Cosine برای نرخ یادگیری، Weight Decay و Momentum
    niter_per_ep = len(loader)
    lr_schedule = cosine_scheduler(config.LR, config.MIN_LR, config.EPOCHS, niter_per_ep, config.WARMUP_EPOCHS)
    wd_schedule = cosine_scheduler(config.WEIGHT_DECAY, config.WEIGHT_DECAY_END, config.EPOCHS, niter_per_ep)
    momentum_schedule = cosine_scheduler(config.EMA_MOMENTUM_BASE, 1.0, config.EPOCHS, niter_per_ep)

    best_loss = float("inf")

    for epoch in range(config.EPOCHS):
        student.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for step, crops in enumerate(pbar):
            it = epoch * niter_per_ep + step
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[it]

            crops = [c.to(device, non_blocking=True) for c in crops]

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                # Teacher فقط ۲ کراپ سراسری (Global) اول را می‌بیند
                teacher_output = teacher(crops[:2])
                # Student تمام ۶ کراپ (۲ سراسری + ۴ محلی) را می‌بیند
                student_output = student(crops)
                loss = dino_loss(student_output, teacher_output, epoch) / config.ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % config.ACCUMULATION_STEPS == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # آپدیت وزن‌های Teacher به روش EMA از روی Student
                with torch.no_grad():
                    m = momentum_schedule[it]
                    for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            step_loss = loss.item() * config.ACCUMULATION_STEPS
            total_loss += step_loss
            pbar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{lr_schedule[it]:.2e}")

        avg_loss = total_loss / len(loader)
        print(f"👉 Epoch {epoch+1:02d} | Avg DINO Loss: {avg_loss:.4f}")

        # ذخیره چک‌پوینت‌ها (در DINO معمولاً بک‌بونِ Teacher قوی‌ترین ویژگی‌ها را دارد)
        ckpt = {
            "epoch": epoch + 1,
            "arch": config.BACKBONE,
            "teacher_backbone": teacher.backbone.state_dict(),
            "student_backbone": student.backbone.state_dict(),
            "loss": avg_loss,
        }
        torch.save(ckpt, config.OUTPUT_DIR / "latest_dino_xray.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, config.OUTPUT_DIR / "best_dino_xray.pt")

if __name__ == "__main__":
    main()