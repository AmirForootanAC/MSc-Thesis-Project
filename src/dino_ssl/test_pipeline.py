import torch
from src.dino_ssl.dataset import UnlabeledRadiographDataset, SingleImageLabeledRadiographDataset
from src.dino_ssl.model import build_dino_models
from src.dino_ssl.evaluate_xray import SingleXrayClassifier
from src.dino_ssl import config

def main():
    print("="*50)
    print("🧪 STARTING DINO PIPELINE SANITY CHECK")
    print("="*50)

    try:
        # ۱. تست دیتاست پیش‌آموزش DINO و کراپ‌ها
        print("\n[1/4] Testing Unlabeled DINO Dataset & Transforms...")
        ssl_ds = UnlabeledRadiographDataset(split="validation", enhance=True)
        crops = ssl_ds[0]
        print(f"✅ Success! Generated {len(crops)} crops.")
        print(f"   Shapes: {[list(c.shape) for c in crops]}")

        # ۲. تست عبور دیتا از شبکه‌های Teacher و Student
        print("\n[2/4] Testing DINO Models (Forward Pass)...")
        student, teacher = build_dino_models(arch=config.BACKBONE, out_dim=config.OUT_DIM)
        # اضافه کردن بُعد Batch (شبیه‌سازی بچ‌سایز ۱)
        batch_crops = [c.unsqueeze(0) for c in crops]
        stu_out = student(batch_crops)
        print(f"✅ Success! Student output shape: {list(stu_out.shape)} (Expected: [6, {config.OUT_DIM}])")

        # ۳. تست دیتاست طبقه‌بندی تک‌تصویری
        print("\n[3/4] Testing Single-Image Labeled Dataset...")
        eval_ds = SingleImageLabeledRadiographDataset(split="validation", is_train=True, enhance=True)
        img, lbl = eval_ds[0]
        print(f"✅ Success! Image shape: {list(img.shape)}, Label shape: {list(lbl.shape)}")

        # ۴. تست شبکه نهایی کلاسیفایر
        print("\n[4/4] Testing X-Ray Classifier (Forward Pass)...")
        clf = SingleXrayClassifier(arch=config.BACKBONE, num_classes=len(config.LABEL_NAMES))
        logits = clf(img.unsqueeze(0))
        print(f"✅ Success! Classifier logits shape: {list(logits.shape)} (Expected: [1, {len(config.LABEL_NAMES)}])")

        print("\n" + "🎉 "*10)
        print("ALL TESTS PASSED SUCCESSFULLY! PIPELINE IS READY.")
        print("🎉 "*10 + "\n")

    except Exception as e:
        print("\n❌ ERROR DETECTED:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()