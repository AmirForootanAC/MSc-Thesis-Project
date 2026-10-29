import ast
from pathlib import Path
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from . import config
from .transforms import DINORadiographTransforms


def parse_filenames(val) -> list[str]:
    if pd.isna(val):
        return []
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ["nan", "none", "[]"]:
        return []
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            parsed = ast.literal_eval(val_str)
            return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    for sep in [";", ",", "|"]:
        if sep in val_str:
            return [x.strip() for x in val_str.split(sep) if x.strip()]
    return [val_str]


def enhance_xray_pil(pil_img: Image.Image) -> Image.Image:
    """افزایش کنتراست رادیوگرافی با PIL خالص (بدون نیاز به cv2)"""
    gray = pil_img.convert("L")
    enhanced = ImageOps.autocontrast(gray, cutoff=1)
    return enhanced.convert("RGB")


def resolve_image_path(fname: str) -> Path:
    candidates = [
        config.IMAGE_ROOT / fname,
        config.IMAGE_ROOT / "radiographs" / fname,
        config.IMAGE_ROOT / "radiograph" / fname,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


class UnlabeledRadiographDataset(Dataset):
    """
    دیتاست مرحله اول (SSL Pretraining با DINO):
    تک‌تک تصاویر X-ray به صورت کاملاً مستقل و جداگانه در آموزش شرکت می‌کنند.
    """
    def __init__(self, split="train", enhance=True):
        df = pd.read_csv(config.DATASET_PATH)
        if split != "all":
            df = df[df["split"] == split].copy()

        self.enhance = enhance
        self.transform = DINORadiographTransforms(
            global_crops_size=config.GLOBAL_CROPS_SIZE,
            local_crops_size=config.LOCAL_CROPS_SIZE,
            local_crops_number=config.LOCAL_CROPS_NUMBER,
        )

        # استخراج تک‌تک تصاویر به عنوان نمونه‌های مستقل
        all_files = []
        for val in df["radiographs"]:
            all_files.extend(parse_filenames(val))
        
        self.file_list = list(dict.fromkeys(all_files))
        print(f"🦷 [DINO SSL] Loaded {len(self.file_list)} individual radiographs for '{split}'.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img_path = resolve_image_path(fname)
        try:
            img = Image.open(img_path).convert("RGB")
            if self.enhance:
                img = enhance_xray_pil(img)
        except Exception:
            img = Image.new("RGB", (config.GLOBAL_CROPS_SIZE, config.GLOBAL_CROPS_SIZE), (128, 128, 128))

        return self.transform(img)


class SingleImageLabeledRadiographDataset(Dataset):
    """
    دیتاست مرحله دوم (Supervised Fine-Tuning & Evaluation):
    بدون هیچ‌گونه تجمیع ویزیت! هر تصویر به صورت یک ردیف مستقل با لیبل مربوطه در آموزش شرکت می‌کند.
    """
    def __init__(self, split="train", is_train=True, enhance=True):
        df = pd.read_csv(config.DATASET_PATH)
        df = df[df["split"] == split].copy()

        self.enhance = enhance
        self.samples = []

        # باز کردن (Flatten/Explode) تصاویر هر ویزیت به نمونه‌های تک‌تصویری مستقل
        for _, row in df.iterrows():
            fnames = parse_filenames(row["radiographs"])
            if not fnames:
                continue
            labels = row[config.LABEL_NAMES].astype(float).values
            for fn in fnames:
                self.samples.append((fn, labels))

        print(f"🦷 [Single-Image Finetune] '{split}' split: {len(self.samples)} individual labeled radiographs.")

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(config.GLOBAL_CROPS_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config.GLOBAL_CROPS_SIZE, config.GLOBAL_CROPS_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, labels = self.samples[idx]
        img_path = resolve_image_path(fname)
        try:
            img = Image.open(img_path).convert("RGB")
            if self.enhance:
                img = enhance_xray_pil(img)
        except Exception:
            img = Image.new("RGB", (config.GLOBAL_CROPS_SIZE, config.GLOBAL_CROPS_SIZE), (128, 128, 128))

        img_tensor = self.transform(img)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        return img_tensor, label_tensor