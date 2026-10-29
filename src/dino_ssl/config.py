from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "results/six_label_patient_level_dataset/labeled_dataset.csv"
IMAGE_ROOT = PROJECT_ROOT / "data/images"  # اگر مسیر عکس‌هایت در سیستم لوکال فرق دارد اینجا اصلاح کن
OUTPUT_DIR = PROJECT_ROOT / "results/dino_radiograph"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
PIN_MEMORY = True

# معماری بک‌بون: می‌توانی بین convnext_tiny و densenet121 تغییر دهی
BACKBONE = "convnext_tiny"
OUT_DIM = 8192              # بعد خروجی DINO Head (برای رادیوگرافی 8192 عالی است و از اورفیت جلوگیری می‌کند)

# تنظیمات Multi-Crop
GLOBAL_CROPS_SIZE = 224
LOCAL_CROPS_SIZE = 96
LOCAL_CROPS_NUMBER = 4      # ۲ کراپ سراسری + ۴ کراپ محلی = ۶ تصویر از هر رادیوگرافی

# تنظیمات آموزش DINO
BATCH_SIZE = 16
ACCUMULATION_STEPS = 2      # بچ‌سایز موثر = ۳۲
EPOCHS = 50
WARMUP_EPOCHS = 5
LR = 5e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 0.04
WEIGHT_DECAY_END = 0.4

# دماها (Temperatures) - قلب پایداری DINO
WARMUP_TEACHER_TEMP = 0.04
TEACHER_TEMP = 0.07
WARMUP_TEACHER_TEMP_EPOCHS = 10
STUDENT_TEMP = 0.1
CENTER_MOMENTUM = 0.9
EMA_MOMENTUM_BASE = 0.996   # مومنتوم آپدیت شبکه Teacher از روی Student

LABEL_NAMES = [
    "label_caries",
    "label_gingivitis",
    "label_malocclusion",
    "label_pulpitis",
]