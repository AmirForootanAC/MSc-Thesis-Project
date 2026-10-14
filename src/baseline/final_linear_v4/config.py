"""Configuration for Final Linear V4 supervised baselines."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    PROJECT_ROOT
    / "results"
    / "six_label_patient_level_dataset"
    / "labeled_dataset.csv"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "COde-Dataset"
    / "Images"
)


SCENARIOS = {
    "text_only": ("text",),
    "image_only": ("photograph",),
    "xray_only": ("radiograph",),
    "image_text": ("photograph", "text"),
    "image_xray": ("photograph", "radiograph"),
    "text_xray": ("text", "radiograph"),
    "full_multimodal": ("photograph", "radiograph", "text"),
}


ACTIVE_SCENARIO = "full_multimodal"
ACTIVE_MODALITIES = SCENARIOS[ACTIVE_SCENARIO]


RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "final_linear_v4"
)


EXPECTED_COUNTS = {
    "train": 2935,
    "validation": 627,
    "test": 633,
}


LABEL_NAMES = [
    "label_caries",
    "label_gingivitis",
    "label_malocclusion",
    "label_pulpitis",
    "label_tooth_loss",
    "label_tooth_structure_loss",
]

NUM_LABELS = len(LABEL_NAMES)


TEXT_COLUMNS = [
    "chief_complaint",
    "present_illness",
    "past_medical_record",
    "examination",
]

TEXT_MODEL_NAME = "distilbert-base-uncased"
TEXT_MAX_LENGTH = 256


IMAGE_SIZE = 256

PRETRAINED_IMAGE_ENCODER = True
FREEZE_IMAGE_ENCODER = True
# نقطه تعادل دقیق بین تست اول (4) و تست دوم (2): فریز 3 لایه از 6 لایه
FREEZE_TEXT_LAYERS = 3
TEXT_INTERNAL_DROPOUT = 0.20


BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1

NUM_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 7


BACKBONE_LR = 1.2e-5
IMAGE_HEAD_LR = 6e-5
CLASSIFIER_LR = 8e-5
TEXT_LR = 1e-5

WEIGHT_DECAY = 1e-2


WARMUP_EPOCHS = 2

DROPOUT = 0.45
MODALITY_DROPOUT = 0.0
LABEL_SMOOTHING = 0.05
MAX_POS_WEIGHT = 6.0

GRADIENT_CLIP_NORM = 1.0

TOP_K_CHECKPOINTS = 3

USE_TTA = True


DEFAULT_THRESHOLD = 0.5

THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.70
THRESHOLD_STEP = 0.05


SEED = 42

DEVICE = "cuda"

NUM_WORKERS = 0
PIN_MEMORY = True