"""Configuration for strong complete-case supervised baselines."""

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
    "text_only": ("text",),#DONE
    "image_only": ("photograph",),
    "xray_only": ("radiograph",),
    "image_text": ("photograph", "text"),#DONE
    "image_xray": ("photograph", "radiograph"),#DONE
    "text_xray": ("text", "radiograph"),#DONE
    "full_multimodal": ("photograph", "radiograph", "text"),#DONE
}


ACTIVE_SCENARIO = "full_multimodal"
ACTIVE_MODALITIES = SCENARIOS[ACTIVE_SCENARIO]


RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "final_linear_v3"
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
FREEZE_IMAGE_ENCODER = False


BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1

NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10


BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4
TEXT_LR = 2e-5

WEIGHT_DECAY = 1e-4


WARMUP_EPOCHS = 3

DROPOUT = 0.50

GRADIENT_CLIP_NORM = 1.0

TOP_K_CHECKPOINTS = 3

USE_TTA = True


DEFAULT_THRESHOLD = 0.5

THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.05


SEED = 42

DEVICE = "cuda"

NUM_WORKERS = 0
PIN_MEMORY = True