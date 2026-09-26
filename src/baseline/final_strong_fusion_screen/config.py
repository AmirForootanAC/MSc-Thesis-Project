"""Fast screening configuration for multimodal fusion."""

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


SOURCE_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "final_strong_v2"
    / "full_multimodal"
    / "checkpoints"
    / "epoch_009.pt"
)


RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "final_strong_fusion_screen"
)


SCENARIOS = {
    "text_only": ("text",),
    "image_text": ("photograph", "text"),
    "full_multimodal": (
        "photograph",
        "radiograph",
        "text",
    ),
}


ACTIVE_SCENARIOS = [
    "text_only",
    "image_text",
    "full_multimodal",
]


EXPECTED_COUNTS = {
    "train": 2935,
    "validation": 627,
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


BATCH_SIZE = 16


SCREEN_TRAIN_SAMPLES = 1000

NUM_EPOCHS = 9

EARLY_STOPPING_PATIENCE = 5


FUSION_DIM = 256
FUSION_HIDDEN_DIM = 128

DROPOUT = 0.20


LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


SEED = 42


DEVICE = "cuda"

NUM_WORKERS = 0

PIN_MEMORY = True


CACHE_VERSION = "v1"

CACHE_ROOT = (
    RESULT_ROOT
    / "feature_cache"
)