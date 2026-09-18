"""Configuration for strong X-ray-only experiment v2."""

from pathlib import Path

from src.baseline.final import config as final_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = final_config.DATASET_PATH
IMAGE_ROOT = final_config.IMAGE_ROOT

LABEL_NAMES = final_config.LABEL_NAMES
NUM_LABELS = final_config.NUM_LABELS

EXPECTED_COUNTS = final_config.EXPECTED_COUNTS

MODALITIES = ("radiograph",)

IMAGE_SIZE = 256

PRETRAINED = True

BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1

NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4

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

NUM_WORKERS = 0
PIN_MEMORY = True

DEVICE = "cuda"

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "test_xray_strong_v2"
    / "xray_only_max_strength"
)