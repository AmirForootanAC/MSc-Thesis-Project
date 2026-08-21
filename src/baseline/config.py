"""
Configuration for COde baseline experiments.
"""

from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# ============================================================
# Dataset
# ============================================================

DATASET_PATH = (
    PROJECT_ROOT
    /
    "results"
    /
    "six_label_patient_level_dataset"
    /
    "labeled_dataset.csv"
)


IMAGE_ROOT = (
    PROJECT_ROOT
    /
    "data"
    /
    "raw"
    /
    "COde-Dataset"
    /
    "Images"
)


# ============================================================
# Results
# ============================================================

RESULT_ROOT = (
    PROJECT_ROOT
    /
    "results"
    /
    "baseline"
)


EXPERIMENT_NAME = (
    "photograph_only_6label"
)


# ============================================================
# Modality
# ============================================================

MODALITY = "photograph"

REQUIRE_MODALITY = "photograph"


# ============================================================
# Splits
# ============================================================

TRAIN_SPLIT = "train"

VALID_SPLIT = "validation"

TEST_SPLIT = "test"


# ============================================================
# Labels
# ============================================================

NUM_LABELS = 6


LABEL_NAMES = [
    "label_caries",
    "label_gingivitis",
    "label_malocclusion",
    "label_pulpitis",
    "label_tooth_loss",
    "label_tooth_structure_loss",
]


# ============================================================
# Image
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# Training
# ============================================================

BATCH_SIZE = 8

NUM_EPOCHS = 20

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 10


# ============================================================
# Hardware
# ============================================================

DEVICE = "cuda"

PRETRAINED = True

FREEZE_ENCODER = False