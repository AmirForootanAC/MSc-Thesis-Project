"""
Configuration for SSL multimodal fusion experiments.
"""

from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


# ============================================================
# Dataset
# ============================================================

DATASET_PATH = (
    PROJECT_ROOT
    /
    "results"
    /
    "labeled_patient_level_dataset"
    /
    "labeled_dataset.csv"
)


# ============================================================
# Modalities
# ============================================================

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


RADIOGRAPH_ROOT = (
    PROJECT_ROOT
    /
    "data"
    /
    "raw"
    /
    "COde-Dataset"
    /
    "Radiographs"
)


# ============================================================
# SSL checkpoint
# ============================================================

SSL_CHECKPOINT = (
    PROJECT_ROOT
    /
    "results"
    /
    "ssl_pretraining"
    /
    "full_dynamic"
    /
    "best_ssl_model.pt"
)


# ============================================================
# Labels
# ============================================================

LABEL_NAMES = [
    "label_caries",
    "label_gingivitis",
    "label_malocclusion",
    "label_pulpitis",
    "label_tooth_loss",
    "label_tooth_structure_loss",
]