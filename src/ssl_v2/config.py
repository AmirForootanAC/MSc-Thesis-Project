"""
Centralized Configuration for SSL v2 and 4-Scenario Multimodal Fusion.

Optimized for Dual NVIDIA Tesla T4 GPUs (2 x 16GB VRAM) on Kaggle.
"""

from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# Dataset Paths
# ============================================================

DATASET_FOUR_LABEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "four_label_patient_level_dataset"
    / "labeled_dataset.csv"
)

DATASET_FALLBACK_SIX_LABEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "six_label_patient_level_dataset"
    / "labeled_dataset.csv"
)

# Use the 4-label dataset if generated; otherwise read the 6-label CSV
# and filter the 4 target columns on the fly.
DATASET_PATH = (
    DATASET_FOUR_LABEL_PATH
    if DATASET_FOUR_LABEL_PATH.exists()
    else DATASET_FALLBACK_SIX_LABEL_PATH
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "COde-Dataset"
    / "Images"
)

SSL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ssl_v2"
    / "pretraining"
)

REPRESENTATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "ssl_v2"
    / "representations"
)

DOWNSTREAM_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ssl_v2"
    / "downstream_4scenarios"
)


# ============================================================
# 4 Execution Scenarios (Downstream & Fusion)
# ============================================================

SCENARIOS = {
    "image_only": ("photograph",),
    "xray_only": ("radiograph",),
    "text_only": ("text",),
    "full_multimodal": ("photograph", "radiograph", "text"),
}

ACTIVE_SCENARIO = "full_multimodal"
ACTIVE_MODALITIES = SCENARIOS[ACTIVE_SCENARIO]


# ============================================================
# 4 Canonical Clinical Labels & Authoritative Counts
# ============================================================

LABEL_NAMES = [
    "label_caries",
    "label_gingivitis",
    "label_malocclusion",
    "label_pulpitis",
]

NUM_LABELS = len(LABEL_NAMES)  # 4

TEXT_COLUMNS = [
    "chief_complaint",
    "present_illness",
    "past_medical_record",
    "examination",
]

EXPECTED_SSL_COUNTS = {
    "train": 6129,
    "validation": 1330,
    "test": 1316,
}

EXPECTED_COMPLETE_CASE_COUNTS = {
    "train": 2935,
    "validation": 627,
    "test": 633,
}


# ============================================================
# Specialized Backbones & Dimensions
# ============================================================

# 1. Intraoral Photographs -> ConvNeXt-Tiny (768-d)
PHOTO_BACKBONE = "convnext_tiny"
PHOTO_FEATURE_DIM = 768

# 2. Dental Radiographs -> DenseNet121 (1024-d) + CLAHE preprocessing
RADIO_BACKBONE = "densenet121"
RADIO_FEATURE_DIM = 1024
USE_CLAHE_FOR_XRAY = True

# 3. Clinical Text -> Bio_ClinicalBERT (768-d)
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
TEXT_FEATURE_DIM = 768
TEXT_MAX_LENGTH = 256

# Shared SSL latent & attention space
SHARED_EMBED_DIM = 256
PROJECTION_DIM = 128


# ============================================================
# Multi-View (No-Averaging) Configuration for 2x T4 GPUs
# ============================================================

IMAGE_SIZE = 256

# P75 of photographs is 8; P95 of radiographs is 3.
# During training, we sample up to MAX_VIEWS without replacement
# so every view is processed at full resolution without OOM.
# During evaluation/extraction, ALL views of the visit are processed.
MAX_PHOTO_VIEWS_TRAIN = 8
MAX_RADIO_VIEWS_TRAIN = 3

MAX_PHOTO_VIEWS_EVAL = 32
MAX_RADIO_VIEWS_EVAL = 16


# ============================================================
# SSL v2 Pretraining Hyperparameters
# ============================================================

SSL_BATCH_SIZE = 16          # 8 per T4 GPU across 2 GPUs
SSL_GRAD_ACCUM_STEPS = 2     # Effective batch size = 32 visits (~250 images/step)
SSL_EPOCHS = 40
SSL_WARMUP_EPOCHS = 3

SSL_VISION_LR = 1e-4
SSL_TEXT_LR = 3e-5
SSL_HEAD_LR = 3e-4
SSL_WEIGHT_DECAY = 1e-2

SSL_TEMPERATURE = 0.07
SSL_SOFT_TARGET_ALPHA = 0.25   # Weight for MedCLIP-style clinical semantic soft target
SSL_LOCAL_VIEW_WEIGHT = 0.35   # Weight for GLoRIA-style view-to-visit alignment
SSL_INTRA_VIEW_WEIGHT = 0.30   # Weight for multi-view intra-visit consistency


# ============================================================
# Downstream / Fusion v2 Hyperparameters (4 Scenarios)
# ============================================================

# Mode can be:
#   - "end_to_end_finetune": fine-tunes SSL encoders + Gated Cross-Modal Fusion
#   - "frozen_ssl_fusion"  : trains Gated Cross-Modal Fusion on pre-extracted SSL features
DOWNSTREAM_MODE = "end_to_end_finetune"

DOWNSTREAM_BATCH_SIZE = 8
DOWNSTREAM_GRAD_ACCUM_STEPS = 2  # Effective batch size = 16
DOWNSTREAM_EPOCHS = 35
EARLY_STOPPING_PATIENCE = 7
WARMUP_EPOCHS = 2

BACKBONE_LR = 1.5e-5
TEXT_LR = 1.2e-5
FUSION_HEAD_LR = 8e-5
WEIGHT_DECAY = 1e-2

FREEZE_PHOTO_STAGES = 2      # Freeze early stages 0 & 1 of ConvNeXt-Tiny
FREEZE_RADIO_BLOCKS = 2      # Freeze denseblock1 & denseblock2 of DenseNet121
FREEZE_TEXT_LAYERS = 4       # Freeze bottom 4 layers of Bio_ClinicalBERT (out of 12)

DROPOUT = 0.35
MODALITY_DROPOUT = 0.15
LABEL_SMOOTHING = 0.05
MAX_POS_WEIGHT = 5.0
GRADIENT_CLIP_NORM = 1.0

USE_TTA = True
DEFAULT_THRESHOLD = 0.50
THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.70
THRESHOLD_STEP = 0.05
TOP_K_CHECKPOINTS = 3


# ============================================================
# Hardware & Reproducibility (Kaggle 2x T4)
# ============================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_MULTI_GPU = NUM_GPUS > 1
USE_AMP = DEVICE == "cuda"

# 4 workers significantly speeds up multi-view image loading on Kaggle
NUM_WORKERS = 4 if DEVICE == "cuda" else 0
PIN_MEMORY = DEVICE == "cuda"