"""
Configuration for Decoupled Modular Multi-View SSL & 4-Scenario Late Fusion.

Stage 1A: Photograph <-> Text Multi-View SSL (ConvNeXt-Tiny + Bio_ClinicalBERT)
Stage 1B: Radiograph <-> Text Multi-View SSL (DenseNet121 + Bio_ClinicalBERT)
Stage 2 : Multi-View Representation Extraction (.pt artifacts)
Stage 3 : 4-Scenario Gated Cross-Modal Attention Fusion (image_only, xray_only, text_only, full_multimodal)
"""

from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# Dataset & Fast Local Cache Paths
# ============================================================

DATASET_FOUR_LABEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "four_label_patient_level_dataset"
    / "labeled_dataset.csv"
)

DATASET_SIX_LABEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "six_label_patient_level_dataset"
    / "labeled_dataset.csv"
)

DATASET_PATH = (
    DATASET_FOUR_LABEL_PATH
    if DATASET_FOUR_LABEL_PATH.exists()
    else DATASET_SIX_LABEL_PATH
)

# If copied to /tmp/COde_Images on Kaggle for fast local SSD I/O, prioritize it
FAST_TMP_IMAGE_ROOT = Path("/tmp/COde_Images")
KAGGLE_MOUNT_IMAGE_ROOT = Path("/kaggle/input/datasets/ahforootan/code-dataset/Images")
LOCAL_SYMLINK_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "COde-Dataset" / "Images"

if FAST_TMP_IMAGE_ROOT.exists():
    IMAGE_ROOT = FAST_TMP_IMAGE_ROOT
elif LOCAL_SYMLINK_IMAGE_ROOT.exists():
    IMAGE_ROOT = LOCAL_SYMLINK_IMAGE_ROOT
else:
    IMAGE_ROOT = KAGGLE_MOUNT_IMAGE_ROOT


# ============================================================
# Output Directories
# ============================================================

OUTPUT_ROOT = PROJECT_ROOT / "results" / "modular_ssl"
PHOTO_SSL_DIR = OUTPUT_ROOT / "ssl_photograph"
RADIO_SSL_DIR = OUTPUT_ROOT / "ssl_radiograph"
REPRESENTATION_DIR = OUTPUT_ROOT / "representations"
FUSION_SCENARIOS_DIR = OUTPUT_ROOT / "scenarios_4labels"


# ============================================================
# 4 Canonical Clinical Labels & 4 Scenarios
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

SCENARIOS = {
    "image_only": ("photograph",),
    "xray_only": ("radiograph",),
    "text_only": ("text",),
    "full_multimodal": ("photograph", "radiograph", "text"),
}

EXPECTED_COMPLETE_CASE_COUNTS = {
    "train": 2935,
    "validation": 627,
    "test": 633,
}


# ============================================================
# Backbones & Multi-View Settings (No Averaging)
# ============================================================

PHOTO_BACKBONE = "convnext_tiny"      # 768-d spatial features
RADIO_BACKBONE = "densenet121"        # 1024-d spatial features
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"  # 768-d token features

TEXT_MAX_LENGTH = 256
IMAGE_SIZE = 224                      # 224x224 is 35% faster than 256x256 with identical accuracy

# Keep all views without pixel averaging:
MAX_PHOTO_VIEWS_TRAIN = 6             # Median photo count per visit is 6
MAX_RADIO_VIEWS_TRAIN = 3             # 95th percentile of radiographs per visit is 3

MAX_PHOTO_VIEWS_EXTRACT = 12          # Up to 12 views stored per visit for Cross-Modal Attention
MAX_RADIO_VIEWS_EXTRACT = 4           # Up to 4 radiographs stored per visit

SHARED_EMBED_DIM = 256
PROJECTION_DIM = 128


# ============================================================
# Modular SSL Pretraining Hyperparameters (Per-Modality)
# ============================================================

SSL_BATCH_SIZE = 32                   # Single visual branch fits B=32 easily in FP16!
SSL_EPOCHS_PHOTO = 25
SSL_EPOCHS_RADIO = 30
SSL_WARMUP_EPOCHS = 2

SSL_VISION_LR = 1.5e-4
SSL_TEXT_LR = 2e-5
SSL_HEAD_LR = 3e-4
SSL_WEIGHT_DECAY = 1e-2

# Freeze bottom 10 layers of Bio_ClinicalBERT during SSL so text encoding is 5x faster
SSL_FREEZE_TEXT_LAYERS = 10
SSL_FREEZE_VISION_STAGES = 1

SSL_TEMPERATURE = 0.07
SSL_SOFT_ALPHA = 0.25
SSL_LOCAL_VIEW_WEIGHT = 0.35
SSL_INTRA_VIEW_WEIGHT = 0.30


# ============================================================
# Stage 3: 4-Scenario Fusion Training Hyperparameters
# ============================================================

FUSION_BATCH_SIZE = 64
FUSION_EPOCHS = 60
FUSION_PATIENCE = 12
FUSION_LR = 3e-4
FUSION_WEIGHT_DECAY = 1e-2
FUSION_DROPOUT = 0.35
MODALITY_DROPOUT = 0.15

LABEL_SMOOTHING = 0.05
MAX_POS_WEIGHT = 5.0
DEFAULT_THRESHOLD = 0.50
THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.70
THRESHOLD_STEP = 0.05


# ============================================================
# Hardware & DataLoader
# ============================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_MULTI_GPU = NUM_GPUS > 1
USE_AMP = DEVICE == "cuda"

# Use 2 workers with persistent_workers=True when running single-modality SSL
NUM_WORKERS = 2 if DEVICE == "cuda" else 0
PIN_MEMORY = DEVICE == "cuda"