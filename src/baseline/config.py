"""
Configuration for COde baseline experiments.
"""


from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


DATASET_PATH = (
    PROJECT_ROOT
    /
    "results"
    /
    "labeled_patient_level_dataset"
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


RESULT_ROOT = (
    PROJECT_ROOT
    /
    "results"
    /
    "baseline"
)


EXPERIMENT_NAME = (
    "radiograph_only_finetune"
)

MODALITY = "radiograph"

REQUIRE_MODALITY = "radiograph"


TRAIN_SPLIT = "train"

VALID_SPLIT = "validation"

TEST_SPLIT = "test"


NUM_LABELS = 13


IMAGE_SIZE = 224


BATCH_SIZE = 16


NUM_EPOCHS = 20


LEARNING_RATE = 1e-5


WEIGHT_DECAY = 1e-4


EARLY_STOPPING_PATIENCE = 5


DEVICE = "cuda"


PRETRAINED = True


FREEZE_ENCODER = False


LABEL_NAMES = [
    "label_gingivitis",
    "label_class_ii_malocclusion",
    "label_dental_crowding",
    "label_tooth_structure_loss",
    "label_dental_caries",
    "label_convex_profile",
    "label_mandibular_skeletal_asymmetry",
    "label_periodontitis",
    "label_class_iii_malocclusion",
    "label_pulpitis",
    "label_deep_overbite",
    "label_class_i_malocclusion",
    "label_tooth_loss",
]