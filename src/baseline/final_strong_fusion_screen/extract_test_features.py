"""Extract frozen V2 representations for the held-out test set."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.baseline.final_strong_fusion_screen import config

from src.baseline.final_strong_v2.collate import (
    complete_case_collate,
)
from src.baseline.final_strong_v2.dataset import (
    CompleteCaseDataset,
)
from src.baseline.final_strong_v2.model import (
    FinalStrongV2,
)
from src.baseline.final_strong_v2.transforms import (
    get_eval_transform,
)


# ================================================================
# Constants
# ================================================================

EXPECTED_TEST_COUNT = 633


# ================================================================
# Device
# ================================================================

def get_device():
    if (
        config.DEVICE == "cuda"
        and torch.cuda.is_available()
    ):
        return torch.device("cuda")

    return torch.device("cpu")


# ================================================================
# Batch handling
# ================================================================

def move_image_lists(
    image_lists,
    device,
):
    result = []

    for image_list in image_lists:
        result.append(
            [
                image.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                for image in image_list
            ]
        )

    return result


def move_batch(
    batch,
    device,
):
    result = {
        "labels": batch["labels"].to(
            device,
            dtype=torch.float32,
        )
    }

    if "images" in batch:
        result["images"] = move_image_lists(
            batch["images"],
            device,
        )

    if "radiographs" in batch:
        result["radiographs"] = move_image_lists(
            batch["radiographs"],
            device,
        )

    if "input_ids" in batch:
        result["input_ids"] = batch[
            "input_ids"
        ].to(device)

        result["attention_mask"] = batch[
            "attention_mask"
        ].to(device)

    return result


# ================================================================
# Dataset
# ================================================================

def build_test_dataset():
    modalities = (
        "photograph",
        "radiograph",
        "text",
    )

    transform = get_eval_transform(
        image_size=config.IMAGE_SIZE
    )

    dataset = CompleteCaseDataset(
        csv_path=config.DATASET_PATH,
        split="test",
        image_root=config.IMAGE_ROOT,
        modalities=modalities,
        tokenizer_name=config.TEXT_MODEL_NAME,
        max_length=config.TEXT_MAX_LENGTH,
        photograph_transform=transform,
        radiograph_transform=transform,
    )

    print(
        f"test: {len(dataset)} complete-case samples | "
        f"{modalities}"
    )

    if len(dataset) != EXPECTED_TEST_COUNT:
        raise RuntimeError(
            f"Test count mismatch: "
            f"expected {EXPECTED_TEST_COUNT}, "
            f"found {len(dataset)}"
        )

    return dataset


def build_test_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=complete_case_collate,
    )


# ================================================================
# Source model
# ================================================================

def build_source_model(device):
    model = FinalStrongV2(
        modalities=(
            "photograph",
            "radiograph",
            "text",
        ),
        text_model_name=config.TEXT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
        pretrained_image=True,
        freeze_image_encoder=False,
    ).to(device)

    checkpoint = torch.load(
        config.SOURCE_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    print()
    print("=" * 80)
    print("SOURCE CHECKPOINT")
    print("=" * 80)
    print(
        f"Path:          {config.SOURCE_CHECKPOINT}"
    )
    print(
        f"Epoch:         {checkpoint.get('epoch')}"
    )
    print(
        f"Val Macro F1:  "
        f"{checkpoint.get('val_macro_f1'):.4f}"
    )
    print("=" * 80)

    return model


# ================================================================
# Feature extraction
# ================================================================

@torch.no_grad()
def extract_test_features(
    model,
    loader,
    device,
):
    text_features = []
    photograph_features = []
    radiograph_features = []
    labels = []

    progress = tqdm(
        loader,
        desc="Extracting test features",
        dynamic_ncols=True,
    )

    for raw_batch in progress:
        batch = move_batch(
            raw_batch,
            device,
        )

        # --------------------------------------------------------
        # Text
        # --------------------------------------------------------

        outputs = model.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch[
                "attention_mask"
            ],
        )

        text_features.append(
            outputs.last_hidden_state[
                :,
                0,
            ]
            .float()
            .cpu()
        )

        # --------------------------------------------------------
        # Photograph
        # --------------------------------------------------------

        photograph = (
            model.photograph_aggregator(
                batch["images"]
            )
        )

        photograph_features.append(
            photograph.float().cpu()
        )

        # --------------------------------------------------------
        # Radiograph
        # --------------------------------------------------------

        radiograph = (
            model.radiograph_aggregator(
                batch["radiographs"]
            )
        )

        radiograph_features.append(
            radiograph.float().cpu()
        )

        # --------------------------------------------------------
        # Labels
        # --------------------------------------------------------

        labels.append(
            batch["labels"]
            .float()
            .cpu()
        )

    result = {
        "labels": torch.cat(
            labels,
            dim=0,
        ),
        "text": torch.cat(
            text_features,
            dim=0,
        ),
        "photograph": torch.cat(
            photograph_features,
            dim=0,
        ),
        "radiograph": torch.cat(
            radiograph_features,
            dim=0,
        ),
    }

    return result


# ================================================================
# Validation
# ================================================================

def validate_features(features):
    required_keys = (
        "labels",
        "text",
        "photograph",
        "radiograph",
    )

    for key in required_keys:
        if key not in features:
            raise RuntimeError(
                f"Missing feature key: {key}"
            )

        if features[key].shape[0] != EXPECTED_TEST_COUNT:
            raise RuntimeError(
                f"{key} sample count mismatch: "
                f"expected {EXPECTED_TEST_COUNT}, "
                f"found {features[key].shape[0]}"
            )

    expected_shapes = {
        "labels": (
            EXPECTED_TEST_COUNT,
            6,
        ),
        "text": (
            EXPECTED_TEST_COUNT,
            768,
        ),
        "photograph": (
            EXPECTED_TEST_COUNT,
            128,
        ),
        "radiograph": (
            EXPECTED_TEST_COUNT,
            128,
        ),
    }

    for key, shape in expected_shapes.items():
        actual = tuple(features[key].shape)

        if actual != shape:
            raise RuntimeError(
                f"{key} shape mismatch: "
                f"expected {shape}, "
                f"found {actual}"
            )

    for key, value in features.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(
                f"Non-finite values found in {key}"
            )


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 80)
    print("V2 FROZEN TEST FEATURE EXTRACTION")
    print("=" * 80)

    device = get_device()

    print(f"Device: {device}")
    print(
        f"Source checkpoint: "
        f"{config.SOURCE_CHECKPOINT}"
    )

    output_dir = Path(
        config.CACHE_ROOT
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "test_v1.pt"
    )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    dataset = build_test_dataset()

    loader = build_test_loader(
        dataset
    )

    # ------------------------------------------------------------
    # Source model
    # ------------------------------------------------------------

    model = build_source_model(
        device
    )

    # ------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------

    features = extract_test_features(
        model=model,
        loader=loader,
        device=device,
    )

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    validate_features(
        features
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    torch.save(
        features,
        output_path,
    )

    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)

    for key, value in features.items():
        print(
            f"{key:<15} "
            f"{tuple(value.shape)}"
        )

    print()
    print(
        f"Saved to: {output_path}"
    )

    print()
    print(
        "TEST SET WAS ONLY USED FOR FEATURE EXTRACTION."
    )
    print(
        "NO TRAINING / NO THRESHOLD OPTIMIZATION / NO TEST TUNING."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
