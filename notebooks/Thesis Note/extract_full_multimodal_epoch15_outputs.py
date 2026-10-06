from pathlib import Path
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Configuration
# ============================================================

from src.baseline.final_linear_v3 import config


# config.py defaults to image_only.
# Override only in memory for this extraction process.
# The actual config.py file is NOT modified.

config.ACTIVE_SCENARIO = "full_multimodal"
config.ACTIVE_MODALITIES = config.SCENARIOS[
    "full_multimodal"
]


# ============================================================
# Project implementation
# ============================================================

from src.baseline.final_linear_v3.train import (
    build_datasets,
    build_model,
    get_device,
    move_batch_to_device,
    model_forward,
    horizontally_flip_image_lists,
    probabilities_to_logits,
    load_checkpoint,
)

from src.baseline.final_linear_v3.collate import (
    complete_case_collate,
)

from src.baseline.final_linear_v3.metrics import (
    compute_metrics,
)


# ============================================================
# Paths
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "baseline"
    / "final_linear_v3"
    / "full_multimodal"
    / "checkpoints"
    / "epoch_015.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_linear_v3"
)

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "full_multimodal_epoch15_test_outputs.npz"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "full_multimodal_epoch15_test_outputs.json"
)


# ============================================================
# Inference settings
# ============================================================

# The original training configuration uses BATCH_SIZE=8.
#
# We intentionally use batch size 1 here to reduce GPU memory
# usage on the RTX 3050 4 GB. Since the model is evaluated in
# eval() mode, BatchNorm running statistics are not updated and
# therefore changing the inference batch size does not alter
# the evaluation procedure.
INFERENCE_BATCH_SIZE = 1

DEVICE = get_device()


# ============================================================
# Exact test prediction with ID collection
# ============================================================

@torch.no_grad()
def predict_and_collect(
    model,
    loader,
    device,
    use_tta=False,
):
    """
    Same evaluation/TTA logic as train.py::predict(),
    with additional collection of checkup_id and patient_id.

    The TTA implementation intentionally follows train.py:
        original logits
        -> sigmoid
        flipped logits
        -> sigmoid
        -> average probabilities
        -> probabilities_to_logits()
    """

    model.eval()

    all_logits = []
    all_labels = []

    all_checkup_ids = []
    all_patient_ids = []

    total = len(loader.dataset)
    processed = 0

    for batch_index, raw_batch in enumerate(
        loader,
        start=1,
    ):

        # ----------------------------------------------------
        # Preserve identifiers
        # ----------------------------------------------------

        all_checkup_ids.extend(
            raw_batch["checkup_id"]
        )

        all_patient_ids.extend(
            raw_batch["patient_id"]
        )

        # ----------------------------------------------------
        # Move batch exactly as train.py does
        # ----------------------------------------------------

        batch = move_batch_to_device(
            raw_batch,
            device,
        )

        labels = batch["labels"]

        # ----------------------------------------------------
        # Original prediction
        # ----------------------------------------------------

        logits = model_forward(
            model,
            batch,
        )

        if logits.shape != labels.shape:
            raise RuntimeError(
                "Logit/label shape mismatch "
                "during extraction: "
                f"logits={tuple(logits.shape)}, "
                f"labels={tuple(labels.shape)}"
            )

        # ----------------------------------------------------
        # Exact TTA logic from train.py::predict()
        # ----------------------------------------------------

        has_images = (
            "images" in batch
            or "radiographs" in batch
        )

        if not use_tta or not has_images:

            final_logits = logits

        else:

            flipped_batch = dict(batch)

            if "images" in batch:

                flipped_batch["images"] = (
                    horizontally_flip_image_lists(
                        batch["images"]
                    )
                )

            if "radiographs" in batch:

                flipped_batch["radiographs"] = (
                    horizontally_flip_image_lists(
                        batch["radiographs"]
                    )
                )

            flipped_logits = model_forward(
                model,
                flipped_batch,
            )

            original_probabilities = (
                torch.sigmoid(logits)
            )

            flipped_probabilities = (
                torch.sigmoid(
                    flipped_logits
                )
            )

            mean_probabilities = (
                original_probabilities
                + flipped_probabilities
            ) / 2.0

            # EXACT conversion used in train.py.
            final_logits = torch.from_numpy(
                probabilities_to_logits(
                    mean_probabilities
                    .detach()
                    .cpu()
                    .numpy()
                )
            ).to(
                device=device,
                dtype=torch.float32,
            )

        # ----------------------------------------------------
        # Collect
        # ----------------------------------------------------

        all_logits.append(
            final_logits.detach()
            .float()
            .cpu()
            .numpy()
        )

        all_labels.append(
            labels.detach()
            .cpu()
            .numpy()
        )

        processed += len(
            raw_batch["checkup_id"]
        )

        if (
            batch_index == 1
            or batch_index % 50 == 0
            or processed == total
        ):
            print(
                f"[Inference] "
                f"{processed}/{total} visits processed",
                flush=True,
            )

    # --------------------------------------------------------
    # Concatenate
    # --------------------------------------------------------

    logits = np.concatenate(
        all_logits,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    probabilities = 1.0 / (
        1.0 + np.exp(-logits)
    )

    predictions = (
        probabilities
        >= config.DEFAULT_THRESHOLD
    ).astype(
        np.int64
    )

    return (
        logits,
        probabilities,
        predictions,
        labels,
        np.asarray(
            all_checkup_ids
        ),
        np.asarray(
            all_patient_ids
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print(
        "Full Multimodal — Epoch 15 "
        "Test Output Extraction"
    )
    print("=" * 72)

    print(
        f"Project root : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Checkpoint   : "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"Device       : "
        f"{DEVICE}"
    )

    print(
        f"Scenario     : "
        f"{config.ACTIVE_SCENARIO}"
    )

    print(
        f"Modalities   : "
        f"{config.ACTIVE_MODALITIES}"
    )

    print(
        f"Use TTA      : "
        f"{config.USE_TTA}"
    )

    print(
        f"Threshold    : "
        f"{config.DEFAULT_THRESHOLD}"
    )

    print(
        f"Batch size   : "
        f"{INFERENCE_BATCH_SIZE}"
    )

    print()

    # ========================================================
    # Safety checks
    # ========================================================

    if config.ACTIVE_SCENARIO != "full_multimodal":
        raise RuntimeError(
            "ACTIVE_SCENARIO must be "
            "'full_multimodal'."
        )

    expected_modalities = (
        "photograph",
        "radiograph",
        "text",
    )

    if tuple(
        config.ACTIVE_MODALITIES
    ) != expected_modalities:

        raise RuntimeError(
            "Unexpected active modalities: "
            f"{config.ACTIVE_MODALITIES}"
        )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Dataset
    # ========================================================

    print(
        "[1/5] Building datasets using "
        "train.py::build_datasets()..."
    )

    (
        _train_dataset,
        _validation_dataset,
        test_dataset,
    ) = build_datasets()

    print()

    print(
        f"Test visits: "
        f"{len(test_dataset)}"
    )

    # The authoritative complete-case test cohort
    # contains exactly 633 visits.
    if len(test_dataset) != 633:

        raise RuntimeError(
            "Expected 633 complete-case "
            f"test visits, found "
            f"{len(test_dataset)}."
        )

    # ========================================================
    # Test DataLoader
    # ========================================================

    # We use the exact same collate function as train.py.
    # Only batch size is reduced for GPU memory safety.
    test_loader = DataLoader(
        test_dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=config.PIN_MEMORY,
        collate_fn=complete_case_collate,
    )

    # ========================================================
    # Model
    # ========================================================

    print(
        "[2/5] Building Full Multimodal model "
        "using train.py::build_model()..."
    )

    model = build_model()

    model = model.to(
        DEVICE
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    # ========================================================
    # Checkpoint
    # ========================================================

    print(
        "[3/5] Loading epoch 15 checkpoint..."
    )

    checkpoint = load_checkpoint(
        model,
        CHECKPOINT_PATH,
        DEVICE,
    )

    checkpoint_epoch = checkpoint.get(
        "epoch"
    )

    best_epoch = checkpoint.get(
        "best_epoch"
    )

    best_val_f1 = checkpoint.get(
        "best_val_f1"
    )

    checkpoint_val_macro_f1 = checkpoint.get(
        "val_macro_f1"
    )

    print(
        f"Checkpoint epoch      : "
        f"{checkpoint_epoch}"
    )

    print(
        f"Best epoch             : "
        f"{best_epoch}"
    )

    print(
        f"Best validation F1    : "
        f"{best_val_f1}"
    )

    print(
        f"Checkpoint val Macro-F1: "
        f"{checkpoint_val_macro_f1}"
    )

    # ========================================================
    # Test inference
    # ========================================================

    print(
        "[4/5] Running final test inference..."
    )

    (
        logits,
        probabilities,
        predictions,
        labels,
        checkup_ids,
        patient_ids,
    ) = predict_and_collect(
        model=model,
        loader=test_loader,
        device=DEVICE,
        use_tta=config.USE_TTA,
    )

    print()

    print("Output shapes:")

    print(
        f"  logits        : "
        f"{logits.shape}"
    )

    print(
        f"  probabilities : "
        f"{probabilities.shape}"
    )

    print(
        f"  predictions   : "
        f"{predictions.shape}"
    )

    print(
        f"  labels        : "
        f"{labels.shape}"
    )

    print(
        f"  checkup IDs   : "
        f"{checkup_ids.shape}"
    )

    print(
        f"  patient IDs   : "
        f"{patient_ids.shape}"
    )

    # ========================================================
    # Metric verification
    # ========================================================

    print(
        "[5/5] Verifying benchmark metrics..."
    )

    metrics = compute_metrics(
        logits,
        labels,
        threshold=config.DEFAULT_THRESHOLD,
        label_names=config.LABEL_NAMES,
    )

    print()

    print("=" * 72)
    print(
        "VERIFICATION METRICS"
    )
    print("=" * 72)

    print(
        f"Macro-F1 : "
        f"{metrics['macro_f1']:.10f}"
    )

    print(
        f"Micro-F1 : "
        f"{metrics['micro_f1']:.10f}"
    )

    print(
        f"AUROC    : "
        f"{metrics['auroc']:.10f}"
    )

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.10f}"
    )

    # ========================================================
    # Expected Table 5 values
    # ========================================================

    expected = {
        "macro_f1": 0.8377295508659238,
        "micro_f1": 0.8954128440366973,
        "auroc": 0.9595941452144533,
        "accuracy": 0.8357030015797788,
    }

    print()

    print(
        "Expected Table 5 values:"
    )

    for key, value in expected.items():

        print(
            f"{key:10s}: "
            f"{value:.10f}"
        )

    # ========================================================
    # Differences
    # ========================================================

    print()

    print(
        "Absolute differences:"
    )

    differences = {}

    for key, expected_value in expected.items():

        actual_value = float(
            metrics[key]
        )

        difference = abs(
            actual_value
            - expected_value
        )

        differences[key] = difference

        print(
            f"{key:10s}: "
            f"{difference:.10e}"
        )

    # ========================================================
    # Output validation
    # ========================================================

    if logits.shape != (
        633,
        config.NUM_LABELS,
    ):

        raise RuntimeError(
            "Unexpected logits shape: "
            f"{logits.shape}"
        )

    if labels.shape != (
        633,
        config.NUM_LABELS,
    ):

        raise RuntimeError(
            "Unexpected labels shape: "
            f"{labels.shape}"
        )

    if len(checkup_ids) != 633:

        raise RuntimeError(
            "Unexpected number of checkup IDs: "
            f"{len(checkup_ids)}"
        )

    if len(patient_ids) != 633:

        raise RuntimeError(
            "Unexpected number of patient IDs: "
            f"{len(patient_ids)}"
        )

    # ========================================================
    # Save raw outputs
    # ========================================================

    np.savez_compressed(
        OUTPUT_NPZ,
        logits=logits,
        probabilities=probabilities,
        predictions=predictions,
        labels=labels,
        checkup_ids=checkup_ids,
        patient_ids=patient_ids,
    )

    # ========================================================
    # Save metadata
    # ========================================================

    metadata = {
        "scenario": (
            config.ACTIVE_SCENARIO
        ),

        "modalities": list(
            config.ACTIVE_MODALITIES
        ),

        "checkpoint": str(
            CHECKPOINT_PATH
        ),

        "checkpoint_epoch": (
            checkpoint_epoch
        ),

        "best_epoch": (
            best_epoch
        ),

        "best_val_f1": (
            best_val_f1
        ),

        "checkpoint_val_macro_f1": (
            checkpoint_val_macro_f1
        ),

        "dataset": str(
            config.DATASET_PATH
        ),

        "image_root": str(
            config.IMAGE_ROOT
        ),

        "test_visits": int(
            len(test_dataset)
        ),

        "num_labels": int(
            config.NUM_LABELS
        ),

        "labels": list(
            config.LABEL_NAMES
        ),

        "threshold": float(
            config.DEFAULT_THRESHOLD
        ),

        "use_tta": bool(
            config.USE_TTA
        ),

        "inference_batch_size": (
            INFERENCE_BATCH_SIZE
        ),

        "metrics": {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            )
        },

        "expected_table_5_metrics": (
            expected
        ),

        "absolute_differences": (
            differences
        ),

        "outputs": {
            "npz": str(
                OUTPUT_NPZ
            ),
        },
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Final message
    # ========================================================

    print()

    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"NPZ : "
        f"{OUTPUT_NPZ}"
    )

    print(
        f"JSON: "
        f"{OUTPUT_JSON}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()