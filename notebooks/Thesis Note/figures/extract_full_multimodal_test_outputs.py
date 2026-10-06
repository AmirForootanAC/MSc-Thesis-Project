from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.baseline.final_linear_v3 import config


# ============================================================
# IMPORTANT
# ============================================================
# config.py defaults to image_only.
# For this extraction only, switch it in memory to
# full_multimodal. The actual config.py file is NOT modified.
# ============================================================

config.ACTIVE_SCENARIO = "full_multimodal"
config.ACTIVE_MODALITIES = config.SCENARIOS["full_multimodal"]


from src.baseline.final_linear_v3.dataset import CompleteCaseDataset
from src.baseline.final_linear_v3.collate import complete_case_collate
from src.baseline.final_linear_v3.metrics import compute_metrics
from src.baseline.final_linear_v3.train import (
    build_model,
    get_device,
    move_batch_to_device,
    model_forward,
    horizontally_flip_image_lists,
    probabilities_to_logits,
    load_checkpoint,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

# Batch size 1 is intentional for the RTX 3050 4 GB.
# Evaluation mode makes the model output independent of batch
# size because BatchNorm running statistics are not updated.
INFERENCE_BATCH_SIZE = 1

DEVICE = get_device()


# ============================================================
# Exact TTA prediction
# ============================================================

def predict_and_collect(model, loader, device):

    model.eval()

    all_logits = []
    all_labels = []
    all_checkup_ids = []
    all_patient_ids = []

    total = len(loader.dataset)
    processed = 0

    with torch.inference_mode():

        for batch_index, raw_batch in enumerate(
            loader,
            start=1,
        ):

            # Keep identifiers before moving tensors.
            all_checkup_ids.extend(
                raw_batch["checkup_id"]
            )

            all_patient_ids.extend(
                raw_batch["patient_id"]
            )

            batch = move_batch_to_device(
                raw_batch,
                device,
            )

            # ----------------------------------------------
            # Original prediction
            # ----------------------------------------------

            logits = model_forward(
                model,
                batch,
            )

            final_logits = logits

            # ----------------------------------------------
            # Exact TTA used in train.py
            # ----------------------------------------------

            if config.USE_TTA and (
                batch.get("images") is not None
                or batch.get("radiographs") is not None
            ):

                flipped_batch = dict(batch)

                if batch.get("images") is not None:

                    flipped_batch["images"] = (
                        horizontally_flip_image_lists(
                            batch["images"]
                        )
                    )

                if batch.get("radiographs") is not None:

                    flipped_batch["radiographs"] = (
                        horizontally_flip_image_lists(
                            batch["radiographs"]
                        )
                    )

                flipped_logits = model_forward(
                    model,
                    flipped_batch,
                )

                original_probs = torch.sigmoid(
                    logits
                )

                flipped_probs = torch.sigmoid(
                    flipped_logits
                )

                mean_probs = (
                    original_probs
                    + flipped_probs
                ) / 2.0

                final_logits = (
                    probabilities_to_logits(
                        mean_probs
                    )
                )

            all_logits.append(
                final_logits.cpu()
            )

            all_labels.append(
                batch["labels"].cpu()
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

    logits = torch.cat(
        all_logits,
        dim=0,
    ).numpy()

    labels = torch.cat(
        all_labels,
        dim=0,
    ).numpy()

    probabilities = 1.0 / (
        1.0 + np.exp(-logits)
    )

    predictions = (
        probabilities
        >= config.DEFAULT_THRESHOLD
    ).astype(np.int64)

    return (
        logits,
        probabilities,
        predictions,
        labels,
        np.asarray(all_checkup_ids),
        np.asarray(all_patient_ids),
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

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Checkpoint   : {CHECKPOINT_PATH}")
    print(f"Device       : {DEVICE}")
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

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if config.ACTIVE_SCENARIO != "full_multimodal":

        raise RuntimeError(
            "Extraction scenario is not full_multimodal."
        )

    expected_modalities = {
        "photograph",
        "radiograph",
        "text",
    }

    if set(config.ACTIVE_MODALITIES) != expected_modalities:

        raise RuntimeError(
            "Expected photograph + radiograph + text."
        )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "[1/5] Building complete-case "
        "test dataset..."
    )

    test_dataset = CompleteCaseDataset(
        csv_path=config.DATASET_PATH,
        split="test",
        image_root=config.IMAGE_ROOT,
        tokenizer_name=config.TEXT_MODEL_NAME,
        image_transform=config.get_eval_transform(),
        active_modalities=config.ACTIVE_MODALITIES,
    )

    print(
        f"Test visits: "
        f"{len(test_dataset)}"
    )

    if len(test_dataset) != 633:

        raise RuntimeError(
            "Expected 633 complete-case "
            f"test visits, found "
            f"{len(test_dataset)}."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        collate_fn=complete_case_collate,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "[2/5] Building Full Multimodal model..."
    )

    model = build_model()

    model = model.to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

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

    print(
        f"Checkpoint epoch  : "
        f"{checkpoint_epoch}"
    )

    print(
        f"Best epoch        : "
        f"{best_epoch}"
    )

    print(
        f"Best validation F1: "
        f"{best_val_f1}"
    )

    # --------------------------------------------------------
    # Test inference
    # --------------------------------------------------------

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
        model,
        test_loader,
        DEVICE,
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

    # --------------------------------------------------------
    # Metric verification
    # --------------------------------------------------------

    print(
        "[5/5] Verifying benchmark metrics..."
    )

    logits_tensor = torch.from_numpy(
        logits
    )

    labels_tensor = torch.from_numpy(
        labels
    )

    metrics = compute_metrics(
        logits_tensor,
        labels_tensor,
        threshold=config.DEFAULT_THRESHOLD,
        label_names=config.LABEL_NAMES,
    )

    print()
    print("=" * 72)
    print("VERIFICATION METRICS")
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

    # --------------------------------------------------------
    # Expected published values
    # --------------------------------------------------------

    expected = {
        "macro_f1": 0.8377295508659238,
        "micro_f1": 0.8954128440366973,
        "auroc": 0.9595941452144533,
        "accuracy": 0.8357030015797788,
    }

    print()
    print("Expected Table 5 values:")

    for key, value in expected.items():

        print(
            f"{key:10s}: "
            f"{value:.10f}"
        )

    print()
    print("Absolute differences:")

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

    # --------------------------------------------------------
    # Save raw outputs
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_NPZ,
        logits=logits,
        probabilities=probabilities,
        predictions=predictions,
        labels=labels,
        checkup_ids=checkup_ids,
        patient_ids=patient_ids,
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "scenario": config.ACTIVE_SCENARIO,
        "modalities": list(
            config.ACTIVE_MODALITIES
        ),
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "checkpoint_epoch": checkpoint_epoch,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_visits": int(
            len(test_dataset)
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
        "labels": list(
            config.LABEL_NAMES
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
        "expected_metrics": expected,
        "absolute_differences": differences,
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
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"NPZ : {OUTPUT_NPZ}"
    )

    print(
        f"JSON: {OUTPUT_JSON}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()