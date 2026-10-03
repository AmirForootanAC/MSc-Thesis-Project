"""Independent random-seed robustness runner for final_linear_v3.

Supports all seven final supervised V3 scenarios.

The current robustness experiment intentionally runs only:
    - text
    - photo_text
    - radiograph_text
    - full_multimodal

Supported seeds for new runs:
    - 123
    - 456

Seed 42 is the existing reference run and is NEVER retrained or resumed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.baseline.final_linear_v3 import config
from src.baseline.final_linear_v3 import train


# ======================================================================
# Scenario definitions
# ======================================================================

SCENARIOS = {
    "text": "text_only",
    "photo": "image_only",
    "radiograph": "xray_only",
    "photo_text": "image_text",
    "photo_radiograph": "image_xray",
    "radiograph_text": "text_xray",
    "full_multimodal": "full_multimodal",
}


# ======================================================================
# Robustness experiment settings
# ======================================================================

NEW_SEEDS = {
    123,
    456,
}


ROBUSTNESS_ROOT = (
    config.PROJECT_ROOT
    / "results"
    / "baseline"
    / "final_linear_v3_robustness"
)


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one independent random-seed robustness "
            "experiment for final_linear_v3."
        )
    )

    parser.add_argument(
        "--scenario",
        required=True,
        choices=sorted(SCENARIOS.keys()),
        help="Robustness scenario name.",
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=sorted(NEW_SEEDS),
        help=(
            "New independent random seed. "
            "Seed 42 is the existing reference run "
            "and must not be retrained."
        ),
    )

    return parser.parse_args()


# ======================================================================
# Test prediction capture
# ======================================================================

def attach_test_prediction_capture():
    """
    Wrap the existing V3 predict() function.

    The underlying prediction implementation is unchanged.

    We only capture the final [Test TTA] logits and labels so that
    the robustness run stores the required prediction artifact.
    """

    captured = {}

    original_predict = train.predict

    def capture_predict(
        model,
        loader,
        device,
        use_tta=False,
        description="[Eval]",
    ):
        logits, labels = original_predict(
            model=model,
            loader=loader,
            device=device,
            use_tta=use_tta,
            description=description,
        )

        if description == "[Test TTA]":
            captured["test_logits"] = np.asarray(
                logits,
                dtype=np.float32,
            )

            captured["test_labels"] = np.asarray(
                labels,
                dtype=np.float32,
            )

        return logits, labels

    train.predict = capture_predict

    return (
        original_predict,
        captured,
    )


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    base_scenario = SCENARIOS[
        args.scenario
    ]

    run_name = (
        f"{args.scenario}_seed{args.seed}"
    )

    output_dir = (
        ROBUSTNESS_ROOT
        / run_name
    )

    modalities = config.SCENARIOS[
        base_scenario
    ]

    # ------------------------------------------------------------------
    # Absolute protection against accidental overwrite / resume.
    # ------------------------------------------------------------------

    if output_dir.exists():
        raise RuntimeError(
            "\nRobustness run already exists:\n"
            f"  {output_dir}\n\n"
            "This runner NEVER resumes or overwrites an existing "
            "robustness run.\n\n"
            "If you intentionally want to restart this experiment "
            "from epoch 1, delete the run directory first."
        )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Configure the existing V3 training code.
    #
    # Only the following intentionally change:
    #   - scenario
    #   - modalities
    #   - seed
    #   - output root
    #
    # All architecture and training hyperparameters remain those
    # defined by final_linear_v3.
    # ------------------------------------------------------------------

    config.ACTIVE_SCENARIO = run_name

    config.ACTIVE_MODALITIES = modalities

    config.SEED = args.seed

    config.RESULT_ROOT = (
        ROBUSTNESS_ROOT
    )

    # ------------------------------------------------------------------
    # Capture final test predictions.
    # ------------------------------------------------------------------

    (
        original_predict,
        captured,
    ) = attach_test_prediction_capture()

    print("\n" + "=" * 70)
    print(
        "FINAL LINEAR V3 RANDOM-SEED ROBUSTNESS"
    )
    print("=" * 70)

    print(
        f"Base scenario:      {base_scenario}"
    )

    print(
        f"Run name:           {run_name}"
    )

    print(
        f"Seed:               {args.seed}"
    )

    print(
        f"Modalities:         {modalities}"
    )

    print(
        f"Output directory:   {output_dir}"
    )

    print(
        "Reference seed 42:  NOT USED"
    )

    print(
        "Checkpoint source:  NONE"
    )

    print(
        "Resume:             DISABLED"
    )

    print(
        "Start epoch:        1"
    )

    print(
        "Architecture:       final_linear_v3"
    )

    print(
        "Training protocol:  UNCHANGED"
    )

    print("=" * 70)

    try:
        # --------------------------------------------------------------
        # Existing V3 training function.
        #
        # Because output_dir does not exist before this call, the
        # existing V3 prepare_scenario_output_directory() will create
        # a fresh run and will find no latest.pt.
        # --------------------------------------------------------------

        train.main()

    finally:
        train.predict = original_predict

    # ------------------------------------------------------------------
    # Verify that final test predictions were captured.
    # ------------------------------------------------------------------

    if (
        "test_logits" not in captured
        or "test_labels" not in captured
    ):
        raise RuntimeError(
            "\nFinal test predictions were not captured.\n"
            "The robustness run cannot be considered complete."
        )

    test_logits = captured[
        "test_logits"
    ]

    test_labels = captured[
        "test_labels"
    ]

    # ------------------------------------------------------------------
    # Convert final TTA logits to probabilities.
    #
    # This exactly corresponds to the logits returned by the existing
    # V3 test prediction path after TTA probability averaging.
    # ------------------------------------------------------------------

    clipped_logits = np.clip(
        test_logits,
        -50.0,
        50.0,
    )

    test_probabilities = (
        1.0
        / (
            1.0
            + np.exp(
                -clipped_logits
            )
        )
    )

    test_predictions = (
        test_probabilities
        >= config.DEFAULT_THRESHOLD
    ).astype(
        np.int8
    )

    # ------------------------------------------------------------------
    # Save prediction artifact.
    # ------------------------------------------------------------------

    prediction_path = (
        output_dir
        / "test_predictions.npz"
    )

    np.savez_compressed(
        prediction_path,
        logits=test_logits,
        probabilities=(
            test_probabilities.astype(
                np.float32
            )
        ),
        predictions=test_predictions,
        labels=(
            test_labels.astype(
                np.int8
            )
        ),
    )

    # ------------------------------------------------------------------
    # Save explicit robustness metadata.
    # ------------------------------------------------------------------

    metadata = {
        "experiment":
            "final_linear_v3_random_seed_robustness",

        "run_name":
            run_name,

        "base_scenario":
            base_scenario,

        "scenario_short_name":
            args.scenario,

        "seed":
            int(args.seed),

        "modalities":
            list(modalities),

        "reference_seed_42_used":
            False,

        "checkpoint_source":
            None,

        "resume":
            False,

        "fresh_epoch_1":
            True,

        "output_directory":
            str(output_dir),

        "dataset":
            str(config.DATASET_PATH),

        "expected_counts":
            dict(config.EXPECTED_COUNTS),

        "prediction_artifact":
            "test_predictions.npz",

        "prediction_threshold":
            float(config.DEFAULT_THRESHOLD),

        "tta":
            bool(config.USE_TTA),

        "checkpoint_selection":
            "validation_macro_f1_at_threshold_0.5_without_tta",

        "architecture":
            "final_linear_v3",

        "protocol_note":
            (
                "Only random seed, scenario, and output directory "
                "are intentionally changed relative to final_linear_v3. "
                "All model, preprocessing, optimization, loss, "
                "checkpoint-selection, and evaluation settings remain "
                "unchanged."
            ),
    }

    metadata_path = (
        output_dir
        / "robustness_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Final verification.
    # ------------------------------------------------------------------

    result_path = (
        output_dir
        / "final_results.json"
    )

    if not result_path.exists():
        raise RuntimeError(
            "\nTraining returned but final_results.json "
            "was not created:\n"
            f"  {result_path}"
        )

    print("\n" + "=" * 70)
    print(
        "ROBUSTNESS RUN COMPLETED"
    )
    print("=" * 70)

    print(
        f"Scenario:       {base_scenario}"
    )

    print(
        f"Seed:            {args.seed}"
    )

    print(
        f"Results:         {result_path}"
    )

    print(
        f"Predictions:     {prediction_path}"
    )

    print(
        f"Metadata:        {metadata_path}"
    )

    print(
        "\nReference seed 42 was not retrained or loaded."
    )

    print(
        "Training started from epoch 1."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()