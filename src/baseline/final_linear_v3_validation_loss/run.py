"""Independent V3 rerun for recording validation loss."""

from pathlib import Path

from src.baseline.final_linear_v3 import config
from src.baseline.final_linear_v3.train import main


def run():
    original_root = config.RESULT_ROOT

    new_root = Path(
        "results/baseline/final_linear_v3_validation_loss"
    ).resolve()

    if new_root == Path(original_root).resolve():
        raise RuntimeError("Output root must differ from original.")

    config.RESULT_ROOT = new_root

    print("=" * 70)
    print("V3 VALIDATION LOSS RERUN")
    print(f"Scenario: {config.ACTIVE_SCENARIO}")
    print(f"Seed: {config.SEED}")
    print(f"Original output root: {original_root}")
    print(f"New output root: {new_root}")
    print("=" * 70)

    main()


if __name__ == "__main__":
    run()
