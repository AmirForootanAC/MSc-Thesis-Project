
from pathlib import Path
import json

import matplotlib.pyplot as plt


# Repository root
ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = ROOT / "results" / "baseline" / "final_linear_v3"
OUTPUT_DIR = ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    ("text_only", "Text-only"),
    ("image_xray", "Image + X-ray"),
    ("full_multimodal", "Full Multimodal"),
]


def load_history(scenario):
    path = RESULTS_DIR / scenario / "history.json"

    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        raise ValueError(f"History is empty: {path}")

    return history


def plot_learning_curves():
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12, 12),
        constrained_layout=True,
    )

    for row, (scenario, title) in enumerate(SCENARIOS):
        history = load_history(scenario)

        epochs = [item["epoch"] for item in history]

        train_f1 = [item["train_macro_f1"] for item in history]
        val_f1 = [item["validation_macro_f1"] for item in history]
        train_loss = [item["train_loss"] for item in history]

        # Left: Macro-F1
        ax_f1 = axes[row, 0]
        ax_f1.plot(
            epochs, train_f1,
            marker="o", markersize=3,
            linewidth=1.8, label="Training Macro-F1",
        )
        ax_f1.plot(
            epochs, val_f1,
            marker="o", markersize=3,
            linewidth=1.8, label="Validation Macro-F1",
        )

        ax_f1.set_title(f"{title}: Macro-F1")
        ax_f1.set_xlabel("Epoch")
        ax_f1.set_ylabel("Macro-F1")
        ax_f1.set_ylim(0, 1.02)
        ax_f1.grid(True, alpha=0.3)
        ax_f1.legend(frameon=False)

        # Right: Training loss
        ax_loss = axes[row, 1]
        ax_loss.plot(
            epochs, train_loss,
            marker="o", markersize=3,
            linewidth=1.8, label="Training Loss",
        )

        ax_loss.set_title(f"{title}: Training Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Weighted BCE Loss")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(frameon=False)

    fig.suptitle(
        "Learning Curves Across Three Model Configurations",
        fontsize=15,
    )

    output_path = OUTPUT_DIR / "c10_learning_curves_macro_f1_loss.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    plot_learning_curves()
