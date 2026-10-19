import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ============================================================
# Paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.final_linear_v3 import config, train


# Display/model scenario -> actual directory containing its results.
# The image_xray results are stored under xray_only.
SCENARIOS = {
    "text_only": {
        "directory": "text_only",
        "title": "Clinical Text",
    },
    "image_xray": {
        "directory": "xray_only",
        "title": "Photograph + Radiograph",
    },
    "full_multimodal": {
        "directory": "full_multimodal",
        "title": "Full Multimodal",
    },
}

RESULT_ROOT = PROJECT_ROOT / "results" / "baseline" / "final_linear_v3"
FIGURE_ROOT = PROJECT_ROOT / "results" / "figures"
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

LOSS_RESULTS_PATH = FIGURE_ROOT / "validation_loss_from_checkpoints.json"
OUTPUT_PLOT = FIGURE_ROOT / "c10_learning_curves_macro_f1_loss_4k.png"

EVAL_BATCH_SIZE = 1
NUM_WORKERS = 0


# ============================================================
# Persistent results
# ============================================================

def read_loss_results():
    if not LOSS_RESULTS_PATH.exists():
        return {}

    try:
        data = json.loads(LOSS_RESULTS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        print("WARNING: Could not read previous loss results; starting empty.")
        return {}


def save_loss_results(data):
    LOSS_RESULTS_PATH.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


# ============================================================
# History and checkpoints
# ============================================================

def read_history(scenario_dir):
    history_path = scenario_dir / "history.json"

    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        latest_path = scenario_dir / "checkpoints" / "latest.pt"
        if not latest_path.exists():
            raise FileNotFoundError(
                f"No history.json or latest.pt in {scenario_dir}"
            )
        checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=False,
        )
        history = checkpoint.get("history", [])

    if not history:
        raise RuntimeError(f"Training history is empty: {scenario_dir}")

    # Verify that the fields required for plotting are available.
    required = {
        "epoch",
        "train_loss",
        "train_macro_f1",
        "validation_macro_f1",
    }
    missing = required - set(history[0])
    if missing:
        raise KeyError(
            f"Missing history fields in {scenario_dir}: {sorted(missing)}"
        )

    return sorted(history, key=lambda item: int(item["epoch"]))


def available_checkpoints(scenario_dir, history):
    checkpoint_dir = scenario_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints = {}

    for path in checkpoint_dir.glob("epoch_*.pt"):
        try:
            epoch = int(path.stem.split("_")[-1])
            checkpoints[epoch] = path
        except ValueError:
            continue

    latest_path = checkpoint_dir / "latest.pt"
    if latest_path.exists():
        checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=False,
        )
        latest_epoch = checkpoint.get("epoch")

        if latest_epoch is not None:
            latest_epoch = int(latest_epoch)
            # Prefer the epoch-specific file if it exists.
            checkpoints.setdefault(latest_epoch, latest_path)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    return sorted(checkpoints.items())


# ============================================================
# Validation loss: inference only, no training
# ============================================================

@torch.no_grad()
def evaluate_validation_loss(model, loader, criterion, device, scenario, epoch):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for raw_batch in tqdm(
        loader,
        desc=f"{scenario} | validation loss, epoch {epoch:03d}",
        leave=False,
    ):
        batch = train.move_batch_to_device(raw_batch, device)
        labels = batch["labels"]
        logits = train.model_forward(model, batch)

        if logits.shape != labels.shape:
            raise RuntimeError(
                f"Shape mismatch: logits={tuple(logits.shape)}, "
                f"labels={tuple(labels.shape)}"
            )

        loss = criterion(logits, labels)
        batch_size = labels.shape[0]

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Validation loader is empty.")

    return total_loss / total_samples


def evaluate_scenario(scenario, directory_name):
    print("\n" + "=" * 76)
    print(f"VALIDATION LOSS RECOMPUTATION: {scenario}")
    print(f"Source directory: {directory_name}")
    print("=" * 76)

    scenario_dir = RESULT_ROOT / directory_name
    history = read_history(scenario_dir)
    checkpoints = available_checkpoints(scenario_dir, history)

    # Verify that the directory metadata matches the requested architecture.
    metadata_path = scenario_dir / "final_results.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        recorded_scenario = metadata.get("scenario")
        recorded_modalities = metadata.get("modalities")
        expected_modalities = config.SCENARIOS[scenario]

        if recorded_scenario != scenario:
            raise RuntimeError(
                f"Scenario mismatch in {metadata_path}: "
                f"expected {scenario!r}, got {recorded_scenario!r}"
            )

        if list(recorded_modalities or []) != list(expected_modalities):
            raise RuntimeError(
                f"Modality mismatch in {metadata_path}: "
                f"expected {list(expected_modalities)}, got {recorded_modalities}"
            )

    # Match the architecture to the intended scenario.
    config.ACTIVE_SCENARIO = scenario
    config.ACTIVE_MODALITIES = config.SCENARIOS[scenario]

    # Checkpoint weights replace initialization weights; no pretrained
    # image download is needed.
    config.PRETRAINED_IMAGE_ENCODER = False
    config.BATCH_SIZE = EVAL_BATCH_SIZE
    config.NUM_WORKERS = NUM_WORKERS
    config.PIN_MEMORY = torch.cuda.is_available()

    device = train.get_device()
    print(f"Device: {device}")
    print(f"Modalities: {config.ACTIVE_MODALITIES}")
    print(f"Available checkpoint epochs: {[epoch for epoch, _ in checkpoints]}")

    train_dataset, validation_dataset, _ = train.build_datasets()
    validation_loader = train.make_loader(validation_dataset, shuffle=False)
    criterion = train.build_criterion(train_dataset, device)
    model = train.build_model().to(device)

    all_results = read_loss_results()
    scenario_results = all_results.setdefault(scenario, {})

    try:
        for epoch, checkpoint_path in checkpoints:
            print(f"\nLoading checkpoint: {checkpoint_path}")

            checkpoint = train.load_checkpoint(model, checkpoint_path, device)
            checkpoint_epoch = int(checkpoint.get("epoch", epoch))

            # Do not silently associate a checkpoint with a different epoch.
            if checkpoint_epoch != epoch:
                raise RuntimeError(
                    f"Checkpoint epoch mismatch: file epoch={epoch}, "
                    f"stored epoch={checkpoint_epoch}"
                )

            # Reuse a previously calculated value for the same checkpoint.
            previous = scenario_results.get(str(checkpoint_epoch))
            if (
                previous
                and previous.get("checkpoint") == checkpoint_path.name
                and "validation_loss" in previous
            ):
                print(
                    f"Already evaluated: validation loss = "
                    f"{previous['validation_loss']:.6f}; preserving saved value."
                )
                continue

            val_loss = evaluate_validation_loss(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                scenario=scenario,
                epoch=checkpoint_epoch,
            )

            scenario_results[str(checkpoint_epoch)] = {
                "epoch": checkpoint_epoch,
                "checkpoint": checkpoint_path.name,
                "validation_loss": float(val_loss),
            }

            save_loss_results(all_results)
            print(
                f"Epoch {checkpoint_epoch:03d} | "
                f"Validation Loss: {val_loss:.6f}"
            )

    finally:
        del model, criterion, validation_loader
        del train_dataset, validation_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return history


# ============================================================
# Publication-quality 4K plot
# ============================================================

def make_plot(all_histories, all_losses):
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "figure.titlesize": 20,
        "lines.linewidth": 2.6,
        "savefig.facecolor": "white",
    })

    # Exact 4K UHD dimensions: 19.2 x 10.8 inches at 200 DPI.
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(19.2, 10.8),
        dpi=200,
    )

    for row, (scenario, settings) in enumerate(SCENARIOS.items()):
        history = all_histories[scenario]
        losses = all_losses.get(scenario, {})
        title = settings["title"]

        epochs = [int(item["epoch"]) for item in history]
        train_f1 = [float(item["train_macro_f1"]) for item in history]
        val_f1 = [float(item["validation_macro_f1"]) for item in history]
        train_loss = [float(item["train_loss"]) for item in history]

        best_index = int(np.argmax(val_f1))
        best_epoch = epochs[best_index]
        best_f1 = val_f1[best_index]

        ax_f1 = axes[row, 0]
        ax_loss = axes[row, 1]

        # ----------------------------------------------------
        # Macro-F1
        # ----------------------------------------------------
        ax_f1.plot(
            epochs, train_f1,
            marker="o", markersize=4,
            label="Training Macro-F1",
        )
        ax_f1.plot(
            epochs, val_f1,
            marker="o", markersize=4,
            label="Validation Macro-F1",
        )

        ax_f1.axvline(
            best_epoch,
            linestyle="--",
            linewidth=2.0,
            color="black",
            alpha=0.85,
        )
        ax_f1.scatter(
            [best_epoch], [best_f1],
            s=110, zorder=5,
            edgecolors="black",
        )
        ax_f1.annotate(
            f"Best epoch: {best_epoch}\nMacro-F1 = {best_f1:.4f}",
            xy=(best_epoch, best_f1),
            xytext=(14, -58 if best_f1 > 0.88 else 14),
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                fc="white",
                ec="gray",
                alpha=0.95,
            ),
            arrowprops=dict(arrowstyle="->", linewidth=1.4),
        )

        ax_f1.set_title(f"{title} — Macro-F1", pad=12)
        ax_f1.set_xlabel("Epoch")
        ax_f1.set_ylabel("Macro-F1")
        ax_f1.set_ylim(0, 1.04)
        ax_f1.grid(True, alpha=0.28)
        ax_f1.legend(loc="lower right", framealpha=0.95)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------
        ax_loss.plot(
            epochs, train_loss,
            marker="o", markersize=4,
            label="Training Loss",
        )

        # Validation loss is only recoverable for saved checkpoints.
        val_epochs = sorted(
            int(key)
            for key, value in losses.items()
            if "validation_loss" in value
        )
        val_losses = [
            float(losses[str(epoch)]["validation_loss"])
            for epoch in val_epochs
        ]

        if val_epochs:
            ax_loss.plot(
                val_epochs,
                val_losses,
                linestyle="None",
                marker="D",
                markersize=8,
                label="Validation Loss (saved checkpoints)",
                zorder=5,
            )

            for epoch, value in zip(val_epochs, val_losses):
                ax_loss.annotate(
                    f"{value:.3f}",
                    (epoch, value),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=12,
                )

        ax_loss.axvline(
            best_epoch,
            linestyle="--",
            linewidth=2.0,
            color="black",
            alpha=0.85,
            label=f"Best Macro-F1 epoch: {best_epoch}",
        )

        # Label the same best Macro-F1 epoch on the loss panel too.
        y_min, y_max = ax_loss.get_ylim()
        label_y = y_max - 0.08 * (y_max - y_min)
        ax_loss.annotate(
            f"Best epoch: {best_epoch}\nMacro-F1 = {best_f1:.4f}",
            xy=(best_epoch, label_y),
            xytext=(12, 0),
            textcoords="offset points",
            fontsize=13,
            fontweight="bold",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.35",
                fc="white",
                ec="gray",
                alpha=0.95,
            ),
        )

        ax_loss.set_title(f"{title} — Loss", pad=12)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Class-weighted BCE Loss")
        ax_loss.grid(True, alpha=0.28)
        ax_loss.legend(loc="best", framealpha=0.95)

        # Keep epoch labels within the available history.
        for ax in (ax_f1, ax_loss):
            ax.set_xlim(min(epochs) - 0.7, max(epochs) + 0.7)

    # No overall title, as requested.
    fig.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    fig.savefig(
        OUTPUT_PLOT,
        dpi=200,
        facecolor="white",
        bbox_inches=None,
    )
    plt.close(fig)

    with Image.open(OUTPUT_PLOT) as image:
        if image.size != (3840, 2160):
            raise RuntimeError(
                f"Unexpected image size {image.size}; expected (3840, 2160)."
            )

    print(f"\n4K plot saved to:\n{OUTPUT_PLOT}")
    print("Image dimensions: 3840 x 2160 pixels")
    print(f"Loss values saved to:\n{LOSS_RESULTS_PATH}")


# ============================================================
# Main
# ============================================================

def main():
    all_histories = {}

    for scenario, settings in SCENARIOS.items():
        all_histories[scenario] = evaluate_scenario(
            scenario=scenario,
            directory_name=settings["directory"],
        )

    all_losses = read_loss_results()

    # Ensure the plot has a saved validation-loss point for each scenario.
    for scenario in SCENARIOS:
        if not all_losses.get(scenario):
            print(
                f"WARNING: No validation-loss checkpoint values available "
                f"for {scenario}."
            )

    make_plot(all_histories, all_losses)


if __name__ == "__main__":
    main()
