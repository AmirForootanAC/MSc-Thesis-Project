from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Figure 3 — Overall Performance Across the Seven Configurations
# ============================================================

# Output directory
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Final test results from final_linear_v3
configurations = [
    "Photograph-only",
    "Radiograph-only",
    "Clinical Text",
    "Photograph + Radiograph",
    "Photograph + Text",
    "Radiograph + Text",
    "Full Multimodal",
]

macro_f1 = np.array([
    0.5422,
    0.5329,
    0.7990,
    0.5250,
    0.8038,
    0.8128,
    0.8377,
])

auroc = np.array([
    0.8598,
    0.8622,
    0.9660,
    0.8594,
    0.9690,
    0.9629,
    0.9596,
])


# Plot settings
x = np.arange(len(configurations))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 6.5))

bars_f1 = ax.bar(
    x - width / 2,
    macro_f1,
    width,
    label="Macro-F1",
)

bars_auroc = ax.bar(
    x + width / 2,
    auroc,
    width,
    label="AUROC",
)


# Axis labels and title
ax.set_ylabel("Score", fontsize=12)
ax.set_xlabel("Modality Configuration", fontsize=12)

ax.set_title(
    "Overall Performance Across the Seven Modality Configurations",
    fontsize=14,
    pad=12,
)

ax.set_xticks(x)
ax.set_xticklabels(
    configurations,
    rotation=25,
    ha="right",
)

ax.set_ylim(0, 1.05)

ax.set_yticks(np.arange(0, 1.01, 0.1))

ax.legend(
    loc="upper left",
    frameon=True,
)


# Add numerical values above each bar
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


add_value_labels(bars_f1)
add_value_labels(bars_auroc)


# Improve layout
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()


# Save high-resolution outputs
svg_path = OUTPUT_DIR / "figure_3_overall_performance.svg"
png_path = OUTPUT_DIR / "figure_3_overall_performance.png"

fig.savefig(svg_path, bbox_inches="tight")
fig.savefig(png_path, dpi=600, bbox_inches="tight")

plt.close(fig)

print(f"Saved: {svg_path}")
print(f"Saved: {png_path}")