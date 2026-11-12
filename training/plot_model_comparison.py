"""Model comparison figure for the README.

Values are the validation results printed by train_zone.py, train_zone_dark.py and export_small.py
(validation set = held-out clips edges_trace and led_on_center; "dark" = the same frames at 0.125x brightness).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
NORMAL, DARK = "#2a78d6", "#eb6834"

MODELS = ["v1 model\n(close-up data)", "Zone fine-tune\n640 px", "+ dark augment.\n640 px", "+ dark augment.\n416 px (deployed)"]
BOX = {"normal": [0.001, 0.918, 0.921, 0.961], "dark": [None, 0.643, 0.916, 0.936]}
POSE = {"normal": [0.000, 0.321, 0.394, 0.393], "dark": [None, 0.222, 0.303, 0.486]}


def panel(ax, data, title):
    ax.set_facecolor(SURF)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK2)
    x = np.arange(len(MODELS))
    w = 0.36
    for k, (key, col, lab) in enumerate((("normal", NORMAL, "Normal lighting"), ("dark", DARK, "Dark (0.125x)"))):
        for xi, v in zip(x + (k - 0.5) * (w + 0.03), data[key]):
            if v is None:
                ax.text(xi, 0.02, "not\nmeasured", ha="center", va="bottom", fontsize=8, color=INK2)
                continue
            ax.bar(xi, v, w, color=col, edgecolor=SURF, lw=1.5, label=lab if xi == x[1] + (k - 0.5) * (w + 0.03) else None)
            ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, MODELS, color=INK, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_title(title, loc="left", color=INK, fontsize=12)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), dpi=200, facecolor=SURF)
    panel(axs[0], BOX, "Hand detection (box mAP50)")
    panel(axs[1], POSE, "Keypoints (pose mAP50)")
    axs[0].legend(frameon=False, loc="upper left", labelcolor=INK, fontsize=10)
    fig.suptitle("YOLOv8n-pose on the real work-zone view (held-out clips)", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig("docs/images/training-results.png", facecolor=SURF)


if __name__ == "__main__":
    main()
