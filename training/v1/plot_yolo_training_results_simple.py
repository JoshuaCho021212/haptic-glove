"""Portfolio figure (plain-language cut): YOLOv8n-pose hand-tracking training curves.

A simplified companion to plot_yolo_training_results.py for a general audience:
two panels instead of three (bounding-box loss dropped; box accuracy stays as a
line in panel A), plain-language labels, and a shorter footnote. Same data,
same colours (orange = box, blue = keypoints).

Reads only runs/pose/train-7/results.csv.
Style matches SEGMENTATION/figures/plot_finetune_batchnorm_simple.py.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "pose" / "train-7"
OUT = Path(__file__).resolve().parent / "yolo_training_results_simple.png"
CLOSE_MOSAIC = 10  # args.yaml: mosaic augmentation switched off for the last 10 epochs

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BOX = "#eb6834"
POSE = "#2a78d6"

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 1,
})

with open(RUN / "results.csv", newline="") as f:
    rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]
col = {k: [r[k] for r in rows] for k in rows[0]}
epochs = [int(e) for e in col["epoch"]]
last = epochs[-1]


def peak(key):
    i = col[key].index(max(col[key]))
    return epochs[i], col[key][i]


fig = plt.figure(figsize=(10.5, 5.0), dpi=300, facecolor="white")
grid = fig.add_gridspec(1, 2, wspace=0.62, left=0.075, right=0.87, top=0.66, bottom=0.26)
axes = [fig.add_subplot(grid[0, i]) for i in range(2)]


def panel(ax, title, note, ylim, yticks, ylabel, series):
    """series: list of (values, color, alpha, name), upper line first."""
    ax.set_facecolor("white")
    ax.set_xlim(1, last)
    ax.set_xticks([1, 25, 50, 75, 100, 125, 150])
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=6)
    ax.set_xlabel("Training round")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=30)
    ax.text(0, 1.035, note, transform=ax.transAxes, color=INK_2, fontsize=10, va="bottom")
    for values, color, alpha, name in series:
        ax.plot(epochs, values, color=color, linewidth=1.6, alpha=alpha,
                solid_joinstyle="round", solid_capstyle="round")
        # Direct label at the line end: bold final value, then the series name.
        ax.annotate(f"{values[-1]:.2f}", (last, values[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=10, fontweight="bold", va="center", annotation_clip=False)
        ax.annotate(name, (last, values[-1]), xytext=(33, 0), textcoords="offset points",
                    color=INK_2, fontsize=9, va="center", annotation_clip=False)


panel(axes[0], "How well it tracks hands it hasn't seen",
      "Accuracy on 2 held-out video clips; higher is better",
      (0, 1.0), [0, 0.25, 0.5, 0.75, 1.0], "Accuracy",
      [(col["metrics/mAP50-95(B)"], BOX, 1.0, "whole hand"),
       (col["metrics/mAP50-95(P)"], POSE, 1.0, "21 hand points")])

panel(axes[1], "How training progressed",
      "Error on new clips levels off early and stays flat",
      (0, 9), [0, 2, 4, 6, 8], "Error",
      [(col["val/pose_loss"], POSE, 1.0, "new clips"),
       (col["train/pose_loss"], POSE, 0.45, "training clips")])

fig.text(0.075, 0.955, "The hand tracker learned quickly and holds up on new videos",
         color=INK, fontsize=16, fontweight="bold", va="top")
fig.text(0.075, 0.885,
         "A small AI model learned to place 21 points on a hand, trained on 18 video clips "
         "and tested on 2 it never saw.",
         color=INK_2, fontsize=11, va="top")

pose_best, box_best = peak("metrics/mAP50-95(P)"), peak("metrics/mAP50-95(B)")
fig.text(0.075, 0.03,
         f"YOLOv8n-pose, 150 epochs on 3,332 labeled frames. Accuracy is mAP50-95 at the final epoch "
         f"(peaks: points {pose_best[1]:.2f}, hand {box_best[1]:.2f}).\n"
         f"Error is keypoint loss; the dip in training error near the end is an augmentation change "
         f"for the last {CLOSE_MOSAIC} epochs.",
         color=MUTED, fontsize=7.5, va="bottom", linespacing=1.5)

fig.savefig(OUT, dpi=300, facecolor="white")
print(f"wrote {OUT}")
print(f"pose mAP50-95 final {col['metrics/mAP50-95(P)'][-1]:.4f} peak {pose_best}  "
      f"box mAP50-95 final {col['metrics/mAP50-95(B)'][-1]:.4f} peak {box_best}")
