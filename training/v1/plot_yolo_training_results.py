"""Portfolio figure: YOLOv8n-pose hand-keypoint training curves.

Reads only runs/pose/train-7/results.csv (150 epochs, 18 clips / 3,332 frames, HSV augmentation).
Columns used:
  metrics/mAP50-95(P), metrics/mAP50-95(B)   panel A
  train/pose_loss, val/pose_loss             panel B
  train/box_loss, val/box_loss               panel C
  metrics/mAP50(P), metrics/mAP50(B)         footer text only (saturated at 0.995)
Style matches SEGMENTATION/figures/plot_finetune_batchnorm.py.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "pose" / "train-7"
OUT = Path(__file__).resolve().parent / "yolo_training_results.png"
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


def peak(key, best=max):
    i = col[key].index(best(col[key]))
    return epochs[i], col[key][i]


fig = plt.figure(figsize=(12, 5.0), dpi=300, facecolor="white")
grid = fig.add_gridspec(1, 3, wspace=0.55, left=0.055, right=0.91, top=0.72, bottom=0.20)
axes = [fig.add_subplot(grid[0, i]) for i in range(3)]


def panel(ax, title, note, ylim, yticks, series):
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
    ax.set_xlabel("Training epoch")
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=22)
    ax.text(0, 1.02, note, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
    for i, (values, color, alpha, name) in enumerate(series):
        ax.plot(epochs, values, color=color, linewidth=1.6, alpha=alpha,
                solid_joinstyle="round", solid_capstyle="round")
        # Direct label at the line end: bold final value, then the series name.
        ax.annotate(f"{values[-1]:.2f}", (last, values[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=10, fontweight="bold", va="center", annotation_clip=False)
        ax.annotate(name, (last, values[-1]), xytext=(33, 0), textcoords="offset points",
                    color=INK_2, fontsize=8.5, va="center", annotation_clip=False)


def mosaic_marker(ax):
    ax.axvline(last - CLOSE_MOSAIC + 0.5, color=AXIS, linewidth=1, linestyle=(0, (2, 2)), zorder=0)


pose_best, box_best = peak("metrics/mAP50-95(P)"), peak("metrics/mAP50-95(B)")
panel(axes[0], "A   Accuracy on held-out clips",
      "mAP50-95; higher is better",
      (0, 1.0), [0, 0.25, 0.5, 0.75, 1.0],
      [(col["metrics/mAP50-95(B)"], BOX, 1.0, "hand box"),
       (col["metrics/mAP50-95(P)"], POSE, 1.0, "keypoints")])
axes[0].set_ylabel("mAP50-95")

panel(axes[1], "B   Keypoint (pose) loss",
      "Validation loss plateaus early and stays flat",
      (0, 9), [0, 2, 4, 6, 8],
      [(col["val/pose_loss"], POSE, 1.0, "validation"),
       (col["train/pose_loss"], POSE, 0.45, "training")])
axes[1].set_ylabel("Loss")
mosaic_marker(axes[1])

val_box_min = peak("val/box_loss", min)
panel(axes[2], "C   Bounding-box loss",
      f"Validation loss bottoms out at epoch {val_box_min[0]}, then drifts up",
      (0, 1.05), [0, 0.25, 0.5, 0.75, 1.0],
      [(col["val/box_loss"], BOX, 1.0, "validation"),
       (col["train/box_loss"], BOX, 0.45, "training")])
axes[2].set_ylabel("Loss")
mosaic_marker(axes[2])

fig.text(0.055, 0.965, "YOLOv8n-pose learns 21-point hand tracking that holds up on unseen video clips",
         color=INK, fontsize=14, fontweight="bold", va="top")
fig.text(0.055, 0.905,
         "COCO-pretrained YOLOv8n-pose fine-tuned for 150 epochs on 3,332 CVAT-labeled frames from 18 clips, with HSV colour augmentation.\n"
         "Validation is 346 frames from 2 clips held out entirely from training; lighter lines are training loss.",
         color=INK_2, fontsize=10, va="top")
fig.text(0.055, 0.035,
         f"Final epoch {last}: pose mAP50-95 {col['metrics/mAP50-95(P)'][-1]:.3f}, box mAP50-95 {col['metrics/mAP50-95(B)'][-1]:.3f} "
         f"(peaks {pose_best[1]:.3f} at epoch {pose_best[0]}, {box_best[1]:.3f} at epoch {box_best[0]}). "
         f"mAP50 is {col['metrics/mAP50(P)'][-1]:.3f} for both, so the stricter mAP50-95 is plotted.\n"
         f"Dotted line: mosaic augmentation off for the last {CLOSE_MOSAIC} epochs, causing the step in training loss.",
         color=MUTED, fontsize=8.5, va="bottom")

fig.savefig(OUT, dpi=300, facecolor="white")
print(f"wrote {OUT}")
print(f"pose mAP50-95 final {col['metrics/mAP50-95(P)'][-1]:.4f} peak {pose_best}  "
      f"box mAP50-95 final {col['metrics/mAP50-95(B)'][-1]:.4f} peak {box_best}")
