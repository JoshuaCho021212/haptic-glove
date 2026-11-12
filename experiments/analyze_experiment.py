"""Summarize the edge-tracing pilot study and render the README figures.

Usage:
    python experiments/analyze_experiment.py experiments/experiment_results.csv --out docs/images
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
COND_NAMES = {1: "Glove only", 2: "Glove + vibration", 3: "Glove + vibration + sound"}
ORDERS = {1: "1 > 2 > 3", 2: "2 > 3 > 1", 3: "3 > 1 > 2"}


def style(ax):
    ax.set_facecolor(SURF)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="docs/images")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

    df = pd.read_csv(args.csv)
    metrics = ["outside_pct", "exits", "mean_dist_mm", "duration_s", "in_band_pct"]
    per = df.groupby(["participant", "condition"])[metrics].mean()
    summ = per.groupby("condition").agg(["mean", "std"])

    print(f"{len(df)} trials, {(df.valid == 'yes').sum()} valid")
    for m in metrics:
        wide = per[m].unstack()
        p = stats.friedmanchisquare(*[wide[c] for c in (1, 2, 3)]).pvalue
        row = "  ".join(f"{COND_NAMES[c]}: {summ.loc[c, (m, 'mean')]:.1f} ± {summ.loc[c, (m, 'std')]:.1f}" for c in (1, 2, 3))
        print(f"{m:14s} {row}   Friedman p={p:.3f}")

    x = np.arange(3)
    labels = ["Glove only", "Glove +\nvibration", "Glove + vibration\n+ sound"]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200, facecolor=SURF)
    style(ax)
    w = 0.24
    for k, p in enumerate(sorted(df.participant.unique())):
        vals = [per.loc[(p, c), "outside_pct"] for c in (1, 2, 3)]
        xs = x + (k - 1) * (w + 0.02)
        ax.bar(xs, vals, w, color=SERIES[k], edgecolor=SURF, lw=1.5, label=f"Session {p} (order {ORDERS.get(p, '')})")
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, labels, color=INK)
    ax.set_ylabel("Time outside the zone (% of trial)", color=INK2)
    ax.set_ylim(0, max(60, per.outside_pct.max() * 1.15))
    ax.set_title("Time outside the safe zone, per session (mean of 3 trials)", loc="left", color=INK, fontsize=13, pad=14)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "results-outside-by-session.png", facecolor=SURF)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4.4), dpi=200, facecolor=SURF)
    specs = [("outside_pct", "Time outside zone (%)", "{:.1f}%"),
             ("mean_dist_mm", "Mean distance from edge (mm)", "{:.1f}"),
             ("duration_s", "Time per lap (s)", "{:.1f}")]
    for ax, (m, title, fmt) in zip(axs, specs):
        style(ax)
        mu, sd = summ[(m, "mean")].values, summ[(m, "std")].values
        ax.bar(x, mu, 0.55, color=SERIES[0], edgecolor=SURF, lw=1.5)
        ax.errorbar(x, mu, yerr=sd, fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4)
        for xi, mi, si in zip(x, mu, sd):
            ax.text(xi, mi + si + max(mu) * 0.03, fmt.format(mi), ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_xticks(x, ["Glove\nonly", "+ Vibration", "+ Vibration\n+ sound"], color=INK, fontsize=10)
        ax.set_title(title, loc="left", color=INK, fontsize=12)
        ax.set_ylim(0, max(mu + sd) * 1.25)
    fig.suptitle(f"Pilot results by condition (mean ± SD across {df.participant.nunique()} sessions, {len(df)} trials)",
                 x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "results-condition-summary.png", facecolor=SURF)

    rows = []
    for c in (1, 2, 3):
        sub = df[df.condition == c]
        rows.append([COND_NAMES[c],
                     f"{summ.loc[c, ('outside_pct', 'mean')]:.1f} ± {summ.loc[c, ('outside_pct', 'std')]:.1f}",
                     f"{summ.loc[c, ('exits', 'mean')]:.1f}",
                     f"{summ.loc[c, ('mean_dist_mm', 'mean')]:.1f}",
                     f"{summ.loc[c, ('in_band_pct', 'mean')]:.1f}",
                     f"{summ.loc[c, ('duration_s', 'mean')]:.1f}",
                     f"{(sub.valid == 'yes').sum()} / {len(sub)}"])
    cols = ["Condition", "Time outside (%)", "Exits / lap", "Mean dist. (mm)", "In 20 mm band (%)", "Lap time (s)", "Valid trials"]
    fig, ax = plt.subplots(figsize=(12, 2.3), dpi=200, facecolor=SURF)
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center", colLoc="center",
                  colWidths=[0.25, 0.14, 0.09, 0.13, 0.15, 0.11, 0.11])
    tb.auto_set_font_size(False)
    tb.set_fontsize(11)
    tb.scale(1, 2.0)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURF)
        cell.get_text().set_color(INK if r else INK2)
        if r == 0:
            cell.get_text().set_fontweight("bold")
        if c == 0 and r:
            cell.get_text().set_ha("left")
            cell.PAD = 0.04
    ax.set_title("Edge-tracing pilot study: summary by condition", loc="left", color=INK, fontsize=13, pad=6)
    fig.tight_layout()
    fig.savefig(out / "results-table.png", facecolor=SURF)
    print("figures saved to", out)


if __name__ == "__main__":
    main()
