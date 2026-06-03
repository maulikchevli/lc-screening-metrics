#!/usr/bin/env python3
"""Reliability diagrams (equal-width predicted-probability bins) for the
three imaging models on their un-calibrated sigmoid outputs. This is the
canonical calibration plot — equal-width bins make miscalibration in the
high-predicted tail directly visible.

PLCOm omitted (operates on a different risk scale and would compress the
imaging-model curves).

Output:
  eval_output/reliability_diagrams.pdf
  eval_output/reliability_diagrams.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import _include, OUTPUT_DIR

# (paper label, score-column template, color, marker)
MODELS = [
    ("LungEvaty", "risk_score",        "#1f3a93", "o"),
    ("Pillar-0",  "sybil_risk_{y}",    "#e07b00", "s"),
    ("Sybil",     "sv1r_risk_{y}",     "#555555", "D"),
]
YEARS = (1, 3, 6)
N_BINS = 10


def equal_width_bins(scores, labels, n_bins=N_BINS):
    """Equal-width bins on [0, 1]. Returns (mean_pred, mean_obs, n) per bin
    that has ≥ 2 scans."""
    bins = np.linspace(0, 1, n_bins + 1)
    out_c, out_f, out_n = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (scores >= lo) & (scores < hi) if i < n_bins - 1 \
            else (scores >= lo) & (scores <= hi)
        if m.sum() < 2:
            continue
        out_c.append(float(np.mean(scores[m])))
        out_f.append(float(np.mean(labels[m])))
        out_n.append(int(m.sum()))
    return np.array(out_c), np.array(out_f), np.array(out_n)


def main():
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=False)

    for ax, year in zip(axes, YEARS):
        fu = year - 1
        include = np.array([_include(t, e, fu)[0]
                            for t, e in zip(comp["time_at_event"],
                                            comp["gold"])])
        labels = np.array([_include(t, e, fu)[1]
                           for t, e in zip(comp["time_at_event"],
                                           comp["gold"])])
        keep = include & ~np.isnan(comp["risk_score"].values.astype(float))
        sub_labels = labels[keep].astype(int)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.7, zorder=1,
                label="Perfect calibration")

        for paper, tmpl, color, marker in MODELS:
            col = tmpl.format(y=year)
            scores = comp[col].values.astype(float)[keep]
            valid = ~np.isnan(scores)
            sc = np.clip(scores[valid], 1e-7, 1 - 1e-7)
            lab = sub_labels[valid]
            mp, mo, _ = equal_width_bins(scores[valid], lab)
            brier = brier_score_loss(lab, sc)
            ax.plot(mp, mo, color=color, marker=marker, linestyle="-",
                    linewidth=1.4, markersize=6,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.5,
                    label=f"{paper} (Brier={brier:.4f})",
                    zorder=3)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("Predicted probability")
        if year == 1:
            ax.set_ylabel("Observed event rate")
        ax.set_title(f"Year {year}", fontsize=10)
        ax.grid(True, which="major", alpha=0.3, linewidth=0.5, zorder=0)
        ax.tick_params(which="both", direction="in", length=3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(loc="upper left", fontsize=8, frameon=False)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.92,
                        bottom=0.13, wspace=0.20)
    for ext in ("pdf", "png"):
        path = os.path.join(OUTPUT_DIR, f"reliability_diagrams.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
