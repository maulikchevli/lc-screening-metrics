#!/usr/bin/env python3
"""3-panel PR figure for the Layer 2 section of the LDCT screening paper.

Reads eval_output/task3_pr_curves.json and produces:
  eval_output/pr_curves.pdf
  eval_output/pr_curves.png

Layout: three panels side-by-side (Year 1, 3, 6). Each panel has four
curves (LungEvaty raw, Pillar-0, Sybil v1.0 raw, PLCOm2012) with
operating points at flag rates 10/20/40/70% (cat-aggregated decile
boundaries) plus recall ≥ 95% (high-sensitivity safety regime).

Pillar-0 scores are the user reproduction (sybil_risk_* in comp_df is
patched; the JSON key changed from 'sybil15' → 'pillar0' in task3).
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

OUT_DIR = os.environ.get("SCREENING_METRICS_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_output"))
JSON_PATH = os.path.join(OUT_DIR, "task3_pr_curves.json")

MODEL_STYLE = {
    "lungevaty":   dict(label="LungEvaty (raw)",      color="#1f3a93",
                        linestyle="-",  lw=1.4),
    "pillar0":     dict(label="Pillar-0",             color="#e07b00",
                        linestyle="-",  lw=1.4),
    "sybilv1_raw": dict(label="Sybil v1.0 (raw)",     color="#555555",
                        linestyle="-",  lw=1.4),
    "plcom":       dict(label="PLCOm2012 (clinical)", color="#4ca64c",
                        linestyle="--", lw=1.4),
}
MODEL_ORDER = ["lungevaty", "pillar0", "sybilv1_raw", "plcom"]

OP_STYLE = {
    "FR10": dict(marker="o", label="flag 10%"),
    "FR20": dict(marker="s", label="flag 20%"),
    "FR40": dict(marker="D", label="flag 40%"),
    "FR70": dict(marker="v", label="flag 70%"),
    "R95":  dict(marker="^", label="recall ≥ 95%"),
}
OP_ORDER = ["FR10", "FR20", "FR40", "FR70", "R95"]


def prevalence_from_curve(curve):
    """Precision at threshold 0 (all-predicted-positive) ≈ prevalence."""
    t0 = min(curve, key=lambda p: p[0])
    return t0[2]


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), constrained_layout=False)

    Y_LOW = 0.005
    Y_HIGH = 1.0
    MAJOR_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    MINOR_TICKS = [0.003, 0.03, 0.3]

    def _fmt_tick(v, _):
        if v >= 0.1:
            return f"{v:.1f}"
        if abs(v - 0.05) < 1e-6:
            return "0.05"
        if abs(v - 0.02) < 1e-6:
            return "0.02"
        if abs(v - 0.01) < 1e-6:
            return "0.01"
        return f"{v:g}"

    for ax, year in zip(axes, (1, 3, 6)):
        prev = None
        for mname in MODEL_ORDER:
            key = f"{mname}_year{year}"
            if key not in data:
                continue
            entry = data[key]
            curve = entry["curve"]
            recall = np.array([pt[1] for pt in curve])
            prec = np.array([pt[2] for pt in curve], dtype=float)
            # clip non-positive precision so log axis doesn't choke
            prec_plot = np.where(prec > 0, prec, np.nan)
            style = MODEL_STYLE[mname]
            ax.plot(recall, prec_plot,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["lw"],
                    label=style["label"],
                    zorder=2)
            if prev is None:
                prev = prevalence_from_curve(curve)
            for tname in OP_ORDER:
                opstyle = OP_STYLE[tname]
                op = entry["operating_points"].get(tname)
                if not op or not op.get("achievable"):
                    continue
                r = op["recall"]; p = op["precision"]
                if p is None or p <= 0:
                    continue
                ax.plot(r, p,
                        marker=opstyle["marker"],
                        markersize=5.5,
                        markerfacecolor=style["color"],
                        markeredgecolor="white",
                        markeredgewidth=0.6,
                        linestyle="None",
                        zorder=4)

        # prevalence reference line + label at right edge
        if prev is not None:
            ax.axhline(prev, color="#888888", linestyle=":", linewidth=0.9,
                       zorder=1)
            ax.annotate(rf"$\hat{{\pi}}={prev*100:.2f}\%$",
                        xy=(1.0, prev), xycoords=("axes fraction", "data"),
                        xytext=(-3, 4), textcoords="offset points",
                        ha="right", va="bottom",
                        fontsize=8, color="#555555", zorder=5)

        mode = data[f"lungevaty_year{year}"]["mode"].upper()
        n = data[f"lungevaty_year{year}"]["n"]
        pi_str = f"$\\hat{{\\pi}} = {prev*100:.2f}\\%$" if prev is not None else ""
        ax.set_title(f"Year {year}\n{pi_str}  ({mode}, $N={n}$)", fontsize=10)
        ax.set_xlabel("Recall (sensitivity)")
        ax.set_xlim(0.0, 1.02)

        # log y-axis with explicit major & minor tick locations
        ax.set_yscale("log")
        ax.set_ylim(Y_LOW, Y_HIGH)
        ax.yaxis.set_major_locator(FixedLocator(MAJOR_TICKS))
        ax.yaxis.set_minor_locator(FixedLocator(MINOR_TICKS))
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_tick))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))

        # gridlines: major heavier, minor lighter
        ax.grid(True, which="major", alpha=0.35, linewidth=0.6, zorder=0)
        ax.grid(True, which="minor", alpha=0.2,  linewidth=0.5, zorder=0)
        ax.tick_params(which="both", direction="in", length=3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Precision (PPV)")

    # ---------------- shared legend below the figure ----------------
    model_handles = [
        Line2D([0], [0],
               color=MODEL_STYLE[m]["color"],
               linestyle=MODEL_STYLE[m]["linestyle"],
               linewidth=1.6,
               label=MODEL_STYLE[m]["label"])
        for m in MODEL_ORDER
    ]
    op_handles = [
        Line2D([0], [0],
               marker=OP_STYLE[t]["marker"],
               color="black",
               markerfacecolor="black",
               markeredgecolor="white",
               markeredgewidth=0.6,
               markersize=6,
               linestyle="None",
               label=OP_STYLE[t]["label"])
        for t in OP_ORDER
    ]

    fig.subplots_adjust(left=0.07, right=0.985, top=0.85,
                        bottom=0.28, wspace=0.30)
    leg1 = fig.legend(handles=model_handles,
                      loc="lower center",
                      bbox_to_anchor=(0.27, 0.005),
                      ncol=2, frameon=False, handlelength=2.4,
                      columnspacing=1.4)
    fig.add_artist(leg1)
    fig.legend(handles=op_handles,
               loc="lower center",
               bbox_to_anchor=(0.75, 0.005),
               ncol=5, frameon=False, handlelength=1.5,
               columnspacing=1.0, title="Operating points",
               title_fontsize=9)

    pdf_path = os.path.join(OUT_DIR, "pr_curves.pdf")
    png_path = os.path.join(OUT_DIR, "pr_curves.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
