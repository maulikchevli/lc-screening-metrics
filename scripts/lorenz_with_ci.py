#!/usr/bin/env python3
"""Lorenz / concentration curves with 95% bootstrap CI bands.

Nature Medicine-style figures:
  - Helvetica/Arial sans-serif, 7pt body, 6pt ticks/legend.
  - Wong (2011) colorblind-safe palette.
  - Two-spine layout (bottom+left), outward ticks, no embedded title.
  - PDF (vector, TrueType-embedded) + 600 dpi PNG output per figure.
  - CI bands de-emphasized (alpha=0.10) so point-estimate lines dominate.
  - Inline annotation showing n / cancers / prevalence.

Year-1 outputs (single time point, requested):
  eval_output/lorenz_y1_linear_ci.{pdf,png}         (single panel, 89 mm)
  eval_output/lorenz_y1_semilog_ci.{pdf,png}        (single panel, 89 mm)
  eval_output/lorenz_y1_composite_ci.{pdf,png}      (2-panel a/b, 180 mm)

Multi-horizon outputs (kept for paper compatibility):
  eval_output/lorenz_multihorizon_ci.{pdf,png}      (3-panel y1/y3/y6, 180 mm)

Long-format CSVs:
  eval_output/lorenz_y1_with_ci.csv
  eval_output/lorenz_multihorizon_ci.csv
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import (
    _include, load_participant, PARTICIPANT_CSV, OUTPUT_DIR,
)
from decile_stratification_devfit import (
    fit_lung_platt, fit_plcom_platt, fit_pillar_platt,
    attach_user_pillar, build_plcom_dev, _logit, _plcom_extra_cols,
)
from uspstf_plcom import plcom2012_risk

N_BOOT = 1000
RNG_SEED = 42
N_X = 200
LUNGRADS_CSV = os.path.join(OUTPUT_DIR, "nlst_lungrads_pinsky.csv")
LR_ORDER = {"4X": 6, "4B": 5, "4A": 4, "3": 3, "2": 2, "1": 1}

# ---- Nature Medicine style ----
def setup_nm_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        # Nimbus Sans is a metric-compatible Helvetica clone available in
        # most Linux font stacks; falls back to DejaVu Sans if absent.
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 7,
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "legend.columnspacing": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# Wong (Nat Methods 8:441, 2011) colorblind-safe palette
COLORS = {
    "LungEvaty": "#D55E00",  # vermillion
    "Pillar-0":  "#009E73",  # bluish green
    "Sybil":     "#0072B2",  # blue
    "PLCOm2012": "#E69F00",  # orange
    "Lung-RADS": "#000000",  # black
}
LINESTYLES = {
    "LungEvaty": "-",
    "Pillar-0":  "-",
    "Sybil":     "-",
    "PLCOm2012": (0, (5, 2)),    # dashed (redundant channel)
    "Lung-RADS": (0, (1, 1.4)),  # dotted (redundant channel)
}
MODEL_ORDER = ["LungEvaty", "Pillar-0", "Sybil", "PLCOm2012", "Lung-RADS"]
CI_ALPHA = 0.12


def _pct_int(x, _pos):
    if x <= 0: return ""
    if x >= 0.1: return f"{int(round(x*100))}"
    if x >= 0.01: return f"{x*100:.0f}"
    if x >= 0.001: return f"{x*100:.1f}"
    return f"{x*100:.2f}"


def attach_lr(comp):
    lr = pd.read_csv(LUNGRADS_CSV)
    lr["pid"] = lr["pid"].astype(str)
    lr["study_yr"] = lr["study_yr"].astype(int)
    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    comp["study_yr"] = comp["screen_timepoint"].astype(int)
    return comp.merge(lr[["pid", "study_yr", "lungrads_v10"]],
                      on=["pid", "study_yr"], how="left")


def build_scores_all_years(comp):
    print("[setup] Fitting calibrators…")
    lung_platt = fit_lung_platt()
    part = load_participant(PARTICIPANT_CSV)
    dev_plcom = build_plcom_dev(part)
    plcom_platt = fit_plcom_platt(dev_plcom)
    pillar_platt = fit_pillar_platt()
    comp = attach_user_pillar(comp)
    base_cols = ["pid", "age", "race", "pkyr", "smokeyr", "cigsmok"]
    base_cols = [c for c in base_cols if c in part.columns]
    extras = _plcom_extra_cols(part)
    base = part[base_cols + extras].drop_duplicates("pid").copy()
    base["pid"] = base["pid"].astype(str)
    pid_to_feat = base.set_index("pid")
    le_vec = np.array([
        [float(v) for v in r.strip("[]").split(",")]
        for r in comp["cancer_risk"].values
    ])
    out = {}
    for year in [1, 3, 6]:
        le_raw = le_vec[:, year - 1]
        le_cal = lung_platt[year].predict_proba(
            _logit(le_raw).reshape(-1, 1))[:, 1]
        syb_cal = comp[f"sv1c_risk_{year}"].values.astype(float)
        pi_raw = comp[f"pillar_user_y{year}"].values.astype(float)
        mp = ~np.isnan(pi_raw)
        pi_cal = np.full(len(comp), np.nan)
        pi_cal[mp] = pillar_platt[year].predict_proba(
            _logit(pi_raw[mp]).reshape(-1, 1))[:, 1]
        plcom_raw = np.full(len(comp), np.nan)
        for i, pid in enumerate(comp["pid"].values):
            if pid in pid_to_feat.index:
                plcom_raw[i] = float(
                    plcom2012_risk(pid_to_feat.loc[pid], year=year))
        mm = ~np.isnan(plcom_raw)
        pl_cal = np.full(len(comp), np.nan)
        pl_cal[mm] = plcom_platt[year].predict_proba(
            _logit(plcom_raw[mm]).reshape(-1, 1))[:, 1]
        out[year] = {"LungEvaty": le_cal, "Sybil": syb_cal,
                     "Pillar-0": pi_cal, "PLCOm2012": pl_cal}
    return comp, out


def lorenz_curve_interp(scores, labels, x_grid):
    n = len(scores)
    if n == 0:
        return np.full_like(x_grid, np.nan, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = labels[order].astype(float)
    n_pos = y_sorted.sum()
    if n_pos == 0:
        return np.full_like(x_grid, np.nan, dtype=float)
    x_cum = np.arange(1, n + 1) / n
    y_cum = np.cumsum(y_sorted) / n_pos
    x_full = np.concatenate([[0.0], x_cum])
    y_full = np.concatenate([[0.0], y_cum])
    return np.interp(x_grid, x_full, y_full)


def _boot_worker(args):
    (year, model, scores, labels, pid_to_rows, pids_uniq, x_grid,
     n_boot, seed) = args
    rng = np.random.default_rng(seed)
    nan_mask = ~np.isnan(scores)
    sel_pt = np.where(nan_mask)[0]
    point = lorenz_curve_interp(scores[sel_pt], labels[sel_pt], x_grid)
    n_inc_pt = int(sel_pt.size)
    n_pos_pt = int(labels[sel_pt].sum())
    boot = np.zeros((n_boot, len(x_grid)))
    n_u = len(pids_uniq)
    for b in range(n_boot):
        sample = rng.choice(pids_uniq, size=n_u, replace=True)
        idx_all = np.concatenate([pid_to_rows[p] for p in sample])
        idx = idx_all[nan_mask[idx_all]]
        boot[b] = lorenz_curve_interp(scores[idx], labels[idx], x_grid)
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return year, model, point, lo, hi, n_inc_pt, n_pos_pt


def lr_anchor_flag_rates_full(lr_score, mask):
    n_cohort = int(mask.sum())
    out = {}
    for floor, lab in [(LR_ORDER["4X"], "LR≥4X"),
                       (LR_ORDER["4B"], "LR≥4B"),
                       (LR_ORDER["4A"], "LR≥4A"),
                       (LR_ORDER["3"],  "LR≥3")]:
        n_flag = int((lr_score[mask] >= floor).sum())
        out[lab] = n_flag / n_cohort if n_cohort else np.nan
    return out


def recall_at_lr_match(scores, labels, lr_score, mask_inc, lr_rank_floor):
    m = mask_inc & ~np.isnan(scores)
    if not m.any():
        return np.nan
    s = scores[m]
    y = labels[m].astype(int)
    lr = lr_score[m]
    n_flag = int((lr >= lr_rank_floor).sum())
    if n_flag == 0 or y.sum() == 0:
        return np.nan
    top = np.argsort(-s, kind="mergesort")[:n_flag]
    return int(y[top].sum()) / int(y.sum())


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def _add_anchor_markers(ax, x_grid, results, lr_4b_fr, lr_anchors,
                        which="full"):
    """Vertical guides at LR anchors. Labels rendered vertically (rotation=90)
    above the plot so adjacent anchors do not overlap at semi-log scale.

    which='full' shows all four LR anchors (semilog plot); 'minimal' shows
    only LR≥4B (linear plot).
    """
    if which == "full":
        labels_to_show = [("LR≥4X", lr_anchors.get("LR≥4X", np.nan)),
                          ("LR≥4B", lr_anchors.get("LR≥4B", np.nan)),
                          ("LR≥4A", lr_anchors.get("LR≥4A", np.nan)),
                          ("LR≥3",  lr_anchors.get("LR≥3",  np.nan))]
    else:
        labels_to_show = [("LR≥4B", lr_anchors.get("LR≥4B", np.nan))]
    for lab, fr in labels_to_show:
        if not np.isfinite(fr) or fr <= 0:
            continue
        ax.axvline(fr, color="#888888", lw=0.35, ls=(0, (2, 2)), alpha=0.7,
                   zorder=0.5)
        ax.text(fr, 1.01, lab, rotation=90, fontsize=5.4, color="#555555",
                va="bottom", ha="center", transform=ax.get_xaxis_transform())


def _plot_lorenz_panel(ax, year, results, lr_anchors, lr_4b_fr, x_grid,
                       scale, show_legend=True, n_meta=None,
                       anchors_which="full", show_no_disc_label=True):
    # CI bands first
    for m in MODEL_ORDER:
        r = results.get((year, m))
        if r is None:
            continue
        if scale == "semilog":
            xs = x_grid[1:]; lo = r["lo"][1:]; hi = r["hi"][1:]
        else:
            xs = x_grid; lo = r["lo"]; hi = r["hi"]
        ax.fill_between(xs, lo, hi, color=COLORS[m], alpha=CI_ALPHA,
                        linewidth=0, zorder=1)
    # No-discrimination diagonal
    if scale == "semilog":
        xref = np.logspace(np.log10(max(x_grid[1], 1e-4)), 0, 200)
        ax.plot(xref, xref, color="#555555", lw=0.55,
                dashes=(2, 2), alpha=0.7, zorder=1.5,
                label="No discrimination" if show_no_disc_label else None)
    else:
        ax.plot([0, 1], [0, 1], color="#555555", lw=0.55, dashes=(2, 2),
                alpha=0.7, zorder=1.5,
                label="No discrimination" if show_no_disc_label else None)
    # Model lines
    for m in MODEL_ORDER:
        r = results.get((year, m))
        if r is None:
            continue
        if scale == "semilog":
            xs = x_grid[1:]; ys = r["point"][1:]
        else:
            xs = x_grid; ys = r["point"]
        lw = 1.3 if m == "Lung-RADS" else 1.5
        ax.plot(xs, ys, color=COLORS[m], lw=lw, ls=LINESTYLES[m],
                label=m, zorder=2.5)
    # LR-anchored guides (and label LR4B)
    _add_anchor_markers(ax, x_grid, results, lr_4b_fr, lr_anchors,
                        which=anchors_which)
    # Axes formatting
    ax.set_ylim(0, 1.02)
    if scale == "semilog":
        ax.set_xscale("log")
        n_inc = n_meta[0] if n_meta else len(x_grid)
        ax.set_xlim(0.5 / max(n_inc, 1), 1)
        ax.set_xlabel("Scans flagged (%, log scale)")
    else:
        ax.set_xlim(0, 1)
        ax.set_xlabel("Scans flagged (%, linear scale)")
    ax.set_ylabel(f"Year-{year} cancers captured (%)")
    ax.xaxis.set_major_formatter(FuncFormatter(_pct_int))
    ax.yaxis.set_major_formatter(FuncFormatter(_pct_int))
    _style_axes(ax)
    if show_legend:
        leg = ax.legend(loc="lower right" if scale == "linear" else "upper left",
                        fontsize=5.8, handlelength=1.5,
                        borderpad=0.35, labelspacing=0.4)


def _legend_with_recall(ax, year, results, recall_by_model, scale,
                        loc=None):
    """Replace stock legend with a custom one that includes recall@LR≥4B
    as a small numeric column next to each model name."""
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for lab in labels:
        m = lab
        if m in recall_by_model and np.isfinite(recall_by_model[m]):
            new_labels.append(f"{m}  ({recall_by_model[m]*100:.1f}%)")
        else:
            new_labels.append(m)
    if loc is None:
        loc = "lower right" if scale == "linear" else "upper left"
    ax.legend(handles, new_labels, loc=loc, fontsize=5.8,
              handlelength=1.6, borderpad=0.35, labelspacing=0.4,
              title="Model  (recall @ LR≥4B)", title_fontsize=5.6)


def _add_meta_annotation(ax, n_inc, n_pos, scale, cohort_label=""):
    prev = n_pos / n_inc if n_inc else np.nan
    pre = f"{cohort_label}: " if cohort_label else ""
    txt = (f"{pre}n = {n_inc:,} scans · {n_pos} year-1 cancers · "
           f"prevalence = {prev*100:.2f}%")
    ax.text(0.0, -0.18, txt, transform=ax.transAxes,
            fontsize=5.6, color="#444444", ha="left", va="top")


def save_dual(fig, out_path_pdf, out_path_png):
    fig.savefig(out_path_pdf, format="pdf")
    fig.savefig(out_path_png, format="png", dpi=600)
    plt.close(fig)
    print(f"→ {out_path_pdf}")
    print(f"→ {out_path_png}")


def plot_single_year(year, scale, results, lr_anchors, x_grid, n_meta,
                     recall_by_model, out_stem, cohort_label=""):
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    lr_4b_fr = lr_anchors.get("LR≥4B", np.nan)
    anchors_which = "full" if scale == "semilog" else "minimal"
    _plot_lorenz_panel(ax, year, results, lr_anchors, lr_4b_fr, x_grid,
                       scale, show_legend=False, n_meta=n_meta,
                       anchors_which=anchors_which)
    _legend_with_recall(ax, year, results, recall_by_model, scale)
    _add_meta_annotation(ax, n_meta[0], n_meta[1], scale, cohort_label)
    save_dual(fig, f"{out_stem}.pdf", f"{out_stem}.png")


def plot_alltp_vs_t0_compare(year, scale, results_full, results_t0,
                             lr_anchors_full, lr_anchors_t0, x_grid,
                             meta_full, meta_t0, recall_full, recall_t0,
                             out_stem):
    """Side-by-side a/b: full cohort vs T0-only prevalence-round subset.

    Cohort identity + n / cancers / prevalence are folded into the sub-
    title so the bottom-of-axis meta annotation is omitted (it overlaps
    the x-axis label on the compact 7.2"-wide composite).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    lr_4b_full = lr_anchors_full.get("LR≥4B", np.nan)
    lr_4b_t0 = lr_anchors_t0.get("LR≥4B", np.nan)
    anchors_which = "full" if scale == "semilog" else "minimal"
    _plot_lorenz_panel(ax1, year, results_full, lr_anchors_full, lr_4b_full,
                       x_grid, scale, show_legend=False, n_meta=meta_full,
                       anchors_which=anchors_which)
    _legend_with_recall(ax1, year, results_full, recall_full, scale)
    _plot_lorenz_panel(ax2, year, results_t0, lr_anchors_t0, lr_4b_t0,
                       x_grid, scale, show_legend=False, n_meta=meta_t0,
                       anchors_which=anchors_which)
    _legend_with_recall(ax2, year, results_t0, recall_t0, scale)
    # Short cohort labels; n / cancers / prevalence belong in the LaTeX
    # caption. Naming makes the LR information asymmetry explicit.
    subs = [
        (ax1, "a", "Year 1, all timepoints (LR w/ prior)"),
        (ax2, "b", "Year 1, T0 only (single-scan, LR symmetric)"),
    ]
    for ax, lbl, sub in subs:
        ax.text(-0.18, 1.26, lbl, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", va="top", ha="left")
        ax.text(0.5, 1.14, sub, transform=ax.transAxes,
                fontsize=7.0, color="#222222", ha="center", va="bottom")
    fig.tight_layout(w_pad=2.0, rect=[0, 0, 1, 0.86])
    save_dual(fig, f"{out_stem}.pdf", f"{out_stem}.png")


def plot_composite_y1(year, results, lr_anchors, x_grid, n_meta,
                      recall_by_model, out_stem):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    lr_4b_fr = lr_anchors.get("LR≥4B", np.nan)
    # Panel (a): linear — no legend (panel b carries the shared legend).
    _plot_lorenz_panel(ax1, year, results, lr_anchors, lr_4b_fr, x_grid,
                       "linear", show_legend=False, n_meta=n_meta,
                       anchors_which="minimal")
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()
    # Panel (b): semi-log — recall@LR≥4B legend in lower-right (the upper-
    # left corner is occupied by the LR≥4X anchor label).
    _plot_lorenz_panel(ax2, year, results, lr_anchors, lr_4b_fr, x_grid,
                       "semilog", show_legend=False, n_meta=n_meta,
                       anchors_which="full")
    _legend_with_recall(ax2, year, results, recall_by_model, "semilog",
                        loc="lower right")
    # Panel labels
    for ax, lbl in [(ax1, "a"), (ax2, "b")]:
        ax.text(-0.18, 1.08, lbl, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", va="bottom", ha="left")
    _add_meta_annotation(ax1, n_meta[0], n_meta[1], "linear")
    fig.tight_layout(w_pad=2.0)
    save_dual(fig, f"{out_stem}.pdf", f"{out_stem}.png")


def plot_multihorizon(results, lr_anchors_by_year, x_grid, meta_by_year,
                      recall_by_year_model, out_stem):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=True)
    for ax, year, lbl in zip(axes, [1, 3, 6], ["a", "b", "c"]):
        n_meta = meta_by_year[year]
        lr_anchors = lr_anchors_by_year[year]
        lr_4b_fr = lr_anchors.get("LR≥4B", np.nan)
        _plot_lorenz_panel(ax, year, results, lr_anchors, lr_4b_fr, x_grid,
                           "semilog", show_legend=False, n_meta=n_meta,
                           anchors_which="full",
                           show_no_disc_label=(year == 1))
        recall_map = {m: recall_by_year_model.get((year, m), np.nan)
                      for m in MODEL_ORDER}
        # Per-panel legend with the horizon-specific recall@LR≥4B (the
        # values differ across horizons, so they belong on every panel).
        _legend_with_recall(ax, year, results, recall_map, "semilog")
        # The vertical LR-anchor labels extend up to ~y=1.10 transAxes;
        # place the sub-title above that, and the panel label one row
        # higher still.
        ax.text(-0.18, 1.32, lbl, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", va="top", ha="left")
        suff = " (complete-case)" if year == 6 else ""
        ax.text(0.5, 1.18,
                f"Year {year} · n = {n_meta[0]:,} · cancers = {n_meta[1]}{suff}",
                transform=ax.transAxes, fontsize=6.4, color="#222222",
                ha="center", va="bottom")
    fig.tight_layout(w_pad=1.5, rect=[0, 0, 1, 0.85])
    save_dual(fig, f"{out_stem}.pdf", f"{out_stem}.png")


def main():
    setup_nm_style()
    print("=" * 78)
    print(f"LORENZ CURVES — NM style (n_boot={N_BOOT}, paired)")
    print("=" * 78)
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    comp = attach_lr(comp)
    comp, year_to_scores = build_scores_all_years(comp)

    times = comp["time_at_event"].values
    golds = comp["gold"].values
    pids_all = comp["pid"].values
    lr_score_all = np.array([LR_ORDER.get(str(c), 0)
                             for c in comp["lungrads_v10"].astype(str).values],
                            dtype=float)

    n_total = len(comp)
    x_grid = np.concatenate([[0.0],
                             np.logspace(np.log10(1.0 / n_total), 0, N_X)])

    tasks = []
    meta_by_year = {}
    lr_anchors_by_year = {}
    recall_by_year_model = {}
    for year in [1, 3, 6]:
        fu = year - 1
        inc = np.array([_include(t, g, fu)[0] for t, g in zip(times, golds)])
        lbl = np.array([_include(t, g, fu)[1] for t, g in zip(times, golds)])
        idx_inc = np.where(inc)[0]
        n_inc = int(inc.sum())
        n_pos = int(lbl[inc].sum())
        meta_by_year[year] = (n_inc, n_pos)
        lr_anchors_by_year[year] = lr_anchor_flag_rates_full(
            lr_score_all, inc)
        p_inc = pids_all[idx_inc]
        pid_to_rows = {}
        for i, pp in enumerate(p_inc):
            pid_to_rows.setdefault(pp, []).append(i)
        pid_to_rows = {k: np.asarray(v) for k, v in pid_to_rows.items()}
        pids_uniq = np.array(sorted(pid_to_rows.keys()))
        seed_year = RNG_SEED + year * 1000
        for m, sc in year_to_scores[year].items():
            tasks.append((year, m, sc[idx_inc], lbl[idx_inc],
                          pid_to_rows, pids_uniq, x_grid, N_BOOT, seed_year))
            recall_by_year_model[(year, m)] = recall_at_lr_match(
                sc, lbl, lr_score_all, inc, LR_ORDER["4B"])
        lr_score_for_year = np.where(lr_score_all > 0,
                                     lr_score_all, np.nan).astype(float)
        tasks.append((year, "Lung-RADS",
                      lr_score_for_year[idx_inc], lbl[idx_inc],
                      pid_to_rows, pids_uniq, x_grid, N_BOOT, seed_year))
        recall_by_year_model[(year, "Lung-RADS")] = recall_at_lr_match(
            lr_score_for_year, lbl, lr_score_all, inc, LR_ORDER["4B"])

    print(f"[run] Bootstrap {len(tasks)} cells × {N_BOOT}…")
    results = {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(_boot_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            year, m, pt, lo, hi, n_inc, n_pos = f.result()
            results[(year, m)] = {"point": pt, "lo": lo, "hi": hi,
                                  "n_inc": n_inc, "n_pos": n_pos}
            print(f"  ✓ ({i:2}/{len(tasks)}) y{year} {m}: "
                  f"n_eff={n_inc}, cancers={n_pos}")

    # CSVs (unchanged structure)
    rows_y1 = []
    for m in MODEL_ORDER:
        r = results.get((1, m))
        if r is None:
            continue
        for i, x in enumerate(x_grid):
            rows_y1.append(dict(model=m, x=x,
                                y_point=r["point"][i],
                                y_lo=r["lo"][i], y_hi=r["hi"][i],
                                n_eff=r["n_inc"], n_pos=r["n_pos"]))
    pd.DataFrame(rows_y1).to_csv(
        os.path.join(OUTPUT_DIR, "lorenz_y1_with_ci.csv"), index=False)
    print("→ lorenz_y1_with_ci.csv")

    rows_mh = []
    for year in [1, 3, 6]:
        for m in MODEL_ORDER:
            r = results.get((year, m))
            if r is None:
                continue
            for i, x in enumerate(x_grid):
                rows_mh.append(dict(year=year, model=m, x=x,
                                    y_point=r["point"][i],
                                    y_lo=r["lo"][i], y_hi=r["hi"][i],
                                    n_eff=r["n_inc"], n_pos=r["n_pos"]))
    pd.DataFrame(rows_mh).to_csv(
        os.path.join(OUTPUT_DIR, "lorenz_multihorizon_ci.csv"), index=False)
    print("→ lorenz_multihorizon_ci.csv")

    recall_y1 = {m: recall_by_year_model.get((1, m), np.nan)
                 for m in MODEL_ORDER}
    plot_single_year(1, "linear", results, lr_anchors_by_year[1], x_grid,
                     meta_by_year[1], recall_y1,
                     os.path.join(OUTPUT_DIR, "lorenz_y1_linear_ci"))
    plot_single_year(1, "semilog", results, lr_anchors_by_year[1], x_grid,
                     meta_by_year[1], recall_y1,
                     os.path.join(OUTPUT_DIR, "lorenz_y1_semilog_ci"))
    plot_composite_y1(1, results, lr_anchors_by_year[1], x_grid,
                      meta_by_year[1], recall_y1,
                      os.path.join(OUTPUT_DIR, "lorenz_y1_composite_ci"))
    plot_multihorizon(results, lr_anchors_by_year, x_grid, meta_by_year,
                      recall_by_year_model,
                      os.path.join(OUTPUT_DIR, "lorenz_multihorizon_ci"))

    # ------------------------------------------------------------------
    # T0-only year-1 (single-scan, prevalence-round) cohort
    # ------------------------------------------------------------------
    print("\n[T0] Year-1 cancer concentration on prevalence-round (T0) "
          "scans only — information-symmetric vs Lung-RADS.")
    tp_mask = comp["screen_timepoint"].astype(int).values == 0
    fu = 0  # year 1
    inc_t0 = np.array([_include(t, g, fu)[0] for t, g in zip(times, golds)])
    inc_t0 = inc_t0 & tp_mask
    lbl_t0 = np.array([_include(t, g, fu)[1] for t, g in zip(times, golds)])
    idx_t0 = np.where(inc_t0)[0]
    n_inc_t0 = int(inc_t0.sum())
    n_pos_t0 = int(lbl_t0[inc_t0].sum())
    meta_t0 = (n_inc_t0, n_pos_t0)
    lr_anchors_t0 = lr_anchor_flag_rates_full(lr_score_all, inc_t0)
    p_t0 = pids_all[idx_t0]
    pid_to_rows_t0 = {}
    for i, pp in enumerate(p_t0):
        pid_to_rows_t0.setdefault(pp, []).append(i)
    pid_to_rows_t0 = {k: np.asarray(v) for k, v in pid_to_rows_t0.items()}
    pids_uniq_t0 = np.array(sorted(pid_to_rows_t0.keys()))
    seed_t0 = RNG_SEED + 1 * 1000 + 7

    tasks_t0 = []
    recall_t0 = {}
    for m, sc in year_to_scores[1].items():
        tasks_t0.append((1, m, sc[idx_t0], lbl_t0[idx_t0],
                         pid_to_rows_t0, pids_uniq_t0, x_grid, N_BOOT, seed_t0))
        recall_t0[m] = recall_at_lr_match(
            sc, lbl_t0, lr_score_all, inc_t0, LR_ORDER["4B"])
    lr_score_for_t0 = np.where(lr_score_all > 0,
                               lr_score_all, np.nan).astype(float)
    tasks_t0.append((1, "Lung-RADS",
                     lr_score_for_t0[idx_t0], lbl_t0[idx_t0],
                     pid_to_rows_t0, pids_uniq_t0, x_grid, N_BOOT, seed_t0))
    recall_t0["Lung-RADS"] = recall_at_lr_match(
        lr_score_for_t0, lbl_t0, lr_score_all, inc_t0, LR_ORDER["4B"])

    print(f"[T0 run] Bootstrap {len(tasks_t0)} cells × {N_BOOT} "
          f"(n_T0={n_inc_t0}, cancers={n_pos_t0})…")
    results_t0 = {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(_boot_worker, t) for t in tasks_t0]
        for i, f in enumerate(as_completed(futures), 1):
            yr, m, pt, lo, hi, n_inc, n_pos = f.result()
            results_t0[(yr, m)] = {"point": pt, "lo": lo, "hi": hi,
                                   "n_inc": n_inc, "n_pos": n_pos}
            print(f"  ✓ ({i:2}/{len(tasks_t0)}) y1 T0 {m}: "
                  f"n_eff={n_inc}, cancers={n_pos}")

    # T0 CSV
    rows_t0 = []
    for m in MODEL_ORDER:
        r = results_t0.get((1, m))
        if r is None:
            continue
        for i, x in enumerate(x_grid):
            rows_t0.append(dict(model=m, x=x,
                                y_point=r["point"][i],
                                y_lo=r["lo"][i], y_hi=r["hi"][i],
                                n_eff=r["n_inc"], n_pos=r["n_pos"]))
    pd.DataFrame(rows_t0).to_csv(
        os.path.join(OUTPUT_DIR, "lorenz_y1_T0_with_ci.csv"), index=False)
    print("→ lorenz_y1_T0_with_ci.csv")

    # T0 standalone figures
    plot_single_year(1, "linear", results_t0, lr_anchors_t0, x_grid,
                     meta_t0, recall_t0,
                     os.path.join(OUTPUT_DIR, "lorenz_y1_T0_linear_ci"),
                     cohort_label="T0 only")
    plot_single_year(1, "semilog", results_t0, lr_anchors_t0, x_grid,
                     meta_t0, recall_t0,
                     os.path.join(OUTPUT_DIR, "lorenz_y1_T0_semilog_ci"),
                     cohort_label="T0 only")
    # All vs T0 comparison (semilog — the standard view)
    plot_alltp_vs_t0_compare(
        1, "semilog", results, results_t0,
        lr_anchors_by_year[1], lr_anchors_t0, x_grid,
        meta_by_year[1], meta_t0,
        {m: recall_by_year_model.get((1, m), np.nan) for m in MODEL_ORDER},
        recall_t0,
        os.path.join(OUTPUT_DIR, "lorenz_y1_alltp_vs_T0_semilog_ci"))
    print("\nDone.")


if __name__ == "__main__":
    main()
