#!/usr/bin/env python3
"""Decision Curve Analysis with 95% bootstrap CI bands.

Nature Medicine-style figures:
  - Helvetica/Arial sans-serif, 7pt body, 6pt ticks/legend.
  - Wong (2011) colorblind-safe palette.
  - Two-spine layout (bottom+left), outward ticks, no embedded title.
  - PDF (vector, TrueType-embedded) + 600 dpi PNG output per figure.
  - CI bands de-emphasized (alpha=0.12) so point-estimate lines dominate.
  - Net benefit expressed as "per 1,000 patients" (NB × 1000) for legibility,
    consistent with the convention used by Vickers et al.
  - Inline annotation showing n / cancers / prevalence.

Year-1 outputs (single time point, requested):
  eval_output/dca_y1_full_ci.{pdf,png}              (single panel)
  eval_output/dca_y1_zoomed_ci.{pdf,png}            (single panel, p_t ≤ 0.4)
  eval_output/dca_y1_composite_ci.{pdf,png}        (2-panel a/b, 180 mm)

Multi-horizon outputs (kept for paper compatibility):
  eval_output/dca_multihorizon_full_ci.{pdf,png}    (3-panel y1/y3/y6)
  eval_output/dca_multihorizon_zoomed_ci.{pdf,png}  (3-panel y1/y3/y6)

Long-format CSV (NB on the [0,1] scale; multiply by 1000 for the figure scale):
  eval_output/dca_with_ci_y1y3y6.csv
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from lifelines import KaplanMeierFitter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import _include, load_participant, PARTICIPANT_CSV, OUTPUT_DIR
from decile_stratification_devfit import (
    fit_lung_platt, fit_plcom_platt, fit_pillar_platt,
    attach_user_pillar, build_plcom_dev, _logit, _plcom_extra_cols,
)
from uspstf_plcom import plcom2012_risk

warnings.filterwarnings("ignore")

PT_GRID = np.linspace(0.001, 0.99, 200)
ZOOM_MAX = 0.40
# Vertical reference lines at the Lung-RADS-anchored p_t values. Labels
# match the discrete LR_DCA_POINTS markers below them so the figure reads
# consistently top-to-bottom (label → guide → marker).
LR_MARKERS = [0.01, 0.05, 0.15]
LR_MARKER_LABELS = {0.01: "LR≥3", 0.05: "LR≥4A", 0.15: "LR≥4B"}
# Lung-RADS as a discrete classifier on DCA. Each entry is
# (legend label, LR rank floor, p_t at which to evaluate). p_t matches the
# lower bound of each LR category's risk-of-malignancy range.
LR_ORDER = {"4X": 6, "4B": 5, "4A": 4, "3": 3, "2": 2, "1": 1}
LR_DCA_POINTS = [
    ("≥3",  LR_ORDER["3"],  0.01),
    ("≥4A", LR_ORDER["4A"], 0.05),
    ("≥4B", LR_ORDER["4B"], 0.15),
]
LUNGRADS_CSV = os.path.join(OUTPUT_DIR, "nlst_lungrads_pinsky.csv")
N_BOOT = 1000
RNG_SEED = 42

# ---- NM style ----
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
    "LungEvaty": "#D55E00",
    "Pillar-0":  "#009E73",
    "Sybil":     "#0072B2",
    "PLCOm2012": "#E69F00",
}
LINESTYLES = {
    "LungEvaty": "-",
    "Pillar-0":  "-",
    "Sybil":     "-",
    "PLCOm2012": (0, (5, 2)),
}
MODEL_ORDER = ["LungEvaty", "Pillar-0", "Sybil", "PLCOm2012"]
CI_ALPHA = 0.12


def _pct_int(x, _pos):
    if x >= 0.1: return f"{int(round(x*100))}"
    if x > 0: return f"{x*100:.1f}"
    return ""


def attach_lr(comp):
    """Merge Pinsky-derived Lung-RADS categories onto comp."""
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


def _ipcw_weights_paper(times, events, fu):
    kmf_c = KaplanMeierFitter()
    kmf_c.fit(times, 1 - events)
    kmf_e = KaplanMeierFitter()
    kmf_e.fit(times, events)
    prev = float(1.0 - kmf_e.predict(fu))
    G_t = float(max(kmf_c.predict(fu), 0.01))
    is_tp = (events == 1) & (times <= fu)
    is_fp = (events == 0) & (times > fu)
    tp_w = np.zeros(len(times))
    if is_tp.any():
        unique_tp_t = np.unique(times[is_tp])
        G_lookup = {t: float(max(kmf_c.predict(t), 0.01))
                    for t in unique_tp_t}
        for i in np.where(is_tp)[0]:
            tp_w[i] = 1.0 / G_lookup[times[i]]
    fp_w = 1.0 / G_t
    return tp_w, fp_w, is_tp, is_fp, prev


def nb_curve_ipcw_fast(scores, times, events, fu, pt_grid):
    tp_w, fp_w, is_tp, is_fp, prev = _ipcw_weights_paper(times, events, fu)
    n = len(scores)
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    tp_w_sorted = np.where(is_tp[order], tp_w[order], 0.0)
    fp_mask_sorted = is_fp[order].astype(int)
    cum_tp = np.cumsum(tp_w_sorted)
    cum_fp = np.cumsum(fp_mask_sorted)
    nb = np.full(len(pt_grid), np.nan)
    nb_all = np.zeros_like(pt_grid)
    for i, pt in enumerate(pt_grid):
        k = int(np.searchsorted(-s_sorted, -pt, side="right"))
        w = pt / (1 - pt) if 0 < pt < 1 else 0.0
        if k > 0:
            tp_r = cum_tp[k - 1] / n
            fp_r = cum_fp[k - 1] * fp_w / n
            nb[i] = tp_r - fp_r * w
        nb_all[i] = prev - (1 - prev) * w
    return nb, nb_all


def nb_curve_cc(scores, labels, pt_grid):
    n = len(scores)
    if n == 0 or labels.sum() == 0:
        return np.full_like(pt_grid, np.nan), np.full_like(pt_grid, np.nan)
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = labels[order].astype(int)
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    prev = float(labels.mean())
    nb = np.full(len(pt_grid), np.nan)
    nb_all = np.zeros_like(pt_grid)
    for i, pt in enumerate(pt_grid):
        k = int(np.searchsorted(-s_sorted, -pt, side="right"))
        w = pt / (1 - pt) if 0 < pt < 1 else 0.0
        if k > 0:
            tp_r = cum_tp[k - 1] / n
            fp_r = cum_fp[k - 1] / n
            nb[i] = tp_r - fp_r * w
        nb_all[i] = prev - (1 - prev) * w
    return nb, nb_all


def compute_nb_only(scores, times, events, year, pt_grid):
    fu = year - 1
    if year <= 3:
        return nb_curve_ipcw_fast(scores, times, events, fu, pt_grid)
    inc = np.array([_include(ti, ei, fu)[0] for ti, ei in zip(times, events)])
    lbl = np.array([_include(ti, ei, fu)[1] for ti, ei in zip(times, events)])
    return nb_curve_cc(scores[inc], lbl[inc], pt_grid)


def compute_treat_all(times, events, year, pt_grid):
    fu = year - 1
    if year <= 3:
        kmf_e = KaplanMeierFitter()
        kmf_e.fit(times, events)
        prev = float(1.0 - kmf_e.predict(fu))
    else:
        inc = np.array([_include(ti, ei, fu)[0] for ti, ei in zip(times, events)])
        lbl = np.array([_include(ti, ei, fu)[1] for ti, ei in zip(times, events)])
        if inc.sum() == 0:
            return np.full_like(pt_grid, np.nan)
        prev = float(lbl[inc].mean())
    out = np.zeros_like(pt_grid)
    for i, pt in enumerate(pt_grid):
        w = pt / (1 - pt) if 0 < pt < 1 else 0.0
        out[i] = prev - (1 - prev) * w
    return out


def _model_boot_worker(args):
    (model, scores, times, events, year, pt_grid,
     pid_to_rows, pids_uniq, n_boot, seed) = args
    rng = np.random.default_rng(seed)
    nan_mask = ~np.isnan(scores)
    n_u = len(pids_uniq)
    boot_nb = np.zeros((n_boot, len(pt_grid)))
    for b in range(n_boot):
        sample = rng.choice(pids_uniq, size=n_u, replace=True)
        idx_all = np.concatenate([pid_to_rows[p] for p in sample])
        idx = idx_all[nan_mask[idx_all]]
        if len(idx) < 20 or events[idx].sum() == 0:
            boot_nb[b] = np.nan
            continue
        try:
            nb, _ = compute_nb_only(scores[idx], times[idx], events[idx],
                                    year, pt_grid)
        except Exception:
            nb = np.full(len(pt_grid), np.nan)
        boot_nb[b] = nb
    lo = np.nanpercentile(boot_nb, 2.5, axis=0)
    hi = np.nanpercentile(boot_nb, 97.5, axis=0)
    return ("model", model, year, lo, hi)


def lr_nb_at_pt_idx(times, events, lr_score, lr_floor, year, pt):
    """NB for the discrete rule 'treat if LR ≥ lr_floor' at p_t.

    Mirrors the IPCW (y1/y3) / complete-case (y6) convention used by the
    continuous model curves, so the LR markers are on the same NB scale.
    Subjects with missing LR (lr_score == 0) are excluded.
    """
    valid = lr_score > 0
    if not valid.any():
        return np.nan
    t = times[valid].astype(float)
    e = events[valid].astype(int)
    pos = lr_score[valid] >= lr_floor
    fu = year - 1
    if year <= 3:
        tp_w, fp_w, is_tp, is_fp, prev = _ipcw_weights_paper(t, e, fu)
        n = len(t)
        tp_r = float(tp_w[pos & is_tp].sum()) / n
        fp_r = float((pos & is_fp).sum()) * fp_w / n
    else:
        inc = np.array([_include(ti, ei, fu)[0] for ti, ei in zip(t, e)])
        lbl = np.array([_include(ti, ei, fu)[1] for ti, ei in zip(t, e)])
        if not inc.any():
            return np.nan
        pos_inc = pos[inc]
        lbl_inc = lbl[inc]
        n = int(inc.sum())
        tp_r = int((pos_inc & (lbl_inc == 1)).sum()) / n
        fp_r = int((pos_inc & (lbl_inc == 0)).sum()) / n
    w = pt / (1 - pt) if 0 < pt < 1 else 0.0
    return tp_r - fp_r * w


def _lr_boot_worker(args):
    """Bootstrap CI for one (year, LR rule, p_t) operating point."""
    (year, lr_label, lr_floor, pt, times, events, lr_score,
     pid_to_rows, pids_uniq, n_boot, seed) = args
    rng = np.random.default_rng(seed)
    n_u = len(pids_uniq)
    point = lr_nb_at_pt_idx(times, events, lr_score, lr_floor, year, pt)
    boot = np.zeros(n_boot)
    for b in range(n_boot):
        sample = rng.choice(pids_uniq, size=n_u, replace=True)
        idx = np.concatenate([pid_to_rows[p] for p in sample])
        try:
            boot[b] = lr_nb_at_pt_idx(times[idx], events[idx],
                                      lr_score[idx], lr_floor, year, pt)
        except Exception:
            boot[b] = np.nan
    lo = float(np.nanpercentile(boot, 2.5))
    hi = float(np.nanpercentile(boot, 97.5))
    return ("lr", year, lr_label, pt, point, lo, hi)


def _treat_all_boot_worker(args):
    (year, times, events, pt_grid,
     pid_to_rows, pids_uniq, n_boot, seed) = args
    rng = np.random.default_rng(seed)
    n_u = len(pids_uniq)
    boot = np.zeros((n_boot, len(pt_grid)))
    for b in range(n_boot):
        sample = rng.choice(pids_uniq, size=n_u, replace=True)
        idx = np.concatenate([pid_to_rows[p] for p in sample])
        try:
            boot[b] = compute_treat_all(times[idx], events[idx], year, pt_grid)
        except Exception:
            boot[b] = np.nan
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return ("treat_all", year, lo, hi)


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def _ylim_for(point_by_model, pt_grid, max_pt):
    """Y-limits framed around model curves in the visible region."""
    sub = pt_grid <= max_pt
    vals = []
    for nb in point_by_model.values():
        v = nb[sub]
        v = v[np.isfinite(v)]
        vals.append(v)
    if not vals or not any(len(v) for v in vals):
        return -0.005, 0.01
    arr = np.concatenate([v for v in vals if len(v)])
    nb_max = max(0.001, float(np.nanmax(arr)))
    nb_min = float(np.nanmin(arr))
    return min(nb_min, 0.0) - nb_max * 0.10, nb_max * 1.18


def _plot_lr_markers(ax, year, lr_markers, pt_max):
    """Plot Lung-RADS discrete operating points (LR≥3, LR≥4A, LR≥4B)
    with vertical 95% CI bars. Returns True if at least one marker was
    drawn (so the caller can decide whether to add the Lung-RADS legend
    entry)."""
    if lr_markers is None:
        return False
    plotted_any = False
    legend_added = False
    for (yr, lab), (nb, lo, hi, pt) in lr_markers.items():
        if yr != year or pt > pt_max or not np.isfinite(nb):
            continue
        nb_k = nb * 1000
        lo_k = lo * 1000
        hi_k = hi * 1000
        yerr = [[max(0.0, nb_k - lo_k)], [max(0.0, hi_k - nb_k)]]
        ax.errorbar(pt, nb_k, yerr=yerr, fmt="o", color="black",
                    markersize=3.5, capsize=2.0, elinewidth=0.7,
                    capthick=0.7, zorder=3.0,
                    label="Lung-RADS" if not legend_added else None)
        legend_added = True
        # Short LR category label, offset slightly to the right
        ax.text(pt, nb_k, f"  {lab}", fontsize=5.4, color="black",
                va="center", ha="left", zorder=3.0)
        plotted_any = True
    return plotted_any


def _plot_dca_panel(ax, year, point, ci, point_all, pt_max,
                    show_legend=True, n_meta=None,
                    show_treat_labels=True,
                    show_lr_labels=True,
                    lr_markers=None):
    point_by_model = {m: point[(year, m)] for m in MODEL_ORDER
                      if (year, m) in point}
    sub = PT_GRID <= pt_max
    pt_sub = PT_GRID[sub]
    # Reference lines first (lowest zorder)
    nb_all = point_all[year]
    ax.plot(pt_sub, nb_all[sub] * 1000, color="#555555", lw=0.7,
            dashes=(4, 2), alpha=0.85,
            label="Treat all" if show_treat_labels else None, zorder=1.5)
    ax.axhline(0, color="black", lw=0.45, alpha=0.7,
               label="Treat none" if show_treat_labels else None, zorder=1.5)
    # Model CI bands (low zorder)
    for m in MODEL_ORDER:
        if (year, m) not in point:
            continue
        lo, hi = ci[(year, m)]
        ax.fill_between(pt_sub, lo[sub] * 1000, hi[sub] * 1000,
                        color=COLORS[m], alpha=CI_ALPHA, linewidth=0,
                        zorder=1)
    # Model lines (top zorder)
    for m in MODEL_ORDER:
        if (year, m) not in point:
            continue
        ax.plot(pt_sub, point[(year, m)][sub] * 1000,
                color=COLORS[m], lw=1.5, ls=LINESTYLES[m],
                label=m, zorder=2.5)
    # Lung-RADS discrete operating points
    _plot_lr_markers(ax, year, lr_markers, pt_max)
    # LR-anchored markers. On the wide p_t∈[0,0.99] view the five anchors
    # cluster in the leftmost 1.5% of the axis and would render as a smear,
    # so we skip them entirely. On the zoomed view we draw vertical guides
    # and rotate labels vertically; the LR2/3 (1%) and LR3-top (2%) labels
    # are staggered to two y-rows to avoid touching.
    draw_lr_markers = pt_max <= ZOOM_MAX + 1e-6
    label_anchors = show_lr_labels and draw_lr_markers
    if draw_lr_markers:
        for pt_mark in LR_MARKERS:
            if pt_mark > pt_max:
                continue
            ax.axvline(pt_mark, color="#888888", lw=0.35,
                       ls=(0, (2, 2)), alpha=0.7, zorder=0.5)
            if label_anchors:
                # Three anchors at 1/5/15% are well-spaced even on the
                # 0–40% zoomed view, so no staggering is needed.
                ax.text(pt_mark, 1.01,
                        LR_MARKER_LABELS[pt_mark],
                        rotation=90, fontsize=5.2, color="#555555",
                        va="bottom", ha="center",
                        transform=ax.get_xaxis_transform())
    ax.set_xlim(0, pt_max)
    ylim_lo, ylim_hi = _ylim_for(point_by_model, PT_GRID, pt_max)
    ax.set_ylim(ylim_lo * 1000, ylim_hi * 1000)
    ax.xaxis.set_major_formatter(FuncFormatter(_pct_int))
    ax.set_xlabel(r"Threshold probability, $p_t$ (%)")
    ax.set_ylabel("Net benefit (per 1,000 patients)")
    _style_axes(ax)
    if show_legend:
        leg = ax.legend(loc="upper right", fontsize=5.8,
                        handlelength=1.6, borderpad=0.35, labelspacing=0.4)


def _add_meta_annotation(ax, n_meta, year, cohort_label=""):
    n_inc, n_pos = n_meta
    prev = n_pos / n_inc if n_inc else np.nan
    suff = " (complete-case)" if year == 6 else " (IPCW)"
    pre = f"{cohort_label}: " if cohort_label else ""
    txt = (f"{pre}Year {year}{suff} · n = {n_inc:,} scans · "
           f"{n_pos} cancers · prevalence = {prev*100:.2f}%")
    ax.text(0.0, -0.18, txt, transform=ax.transAxes,
            fontsize=5.6, color="#444444", ha="left", va="top")


def save_dual(fig, out_path_pdf, out_path_png):
    fig.savefig(out_path_pdf, format="pdf")
    fig.savefig(out_path_png, format="png", dpi=600)
    plt.close(fig)
    print(f"→ {out_path_pdf}")
    print(f"→ {out_path_png}")


def plot_dca_single(year, point, ci, point_all, pt_max, n_meta, out_stem,
                    cohort_label="", lr_markers=None):
    fig, ax = plt.subplots(figsize=(4.3, 3.3))
    _plot_dca_panel(ax, year, point, ci, point_all, pt_max,
                    show_legend=True, n_meta=n_meta,
                    lr_markers=lr_markers)
    _add_meta_annotation(ax, n_meta, year, cohort_label)
    save_dual(fig, f"{out_stem}.pdf", f"{out_stem}.png")


def plot_dca_alltp_vs_t0_compare(point_full, ci_full, point_all_full,
                                 point_t0, ci_t0, point_all_t0,
                                 meta_full, meta_t0, pt_max, out_stem,
                                 lr_markers_full=None,
                                 lr_markers_t0=None):
    """Side-by-side: full cohort vs T0-only (prevalence round) DCA. y1 only.

    Both panels share y-axis (per 1,000 patients); cohort identity + n /
    cancers / prevalence are consolidated into the panel sub-title (top)
    so the bottom-of-axis meta annotation is not needed and cannot
    collide with the x-axis label on a compact panel.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
    _plot_dca_panel(ax1, 1, point_full, ci_full, point_all_full, pt_max,
                    show_legend=True, n_meta=meta_full,
                    lr_markers=lr_markers_full)
    _plot_dca_panel(ax2, 1, point_t0, ci_t0, point_all_t0, pt_max,
                    show_legend=False, n_meta=meta_t0,
                    lr_markers=lr_markers_t0)
    # Short cohort labels; n / cancers / prevalence belong in the LaTeX
    # caption. Naming makes the LR information asymmetry explicit.
    subs = [
        (ax1, "a", "Year 1, all timepoints (LR w/ prior)"),
        (ax2, "b", "Year 1, T0 only (single-scan, LR symmetric)"),
    ]
    for ax, lbl, sub in subs:
        ax.text(-0.18, 1.38, lbl, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", va="top", ha="left")
        ax.text(0.5, 1.22, sub, transform=ax.transAxes,
                fontsize=7.0, color="#222222", ha="center", va="bottom")
    fig.tight_layout(w_pad=2.0, rect=[0, 0, 1, 0.84])
    save_dual(fig, f"{out_stem}.pdf", f"{out_stem}.png")


def plot_dca_composite_y1(point, ci, point_all, n_meta, out_stem,
                          lr_markers=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    _plot_dca_panel(ax1, 1, point, ci, point_all, 0.99,
                    show_legend=False, n_meta=n_meta,
                    lr_markers=lr_markers)
    _plot_dca_panel(ax2, 1, point, ci, point_all, ZOOM_MAX,
                    show_legend=True, n_meta=n_meta,
                    lr_markers=lr_markers)
    # Panel labels
    for ax, lbl in [(ax1, "a"), (ax2, "b")]:
        ax.text(-0.18, 1.06, lbl, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", va="bottom", ha="left")
    _add_meta_annotation(ax1, n_meta, 1)
    fig.tight_layout(w_pad=2.0)
    save_dual(fig, f"{out_stem}.pdf", f"{out_stem}.png")


def plot_dca_multihorizon(point, ci, point_all, meta_by_year, pt_max,
                          out_stem, lr_markers=None):
    # Taller canvas + sub-title pushed further above the staggered LR
    # labels (LR3-top sits at y=1.08, extends to ~y=1.17). Matches the
    # spacing used by the all-vs-T0 comparison plot.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.7))
    legend_year = 3 if pt_max <= ZOOM_MAX + 1e-6 else 1
    for ax, year, lbl in zip(axes, [1, 3, 6], ["a", "b", "c"]):
        _plot_dca_panel(ax, year, point, ci, point_all, pt_max,
                        show_legend=(year == legend_year),
                        show_lr_labels=True,
                        lr_markers=lr_markers)
        ax.text(-0.18, 1.38, lbl, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", va="top", ha="left")
        n_inc, n_pos = meta_by_year[year]
        suff = " (complete-case)" if year == 6 else ""
        ax.text(0.5, 1.22,
                f"Year {year} · n = {n_inc:,} · cancers = {n_pos}{suff}",
                transform=ax.transAxes, fontsize=6.4, color="#222222",
                ha="center", va="bottom")
    fig.tight_layout(w_pad=1.5, rect=[0, 0, 1, 0.84])
    save_dual(fig, f"{out_stem}.pdf", f"{out_stem}.png")


def main():
    setup_nm_style()
    print("=" * 78)
    print(f"DCA — NM style (n_boot={N_BOOT}, paired)")
    print("=" * 78)
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    comp = attach_lr(comp)
    comp, year_to_scores = build_scores_all_years(comp)
    times = comp["time_at_event"].values.astype(float)
    events = comp["gold"].values.astype(int)
    pids = comp["pid"].values
    lr_score = np.array([LR_ORDER.get(str(c), 0)
                         for c in comp["lungrads_v10"].astype(str).values],
                        dtype=float)

    pid_to_rows = {}
    for p, i in zip(pids, np.arange(len(pids))):
        pid_to_rows.setdefault(p, []).append(i)
    pid_to_rows = {k: np.asarray(v) for k, v in pid_to_rows.items()}
    pids_uniq = np.array(sorted(pid_to_rows.keys()))

    # Point estimates
    point = {}
    point_all = {}
    meta_by_year = {}
    for year in [1, 3, 6]:
        for m, sc in year_to_scores[year].items():
            mask = ~np.isnan(sc)
            nb, _ = compute_nb_only(sc[mask], times[mask], events[mask],
                                    year, PT_GRID)
            point[(year, m)] = nb
        point_all[year] = compute_treat_all(times, events, year, PT_GRID)
        fu = year - 1
        inc = np.array([_include(t, g, fu)[0] for t, g in zip(times, events)])
        lbl = np.array([_include(t, g, fu)[1] for t, g in zip(times, events)])
        n_inc = int(inc.sum())
        n_pos = int(lbl[inc].sum())
        meta_by_year[year] = (n_inc, n_pos)

    tasks_model = []
    for year in [1, 3, 6]:
        seed_year = RNG_SEED + year * 1000
        for m, sc in year_to_scores[year].items():
            tasks_model.append((m, sc, times, events, year, PT_GRID,
                                pid_to_rows, pids_uniq, N_BOOT, seed_year))
    tasks_all = []
    for year in [1, 3, 6]:
        tasks_all.append((year, times, events, PT_GRID,
                          pid_to_rows, pids_uniq, N_BOOT,
                          RNG_SEED + year * 1000))
    # Lung-RADS discrete operating points (3 markers per year, shared
    # resamples with the model curves for paired CIs).
    tasks_lr = []
    for year in [1, 3, 6]:
        seed_year = RNG_SEED + year * 1000
        for lab, floor, pt in LR_DCA_POINTS:
            tasks_lr.append((year, lab, floor, pt, times, events, lr_score,
                             pid_to_rows, pids_uniq, N_BOOT, seed_year))

    print(f"[run] Bootstrap n={N_BOOT}, {len(tasks_model)} model cells + "
          f"{len(tasks_all)} treat-all + {len(tasks_lr)} LR markers …")
    ci = {}
    ci_all = {}
    lr_markers = {}  # (year, label) -> (nb, lo, hi, pt)
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(_model_boot_worker, t) for t in tasks_model]
        futs += [ex.submit(_treat_all_boot_worker, t) for t in tasks_all]
        futs += [ex.submit(_lr_boot_worker, t) for t in tasks_lr]
        done = 0
        for f in as_completed(futs):
            r = f.result()
            done += 1
            if r[0] == "model":
                _, m, year, lo, hi = r
                ci[(year, m)] = (lo, hi)
                print(f"  ✓ ({done:2}/{len(futs)}) y{year} {m}")
            elif r[0] == "treat_all":
                _, year, lo, hi = r
                ci_all[year] = (lo, hi)
                print(f"  ✓ ({done:2}/{len(futs)}) y{year} treat-all")
            else:  # "lr"
                _, year, lab, pt, point_v, lo, hi = r
                lr_markers[(year, lab)] = (point_v, lo, hi, pt)
                print(f"  ✓ ({done:2}/{len(futs)}) y{year} LR {lab}")

    # CSV (NB on [0,1] scale)
    rows = []
    for year in [1, 3, 6]:
        nb_all = point_all[year]
        all_lo, all_hi = ci_all[year]
        for m in MODEL_ORDER:
            if (year, m) not in point:
                continue
            nb = point[(year, m)]
            lo, hi = ci[(year, m)]
            for i, pt in enumerate(PT_GRID):
                rows.append(dict(year=year, model=m, p_t=pt,
                                 nb=nb[i], nb_lo=lo[i], nb_hi=hi[i],
                                 nb_treat_all=nb_all[i],
                                 nb_treat_all_lo=all_lo[i],
                                 nb_treat_all_hi=all_hi[i]))
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "dca_with_ci_y1y3y6.csv"), index=False)
    print("→ dca_with_ci_y1y3y6.csv")

    # LR markers CSV (one row per (cohort, year, LR rule))
    lr_rows = []
    for (yr, lab), (nb, lo, hi, pt) in lr_markers.items():
        lr_rows.append(dict(cohort="all", year=yr, lr_rule=lab, p_t=pt,
                            nb=nb, nb_lo=lo, nb_hi=hi))
    pd.DataFrame(lr_rows).to_csv(
        os.path.join(OUTPUT_DIR, "dca_lr_markers_full.csv"), index=False)
    print("→ dca_lr_markers_full.csv")

    n_meta_y1 = meta_by_year[1]
    plot_dca_single(1, point, ci, point_all, pt_max=0.99, n_meta=n_meta_y1,
                    out_stem=os.path.join(OUTPUT_DIR, "dca_y1_full_ci"),
                    lr_markers=lr_markers)
    plot_dca_single(1, point, ci, point_all, pt_max=ZOOM_MAX, n_meta=n_meta_y1,
                    out_stem=os.path.join(OUTPUT_DIR, "dca_y1_zoomed_ci"),
                    lr_markers=lr_markers)
    plot_dca_composite_y1(point, ci, point_all, n_meta_y1,
                          os.path.join(OUTPUT_DIR, "dca_y1_composite_ci"),
                          lr_markers=lr_markers)
    plot_dca_multihorizon(point, ci, point_all, meta_by_year, 0.99,
                          os.path.join(OUTPUT_DIR, "dca_multihorizon_full_ci"),
                          lr_markers=lr_markers)
    plot_dca_multihorizon(point, ci, point_all, meta_by_year, ZOOM_MAX,
                          os.path.join(OUTPUT_DIR, "dca_multihorizon_zoomed_ci"),
                          lr_markers=lr_markers)

    # ------------------------------------------------------------------
    # T0-only year-1 (single-scan, prevalence-round) cohort
    # ------------------------------------------------------------------
    print("\n[T0] DCA on prevalence-round (T0) scans only.")
    tp_mask = comp["screen_timepoint"].astype(int).values == 0
    idx_t0_full = np.where(tp_mask)[0]
    times_t0 = times[idx_t0_full]
    events_t0 = events[idx_t0_full]
    pids_t0 = pids[idx_t0_full]
    pid_to_rows_t0 = {}
    for p, i in zip(pids_t0, np.arange(len(pids_t0))):
        pid_to_rows_t0.setdefault(p, []).append(i)
    pid_to_rows_t0 = {k: np.asarray(v) for k, v in pid_to_rows_t0.items()}
    pids_uniq_t0 = np.array(sorted(pid_to_rows_t0.keys()))

    # T0 point estimates (year-1 only)
    point_t0 = {}
    for m, sc in year_to_scores[1].items():
        sc_t0 = sc[idx_t0_full]
        mask = ~np.isnan(sc_t0)
        nb, _ = compute_nb_only(sc_t0[mask], times_t0[mask], events_t0[mask],
                                1, PT_GRID)
        point_t0[(1, m)] = nb
    point_all_t0 = {1: compute_treat_all(times_t0, events_t0, 1, PT_GRID)}

    # T0 cohort metadata
    inc_t0 = np.array([_include(t, g, 0)[0]
                       for t, g in zip(times_t0, events_t0)])
    lbl_t0 = np.array([_include(t, g, 0)[1]
                       for t, g in zip(times_t0, events_t0)])
    n_inc_t0 = int(inc_t0.sum())
    n_pos_t0 = int(lbl_t0[inc_t0].sum())
    meta_t0 = (n_inc_t0, n_pos_t0)

    # T0 bootstrap (models + treat-all + LR markers, all paired seed)
    seed_t0 = RNG_SEED + 1 * 1000 + 7
    tasks_t0 = []
    for m, sc in year_to_scores[1].items():
        tasks_t0.append((m, sc[idx_t0_full], times_t0, events_t0, 1, PT_GRID,
                         pid_to_rows_t0, pids_uniq_t0, N_BOOT, seed_t0))
    tasks_t0_all = [(1, times_t0, events_t0, PT_GRID,
                     pid_to_rows_t0, pids_uniq_t0, N_BOOT, seed_t0)]
    lr_score_t0 = lr_score[idx_t0_full]
    tasks_t0_lr = []
    for lab, floor, pt in LR_DCA_POINTS:
        tasks_t0_lr.append((1, lab, floor, pt, times_t0, events_t0,
                            lr_score_t0, pid_to_rows_t0, pids_uniq_t0,
                            N_BOOT, seed_t0))

    print(f"[T0 run] Bootstrap n={N_BOOT}, {len(tasks_t0)} model cells + "
          f"1 treat-all + {len(tasks_t0_lr)} LR markers "
          f"(n_T0={n_inc_t0}, cancers={n_pos_t0})…")
    ci_t0 = {}
    ci_all_t0 = {}
    lr_markers_t0 = {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(_model_boot_worker, t) for t in tasks_t0]
        futs += [ex.submit(_treat_all_boot_worker, t) for t in tasks_t0_all]
        futs += [ex.submit(_lr_boot_worker, t) for t in tasks_t0_lr]
        done = 0
        for f in as_completed(futs):
            r = f.result()
            done += 1
            if r[0] == "model":
                _, m, year, lo, hi = r
                ci_t0[(year, m)] = (lo, hi)
                print(f"  ✓ ({done}/{len(futs)}) T0 y{year} {m}")
            elif r[0] == "treat_all":
                _, year, lo, hi = r
                ci_all_t0[year] = (lo, hi)
                print(f"  ✓ ({done}/{len(futs)}) T0 y{year} treat-all")
            else:  # "lr"
                _, year, lab, pt, point_v, lo, hi = r
                lr_markers_t0[(year, lab)] = (point_v, lo, hi, pt)
                print(f"  ✓ ({done}/{len(futs)}) T0 y{year} LR {lab}")

    # T0 CSV (y1)
    rows_t0 = []
    nb_all = point_all_t0[1]
    all_lo, all_hi = ci_all_t0[1]
    for m in MODEL_ORDER:
        if (1, m) not in point_t0:
            continue
        nb = point_t0[(1, m)]
        lo, hi = ci_t0[(1, m)]
        for i, pt in enumerate(PT_GRID):
            rows_t0.append(dict(year=1, model=m, p_t=pt,
                                nb=nb[i], nb_lo=lo[i], nb_hi=hi[i],
                                nb_treat_all=nb_all[i],
                                nb_treat_all_lo=all_lo[i],
                                nb_treat_all_hi=all_hi[i]))
    pd.DataFrame(rows_t0).to_csv(
        os.path.join(OUTPUT_DIR, "dca_y1_T0_with_ci.csv"), index=False)
    print("→ dca_y1_T0_with_ci.csv")
    lr_rows_t0 = []
    for (yr, lab), (nb, lo, hi, pt) in lr_markers_t0.items():
        lr_rows_t0.append(dict(cohort="T0", year=yr, lr_rule=lab, p_t=pt,
                               nb=nb, nb_lo=lo, nb_hi=hi))
    pd.DataFrame(lr_rows_t0).to_csv(
        os.path.join(OUTPUT_DIR, "dca_lr_markers_T0.csv"), index=False)
    print("→ dca_lr_markers_T0.csv")

    # T0 standalone figures
    plot_dca_single(1, point_t0, ci_t0, point_all_t0, pt_max=0.99,
                    n_meta=meta_t0,
                    out_stem=os.path.join(OUTPUT_DIR, "dca_y1_T0_full_ci"),
                    cohort_label="T0 only",
                    lr_markers=lr_markers_t0)
    plot_dca_single(1, point_t0, ci_t0, point_all_t0, pt_max=ZOOM_MAX,
                    n_meta=meta_t0,
                    out_stem=os.path.join(OUTPUT_DIR, "dca_y1_T0_zoomed_ci"),
                    cohort_label="T0 only",
                    lr_markers=lr_markers_t0)
    # All-vs-T0 comparison (zoomed — the standard clinical view)
    plot_dca_alltp_vs_t0_compare(
        point, ci, point_all, point_t0, ci_t0, point_all_t0,
        meta_by_year[1], meta_t0, ZOOM_MAX,
        os.path.join(OUTPUT_DIR, "dca_y1_alltp_vs_T0_zoomed_ci"),
        lr_markers_full=lr_markers, lr_markers_t0=lr_markers_t0)
    print("\nDone.")


if __name__ == "__main__":
    main()
