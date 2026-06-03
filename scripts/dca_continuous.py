#!/usr/bin/env python3
"""Continuous DCA curves with LR-anchored operating-point markers.

NB(p_t) = TPR_w * π - FPR_w * (1-π) * (p_t / (1-p_t))

Continuous sweep across p_t ∈ [0, 0.2] for y1/y3/y6 × {LungEvaty, Pillar-0,
Sybil, PLCOm}. Treat-all (intervene on everyone) and treat-none (intervene
on nobody) reference curves shown for context. Markers placed at LR-aligned
p_t = {1%, 2%, 5%, 15%}.

Patient-clustered bootstrap n=500 for CI bands (lighter than the discrete
table because we're computing 100+ thresholds per curve).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import (
    _include, _ipcw_weights, load_participant, PARTICIPANT_CSV, OUTPUT_DIR,
)
from decile_stratification_devfit import (
    fit_lung_platt, fit_plcom_platt, fit_pillar_platt,
    attach_user_pillar, build_plcom_dev, _logit, _plcom_extra_cols,
)
from uspstf_plcom import plcom2012_risk

PT_GRID = np.linspace(0.001, 0.20, 100)
LR_MARKERS = [0.01, 0.02, 0.05, 0.15]
LR_MARKER_LABELS = {0.01: "LR2/3", 0.02: "LR3 top",
                    0.05: "LR4A start", 0.15: "LR4B start"}
N_BOOT = 500
RNG_SEED = 42
colors = {"LungEvaty": "#d62728", "Sybil": "#1f77b4",
          "Pillar-0": "#2ca02c", "PLCOm2012": "#ff7f0e"}


def _pct_fmt(x, _pos):
    return f"{x*100:.0f}%" if x >= 0.01 else (f"{x*100:.1f}%" if x > 0 else "0%")


def build_scores_all_years(comp):
    print("[setup] Fitting calibrators …")
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
                plcom_raw[i] = float(plcom2012_risk(pid_to_feat.loc[pid], year=year))
        mm = ~np.isnan(plcom_raw)
        pl_cal = np.full(len(comp), np.nan)
        pl_cal[mm] = plcom_platt[year].predict_proba(
            _logit(plcom_raw[mm]).reshape(-1, 1))[:, 1]
        out[year] = {"LungEvaty": le_cal, "Sybil": syb_cal,
                     "Pillar-0": pi_cal, "PLCOm2012": pl_cal}
    return comp, out


def nb_curve_ipcw(scores, times, events, fu, pt_grid):
    """Vectorized continuous NB curve via IPCW weights."""
    tp_w, fp_w, is_tp, is_fp, prev = _ipcw_weights(times, events, fu)
    n = len(scores)
    # Sort by descending score; pre-compute cumulative tp_w and fp_count
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    tp_w_sorted = np.where(is_tp[order], tp_w[order], 0.0)
    fp_mask_sorted = is_fp[order].astype(int)
    cum_tp = np.cumsum(tp_w_sorted)
    cum_fp = np.cumsum(fp_mask_sorted)
    nb_out = np.zeros_like(pt_grid)
    nb_all_out = np.zeros_like(pt_grid)
    for i, pt in enumerate(pt_grid):
        # All scans with score >= pt are positives
        k = int(np.searchsorted(-s_sorted, -pt, side="right"))
        tp_r = (cum_tp[k - 1] if k > 0 else 0.0) / n
        fp_r = (cum_fp[k - 1] if k > 0 else 0.0) * fp_w / n
        w = pt / (1 - pt) if 0 < pt < 1 else 0.0
        nb_out[i] = tp_r - fp_r * w
        nb_all_out[i] = prev - (1 - prev) * w
    return nb_out, nb_all_out


def nb_curve_cc(scores, labels, pt_grid):
    """Complete-case continuous NB curve."""
    n = len(scores)
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = labels[order].astype(int)
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    prev = labels.mean()
    nb_out = np.zeros_like(pt_grid)
    nb_all_out = np.zeros_like(pt_grid)
    for i, pt in enumerate(pt_grid):
        k = int(np.searchsorted(-s_sorted, -pt, side="right"))
        tp_r = (cum_tp[k - 1] if k > 0 else 0.0) / n
        fp_r = (cum_fp[k - 1] if k > 0 else 0.0) / n
        w = pt / (1 - pt) if 0 < pt < 1 else 0.0
        nb_out[i] = tp_r - fp_r * w
        nb_all_out[i] = prev - (1 - prev) * w
    return nb_out, nb_all_out


def compute_curve(scores, times, events, year, pt_grid):
    mask = ~np.isnan(scores)
    s = scores[mask]
    t = times[mask].astype(float)
    e = events[mask].astype(int)
    fu = year - 1
    if year <= 3:
        return nb_curve_ipcw(s, t, e, fu, pt_grid)
    else:
        inc = np.array([_include(ti, ei, fu)[0] for ti, ei in zip(t, e)])
        lbl = np.array([_include(ti, ei, fu)[1] for ti, ei in zip(t, e)])
        return nb_curve_cc(s[inc], lbl[inc], pt_grid)


def _boot_worker(args):
    (model, scores, times, events, year, pt_grid, pid_to_rows, pids_uniq,
     n_boot, seed) = args
    rng = np.random.default_rng(seed)
    boot_nb = np.zeros((n_boot, len(pt_grid)))
    for b in range(n_boot):
        sample = rng.choice(pids_uniq, size=len(pids_uniq), replace=True)
        idx = np.concatenate([pid_to_rows[p] for p in sample])
        try:
            nb, _ = compute_curve(scores[idx], times[idx], events[idx],
                                  year, pt_grid)
        except Exception:
            nb = np.full(len(pt_grid), np.nan)
        boot_nb[b] = nb
    lo = np.nanpercentile(boot_nb, 2.5, axis=0)
    hi = np.nanpercentile(boot_nb, 97.5, axis=0)
    return (model, year, lo, hi)


def main():
    print("=" * 78)
    print("CONTINUOUS DCA CURVES (p_t ∈ [0, 0.2], y1/y3/y6, IPCW for y1/y3)")
    print("=" * 78)
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    comp, year_to_scores = build_scores_all_years(comp)

    times = comp["time_at_event"].values.astype(float)
    events = comp["gold"].values.astype(int)
    pids = comp["pid"].values
    pid_to_rows = {}
    for p, i in zip(pids, np.arange(len(pids))):
        pid_to_rows.setdefault(p, []).append(i)
    pid_to_rows = {k: np.asarray(v) for k, v in pid_to_rows.items()}
    pids_uniq = np.array(list(pid_to_rows.keys()))

    # Point estimates
    point = {}  # (year, model) -> (nb, nb_all)
    for year in [1, 3, 6]:
        for model, sc in year_to_scores[year].items():
            mask = ~np.isnan(sc)
            s = sc[mask]; t = times[mask]; e = events[mask]
            fu = year - 1
            if year <= 3:
                nb, nb_all = nb_curve_ipcw(s, t, e, fu, PT_GRID)
            else:
                inc = np.array([_include(ti, ei, fu)[0] for ti, ei in zip(t, e)])
                lbl = np.array([_include(ti, ei, fu)[1] for ti, ei in zip(t, e)])
                nb, nb_all = nb_curve_cc(s[inc], lbl[inc], PT_GRID)
            point[(year, model)] = (nb, nb_all)

    # Bootstrap bands
    print(f"[run] Bootstrap n={N_BOOT} for CI bands (12 cells)…")
    tasks = []
    for year in [1, 3, 6]:
        for model, sc in year_to_scores[year].items():
            tasks.append((model, sc, times, events, year, PT_GRID,
                          pid_to_rows, pids_uniq, N_BOOT, RNG_SEED))
    ci_band = {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(_boot_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            model, year, lo, hi = f.result()
            ci_band[(year, model)] = (lo, hi)
            print(f"  … {i}/{len(tasks)}: y{year} {model}")

    # Save raw curves
    rows = []
    for (year, model), (nb, nb_all) in point.items():
        lo, hi = ci_band[(year, model)]
        for i, pt in enumerate(PT_GRID):
            rows.append(dict(year=year, model=model, p_t=pt,
                             nb=nb[i], nb_lo=lo[i], nb_hi=hi[i],
                             nb_treat_all=nb_all[i]))
    df_curve = pd.DataFrame(rows)
    df_curve.to_csv(os.path.join(OUTPUT_DIR, "dca_continuous_y1y3y6.csv"),
                    index=False)
    print(f"\n→ dca_continuous_y1y3y6.csv")

    # Figure: 3-panel y1/y3/y6
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    for ax, year in zip(axes, [1, 3, 6]):
        nb_all = point[(year, "LungEvaty")][1]  # same for all models
        # Treat-all and treat-none reference lines
        ax.plot(PT_GRID, nb_all, color="grey", lw=1.0, ls="--",
                label="Treat all", alpha=0.7)
        ax.axhline(0, color="black", lw=0.7, alpha=0.7, label="Treat none")
        # Model curves with CI bands
        for model in ["LungEvaty", "Pillar-0", "Sybil", "PLCOm2012"]:
            nb, _ = point[(year, model)]
            lo, hi = ci_band[(year, model)]
            ax.fill_between(PT_GRID, lo, hi, color=colors[model], alpha=0.18)
            ax.plot(PT_GRID, nb, color=colors[model], lw=1.6, label=model)
        # LR markers
        for pt_mark in LR_MARKERS:
            ax.axvline(pt_mark, color="grey", lw=0.6, ls=":", alpha=0.5)
            ax.text(pt_mark, ax.get_ylim()[1] * 0.95 if year == 1 else 0.95,
                    LR_MARKER_LABELS[pt_mark], rotation=90, fontsize=7,
                    color="grey", va="top", ha="right",
                    transform=ax.get_xaxis_transform())
        ax.set_xlim(0, 0.20)
        ax.xaxis.set_major_formatter(FuncFormatter(_pct_fmt))
        ax.set_xlabel("Threshold probability p_t")
        if year == 1:
            ax.set_ylabel("Net benefit")
        ax.set_title(f"y{year} ({'IPCW' if year<=3 else 'complete-case'})",
                     fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        # Set y-lim to a sensible region
        nb_max = max(point[(year, m)][0].max() for m in
                     ["LungEvaty", "Pillar-0", "Sybil"])
        ax.set_ylim(-nb_max * 0.1, nb_max * 1.15)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "dca_continuous.png")
    plt.savefig(out, dpi=150)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
