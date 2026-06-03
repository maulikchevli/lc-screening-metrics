#!/usr/bin/env python3
"""Lung-RADS-anchored risk bins and Lorenz/concentration curves.

Two analyses:

1. Risk-anchored bins (Lung-RADS thresholds): <1%, 1-2%, 2-5%, 5-15%, >=15%.
   For each model, report N scans, N y1 cancers, observed y1 incidence per bin.
   Also report actual Lung-RADS categories (Pinsky algorithm) as a baseline.

2. Lorenz / concentration curves at y1: cumulative fraction of cancers vs
   cumulative fraction of scans, sorted by descending model score. Gini = 2 *
   area between curve and diagonal.

Calibrated probabilities used throughout (Platt for LungEvaty/PLCOm/Pillar-0,
author-released calibrator for Sybil v1.0).
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LogisticRegression


def _pct_fmt(x, _pos):
    if x <= 0:
        return ""
    if x >= 0.1:
        return f"{x * 100:.0f}%"
    if x >= 0.01:
        return f"{x * 100:.0f}%"
    if x >= 0.001:
        return f"{x * 100:.1f}%"
    return f"{x * 100:.2f}%"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import _include, load_participant, PARTICIPANT_CSV, OUTPUT_DIR
from decile_stratification_devfit import (
    fit_lung_platt, fit_plcom_platt, fit_pillar_platt,
    attach_user_pillar, build_plcom_dev, _logit,
)

LUNGRADS_CSV = os.path.join(OUTPUT_DIR, "nlst_lungrads_pinsky.csv")
LR_BIN_EDGES = [0.0, 0.01, 0.02, 0.05, 0.15, 1.01]
LR_BIN_LABELS = ["B1: <1%", "B2: 1-2%", "B3: 2-5%", "B4: 5-15%", "B5: >=15%"]


def attach_lungrads(comp):
    lr = pd.read_csv(LUNGRADS_CSV)
    lr["pid"] = lr["pid"].astype(str)
    lr["study_yr"] = lr["study_yr"].astype(int)
    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    comp["study_yr"] = comp["screen_timepoint"].astype(int)
    comp = comp.merge(
        lr[["pid", "study_yr", "lungrads_v10"]],
        on=["pid", "study_yr"], how="left",
    )
    return comp


def y1_label_array(comp):
    """Scan-relative year-1 inclusion/label."""
    times = comp["time_at_event"].values
    golds = comp["gold"].values
    inc = np.zeros(len(comp), bool)
    lbl = np.zeros(len(comp), int)
    for i, (t, g) in enumerate(zip(times, golds)):
        ok, y = _include(t, g, 0)
        inc[i] = ok
        lbl[i] = y
    return inc, lbl


def build_y1_calibrated_scores(comp):
    """Return dict model -> calibrated y1 probability vector."""
    print("[setup] Fitting LungEvaty Platt on dev …")
    lung_platt = fit_lung_platt()

    print("[setup] Computing PLCOm on test cohort …")
    part = load_participant(PARTICIPANT_CSV)
    dev_plcom = build_plcom_dev(part)
    plcom_platt = fit_plcom_platt(dev_plcom)
    base_cols = ["pid", "age", "race", "pkyr", "smokeyr", "cigsmok"]
    base_cols = [c for c in base_cols if c in part.columns]
    from uspstf_plcom import plcom2012_risk
    from decile_stratification_devfit import _plcom_extra_cols as _xc
    extras = _xc(part)
    base = part[base_cols + extras].drop_duplicates("pid").copy()
    base["pid"] = base["pid"].astype(str)
    test_p = comp[["pid"]].drop_duplicates().merge(base, on="pid", how="left")
    plcom_y1_raw = pd.Series(index=comp.index, dtype=float)
    pid_to_feat = test_p.set_index("pid")
    for i, pid in enumerate(comp["pid"].values):
        r = pid_to_feat.loc[pid] if pid in pid_to_feat.index else None
        plcom_y1_raw.iat[i] = float(plcom2012_risk(r, year=1)) if r is not None else np.nan

    mask = ~plcom_y1_raw.isna()
    plcom_y1_cal = np.full(len(comp), np.nan)
    plcom_y1_cal[mask] = plcom_platt[1].predict_proba(
        _logit(plcom_y1_raw[mask].values).reshape(-1, 1))[:, 1]

    print("[setup] Attaching Pillar-0 + fitting Platt on dev …")
    comp = attach_user_pillar(comp)
    pillar_platt = fit_pillar_platt()
    pillar_y1_raw = comp["pillar_user_y1"].values.astype(float)
    mask_p = ~np.isnan(pillar_y1_raw)
    pillar_y1_cal = np.full(len(comp), np.nan)
    pillar_y1_cal[mask_p] = pillar_platt[1].predict_proba(
        _logit(pillar_y1_raw[mask_p]).reshape(-1, 1))[:, 1]

    print("[setup] Building LungEvaty calibrated y1 …")
    lung_y1_raw = comp["risk_score"].values.astype(float)
    lung_y1_cal = lung_platt[1].predict_proba(
        _logit(lung_y1_raw).reshape(-1, 1))[:, 1]

    return comp, {
        "LungEvaty": lung_y1_cal,
        "Sybil":     comp["sv1c_risk_1"].values.astype(float),
        "Pillar-0":  pillar_y1_cal,
        "PLCOm2012": plcom_y1_cal,
    }


def risk_bin_table(comp, scores_by_model, inc_y1, lbl_y1):
    """For each model: per-bin N, cancers, incidence."""
    rows = []
    for model, scores in scores_by_model.items():
        mask = inc_y1 & ~np.isnan(scores)
        s = scores[mask]; y = lbl_y1[mask]
        bins = pd.cut(s, bins=LR_BIN_EDGES, labels=LR_BIN_LABELS,
                      include_lowest=True, right=False)
        for b in LR_BIN_LABELS:
            sel = (bins == b)
            n = int(sel.sum())
            cancers = int(y[sel].sum()) if n else 0
            inc_rate = cancers / n if n else np.nan
            rows.append(dict(model=model, bin=b, n=n, cancers=cancers,
                             incidence=inc_rate))
    return pd.DataFrame(rows)


def lungrads_bin_table(comp, inc_y1, lbl_y1):
    """Actual Lung-RADS distribution + y1 incidence."""
    rows = []
    lr = comp["lungrads_v10"].astype(str).values
    for cat in ["1", "2", "3", "4A", "4B", "4X"]:
        sel = inc_y1 & (lr == cat)
        n = int(sel.sum())
        cancers = int(lbl_y1[sel].sum())
        inc_rate = cancers / n if n else np.nan
        rows.append(dict(category=f"LR{cat}", n=n, cancers=cancers,
                         incidence=inc_rate))
    return pd.DataFrame(rows)


def lungrads_reclassification(comp, scores_by_model, inc_y1, lbl_y1,
                              high_thr=0.15):
    """For each LR category × model, report:
       - N scans in LR cat
       - N y1 cancers in LR cat
       - mean predicted p̂_y1 (calibration check within LR bin)
       - N scans the model assigns to B5 (>= high_thr)
       - N cancers the model assigns to B5 (model-LR agreement on positives)
       - N cancers the model misses despite LR (in LR4A/4B/4X but not B5)
    """
    lr = comp["lungrads_v10"].astype(str).values
    rows = []
    for cat in ["1", "2", "3", "4A", "4B", "4X"]:
        cat_mask = inc_y1 & (lr == cat)
        n = int(cat_mask.sum())
        cancers = int(lbl_y1[cat_mask].sum())
        for model, s in scores_by_model.items():
            sm = ~np.isnan(s)
            m = cat_mask & sm
            n_m = int(m.sum())
            mean_p = float(np.nanmean(s[m])) if n_m else np.nan
            high = m & (s >= high_thr)
            high_cancers = int(lbl_y1[high].sum())
            n_high = int(high.sum())
            rows.append(dict(
                lr_cat=f"LR{cat}", model=model,
                n=n_m, cancers=cancers,
                mean_p_hat=mean_p,
                n_in_B5=n_high,
                cancers_in_B5=high_cancers,
                cancers_missed_by_B5=cancers - high_cancers,
                pct_flagged_B5=n_high / n_m if n_m else np.nan,
            ))
    return pd.DataFrame(rows)


def lr_reclass_wide(reclass_df):
    """Pivot to one row per LR category, columns per model."""
    pieces = []
    base = reclass_df[reclass_df.model == reclass_df.model.iloc[0]][
        ["lr_cat", "n", "cancers"]].rename(columns={"n": "N", "cancers": "y1_ca"})
    pieces.append(base.set_index("lr_cat"))
    for m, g in reclass_df.groupby("model"):
        sub = g.set_index("lr_cat")[
            ["mean_p_hat", "pct_flagged_B5", "cancers_in_B5"]]
        sub.columns = [f"{m}_p̂_mean", f"{m}_%B5", f"{m}_ca∈B5"]
        pieces.append(sub)
    wide = pd.concat(pieces, axis=1)
    return wide.reset_index()


def operating_points(scores_by_model, inc_mask, lbl_y1, flag_rates,
                     scope_label="all"):
    """For each model and each top-K flag rate, report cancers captured +
    precision. flag_rates is a list of fractions in (0, 1]."""
    rows = []
    total_pos = int(lbl_y1[inc_mask].sum())
    n_total = int(inc_mask.sum())
    for model, sc in scores_by_model.items():
        mask = inc_mask & ~np.isnan(sc)
        s = sc[mask]; y = lbl_y1[mask]
        order = np.argsort(-s, kind="mergesort")
        y_sorted = y[order].astype(int)
        cum_pos = np.cumsum(y_sorted)
        n_eval = len(y_sorted)
        for fr in flag_rates:
            k = max(1, int(round(fr * n_total)))
            k = min(k, n_eval)
            cap = int(cum_pos[k - 1])
            rows.append(dict(
                scope=scope_label, model=model,
                flag_rate=fr, n_flag=k,
                cancers_captured=cap,
                precision=cap / k,
                recall=cap / total_pos if total_pos else np.nan,
            ))
    return pd.DataFrame(rows)


def lr_operating_point(comp, lr_score, inc_mask, lbl_y1, lr_threshold_label,
                       lr_threshold_rank, scope_label="all"):
    """Lung-RADS operating point at a category threshold (e.g., '4X+',
    '4B+', '4A+'). rank: minimum CAT_ORDER rank to flag."""
    mask = inc_mask & (lr_score >= lr_threshold_rank)
    n = int(mask.sum())
    cap = int(lbl_y1[mask].sum())
    n_total = int(inc_mask.sum())
    total_pos = int(lbl_y1[inc_mask].sum())
    return dict(scope=scope_label, model=f"Lung-RADS ≥{lr_threshold_label}",
                flag_rate=n / n_total if n_total else np.nan,
                n_flag=n, cancers_captured=cap,
                precision=cap / n if n else np.nan,
                recall=cap / total_pos if total_pos else np.nan)


def lorenz_points(scores, labels, mask):
    """Lorenz curve: cumul fraction of cancers (y) vs cumul fraction of scans
    (x), sorted by descending score. Returns (x, y, gini)."""
    s = scores[mask]; y = labels[mask]
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order].astype(float)
    n = len(y_sorted); pos = y_sorted.sum()
    if pos == 0 or n == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.0
    x = np.arange(1, n + 1) / n
    yc = np.cumsum(y_sorted) / pos
    x = np.concatenate([[0.0], x])
    yc = np.concatenate([[0.0], yc])
    # Gini = 2*area between curve and diagonal, in [0,1].
    auc = np.trapz(yc, x)
    gini = 2 * (auc - 0.5)
    return x, yc, gini


def main():
    print("=" * 78)
    print("LUNG-RADS-ANCHORED RISK BINS + LORENZ CURVES (y1)")
    print("=" * 78)

    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    comp = attach_lungrads(comp)

    inc_y1, lbl_y1 = y1_label_array(comp)
    print(f"\nTest cohort: n={len(comp)} scans, y1 includable={inc_y1.sum()}, "
          f"y1 cancers={int(lbl_y1[inc_y1].sum())}")

    comp, scores = build_y1_calibrated_scores(comp)

    # 1. Risk-anchored bins per model
    print("\n" + "-" * 78)
    print("RISK-ANCHORED BINS (Lung-RADS thresholds)")
    print("-" * 78)
    rb = risk_bin_table(comp, scores, inc_y1, lbl_y1)
    for m in scores.keys():
        sub = rb[rb.model == m]
        print(f"\n{m}:")
        print(sub.to_string(index=False))
    rb_path = os.path.join(OUTPUT_DIR, "lungrads_risk_bins_y1.csv")
    rb.to_csv(rb_path, index=False)
    print(f"\n→ {rb_path}")

    # 1b. Actual Lung-RADS distribution
    print("\n" + "-" * 78)
    print("ACTUAL LUNG-RADS DISTRIBUTION (Pinsky algorithm)")
    print("-" * 78)
    lrt = lungrads_bin_table(comp, inc_y1, lbl_y1)
    print(lrt.to_string(index=False))
    lrt.to_csv(os.path.join(OUTPUT_DIR, "lungrads_actual_y1.csv"), index=False)

    # 1c. LR-binned reclassification: per LR category × model
    print("\n" + "-" * 78)
    print("LR-BINNED RECLASSIFICATION (rows=LR cat, p̂ + B5 stats per model)")
    print("-" * 78)
    reclass = lungrads_reclassification(comp, scores, inc_y1, lbl_y1,
                                        high_thr=0.15)
    reclass.to_csv(
        os.path.join(OUTPUT_DIR, "lungrads_reclass_y1_long.csv"), index=False)
    wide = lr_reclass_wide(reclass)
    # Round for display
    disp = wide.copy()
    for c in disp.columns:
        if "p̂" in c:
            disp[c] = disp[c].map(lambda v: f"{v*100:.2f}%" if pd.notna(v) else "")
        elif "%B5" in c:
            disp[c] = disp[c].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "")
    print(disp.to_string(index=False))
    wide.to_csv(
        os.path.join(OUTPUT_DIR, "lungrads_reclass_y1_wide.csv"), index=False)
    print(f"\n→ eval_output/lungrads_reclass_y1_{{long,wide}}.csv")

    # 2. Lorenz curves
    print("\n" + "-" * 78)
    print("LORENZ / CONCENTRATION CURVES")
    print("-" * 78)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="No discrimination")
    colors = {"LungEvaty": "#d62728", "Sybil": "#1f77b4",
              "Pillar-0": "#2ca02c", "PLCOm2012": "#ff7f0e"}
    gini_rows = []
    for m, sc in scores.items():
        mask = inc_y1 & ~np.isnan(sc)
        x, y, g = lorenz_points(sc, lbl_y1, mask)
        ax.plot(x, y, color=colors[m], lw=1.6, label=f"{m} (Gini={g:.3f})")
        gini_rows.append(dict(model=m, gini=g, n_eval=int(mask.sum()),
                              n_cancers=int(lbl_y1[mask].sum())))
        print(f"  {m:<12} Gini={g:.4f}  n_eval={mask.sum():>5}  "
              f"cancers={int(lbl_y1[mask].sum())}")

    # Add Lung-RADS as a step function: treat 4X>4B>4A>3>2>1 as descending score
    lr_order = {"4X": 6, "4B": 5, "4A": 4, "3": 3, "2": 2, "1": 1}
    lr = comp["lungrads_v10"].astype(str).values
    lr_score = np.array([lr_order.get(c, 0) for c in lr], dtype=float)
    mask_lr = inc_y1 & (lr_score > 0)
    x, y, g_lr = lorenz_points(lr_score, lbl_y1, mask_lr)
    ax.plot(x, y, color="#444", lw=1.6, ls=":",
            label=f"Lung-RADS (Gini={g_lr:.3f})")
    gini_rows.append(dict(model="Lung-RADS", gini=g_lr,
                          n_eval=int(mask_lr.sum()),
                          n_cancers=int(lbl_y1[mask_lr].sum())))
    print(f"  {'Lung-RADS':<12} Gini={g_lr:.4f}  n_eval={mask_lr.sum():>5}  "
          f"cancers={int(lbl_y1[mask_lr].sum())}")

    ax.set_xlabel("Cumulative fraction of scans (highest risk first)")
    ax.set_ylabel("Cumulative fraction of year-1 cancers captured")
    ax.set_title("Lorenz / concentration curves — NLST test, y1")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3)
    out_png = os.path.join(OUTPUT_DIR, "lorenz_y1.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\n→ {out_png}")

    # Zoomed view of the top decile
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5,
            label="No discrimination")
    for m, sc in scores.items():
        mask = inc_y1 & ~np.isnan(sc)
        x, y, g = lorenz_points(sc, lbl_y1, mask)
        ax.plot(x, y, color=colors[m], lw=1.6, label=f"{m} (Gini={g:.3f})")
    x_lr, y_lr, _ = lorenz_points(lr_score, lbl_y1, mask_lr)
    ax.plot(x_lr, y_lr, color="#444", lw=1.6, ls=":",
            label=f"Lung-RADS (Gini={g_lr:.3f})")

    # Vertical guides at Lung-RADS-like population fractions
    lr_counts = comp["lungrads_v10"].astype(str).value_counts()
    n_total = int(mask_lr.sum())
    f_4x  = lr_counts.get("4X", 0) / n_total
    f_4b  = (lr_counts.get("4X", 0) + lr_counts.get("4B", 0)) / n_total
    f_4a  = f_4b + lr_counts.get("4A", 0) / n_total
    f_3   = f_4a + lr_counts.get("3", 0) / n_total
    for f, lab in [(f_4x, "LR4X"), (f_4b, "+4B"), (f_4a, "+4A"), (f_3, "+3")]:
        ax.axvline(f, color="grey", lw=0.7, ls="--", alpha=0.4)
        ax.text(f, 0.02, lab, rotation=90, fontsize=7, color="grey",
                va="bottom", ha="right")

    ax.set_xlim(0, 0.10); ax.set_ylim(0, 1.01)
    ax.set_xlabel("Cumulative fraction of scans (top 10%)")
    ax.set_ylabel("Cumulative fraction of year-1 cancers captured")
    ax.set_title("Lorenz curves — top 10% zoom (y1)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_zoom = os.path.join(OUTPUT_DIR, "lorenz_y1_top10.png")
    plt.savefig(out_zoom, dpi=150)
    print(f"→ {out_zoom}")

    pd.DataFrame(gini_rows).to_csv(
        os.path.join(OUTPUT_DIR, "lorenz_gini_y1.csv"), index=False)

    # -- (A) T0-ONLY fair comparison (no longitudinal asymmetry) ------------
    print("\n" + "-" * 78)
    print("(A) T0-ONLY (info-symmetric) Lorenz + LR reclass")
    print("-" * 78)
    t0 = comp["screen_timepoint"].astype(int).values == 0
    inc_t0 = inc_y1 & t0
    print(f"  T0: n_scans={int(t0.sum())}, y1_incl={int(inc_t0.sum())}, "
          f"y1_ca={int(lbl_y1[inc_t0].sum())}")
    rb_t0 = risk_bin_table(comp, scores, inc_t0, lbl_y1)
    rb_t0.to_csv(os.path.join(OUTPUT_DIR, "lungrads_risk_bins_y1_T0.csv"),
                 index=False)
    reclass_t0 = lungrads_reclassification(comp, scores, inc_t0, lbl_y1, 0.15)
    wide_t0 = lr_reclass_wide(reclass_t0)
    disp_t0 = wide_t0.copy()
    for c in disp_t0.columns:
        if "p̂" in c:
            disp_t0[c] = disp_t0[c].map(
                lambda v: f"{v*100:.2f}%" if pd.notna(v) else "")
        elif "%B5" in c:
            disp_t0[c] = disp_t0[c].map(
                lambda v: f"{v*100:.1f}%" if pd.notna(v) else "")
    print("\n  LR reclass table (T0 only):")
    print(disp_t0.to_string(index=False))
    wide_t0.to_csv(os.path.join(OUTPUT_DIR, "lungrads_reclass_y1_T0_wide.csv"),
                   index=False)

    # T0 Lorenz (semi-log)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for m, sc in scores.items():
        mask = inc_t0 & ~np.isnan(sc)
        x, y, g = lorenz_points(sc, lbl_y1, mask)
        ax.plot(x[1:], y[1:], color=colors[m], lw=1.6,
                label=f"{m} (Gini={g:.3f})")
    mask_lr_t0 = inc_t0 & (lr_score > 0)
    x_lr, y_lr, g_lr_t0 = lorenz_points(lr_score, lbl_y1, mask_lr_t0)
    ax.plot(x_lr[1:], y_lr[1:], color="#444", lw=1.6, ls=":",
            label=f"Lung-RADS (Gini={g_lr_t0:.3f})")
    xref = np.logspace(-3.5, 0, 100)
    ax.plot(xref, xref, "k--", lw=0.8, alpha=0.5, label="No discrimination")
    ax.set_xscale("log")
    ax.set_xlim(1 / max(int(inc_t0.sum()), 1), 1); ax.set_ylim(0, 1.01)
    ax.xaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax.set_xlabel("Cumulative fraction of scans flagged (log scale)")
    ax.set_ylabel("Cumulative fraction of year-1 cancers captured")
    ax.set_title("Lorenz — T0 only (info-symmetric; Lung-RADS uses single-scan rules)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    out_t0 = os.path.join(OUTPUT_DIR, "lorenz_y1_T0_semilog.png")
    plt.savefig(out_t0, dpi=150); print(f"  → {out_t0}")

    # -- (B) PER-TIMEPOINT stratified Lorenz ---------------------------------
    print("\n" + "-" * 78)
    print("(B) PER-TIMEPOINT stratified Lorenz (T0 / T1 / T2)")
    print("-" * 78)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    tp_vals = comp["screen_timepoint"].astype(int).values
    gini_panel_rows = []
    for ax, tp in zip(axes, [0, 1, 2]):
        tp_mask = inc_y1 & (tp_vals == tp)
        n_tp = int(tp_mask.sum())
        ca_tp = int(lbl_y1[tp_mask].sum())
        print(f"  T{tp}: n_eval={n_tp}, y1_ca={ca_tp}")
        for m, sc in scores.items():
            mask = tp_mask & ~np.isnan(sc)
            x, y, g = lorenz_points(sc, lbl_y1, mask)
            ax.plot(x[1:], y[1:], color=colors[m], lw=1.6,
                    label=f"{m} (G={g:.3f})")
            gini_panel_rows.append(dict(timepoint=f"T{tp}", model=m, gini=g,
                                        n_eval=int(mask.sum()),
                                        n_cancers=int(lbl_y1[mask].sum())))
        m_lr = tp_mask & (lr_score > 0)
        x_lr, y_lr, g_lr_tp = lorenz_points(lr_score, lbl_y1, m_lr)
        ax.plot(x_lr[1:], y_lr[1:], color="#444", lw=1.6, ls=":",
                label=f"Lung-RADS (G={g_lr_tp:.3f})")
        gini_panel_rows.append(dict(timepoint=f"T{tp}", model="Lung-RADS",
                                    gini=g_lr_tp, n_eval=int(m_lr.sum()),
                                    n_cancers=int(lbl_y1[m_lr].sum())))
        ax.plot(xref, xref, "k--", lw=0.8, alpha=0.4)
        ax.set_xscale("log")
        ax.set_xlim(1 / max(n_tp, 1), 1); ax.set_ylim(0, 1.01)
        ax.xaxis.set_major_formatter(FuncFormatter(_pct_fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
        info = " (Lung-RADS single-scan)" if tp == 0 else " (Lung-RADS uses prior CT)"
        ax.set_title(f"T{tp}: n={n_tp}, y1 ca={ca_tp}{info}", fontsize=11)
        ax.set_xlabel("Cumul. fraction of scans flagged")
        if tp == 0:
            ax.set_ylabel("Cumul. fraction of year-1 cancers captured")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    out_panel = os.path.join(OUTPUT_DIR, "lorenz_y1_by_timepoint.png")
    plt.savefig(out_panel, dpi=150); print(f"  → {out_panel}")
    pd.DataFrame(gini_panel_rows).to_csv(
        os.path.join(OUTPUT_DIR, "lorenz_gini_y1_by_timepoint.csv"),
        index=False)

    # -- OPERATING POINTS: ultra-selective + LR-comparable -----------------
    print("\n" + "-" * 78)
    print("OPERATING POINTS (top-K flag rates; precision + recall)")
    print("-" * 78)
    # Matches the lr_score scale used elsewhere in this script
    # (lr_order = {"4X": 6, "4B": 5, "4A": 4, "3": 3, "2": 2, "1": 1}).
    LR_CAT_ORDER = {"1": 1, "2": 2, "3": 3, "4A": 4, "4B": 5, "4X": 6}
    fr_grid = [0.001, 0.002, 0.004, 0.005, 0.01, 0.02, 0.04, 0.05, 0.10]

    # All timepoints
    op_all = operating_points(scores, inc_y1, lbl_y1, fr_grid, "all_TP")
    lr_pts_all = [
        lr_operating_point(comp, lr_score, inc_y1, lbl_y1,
                           "4X", LR_CAT_ORDER["4X"], "all_TP"),
        lr_operating_point(comp, lr_score, inc_y1, lbl_y1,
                           "4B", LR_CAT_ORDER["4B"], "all_TP"),
        lr_operating_point(comp, lr_score, inc_y1, lbl_y1,
                           "4A", LR_CAT_ORDER["4A"], "all_TP"),
        lr_operating_point(comp, lr_score, inc_y1, lbl_y1,
                           "3",  LR_CAT_ORDER["3"],  "all_TP"),
    ]
    # T0 only
    op_t0 = operating_points(scores, inc_t0, lbl_y1, fr_grid, "T0_only")
    lr_pts_t0 = [
        lr_operating_point(comp, lr_score, inc_t0, lbl_y1,
                           "4X", LR_CAT_ORDER["4X"], "T0_only"),
        lr_operating_point(comp, lr_score, inc_t0, lbl_y1,
                           "4B", LR_CAT_ORDER["4B"], "T0_only"),
        lr_operating_point(comp, lr_score, inc_t0, lbl_y1,
                           "4A", LR_CAT_ORDER["4A"], "T0_only"),
        lr_operating_point(comp, lr_score, inc_t0, lbl_y1,
                           "3",  LR_CAT_ORDER["3"],  "T0_only"),
    ]
    op_all_full = pd.concat([op_all, pd.DataFrame(lr_pts_all)],
                            ignore_index=True)
    op_t0_full = pd.concat([op_t0, pd.DataFrame(lr_pts_t0)], ignore_index=True)

    def _fmt_op(df):
        d = df.copy()
        d["flag_rate"] = (d["flag_rate"] * 100).map(
            lambda v: f"{v:.2f}%" if pd.notna(v) else "")
        d["precision"] = (d["precision"] * 100).map(
            lambda v: f"{v:.1f}%" if pd.notna(v) else "")
        d["recall"] = (d["recall"] * 100).map(
            lambda v: f"{v:.1f}%" if pd.notna(v) else "")
        return d

    print("\n  All timepoints:")
    print(_fmt_op(op_all_full).to_string(index=False))
    print("\n  T0 only:")
    print(_fmt_op(op_t0_full).to_string(index=False))
    op_all_full.to_csv(
        os.path.join(OUTPUT_DIR, "operating_points_y1_all_TP.csv"), index=False)
    op_t0_full.to_csv(
        os.path.join(OUTPUT_DIR, "operating_points_y1_T0.csv"), index=False)

    # -- MULTI-HORIZON Lorenz + operating points (y1, y3, y6) -------------
    print("\n" + "=" * 78)
    print("MULTI-HORIZON LR-ADJACENT ANALYSIS")
    print("=" * 78)

    def label_for_year(year_idx):
        """Complete-case labels at horizon year (0-indexed fu)."""
        times = comp["time_at_event"].values
        golds = comp["gold"].values
        inc = np.zeros(len(comp), bool)
        lbl = np.zeros(len(comp), int)
        for i, (t, g) in enumerate(zip(times, golds)):
            ok, y = _include(t, g, year_idx)
            inc[i] = ok
            lbl[i] = y
        return inc, lbl

    # Parse LE 6-vector once for non-y1 years
    le_vec = np.array([
        [float(v) for v in r.strip("[]").split(",")]
        for r in comp["cancer_risk"].values
    ])  # shape (N, 6)

    def scores_for_year(year):
        """Return calibrated p̂_year for all 4 models."""
        le_raw = le_vec[:, year - 1]
        le_cal = lung_platt[year].predict_proba(
            _logit(le_raw).reshape(-1, 1))[:, 1]

        syb_cal = comp[f"sv1c_risk_{year}"].values.astype(float)

        pi_raw = comp[f"pillar_user_y{year}"].values.astype(float)
        mp = ~np.isnan(pi_raw)
        pi_cal = np.full(len(comp), np.nan)
        pi_cal[mp] = pillar_platt[year].predict_proba(
            _logit(pi_raw[mp]).reshape(-1, 1))[:, 1]

        plcom_raw_y = pd.Series(index=comp.index, dtype=float)
        for i, pid in enumerate(comp["pid"].values):
            r = pid_to_feat.loc[pid] if pid in pid_to_feat.index else None
            plcom_raw_y.iat[i] = (
                float(plcom2012_risk(r, year=year)) if r is not None
                else np.nan)
        mm = ~plcom_raw_y.isna()
        pl_cal = np.full(len(comp), np.nan)
        pl_cal[mm] = plcom_platt[year].predict_proba(
            _logit(plcom_raw_y[mm].values).reshape(-1, 1))[:, 1]

        return {"LungEvaty": le_cal, "Sybil": syb_cal,
                "Pillar-0": pi_cal, "PLCOm2012": pl_cal}

    # Expose pid_to_feat / pillar_platt / plcom_platt / lung_platt to scope
    pillar_platt = fit_pillar_platt()
    lung_platt = fit_lung_platt()
    part = load_participant(PARTICIPANT_CSV)
    dev_plcom = build_plcom_dev(part)
    plcom_platt = fit_plcom_platt(dev_plcom)
    base_cols = ["pid", "age", "race", "pkyr", "smokeyr", "cigsmok"]
    base_cols = [c for c in base_cols if c in part.columns]
    from decile_stratification_devfit import _plcom_extra_cols as _xc
    extras = _xc(part)
    base = part[base_cols + extras].drop_duplicates("pid").copy()
    base["pid"] = base["pid"].astype(str)
    pid_to_feat = base.set_index("pid")
    from uspstf_plcom import plcom2012_risk

    HORIZONS = [1, 3, 6]
    multi_op_rows = []
    multi_gini_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)
    for ax, year in zip(axes, HORIZONS):
        fu = year - 1
        inc_h, lbl_h = label_for_year(fu)
        n_h = int(inc_h.sum()); ca_h = int(lbl_h[inc_h].sum())
        print(f"\n  y{year}: n_incl={n_h}, cancers={ca_h}")
        scores_h = scores_for_year(year)

        for m, sc in scores_h.items():
            mask = inc_h & ~np.isnan(sc)
            x, y, g = lorenz_points(sc, lbl_h, mask)
            ax.plot(x[1:], y[1:], color=colors[m], lw=1.6,
                    label=f"{m} (G={g:.3f})")
            multi_gini_rows.append(dict(year=year, model=m, gini=g,
                                        n_eval=int(mask.sum()),
                                        n_cancers=int(lbl_h[mask].sum())))
        m_lr = inc_h & (lr_score > 0)
        x_lr, y_lr, g_lr_h = lorenz_points(lr_score, lbl_h, m_lr)
        ax.plot(x_lr[1:], y_lr[1:], color="#444", lw=1.6, ls=":",
                label=f"Lung-RADS (G={g_lr_h:.3f})")
        multi_gini_rows.append(dict(year=year, model="Lung-RADS",
                                    gini=g_lr_h, n_eval=int(m_lr.sum()),
                                    n_cancers=int(lbl_h[m_lr].sum())))

        # Per-horizon operating points
        op_y = operating_points(scores_h, inc_h, lbl_h, fr_grid,
                                scope_label=f"y{year}")
        for label, rank in [("4X", LR_CAT_ORDER["4X"]),
                            ("4B", LR_CAT_ORDER["4B"]),
                            ("4A", LR_CAT_ORDER["4A"]),
                            ("3",  LR_CAT_ORDER["3"])]:
            op_y = pd.concat([op_y, pd.DataFrame([lr_operating_point(
                comp, lr_score, inc_h, lbl_h, label, rank,
                scope_label=f"y{year}")])], ignore_index=True)
        multi_op_rows.append(op_y)

        # Mark LR ≥4B operating point
        lr4b = op_y[op_y.model == "Lung-RADS ≥4B"].iloc[0]
        ax.plot(lr4b.flag_rate, lr4b.recall, marker="^", color="#222",
                markersize=8, markeredgecolor="white", markeredgewidth=1.2,
                zorder=5)

        ax.plot(xref, xref, "k--", lw=0.8, alpha=0.4)
        ax.set_xscale("log")
        ax.set_xlim(1 / max(n_h, 1), 1); ax.set_ylim(0, 1.01)
        ax.xaxis.set_major_formatter(FuncFormatter(_pct_fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
        ax.set_xlabel("Cumul. fraction of scans flagged")
        if year == 1:
            ax.set_ylabel("Cumul. fraction of cancers captured")
        ax.set_title(f"y{year}: n={n_h}, cancers={ca_h} (complete-case)",
                     fontsize=11)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_mh = os.path.join(OUTPUT_DIR, "lorenz_multihorizon.png")
    plt.savefig(out_mh, dpi=150); print(f"\n→ {out_mh}")

    multi_op_full = pd.concat(multi_op_rows, ignore_index=True)
    multi_op_full.to_csv(
        os.path.join(OUTPUT_DIR, "operating_points_multihorizon.csv"),
        index=False)
    pd.DataFrame(multi_gini_rows).to_csv(
        os.path.join(OUTPUT_DIR, "lorenz_gini_multihorizon.csv"), index=False)
    print(f"→ operating_points_multihorizon.csv")
    print(f"→ lorenz_gini_multihorizon.csv")

    # -- COMBINED 2-PANEL FIGURE -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, scope, scope_label, inc_mask in [
        (axes[0], "all_TP", "All timepoints (LR uses prior CT at T1/T2)", inc_y1),
        (axes[1], "T0_only", "T0 only (info-symmetric)", inc_t0),
    ]:
        n_scope = int(inc_mask.sum())
        ca_scope = int(lbl_y1[inc_mask].sum())
        for m, sc in scores.items():
            mask = inc_mask & ~np.isnan(sc)
            x, y, g = lorenz_points(sc, lbl_y1, mask)
            ax.plot(x[1:], y[1:], color=colors[m], lw=1.7,
                    label=f"{m} (Gini={g:.3f})")
        m_lr = inc_mask & (lr_score > 0)
        x_lr, y_lr, g_lr_s = lorenz_points(lr_score, lbl_y1, m_lr)
        ax.plot(x_lr[1:], y_lr[1:], color="#444", lw=1.7, ls=":",
                label=f"Lung-RADS (Gini={g_lr_s:.3f})")

        ax.plot(xref, xref, "k--", lw=0.8, alpha=0.45,
                label="No discrimination")

        # Operating-point markers for LR thresholds
        op_rows = op_all_full if scope == "all_TP" else op_t0_full
        for lr_label, marker, ms, color in [
            ("Lung-RADS ≥4X", "s", 9, "#222"),
            ("Lung-RADS ≥4B", "^", 9, "#222"),
        ]:
            row = op_rows[op_rows.model == lr_label].iloc[0]
            ax.plot(row.flag_rate, row.recall, marker=marker, color=color,
                    markersize=ms, markeredgecolor="white",
                    markeredgewidth=1.2, zorder=5)
            ax.annotate(
                f"{lr_label.replace('Lung-RADS ', 'LR')}\n"
                f"{row.flag_rate*100:.2f}%, prec={row.precision*100:.1f}%, "
                f"recall={row.recall*100:.1f}%",
                xy=(row.flag_rate, row.recall),
                xytext=(8, -28 if "4X" in lr_label else 12),
                textcoords="offset points",
                fontsize=8, color="#222",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#555"),
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="#999", alpha=0.9),
            )

        # LungEvaty matched-volume marker at LR ≥4X flag rate
        lr_4x_fr = op_rows[op_rows.model == "Lung-RADS ≥4X"].iloc[0].flag_rate
        le_row = op_rows[
            (op_rows.model == "LungEvaty")
            & (np.isclose(op_rows.flag_rate, round(lr_4x_fr, 3), atol=0.001))]
        if len(le_row) == 0:
            # interpolate from operating_points table at closest fr
            le_sub = op_rows[op_rows.model == "LungEvaty"].copy()
            idx = (le_sub.flag_rate - lr_4x_fr).abs().idxmin()
            le_row = le_sub.loc[[idx]]
        if len(le_row):
            r = le_row.iloc[0]
            ax.plot(r.flag_rate, r.recall, marker="o",
                    color=colors["LungEvaty"], markersize=10,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=6)
            ax.annotate(
                f"LungEvaty @ {r.flag_rate*100:.2f}%\n"
                f"prec={r.precision*100:.1f}%, recall={r.recall*100:.1f}%",
                xy=(r.flag_rate, r.recall),
                xytext=(8, 30),
                textcoords="offset points",
                fontsize=8, color=colors["LungEvaty"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.6,
                                color=colors["LungEvaty"]),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=colors["LungEvaty"], alpha=0.95),
            )

        ax.set_xscale("log")
        ax.set_xlim(0.5 / n_scope, 1); ax.set_ylim(0, 1.01)
        ax.xaxis.set_major_formatter(FuncFormatter(_pct_fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
        ax.set_xlabel("Cumulative fraction of scans flagged (log scale)")
        if scope == "all_TP":
            ax.set_ylabel("Cumulative fraction of year-1 cancers captured")
        ax.set_title(f"{scope_label}\nn={n_scope}, y1 cancers={ca_scope}",
                     fontsize=11)
        ax.legend(loc="upper left", fontsize=8.5)
        ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    out_combo = os.path.join(OUTPUT_DIR, "lorenz_y1_paper_figure.png")
    plt.savefig(out_combo, dpi=150); print(f"\n→ {out_combo}")

    # -- Semi-log (log X, linear Y) view -----------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for m, sc in scores.items():
        mask = inc_y1 & ~np.isnan(sc)
        x, y, g = lorenz_points(sc, lbl_y1, mask)
        ax.plot(x[1:], y[1:], color=colors[m], lw=1.6,
                label=f"{m} (Gini={g:.3f})")
    x_lr, y_lr, _ = lorenz_points(lr_score, lbl_y1, mask_lr)
    ax.plot(x_lr[1:], y_lr[1:], color="#444", lw=1.6, ls=":",
            label=f"Lung-RADS (Gini={g_lr:.3f})")
    # Diagonal becomes y = x on linear y, log x
    xref = np.logspace(-4, 0, 100)
    ax.plot(xref, xref, "k--", lw=0.8, alpha=0.5, label="No discrimination")
    for f, lab in [(f_4x, "LR4X"), (f_4b, "+4B"), (f_4a, "+4A"), (f_3, "+3")]:
        ax.axvline(f, color="grey", lw=0.7, ls="--", alpha=0.4)
        ax.text(f, 0.02, lab, rotation=90, fontsize=7, color="grey",
                va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlim(1 / 6223, 1); ax.set_ylim(0, 1.01)
    ax.xaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, p: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax.set_xlabel("Cumulative fraction of scans flagged (log scale)")
    ax.set_ylabel("Cumulative fraction of year-1 cancers captured")
    ax.set_title("Lorenz curves — semi-log (linear Y, log X)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    out_sl = os.path.join(OUTPUT_DIR, "lorenz_y1_semilog.png")
    plt.savefig(out_sl, dpi=150); print(f"→ {out_sl}")

    # -- Log-log view ------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for m, sc in scores.items():
        mask = inc_y1 & ~np.isnan(sc)
        x, y, g = lorenz_points(sc, lbl_y1, mask)
        nz = (x > 0) & (y > 0)
        ax.plot(x[nz], y[nz], color=colors[m], lw=1.6,
                label=f"{m} (Gini={g:.3f})")
    x_lr, y_lr, _ = lorenz_points(lr_score, lbl_y1, mask_lr)
    nz = (x_lr > 0) & (y_lr > 0)
    ax.plot(x_lr[nz], y_lr[nz], color="#444", lw=1.6, ls=":",
            label=f"Lung-RADS (Gini={g_lr:.3f})")
    ax.plot(xref, xref, "k--", lw=0.8, alpha=0.5, label="No discrimination")
    for f, lab in [(f_4x, "LR4X"), (f_4b, "+4B"), (f_4a, "+4A"), (f_3, "+3")]:
        ax.axvline(f, color="grey", lw=0.7, ls="--", alpha=0.4)
        ax.text(f, 0.011, lab, rotation=90, fontsize=7, color="grey",
                va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1 / 6223, 1); ax.set_ylim(1 / 82, 1.01)
    ax.xaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax.set_xlabel("Cumulative fraction of scans flagged (log scale)")
    ax.set_ylabel("Cumulative fraction of year-1 cancers captured (log)")
    ax.set_title("Lorenz curves — log-log")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    out_ll = os.path.join(OUTPUT_DIR, "lorenz_y1_loglog.png")
    plt.savefig(out_ll, dpi=150); print(f"→ {out_ll}")


if __name__ == "__main__":
    main()
