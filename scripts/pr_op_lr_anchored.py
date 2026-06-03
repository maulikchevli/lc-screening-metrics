#!/usr/bin/env python3
"""PR-curve operating points at LR-anchored + ultra-selective + R95 anchors.

Multi-horizon (y1/y3/y6), all 4 models.

Anchors (replaces FR10/FR20/FR40/FR70 from task3_pr_curves_op_points.csv):
- LR ≥4X-matched flag rate (LR's most-selective)
- LR ≥4B-matched flag rate (LR's workup-recommended volume)
- LR ≥4A-matched flag rate (LR's short-interval CT volume)
- Ultra-selective: top 0.1%, 0.2%, 0.4% by AI score
- R95: smallest threshold achieving sensitivity ≥ 95%   (kept; clinically meaningful)

Output: pr_op_lr_anchored.csv with point estimates + patient-clustered
bootstrap n=1000 CIs for precision and recall.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import (
    _include, load_participant, PARTICIPANT_CSV, OUTPUT_DIR,
)
from decile_stratification_devfit import (
    fit_lung_platt, fit_plcom_platt, fit_pillar_platt,
    attach_user_pillar, build_plcom_dev, _logit, _plcom_extra_cols,
)
from uspstf_plcom import plcom2012_risk

LUNGRADS_CSV = os.path.join(OUTPUT_DIR, "nlst_lungrads_pinsky.csv")
LR_ORDER = {"4X": 6, "4B": 5, "4A": 4, "3": 3, "2": 2, "1": 1}
ULTRA_SELECT = [0.001, 0.002, 0.004]
N_BOOT = 1000
RNG_SEED = 42


def attach_lungrads(comp):
    lr = pd.read_csv(LUNGRADS_CSV)
    lr["pid"] = lr["pid"].astype(str)
    lr["study_yr"] = lr["study_yr"].astype(int)
    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    comp["study_yr"] = comp["screen_timepoint"].astype(int)
    return comp.merge(lr[["pid", "study_yr", "lungrads_v10"]],
                      on=["pid", "study_yr"], how="left")


def build_scores_all_years(comp):
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


def metrics_at_top_k(s, y, k):
    """Top-k by score → precision/recall."""
    order = np.argsort(-s, kind="mergesort")[:k]
    n_pos = int(y.sum())
    cap = int(y[order].sum())
    return cap / k if k else np.nan, cap / n_pos if n_pos else np.nan


def metrics_at_r95(s, y):
    """Threshold giving sensitivity ≥ 95% with smallest flag set."""
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.nan, np.nan, np.nan
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order].astype(int)
    cum_pos = np.cumsum(y_sorted)
    needed = int(np.ceil(0.95 * n_pos))
    if cum_pos.max() < needed:
        return np.nan, np.nan, np.nan
    idx = int(np.searchsorted(cum_pos, needed)) + 1
    k = idx
    cap = int(cum_pos[k - 1])
    return cap / k, cap / n_pos, k / len(s)


def _boot_worker(args):
    (model, scores, labels, lr_score, year, anchor, anchor_val,
     pid_to_rows, pids_uniq, n_boot, seed) = args
    rng = np.random.default_rng(seed)
    mask_full = ~np.isnan(scores)
    s = scores[mask_full]; y = labels[mask_full]; lr = lr_score[mask_full]

    def compute_one(s, y, lr):
        if anchor == "ultra":
            k = max(1, int(round(anchor_val * len(s))))
            k = min(k, len(s))
            return metrics_at_top_k(s, y, k) + (k / len(s),)
        elif anchor == "lr_match":
            rank_floor = LR_ORDER[anchor_val]
            n_flag = int((lr >= rank_floor).sum())
            if n_flag == 0:
                return (np.nan, np.nan, 0.0)
            return metrics_at_top_k(s, y, n_flag) + (n_flag / len(s),)
        elif anchor == "lr_own":
            rank_floor = LR_ORDER[anchor_val]
            sel = lr >= rank_floor
            n_flag = int(sel.sum())
            if n_flag == 0:
                return (np.nan, np.nan, 0.0)
            cap = int(y[sel].sum())
            n_pos = int(y.sum())
            return (cap / n_flag, cap / n_pos if n_pos else np.nan,
                    n_flag / len(s))
        elif anchor == "r95":
            return metrics_at_r95(s, y)
        else:
            return (np.nan, np.nan, np.nan)

    p, r, fr = compute_one(s, y, lr)
    boot_p, boot_r = [], []
    for _ in range(n_boot):
        sample = rng.choice(pids_uniq, size=len(pids_uniq), replace=True)
        idx = np.concatenate([pid_to_rows[p_] for p_ in sample])
        m = np.zeros(len(scores), bool); m[idx] = True
        m = m & mask_full
        if m.sum() < 20:
            boot_p.append(np.nan); boot_r.append(np.nan); continue
        bp, br, _ = compute_one(scores[m], labels[m], lr_score[m])
        boot_p.append(bp); boot_r.append(br)
    bp = np.array(boot_p); br = np.array(boot_r)
    return dict(model=model, year=year, anchor=anchor, anchor_val=str(anchor_val),
                flag_rate=fr, precision=p, recall=r,
                prec_lo=float(np.nanpercentile(bp, 2.5)) if np.isfinite(bp).any() else np.nan,
                prec_hi=float(np.nanpercentile(bp, 97.5)) if np.isfinite(bp).any() else np.nan,
                rec_lo=float(np.nanpercentile(br, 2.5)) if np.isfinite(br).any() else np.nan,
                rec_hi=float(np.nanpercentile(br, 97.5)) if np.isfinite(br).any() else np.nan)


def label_for_year(comp, year):
    times = comp["time_at_event"].values
    golds = comp["gold"].values
    fu = year - 1
    inc = np.array([_include(t, g, fu)[0] for t, g in zip(times, golds)])
    lbl = np.array([_include(t, g, fu)[1] for t, g in zip(times, golds)])
    return inc, lbl


def main():
    print("=" * 78)
    print("PR OPERATING POINTS at LR-anchored + ultra-selective + R95")
    print("=" * 78)
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    comp = attach_lungrads(comp)
    comp, year_to_scores = build_scores_all_years(comp)
    lr_score = np.array([LR_ORDER.get(str(c), 0)
                         for c in comp["lungrads_v10"].astype(str).values],
                        dtype=float)
    pids = comp["pid"].values
    pid_to_rows = {}
    for p, i in zip(pids, np.arange(len(pids))):
        pid_to_rows.setdefault(p, []).append(i)
    pid_to_rows = {k: np.asarray(v) for k, v in pid_to_rows.items()}
    pids_uniq = np.array(list(pid_to_rows.keys()))

    anchors = (
        [("ultra", u) for u in ULTRA_SELECT]
        + [("lr_match", c) for c in ["4X", "4B", "4A", "3"]]
        + [("lr_own", c) for c in ["4X", "4B", "4A", "3"]]
        + [("r95", None)]
    )

    tasks = []
    for year in [1, 3, 6]:
        inc, lbl = label_for_year(comp, year)
        # Restrict comp arrays to includable scans
        for model, sc in year_to_scores[year].items():
            sc_m = np.where(inc, sc, np.nan)
            for anchor, anchor_val in anchors:
                if anchor == "lr_own" and model != "LungEvaty":
                    continue  # lr_own is a reference per-year, not per-model
                tasks.append((
                    model, sc_m, lbl, lr_score, year, anchor, anchor_val,
                    pid_to_rows, pids_uniq, N_BOOT, RNG_SEED,
                ))

    print(f"[run] {len(tasks)} cells × {N_BOOT} bootstrap …")
    rows = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(_boot_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            if i % 20 == 0:
                print(f"  … {i}/{len(tasks)} done")
    df = pd.DataFrame(rows).sort_values(["year", "anchor", "anchor_val", "model"])
    out = os.path.join(OUTPUT_DIR, "pr_op_lr_anchored.csv")
    df.to_csv(out, index=False)
    print(f"\n→ {out}")
    # Compact print
    for year in [1, 3, 6]:
        print(f"\n  y{year}:")
        sub = df[df.year == year]
        for _, r in sub.iterrows():
            a = f"{r.anchor}:{r.anchor_val}"
            print(f"    {a:<16} {r.model:<10} fr={r.flag_rate*100:.2f}% "
                  f"prec={r.precision*100:.1f}% recall={r.recall*100:.1f}% "
                  f"[CI: {r.prec_lo*100:.1f}-{r.prec_hi*100:.1f}, "
                  f"{r.rec_lo*100:.1f}-{r.rec_hi*100:.1f}]")


if __name__ == "__main__":
    main()
