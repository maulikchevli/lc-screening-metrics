#!/usr/bin/env python3
"""Calibration assessment per Lung-RADS category.

For each LR category (LR1, LR2, LR3, LR4A, LR4B, LR4X) and each model:
- Mean predicted p̂_y1 (calibrated)
- Mean observed cancer rate (y1 complete-case)
- Difference (calibration error within the LR bin)
- Patient-clustered bootstrap n=1000 CIs on the difference

Replaces the per-decile rows in decile_calibration_devfit.csv. The
continuous calibration belts (calibration_belt_*.png) are bin-free and
need no replacement.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import _include, load_participant, PARTICIPANT_CSV, OUTPUT_DIR
from decile_stratification_devfit import (
    fit_lung_platt, fit_plcom_platt, fit_pillar_platt,
    attach_user_pillar, build_plcom_dev, _logit, _plcom_extra_cols,
)
from uspstf_plcom import plcom2012_risk

LUNGRADS_CSV = os.path.join(OUTPUT_DIR, "nlst_lungrads_pinsky.csv")
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


def build_scores_year(comp, year):
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
    return comp, {"LungEvaty": le_cal, "Sybil": syb_cal,
                  "Pillar-0": pi_cal, "PLCOm2012": pl_cal}


def y1_inc_lbl(comp):
    times = comp["time_at_event"].values
    golds = comp["gold"].values
    inc = np.zeros(len(comp), bool)
    lbl = np.zeros(len(comp), int)
    for i, (t, g) in enumerate(zip(times, golds)):
        ok, y = _include(t, g, 0)
        inc[i] = ok; lbl[i] = y
    return inc, lbl


def _boot_worker(args):
    (sg_var, lr_cat, model, scores, labels, level_mask, pids, n_boot, seed) = args
    rng = np.random.default_rng(seed)
    sub_idx = np.where(level_mask)[0]
    if len(sub_idx) < 5:
        return None
    pid_to_rows = {}
    for p, i in zip(pids[sub_idx], sub_idx):
        pid_to_rows.setdefault(p, []).append(i)
    pid_to_rows = {k: np.asarray(v) for k, v in pid_to_rows.items()}
    sub_pids = np.array(list(pid_to_rows.keys()))

    def stats(m):
        sm = m & ~np.isnan(scores)
        if sm.sum() == 0:
            return np.nan, np.nan, np.nan
        mp = float(np.nanmean(scores[sm]))
        mo = float(labels[sm].mean())
        return mp, mo, mp - mo

    mp_pt, mo_pt, diff_pt = stats(level_mask)
    diffs = []
    for _ in range(n_boot):
        sample = rng.choice(sub_pids, size=len(sub_pids), replace=True)
        idx = np.concatenate([pid_to_rows[p] for p in sample])
        m = np.zeros(len(scores), bool); m[idx] = True
        _, _, d = stats(m)
        diffs.append(d)
    diffs = np.array(diffs)
    return dict(lr_cat=lr_cat, model=model,
                n=int(level_mask.sum()), n_cancers=int(labels[level_mask].sum()),
                mean_pred=mp_pt, mean_obs=mo_pt, diff=diff_pt,
                diff_lo=float(np.nanpercentile(diffs, 2.5)),
                diff_hi=float(np.nanpercentile(diffs, 97.5)))


def main():
    print("=" * 78)
    print("CALIBRATION per Lung-RADS category (y1)")
    print("=" * 78)
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    comp = attach_lungrads(comp)
    comp, scores = build_scores_year(comp, 1)
    inc_y1, lbl_y1 = y1_inc_lbl(comp)
    pids = comp["pid"].values
    lr_arr = comp["lungrads_v10"].astype(str).values

    tasks = []
    for cat in ["1", "2", "3", "4A", "4B", "4X"]:
        cat_mask = inc_y1 & (lr_arr == cat)
        for model, sc in scores.items():
            tasks.append((f"LR{cat}", f"LR{cat}", model, sc, lbl_y1,
                          cat_mask, pids, N_BOOT, RNG_SEED))

    print(f"[run] {len(tasks)} cells × {N_BOOT} bootstrap …")
    rows = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(_boot_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            if r is not None:
                rows.append(r)

    df = pd.DataFrame(rows).sort_values(["lr_cat", "model"])
    out = os.path.join(OUTPUT_DIR, "calib_lr_anchored_y1.csv")
    df.to_csv(out, index=False)
    print(f"\n→ {out}")

    # Compact display
    for cat in ["LR1", "LR2", "LR3", "LR4A", "LR4B", "LR4X"]:
        sub = df[df.lr_cat == cat]
        if not len(sub): continue
        print(f"\n  {cat} (N={sub.iloc[0].n}, ca={sub.iloc[0].n_cancers}):")
        for _, r in sub.iterrows():
            print(f"    {r.model:<10} pred={r.mean_pred*100:.2f}% "
                  f"obs={r.mean_obs*100:.2f}% diff={r.diff*100:+.2f}pp "
                  f"[{r.diff_lo*100:+.2f},{r.diff_hi*100:+.2f}]")


if __name__ == "__main__":
    main()
