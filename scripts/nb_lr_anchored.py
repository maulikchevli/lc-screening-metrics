#!/usr/bin/env python3
"""DCA / Net-Benefit at LR-anchored probability thresholds.

Threshold grid p_t ∈ {0.01, 0.02, 0.05, 0.10, 0.15} — aligned to Lung-RADS
risk-of-malignancy reference points (LR3 starts at 1%; LR4A spans 5–15%;
LR4B begins at 15%).

Years: 1, 3, 6. IPCW (Uno) for y1/y3 (consistent with paper convention);
complete-case for y6.

Patient-clustered bootstrap n=1000 with shared resample indices for paired
model comparison.

Replaces the decile_cum anchor rows in nb_at_decile_thresholds_devfit.csv.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import (
    _include, _ipcw_weights, load_participant, PARTICIPANT_CSV, OUTPUT_DIR,
)
from decile_stratification_devfit import (
    fit_lung_platt, fit_plcom_platt, fit_pillar_platt,
    attach_user_pillar, build_plcom_dev, _logit,
    _ipcw_nb, _cc_nb, _plcom_extra_cols,
)
from uspstf_plcom import plcom2012_risk

P_T_GRID = [0.01, 0.02, 0.05, 0.10, 0.15]
YEARS_IPCW = [1, 3]
YEARS_CC = [6]
N_BOOT = 1000
RNG_SEED = 42


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

    out = {}  # year -> {model_name: scores}
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


def nb_one(scores, times, events, mask, year, p_t):
    s = scores[mask]
    t = times[mask].astype(float)
    e = events[mask].astype(int)
    fu = year - 1
    if year <= 3:
        return _ipcw_nb(s, t, e, fu, p_t)[0]
    else:
        # complete-case
        inc = np.array([
            _include(ti, ei, fu)[0] for ti, ei in zip(t, e)])
        if not inc.any():
            return np.nan
        lbl = np.array([_include(ti, ei, fu)[1]
                        for ti, ei in zip(t, e)])
        return _cc_nb(s[inc], lbl[inc], p_t)[0]


def _boot_worker(args):
    (model, scores, times, events, year, p_t, pid_to_rows, pids_uniq,
     n_boot, seed) = args
    rng = np.random.default_rng(seed)
    mask_full = ~np.isnan(scores)
    point_nb = nb_one(scores, times, events, mask_full, year, p_t)
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(pids_uniq, size=len(pids_uniq), replace=True)
        idx = np.concatenate([pid_to_rows[p] for p in sample])
        m = np.zeros(len(scores), bool); m[idx] = True
        m = m & mask_full
        if m.sum() < 20:
            boot.append(np.nan); continue
        try:
            v = nb_one(scores, times, events, m, year, p_t)
        except Exception:
            v = np.nan
        boot.append(v)
    boot = np.array(boot, dtype=float)
    lo = float(np.nanpercentile(boot, 2.5)) if np.isfinite(boot).any() else np.nan
    hi = float(np.nanpercentile(boot, 97.5)) if np.isfinite(boot).any() else np.nan
    return dict(model=model, year=year, p_t=p_t,
                nb=point_nb, nb_lo=lo, nb_hi=hi)


def main():
    print("=" * 78)
    print("DCA / NB at LR-anchored thresholds (p_t = 1%, 2%, 5%, 10%, 15%)")
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

    tasks = []
    for year in [1, 3, 6]:
        for p_t in P_T_GRID:
            for model, sc in year_to_scores[year].items():
                tasks.append((model, sc, times, events, year, p_t,
                              pid_to_rows, pids_uniq, N_BOOT, RNG_SEED))

    print(f"[run] {len(tasks)} cells × {N_BOOT} bootstrap (process pool) …")
    rows = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(_boot_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            if i % 10 == 0:
                print(f"  … {i}/{len(tasks)} done")

    df = pd.DataFrame(rows).sort_values(["year", "p_t", "model"])
    out = os.path.join(OUTPUT_DIR, "nb_lr_anchored.csv")
    df.to_csv(out, index=False)
    print(f"\n→ {out}")

    # Compact display
    for year in [1, 3, 6]:
        sub = df[df.year == year]
        print(f"\n  y{year}:")
        for p_t in P_T_GRID:
            row = sub[sub.p_t == p_t]
            line = f"    p_t={int(p_t*100)}%: "
            for m in ["LungEvaty", "Pillar-0", "Sybil", "PLCOm2012"]:
                r = row[row.model == m]
                if len(r):
                    rr = r.iloc[0]
                    line += (f"{m}: {rr.nb*1000:.2f} "
                             f"[{rr.nb_lo*1000:.2f},{rr.nb_hi*1000:.2f}]  ")
            print(line.rstrip())


if __name__ == "__main__":
    main()
