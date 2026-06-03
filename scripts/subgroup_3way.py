#!/usr/bin/env python3
"""Compute year-1 subgroup PR-AUC + Sp@90% with paired bootstrap CIs for
all 3 raw models (Lungevity, Sybil 1.5, Sybil v1.0 raw) on the 4-way set."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import (
    _include, _specificity_at_sensitivity, _subgroup_mask,
    OUTPUT_DIR,
)

RNG_SEED = 42
N_BOOT = 1000


def _ci(arr):
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def _resample(rng, pids, pid_to_rows):
    unique = np.array(sorted(set(pids)))
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([pid_to_rows[p] for p in sampled])


def main():
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)

    masks = _subgroup_mask(comp)
    rows = []
    for name, mask in masks.items():
        sub = comp[mask].copy()
        if len(sub) < 20:
            continue
        # Year 1, with raw v1.0 dropping NaN (year-1 has none anyway)
        L, S15, V1r, labels, pids = [], [], [], [], []
        for _, r in sub.iterrows():
            inc, lbl = _include(r["time_at_event"], r["gold"], 0)
            v1r = r["sv1r_risk_1"]
            if inc and not np.isnan(v1r):
                L.append(r["risk_score"])
                S15.append(r["sybil_risk_1"])
                V1r.append(v1r)
                labels.append(lbl)
                pids.append(r["pid"])
        L, S15, V1r = map(np.array, (L, S15, V1r))
        y = np.array(labels); pids = np.array(pids)
        if y.sum() == 0:
            continue
        pid_to_rows = {p: np.where(pids == p)[0] for p in set(pids)}

        def stats(s):
            try:
                ap = average_precision_score(y, s)
                fpr, tpr, _ = roc_curve(y, s)
                sp = _specificity_at_sensitivity(fpr, tpr, 0.90)
                return ap, sp
            except Exception:
                return np.nan, np.nan

        ap_l, sp_l = stats(L)
        ap_15, sp_15 = stats(S15)
        ap_v1, sp_v1 = stats(V1r)

        rng = np.random.default_rng(RNG_SEED + hash(name) % 1000)
        b_ap_l, b_sp_l = [], []
        b_ap_15, b_sp_15 = [], []
        b_ap_v1, b_sp_v1 = [], []
        for _ in range(N_BOOT):
            idx = _resample(rng, pids, pid_to_rows)
            yl = y[idx]
            if yl.sum() == 0 or yl.sum() == len(yl):
                continue
            try:
                ap_lb, sp_lb = stats_inner(yl, L[idx])
                ap_15b, sp_15b = stats_inner(yl, S15[idx])
                ap_v1b, sp_v1b = stats_inner(yl, V1r[idx])
                if not np.isnan(ap_lb): b_ap_l.append(ap_lb)
                if not np.isnan(sp_lb): b_sp_l.append(sp_lb)
                if not np.isnan(ap_15b): b_ap_15.append(ap_15b)
                if not np.isnan(sp_15b): b_sp_15.append(sp_15b)
                if not np.isnan(ap_v1b): b_ap_v1.append(ap_v1b)
                if not np.isnan(sp_v1b): b_sp_v1.append(sp_v1b)
            except Exception:
                pass

        rows.append(dict(
            Subgroup=name, N=len(y), N_cancer=int(y.sum()),
            L_PRAUC=ap_l, L_PRAUC_lo=_ci(b_ap_l)[0], L_PRAUC_hi=_ci(b_ap_l)[1],
            L_Sp90=sp_l, L_Sp90_lo=_ci(b_sp_l)[0], L_Sp90_hi=_ci(b_sp_l)[1],
            S15_PRAUC=ap_15, S15_PRAUC_lo=_ci(b_ap_15)[0], S15_PRAUC_hi=_ci(b_ap_15)[1],
            S15_Sp90=sp_15, S15_Sp90_lo=_ci(b_sp_15)[0], S15_Sp90_hi=_ci(b_sp_15)[1],
            V1r_PRAUC=ap_v1, V1r_PRAUC_lo=_ci(b_ap_v1)[0], V1r_PRAUC_hi=_ci(b_ap_v1)[1],
            V1r_Sp90=sp_v1, V1r_Sp90_lo=_ci(b_sp_v1)[0], V1r_Sp90_hi=_ci(b_sp_v1)[1],
        ))
        print(f"  {name}: N={len(y)}, ca={int(y.sum())}")

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "subgroup_3way.csv"), index=False)


def stats_inner(y, s):
    if y.sum() == 0:
        return np.nan, np.nan
    try:
        ap = average_precision_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        sp = _specificity_at_sensitivity(fpr, tpr, 0.90)
        return ap, sp
    except Exception:
        return np.nan, np.nan


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
