#!/usr/bin/env python3
"""Sensitivity check: PLCOm2012 DCA at year-1 and year-3 WITHOUT horizon
rescaling — apply the raw 6-year PLCOm risk directly against year-1 and
year-3 thresholds.

Compares against the rescaled values in audit_dca_raw.csv (which uses
KM-cumulative-incidence ratio scaling: 0.2796 for year 1, 0.6431 for year 3).

Output: eval_output/audit_plco_no_rescale.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import _ipcw_weights, OUTPUT_DIR
from uspstf_plcom import plcom2012_risk, NLST_PERSONAL_CANCER_COLS

RNG_SEED = 42
N_BOOT = 500
THRESHOLDS = (0.01, 0.05, 0.10)


def _ci(arr):
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def main():
    print("Loading data + computing unrescaled 6-yr PLCOm …")
    from evaluate_survival import (
        load_participant, PARTICIPANT_CSV,
    )
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    part = load_participant(PARTICIPANT_CSV)
    part["pid"] = part["pid"].astype(str)
    extra_cols = ["height", "weight", "educat", "age_quit", "ethnic",
                  "diagcopd", "diagemph", "diagchro",
                  "famfather", "fammother", "fambrother", "famsister", "famchild"]
    extra_cols += list(NLST_PERSONAL_CANCER_COLS)
    extra_cols = [c for c in extra_cols if c in part.columns]
    extra = part[["pid"] + extra_cols].drop_duplicates("pid")
    comp = comp.merge(extra, on="pid", how="left")

    # Compute the UNRESCALED 6-year PLCOm risk (year=6 in our function)
    comp["plcom_no_rescale"] = comp.apply(
        lambda r: plcom2012_risk(r, year=6), axis=1)
    print(f"  N = {len(comp)}, "
          f"PLCOm_6yr range: {comp['plcom_no_rescale'].min():.4f} - "
          f"{comp['plcom_no_rescale'].max():.4f}, "
          f"median = {comp['plcom_no_rescale'].median():.4f}")

    times = comp["time_at_event"].values.astype(float)
    events = comp["gold"].values.astype(int)
    pids = comp["pid"].values
    plco_unrescaled = comp["plcom_no_rescale"].values.astype(float)
    n = len(comp)
    pid_to_rows = {p: np.where(pids == p)[0] for p in set(pids)}
    unique_pids = np.array(sorted(set(pids)))

    rows = []
    for year in (1, 3):
        fu = year - 1
        for p_t in THRESHOLDS:
            tp_w, fp_w, is_tp, is_fp, prev = _ipcw_weights(times, events, fu)
            pos = plco_unrescaled >= p_t
            tp_r = float(tp_w[pos & is_tp].sum()) / n
            fp_r = float((pos & is_fp).sum()) * fp_w / n
            w = p_t / (1 - p_t)
            nb = tp_r - fp_r * w

            rng = np.random.default_rng(
                RNG_SEED + year * 100 + int(p_t * 100) + 4444)
            boot = []
            for _ in range(N_BOOT):
                sampled = rng.choice(unique_pids,
                                     size=len(unique_pids), replace=True)
                idx = np.concatenate([pid_to_rows[p] for p in sampled])
                ti, ei = times[idx], events[idx]
                if ei.sum() == 0:
                    continue
                try:
                    tp_w_b, fp_w_b, is_tp_b, is_fp_b, _ = \
                        _ipcw_weights(ti, ei, fu)
                    pos_b = plco_unrescaled[idx] >= p_t
                    tp_rb = float(tp_w_b[pos_b & is_tp_b].sum()) / n
                    fp_rb = float((pos_b & is_fp_b).sum()) * fp_w_b / n
                    boot.append(tp_rb - fp_rb * w)
                except Exception:
                    pass
            lo, hi = _ci(boot)
            rows.append(dict(Year=year, p_t=p_t,
                             PLCO_no_rescale_NB=nb,
                             lo=lo, hi=hi))
            print(f"  year {year}, p_t={p_t}:  NB={nb:+.4f}  "
                  f"[{lo:+.4f}, {hi:+.4f}]")

    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, "audit_plco_no_rescale.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")

    # ---- Comparison against rescaled values ----
    rescaled = pd.read_csv(os.path.join(OUTPUT_DIR, "audit_dca_raw.csv"))
    rescaled = rescaled[rescaled.Year.isin([1, 3])].copy()

    print("\n" + "=" * 80)
    print("PLCOm2012 RESCALED vs NO-RESCALING comparison")
    print("=" * 80)
    print(f"{'Year':<5}{'p_t':<6} {'rescaled':>26} {'no rescale':>26}  delta")
    for _, r in df.iterrows():
        rsc = rescaled[(rescaled.Year == r.Year) &
                       (rescaled.p_t == r.p_t)].iloc[0]
        d = r.PLCO_no_rescale_NB - rsc.PLCO_NB
        print(f"  {int(r.Year):<3}  {r.p_t:<5} "
              f"{rsc.PLCO_NB:+.4f} [{rsc.PLCO_lo:+.4f},{rsc.PLCO_hi:+.4f}]  "
              f"  {r.PLCO_no_rescale_NB:+.4f} [{r.lo:+.4f},{r.hi:+.4f}]"
              f"   {d:+.4f}")

    # ---- Qualitative finding checks ----
    print("\n" + "=" * 80)
    print("QUALITATIVE FINDING CHECKS")
    print("=" * 80)

    print("\nFinding 1: PLCOm year-1 strict thresholds (pt=5%, 10%) NB sign")
    for p_t in (0.05, 0.10):
        rsc = rescaled[(rescaled.Year == 1) & (rescaled.p_t == p_t)].iloc[0]
        nor = df[(df.Year == 1) & (df.p_t == p_t)].iloc[0]
        print(f"  pt={p_t:.2f}: rescaled NB={rsc.PLCO_NB:+.4f} "
              f"({'≈0' if abs(rsc.PLCO_NB) < 0.001 else 'pos' if rsc.PLCO_NB > 0 else 'neg'})  "
              f"|  no-rescale NB={nor.PLCO_no_rescale_NB:+.4f} "
              f"({'≈0' if abs(nor.PLCO_no_rescale_NB) < 0.001 else 'pos' if nor.PLCO_no_rescale_NB > 0 else 'neg'})")

    print("\nFinding 2: imaging-vs-PLCOm ordering at year-1 strict thresholds")
    for p_t in (0.05, 0.10):
        rsc = rescaled[(rescaled.Year == 1) & (rescaled.p_t == p_t)].iloc[0]
        nor = df[(df.Year == 1) & (df.p_t == p_t)].iloc[0]
        print(f"  pt={p_t:.2f}:")
        print(f"    Lung   = {rsc.Lung_NB:+.4f}")
        print(f"    S15    = {rsc.S15_NB:+.4f}")
        print(f"    V1r    = {rsc.V1r_NB:+.4f}")
        print(f"    PLCO (rescaled)  = {rsc.PLCO_NB:+.4f}")
        print(f"    PLCO (no-rescale) = {nor.PLCO_no_rescale_NB:+.4f}")
        # Check if PLCO ordering relative to imaging changes
        imaging_max = max(rsc.Lung_NB, rsc.S15_NB, rsc.V1r_NB)
        flip = (rsc.PLCO_NB > imaging_max) != (nor.PLCO_no_rescale_NB > imaging_max)
        print(f"    PLCO > best imaging? rescaled={rsc.PLCO_NB > imaging_max},"
              f" no-rescale={nor.PLCO_no_rescale_NB > imaging_max}, flip={flip}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
