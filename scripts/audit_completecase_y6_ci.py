#!/usr/bin/env python3
"""Complete-case year-6 NB with patient-clustered bootstrap 95% CIs for all
strategies, including Lungevity (Platt) — fills the hole in the calibrated
DCA table at year 6.

Outputs:
  eval_output/audit_completecase_year6_ci.csv
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import _include, OUTPUT_DIR
from uspstf_plcom import plcom2012_risk, uspstf_2021_eligible, NLST_PERSONAL_CANCER_COLS

EPS = 1e-7
RNG_SEED = 42
N_BOOT = 500
THRESHOLDS = (0.01, 0.05, 0.10)
DEV_JSON = os.environ.get("LUNGEVATY_DEV_JSON", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions_lungevaty_dev.json"))


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _ci(arr):
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def fit_lung_platt_y6():
    """Fit Platt for year-6 horizon on Lungevity dev predictions."""
    with open(DEV_JSON) as f:
        dev = json.load(f)
    dev_s, dev_y = [], []
    fu = 5
    for r in dev:
        inc, lbl = _include(r["time_at_event"], r["y"], fu)
        if inc:
            dev_s.append(float(r["cancer_risk"][0]))
            dev_y.append(lbl)
    dev_s = np.array(dev_s); dev_y = np.array(dev_y)
    clf = LogisticRegression(C=1e6, solver="lbfgs")
    clf.fit(_logit(dev_s).reshape(-1, 1), dev_y)
    return clf


def main():
    print("Loading 4-way matched cohort + recomputing PLCO/USPSTF …")
    from evaluate_survival import (
        load_predictions, load_participant, load_lung_cancer, merge_data,
        PARTICIPANT_CSV,
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
    comp["uspstf"] = comp.apply(uspstf_2021_eligible, axis=1)
    comp["plcom_y6"] = comp.apply(lambda r: plcom2012_risk(r, year=6), axis=1)

    # Year-6 complete-case filter
    fu = 5
    mask = comp.apply(
        lambda r: _include(r["time_at_event"], r["gold"], fu)[0], axis=1).values
    sub = comp[mask].copy().reset_index(drop=True)
    labels = sub.apply(
        lambda r: _include(r["time_at_event"], r["gold"], fu)[1], axis=1).values
    n = len(sub)
    pids = sub["pid"].values
    print(f"  Year-6 complete-case: N={n}, cancers={int(labels.sum())}, "
          f"observed prevalence={labels.mean():.4f}")

    # Fit Lungevity Platt for year 6 + apply to test
    print("  Fitting Lungevity Platt (year 6) on dev …")
    clf = fit_lung_platt_y6()
    lung_raw = sub["risk_score"].values.astype(float)
    lung_platt = clf.predict_proba(_logit(lung_raw).reshape(-1, 1))[:, 1]
    sub["lung_platt"] = lung_platt

    pid_to_rows = {p: np.where(pids == p)[0] for p in set(pids)}
    unique_pids = np.array(sorted(set(pids)))

    def nb_complete_case(scores_in, p_t, mask_valid=None):
        s = scores_in
        y = labels
        if mask_valid is not None:
            s = s[mask_valid]; y = y[mask_valid]
        n_v = len(s)
        if n_v == 0:
            return np.nan
        pos = s >= p_t
        tp = (pos & (y == 1)).sum() / n_v
        fp = (pos & (y == 0)).sum() / n_v
        w = p_t / (1 - p_t)
        return tp - fp * w

    def nb_uspstf_cc(flags, p_t):
        n_v = len(flags)
        pos = flags == 1
        tp = (pos & (labels == 1)).sum() / n_v
        fp = (pos & (labels == 0)).sum() / n_v
        w = p_t / (1 - p_t)
        return tp - fp * w

    strategies = [
        ("Lungevity_raw", sub["risk_score"].values.astype(float)),
        ("Lungevity_Platt", sub["lung_platt"].values.astype(float)),
        ("Sybil15_raw", sub["sybil_risk_6"].values.astype(float)),
        ("Sybilv1_raw", sub["sv1r_risk_6"].values.astype(float)),
        ("Sybilv1_cal", sub["sv1c_risk_6"].values.astype(float)),
        ("PLCOm2012", sub["plcom_y6"].values.astype(float)),
    ]

    # Compute point estimates + bootstrap CIs
    rows = []
    for p_t in THRESHOLDS:
        row = dict(p_t=p_t, N=n, prev=float(labels.mean()))
        # USPSTF first
        u = sub["uspstf"].values.astype(int)
        nb_u = nb_uspstf_cc(u, p_t)
        # bootstrap USPSTF
        rng = np.random.default_rng(RNG_SEED + int(p_t * 100) + 9999)
        boot = []
        for _ in range(N_BOOT):
            sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
            idx = np.concatenate([pid_to_rows[p] for p in sampled])
            yi = labels[idx]; ui = u[idx]
            n_v = len(yi)
            if n_v == 0 or yi.sum() == 0:
                continue
            pos = ui == 1
            tp = (pos & (yi == 1)).sum() / n_v
            fp = (pos & (yi == 0)).sum() / n_v
            w = p_t / (1 - p_t)
            boot.append(tp - fp * w)
        lo, hi = _ci(boot)
        row["USPSTF_NB"] = nb_u
        row["USPSTF_lo"] = lo
        row["USPSTF_hi"] = hi

        # Each continuous-score strategy
        for name, scores in strategies:
            valid = ~np.isnan(scores)
            scores_v = scores[valid]
            labels_v = labels[valid]
            pids_v = pids[valid]
            n_v = len(scores_v)
            pos = scores_v >= p_t
            tp = (pos & (labels_v == 1)).sum() / n_v
            fp = (pos & (labels_v == 0)).sum() / n_v
            w = p_t / (1 - p_t)
            nb_pt = tp - fp * w

            pid_to_rows_v = {p: np.where(pids_v == p)[0] for p in set(pids_v)}
            unique_pids_v = np.array(sorted(set(pids_v)))
            rng = np.random.default_rng(
                RNG_SEED + int(p_t * 100) + sum(ord(c) for c in name))
            boot = []
            for _ in range(N_BOOT):
                sampled = rng.choice(unique_pids_v,
                                     size=len(unique_pids_v), replace=True)
                idx = np.concatenate([pid_to_rows_v[p] for p in sampled])
                si = scores_v[idx]; yi = labels_v[idx]
                if yi.sum() == 0:
                    continue
                pos_b = si >= p_t
                tp_b = (pos_b & (yi == 1)).sum() / len(si)
                fp_b = (pos_b & (yi == 0)).sum() / len(si)
                boot.append(tp_b - fp_b * w)
            lo, hi = _ci(boot)
            row[f"{name}_NB"] = nb_pt
            row[f"{name}_lo"] = lo
            row[f"{name}_hi"] = hi
        rows.append(row)
        print(f"  p_t={p_t}: done")

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "audit_completecase_year6_ci.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    # Also print a formatted table for the report
    print()
    print("=" * 90)
    print("COMPLETE-CASE YEAR-6 NB WITH 95% CIs")
    print("=" * 90)
    print(f"  N = {n}, cancers = {int(labels.sum())}, observed prevalence = {labels.mean():.4f}")
    print()
    fmt = lambda nb, lo, hi: f"{nb:+.4f} [{lo:+.4f},{hi:+.4f}]"
    cols = [("Lungevity (raw)", "Lungevity_raw"),
            ("Lungevity (Platt)", "Lungevity_Platt"),
            ("Sybil 1.5", "Sybil15_raw"),
            ("Sybil v1.0 raw", "Sybilv1_raw"),
            ("Sybil v1.0 cal", "Sybilv1_cal"),
            ("PLCOm2012", "PLCOm2012"),
            ("USPSTF", "USPSTF")]
    for label, key in cols:
        print(f"  {label:<22}", end="")
        for p_t in THRESHOLDS:
            r = df[df.p_t == p_t].iloc[0]
            nb = r[f"{key}_NB"]; lo = r[f"{key}_lo"]; hi = r[f"{key}_hi"]
            print(f"  pt={p_t:.2f}: {fmt(nb, lo, hi):<28}", end="")
        print()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
