#!/usr/bin/env python3
"""Patient-clustered bootstrap for the subgroup-equity headline ranges
(max-min recall across powered levels of sex, age tertile, pack-year
quartile, and screening timepoint) at the LR>=4B-matched flag rate,
year~1.

Recall at a flag-rate-matched threshold is rank-invariant to monotonic
calibration, so raw model scores are used directly:
  - LungEvaty: comp_df['risk_score'] (raw transformer logit-prob)
  - Pillar-0:  predictions_pillar_test.csv prob_y1 (3-seed ensemble),
               attached by series_uid (with pid_tp fallback)
  - Sybil:     comp_df['sv1c_risk_1'] (Sybil v1.0, author-calibrated;
               author calibrator is non-monotonic but we use it as
               shipped to match the rest of the paper)

Output: one line per (model, subgroup_var) with range point estimate
and 95% percentile CI.
"""
import os
import sys

import numpy as np
import pandas as pd

OUTPUT_DIR = os.environ.get("SCREENING_METRICS_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_output"))
PILLAR_USER_TEST_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions_pillar_test.csv")

LR_ORDER = {"4X": 6, "4B": 5, "4A": 4, "3": 3, "2": 2, "1": 1}
LR4B_RANK = LR_ORDER["4B"]
N_BOOT = 1000
RNG_SEED = 42
MIN_CA = 5  # require at least 5 cancers in a level to score it
MODELS = ["LungEvaty", "Pillar-0", "Sybil"]


def _include(censor_time: float, gold: int, followup: int) -> tuple:
    valid_pos = bool(gold) and (censor_time <= followup) and (censor_time > -1)
    valid_neg = censor_time >= followup
    return (valid_pos or valid_neg), int(valid_pos)


def attach_pillar(comp):
    user = pd.read_csv(PILLAR_USER_TEST_CSV)
    user["series_uid"] = user["series_uid"].astype(str)
    user["patient_id"] = user["patient_id"].astype(str)
    user["timepoint"] = user["timepoint"].astype(int)
    user["pid_tp"] = user["patient_id"] + "_" + user["timepoint"].astype(str)
    u1 = user.drop_duplicates("series_uid")[["series_uid", "prob_y1"]]
    u1 = u1.rename(columns={"series_uid": "series", "prob_y1": "pillar_user_y1"})
    out = comp.merge(u1, on="series", how="left")
    miss = out["pillar_user_y1"].isna()
    if miss.any():
        u2 = user.drop_duplicates("pid_tp")[["pid_tp", "prob_y1"]].rename(
            columns={"prob_y1": "pillar_user_y1_fb"})
        fb = out.loc[miss, ["pid_tp"]].merge(u2, on="pid_tp", how="left")
        out.loc[miss, "pillar_user_y1"] = fb["pillar_user_y1_fb"].values
    return out


def attach_lungrads(comp):
    lr = pd.read_csv(os.path.join(OUTPUT_DIR, "nlst_lungrads_pinsky.csv"))
    lr["pid"] = lr["pid"].astype(str)
    lr["study_yr"] = lr["study_yr"].astype(int)
    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    comp["study_yr"] = comp["screen_timepoint"].astype(int)
    return comp.merge(lr[["pid", "study_yr", "lungrads_v10"]],
                      on=["pid", "study_yr"], how="left")


def make_levels(comp):
    out = {}
    g = comp["gender"].values
    out["Sex"] = [("Male", g == 1), ("Female", g == 2)]

    ages = comp["age"].values.astype(float)
    a33, a67 = np.percentile(ages, [33.33, 66.67])
    out["Age tertile"] = [
        ("low", ages <= a33),
        ("mid", (ages > a33) & (ages <= a67)),
        ("high", ages > a67),
    ]

    pk = comp["pkyr"].values.astype(float)
    q1, q2, q3 = np.percentile(pk, [25, 50, 75])
    out["Pack-year quartile"] = [
        ("Q1", pk <= q1),
        ("Q2", (pk > q1) & (pk <= q2)),
        ("Q3", (pk > q2) & (pk <= q3)),
        ("Q4", pk > q3),
    ]

    tp = comp["screen_timepoint"].astype(int).values
    out["Screening timepoint"] = [
        ("T0", tp == 0),
        ("T1", tp == 1),
        ("T2", tp == 2),
    ]
    return out


def recall_at_lr4b_match(scores, labels, lr_score, idx):
    s = scores[idx]
    y = labels[idx]
    lr = lr_score[idx]
    ca_total = int(y.sum())
    if ca_total < MIN_CA:
        return np.nan
    n_flag = int((lr >= LR4B_RANK).sum())
    if n_flag == 0:
        return np.nan
    keep = ~np.isnan(s)
    if not keep.any():
        return np.nan
    s_k = s[keep]
    y_k = y[keep]
    order = np.argsort(-s_k, kind="mergesort")[:n_flag]
    cap = int(y_k[order].sum())
    return cap / ca_total


def main():
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    comp["series"] = comp["series"].astype(str)
    comp["screen_timepoint"] = comp["screen_timepoint"].astype(int)
    comp["pid_tp"] = comp["pid"] + "_" + comp["screen_timepoint"].astype(str)
    comp = attach_pillar(comp)
    comp = attach_lungrads(comp)

    times = comp["time_at_event"].values
    golds = comp["gold"].values
    inc = np.zeros(len(comp), bool)
    lbl = np.zeros(len(comp), int)
    for i, (t, g) in enumerate(zip(times, golds)):
        ok, y = _include(t, g, 0)
        inc[i] = ok
        lbl[i] = y

    lr_score = np.array([LR_ORDER.get(str(c), 0)
                         for c in comp["lungrads_v10"].astype(str).values],
                        dtype=float)
    pids = comp["pid"].values

    scores = {
        "LungEvaty": comp["risk_score"].values.astype(float),
        "Pillar-0":  comp["pillar_user_y1"].values.astype(float),
        "Sybil":     comp["sv1c_risk_1"].values.astype(float),
    }

    levels_by_var = make_levels(comp)
    sub_y1_mask = inc.copy()

    # Point estimates
    print("[point] subgroup recall ranges at LR>=4B match, year-1")
    point_ranges = {}
    point_recalls = {}
    for var, levels in levels_by_var.items():
        for model in MODELS:
            recs = []
            for (lab, m) in levels:
                idx = np.where(sub_y1_mask & m)[0]
                r = recall_at_lr4b_match(scores[model], lbl, lr_score, idx)
                recs.append((lab, r))
                point_recalls[(var, lab, model)] = r
            valid = [r for _, r in recs if not np.isnan(r)]
            rng = (max(valid) - min(valid)) if len(valid) >= 2 else np.nan
            point_ranges[(var, model)] = rng
            print(f"  {var:22s} {model:9s} range={100*rng:5.1f}pp  "
                  f"recalls={ [f'{lab}={100*r:.1f}' if not np.isnan(r) else f'{lab}=NA' for lab, r in recs] }")

    # Bootstrap
    print(f"\n[bootstrap] n_boot={N_BOOT} patient-clustered")
    sub_idx = np.where(sub_y1_mask)[0]
    sub_pids = pids[sub_idx]
    pid_to_rows = {}
    for p, i in zip(sub_pids, sub_idx):
        pid_to_rows.setdefault(p, []).append(i)
    pid_to_rows = {k: np.asarray(v, dtype=np.int64)
                   for k, v in pid_to_rows.items()}
    unique_pids = np.array(list(pid_to_rows.keys()))

    boot_ranges = {(var, model): [] for var in levels_by_var for model in MODELS}
    rng = np.random.default_rng(RNG_SEED)
    for it in range(N_BOOT):
        sample = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx_b = np.concatenate([pid_to_rows[p] for p in sample])
        # Subset arrays
        for var, levels in levels_by_var.items():
            for model in MODELS:
                recs = []
                for (lab, m) in levels:
                    mb = m[idx_b]
                    sel = idx_b[mb]
                    r = recall_at_lr4b_match(scores[model], lbl, lr_score, sel)
                    if not np.isnan(r):
                        recs.append(r)
                if len(recs) >= 2:
                    boot_ranges[(var, model)].append(max(recs) - min(recs))
        if (it + 1) % 100 == 0:
            print(f"  iter {it + 1}/{N_BOOT}")

    print("\n[result] range CIs (pp)")
    print(f"{'Subgroup var':24s} {'Model':10s} {'pt':>6s}  {'95% CI':>16s}")
    for var in levels_by_var:
        for model in MODELS:
            pt = point_ranges[(var, model)]
            vals = boot_ranges[(var, model)]
            if len(vals) == 0:
                ci = (np.nan, np.nan)
            else:
                ci = (float(np.percentile(vals, 2.5)),
                      float(np.percentile(vals, 97.5)))
            print(f"{var:24s} {model:10s} {100*pt:5.1f}  "
                  f"[{100*ci[0]:5.1f}, {100*ci[1]:5.1f}]")


if __name__ == "__main__":
    main()
