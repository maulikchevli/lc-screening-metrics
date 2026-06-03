#!/usr/bin/env python3
"""One-cell patient-clustered bootstrap for LR-own recall at the
LR>=4B (workup) flag rate, restricted to T0 (prevalence round).

This fills the only CI cell left in tab:lr_anchored_caps. The main
subgroup pipeline (subgroup_lr_anchored.py) computes lr_subgroup_recall
on the point estimate but doesn't accumulate it through the bootstrap
loop; here we accumulate just that quantity for one (subgroup,level)
cell.
"""
import os
import sys

import numpy as np
import pandas as pd

OUTPUT_DIR = os.environ.get("SCREENING_METRICS_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_output"))
def _include(censor_time: float, gold: int, followup: int,
             fup_lower_bound: int = -1) -> tuple:
    valid_pos = bool(gold) and (censor_time <= followup) and (censor_time > fup_lower_bound)
    valid_neg = censor_time >= followup
    return (valid_pos or valid_neg), int(valid_pos)


LR_ORDER = {"4X": 6, "4B": 5, "4A": 4, "3": 3, "2": 2, "1": 1}
LR4B_RANK = LR_ORDER["4B"]
N_BOOT = 1000
RNG_SEED = 42


def main():
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)

    lr = pd.read_csv(os.path.join(OUTPUT_DIR, "nlst_lungrads_pinsky.csv"))
    lr["pid"] = lr["pid"].astype(str)
    lr["study_yr"] = lr["study_yr"].astype(int)
    comp["study_yr"] = comp["screen_timepoint"].astype(int)
    comp = comp.merge(lr[["pid", "study_yr", "lungrads_v10"]],
                      on=["pid", "study_yr"], how="left")

    lr_score = np.array([LR_ORDER.get(str(c), 0)
                         for c in comp["lungrads_v10"].astype(str).values],
                        dtype=float)
    pids = comp["pid"].values
    times = comp["time_at_event"].values
    golds = comp["gold"].values
    tps = comp["screen_timepoint"].astype(int).values

    inc = np.zeros(len(comp), bool)
    lbl = np.zeros(len(comp), int)
    for i, (t, g) in enumerate(zip(times, golds)):
        ok, y = _include(t, g, 0)
        inc[i] = ok
        lbl[i] = y

    t0_mask = inc & (tps == 0)
    idx_t0 = np.where(t0_mask)[0]
    pids_t0 = pids[idx_t0]
    lr_t0 = lr_score[idx_t0]
    lbl_t0 = lbl[idx_t0]

    flag_mask = lr_t0 >= LR4B_RANK
    n_total_ca = int(lbl_t0.sum())
    n_caught_ca = int(lbl_t0[flag_mask].sum())
    n_flag = int(flag_mask.sum())
    point_recall = n_caught_ca / n_total_ca if n_total_ca else float("nan")
    point_prec = n_caught_ca / n_flag if n_flag else float("nan")

    print(f"[T0] n={len(idx_t0)}, n_cancers={n_total_ca}, "
          f"n_flagged(LR>=4B)={n_flag}, n_caught={n_caught_ca}")
    print(f"[T0] LR-own recall point estimate = {point_recall:.4f}")
    print(f"[T0] LR-own precision point estimate = {point_prec:.4f}")

    pid_to_rows = {}
    for p, i in zip(pids_t0, idx_t0):
        pid_to_rows.setdefault(p, []).append(i)
    pid_to_rows = {k: np.asarray(v, dtype=np.int64)
                   for k, v in pid_to_rows.items()}
    unique_pids = np.array(list(pid_to_rows.keys()))
    print(f"[T0] unique pids = {len(unique_pids)}")

    rng = np.random.default_rng(RNG_SEED)
    recalls, precs = [], []
    for _ in range(N_BOOT):
        sample_pids = rng.choice(unique_pids, size=len(unique_pids),
                                 replace=True)
        idx = np.concatenate([pid_to_rows[p] for p in sample_pids])
        lr_s = lr_score[idx]
        lbl_s = lbl[idx]
        ca_total = int(lbl_s.sum())
        if ca_total == 0:
            recalls.append(np.nan)
            precs.append(np.nan)
            continue
        flag_s = lr_s >= LR4B_RANK
        ca_caught = int(lbl_s[flag_s].sum())
        n_flag_s = int(flag_s.sum())
        recalls.append(ca_caught / ca_total)
        precs.append(ca_caught / n_flag_s if n_flag_s else np.nan)

    rec_lo = float(np.nanpercentile(recalls, 2.5))
    rec_hi = float(np.nanpercentile(recalls, 97.5))
    prec_lo = float(np.nanpercentile(precs, 2.5))
    prec_hi = float(np.nanpercentile(precs, 97.5))
    print(f"[T0] Recall 95% CI = [{rec_lo:.4f}, {rec_hi:.4f}]")
    print(f"[T0] Precision 95% CI = [{prec_lo:.4f}, {prec_hi:.4f}]")
    print(f"[T0] As percentages: recall = {100*point_recall:.1f}% "
          f"[{100*rec_lo:.1f}, {100*rec_hi:.1f}]")


if __name__ == "__main__":
    main()
