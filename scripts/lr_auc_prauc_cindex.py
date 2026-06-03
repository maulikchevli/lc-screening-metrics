#!/usr/bin/env python3
"""Lung-RADS AUC, PR-AUC, and Uno IPCW C-index with patient-clustered
bootstrap CIs — matching the Table 1 machinery for the imaging models.

Score construction
------------------
Per-scan derived Lung-RADS v1.0 category (Pinsky 2015 reproduction) is
mapped to an ordinal rank:
  LR1=1, LR2=2, LR3=3, LR4A=4, LR4B=5, LR4X=6
Scans with no LR assignment (lungrads_v10 NaN / unrecognized) are
dropped — they cannot be ranked. The dropped count is reported.

Filter (per horizon)
--------------------
Same `_include` rule used by the imaging-model AUC pipeline
(complete-case at each year). The C-index uses the full time-to-event
without horizon binning, IPCW (Uno) truncated at year 6.

Bootstrap
---------
Patient-clustered, n=1000, with seeds matching the imaging-model
scripts so paired diffs are computable later if wanted:
  AUC / PR-AUC : seed 20260519 (paired_diffs_auc_prauc.py)
  C-index      : seed 20260518 (cindex_with_ci.py)

Outputs
-------
eval_output/lr_auc_prauc_with_ci.csv      (one row per horizon)
eval_output/lr_cindex_with_ci.csv         (single row)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_survival import _include, OUTPUT_DIR
from vital.metrics import concordance_index, get_censoring_dist

COMP_DF = os.path.join(OUTPUT_DIR, "four_way_comp_df.csv")
LR_CSV = os.path.join(OUTPUT_DIR, "nlst_lungrads_pinsky.csv")
N_BOOT = 1000
RNG_SEED_AUC = 20260519
RNG_SEED_CIDX = 20260518

LR_ORDER = {"1": 1, "2": 2, "3": 3, "4A": 4, "4B": 5, "4X": 6}


def attach_lr(comp):
    lr = pd.read_csv(LR_CSV)
    lr["pid"] = lr["pid"].astype(str)
    lr["study_yr"] = lr["study_yr"].astype(int)
    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    comp["study_yr"] = comp["screen_timepoint"].astype(int)
    return comp.merge(lr[["pid", "study_yr", "lungrads_v10"]],
                      on=["pid", "study_yr"], how="left")


def to_lr_rank(values):
    """str/NaN array -> float array; unmapped -> NaN (dropped downstream)."""
    out = np.array([LR_ORDER.get(str(c), np.nan) for c in values], dtype=float)
    return out


def compute_auc_prauc():
    print(f"Loading {COMP_DF} …")
    comp = pd.read_csv(COMP_DF)
    n_raw, p_raw = len(comp), comp["pid"].nunique()
    print(f"  raw: {n_raw} scans, {p_raw} pids, {int(comp['gold'].sum())} cancers")

    print("Attaching Lung-RADS per-scan categories …")
    comp = attach_lr(comp)
    lr_rank = to_lr_rank(comp["lungrads_v10"].values)
    n_missing = int(np.isnan(lr_rank).sum())
    print(f"  LR-mapped:  {len(lr_rank) - n_missing}/{len(lr_rank)} scans  "
          f"(missing/unrecognized dropped: {n_missing})")

    valid = ~np.isnan(lr_rank)
    comp = comp.loc[valid].reset_index(drop=True)
    lr_rank = lr_rank[valid]
    print(f"  LR-valid cohort: {len(comp)} scans, {comp['pid'].nunique()} pids")

    rng = np.random.default_rng(RNG_SEED_AUC)
    rows = []

    for year in range(1, 7):
        fu = year - 1
        keep_mask = np.array([_include(t, e, fu)[0]
                              for t, e in zip(comp["time_at_event"], comp["gold"])])
        labels_all = np.array([_include(t, e, fu)[1]
                               for t, e in zip(comp["time_at_event"], comp["gold"])])
        sub_idx = np.where(keep_mask)[0]
        sub_scores = lr_rank[sub_idx]
        sub_labels = labels_all[sub_idx]
        sub_pids = comp["pid"].values[sub_idx]

        n = len(sub_idx)
        n_ca = int(sub_labels.sum())
        print(f"\nYear {year}: N={n}, cancers={n_ca}")

        point_auc = float(roc_auc_score(sub_labels, sub_scores))
        point_prauc = float(average_precision_score(sub_labels, sub_scores))

        unique_pids = np.unique(sub_pids)
        pid_to_rows = {p: np.where(sub_pids == p)[0] for p in unique_pids}

        boot_auc, boot_prauc = [], []
        for b in range(N_BOOT):
            sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
            idx = np.concatenate([pid_to_rows[p] for p in sampled])
            yb = sub_labels[idx]
            if yb.sum() == 0 or len(np.unique(yb)) < 2:
                continue
            sb = sub_scores[idx]
            boot_auc.append(float(roc_auc_score(yb, sb)))
            boot_prauc.append(float(average_precision_score(yb, sb)))
            if (b + 1) % 250 == 0:
                print(f"    bootstrap {b+1}/{N_BOOT}")

        boot_auc = np.array(boot_auc)
        boot_prauc = np.array(boot_prauc)
        auc_lo, auc_hi = np.percentile(boot_auc, [2.5, 97.5])
        pr_lo, pr_hi = np.percentile(boot_prauc, [2.5, 97.5])

        rows.append(dict(
            year=year, N=n, N_cancer=n_ca,
            auc=point_auc, auc_lo=float(auc_lo), auc_hi=float(auc_hi),
            prauc=point_prauc, prauc_lo=float(pr_lo), prauc_hi=float(pr_hi),
            n_boot=int(len(boot_auc)),
        ))

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "lr_auc_prauc_with_ci.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return df, comp, lr_rank


def compute_cindex(comp, lr_rank):
    """Uno IPCW C-index on the LR-valid cohort, truncated at year 6."""
    print("\n" + "=" * 64)
    print("UNO IPCW C-INDEX (Lung-RADS, ordinal score)")
    print("=" * 64)

    times = comp["time_at_event"].values.astype(float)
    events = comp["gold"].values.astype(int)
    # C-index expects a per-horizon score matrix; LR is one score across all
    # horizons, so we tile across 6 columns to match the imaging machinery.
    scores = np.tile(lr_rank.reshape(-1, 1), (1, 6))

    pids = comp["pid"].values
    unique_pids = np.unique(pids)
    pid_to_rows = {p: np.where(pids == p)[0] for p in unique_pids}

    def cindex_on_rows(idx):
        sub_t = times[idx]
        sub_e = events[idx]
        sub_s = scores[idx]
        cens = get_censoring_dist([
            {"time_at_event": int(t), "y": int(e)}
            for t, e in zip(sub_t, sub_e)
        ])
        return float(concordance_index(sub_t, sub_s, sub_e, cens))

    point = cindex_on_rows(np.arange(len(times)))
    print(f"  point C-index: {point:.4f}")

    rng = np.random.default_rng(RNG_SEED_CIDX)
    boot = np.full(N_BOOT, np.nan)
    for b in range(N_BOOT):
        sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = np.concatenate([pid_to_rows[p] for p in sampled])
        try:
            boot[b] = cindex_on_rows(idx)
        except Exception:
            pass
        if (b + 1) % 100 == 0:
            print(f"    bootstrap {b+1}/{N_BOOT}")

    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  95% CI: [{lo:.4f}, {hi:.4f}]  (n_boot={len(boot)})")

    df = pd.DataFrame([dict(
        model="Lung-RADS", c_index=point,
        ci_lo=float(lo), ci_hi=float(hi),
        ci_width=float(hi - lo), n_boot=int(len(boot)),
    )])
    out_path = os.path.join(OUTPUT_DIR, "lr_cindex_with_ci.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return df


def print_table(auc_df, cidx_df):
    print("\n" + "=" * 72)
    print("LUNG-RADS Layer-1 metrics, deployment form")
    print("=" * 72)
    print(f"{'Year':>4}  {'N':>6}  {'N_ca':>5}  "
          f"{'AUC':>20}  {'PR-AUC':>20}")
    print("-" * 72)
    for _, r in auc_df.iterrows():
        auc_s = f"{r.auc:.3f} [{r.auc_lo:.3f}, {r.auc_hi:.3f}]"
        pr_s = f"{r.prauc:.3f} [{r.prauc_lo:.3f}, {r.prauc_hi:.3f}]"
        print(f"{int(r.year):>4}  {int(r.N):>6}  {int(r.N_cancer):>5}  "
              f"{auc_s:>20}  {pr_s:>20}")
    r = cidx_df.iloc[0]
    print("-" * 72)
    print(f"C-index (Uno IPCW, trunc y6):  "
          f"{r.c_index:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]")


def main():
    auc_df, comp_valid, lr_rank = compute_auc_prauc()
    cidx_df = compute_cindex(comp_valid, lr_rank)
    print_table(auc_df, cidx_df)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
