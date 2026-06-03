#!/usr/bin/env python3
"""Paired-difference bootstrap CIs for imaging-model vs Lung-RADS on AUC
(horizons y1-y6) and Uno IPCW C-index (single number), companion to
Table 1 with the LR column.

Mirrors the supplementary Table S3 machinery (paired_diffs_auc_prauc.py)
and the C-index machinery (cindex_with_ci.py): patient-clustered
bootstrap, n=1000, same seeds, so the LR comparisons sit on the same
resamples as the existing imaging-vs-imaging tables.

Pairs: (LungEvaty - LR), (Pillar-0 - LR), (Sybil - LR).

PR-AUC is intentionally not reported for LR (ordinal 6-bin score yields
a step-function curve that is not directly comparable to the imaging
models' continuous-score PR-AUC; see paper discussion).

Outputs:
  eval_output/paired_diffs_vs_lr_auc.csv     (one row per (year, model))
  eval_output/paired_diffs_vs_lr_cindex.csv  (one row per model)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_survival import _include, OUTPUT_DIR
from layer1_layer3_full import attach_all_deployment_forms, MODELS, RNG_SEED, N_BOOT
from vital.metrics import concordance_index, get_censoring_dist

LR_CSV = os.path.join(OUTPUT_DIR, "nlst_lungrads_pinsky.csv")
LR_ORDER = {"1": 1, "2": 2, "3": 3, "4A": 4, "4B": 5, "4X": 6}

# Match cindex_with_ci.py seed so paired C-index diff lines up with the
# existing imaging C-index numbers.
CIDX_SEED = 20260518

IMAGING_MODELS = [m for m in MODELS if m[0] != "PLCOm2012"]


def attach_lr(comp):
    lr = pd.read_csv(LR_CSV)
    lr["pid"] = lr["pid"].astype(str)
    lr["study_yr"] = lr["study_yr"].astype(int)
    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    comp["study_yr"] = comp["screen_timepoint"].astype(int)
    return comp.merge(lr[["pid", "study_yr", "lungrads_v10"]],
                      on=["pid", "study_yr"], how="left")


def auc_paired_diffs(comp):
    rng = np.random.default_rng(RNG_SEED)
    lr_rank_all = np.array(
        [LR_ORDER.get(str(c), np.nan) for c in comp["lungrads_v10"].values],
        dtype=float,
    )

    rows = []
    for year in range(1, 7):
        fu = year - 1
        keep = np.array([_include(t, e, fu)[0]
                         for t, e in zip(comp["time_at_event"], comp["gold"])])
        labels = np.array([_include(t, e, fu)[1]
                           for t, e in zip(comp["time_at_event"], comp["gold"])])
        sub = comp.loc[keep].reset_index(drop=True)
        sub_labels = labels[keep]
        sub_lr = lr_rank_all[keep]

        score_arrays = {"LR": sub_lr}
        for name, tmpl in IMAGING_MODELS:
            score_arrays[name] = sub[tmpl.format(y=year)].values.astype(float)

        # Joint-valid mask (drop NaN in any score, incl. LR)
        joint = np.ones(len(sub), dtype=bool)
        for v in score_arrays.values():
            joint &= ~np.isnan(v)
        sub2 = sub.loc[joint].reset_index(drop=True)
        labels2 = sub_labels[joint]
        for k in score_arrays:
            score_arrays[k] = score_arrays[k][joint]

        n = len(sub2)
        n_ca = int(labels2.sum())
        print(f"\nYear {year}: N={n}, cancers={n_ca}")

        pids = sub2["pid"].values
        unique_pids = np.array(sorted(set(pids.tolist())))
        pid_to_rows = {p: np.where(pids == p)[0] for p in unique_pids}

        boot = {k: [] for k in score_arrays}
        for b in range(N_BOOT):
            sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
            idx = np.concatenate([pid_to_rows[p] for p in sampled])
            yb = labels2[idx]
            if yb.sum() == 0 or len(np.unique(yb)) < 2:
                continue
            for k in score_arrays:
                boot[k].append(float(roc_auc_score(yb, score_arrays[k][idx])))
            if (b + 1) % 250 == 0:
                print(f"    bootstrap {b+1}/{N_BOOT}")

        boot = {k: np.array(v) for k, v in boot.items()}
        point = {k: float(roc_auc_score(labels2, score_arrays[k]))
                 for k in score_arrays}

        for name, _ in IMAGING_MODELS:
            diff = boot[name] - boot["LR"]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            pt = point[name] - point["LR"]
            rows.append(dict(
                year=year, N=n, N_cancer=n_ca,
                model=name,
                model_auc=point[name],
                lr_auc=point["LR"],
                diff=pt,
                diff_lo=float(lo),
                diff_hi=float(hi),
                crosses_zero=bool(lo < 0 < hi),
                n_boot=int(len(diff)),
            ))

    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, "paired_diffs_vs_lr_auc.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


def cindex_paired_diffs(comp):
    print("\n" + "=" * 64)
    print("PAIRED C-INDEX DIFFERENCES (imaging - LR), Uno IPCW")
    print("=" * 64)
    lr_rank_all = np.array(
        [LR_ORDER.get(str(c), np.nan) for c in comp["lungrads_v10"].values],
        dtype=float,
    )

    # Joint-valid: LR must be present AND all imaging models valid for y1..y6.
    valid = ~np.isnan(lr_rank_all)
    for name, tmpl in IMAGING_MODELS:
        for y in range(1, 7):
            valid &= ~np.isnan(comp[tmpl.format(y=y)].values.astype(float))
    sub = comp.loc[valid].reset_index(drop=True)
    lr_rank = lr_rank_all[valid]
    print(f"  Joint-valid cohort: {len(sub)} scans, "
          f"{sub['pid'].nunique()} pids, {int(sub['gold'].sum())} cancers")

    times = sub["time_at_event"].values.astype(float)
    events = sub["gold"].values.astype(int)
    scores = {
        "LR": np.tile(lr_rank.reshape(-1, 1), (1, 6)),
    }
    for name, tmpl in IMAGING_MODELS:
        scores[name] = np.column_stack(
            [sub[tmpl.format(y=y)].values for y in range(1, 7)]
        )

    def cindex_on_rows(s2d, idx):
        sub_t = times[idx]
        sub_e = events[idx]
        sub_s = s2d[idx]
        cens = get_censoring_dist([
            {"time_at_event": int(t), "y": int(e)}
            for t, e in zip(sub_t, sub_e)
        ])
        return float(concordance_index(sub_t, sub_s, sub_e, cens))

    full_idx = np.arange(len(times))
    point = {k: cindex_on_rows(s, full_idx) for k, s in scores.items()}

    pids = sub["pid"].values
    unique_pids = np.unique(pids)
    pid_to_rows = {p: np.where(pids == p)[0] for p in unique_pids}

    rng = np.random.default_rng(CIDX_SEED)
    boot = {k: np.full(N_BOOT, np.nan) for k in scores}
    for b in range(N_BOOT):
        sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = np.concatenate([pid_to_rows[p] for p in sampled])
        for k, s in scores.items():
            try:
                boot[k][b] = cindex_on_rows(s, idx)
            except Exception:
                pass
        if (b + 1) % 100 == 0:
            print(f"    bootstrap {b+1}/{N_BOOT}")

    rows = []
    for name, _ in IMAGING_MODELS:
        diff = boot[name] - boot["LR"]
        diff = diff[~np.isnan(diff)]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        pt = point[name] - point["LR"]
        rows.append(dict(
            model=name,
            model_cindex=point[name],
            lr_cindex=point["LR"],
            diff=pt,
            diff_lo=float(lo),
            diff_hi=float(hi),
            crosses_zero=bool(lo < 0 < hi),
            n_boot=int(len(diff)),
        ))
    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, "paired_diffs_vs_lr_cindex.csv")
    df.to_csv(out, index=False)
    print(f"  Saved: {out}")
    return df


def print_report(auc_df, cidx_df):
    print("\n" + "=" * 88)
    print("PAIRED AUC DIFFS (imaging - LR)  patient-clustered bootstrap n=1000, seed=20260519")
    print("=" * 88)
    print(f"{'Year':>4}  {'Model':<10}  {'imaging':>8}  {'LR':>8}  "
          f"{'diff':>8}  {'95% CI':<22}  sig")
    print("-" * 88)
    for _, r in auc_df.iterrows():
        sig = "*" if not r.crosses_zero else " "
        ci = f"[{r.diff_lo:+.4f}, {r.diff_hi:+.4f}]"
        print(f"{int(r.year):>4}  {r.model:<10}  "
              f"{r.model_auc:>8.3f}  {r.lr_auc:>8.3f}  "
              f"{r['diff']:>+8.4f}  {ci:<22}  {sig}")

    print("\n" + "=" * 88)
    print("PAIRED C-INDEX DIFFS (imaging - LR)  patient-clustered bootstrap n=1000, seed=20260518")
    print("=" * 88)
    print(f"{'Model':<10}  {'imaging':>8}  {'LR':>8}  {'diff':>8}  "
          f"{'95% CI':<22}  sig")
    print("-" * 88)
    for _, r in cidx_df.iterrows():
        sig = "*" if not r.crosses_zero else " "
        ci = f"[{r.diff_lo:+.4f}, {r.diff_hi:+.4f}]"
        print(f"{r.model:<10}  {r.model_cindex:>8.3f}  {r.lr_cindex:>8.3f}  "
              f"{r['diff']:>+8.4f}  {ci:<22}  {sig}")


def main():
    print(f"Loading {OUTPUT_DIR}/four_way_comp_df.csv …")
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    print(f"  raw: {len(comp)} scans, {comp['pid'].nunique()} pids, "
          f"{int(comp['gold'].sum())} cancers")

    print("Attaching deployment-form imaging scores …")
    comp = attach_all_deployment_forms(comp)

    print("Attaching Lung-RADS per-scan categories …")
    comp = attach_lr(comp)
    n_missing = int(comp["lungrads_v10"].isna().sum())
    print(f"  LR coverage: {len(comp) - n_missing}/{len(comp)} "
          f"(missing dropped per-horizon by joint-valid mask)")

    auc_df = auc_paired_diffs(comp)
    cidx_df = cindex_paired_diffs(comp)
    print_report(auc_df, cidx_df)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
