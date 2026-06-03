#!/usr/bin/env python3
"""Paired-difference bootstrap CIs for imaging-vs-Lung-RADS recall at the
LR-anchored Cat>=4B flag rate (4.11%), companion to Table 2 (cancer capture).

For each (year, cohort-subset), at the FR = 4.11% flag rate that matches
Lung-RADS' Cat>=4B workup volume:
  - LR own recall: fraction of cancers caught by the LR>=4B set
  - Imaging recall: fraction of cancers caught at the AI threshold producing
    flag rate <= 4.11% (matching LR's volume)

Paired patient-clustered bootstrap (n=1000): per iteration, both LR recall
and each imaging-model recall are computed on the same resample, so
paired (imaging - LR) diffs are valid.

Cohort subsets:
  - y1 all-TP   : full matched cohort at year 1
  - y1 T0-only  : prevalence round only (info-symmetric comparison)
  - y3 all-TP   : full matched cohort at year 3
  - y6 all-TP   : complete-case subset at year 6 (n=4339)

Output: eval_output/paired_diffs_lr_capture.csv
"""
import os
import sys
import numpy as np
import pandas as pd

OUTPUT_DIR = os.environ.get("SCREENING_METRICS_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_output"))
LR_CSV = os.path.join(OUTPUT_DIR, "nlst_lungrads_pinsky.csv")
N_BOOT = 1000
RNG_SEED = 20260527
FR_TARGET = 0.0411  # LR Cat>=4B-matched flag rate

LR_ORDER = {"1": 1, "2": 2, "3": 3, "4A": 4, "4B": 5, "4X": 6}
LR4B_RANK = LR_ORDER["4B"]

MODELS = [
    ("LungEvaty",  "lungevity_platt_y{y}"),
    ("Pillar-0",   "pillar_platt_y{y}"),
    ("Sybil",      "sv1c_risk_{y}"),
]


def _include(censor_time, gold, fu, fup_lower_bound=-1):
    valid_pos = bool(gold) and (censor_time <= fu) and (censor_time > fup_lower_bound)
    valid_neg = censor_time >= fu
    return (valid_pos or valid_neg), int(valid_pos)


def recall_at_fr(scores, labels, fr_target):
    """Recall at the threshold that gives flag rate <= fr_target."""
    n = len(scores)
    n_flag = max(1, int(np.round(fr_target * n)))
    # Sort descending; top n_flag scans flagged
    order = np.argsort(-scores, kind="mergesort")
    flag_idx = order[:n_flag]
    flag = np.zeros(n, dtype=bool)
    flag[flag_idx] = True
    n_ca = int(labels.sum())
    if n_ca == 0:
        return np.nan
    return float(np.sum(flag & (labels == 1))) / n_ca


def lr_recall(lr_score, labels, lr_threshold_rank):
    """Recall of the LR rule flagging LR-rank >= threshold."""
    flag = lr_score >= lr_threshold_rank
    n_ca = int(labels.sum())
    if n_ca == 0:
        return np.nan
    return float(np.sum(flag & (labels == 1))) / n_ca


def main():
    print("Loading comp_df ...")
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)

    print("Attaching deployment-form imaging scores ...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from layer1_layer3_full import attach_all_deployment_forms
    comp = attach_all_deployment_forms(comp)

    print("Attaching Lung-RADS per-scan categories ...")
    lr = pd.read_csv(LR_CSV)
    lr["pid"] = lr["pid"].astype(str)
    lr["study_yr"] = lr["study_yr"].astype(int)
    comp["study_yr"] = comp["screen_timepoint"].astype(int)
    comp = comp.merge(lr[["pid", "study_yr", "lungrads_v10"]],
                      on=["pid", "study_yr"], how="left")
    lr_score = np.array([LR_ORDER.get(str(c), 0)
                         for c in comp["lungrads_v10"].astype(str).values],
                        dtype=float)

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    diff_rows = []

    subsets = [
        ("y1 all-TP", 1, None),
        ("y1 T0-only", 1, 0),
        ("y3 all-TP", 3, None),
        ("y6 CC", 6, None),
    ]

    for label, year, tp_filter in subsets:
        fu = year - 1
        keep = np.array([_include(t, g, fu)[0]
                         for t, g in zip(comp["time_at_event"], comp["gold"])])
        labels = np.array([_include(t, g, fu)[1]
                           for t, g in zip(comp["time_at_event"], comp["gold"])])
        if tp_filter is not None:
            keep &= (comp["screen_timepoint"].astype(int).values == tp_filter)
        sub_idx = np.where(keep)[0]

        # Joint-valid mask across all imaging models
        joint = np.ones(len(sub_idx), dtype=bool)
        for name, tmpl in MODELS:
            col = tmpl.format(y=year)
            vals = comp[col].values[sub_idx].astype(float)
            joint &= ~np.isnan(vals)
        sub_idx = sub_idx[joint]
        sub_labels = labels[sub_idx]
        sub_lr = lr_score[sub_idx]
        sub_pids = comp["pid"].values[sub_idx]
        score_arrays = {name: comp[tmpl.format(y=year)].values[sub_idx].astype(float)
                        for name, tmpl in MODELS}

        n = len(sub_idx)
        n_ca = int(sub_labels.sum())
        print(f"\n{label}: N={n}, cancers={n_ca}")

        # Point estimates
        point_lr = lr_recall(sub_lr, sub_labels, LR4B_RANK)
        point_models = {name: recall_at_fr(score_arrays[name], sub_labels, FR_TARGET)
                        for name in score_arrays}

        # Patient-clustered paired bootstrap
        unique_pids = np.array(sorted(set(sub_pids.tolist())))
        pid_to_rows = {p: np.where(sub_pids == p)[0] for p in unique_pids}

        boot_lr = []
        boot_models = {name: [] for name in score_arrays}

        for b in range(N_BOOT):
            sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
            idx = np.concatenate([pid_to_rows[p] for p in sampled])
            yb = sub_labels[idx]
            if yb.sum() == 0:
                continue
            boot_lr.append(lr_recall(sub_lr[idx], yb, LR4B_RANK))
            for name in score_arrays:
                boot_models[name].append(
                    recall_at_fr(score_arrays[name][idx], yb, FR_TARGET))
            if (b + 1) % 200 == 0:
                print(f"    bootstrap {b+1}/{N_BOOT}")

        boot_lr = np.array(boot_lr)
        for name in score_arrays:
            boot_models[name] = np.array(boot_models[name])

        # LR own recall row
        lo, hi = np.percentile(boot_lr, [2.5, 97.5])
        rows.append(dict(
            subset=label, year=year, n=n, n_ca=n_ca,
            strategy="Lung-RADS (LR>=4B)",
            recall=point_lr,
            ci_lo=float(lo), ci_hi=float(hi),
        ))
        # Imaging rows and paired diffs
        for name in score_arrays:
            arr = boot_models[name]
            lo_m, hi_m = np.percentile(arr, [2.5, 97.5])
            rows.append(dict(
                subset=label, year=year, n=n, n_ca=n_ca,
                strategy=name,
                recall=point_models[name],
                ci_lo=float(lo_m), ci_hi=float(hi_m),
            ))
            diff = arr - boot_lr
            d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
            diff_rows.append(dict(
                subset=label, year=year, model=name,
                diff=point_models[name] - point_lr,
                diff_lo=float(d_lo), diff_hi=float(d_hi),
                crosses_zero=bool(d_lo < 0 < d_hi),
            ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "paired_diffs_lr_capture.csv"), index=False)
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(os.path.join(OUTPUT_DIR, "paired_diffs_lr_capture_diffs.csv"),
                   index=False)

    print("\n" + "=" * 90)
    print(f"RECALL AT LR Cat>=4B MATCHED FLAG RATE ({FR_TARGET*100:.2f}%), %")
    print("Paired patient-clustered bootstrap (n=1000)")
    print("=" * 90)
    for _, r in df.iterrows():
        print(f"{r.subset:<14} {r.strategy:<22} "
              f"recall = {r.recall*100:.1f}% "
              f"[{r.ci_lo*100:.1f}%, {r.ci_hi*100:.1f}%]")

    print("\n" + "=" * 90)
    print("PAIRED (IMAGING - LR) RECALL DIFFS (pp); * = CI excludes zero")
    print("=" * 90)
    for _, r in diff_df.iterrows():
        sig = "*" if not r.crosses_zero else " "
        print(f"  {r.subset:<14} {r.model:<10} - LR: "
              f"diff = {r['diff']*100:+.1f} pp "
              f"[{r.diff_lo*100:+.1f}, {r.diff_hi*100:+.1f}]{sig}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
