#!/usr/bin/env python3
"""Paired-difference bootstrap CIs for AUC and PR-AUC at horizons 1-6.

Patient-clustered paired bootstrap, n=1000. Same indices per iteration
across all models -> per-iteration metric arrays are index-aligned ->
paired difference CI = percentiles of element-wise difference.

Output: eval_output/auc_prauc_paired_diffs.csv (one row per (year, metric,
model_pair); columns include point diff, 95% paired CI, and a
crosses_zero flag).

Pairs considered: LungEvaty vs Pillar-0, LungEvaty vs Sybil, Pillar-0 vs Sybil.
PLCOm comparisons omitted (gap to imaging is large and obvious).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import _include, OUTPUT_DIR
from layer1_layer3_full import attach_all_deployment_forms, MODELS, RNG_SEED, N_BOOT

PAIRS = [
    ("LungEvaty", "Pillar-0"),
    ("LungEvaty", "Sybil"),
    ("Pillar-0", "Sybil"),
]


def main():
    print(f"Loading {os.path.join(OUTPUT_DIR, 'four_way_comp_df.csv')} ...")
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    print(f"  raw: {len(comp)} scans, {comp['pid'].nunique()} pids")

    print("Attaching deployment-form scores (Platt for LE/P0/PLCOm; Sybil author cal) ...")
    comp = attach_all_deployment_forms(comp)

    rng = np.random.default_rng(RNG_SEED)
    rows = []

    for year in range(1, 7):
        fu = year - 1
        keep_mask = np.array([_include(t, e, fu)[0]
                              for t, e in zip(comp["time_at_event"], comp["gold"])])
        labels = np.array([_include(t, e, fu)[1]
                           for t, e in zip(comp["time_at_event"], comp["gold"])])
        sub = comp.loc[keep_mask].reset_index(drop=True)
        sub_labels = labels[keep_mask]

        score_arrays = {}
        for name, tmpl in MODELS:
            col = tmpl.format(y=year)
            vals = sub[col].values.astype(float)
            valid = ~np.isnan(vals)
            score_arrays[name] = (vals, valid)
        joint = np.ones(len(sub), dtype=bool)
        for _, (v, vmask) in score_arrays.items():
            joint &= vmask
        sub2 = sub.loc[joint].reset_index(drop=True)
        labels2 = sub_labels[joint]
        score_arrays = {n: v[joint] for n, (v, _) in score_arrays.items()}

        n = len(sub2)
        n_ca = int(labels2.sum())
        print(f"\nYear {year}: N={n}, cancers={n_ca}")

        pids = sub2["pid"].values
        unique_pids = np.array(sorted(set(pids.tolist())))
        pid_to_rows = {p: np.where(pids == p)[0] for p in unique_pids}

        boot_auc = {name: [] for name, _ in MODELS}
        boot_prauc = {name: [] for name, _ in MODELS}

        for b in range(N_BOOT):
            sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
            idx = np.concatenate([pid_to_rows[p] for p in sampled])
            yb = labels2[idx]
            if yb.sum() == 0 or len(np.unique(yb)) < 2:
                continue
            for name in score_arrays:
                sb = score_arrays[name][idx]
                boot_auc[name].append(float(roc_auc_score(yb, sb)))
                boot_prauc[name].append(float(average_precision_score(yb, sb)))
            if (b + 1) % 200 == 0:
                print(f"    bootstrap {b+1}/{N_BOOT}")

        for name in score_arrays:
            boot_auc[name] = np.array(boot_auc[name])
            boot_prauc[name] = np.array(boot_prauc[name])

        point_auc = {name: float(roc_auc_score(labels2, score_arrays[name]))
                     for name in score_arrays}
        point_prauc = {name: float(average_precision_score(labels2, score_arrays[name]))
                       for name in score_arrays}

        for m1, m2 in PAIRS:
            for metric_name, boot_dict, point_dict in [
                ("auc", boot_auc, point_auc),
                ("prauc", boot_prauc, point_prauc),
            ]:
                diff = boot_dict[m1] - boot_dict[m2]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                pt_diff = point_dict[m1] - point_dict[m2]
                rows.append(dict(
                    year=year, N=n, N_cancer=n_ca,
                    metric=metric_name,
                    model_a=m1, model_b=m2,
                    diff=pt_diff,
                    diff_lo=float(lo),
                    diff_hi=float(hi),
                    crosses_zero=bool(lo < 0 < hi),
                    n_boot=int(len(diff)),
                ))

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "auc_prauc_paired_diffs.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 88)
    print("PAIRED-DIFFERENCE 95% CIs (patient-clustered bootstrap, n=1000)")
    print("Format: diff [lo, hi]; * = CI excludes zero (significantly different)")
    print("=" * 88)
    for metric in ["auc", "prauc"]:
        print(f"\n{metric.upper()}")
        for year in range(1, 7):
            for m1, m2 in PAIRS:
                r = df[(df.year == year) & (df.metric == metric)
                       & (df.model_a == m1) & (df.model_b == m2)].iloc[0]
                sig = "*" if not r.crosses_zero else " "
                print(f"  Y{year} {m1:<10} - {m2:<10}: "
                      f"{r['diff']:+.4f} [{r.diff_lo:+.4f}, {r.diff_hi:+.4f}]{sig}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
