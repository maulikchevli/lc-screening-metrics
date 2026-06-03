#!/usr/bin/env python3
"""Lung-RADS NB at LR-anchored probability thresholds, paired bootstrap CIs.

For each p_t in {0.01, 0.02, 0.05, 0.10, 0.15}, compute the NB of the
Lung-RADS categorical rule whose published Risk-of-Malignancy lower bound
brackets p_t (LR v1.1 RoM thresholds; Pinsky 2015 retrospective assignment):

    p_t = 0.01, 0.02  -> flag if LR >= 3   (LR3 RoM = 1-2%)
    p_t = 0.05, 0.10  -> flag if LR >= 4A  (LR4A RoM = 5-15%)
    p_t = 0.15        -> flag if LR >= 4B  (LR4B RoM >= 15%)

NB(rule, p_t) = w_TP/N - w_FP/N * p_t/(1-p_t)
  y1, y3: IPCW-weighted (KM censoring distribution)
  y6:     complete-case on the n=4,339 subset (matches paper convention)

Bootstrap: patient-clustered, n=1000, paired with all imaging models in
the same iterations -> per-iteration (imaging - LR) differences are valid
paired draws.

Internal sensitivity check: also compute the "best LR rule per p_t"
envelope. If the deployed-rule NB is within a small NB*1000 of the
envelope at every cell, no need to discuss; otherwise flag.

Outputs:
    eval_output/lungrads_dca.csv             (NB per (year, p_t, strategy))
    eval_output/lungrads_dca_paired_diffs.csv (paired imaging - LR diffs)
    eval_output/lungrads_dca_envelope.csv    (sensitivity check)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import _include, _ipcw_weights, OUTPUT_DIR
from layer1_layer3_full import attach_all_deployment_forms

LR_CSV = os.path.join(os.environ.get("SCREENING_METRICS_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_output")), "nlst_lungrads_pinsky.csv")
P_T_GRID = [0.01, 0.02, 0.05, 0.10, 0.15]
P_T_TO_LR_RULE = {
    0.01: "LR3+",
    0.02: "LR3+",
    0.05: "LR4A+",
    0.10: "LR4A+",
    0.15: "LR4B+",
}
LR_RULES = {
    "LR3+":  lambda lr: np.isin(lr, ["3", "4A", "4B", "4X"]),
    "LR4A+": lambda lr: np.isin(lr, ["4A", "4B", "4X"]),
    "LR4B+": lambda lr: np.isin(lr, ["4B", "4X"]),
}
MODELS = [
    ("LungEvaty",  "lungevity_platt_y{y}"),
    ("Pillar-0",   "pillar_platt_y{y}"),
    ("Sybil",      "sv1c_risk_{y}"),
    ("PLCOm2012",  "plcom_platt_y{y}"),
]
N_BOOT = 1000
RNG_SEED = 20260526


def nb_categorical_ipcw(flag, times, events, fu, pt):
    """NB for a binary flag rule under IPCW weights at horizon fu."""
    tp_w, fp_w, is_tp, is_fp, _ = _ipcw_weights(times, events, fu)
    n = len(flag)
    w_tp = float(np.sum(tp_w[is_tp & flag]))
    w_fp = float(np.sum(fp_w * (is_fp & flag).astype(float)))
    return (w_tp - w_fp * (pt / (1 - pt))) / n


def nb_continuous_ipcw(scores, times, events, fu, pt):
    """NB for a continuous score thresholded at pt under IPCW weights."""
    return nb_categorical_ipcw(scores >= pt, times, events, fu, pt)


def nb_categorical_cc(flag, labels, pt):
    """NB for a binary flag rule on complete-case labels at horizon."""
    n = len(flag)
    tp = int(np.sum(flag & (labels == 1)))
    fp = int(np.sum(flag & (labels == 0)))
    return (tp - fp * (pt / (1 - pt))) / n


def nb_continuous_cc(scores, labels, pt):
    return nb_categorical_cc(scores >= pt, labels, pt)


def attach_lungrads(comp):
    """Merge LR v1.0 categorical assignment from Pinsky reproduction CSV."""
    lr = pd.read_csv(LR_CSV, dtype={"pid": str, "study_yr": int,
                                    "lungrads_v10": str})
    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    comp["screen_timepoint"] = comp["screen_timepoint"].astype(int)
    lr = lr.rename(columns={"study_yr": "screen_timepoint"})
    lr = lr[["pid", "screen_timepoint", "lungrads_v10"]].drop_duplicates(
        ["pid", "screen_timepoint"])
    out = comp.merge(lr, on=["pid", "screen_timepoint"], how="left")
    miss = out["lungrads_v10"].isna().sum()
    if miss:
        print(f"  warning: {miss} scans missing LR assignment "
              f"(likely matched-cohort/LR-cohort mismatch)")
    return out


def main():
    print("Loading comp_df ...")
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    print(f"  raw: {len(comp)} scans")

    print("Attaching deployment-form scores (LE/P0 Platt; Sybil author; PLCOm Platt) ...")
    comp = attach_all_deployment_forms(comp)

    print("Attaching Pinsky LR v1.0 per-scan categories ...")
    comp = attach_lungrads(comp)
    lr_cats = comp["lungrads_v10"].astype(str).values
    # Pre-compute LR rule masks
    lr_masks = {name: rule(lr_cats) for name, rule in LR_RULES.items()}
    for name, m in lr_masks.items():
        print(f"  {name}: {int(m.sum())} / {len(m)} scans flagged "
              f"({100*m.mean():.2f}%)")

    rng = np.random.default_rng(RNG_SEED)
    out_rows = []
    diff_rows = []
    env_rows = []

    for year in [1, 3, 6]:
        fu = year - 1
        cc = (year == 6)  # complete-case for year 6 only

        if cc:
            # Complete-case subset: drop censored-before-y6
            keep = np.array([_include(t, e, fu)[0]
                             for t, e in zip(comp["time_at_event"],
                                             comp["gold"])])
            labels = np.array([_include(t, e, fu)[1]
                               for t, e in zip(comp["time_at_event"],
                                               comp["gold"])])
            sub = comp.loc[keep].reset_index(drop=True)
            sub_labels = labels[keep]
            lr_sub_masks = {n: m[keep] for n, m in lr_masks.items()}
            print(f"\nYear {year} (complete-case): N={len(sub)}, cancers={int(sub_labels.sum())}")
        else:
            sub = comp
            lr_sub_masks = lr_masks
            print(f"\nYear {year} (IPCW): N={len(sub)}")

        # Joint-valid mask across all imaging models
        score_cols = {name: tmpl.format(y=year) for name, tmpl in MODELS}
        joint = np.ones(len(sub), dtype=bool)
        for name, col in score_cols.items():
            joint &= ~np.isnan(sub[col].values)
        sub2 = sub.loc[joint].reset_index(drop=True)
        lr_masks_y = {n: m[joint] for n, m in lr_sub_masks.items()}
        scores_y = {name: sub2[col].values.astype(float)
                    for name, col in score_cols.items()}
        if cc:
            labels_y = sub_labels[joint]
        times_y = sub2["time_at_event"].values.astype(float)
        events_y = sub2["gold"].values.astype(int)
        n_y = len(sub2)
        print(f"  joint-valid subset: N={n_y}")

        # Point estimates
        point_lr = {}
        for pt in P_T_GRID:
            rule_name = P_T_TO_LR_RULE[pt]
            flag = lr_masks_y[rule_name]
            if cc:
                nb = nb_categorical_cc(flag, labels_y, pt)
            else:
                nb = nb_categorical_ipcw(flag, times_y, events_y, fu, pt)
            point_lr[(pt, rule_name)] = nb

        point_envelope = {}
        for pt in P_T_GRID:
            best_nb, best_rule = -np.inf, None
            for rname, m in lr_masks_y.items():
                if cc:
                    nb = nb_categorical_cc(m, labels_y, pt)
                else:
                    nb = nb_categorical_ipcw(m, times_y, events_y, fu, pt)
                if nb > best_nb:
                    best_nb, best_rule = nb, rname
            point_envelope[pt] = (best_nb, best_rule)

        point_models = {name: {} for name in scores_y}
        for name in scores_y:
            for pt in P_T_GRID:
                if cc:
                    nb = nb_continuous_cc(scores_y[name], labels_y, pt)
                else:
                    nb = nb_continuous_ipcw(scores_y[name], times_y, events_y, fu, pt)
                point_models[name][pt] = nb

        # Bootstrap (paired patient-clustered)
        pids = sub2["pid"].values
        unique_pids = np.array(sorted(set(pids.tolist())))
        pid_to_rows = {p: np.where(pids == p)[0] for p in unique_pids}

        boot_lr = {pt: [] for pt in P_T_GRID}
        boot_models = {name: {pt: [] for pt in P_T_GRID} for name in scores_y}

        for b in range(N_BOOT):
            sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
            idx = np.concatenate([pid_to_rows[p] for p in sampled])
            t_b = times_y[idx]; e_b = events_y[idx]
            if cc:
                yb = labels_y[idx]
                if yb.sum() == 0:
                    continue
            for pt in P_T_GRID:
                rule_name = P_T_TO_LR_RULE[pt]
                fb = lr_masks_y[rule_name][idx]
                if cc:
                    boot_lr[pt].append(nb_categorical_cc(fb, yb, pt))
                else:
                    try:
                        boot_lr[pt].append(
                            nb_categorical_ipcw(fb, t_b, e_b, fu, pt))
                    except Exception:
                        boot_lr[pt].append(np.nan)
            for name in scores_y:
                s_b = scores_y[name][idx]
                for pt in P_T_GRID:
                    if cc:
                        boot_models[name][pt].append(
                            nb_continuous_cc(s_b, yb, pt))
                    else:
                        try:
                            boot_models[name][pt].append(
                                nb_continuous_ipcw(s_b, t_b, e_b, fu, pt))
                        except Exception:
                            boot_models[name][pt].append(np.nan)
            if (b + 1) % 200 == 0:
                print(f"    bootstrap {b+1}/{N_BOOT}")

        # CIs and paired diffs
        for pt in P_T_GRID:
            rule_name = P_T_TO_LR_RULE[pt]
            lr_arr = np.array([v for v in boot_lr[pt] if not np.isnan(v)])
            lr_lo, lr_hi = np.percentile(lr_arr, [2.5, 97.5])
            out_rows.append(dict(
                year=year, p_t=pt, strategy=f"Lung-RADS ({rule_name})",
                nb=point_lr[(pt, rule_name)],
                ci_lo=float(lr_lo), ci_hi=float(lr_hi),
            ))
            env_nb, env_rule = point_envelope[pt]
            env_rows.append(dict(
                year=year, p_t=pt,
                deployed_rule=rule_name, deployed_nb=point_lr[(pt, rule_name)],
                best_rule=env_rule, best_nb=env_nb,
                nb_gap_x1000=(env_nb - point_lr[(pt, rule_name)]) * 1000,
            ))
            for name in scores_y:
                m_arr = np.array([v for v in boot_models[name][pt]
                                  if not np.isnan(v)])
                m_lo, m_hi = np.percentile(m_arr, [2.5, 97.5])
                out_rows.append(dict(
                    year=year, p_t=pt, strategy=name,
                    nb=point_models[name][pt],
                    ci_lo=float(m_lo), ci_hi=float(m_hi),
                ))
                if len(lr_arr) == len(m_arr):
                    diff = m_arr - lr_arr
                    d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
                    diff_rows.append(dict(
                        year=year, p_t=pt, model=name,
                        lr_rule=rule_name,
                        diff=point_models[name][pt] - point_lr[(pt, rule_name)],
                        diff_lo=float(d_lo), diff_hi=float(d_hi),
                        crosses_zero=bool(d_lo < 0 < d_hi),
                    ))

    df = pd.DataFrame(out_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "lungrads_dca.csv"), index=False)
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(os.path.join(OUTPUT_DIR, "lungrads_dca_paired_diffs.csv"),
                   index=False)
    env_df = pd.DataFrame(env_rows)
    env_df.to_csv(os.path.join(OUTPUT_DIR, "lungrads_dca_envelope.csv"),
                  index=False)
    print(f"\nSaved: {OUTPUT_DIR}/lungrads_dca.csv "
          f"({len(df)} rows)")

    print("\n" + "=" * 90)
    print("LUNG-RADS NB AT LR-ANCHORED PROBABILITY THRESHOLDS (NB x 1000)")
    print("Paired patient-clustered bootstrap (n=1000)")
    print("=" * 90)
    print(f"{'Year':>4} {'p_t':>5} {'Strategy':<24} {'NB*1000':>10} {'95% CI':<26}")
    print("-" * 90)
    for _, r in df.iterrows():
        ci = f"[{r.ci_lo*1000:+.1f}, {r.ci_hi*1000:+.1f}]"
        print(f"{int(r.year):>4} {r.p_t:>5.2f} {r.strategy:<24} "
              f"{r.nb*1000:>10.2f} {ci:<26}")

    print("\n" + "=" * 90)
    print("PAIRED IMAGING - LR DIFFS (NB x 1000); * = CI excludes zero")
    print("=" * 90)
    for _, r in diff_df.iterrows():
        sig = "*" if not r.crosses_zero else " "
        print(f"  y{int(r.year)} p_t={r.p_t:.2f} {r.model:<10} - LR({r.lr_rule:<5}): "
              f"diff = {r['diff']*1000:+.2f} "
              f"[{r.diff_lo*1000:+.2f}, {r.diff_hi*1000:+.2f}]{sig}")

    print("\n" + "=" * 90)
    print("DEPLOYED-RULE vs BEST-LR-RULE ENVELOPE (sensitivity check)")
    print("If gap_x1000 is small, the deployed lower-bracket rule is near-optimal.")
    print("=" * 90)
    for _, r in env_df.iterrows():
        same = " (same rule)" if r.deployed_rule == r.best_rule else ""
        print(f"  y{int(r.year)} p_t={r.p_t:.2f}: deployed={r.deployed_rule} "
              f"NB={r.deployed_nb*1000:+.2f}  |  best={r.best_rule} "
              f"NB={r.best_nb*1000:+.2f}  |  gap = {r.nb_gap_x1000:+.2f}{same}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
