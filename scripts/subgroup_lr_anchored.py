#!/usr/bin/env python3
"""Subgroup analysis under the Lorenz/Lung-RADS-anchored framework.

For each subgroup level × model, compute (with patient-clustered bootstrap):
- PR-AUC                (reference, keeps continuity with field's metric)
- Gini coefficient      (replaces top-D capture)
- Capture @ LR ≥4B-matched flag rate within subgroup (precision + recall)

Subgroups: sex, age tertile, race, pack-year quartile, screening timepoint.
"""
import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import _include, load_participant, PARTICIPANT_CSV, OUTPUT_DIR
from decile_stratification_devfit import (
    fit_lung_platt, fit_plcom_platt, fit_pillar_platt,
    attach_user_pillar, build_plcom_dev, _logit,
    _make_subgroup_levels, _plcom_extra_cols,
)
from uspstf_plcom import plcom2012_risk

LUNGRADS_CSV = os.path.join(OUTPUT_DIR, "nlst_lungrads_pinsky.csv")
LR_ORDER = {"4X": 6, "4B": 5, "4A": 4, "3": 3, "2": 2, "1": 1}
LR4B_RANK = LR_ORDER["4B"]  # ≥4B means rank ≥ 5 (LR4B + LR4X)
N_BOOT = 1000
RNG_SEED = 42


def attach_lungrads(comp):
    lr = pd.read_csv(LUNGRADS_CSV)
    lr["pid"] = lr["pid"].astype(str)
    lr["study_yr"] = lr["study_yr"].astype(int)
    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    comp["study_yr"] = comp["screen_timepoint"].astype(int)
    return comp.merge(lr[["pid", "study_yr", "lungrads_v10"]],
                      on=["pid", "study_yr"], how="left")


def build_y1_scores(comp):
    print("[setup] Fitting Platt calibrators and computing scores …")
    lung_platt = fit_lung_platt()
    part = load_participant(PARTICIPANT_CSV)
    dev_plcom = build_plcom_dev(part)
    plcom_platt = fit_plcom_platt(dev_plcom)

    # LungEvaty Platt-y1
    lung_raw = comp["risk_score"].values.astype(float)
    lung_cal = lung_platt[1].predict_proba(
        _logit(lung_raw).reshape(-1, 1))[:, 1]

    # PLCOm Platt-y1
    base_cols = ["pid", "age", "race", "pkyr", "smokeyr", "cigsmok"]
    base_cols = [c for c in base_cols if c in part.columns]
    extras = _plcom_extra_cols(part)
    base = part[base_cols + extras].drop_duplicates("pid").copy()
    base["pid"] = base["pid"].astype(str)
    pid_to_feat = base.set_index("pid")
    plcom_raw = np.full(len(comp), np.nan)
    for i, pid in enumerate(comp["pid"].values):
        if pid in pid_to_feat.index:
            plcom_raw[i] = float(plcom2012_risk(pid_to_feat.loc[pid], year=1))
    m = ~np.isnan(plcom_raw)
    plcom_cal = np.full(len(comp), np.nan)
    plcom_cal[m] = plcom_platt[1].predict_proba(
        _logit(plcom_raw[m]).reshape(-1, 1))[:, 1]

    # Pillar-0 Platt-y1
    comp = attach_user_pillar(comp)
    pillar_platt = fit_pillar_platt()
    pi_raw = comp["pillar_user_y1"].values.astype(float)
    mp = ~np.isnan(pi_raw)
    pi_cal = np.full(len(comp), np.nan)
    pi_cal[mp] = pillar_platt[1].predict_proba(
        _logit(pi_raw[mp]).reshape(-1, 1))[:, 1]

    return comp, {
        "LungEvaty": lung_cal,
        "Sybil":     comp["sv1c_risk_1"].values.astype(float),
        "Pillar-0":  pi_cal,
        "PLCOm2012": plcom_cal,
    }


def y1_inc_lbl(comp):
    times = comp["time_at_event"].values
    golds = comp["gold"].values
    inc = np.zeros(len(comp), bool)
    lbl = np.zeros(len(comp), int)
    for i, (t, g) in enumerate(zip(times, golds)):
        ok, y = _include(t, g, 0)
        inc[i] = ok; lbl[i] = y
    return inc, lbl


def lorenz_gini(scores, labels):
    """Gini = 2*(AUC under Lorenz - 0.5). Both inputs are 1-D arrays."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = labels[order].astype(float)
    n = len(y_sorted); pos = y_sorted.sum()
    x = np.arange(1, n + 1) / n
    yc = np.cumsum(y_sorted) / pos
    x = np.concatenate([[0.0], x]); yc = np.concatenate([[0.0], yc])
    auc = np.trapz(yc, x)
    return 2 * (auc - 0.5)


def compute_metrics(scores, labels, lr_score, mask):
    """Returns dict of pr_auc, gini, lr4b_match_n, lr4b_match_prec,
    lr4b_match_recall. lr_score uses the 1..6 LR_ORDER scale."""
    m = mask & ~np.isnan(scores)
    s = scores[m]; y = labels[m].astype(int); lr = lr_score[m]
    if y.sum() < 2 or len(y) < 5:
        return dict(pr_auc=np.nan, gini=np.nan, n=int(m.sum()),
                    n_cancers=int(y.sum()),
                    lr4b_n_flag=0, lr4b_prec=np.nan, lr4b_recall=np.nan,
                    lr_subgroup_recall=np.nan, lr_subgroup_prec=np.nan)
    pr_auc = float(average_precision_score(y, s))
    gini = lorenz_gini(s, y)
    # LR ≥4B within subgroup: n_flag = (lr >= 5).sum()
    lr_flag_mask = (lr >= LR4B_RANK)
    n_flag = int(lr_flag_mask.sum())
    lr_cap = int(y[lr_flag_mask].sum())
    lr_prec = lr_cap / n_flag if n_flag else np.nan
    lr_recall = lr_cap / y.sum() if y.sum() else np.nan
    if n_flag > 0:
        # Pick top n_flag by AI score
        order = np.argsort(-s, kind="mergesort")[:n_flag]
        ai_cap = int(y[order].sum())
        prec = ai_cap / n_flag
        recall = ai_cap / y.sum() if y.sum() else np.nan
    else:
        prec, recall = np.nan, np.nan
    return dict(pr_auc=pr_auc, gini=gini, n=int(m.sum()),
                n_cancers=int(y.sum()),
                lr4b_n_flag=n_flag, lr4b_prec=prec, lr4b_recall=recall,
                lr_subgroup_prec=lr_prec, lr_subgroup_recall=lr_recall)


def _bootstrap_worker(args):
    """Worker for one (subgroup_var, level, model) bootstrap pass."""
    (scores, labels, lr_score, level_mask, pids, n_boot, seed,
     sg_var, level_label, model) = args
    rng = np.random.default_rng(seed)
    sub_idx = np.where(level_mask)[0]
    sub_pids_arr = pids[sub_idx]
    pid_to_rows = {}
    for p, i in zip(sub_pids_arr, sub_idx):
        pid_to_rows.setdefault(p, []).append(i)
    pid_to_rows = {k: np.asarray(v, dtype=np.int64)
                   for k, v in pid_to_rows.items()}
    sub_pids = np.array(list(pid_to_rows.keys()))
    if len(sub_pids) < 5:
        return (sg_var, level_label, model, None, None)

    point = compute_metrics(scores, labels, lr_score, level_mask)
    boot = {k: [] for k in ("pr_auc", "gini", "lr4b_prec", "lr4b_recall")}
    for _ in range(n_boot):
        sample_pids = rng.choice(sub_pids, size=len(sub_pids), replace=True)
        idx = np.concatenate([pid_to_rows[p] for p in sample_pids])
        m = np.zeros(len(scores), bool); m[idx] = True
        met = compute_metrics(scores, labels, lr_score, m)
        for k in boot:
            boot[k].append(met[k])
    ci = {k: (float(np.nanpercentile(v, 2.5)),
              float(np.nanpercentile(v, 97.5))) for k, v in boot.items()}
    return (sg_var, level_label, model, point, ci)


def main():
    print("=" * 78)
    print("SUBGROUP ANALYSIS — LR-anchored metrics (PR-AUC ref + Gini + LR≥4B match)")
    print("=" * 78)
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    comp = attach_lungrads(comp)
    comp, scores = build_y1_scores(comp)
    inc_y1, lbl_y1 = y1_inc_lbl(comp)

    lr_score = np.array([LR_ORDER.get(str(c), 0)
                         for c in comp["lungrads_v10"].astype(str).values],
                        dtype=float)
    pids = comp["pid"].values

    subgroups = _make_subgroup_levels(comp)

    tasks = []
    for sg_var, levels in subgroups.items():
        for level_label, level_mask in levels:
            full_mask = inc_y1 & level_mask
            for model, sc in scores.items():
                tasks.append((sc, lbl_y1, lr_score, full_mask, pids,
                              N_BOOT, RNG_SEED, sg_var, level_label, model))

    print(f"[run] Bootstrapping {len(tasks)} cells × {N_BOOT} replicates "
          f"(patient-clustered) using process pool …")

    results = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(_bootstrap_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            if i % 20 == 0:
                print(f"  … {i}/{len(tasks)} done")

    # Assemble long-form CSV
    rows = []
    for sg_var, level, model, point, ci in results:
        if point is None:
            continue
        row = dict(subgroup_var=sg_var, level=level, model=model,
                   n=point["n"], n_cancers=point["n_cancers"],
                   lr_n_flag=point["lr4b_n_flag"],
                   pr_auc=point["pr_auc"],
                   pr_auc_lo=ci["pr_auc"][0], pr_auc_hi=ci["pr_auc"][1],
                   gini=point["gini"],
                   gini_lo=ci["gini"][0], gini_hi=ci["gini"][1],
                   lr4b_match_prec=point["lr4b_prec"],
                   lr4b_match_prec_lo=ci["lr4b_prec"][0],
                   lr4b_match_prec_hi=ci["lr4b_prec"][1],
                   lr4b_match_recall=point["lr4b_recall"],
                   lr4b_match_recall_lo=ci["lr4b_recall"][0],
                   lr4b_match_recall_hi=ci["lr4b_recall"][1],
                   lr_subgroup_prec=point["lr_subgroup_prec"],
                   lr_subgroup_recall=point["lr_subgroup_recall"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR,
                           "subgroup_y1_lr_anchored.csv"), index=False)
    print(f"\n→ eval_output/subgroup_y1_lr_anchored.csv")

    # Compact tables per subgroup variable
    print("\n" + "=" * 78)
    print("COMPACT TABLES per subgroup variable")
    print("=" * 78)

    md_lines = ["# Subgroup analysis (LR-anchored, y1)\n"]
    for sg_var in subgroups:
        sub = df[df.subgroup_var == sg_var]
        if len(sub) == 0: continue
        print(f"\n[{sg_var}]")
        md_lines.append(f"\n## {sg_var}\n")
        md_lines.append("| Level | Model | N | Ca | PR-AUC | Gini | "
                        "AI prec@LR≥4B vol | AI recall@LR≥4B vol | "
                        "LR ≥4B (ref) |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|")
        for lev in sub.level.unique():
            sub_l = sub[sub.level == lev]
            for _, r in sub_l.iterrows():
                low = " ⚠" if r.n_cancers < 10 else ""
                pr = f"{r.pr_auc:.3f} [{r.pr_auc_lo:.3f}, {r.pr_auc_hi:.3f}]"
                g  = f"{r.gini:.3f} [{r.gini_lo:.3f}, {r.gini_hi:.3f}]"
                ap = (f"{r.lr4b_match_prec*100:.1f}% "
                      f"[{r.lr4b_match_prec_lo*100:.1f}, "
                      f"{r.lr4b_match_prec_hi*100:.1f}]")
                ar = (f"{r.lr4b_match_recall*100:.1f}% "
                      f"[{r.lr4b_match_recall_lo*100:.1f}, "
                      f"{r.lr4b_match_recall_hi*100:.1f}]")
                lref = (f"{r.lr_subgroup_prec*100:.1f}% / "
                        f"{r.lr_subgroup_recall*100:.1f}% (n={r.lr_n_flag})")
                md_lines.append(
                    f"| {lev}{low} | {r.model} | {r.n} | {r.n_cancers} | "
                    f"{pr} | {g} | {ap} | {ar} | {lref} |")
                print(f"  {lev:<25}{low} {r.model:<10} N={r.n:<5} ca={r.n_cancers:<3} "
                      f"PR={r.pr_auc:.3f} G={r.gini:.3f} "
                      f"AI@LR≥4B: {r.lr4b_match_prec*100:.1f}%/{r.lr4b_match_recall*100:.1f}%")

    # Headline equity numbers
    print("\n" + "=" * 78)
    print("HEADLINE EQUITY: max - min Gini across levels, per subgroup × model")
    print("=" * 78)
    md_lines.append("\n## Equity summary: Gini range across subgroup levels\n")
    md_lines.append("| Subgroup var | LungEvaty range | Pillar-0 range | "
                    "Sybil range | LR ref (gold-std for context) |")
    md_lines.append("|---|---|---|---|---|")
    for sg_var in subgroups:
        sub = df[df.subgroup_var == sg_var]
        # Underpowered drop
        sub = sub[sub.n_cancers >= 10]
        if len(sub) == 0: continue
        ranges = {}
        for m in ["LungEvaty", "Pillar-0", "Sybil"]:
            g_vals = sub[sub.model == m].gini
            if len(g_vals) > 1:
                ranges[m] = float(g_vals.max() - g_vals.min())
            else:
                ranges[m] = np.nan
        print(f"  {sg_var:<20}  "
              + "  ".join(f"{m}: {ranges[m]:.3f}" for m in ranges))
        md_lines.append(
            f"| {sg_var} | {ranges['LungEvaty']:.3f} | "
            f"{ranges['Pillar-0']:.3f} | {ranges['Sybil']:.3f} | — |")

    out_md = os.path.join(OUTPUT_DIR, "subgroup_y1_lr_anchored.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\n→ {out_md}")


if __name__ == "__main__":
    main()
