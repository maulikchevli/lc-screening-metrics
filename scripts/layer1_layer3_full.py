#!/usr/bin/env python3
"""Compute Layer 1 (discrimination) + Layer 3 (calibration) metrics with
patient-clustered bootstrap CIs for all four paper models in deployment form.

Models (one row per model — deployment form):
  LungEvaty   — Platt-recalibrated on NLST dev (per horizon)
  Pillar-0    — Platt-recalibrated on user-reproduced NLST dev (per horizon)
  Sybil v1.0  — author-released calibrator
  PLCOm2012   — Platt-recalibrated on NLST dev (per horizon)

Metrics per (model, year):
  AUC, PR-AUC          (rank-based; calibration-invariant)
  Brier, Reliability,  (deployment form)
  Resolution, Uncertainty

Bootstrap: paired patient-clustered, n=1000.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import (
    _include, _brier_decomposition, load_participant,
    OUTPUT_DIR, PARTICIPANT_CSV, MAX_FOLLOWUP,
)
from uspstf_plcom import plcom2012_risk
from decile_stratification_devfit import (
    _logit, _plcom_extra_cols,
    fit_lung_platt, fit_plcom_platt, build_plcom_dev,
    fit_pillar_platt, attach_user_pillar,
)

RNG_SEED = 20260519
N_BOOT = 1000

# (paper label, score-column template; {y} = horizon)
MODELS = [
    ("LungEvaty",  "lungevity_platt_y{y}"),
    ("Pillar-0",   "pillar_platt_y{y}"),
    ("Sybil",      "sv1c_risk_{y}"),
    ("PLCOm2012",  "plcom_platt_y{y}"),
]


def attach_all_deployment_forms(comp):
    """Compute Platt for LungEvaty, Pillar-0, PLCOm; Sybil cal already present."""
    part = load_participant(PARTICIPANT_CSV)
    part["pid"] = part["pid"].astype(str)
    extras = _plcom_extra_cols(part)
    ext = part[["pid"] + extras].drop_duplicates("pid").copy()
    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    comp = comp.merge(ext, on="pid", how="left")

    # PLCOm raw
    for y in (1, 2, 3, 4, 5, 6):
        comp[f"plcom_y{y}"] = comp.apply(
            lambda r, _y=y: plcom2012_risk(r, year=_y), axis=1)

    # LungEvaty Platt
    lung_platt = fit_lung_platt()
    for y in range(1, 7):
        if y in lung_platt:
            comp[f"lungevity_platt_y{y}"] = lung_platt[y].predict_proba(
                _logit(comp["risk_score"].values).reshape(-1, 1))[:, 1]
        else:
            # Years 2/4/5 not in YEARS=(1,3,6) of decile pipeline → fit ad-hoc
            comp[f"lungevity_platt_y{y}"] = _fit_platt_year(
                os.environ.get("LUNGEVATY_DEV_JSON", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions_lungevaty_dev.json")),
                "cancer_risk", y, comp["risk_score"].values)

    # Pillar-0 Platt (per the decile script's helper)
    pillar_platt = fit_pillar_platt()
    comp = attach_user_pillar(comp)
    for y in range(1, 7):
        if y in pillar_platt:
            vals = comp[f"pillar_user_y{y}"].values.astype(float)
            mask = ~np.isnan(vals)
            out = np.full_like(vals, np.nan, dtype=float)
            out[mask] = pillar_platt[y].predict_proba(
                _logit(vals[mask]).reshape(-1, 1))[:, 1]
            comp[f"pillar_platt_y{y}"] = out
        else:
            comp[f"pillar_platt_y{y}"] = _fit_pillar_platt_year(y, comp)

    # PLCOm Platt — extend dev to all 6 years before fitting
    dev_plcom = build_plcom_dev(part)
    for y in (2, 4, 5):
        if f"plcom_y{y}" not in dev_plcom.columns:
            dev_plcom[f"plcom_y{y}"] = dev_plcom.apply(
                lambda r, _y=y: plcom2012_risk(r, year=_y), axis=1)
    plcom_platt = fit_plcom_platt(dev_plcom)
    for y in range(1, 7):
        if y in plcom_platt:
            vals = comp[f"plcom_y{y}"].values.astype(float)
            mask = ~np.isnan(vals)
            out = np.full_like(vals, np.nan, dtype=float)
            out[mask] = plcom_platt[y].predict_proba(
                _logit(vals[mask]).reshape(-1, 1))[:, 1]
            comp[f"plcom_platt_y{y}"] = out
        else:
            comp[f"plcom_platt_y{y}"] = _fit_plcom_platt_year(y, dev_plcom, comp)
    return comp


def _fit_platt_year(json_path, key, year, test_raw):
    """Ad-hoc per-year Platt fit on a LungEvaty-style dev JSON."""
    with open(json_path) as f:
        dev = json.load(f)
    fu = year - 1
    sl, yl = [], []
    for r in dev:
        inc, lbl = _include(r["time_at_event"], r["y"], fu)
        if inc:
            sl.append(float(r[key][year - 1]))
            yl.append(lbl)
    s = np.array(sl); y_arr = np.array(yl)
    clf = LogisticRegression(C=1e6, solver="lbfgs")
    clf.fit(_logit(s).reshape(-1, 1), y_arr)
    return clf.predict_proba(_logit(test_raw).reshape(-1, 1))[:, 1]


def _fit_pillar_platt_year(year, comp):
    PILLAR_USER_DEV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions_pillar_dev.csv")
    dev = pd.read_csv(PILLAR_USER_DEV)
    fu = year - 1
    sl, yl = [], []
    for _, r in dev.iterrows():
        inc, lbl = _include(r["time_at_event"], r["y"], fu)
        if inc:
            sl.append(float(r[f"prob_y{year}"]))
            yl.append(int(lbl))
    s = np.array(sl); y_arr = np.array(yl)
    clf = LogisticRegression(C=1e6, solver="lbfgs")
    clf.fit(_logit(s).reshape(-1, 1), y_arr)
    vals = comp[f"pillar_user_y{year}"].values.astype(float)
    mask = ~np.isnan(vals)
    out = np.full_like(vals, np.nan, dtype=float)
    out[mask] = clf.predict_proba(_logit(vals[mask]).reshape(-1, 1))[:, 1]
    return out


def _fit_plcom_platt_year(year, dev_plcom, comp):
    fu = year - 1
    sl, yl = [], []
    for _, r in dev_plcom.iterrows():
        inc, lbl = _include(r["time_at_event"], r["gold"], fu)
        v = r.get(f"plcom_y{year}")
        if v is None or pd.isna(v):
            continue
        if inc:
            sl.append(float(v))
            yl.append(int(lbl))
    s = np.array(sl); y_arr = np.array(yl)
    clf = LogisticRegression(C=1e6, solver="lbfgs")
    clf.fit(_logit(s).reshape(-1, 1), y_arr)
    vals = comp[f"plcom_y{year}"].values.astype(float)
    mask = ~np.isnan(vals)
    out = np.full_like(vals, np.nan, dtype=float)
    out[mask] = clf.predict_proba(_logit(vals[mask]).reshape(-1, 1))[:, 1]
    return out


def metrics_for_subset(scores, labels):
    if labels.sum() == 0 or len(np.unique(labels)) < 2:
        return dict(auc=np.nan, prauc=np.nan, brier=np.nan,
                    reliab=np.nan, resol=np.nan, uncert=np.nan)
    sc = np.clip(scores, 1e-7, 1 - 1e-7)
    rel, res, unc = _brier_decomposition(labels, sc)
    return dict(
        auc=float(roc_auc_score(labels, scores)),
        prauc=float(average_precision_score(labels, scores)),
        brier=float(brier_score_loss(labels, sc)),
        reliab=float(rel), resol=float(res), uncert=float(unc),
    )


def main():
    print("Loading comp_df (Pillar-0 native user reproduction) …")
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    print(f"  N = {len(comp)} scans, {comp['pid'].nunique()} pids")

    print("Attaching deployment-form scores for all 4 models …")
    comp = attach_all_deployment_forms(comp)

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    boot_arrays = {}

    for year in range(1, 7):
        fu = year - 1
        # Cohort filter (matches Layer 2 convention: complete-case via _include)
        keep_mask = np.array([_include(t, e, fu)[0]
                              for t, e in zip(comp["time_at_event"],
                                              comp["gold"])])
        labels = np.array([_include(t, e, fu)[1]
                           for t, e in zip(comp["time_at_event"],
                                           comp["gold"])])
        sub = comp.loc[keep_mask].reset_index(drop=True)
        sub_labels = labels[keep_mask]

        score_arrays = {}
        for name, tmpl in MODELS:
            col = tmpl.format(y=year)
            vals = sub[col].values.astype(float)
            valid = ~np.isnan(vals)
            score_arrays[name] = (vals, valid)
        # Joint-valid mask (all 4 models present)
        joint = np.ones(len(sub), dtype=bool)
        for _, (v, vmask) in score_arrays.items():
            joint &= vmask
        sub2 = sub.loc[joint].reset_index(drop=True)
        labels2 = sub_labels[joint]
        score_arrays = {n: v[joint] for n, (v, _) in score_arrays.items()}

        n = len(sub2)
        n_ca = int(labels2.sum())
        prev = n_ca / n if n else np.nan
        print(f"\nYear {year}: N={n}, cancers={n_ca}, prev={prev:.4f}")

        # Point estimates
        point_ests = {}
        for name in score_arrays:
            point_ests[name] = metrics_for_subset(score_arrays[name], labels2)

        # Patient-clustered paired bootstrap
        pids = sub2["pid"].values
        unique_pids = np.array(sorted(set(pids.tolist())))
        pid_to_rows = {p: np.where(pids == p)[0] for p in unique_pids}
        boot = {name: {k: [] for k in
                       ("auc", "prauc", "brier", "reliab", "resol")}
                for name, _ in MODELS}
        for b in range(N_BOOT):
            sampled = rng.choice(unique_pids, size=len(unique_pids),
                                  replace=True)
            idx = np.concatenate([pid_to_rows[p] for p in sampled])
            yb = labels2[idx]
            if yb.sum() == 0 or len(np.unique(yb)) < 2:
                continue
            for name in score_arrays:
                sb = score_arrays[name][idx]
                m = metrics_for_subset(sb, yb)
                for k in ("auc", "prauc", "brier", "reliab", "resol"):
                    if not np.isnan(m[k]):
                        boot[name][k].append(m[k])
            if (b + 1) % 200 == 0:
                print(f"    bootstrap {b+1}/{N_BOOT}")

        for name in score_arrays:
            row = dict(model=name, year=year, N=n, N_cancer=n_ca, prev=prev)
            for k in ("auc", "prauc", "brier", "reliab", "resol"):
                row[k] = point_ests[name][k]
                arr = np.array(boot[name][k], dtype=float)
                arr = arr[~np.isnan(arr)]
                if len(arr):
                    row[f"{k}_lo"] = float(np.percentile(arr, 2.5))
                    row[f"{k}_hi"] = float(np.percentile(arr, 97.5))
                else:
                    row[f"{k}_lo"] = row[f"{k}_hi"] = np.nan
            row["uncert"] = point_ests[name]["uncert"]
            rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, "layer1_layer3_full.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")

    # Pretty-print markdown tables
    print("\n" + "=" * 80)
    print("AUC")
    print("=" * 80)
    print(_markdown_table(df, "auc"))
    print("\nPR-AUC")
    print(_markdown_table(df, "prauc"))
    print("\nBrier (deployment form)")
    print(_markdown_table(df, "brier"))
    print("\nReliability")
    print(_markdown_table(df, "reliab"))
    print("\nResolution")
    print(_markdown_table(df, "resol"))


def _markdown_table(df, metric):
    lines = []
    cols = [m for m, _ in MODELS]
    lines.append("| Year | " + " | ".join(cols) + " |")
    lines.append("|---|" + "|".join(["---"] * len(cols)) + "|")
    for y in range(1, 7):
        sub = df[df.year == y]
        cells = []
        for c in cols:
            r = sub[sub.model == c]
            if len(r) == 0:
                cells.append("—")
            else:
                v = r[metric].iloc[0]
                lo = r[f"{metric}_lo"].iloc[0]
                hi = r[f"{metric}_hi"].iloc[0]
                if pd.isna(v):
                    cells.append("—")
                else:
                    digits = 4 if metric in ("brier", "reliab", "resol") else 3
                    cells.append(f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]")
        lines.append(f"| {y} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
