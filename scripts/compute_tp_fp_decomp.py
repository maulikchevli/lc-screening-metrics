#!/usr/bin/env python3
"""Self-contained TP/FP decomposition at LR-anchored p_t for §3.4.

Avoids the project's heavy import chain (no torch). Reimplements:
  - _include(t, e, fu): year-y inclusion mask + label
  - _logit
  - Platt scaling (sklearn LogisticRegression) on dev fold
  - per-model deployment-form y1 scores
  - TP/FP at each LR-anchored p_t for each model + LR rule

Output: eval_output/tp_fp_decomp_y1.csv
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bundle root
OUTPUT_DIR = os.environ.get(
    "SCREENING_METRICS_OUTPUT_DIR", os.path.join(ROOT, "eval_output"))
LUNG_DEV_JSON = os.environ.get(
    "LUNGEVATY_DEV_JSON",
    os.path.join(ROOT, "data", "predictions_lungevaty_dev.json"))
PILLAR_USER_DEV_CSV = os.path.join(ROOT, "data", "predictions_pillar_dev.csv")
PILLAR_USER_TEST_CSV = os.path.join(ROOT, "data", "predictions_pillar_test.csv")

YEAR = 1
P_TS = [0.01, 0.02, 0.05, 0.10, 0.15]


def _logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _include(t, e, fu, fup_lower_bound=-1):
    """Canonical _include from evaluate_survival.py.
    `fu` is 0-indexed followup year (0 -> 1-year horizon).
    """
    if pd.isna(t):
        return False, 0
    valid_pos = bool(e) and (t <= fu) and (t > fup_lower_bound)
    valid_neg = t >= fu
    return (valid_pos or valid_neg), int(valid_pos)


def fit_platt(scores, labels):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    keep = ~np.isnan(s)
    s = s[keep]; y = y[keep]
    clf = LogisticRegression(C=1e6, solver="lbfgs")
    clf.fit(_logit(s).reshape(-1, 1), y)
    return clf


def apply_platt(clf, scores):
    s = np.asarray(scores, dtype=float)
    out = np.full_like(s, np.nan, dtype=float)
    mask = ~np.isnan(s)
    out[mask] = clf.predict_proba(_logit(s[mask]).reshape(-1, 1))[:, 1]
    return out


def fit_lung_platt_y1():
    with open(LUNG_DEV_JSON) as f:
        dev = json.load(f)
    s, y = [], []
    fu = YEAR - 1
    for r in dev:
        inc, lbl = _include(r["time_at_event"], r["y"], fu)
        if inc:
            s.append(float(r["cancer_risk"][YEAR - 1]))
            y.append(lbl)
    return fit_platt(s, y)


def fit_pillar_platt_y1():
    dev = pd.read_csv(PILLAR_USER_DEV_CSV)
    s, y = [], []
    fu = YEAR - 1
    for _, r in dev.iterrows():
        inc, lbl = _include(r["time_at_event"], r["y"], fu)
        if inc:
            s.append(float(r[f"prob_y{YEAR}"]))
            y.append(lbl)
    return fit_platt(s, y)


def attach_pillar_test(comp):
    user = pd.read_csv(PILLAR_USER_TEST_CSV)
    user["series_uid"] = user["series_uid"].astype(str)
    user["pid_tp"] = user["patient_id"].astype(str) + "_" + user["timepoint"].astype(str)
    u1 = user.drop_duplicates("series_uid")[["series_uid", f"prob_y{YEAR}"]].rename(
        columns={"series_uid": "series", f"prob_y{YEAR}": "pillar_user_y1"})
    comp = comp.copy()
    comp["series"] = comp["series"].astype(str)
    comp["pid_tp"] = comp["pid"].astype(str) + "_" + comp["screen_timepoint"].astype(int).astype(str)
    out = comp.merge(u1, on="series", how="left")
    miss = out["pillar_user_y1"].isna()
    if miss.any():
        u2 = user.drop_duplicates("pid_tp")[["pid_tp", f"prob_y{YEAR}"]].rename(
            columns={f"prob_y{YEAR}": "pillar_user_y1_fb"})
        out = out.merge(u2, on="pid_tp", how="left")
        out.loc[miss, "pillar_user_y1"] = out.loc[miss, "pillar_user_y1_fb"]
    return out


def main():
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    comp = attach_pillar_test(comp)

    # Attach Lung-RADS v1.0 categories (Pinsky 2015 reproduction)
    lr_csv = os.path.join(OUTPUT_DIR, "nlst_lungrads_pinsky.csv")
    lr_df = pd.read_csv(lr_csv, dtype={"pid": str, "lungrads_v10": str})
    lr_df = lr_df[["pid", "study_yr", "lungrads_v10"]].drop_duplicates(
        ["pid", "study_yr"])
    comp = comp.merge(lr_df, left_on=["pid", "screen_timepoint"],
                      right_on=["pid", "study_yr"], how="left")

    # Build year-1 inclusion mask + labels
    fu = YEAR - 1
    incs, lbls = [], []
    for t, e in zip(comp["time_at_event"], comp["gold"]):
        inc, lbl = _include(t, e, fu)
        incs.append(inc); lbls.append(lbl)
    incs = np.array(incs); lbls = np.array(lbls)

    # Fit Platt
    lung_clf = fit_lung_platt_y1()
    pillar_clf = fit_pillar_platt_y1()

    # Deployment-form y1 scores
    lung_y1 = apply_platt(lung_clf, comp["risk_score"].astype(float).values)
    pil_y1 = apply_platt(pillar_clf, comp["pillar_user_y1"].astype(float).values)
    sybil_y1 = comp["sv1c_risk_1"].astype(float).values  # author calibrator

    # Filter
    keep = incs & ~np.isnan(lung_y1) & ~np.isnan(pil_y1) & ~np.isnan(sybil_y1)
    n_total = int(keep.sum())
    y = lbls[keep].astype(int)
    n_ca = int(y.sum())
    print(f"Year-1 evaluable: N={n_total}, cancers={n_ca}")

    rows = []
    for label, scores in [("LungEvaty", lung_y1), ("Pillar-0", pil_y1), ("Sybil", sybil_y1)]:
        s = scores[keep]
        for pt in P_TS:
            flag = s > pt
            tp = int((flag & (y == 1)).sum())
            fp = int((flag & (y == 0)).sum())
            n = n_total
            flag_rate = (tp + fp) / n
            sens = tp / n_ca if n_ca > 0 else float("nan")
            prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            w = pt / (1 - pt)
            nb = tp / n - (fp / n) * w
            rows.append({
                "model": label, "p_t": pt, "N": n, "n_ca": n_ca,
                "TP": tp, "FP": fp,
                "flag_rate_pct": round(100 * flag_rate, 2),
                "sens_pct": round(100 * sens, 1),
                "prec_pct": round(100 * prec, 1),
                "NB_x1000": round(nb * 1000, 2),
                "FP_per_TP": round(1 / w, 1),
            })

    # Lung-RADS rule per p_t
    if "lungrads_v10" in comp.columns:
        lr = comp["lungrads_v10"].astype(str).values
        for pt, rule_cats in [(0.01, ["3", "4A", "4B", "4X"]),
                               (0.02, ["3", "4A", "4B", "4X"]),
                               (0.05, ["4A", "4B", "4X"]),
                               (0.10, ["4A", "4B", "4X"]),
                               (0.15, ["4B", "4X"])]:
            in_rule = np.isin(lr, rule_cats)
            flag = in_rule & keep
            tp = int((flag & (lbls == 1)).sum())
            fp = int((flag & (lbls == 0)).sum())
            n = n_total
            flag_rate = (tp + fp) / n
            sens = tp / n_ca if n_ca > 0 else float("nan")
            prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            w = pt / (1 - pt)
            nb = tp / n - (fp / n) * w
            rows.append({
                "model": f"LR_rule_{'+'.join(rule_cats)}", "p_t": pt, "N": n, "n_ca": n_ca,
                "TP": tp, "FP": fp,
                "flag_rate_pct": round(100 * flag_rate, 2),
                "sens_pct": round(100 * sens, 1),
                "prec_pct": round(100 * prec, 1),
                "NB_x1000": round(nb * 1000, 2),
                "FP_per_TP": round(1 / w, 1),
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "tp_fp_decomp_y1.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
