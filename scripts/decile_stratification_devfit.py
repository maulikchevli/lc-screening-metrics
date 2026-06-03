#!/usr/bin/env python3
"""DEPLOYMENT-GRADE scan-level DECILE analysis with dev-fit thresholds.

Methodology
-----------
For each model, decile cut points (10/20/.../90 percentile of year-1 score)
are fit on the held-out NLST development fold and frozen. They are then
applied to the matched test fold (n=6,214 scans) for evaluation. This is
the deployment-faithful operating-point selection.

5-category aggregation (Option B: selective top):
  C1 (workup, fr~10%)        = D1
  C2 (3-month CT, fr~10%)    = D2
  C3 (6-month CT, fr~20%)    = D3 + D4
  C4 (12-month CT, fr~30%)   = D5 + D6 + D7
  C5 (24-month CT, fr~30%)   = D8 + D9 + D10

Pillar-0 exception
------------------
Pillar-0 dev predictions are not available; deciles are fit on test for
that model only. Acknowledged as a known limitation tied to data access.

Models
------
- LungEvaty raw   : raw dev cuts on raw test
- LungEvaty Platt : cal dev cuts on cal test (Platt fit on dev raw)
- Sybil raw       : cal dev cuts on cal test (rank-equivalent to raw/raw
                    by monotone Platt). p_t-anchored NB rows are not
                    meaningful for the raw cell — reported with caveat.
- Sybil cal       : cal dev cuts on cal test (author-released calibrator)
- PLCOm raw       : raw dev cuts on raw test (PLCOm computed from features)
- PLCOm Platt     : cal dev cuts on cal test (Platt fit on dev PLCOm raw)
- Pillar-0        : TEST-FIT cuts on test scores (no dev predictions)

Outputs
-------
  eval_output/decile_incidence_devfit.csv
  eval_output/decile_calibration_devfit.csv
  eval_output/nb_at_decile_thresholds_devfit.csv
  eval_output/subgroup_year1_devfit.csv
  eval_output/decile_stratification_devfit.log
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import (
    _include, _ipcw_weights, load_participant,
    PARTICIPANT_CSV, OUTPUT_DIR,
)
from uspstf_plcom import plcom2012_risk, NLST_PERSONAL_CANCER_COLS

LUNG_DEV_JSON = os.environ.get("LUNGEVATY_DEV_JSON", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions_lungevaty_dev.json"))
SYBIL_CAL_DEV_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions_sybil_v1_dev.json")
PILLAR_USER_TEST_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions_pillar_test.csv")
PILLAR_USER_DEV_CSV  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions_pillar_dev.csv")

# Mode toggle: "devfit" (default) freezes cuts on dev and applies to test.
# "testfit" derives cuts from test scores themselves (own-distribution).
# All output filenames carry the matching suffix.
MODE = os.environ.get("DECILE_MODE", "devfit")
assert MODE in ("devfit", "testfit"), f"Invalid DECILE_MODE: {MODE}"
SUFFIX = f"_{MODE}"

EPS = 1e-7
RNG_SEED = 42
N_BOOT_INC = 1000
N_BOOT_NB = 500
N_DECILES = 10
YEARS = (1, 3, 6)
# All 9 decile cumulative cuts (10/20/.../90 %) + 4 cat-aggregated
# boundaries (10/20/40/70 %). The cat boundaries are a strict subset.
DECILE_FLAG_RATES = tuple(round(0.1 * k, 2) for k in range(1, 10))
CAT_BOUNDARIES = (0.10, 0.20, 0.40, 0.70)
CAT_BOUNDARY_LABEL = {0.10: "C1", 0.20: "C1+C2",
                      0.40: "C1+C2+C3", 0.70: "C1+C2+C3+C4"}
PT_THRESHOLDS = (0.01, 0.05, 0.10)
# Decile → category mapping
DECILE_TO_CAT = {1: 1, 2: 2, 3: 3, 4: 3,
                 5: 4, 6: 4, 7: 4,
                 8: 5, 9: 5, 10: 5}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _ci(arr):
    arr = np.array([v for v in arr
                    if v is not None
                    and not (isinstance(v, float) and np.isnan(v))])
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)))


def _resample_idx(rng, pids, pid_to_rows):
    unique = np.array(sorted(set(pids.tolist())))
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([pid_to_rows[p] for p in sampled])


def _plcom_extra_cols(part):
    cols = ["height", "weight", "educat", "age_quit", "ethnic",
            "diagcopd", "diagemph", "diagchro",
            "famfather", "fammother", "fambrother", "famsister", "famchild"]
    cols += list(NLST_PERSONAL_CANCER_COLS)
    return [c for c in cols if c in part.columns]


# ---------------------------------------------------------------------------
# Decile assignment from frozen cut points
# ---------------------------------------------------------------------------
def decile_cut_points(scores):
    """Return 9 percentile cut points (10/20/.../90) of `scores`,
    high → low: cut[k] = 100*(1 - (k+1)/10) percentile, so:
      cut[0] = 90th pctile (top 10% boundary, separates D1 from rest)
      cut[1] = 80th pctile (separates D1+D2 from rest)
      ...
      cut[8] = 10th pctile (separates D1..D9 from D10)
    """
    return [float(np.percentile(scores, 100 * (1 - fr)))
            for fr in DECILE_FLAG_RATES]


def assign_decile_from_cuts(test_scores, cuts):
    """Apply 9 frozen high-to-low cut points → decile label 1..10.
    D1 = top 10% on dev (score ≥ cuts[0]); D10 = bottom 10%.
    """
    d = np.full(len(test_scores), N_DECILES, dtype=int)
    # Walk from least-restrictive cut (10%ile) up to most-restrictive (90%ile).
    # cuts ordered high→low; for D_k the test score must be ≥ cuts[k-1].
    for k in range(N_DECILES, 1, -1):
        d[test_scores >= cuts[k - 2]] = k - 1
    return d


def assign_decile_test_fit(scores):
    """Test-fit fallback for Pillar-0: rank-based 10-way equal split."""
    n = len(scores)
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    d = np.ceil((rank + 1) / (n / N_DECILES)).astype(int)
    return np.clip(d, 1, N_DECILES)


def decile_to_cat(d):
    """Map a decile array (1..10) to category 1..5."""
    return np.array([DECILE_TO_CAT[int(x)] for x in d], dtype=int)


# ---------------------------------------------------------------------------
# Cohort helpers (same convention as quintile script)
# ---------------------------------------------------------------------------
def cohort_for(comp, year, scores_for_eval=None):
    fu = year - 1
    times = comp["time_at_event"].values.astype(float)
    events = comp["gold"].values.astype(int)
    if year == 6:
        mask = np.array([_include(t, e, fu)[0] for t, e in zip(times, events)])
        labels = np.array([_include(t, e, fu)[1] for t, e in zip(times, events)])
        if scores_for_eval is not None:
            mask = mask & ~np.isnan(scores_for_eval)
        return dict(mode="cc", fu=fu, mask=mask, labels=labels)
    mask = (np.ones(len(times), dtype=bool)
            if scores_for_eval is None
            else ~np.isnan(scores_for_eval))
    return dict(mode="ipcw", fu=fu, mask=mask, labels=None)


def bin_incidence(bins, n_bins, times, events, fu, mode):
    """Generic: per-bin observed incidence + top-bin capture."""
    out = {}
    if mode == "ipcw":
        tp_w, _, is_tp, _, _ = _ipcw_weights(times, events, fu)
        total_tp_mass = float(tp_w[is_tp].sum())
        for b in range(1, n_bins + 1):
            mask = bins == b
            if not mask.any():
                out[b] = (np.nan, 0); continue
            inc = float(tp_w[mask & is_tp].sum()) / mask.sum()
            out[b] = (inc, int(mask.sum()))
        capture = (float(tp_w[(bins == 1) & is_tp].sum()) / total_tp_mass
                   if total_tp_mass > 0 else np.nan)
    else:
        labels = events.astype(int)
        total_pos = int(labels.sum())
        for b in range(1, n_bins + 1):
            mask = bins == b
            if not mask.any():
                out[b] = (np.nan, 0); continue
            inc = float(labels[mask].mean())
            out[b] = (inc, int(mask.sum()))
        capture = (int(labels[bins == 1].sum()) / total_pos
                   if total_pos > 0 else np.nan)
    return out, capture


def _ipcw_nb(scores, times, events, fu, threshold):
    tp_w, fp_w, is_tp, is_fp, prev = _ipcw_weights(times, events, fu)
    n = len(times)
    pos = scores >= threshold
    tp_r = float(tp_w[pos & is_tp].sum()) / n
    fp_r = float((pos & is_fp).sum()) * fp_w / n
    w = threshold / (1 - threshold) if 0 < threshold < 1 else 0.0
    nb = tp_r - fp_r * w
    nb_all = prev - (1 - prev) * w
    return nb, nb_all


def _cc_nb(scores, labels, threshold):
    n = len(scores)
    pos = scores >= threshold
    tp = int((pos & (labels == 1)).sum())
    fp = int((pos & (labels == 0)).sum())
    w = threshold / (1 - threshold) if 0 < threshold < 1 else 0.0
    prev = labels.mean()
    return tp / n - (fp / n) * w, prev - (1 - prev) * w


# ---------------------------------------------------------------------------
# Dev fitters
# ---------------------------------------------------------------------------
def fit_lung_platt():
    with open(LUNG_DEV_JSON) as f:
        dev = json.load(f)
    out = {}
    for year in YEARS:
        fu = year - 1
        s, y = [], []
        for r in dev:
            inc, lbl = _include(r["time_at_event"], r["y"], fu)
            if inc:
                s.append(float(r["cancer_risk"][year - 1]))
                y.append(lbl)
        s = np.array(s); y = np.array(y)
        clf = LogisticRegression(C=1e6, solver="lbfgs")
        clf.fit(_logit(s).reshape(-1, 1), y)
        out[year] = clf
    return out


def fit_plcom_platt(dev_df_plcom):
    """Fit per-horizon Platt on PLCOm dev scores (already attached)."""
    out = {}
    for year in YEARS:
        fu = year - 1
        sl, yl = [], []
        col = f"plcom_y{year}"
        for _, r in dev_df_plcom.iterrows():
            inc, lbl = _include(r["time_at_event"], r["gold"], fu)
            v = r[col]
            if inc and not pd.isna(v):
                sl.append(float(v)); yl.append(int(lbl))
        s = np.array(sl); y = np.array(yl)
        clf = LogisticRegression(C=1e6, solver="lbfgs")
        clf.fit(_logit(s).reshape(-1, 1), y)
        out[year] = clf
    return out


def attach_user_pillar(comp):
    """Two-strategy match (series_uid → pid_tp fallback) of user-reproduced
    Pillar-0 test predictions to comp_df. Adds `pillar_user_y{1..6}` cols.
    """
    user = pd.read_csv(PILLAR_USER_TEST_CSV)
    user["series_uid"] = user["series_uid"].astype(str)
    user["patient_id"] = user["patient_id"].astype(str)
    user["timepoint"] = user["timepoint"].astype(int)
    user["pid_tp"] = user["patient_id"] + "_" + user["timepoint"].astype(str)
    cols = [f"prob_y{y}" for y in range(1, 7)]
    rename = {f"prob_y{y}": f"pillar_user_y{y}" for y in range(1, 7)}

    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    comp["screen_timepoint"] = comp["screen_timepoint"].astype(int)
    comp["pid_tp"] = comp["pid"] + "_" + comp["screen_timepoint"].astype(str)

    # Strategy 1: by series_uid
    u1 = user.drop_duplicates("series_uid").rename(
        columns={"series_uid": "series", **rename})[
        ["series"] + [rename[c] for c in cols]]
    out = comp.merge(u1, on="series", how="left")

    # Strategy 2: pid_tp fallback for rows still missing
    miss = out[f"pillar_user_y1"].isna()
    u2 = user.drop_duplicates("pid_tp").rename(columns=rename)[
        ["pid_tp"] + [rename[c] for c in cols]]
    fb = out.loc[miss, ["pid_tp"]].merge(u2, on="pid_tp", how="left")
    for c in cols:
        out.loc[miss, rename[c]] = fb[rename[c]].values
    return out


def fit_pillar_platt():
    """Per-horizon Platt scaling on user-reproduced Pillar-0 dev predictions."""
    dev = pd.read_csv(PILLAR_USER_DEV_CSV)
    out = {}
    for year in YEARS:
        fu = year - 1
        sl, yl = [], []
        for _, r in dev.iterrows():
            inc, lbl = _include(r["time_at_event"], r["y"], fu)
            if inc:
                sl.append(float(r[f"prob_y{year}"]))
                yl.append(int(lbl))
        s = np.array(sl); y = np.array(yl)
        clf = LogisticRegression(C=1e6, solver="lbfgs")
        clf.fit(_logit(s).reshape(-1, 1), y)
        out[year] = clf
    return out


def build_pillar_dev_y1(pillar_platt):
    dev = pd.read_csv(PILLAR_USER_DEV_CSV)
    raw_y1 = dev["prob_y1"].values.astype(float)
    cal_y1 = pillar_platt[1].predict_proba(
        _logit(raw_y1).reshape(-1, 1))[:, 1]
    return raw_y1, cal_y1


def build_lung_dev_y1(lung_platt):
    with open(LUNG_DEV_JSON) as f:
        dev = json.load(f)
    raw_y1 = np.array([float(r["cancer_risk"][0]) for r in dev])
    cal_y1 = lung_platt[1].predict_proba(
        _logit(raw_y1).reshape(-1, 1))[:, 1]
    return raw_y1, cal_y1


def build_sybil_cal_dev_y1():
    with open(SYBIL_CAL_DEV_JSON) as f:
        dev = json.load(f)
    return np.array([float(r["cancer_risk"][0]) for r in dev])


def build_plcom_dev(part):
    """Compute PLCOm on the dev fold (year-1..6) by merging dev pids
    with NLST participant features."""
    with open(LUNG_DEV_JSON) as f:
        dev = json.load(f)
    rows = [dict(pid=str(r["pid"]),
                 time_at_event=float(r["time_at_event"]),
                 gold=int(r["y"]))
            for r in dev]
    dev_df = pd.DataFrame(rows)
    extras = _plcom_extra_cols(part)
    base_cols = ["pid", "age", "race", "pkyr", "smokeyr", "cigsmok"]
    base_cols = [c for c in base_cols if c in part.columns]
    base = part[base_cols + extras].drop_duplicates("pid").copy()
    base["pid"] = base["pid"].astype(str)
    dev_df = dev_df.merge(base, on="pid", how="left")
    for y in YEARS:
        dev_df[f"plcom_y{y}"] = dev_df.apply(
            lambda r, _y=y: plcom2012_risk(r, year=_y), axis=1)
    return dev_df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print(f"DEPLOYMENT-GRADE scan-level DECILE analysis (MODE={MODE})")
    print("=" * 78)

    print("\n[setup] Loading test comp …")
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)

    print("\n[setup] Attaching PLCOm to test cohort …")
    part = load_participant(PARTICIPANT_CSV)
    part["pid"] = part["pid"].astype(str)
    extras = _plcom_extra_cols(part)
    ext = part[["pid"] + extras].drop_duplicates("pid").copy()
    ext["pid"] = ext["pid"].astype(str)
    comp = comp.merge(ext, on="pid", how="left")
    for y in YEARS:
        comp[f"plcom_y{y}"] = comp.apply(
            lambda r, _y=y: plcom2012_risk(r, year=_y), axis=1)

    print("\n[setup] Fitting LungEvaty Platt on dev …")
    lung_platt = fit_lung_platt()
    for y in YEARS:
        comp[f"lungevity_platt_y{y}"] = lung_platt[y].predict_proba(
            _logit(comp["risk_score"].values).reshape(-1, 1))[:, 1]

    print("[setup] Computing PLCOm on dev + fitting per-horizon Platt …")
    dev_plcom = build_plcom_dev(part)
    plcom_platt = fit_plcom_platt(dev_plcom)
    for y in YEARS:
        vals = comp[f"plcom_y{y}"].values.astype(float)
        mask = ~np.isnan(vals)
        out = np.full_like(vals, np.nan, dtype=float)
        out[mask] = plcom_platt[y].predict_proba(
            _logit(vals[mask]).reshape(-1, 1))[:, 1]
        comp[f"plcom_platt_y{y}"] = out

    print("[setup] Attaching user-reproduced Pillar-0 to test cohort …")
    comp = attach_user_pillar(comp)
    n_pillar_matched = int((~comp["pillar_user_y1"].isna()).sum())
    print(f"  user-Pillar-0 matched: {n_pillar_matched}/{len(comp)} scans")

    print("[setup] Fitting Pillar-0 Platt on user dev …")
    pillar_platt = fit_pillar_platt()
    for y in YEARS:
        vals = comp[f"pillar_user_y{y}"].values.astype(float)
        mask = ~np.isnan(vals)
        out_p = np.full_like(vals, np.nan, dtype=float)
        out_p[mask] = pillar_platt[y].predict_proba(
            _logit(vals[mask]).reshape(-1, 1))[:, 1]
        comp[f"pillar_platt_y{y}"] = out_p

    n_scan = len(comp); n_pat = comp["pid"].nunique()
    n_total_ca = int(comp["gold"].sum())
    print(f"\n  Test: N scans={n_scan}, patients={n_pat}, "
          f"total cancers={n_total_ca}")

    # Year-1 cancer count sanity check
    fu1 = 0
    y1_include = np.array([_include(t, e, fu1)[0]
                            for t, e in zip(comp["time_at_event"].values,
                                            comp["gold"].values)])
    y1_lbl = np.array([_include(t, e, fu1)[1]
                       for t, e in zip(comp["time_at_event"].values,
                                       comp["gold"].values)])
    print(f"  Year-1 includable scans = {int(y1_include.sum())}, "
          f"year-1 cancers = {int(y1_lbl[y1_include].sum())}")

    # ---- Build dev y1 score arrays ----------------------------------------
    lung_dev_raw_y1, lung_dev_cal_y1 = build_lung_dev_y1(lung_platt)
    syb_cal_dev_y1 = build_sybil_cal_dev_y1()
    plcom_dev_raw_y1 = dev_plcom["plcom_y1"].values.astype(float)
    plcom_dev_raw_y1 = plcom_dev_raw_y1[~np.isnan(plcom_dev_raw_y1)]
    plcom_dev_cal_y1_all = plcom_platt[1].predict_proba(
        _logit(plcom_dev_raw_y1).reshape(-1, 1))[:, 1]
    pillar_dev_raw_y1, pillar_dev_cal_y1 = build_pillar_dev_y1(pillar_platt)

    cuts = {
        ("LungEvaty", "raw"):   decile_cut_points(lung_dev_raw_y1),
        ("LungEvaty", "platt"): decile_cut_points(lung_dev_cal_y1),
        # Sybil raw uses cal dev cuts via rank invariance (same labels)
        ("Sybil",     "raw"):   decile_cut_points(syb_cal_dev_y1),
        ("Sybil",     "cal"):   decile_cut_points(syb_cal_dev_y1),
        ("PLCOm",     "raw"):   decile_cut_points(plcom_dev_raw_y1),
        ("PLCOm",     "platt"): decile_cut_points(plcom_dev_cal_y1_all),
        ("Pillar-0",  "raw"):   decile_cut_points(pillar_dev_raw_y1),
        ("Pillar-0",  "platt"): decile_cut_points(pillar_dev_cal_y1),
    }

    print("\n[setup] Dev-fit decile thresholds (high → low; D1 boundary first):")
    for k, c in cuts.items():
        print(f"  {k[0]:<12}{k[1]:<6}: "
              + " ".join(f"{v:.5f}" for v in c))

    # ---- Variant registry --------------------------------------------------
    # For each cell: (eval_score_fn, y1_assign_fn, decile_label_fn, source)
    def y1_lung_raw(): return comp["risk_score"].values.astype(float)
    def y1_lung_cal(): return comp["lungevity_platt_y1"].values.astype(float)
    def y1_syb_raw():  return comp["sv1r_risk_1"].values.astype(float)
    def y1_syb_cal():  return comp["sv1c_risk_1"].values.astype(float)
    def y1_plcom_raw():return comp["plcom_y1"].values.astype(float)
    def y1_plcom_cal():return comp["plcom_platt_y1"].values.astype(float)
    def y1_pillar_raw():   return comp["pillar_user_y1"].values.astype(float)
    def y1_pillar_platt(): return comp["pillar_platt_y1"].values.astype(float)

    def eval_score(template):
        def _f(y): return comp[template.format(y=y)].values.astype(float)
        return _f

    # For the Sybil raw cell, decile labels are derived from cal dev cuts
    # applied to cal test scores (rank-equivalent). Evaluation columns
    # are the raw test scores. Q2 mean-predicted will be on raw scale.
    variants = [
        dict(model="LungEvaty", cal="raw",
             y1_for_assign=y1_lung_raw,
             cuts=cuts[("LungEvaty", "raw")],
             eval_tmpl="risk_score",
             source="dev-fit"),
        dict(model="LungEvaty", cal="platt",
             y1_for_assign=y1_lung_cal,
             cuts=cuts[("LungEvaty", "platt")],
             eval_tmpl="lungevity_platt_y{y}",
             source="dev-fit"),
        dict(model="Sybil", cal="raw",
             y1_for_assign=y1_syb_cal,  # cal scores for assignment
             cuts=cuts[("Sybil", "cal")],
             eval_tmpl="sv1r_risk_{y}",
             source="dev-fit-via-cal-rank",
             note="p_t-anchored NB not meaningful for raw scores"),
        dict(model="Sybil", cal="cal",
             y1_for_assign=y1_syb_cal,
             cuts=cuts[("Sybil", "cal")],
             eval_tmpl="sv1c_risk_{y}",
             source="dev-fit"),
        dict(model="PLCOm", cal="raw",
             y1_for_assign=y1_plcom_raw,
             cuts=cuts[("PLCOm", "raw")],
             eval_tmpl="plcom_y{y}",
             source="dev-fit"),
        dict(model="PLCOm", cal="platt",
             y1_for_assign=y1_plcom_cal,
             cuts=cuts[("PLCOm", "platt")],
             eval_tmpl="plcom_platt_y{y}",
             source="dev-fit"),
        dict(model="Pillar-0", cal="raw",
             y1_for_assign=y1_pillar_raw,
             cuts=cuts[("Pillar-0", "raw")],
             eval_tmpl="pillar_user_y{y}",
             source="dev-fit"),
        dict(model="Pillar-0", cal="platt",
             y1_for_assign=y1_pillar_platt,
             cuts=cuts[("Pillar-0", "platt")],
             eval_tmpl="pillar_platt_y{y}",
             source="dev-fit"),
    ]

    # In test-fit mode, all variants use own-distribution cuts. Also redirect
    # Sybil's raw-cell assignment to the actual raw scores (the dev-fit rank-
    # invariance trick is unnecessary when we have raw test scores directly).
    if MODE == "testfit":
        for v in variants:
            v["cuts"] = None
            if v["model"] == "Sybil" and v["cal"] == "raw":
                v["y1_for_assign"] = y1_syb_raw
            v["source"] = "test-fit"

    # ========================================================================
    # ANALYSIS Q1 — per-decile incidence + 5-cat aggregation
    # ========================================================================
    print("\n" + "=" * 78)
    print("Q1 — per-decile incidence + 5-category aggregation")
    print("=" * 78)
    q1_rows = []
    for v in variants:
        y1s = v["y1_for_assign"]()
        valid_y1 = ~np.isnan(y1s)
        if v["cuts"] is None:
            # test-fit fallback
            d_local = assign_decile_test_fit(y1s[valid_y1])
        else:
            d_local = assign_decile_from_cuts(y1s[valid_y1], v["cuts"])
        full_d = np.zeros(len(comp), dtype=int)
        full_d[valid_y1] = d_local
        full_c = np.zeros(len(comp), dtype=int)
        full_c[valid_y1] = decile_to_cat(d_local)

        d_sizes = [int((full_d == d).sum()) for d in range(1, 11)]
        d_fracs = [s / valid_y1.sum() for s in d_sizes]
        c_sizes = [int((full_c == c).sum()) for c in range(1, 6)]
        print(f"\n  {v['model']:<12}{v['cal']:<6} "
              f"[{v['source']}] decile sizes: {d_sizes}")
        print(f"    decile fractions:     "
              f"{[f'{f:.3f}' for f in d_fracs]}")
        print(f"    category sizes:       {c_sizes}")

        for eval_year in YEARS:
            score_eval = eval_score(v["eval_tmpl"])(eval_year)
            data = cohort_for(comp, eval_year, scores_for_eval=score_eval)
            keep = data["mask"] & valid_y1
            d_in = full_d[keep]; c_in = full_c[keep]
            pids_in = comp["pid"].values[keep]

            if eval_year == 6:
                events_in = data["labels"][keep].astype(int)
                times_in = None
                mode = "cc"
            else:
                times_in = comp["time_at_event"].values[keep].astype(float)
                events_in = comp["gold"].values[keep].astype(int)
                mode = "ipcw"

            d_inc, d_cap = bin_incidence(d_in, N_DECILES,
                                          times_in, events_in,
                                          data["fu"], mode)
            c_inc, c_cap = bin_incidence(c_in, 5,
                                          times_in, events_in,
                                          data["fu"], mode)

            # Patient-clustered bootstrap
            rng = np.random.default_rng(
                RNG_SEED + eval_year * 1000
                + sum(ord(c) for c in v["model"])
                + (0 if v["cal"] == "raw" else 1))
            unique_pids = np.array(sorted(set(pids_in.tolist())))
            pid_to_rows = {p: np.where(pids_in == p)[0]
                           for p in unique_pids}
            boot_d = {d: [] for d in range(1, 11)}
            boot_c = {c: [] for c in range(1, 6)}
            boot_d_cap = []; boot_c_cap = []
            for _ in range(N_BOOT_INC):
                idx = _resample_idx(rng, pids_in, pid_to_rows)
                db = d_in[idx]; cb = c_in[idx]
                if mode == "ipcw":
                    try:
                        tb = times_in[idx]; eb = events_in[idx]
                        if eb.sum() == 0:
                            continue
                        bd_inc, bd_cap = bin_incidence(db, 10, tb, eb,
                                                        data["fu"], "ipcw")
                        bc_inc, bc_cap = bin_incidence(cb, 5, tb, eb,
                                                        data["fu"], "ipcw")
                    except Exception:
                        continue
                else:
                    eb = events_in[idx]
                    if eb.sum() == 0:
                        continue
                    bd_inc, bd_cap = bin_incidence(db, 10, None, eb,
                                                    data["fu"], "cc")
                    bc_inc, bc_cap = bin_incidence(cb, 5, None, eb,
                                                    data["fu"], "cc")
                for d in range(1, 11):
                    val = bd_inc[d][0]
                    if not np.isnan(val):
                        boot_d[d].append(val)
                for c in range(1, 6):
                    val = bc_inc[c][0]
                    if not np.isnan(val):
                        boot_c[c].append(val)
                if not np.isnan(bd_cap): boot_d_cap.append(bd_cap)
                if not np.isnan(bc_cap): boot_c_cap.append(bc_cap)

            # Emit decile rows
            for d in range(1, 11):
                inc, n_d = d_inc[d]
                lo, hi = _ci(boot_d[d])
                cap_lo, cap_hi = _ci(boot_d_cap)
                q1_rows.append(dict(
                    model=v["model"], calibration=v["cal"],
                    source=v["source"],
                    eval_year=eval_year, grain="decile", bin=d,
                    obs_incidence=inc, ci_low=lo, ci_high=hi,
                    n_scans=n_d,
                    top_capture=d_cap if d == 1 else np.nan,
                    capture_ci_low=cap_lo if d == 1 else np.nan,
                    capture_ci_high=cap_hi if d == 1 else np.nan,
                ))
            # Emit category rows
            for c in range(1, 6):
                inc, n_c = c_inc[c]
                lo, hi = _ci(boot_c[c])
                cap_lo, cap_hi = _ci(boot_c_cap)
                q1_rows.append(dict(
                    model=v["model"], calibration=v["cal"],
                    source=v["source"],
                    eval_year=eval_year, grain="cat", bin=c,
                    obs_incidence=inc, ci_low=lo, ci_high=hi,
                    n_scans=n_c,
                    top_capture=c_cap if c == 1 else np.nan,
                    capture_ci_low=cap_lo if c == 1 else np.nan,
                    capture_ci_high=cap_hi if c == 1 else np.nan,
                ))

            print(f"    eval_y{eval_year}: D1={d_inc[1][0]:.4f} "
                  f"D5={d_inc[5][0]:.4f} D10={d_inc[10][0]:.4f}  "
                  f"D-cap={d_cap:.3f}  C-cap(C1)={c_cap:.3f}")

    q1_df = pd.DataFrame(q1_rows)
    out_q1 = os.path.join(OUTPUT_DIR, f"decile_incidence{SUFFIX}.csv")
    q1_df.to_csv(out_q1, index=False)
    print(f"\nSaved {out_q1} ({len(q1_df)} rows)")

    # ========================================================================
    # ANALYSIS Q2 — per-decile calibration (mean predicted vs mean observed)
    # ========================================================================
    print("\n" + "=" * 78)
    print("Q2 — per-decile + per-category calibration")
    print("=" * 78)
    q2_rows = []
    for v in variants:
        y1s = v["y1_for_assign"]()
        valid_y1 = ~np.isnan(y1s)
        if v["cuts"] is None:
            d_local = assign_decile_test_fit(y1s[valid_y1])
        else:
            d_local = assign_decile_from_cuts(y1s[valid_y1], v["cuts"])
        full_d = np.zeros(len(comp), dtype=int)
        full_d[valid_y1] = d_local
        full_c = np.zeros(len(comp), dtype=int)
        full_c[valid_y1] = decile_to_cat(d_local)

        for eval_year in YEARS:
            score_eval = eval_score(v["eval_tmpl"])(eval_year)
            data = cohort_for(comp, eval_year, scores_for_eval=score_eval)
            keep = data["mask"] & valid_y1 & ~np.isnan(score_eval)
            d_in = full_d[keep]; c_in = full_c[keep]
            pred_in = score_eval[keep]
            pids_in = comp["pid"].values[keep]
            if eval_year == 6:
                events_in = data["labels"][keep].astype(int)
                times_in = None
                obs_d, _ = bin_incidence(d_in, 10, None, events_in,
                                          data["fu"], "cc")
                obs_c, _ = bin_incidence(c_in, 5, None, events_in,
                                          data["fu"], "cc")
                mode = "cc"
            else:
                times_in = comp["time_at_event"].values[keep].astype(float)
                events_in = comp["gold"].values[keep].astype(int)
                obs_d, _ = bin_incidence(d_in, 10, times_in, events_in,
                                          data["fu"], "ipcw")
                obs_c, _ = bin_incidence(c_in, 5, times_in, events_in,
                                          data["fu"], "ipcw")
                mode = "ipcw"

            mean_pred_d = {d: float(np.mean(pred_in[d_in == d]))
                           if (d_in == d).any() else np.nan
                           for d in range(1, 11)}
            mean_pred_c = {c: float(np.mean(pred_in[c_in == c]))
                           if (c_in == c).any() else np.nan
                           for c in range(1, 6)}

            rng = np.random.default_rng(
                RNG_SEED + 23000 + eval_year * 1000
                + sum(ord(c) for c in v["model"])
                + (0 if v["cal"] == "raw" else 1))
            unique_pids = np.array(sorted(set(pids_in.tolist())))
            pid_to_rows = {p: np.where(pids_in == p)[0]
                           for p in unique_pids}
            boot_diff_d = {d: [] for d in range(1, 11)}
            boot_diff_c = {c: [] for c in range(1, 6)}
            for _ in range(N_BOOT_INC):
                idx = _resample_idx(rng, pids_in, pid_to_rows)
                db = d_in[idx]; cb = c_in[idx]; pb = pred_in[idx]
                if mode == "ipcw":
                    try:
                        tb = times_in[idx]; eb = events_in[idx]
                        if eb.sum() == 0: continue
                        bod, _ = bin_incidence(db, 10, tb, eb,
                                                data["fu"], "ipcw")
                        boc, _ = bin_incidence(cb, 5, tb, eb,
                                                data["fu"], "ipcw")
                    except Exception:
                        continue
                else:
                    eb = events_in[idx]
                    if eb.sum() == 0: continue
                    bod, _ = bin_incidence(db, 10, None, eb,
                                            data["fu"], "cc")
                    boc, _ = bin_incidence(cb, 5, None, eb,
                                            data["fu"], "cc")
                for d in range(1, 11):
                    obsv = bod[d][0]
                    if (db == d).any():
                        pp = float(np.mean(pb[db == d]))
                        if not np.isnan(obsv):
                            boot_diff_d[d].append(obsv - pp)
                for c in range(1, 6):
                    obsv = boc[c][0]
                    if (cb == c).any():
                        pp = float(np.mean(pb[cb == c]))
                        if not np.isnan(obsv):
                            boot_diff_c[c].append(obsv - pp)

            for d in range(1, 11):
                p = mean_pred_d[d]; o = obs_d[d][0]
                diff = (o - p) if (not np.isnan(o) and not np.isnan(p)) else np.nan
                lo, hi = _ci(boot_diff_d[d])
                q2_rows.append(dict(
                    model=v["model"], calibration=v["cal"],
                    source=v["source"],
                    eval_year=eval_year, grain="decile", bin=d,
                    mean_predicted=p, mean_observed=o,
                    diff=diff, diff_lo=lo, diff_hi=hi,
                ))
            for c in range(1, 6):
                p = mean_pred_c[c]; o = obs_c[c][0]
                diff = (o - p) if (not np.isnan(o) and not np.isnan(p)) else np.nan
                lo, hi = _ci(boot_diff_c[c])
                q2_rows.append(dict(
                    model=v["model"], calibration=v["cal"],
                    source=v["source"],
                    eval_year=eval_year, grain="cat", bin=c,
                    mean_predicted=p, mean_observed=o,
                    diff=diff, diff_lo=lo, diff_hi=hi,
                ))
            print(f"  {v['model']:<12}{v['cal']:<6} eval_y{eval_year}: "
                  f"D1 pred={mean_pred_d[1]:.4f} obs={obs_d[1][0]:.4f}  "
                  f"D10 pred={mean_pred_d[10]:.4f} obs={obs_d[10][0]:.4f}")

    q2_df = pd.DataFrame(q2_rows)
    out_q2 = os.path.join(OUTPUT_DIR, f"decile_calibration{SUFFIX}.csv")
    q2_df.to_csv(out_q2, index=False)
    print(f"\nSaved {out_q2} ({len(q2_df)} rows)")

    # ========================================================================
    # ANALYSIS Q3 — NB at all 9 decile cumulative cuts + 4 cat boundaries
    # ========================================================================
    print("\n" + "=" * 78)
    print("Q3 — NB at decile-derived flag rates + cat boundaries + p_t")
    print("=" * 78)
    q3_rows = []
    for v in variants:
        if v["cuts"] is None:
            y1s = v["y1_for_assign"]()
            valid = ~np.isnan(y1s)
            local_cuts = decile_cut_points(y1s[valid])
        else:
            local_cuts = v["cuts"]
        fr_to_t = dict(zip(DECILE_FLAG_RATES, local_cuts))

        for eval_year in YEARS:
            score_eval = eval_score(v["eval_tmpl"])(eval_year)
            data = cohort_for(comp, eval_year, scores_for_eval=score_eval)
            keep = data["mask"] & ~np.isnan(v["y1_for_assign"]())
            s_in = score_eval[keep]
            pids_in = comp["pid"].values[keep]
            if eval_year == 6:
                events_in = data["labels"][keep].astype(int)
                times_in = None
                mode = "cc"
            else:
                times_in = comp["time_at_event"].values[keep].astype(float)
                events_in = comp["gold"].values[keep].astype(int)
                mode = "ipcw"

            anchors = []
            # all 9 decile cumulative cuts
            for fr in DECILE_FLAG_RATES:
                anchors.append(("decile_cum", fr, fr_to_t[fr],
                                CAT_BOUNDARY_LABEL.get(fr, "")))
            # p_t fixed thresholds
            for pt in PT_THRESHOLDS:
                anchors.append(("pt", pt, pt, ""))

            for anchor_type, anchor_val, t, cat_lbl in anchors:
                if mode == "ipcw":
                    nb_pt, nb_all = _ipcw_nb(s_in, times_in, events_in,
                                              data["fu"], t)
                else:
                    nb_pt, nb_all = _cc_nb(s_in, events_in, t)
                realised_fr = float((s_in >= t).mean())

                rng = np.random.default_rng(
                    RNG_SEED + 47000 + eval_year * 100
                    + sum(ord(c) for c in v["model"])
                    + int(anchor_val * 100))
                unique_pids = np.array(sorted(set(pids_in.tolist())))
                pid_to_rows = {p: np.where(pids_in == p)[0]
                               for p in unique_pids}
                nbs = []
                for _ in range(N_BOOT_NB):
                    idx = _resample_idx(rng, pids_in, pid_to_rows)
                    if mode == "ipcw":
                        try:
                            tb = times_in[idx]; eb = events_in[idx]
                            if eb.sum() == 0: continue
                            nb_b, _ = _ipcw_nb(s_in[idx], tb, eb,
                                                data["fu"], t)
                        except Exception:
                            continue
                    else:
                        eb = events_in[idx]
                        if eb.sum() == 0: continue
                        nb_b, _ = _cc_nb(s_in[idx], eb, t)
                    nbs.append(nb_b)
                lo, hi = _ci(nbs)
                q3_rows.append(dict(
                    model=v["model"], calibration=v["cal"],
                    source=v["source"],
                    eval_year=eval_year, anchor_type=anchor_type,
                    anchor_value=anchor_val, cat_label=cat_lbl,
                    threshold=t, realised_flag_rate=realised_fr,
                    nb=nb_pt, ci_low=lo, ci_high=hi,
                    treat_all_nb=nb_all,
                ))

            fr_str = " ".join([
                f"{int(fr*100):>2}%→{[r['nb'] for r in q3_rows if r['model']==v['model'] and r['calibration']==v['cal'] and r['eval_year']==eval_year and r['anchor_type']=='decile_cum' and r['anchor_value']==fr][0]:+.4f}"
                for fr in (0.10, 0.20, 0.40, 0.70)])
            print(f"  {v['model']:<12}{v['cal']:<6} eval_y{eval_year}: "
                  f"{fr_str}")

    q3_df = pd.DataFrame(q3_rows)
    out_q3 = os.path.join(OUTPUT_DIR, f"nb_at_decile_thresholds{SUFFIX}.csv")
    q3_df.to_csv(out_q3, index=False)
    print(f"\nSaved {out_q3} ({len(q3_df)} rows)")

    # ========================================================================
    # ANALYSIS Q4 — subgroup year-1 analysis
    # ========================================================================
    print("\n" + "=" * 78)
    print("Q4 — subgroup year-1 (PR-AUC, top-D capture, top-C capture)")
    print("=" * 78)
    subgroups = _make_subgroup_levels(comp)
    fu = 0
    labels_full = np.array([_include(t, e, fu)[1]
                            for t, e in zip(comp["time_at_event"].values,
                                            comp["gold"].values)])
    include_mask = np.array([_include(t, e, fu)[0]
                             for t, e in zip(comp["time_at_event"].values,
                                             comp["gold"].values)])

    q4_rows = []
    # Q4 needs one cell per paper model (rank-invariant under monotone
    # calibration — see feedback_rank_invariance_no_rerun memory). For LE
    # and PLCOm we pick raw; the consolidator substitutes raw for Platt.
    q4_imaging = [v for v in variants if (v["model"], v["cal"]) in {
        ("LungEvaty", "raw"),
        ("Sybil", "raw"), ("Sybil", "cal"),
        ("Pillar-0", "raw"),
        ("PLCOm", "raw"),
    }]
    for sg_var, levels in subgroups.items():
        for level_label, level_mask in levels:
            mask = level_mask & include_mask
            n_total = int(mask.sum())
            n_pat = comp.loc[mask, "pid"].nunique()
            sub_labels = labels_full[mask]
            sub_pids = comp["pid"].values[mask]
            n_ca = int(sub_labels.sum())

            for v in q4_imaging:
                y1s = v["y1_for_assign"]()
                sub_scores = y1s[mask]
                # Drop NaN scores (e.g., 1 Pillar-0 user scan unmatchable;
                # PLCOm can also be NaN if smokeyr inputs are missing).
                pr_valid = ~np.isnan(sub_scores)
                if (n_ca > 0 and pr_valid.any()
                        and int(sub_labels[pr_valid].sum()) > 0):
                    pr_auc = float(average_precision_score(
                        sub_labels[pr_valid], sub_scores[pr_valid]))
                else:
                    pr_auc = np.nan
                # Top-D and top-C capture within subgroup, using SAME
                # frozen cuts (or test-fit fallback for Pillar-0)
                valid = ~np.isnan(sub_scores)
                if valid.any() and n_ca > 0:
                    if v["cuts"] is None:
                        d_sub = assign_decile_test_fit(sub_scores[valid])
                    else:
                        d_sub = assign_decile_from_cuts(sub_scores[valid],
                                                         v["cuts"])
                    c_sub = decile_to_cat(d_sub)
                    sub_lbl = sub_labels[valid]
                    total = int(sub_lbl.sum())
                    d_cap = (int(sub_lbl[d_sub == 1].sum()) / total
                             if total > 0 else np.nan)
                    c_cap = (int(sub_lbl[c_sub == 1].sum()) / total
                             if total > 0 else np.nan)
                else:
                    d_cap = c_cap = np.nan

                rng = np.random.default_rng(
                    RNG_SEED + 71000 + sum(ord(c) for c in
                                            f"{sg_var}{level_label}{v['model']}"))
                unique_pids = np.array(sorted(set(sub_pids.tolist())))
                pid_to_rows = {p: np.where(sub_pids == p)[0]
                               for p in unique_pids}
                pa_boot = []; dcap_boot = []; ccap_boot = []
                for _ in range(N_BOOT_INC):
                    if len(unique_pids) < 2: continue
                    idx = _resample_idx(rng, sub_pids, pid_to_rows)
                    yb = sub_labels[idx]; sb = sub_scores[idx]
                    if yb.sum() == 0: continue
                    try:
                        pa_boot.append(float(average_precision_score(yb, sb)))
                    except Exception:
                        pass
                    valid_b = ~np.isnan(sb)
                    if not valid_b.any(): continue
                    if v["cuts"] is None:
                        db = assign_decile_test_fit(sb[valid_b])
                    else:
                        db = assign_decile_from_cuts(sb[valid_b], v["cuts"])
                    cb = decile_to_cat(db)
                    yb_v = yb[valid_b]
                    tot_b = int(yb_v.sum())
                    if tot_b > 0:
                        dcap_boot.append(int(yb_v[db == 1].sum()) / tot_b)
                        ccap_boot.append(int(yb_v[cb == 1].sum()) / tot_b)

                pa_lo, pa_hi = _ci(pa_boot)
                dc_lo, dc_hi = _ci(dcap_boot)
                cc_lo, cc_hi = _ci(ccap_boot)
                for metric, val, lo, hi in [
                        ("pr_auc", pr_auc, pa_lo, pa_hi),
                        ("top_d_capture", d_cap, dc_lo, dc_hi),
                        ("top_c_capture", c_cap, cc_lo, cc_hi)]:
                    q4_rows.append(dict(
                        subgroup_var=sg_var, subgroup_level=level_label,
                        n_patients=n_pat, n_scans=n_total, n_cancer=n_ca,
                        model=v["model"], calibration=v["cal"],
                        source=v["source"],
                        metric=metric, value=val,
                        ci_low=lo, ci_high=hi,
                    ))
            print(f"  {sg_var:<18}{level_label:<22}"
                  f"n_pat={n_pat} n_scan={n_total} n_ca={n_ca}")

    q4_df = pd.DataFrame(q4_rows)
    out_q4 = os.path.join(OUTPUT_DIR, f"subgroup_year1{SUFFIX}.csv")
    q4_df.to_csv(out_q4, index=False)
    print(f"\nSaved {out_q4} ({len(q4_df)} rows)")

    # ========================================================================
    # Sanity checks
    # ========================================================================
    print("\n" + "=" * 78)
    print("Sanity checks")
    print("=" * 78)
    # Year-1 cancer events should sum to total y1 cancers across all deciles
    total_y1_ca = int(y1_lbl[y1_include].sum())
    for v in variants:
        y1s = v["y1_for_assign"]()
        valid = ~np.isnan(y1s) & y1_include
        if v["cuts"] is None:
            d_local = assign_decile_test_fit(y1s[valid])
        else:
            d_local = assign_decile_from_cuts(y1s[valid], v["cuts"])
        cas = int(y1_lbl[valid].sum())
        if cas != total_y1_ca:
            print(f"  WARN {v['model']:<12}{v['cal']:<6}: "
                  f"y1 cancers in valid={cas} ≠ total y1 cancers={total_y1_ca}")
        else:
            print(f"  OK   {v['model']:<12}{v['cal']:<6}: y1 cancer sum = "
                  f"{cas} = total")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Subgroup definitions (same as quintile_stratification_scan.py)
# ---------------------------------------------------------------------------
def _make_subgroup_levels(comp):
    out = {}
    out["sex"] = [("Male", comp["gender"].values == 1),
                  ("Female", comp["gender"].values == 2)]
    ages = comp["age"].values.astype(float)
    a33, a67 = np.percentile(ages, [33.33, 66.67])
    out["age_tertile"] = [
        (f"≤{a33:.0f}", ages <= a33),
        (f"{a33:.0f}<age≤{a67:.0f}", (ages > a33) & (ages <= a67)),
        (f">{a67:.0f}", ages > a67),
    ]
    race = comp["race"].values
    out["race"] = [
        ("White", race == 1),
        ("Black/AA", race == 2),
        ("Asian", race == 3),
        ("Other/Native", np.isin(race, [4, 5, 6, 7])),
    ]
    pk = comp["pkyr"].values.astype(float)
    q1, q2, q3 = np.percentile(pk, [25, 50, 75])
    out["pkyr_quartile"] = [
        (f"Q1≤{q1:.0f}", pk <= q1),
        (f"{q1:.0f}<Q2≤{q2:.0f}", (pk > q1) & (pk <= q2)),
        (f"{q2:.0f}<Q3≤{q3:.0f}", (pk > q2) & (pk <= q3)),
        (f"Q4>{q3:.0f}", pk > q3),
    ]
    out["screen_timepoint"] = [
        ("T0", comp["screen_timepoint"].values == 0),
        ("T1", comp["screen_timepoint"].values == 1),
        ("T2", comp["screen_timepoint"].values == 2),
    ]
    return out


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
