#!/usr/bin/env python3
"""
Sybil-style evaluation of the lung cancer risk prediction model on NLST.

Produces:
  - AUC by year (1–6), with 95% bootstrap CI
  - Subgroup AUC (age, sex, race, pack-years, screen timepoint)
  - LungRADS-like comparison (NLST scr_res as proxy)
  - Population characteristics table
  - ROC curve plots (per year, subgroups, LungRADS overlay)

Metric logic mirrors vital/metrics.py:
  For year N (followup index = N-1):
    positive  = gold==1 AND time_at_event <= (N-1)
    negative  = time_at_event >= (N-1)   [sufficient follow-up]
    excluded  = everything else (early censors without cancer)
"""

import json
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             brier_score_loss, log_loss)
from lifelines import KaplanMeierFitter
from vital.metrics import concordance_index, get_censoring_dist

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Paths default to bundle-relative locations so the pipeline runs from any
# clone of the code release. NLST raw tabular data (restricted-access) and the
# raw LungEvaty test/dev JSONs are not shipped — set the env vars below to
# point at your local copies if you intend to rebuild four_way_comp_df.csv
# from scratch (BUILD_COMPDF=1 in run_all.sh).
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT  = os.path.dirname(_SCRIPTS_DIR)
DATA_DIR     = os.path.join(BUNDLE_ROOT, "data")

# NLST tabular: data/nlst_derived/ ships paper-subset CSVs (38 PLCOm-relevant
# cols, 4,535 pids) that satisfy every column the scripts read. Full NLST is
# restricted (https://cdas.cancer.gov/nlst/); set NLST_TAB_DIR to your local
# copy if you have it.
NLST_TAB_DIR    = os.environ.get(
    "NLST_TAB_DIR", "/tank-1/data/NLST/tab-data")
PARTICIPANT_CSV = os.environ.get(
    "NLST_PARTICIPANT_CSV",
    os.path.join(DATA_DIR, "nlst_derived", "participant_paper_subset.csv")
    if os.path.exists(os.path.join(DATA_DIR, "nlst_derived", "participant_paper_subset.csv"))
    else os.path.join(NLST_TAB_DIR, "participant_d040722.csv"))
LUNG_CANCER_CSV = os.environ.get(
    "NLST_LUNG_CANCER_CSV",
    os.path.join(DATA_DIR, "nlst_derived", "lung_cancer_paper_subset.csv")
    if os.path.exists(os.path.join(DATA_DIR, "nlst_derived", "lung_cancer_paper_subset.csv"))
    else os.path.join(NLST_TAB_DIR, "lung_cancer_d040722.csv"))

# Raw LungEvaty test predictions JSON. Shipped in data/. Set
# LUNGEVATY_TEST_JSON to override.
PREDS_PATH = os.environ.get(
    "LUNGEVATY_TEST_JSON",
    os.path.join(DATA_DIR, "predictions_lungevaty_test.json"))

EXAM_MAP_JSON = os.environ.get(
    "NLST_EXAM_MAP_JSON",
    os.path.join(DATA_DIR, "test_mikhael_with_exam.json"))

# Runtime outputs land here. Override with SCREENING_METRICS_OUTPUT_DIR.
OUTPUT_DIR = os.environ.get(
    "SCREENING_METRICS_OUTPUT_DIR",
    os.path.join(BUNDLE_ROOT, "eval_output"))

# Author Sybil-1.5 (= Pillar-0 architecture) prediction tree — used by the
# optional rebuild path. data/sybil1.5 ships seed{0,1,2}/test.csv (flat); the
# original layout was seed{0,1,2}/checkpoints/3/test.csv. load_sybil_predictions
# transparently handles either.
SYBIL_DIR   = os.environ.get(
    "SYBIL15_DIR",
    os.path.join(DATA_DIR, "sybil1.5"))
SYBIL_SEEDS = [0, 1, 2]

# Series-UID alignment lookup. Shipped inside scripts/ in the bundle.
SERIES_MATCH_CSV = os.environ.get(
    "NLST_SERIES_MATCH_CSV",
    os.path.join(_SCRIPTS_DIR, "nlst_accession_match_results.csv"))

# Ensure OUTPUT_DIR exists and seed prebuilt artifacts into it so individual
# scripts can be invoked without a separate setup step. Sources:
#   - data/four_way_comp_df.csv          → OUTPUT_DIR/four_way_comp_df.csv
#   - data/nlst_derived/nlst_lungrads_pinsky.csv  →  OUTPUT_DIR/...
#   - data/nlst_derived/nlst_lungrads_v2022.csv   →  OUTPUT_DIR/...
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _seed(src: str, dst: str) -> None:
    # Copy rather than symlink so scripts that overwrite the destination
    # (four_way_compare.py → four_way_comp_df.csv, derive_lungrads*.py →
    # nlst_lungrads_*.csv) don't write through the link and corrupt the
    # shipped source file.
    if os.path.exists(src) and not os.path.exists(dst):
        import shutil as _shutil
        _shutil.copy2(src, dst)


_seed(os.path.join(DATA_DIR, "four_way_comp_df.csv"),
      os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
_seed(os.path.join(DATA_DIR, "nlst_derived", "nlst_lungrads_pinsky.csv"),
      os.path.join(OUTPUT_DIR, "nlst_lungrads_pinsky.csv"))
_seed(os.path.join(DATA_DIR, "nlst_derived", "nlst_lungrads_v2022.csv"),
      os.path.join(OUTPUT_DIR, "nlst_lungrads_v2022.csv"))

MAX_FOLLOWUP = 6
N_BOOTSTRAP  = 1000
RNG_SEED     = 42

# ---------------------------------------------------------------------------
# NLST scr_res → LungRADS-like mapping
# Based on NLST documentation (pre-LungRADS era):
#   1 = Negative                       → LungRADS 1   (negative)
#   2 = Non-specific / benign finding  → LungRADS 2   (negative)
#   3 = Suspicious finding             → LungRADS 3   (positive for comparison)
#   4 = Positive finding               → LungRADS 4   (positive)
#   5 = Comparison needed, uncertain   → positive
#   6 = Suspicious, other              → positive
#   10–17, 23, 24 = other suspicious   → positive
#   95 = incomplete / 97 = refused / 99 = unknown → excluded (NaN)
# ---------------------------------------------------------------------------
LUNGRADS_NEG = {1, 2}
LUNGRADS_POS = {3, 4, 5, 6, 10, 11, 13, 14, 15, 17, 23, 24}


# ===========================================================================
# Data loading & merging
# ===========================================================================

def load_predictions(path: str) -> pd.DataFrame:
    with open(path) as f:
        preds = json.load(f)
    df = pd.DataFrame(preds)
    # All 6 cancer_risk values are identical per record → scalar score
    df["risk_score"] = df["cancer_risk"].apply(lambda x: float(x[0]))
    df["pid"] = df["pid"].astype(str)
    return df


def load_participant(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["pid"] = df["pid"].astype(str)
    return df


def load_lung_cancer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pid"] = df["pid"].astype(str)
    return df


def merge_data(preds: pd.DataFrame,
               part: pd.DataFrame,
               lc: pd.DataFrame) -> pd.DataFrame:
    demo_cols = [
        "pid", "age", "gender", "race", "pkyr", "smokeyr", "cigsmok",
        "scr_res0", "scr_res1", "scr_res2",
    ]
    merged = preds.merge(part[demo_cols], on="pid", how="left")

    # Per-scan screening result (match timepoint → scr_resN)
    def _scr_res(row):
        tp = row["screen_timepoint"]
        col = f"scr_res{tp}"
        return row[col] if col in row.index else np.nan

    merged["scr_res"] = merged.apply(_scr_res, axis=1)

    # Binary LungRADS-like label (NaN = excluded from LungRADS analysis)
    def _lr(v):
        if v in LUNGRADS_POS:
            return 1
        elif v in LUNGRADS_NEG:
            return 0
        return np.nan

    merged["lungrads_pos"] = merged["scr_res"].apply(_lr)

    # Attach stage for cancer cases (first confirmed cancer per patient)
    lc_cols = ["pid", "de_stag"] + [c for c in lc.columns if "histol" in c.lower() or "morph" in c.lower()]
    lc_first = lc[lc["first_lc"] == 1][lc_cols].copy()
    merged = merged.merge(lc_first, on="pid", how="left")

    return merged


# ===========================================================================
# Sybil data loading
# ===========================================================================

def load_series_match_maps(match_csv_path: str,
                           preds_df: pd.DataFrame) -> tuple:
    """Build all three matching maps from nlst_accession_match_results.csv.

    Strategy 1 (accession):       accession_str  → series_uid
    Strategy 2 (exact_exam):      constructed exam → series_uid
    Strategy 3 (pid_yr_fallback): (pid_str, study_yr_int) → series_uid
        Built from ALL matched series so it catches Sybil rows with
        TCIA-format accessions not present in the CSV's accession column.

    Needs preds_df to look up screen_timepoint per series for strategies 2-3.
    Returns (acc_to_series, exam_to_series, pidyr_to_series).
    """
    match_df = pd.read_csv(match_csv_path, dtype=str)
    matched = match_df[match_df["status"] == "matched"].copy()

    # series → screen_timepoint from our predictions
    s2tp = dict(zip(preds_df["series"].astype(str),
                    preds_df["screen_timepoint"].astype(int)))

    acc_to_series = {}
    exam_to_series = {}
    pidyr_to_series = {}

    for _, row in matched.iterrows():
        series = row["series_uid"]
        acc = row["dicom_accession_number"]
        key = row["match_key"]
        pid = str(row["pid"])
        tp = s2tp.get(series)

        if key == "accession" and pd.notna(acc) and acc.strip() != "":
            acc_to_series[str(acc)] = series

        elif key == "exact_exam":
            # Exam = {pid}{study_yr}{series_uid[-3:]}
            if tp is not None:
                exam_10 = f"{pid}{tp}{series[-3:]}"
                exam_to_series[exam_10] = series
                # Also store int-cast version (Sybil strips leading zeros)
                if exam_10.isdigit():
                    exam_to_series[str(int(exam_10))] = series

        # Strategy 3: build (pid, yr) → series for ALL matched series
        # (not just pid_yr_fallback rows) so it catches Sybil rows with
        # TCIA-format accessions that don't appear in the CSV.
        if tp is not None:
            pidyr_to_series.setdefault((pid, tp), series)

    # Also add any prediction series not in the CSV (covers the 58 series
    # in our predictions that the match CSV doesn't include).
    for _, row in preds_df.iterrows():
        pid = str(row["pid"])
        tp = int(row["screen_timepoint"])
        series = str(row["series"])
        pidyr_to_series.setdefault((pid, tp), series)

    return acc_to_series, exam_to_series, pidyr_to_series


def _parse_sybil_survival(s: str) -> list:
    """Parse '[-3.078 -3.078 ...]' string into list of floats."""
    return [float(x) for x in s.strip("[]").split()]


def load_sybil_predictions(sybil_dir: str, seeds: list,
                           acc_to_series: dict,
                           exam_to_series: dict,
                           pidyr_to_series: dict) -> pd.DataFrame:
    """Load Sybil predictions from all seeds, ensemble by averaging logits.

    Three-strategy matching (mirrors nlst_accession_match_results.csv):
      1. accession → series_uid
      2. exam → series_uid  (constructed exam key)
      3. (pid, study_yr) fallback → series_uid

    Returns DataFrame with columns: series, sybil_risk_1..sybil_risk_6,
    time_at_event, gold.
    """
    seed_dfs = []
    for seed in seeds:
        # Original layout: sybil1.5/seedN/checkpoints/3/test.csv
        # Supplement layout: sybil1.5/seedN/test.csv (flat)
        path = os.path.join(sybil_dir, f"seed{seed}", "checkpoints", "3", "test.csv")
        if not os.path.exists(path):
            alt = os.path.join(sybil_dir, f"seed{seed}", "test.csv")
            if os.path.exists(alt):
                path = alt
        df = pd.read_csv(path)
        df["exam"] = df["exam"].astype(str)
        # Strip .0 suffix from float-parsed accession values
        df["accession"] = (df["accession"].astype(str)
                           .str.replace(r"\.0$", "", regex=True))

        # Parse survival logits into 6 columns
        parsed = df["survival"].apply(_parse_sybil_survival)
        for yr in range(1, 7):
            df[f"logit_{yr}"] = parsed.apply(lambda x: x[yr - 1])

        def _resolve_series(row):
            # Strategy 1: accession → series
            acc = row["accession"]
            if acc in acc_to_series:
                return acc_to_series[acc]
            # Strategy 2: exam → series
            exam = row["exam"]
            if exam in exam_to_series:
                return exam_to_series[exam]
            # Strategy 3: (pid, study_yr) fallback
            # Sybil exam format: {pid(6)}{study_yr(1)}{suffix}
            if len(exam) >= 7 and exam[:6].isdigit():
                pid = exam[:6]
                yr = int(exam[6])
                if (pid, yr) in pidyr_to_series:
                    return pidyr_to_series[(pid, yr)]
            return None

        df["series"] = df.apply(_resolve_series, axis=1)
        df = df.dropna(subset=["series"])

        seed_dfs.append(df)

    # Ensemble: average logits across seeds, then sigmoid → probability
    merged = seed_dfs[0][["series", "accession", "time_at_event", "y"]].copy()
    for yr in range(1, 7):
        cols = [sdf.set_index("series")[f"logit_{yr}"] for sdf in seed_dfs]
        stacked = pd.concat(cols, axis=1)
        merged = merged.set_index("series")
        mean_logit = stacked.mean(axis=1)
        merged[f"sybil_risk_{yr}"] = 1.0 / (1.0 + np.exp(-mean_logit))
        merged = merged.reset_index()

    merged = merged.rename(columns={"y": "gold"})
    return merged


def build_comparison_df(our_df: pd.DataFrame,
                        sybil_df: pd.DataFrame) -> pd.DataFrame:
    """Inner join our predictions with Sybil on series (matched set)."""
    sybil_cols = ["series"] + [f"sybil_risk_{yr}" for yr in range(1, 7)]
    return our_df.merge(sybil_df[sybil_cols], on="series", how="inner")


# ===========================================================================
# Core metric helpers  (mirrors vital/metrics.py logic)
# ===========================================================================

def _include(censor_time: float, gold: int, followup: int,
             fup_lower_bound: int = -1) -> tuple:
    """
    Returns (included, label) for a single exam.

    followup      : 0-indexed year index (0 → 1-year horizon, etc.)
    fup_lower_bound: -1 → include all cancers up to and including `followup`
    """
    valid_pos = bool(gold) and (censor_time <= followup) and (censor_time > fup_lower_bound)
    valid_neg = censor_time >= followup
    return (valid_pos or valid_neg), int(valid_pos)


def _filter_for_year(df: pd.DataFrame, followup: int) -> tuple:
    """Return (scores, labels) arrays for the given followup index."""
    scores, labels = [], []
    for _, row in df.iterrows():
        inc, lbl = _include(row["time_at_event"], row["gold"], followup)
        if inc:
            scores.append(row["risk_score"])
            labels.append(lbl)
    return np.array(scores), np.array(labels)


def _bootstrap_auc(scores: np.ndarray, labels: np.ndarray,
                   n_boot: int = N_BOOTSTRAP) -> tuple:
    """Returns (auc, ci_lo, ci_hi).  NaN if class is degenerate."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan, np.nan, np.nan

    point_auc = roc_auc_score(labels, scores)
    rng = np.random.default_rng(RNG_SEED)
    n = len(labels)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bl, bs = labels[idx], scores[idx]
        if bl.sum() == 0 or bl.sum() == n:
            continue
        try:
            boot.append(roc_auc_score(bl, bs))
        except Exception:
            pass
    boot = np.array(boot)
    return point_auc, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ===========================================================================
# Analysis 1 – AUC by year
# ===========================================================================

def compute_yearly_aucs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in range(1, MAX_FOLLOWUP + 1):
        followup = year - 1
        scores, labels = _filter_for_year(df, followup)
        auc_val, ci_lo, ci_hi = _bootstrap_auc(scores, labels)
        n_cancer = int(labels.sum())
        rows.append(dict(
            Year=year,
            AUC=auc_val, CI_lo=ci_lo, CI_hi=ci_hi,
            N=len(labels), N_cancer=n_cancer, N_control=len(labels) - n_cancer,
        ))
    return pd.DataFrame(rows)


# ===========================================================================
# Analysis 2 – Subgroup AUC
# ===========================================================================

def _subgroup_mask(df: pd.DataFrame) -> dict:
    pkyr_q = df["pkyr"].quantile([0.25, 0.5, 0.75])
    return {
        # Age
        "Age <55":        df["age"] < 55,
        "Age 55–59":      df["age"].between(55, 59),
        "Age 60–64":      df["age"].between(60, 64),
        "Age 65–69":      df["age"].between(65, 69),
        "Age ≥70":        df["age"] >= 70,
        # Sex
        "Male":           df["gender"] == 1,
        "Female":         df["gender"] == 2,
        # Race
        "White":          df["race"] == 1,
        "Black/AA":       df["race"] == 2,
        "Other race":     ~df["race"].isin([1, 2]),
        # Pack-years quartiles
        "Pack-years Q1":  df["pkyr"] <= pkyr_q[0.25],
        "Pack-years Q2":  df["pkyr"].between(pkyr_q[0.25], pkyr_q[0.50]),
        "Pack-years Q3":  df["pkyr"].between(pkyr_q[0.50], pkyr_q[0.75]),
        "Pack-years Q4":  df["pkyr"] > pkyr_q[0.75],
        # Screen timepoint
        "Timepoint 0":    df["screen_timepoint"] == 0,
        "Timepoint 1":    df["screen_timepoint"] == 1,
        "Timepoint 2":    df["screen_timepoint"] == 2,
    }


def compute_subgroup_aucs(df: pd.DataFrame, year: int = 1) -> pd.DataFrame:
    followup = year - 1
    masks = _subgroup_mask(df)
    rows = []
    for name, mask in masks.items():
        sub = df[mask]
        if len(sub) < 20:
            continue
        scores, labels = _filter_for_year(sub, followup)
        if labels.sum() == 0:
            continue
        auc_val, ci_lo, ci_hi = _bootstrap_auc(scores, labels)
        rows.append(dict(
            Subgroup=name,
            AUC=auc_val, CI_lo=ci_lo, CI_hi=ci_hi,
            N=len(labels), N_cancer=int(labels.sum()),
        ))
    return pd.DataFrame(rows)


# ===========================================================================
# Analysis 3 – LungRADS-like comparison
# ===========================================================================

def compute_lungrads_comparison(df: pd.DataFrame, year: int = 1) -> dict:
    followup = year - 1

    # Build per-exam table for this year
    records = []
    for _, row in df.iterrows():
        inc, lbl = _include(row["time_at_event"], row["gold"], followup)
        if inc:
            records.append(dict(
                risk_score=row["risk_score"],
                gold=lbl,
                lungrads_pos=row["lungrads_pos"],
            ))
    sub = pd.DataFrame(records).dropna(subset=["lungrads_pos"])

    gold  = sub["gold"].values
    score = sub["risk_score"].values
    lr    = sub["lungrads_pos"].astype(int).values

    # Model AUC
    model_auc = roc_auc_score(gold, score) if gold.sum() > 0 else np.nan

    # LungRADS sensitivity & specificity
    TP = int(((lr == 1) & (gold == 1)).sum())
    FN = int(((lr == 0) & (gold == 1)).sum())
    TN = int(((lr == 0) & (gold == 0)).sum())
    FP = int(((lr == 1) & (gold == 0)).sum())

    lr_sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    lr_spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan

    # Model sensitivity at LungRADS specificity
    fpr, tpr, _ = roc_curve(gold, score)
    model_spec = 1 - fpr
    idx = int(np.argmin(np.abs(model_spec - lr_spec)))
    model_sens_at_lr_spec = float(tpr[idx])

    return dict(
        year=year,
        n_exams=len(sub),
        n_cancer=int(gold.sum()),
        model_auc=float(model_auc),
        lr_sensitivity=float(lr_sens),
        lr_specificity=float(lr_spec),
        model_sens_at_lr_spec=model_sens_at_lr_spec,
        # For plotting
        fpr=fpr, tpr=tpr,
        lr_fpr=float(1 - lr_spec), lr_tpr=float(lr_sens),
    )


# ===========================================================================
# Analysis 5 – Head-to-head comparison (Lungevity vs Sybil)
# ===========================================================================

def _filter_for_year_comparison(df: pd.DataFrame, followup: int) -> tuple:
    """Return (our_scores, sybil_scores, labels) for the given followup."""
    our, syb, labels = [], [], []
    sybil_col = f"sybil_risk_{followup + 1}"
    for _, row in df.iterrows():
        inc, lbl = _include(row["time_at_event"], row["gold"], followup)
        if inc:
            our.append(row["risk_score"])
            syb.append(row[sybil_col])
            labels.append(lbl)
    return np.array(our), np.array(syb), np.array(labels)


def _bootstrap_auc_pair(scores_a: np.ndarray, scores_b: np.ndarray,
                        labels: np.ndarray,
                        n_boot: int = N_BOOTSTRAP) -> tuple:
    """Bootstrap AUC + CI for two models simultaneously (paired sampling)."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        nan3 = (np.nan, np.nan, np.nan)
        return nan3, nan3

    auc_a = roc_auc_score(labels, scores_a)
    auc_b = roc_auc_score(labels, scores_b)

    rng = np.random.default_rng(RNG_SEED)
    n = len(labels)
    boot_a, boot_b = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bl = labels[idx]
        if bl.sum() == 0 or bl.sum() == n:
            continue
        try:
            boot_a.append(roc_auc_score(bl, scores_a[idx]))
            boot_b.append(roc_auc_score(bl, scores_b[idx]))
        except Exception:
            pass
    boot_a, boot_b = np.array(boot_a), np.array(boot_b)
    ci_a = (float(np.percentile(boot_a, 2.5)), float(np.percentile(boot_a, 97.5)))
    ci_b = (float(np.percentile(boot_b, 2.5)), float(np.percentile(boot_b, 97.5)))
    return (auc_a, *ci_a), (auc_b, *ci_b)


def _compute_censoring_dist(df: pd.DataFrame) -> dict:
    """Build censoring distribution from the comparison DataFrame."""
    records = [{"time_at_event": row["time_at_event"], "y": row["gold"]}
               for _, row in df.iterrows()]
    return get_censoring_dist(records)


def compute_yearly_aucs_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """AUC + PR-AUC + C-index by year for both models on the matched set."""
    # C-index (computed once, uses all timepoints)
    # concordance_index expects 2D scores: (N, max_followup)
    cens_dist = _compute_censoring_dist(df)
    event_times = df["time_at_event"].values
    event_observed = df["gold"].values.astype(int)

    # Our model: single risk score replicated across years
    our_scores_2d = np.tile(df["risk_score"].values[:, None], (1, MAX_FOLLOWUP))
    our_c = concordance_index(event_times, our_scores_2d,
                              event_observed, cens_dist)

    # Sybil: year-specific scores
    syb_scores_2d = np.column_stack(
        [df[f"sybil_risk_{yr}"].values for yr in range(1, MAX_FOLLOWUP + 1)])
    syb_c = concordance_index(event_times, syb_scores_2d,
                              event_observed, cens_dist)

    rows = []
    for year in range(1, MAX_FOLLOWUP + 1):
        followup = year - 1
        our, syb, labels = _filter_for_year_comparison(df, followup)
        (a_auc, a_lo, a_hi), (b_auc, b_lo, b_hi) = \
            _bootstrap_auc_pair(our, syb, labels)

        # PR-AUC (no bootstrap, just point estimate)
        if labels.sum() > 0:
            our_prauc = average_precision_score(labels, our)
            syb_prauc = average_precision_score(labels, syb)
        else:
            our_prauc = syb_prauc = np.nan

        n_cancer = int(labels.sum())
        row = dict(
            Year=year,
            Lungevity_AUC=a_auc, Lungevity_CI_lo=a_lo, Lungevity_CI_hi=a_hi,
            Sybil_AUC=b_auc, Sybil_CI_lo=b_lo, Sybil_CI_hi=b_hi,
            Lungevity_PRAUC=our_prauc, Sybil_PRAUC=syb_prauc,
            N=len(labels), N_cancer=n_cancer, N_control=len(labels) - n_cancer,
        )
        # Add C-index only on first row (it's a single global metric)
        if year == 1:
            row["Lungevity_Cindex"] = our_c
            row["Sybil_Cindex"] = syb_c
        rows.append(row)
    return pd.DataFrame(rows)


def compute_subgroup_aucs_comparison(df: pd.DataFrame,
                                     year: int = 1) -> pd.DataFrame:
    """Subgroup AUC for both models on matched set."""
    followup = year - 1
    masks = _subgroup_mask(df)
    rows = []
    for name, mask in masks.items():
        sub = df[mask]
        if len(sub) < 20:
            continue
        our, syb, labels = _filter_for_year_comparison(sub, followup)
        if labels.sum() == 0:
            continue
        (a_auc, a_lo, a_hi), (b_auc, b_lo, b_hi) = \
            _bootstrap_auc_pair(our, syb, labels)
        rows.append(dict(
            Subgroup=name,
            Lungevity_AUC=a_auc, Lungevity_CI_lo=a_lo, Lungevity_CI_hi=a_hi,
            Sybil_AUC=b_auc, Sybil_CI_lo=b_lo, Sybil_CI_hi=b_hi,
            N=len(labels), N_cancer=int(labels.sum()),
        ))
    return pd.DataFrame(rows)


# ===========================================================================
# Analysis 4 – Population characteristics table
# ===========================================================================

def compute_population_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique patient (first occurrence)."""
    patients = df.drop_duplicates("pid").copy()

    def _fmt(v, fmt=".1f"):
        return f"{v:{fmt}}" if not pd.isna(v) else "N/A"

    def _pct(n, tot):
        return f"{n} ({100*n/tot:.1f}%)" if tot > 0 else "N/A"

    n = len(patients)
    n_cancer = int(patients["gold"].sum())

    rows = [
        ("N patients",           str(n)),
        ("Cancer cases",         _pct(n_cancer, n)),
        ("Age, mean ± SD",       f"{patients['age'].mean():.1f} ± {patients['age'].std():.1f}"),
        ("Male",                 _pct((patients["gender"] == 1).sum(), n)),
        ("Female",               _pct((patients["gender"] == 2).sum(), n)),
        ("White",                _pct((patients["race"] == 1).sum(), n)),
        ("Black/AA",             _pct((patients["race"] == 2).sum(), n)),
        ("Other race",           _pct((~patients["race"].isin([1, 2])).sum(), n)),
        ("Pack-years, med (IQR)",
            f"{patients['pkyr'].median():.0f} "
            f"({patients['pkyr'].quantile(0.25):.0f}–{patients['pkyr'].quantile(0.75):.0f})"),
        ("Smoking years, med (IQR)",
            f"{patients['smokeyr'].median():.0f} "
            f"({patients['smokeyr'].quantile(0.25):.0f}–{patients['smokeyr'].quantile(0.75):.0f})"),
    ]

    # Screen timepoint counts (per scan)
    for tp in [0, 1, 2]:
        n_tp = (df["screen_timepoint"] == tp).sum()
        rows.append((f"Scans at timepoint {tp}", str(n_tp)))

    rows.append(("Total scans", str(len(df))))

    return pd.DataFrame(rows, columns=["Characteristic", "Value"])


# ===========================================================================
# Printing helpers
# ===========================================================================

def _hr(char="─", n=70):
    print(char * n)


def print_population_table(pop_df: pd.DataFrame):
    print("\nPOPULATION CHARACTERISTICS")
    _hr()
    for _, row in pop_df.iterrows():
        print(f"  {row['Characteristic']:<35} {row['Value']}")
    _hr()


def print_yearly_aucs(yearly_df: pd.DataFrame):
    print("\nAUC BY YEAR  (95% CI, bootstrap n={})".format(N_BOOTSTRAP))
    _hr()
    hdr = f"  {'Year':<6} {'AUC':>6}  {'95% CI':<17}  {'N':>6}  {'N_cancer':>8}  {'N_control':>10}"
    print(hdr)
    _hr("·")
    for _, r in yearly_df.iterrows():
        auc_str = f"{r['AUC']:.3f}" if not pd.isna(r["AUC"]) else " N/A "
        ci_str  = f"({r['CI_lo']:.3f}–{r['CI_hi']:.3f})" if not pd.isna(r["CI_lo"]) else "      N/A      "
        print(f"  {int(r['Year']):<6} {auc_str:>6}  {ci_str:<17}  "
              f"{int(r['N']):>6}  {int(r['N_cancer']):>8}  {int(r['N_control']):>10}")
    _hr()


def print_subgroup_table(sg_df: pd.DataFrame, year: int):
    print(f"\nSUBGROUP AUC  ({year}-YEAR HORIZON)")
    _hr()
    hdr = f"  {'Subgroup':<28} {'AUC':>6}  {'95% CI':<17}  {'N':>6}  {'N_cancer':>8}"
    print(hdr)
    _hr("·")
    for _, r in sg_df.iterrows():
        auc_str = f"{r['AUC']:.3f}" if not pd.isna(r["AUC"]) else " N/A "
        ci_str  = f"({r['CI_lo']:.3f}–{r['CI_hi']:.3f})" if not pd.isna(r["CI_lo"]) else "      N/A      "
        print(f"  {r['Subgroup']:<28} {auc_str:>6}  {ci_str:<17}  "
              f"{int(r['N']):>6}  {int(r['N_cancer']):>8}")
    _hr()


def print_lungrads_table(lr_list: list):
    print("\nLUNGRADS-LIKE COMPARISON  (NLST scr_res ≥ 3 = positive)")
    _hr()
    print(f"  {'Year':<5} {'N exams':>8} {'N cancer':>9} {'Model AUC':>10}  "
          f"{'LR Sens':>8}  {'LR Spec':>8}  {'Model Sens @ LR Spec':>22}")
    _hr("·")
    for r in lr_list:
        print(f"  {r['year']:<5} {r['n_exams']:>8} {r['n_cancer']:>9} "
              f"{r['model_auc']:>10.3f}  "
              f"{r['lr_sensitivity']:>8.3f}  {r['lr_specificity']:>8.3f}  "
              f"{r['model_sens_at_lr_spec']:>22.3f}")
    _hr()


# ===========================================================================
# Plots
# ===========================================================================

def plot_auc_by_year(yearly_df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    yr  = yearly_df["Year"].values
    auc = yearly_df["AUC"].values
    lo  = yearly_df["CI_lo"].values
    hi  = yearly_df["CI_hi"].values

    ax.plot(yr, auc, "bo-", lw=2, ms=7, label="Model AUC")
    ax.fill_between(yr, lo, hi, alpha=0.2, color="steelblue", label="95% CI")
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="Chance")
    ax.set_xticks(yr)
    ax.set_xlabel("Prediction horizon (years)")
    ax.set_ylabel("AUC")
    ax.set_title("Model AUC by prediction horizon")
    ax.set_ylim(0.45, 1.0)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out, "auc_by_year.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_roc_all_years(df: pd.DataFrame, out: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, MAX_FOLLOWUP))

    for i, year in enumerate(range(1, MAX_FOLLOWUP + 1)):
        ax = axes[i]
        followup = year - 1
        scores, labels = _filter_for_year(df, followup)
        if labels.sum() == 0:
            ax.set_visible(False)
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_val = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("1 − Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(f"{year}-Year Cancer Risk")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("ROC curves by prediction horizon", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out, "roc_all_years.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_roc_with_lungrads(lr_result: dict, out: str):
    year = lr_result["year"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(lr_result["fpr"], lr_result["tpr"], "b-", lw=2,
            label=f"Model  AUC = {lr_result['model_auc']:.3f}")
    ax.scatter(
        [lr_result["lr_fpr"]], [lr_result["lr_tpr"]],
        color="red", s=120, zorder=5,
        label=(f"LungRADS-like  "
               f"Sens={lr_result['lr_sensitivity']:.3f}, "
               f"Spec={lr_result['lr_specificity']:.3f}"),
    )
    # Model operating point at matched specificity
    fpr_arr = lr_result["fpr"]
    tpr_arr = lr_result["tpr"]
    idx = int(np.argmin(np.abs((1 - fpr_arr) - lr_result["lr_specificity"])))
    ax.scatter([fpr_arr[idx]], [tpr_arr[idx]], color="blue", s=120,
               marker="^", zorder=5,
               label=f"Model @ LR spec  Sens={lr_result['model_sens_at_lr_spec']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"{year}-Year Risk: Model vs LungRADS-like")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(out, f"roc_lungrads_year{year}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_subgroup_forest(sg_df: pd.DataFrame, year: int, out: str):
    sg = sg_df.dropna(subset=["AUC"]).reset_index(drop=True)
    if sg.empty:
        return

    fig, ax = plt.subplots(figsize=(9, max(5, len(sg) * 0.4 + 1)))
    y = np.arange(len(sg))
    ax.barh(y, sg["AUC"],
            xerr=[sg["AUC"] - sg["CI_lo"], sg["CI_hi"] - sg["AUC"]],
            align="center", height=0.6, alpha=0.8,
            capsize=4, color="steelblue", ecolor="navy")
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sg["Subgroup"], fontsize=9)
    ax.set_xlabel("AUC")
    ax.set_title(f"Subgroup AUC  ({year}-year horizon)")
    ax.set_xlim(max(0.3, sg["CI_lo"].min() - 0.05), min(1.0, sg["CI_hi"].max() + 0.05))

    # Annotate AUC values
    for i, row in sg.iterrows():
        ax.text(row["AUC"] + 0.005, i,
                f"{row['AUC']:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out, f"subgroup_forest_year{year}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Comparison printing & plotting
# ===========================================================================

def print_comparison_yearly(comp_df: pd.DataFrame):
    # Print C-index (single global metric)
    r0 = comp_df.iloc[0]
    if "Lungevity_Cindex" in r0 and not pd.isna(r0.get("Lungevity_Cindex")):
        print(f"\nC-INDEX (Uno's, IPCW-weighted, matched set)")
        _hr("=")
        print(f"  Lungevity:  {r0['Lungevity_Cindex']:.3f}")
        print(f"  Sybil:      {r0['Sybil_Cindex']:.3f}")
        _hr("=")

    print("\nHEAD-TO-HEAD AUC COMPARISON  (matched set, 95% CI)")
    _hr("=")
    hdr = (f"  {'Year':<5} {'Lungevity AUC':>14} {'95% CI':<17}"
           f"  {'Sybil AUC':>10} {'95% CI':<17}"
           f"  {'Lung PRAUC':>11} {'Syb PRAUC':>10}"
           f"  {'N':>6} {'N_ca':>5}")
    print(hdr)
    _hr("·")
    for _, r in comp_df.iterrows():
        def _f(v): return f"{v:.3f}" if not pd.isna(v) else " N/A "
        def _ci(lo, hi):
            return f"({lo:.3f}-{hi:.3f})" if not pd.isna(lo) else "     N/A      "
        print(f"  {int(r['Year']):<5} {_f(r['Lungevity_AUC']):>14} "
              f"{_ci(r['Lungevity_CI_lo'], r['Lungevity_CI_hi']):<17}"
              f"  {_f(r['Sybil_AUC']):>10} "
              f"{_ci(r['Sybil_CI_lo'], r['Sybil_CI_hi']):<17}"
              f"  {_f(r['Lungevity_PRAUC']):>11} {_f(r['Sybil_PRAUC']):>10}"
              f"  {int(r['N']):>6} {int(r['N_cancer']):>5}")
    _hr("=")


def print_comparison_subgroup(sg_df: pd.DataFrame, year: int):
    print(f"\nSUBGROUP AUC COMPARISON  ({year}-YEAR, matched set)")
    _hr("=")
    hdr = (f"  {'Subgroup':<25} {'Lung AUC':>9} {'95% CI':<17}"
           f"  {'Syb AUC':>8} {'95% CI':<17}"
           f"  {'N':>6} {'N_ca':>5}")
    print(hdr)
    _hr("·")
    for _, r in sg_df.iterrows():
        def _f(v): return f"{v:.3f}" if not pd.isna(v) else " N/A "
        def _ci(lo, hi):
            return f"({lo:.3f}-{hi:.3f})" if not pd.isna(lo) else "     N/A      "
        print(f"  {r['Subgroup']:<25} {_f(r['Lungevity_AUC']):>9} "
              f"{_ci(r['Lungevity_CI_lo'], r['Lungevity_CI_hi']):<17}"
              f"  {_f(r['Sybil_AUC']):>8} "
              f"{_ci(r['Sybil_CI_lo'], r['Sybil_CI_hi']):<17}"
              f"  {int(r['N']):>6} {int(r['N_cancer']):>5}")
    _hr("=")


def plot_comparison_auc_by_year(comp_df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    yr = comp_df["Year"].values

    ax.plot(yr, comp_df["Lungevity_AUC"], "bo-", lw=2, ms=7, label="Lungevity")
    ax.fill_between(yr, comp_df["Lungevity_CI_lo"], comp_df["Lungevity_CI_hi"],
                    alpha=0.15, color="steelblue")
    ax.plot(yr, comp_df["Sybil_AUC"], "rs-", lw=2, ms=7, label="Sybil (3-seed)")
    ax.fill_between(yr, comp_df["Sybil_CI_lo"], comp_df["Sybil_CI_hi"],
                    alpha=0.15, color="salmon")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(yr)
    ax.set_xlabel("Prediction horizon (years)")
    ax.set_ylabel("AUC")
    ax.set_title("Lungevity vs Sybil — AUC by year (matched set)")
    ax.set_ylim(0.45, 1.0)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out, "comparison_auc_by_year.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_comparison_roc(comp_df_full: pd.DataFrame, year: int, out: str):
    """ROC overlay for both models at a given year."""
    followup = year - 1
    our, syb, labels = _filter_for_year_comparison(comp_df_full, followup)
    if labels.sum() == 0:
        return

    fpr_a, tpr_a, _ = roc_curve(labels, our)
    fpr_b, tpr_b, _ = roc_curve(labels, syb)
    auc_a = roc_auc_score(labels, our)
    auc_b = roc_auc_score(labels, syb)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_a, tpr_a, "b-", lw=2, label=f"Lungevity  AUC={auc_a:.3f}")
    ax.plot(fpr_b, tpr_b, "r-", lw=2, label=f"Sybil      AUC={auc_b:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"{year}-Year Risk: Lungevity vs Sybil")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(out, f"comparison_roc_year{year}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Clinical Evaluation Framework (per paper)
#   §2: Partial AUC + Specificity at Fixed Sensitivity
#   §3: Brier Score (Murphy decomposition) + Log Loss
#   §4: Survival-adapted Decision Curve Analysis
# ===========================================================================

def _compute_pauc_high_sensitivity(fpr, tpr, min_sens=0.90):
    """Partial AUC restricted to sensitivity >= min_sens (paper §2).

    Returns the area under the ROC curve in the high-sensitivity region.
    """
    mask = tpr >= min_sens
    if mask.sum() < 2:
        return np.nan
    return float(np.abs(np.trapz(tpr[mask], fpr[mask])))


def _specificity_at_sensitivity(fpr, tpr, target_sens):
    """Specificity at the lowest FPR achieving >= target_sens (paper §2)."""
    idx = np.where(tpr >= target_sens)[0]
    if len(idx) == 0:
        return np.nan
    return float(1.0 - fpr[idx[0]])


def _brier_decomposition(y_true, y_prob, n_bins=10):
    """Murphy decomposition: BS = reliability - resolution + uncertainty.

    Returns (reliability, resolution, uncertainty).
    Reliability = calibration error (lower is better).
    Resolution  = discrimination sharpness (higher is better).
    Uncertainty = base-rate entropy (data-dependent constant).
    """
    y_bar = np.mean(y_true)
    unc = y_bar * (1 - y_bar)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    n = len(y_true)
    rel = 0.0
    res = 0.0
    for k in range(n_bins):
        mask = bin_idx == k
        n_k = mask.sum()
        if n_k == 0:
            continue
        o_k = np.mean(y_true[mask])
        f_k = np.mean(y_prob[mask])
        rel += n_k * (f_k - o_k) ** 2
        res += n_k * (o_k - y_bar) ** 2

    return rel / n, res / n, unc


def _compute_clinical_metrics(scores, labels):
    """Compute all paper-framework metrics for a given (scores, labels) pair.

    Returns dict with pAUC, specificity@sens, Brier, log loss, decomposition.
    Returns None if data is degenerate (single class).
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None

    fpr, tpr, _ = roc_curve(labels, scores)

    pauc_90 = _compute_pauc_high_sensitivity(fpr, tpr, 0.90)
    spec_at_90 = _specificity_at_sensitivity(fpr, tpr, 0.90)
    spec_at_95 = _specificity_at_sensitivity(fpr, tpr, 0.95)

    s_clip = np.clip(scores, 1e-7, 1 - 1e-7)
    brier = float(brier_score_loss(labels, s_clip))
    ll = float(log_loss(labels, s_clip))
    rel, res, unc = _brier_decomposition(labels, s_clip)

    return dict(
        pAUC_sens90=pauc_90,
        spec_at_sens90=spec_at_90,
        spec_at_sens95=spec_at_95,
        brier_score=brier,
        log_loss_val=ll,
        brier_reliability=rel,
        brier_resolution=res,
        brier_uncertainty=unc,
    )


def compute_clinical_metrics_by_year(df):
    """Compute clinical metrics for each year horizon (standalone)."""
    rows = []
    for year in range(1, MAX_FOLLOWUP + 1):
        scores, labels = _filter_for_year(df, year - 1)
        metrics = _compute_clinical_metrics(scores, labels)
        if metrics is None:
            continue
        metrics["Year"] = year
        metrics["N"] = len(labels)
        metrics["N_cancer"] = int(labels.sum())
        rows.append(metrics)
    return pd.DataFrame(rows)


def _ipcw_weights(times, events, followup):
    """Compute IPCW weights for Survival DCA.

    Uses the Kaplan-Meier estimator of the censoring distribution G(t).
    - TP subjects (event=1, time<=followup): weight = 1 / G(time_i)
    - FP subjects (time>followup):           weight = 1 / G(followup)
    - Censored before followup (event=0, time<=followup): weight = 0

    Returns (tp_weights, fp_weight, is_tp, is_fp, prevalence_t).
    """
    # KM for censoring survival: G(t) = P(not censored by t)
    kmf_cens = KaplanMeierFitter()
    kmf_cens.fit(times, 1 - events)

    # KM for event: prevalence(t) = 1 - S(t)
    kmf_event = KaplanMeierFitter()
    kmf_event.fit(times, events)
    prevalence_t = float(1.0 - kmf_event.predict(followup))

    G_t = float(max(kmf_cens.predict(followup), 0.01))

    is_tp = (events == 1) & (times <= followup)
    # FP-eligible: followed to/past horizon WITHOUT an event. Consistent
    # with `_include` (line ~281) which uses time>=followup for valid_neg.
    # Prior strict `times > followup` was buggy at year 6 because NLST
    # time_at_event maxes at 5 (administrative end of follow-up) so no
    # patient satisfied times>5 and the FP penalty vanished.
    is_fp = (events == 0) & (times >= followup)

    # IPCW weights for TP subjects (one per unique event time)
    tp_weights = np.zeros(len(times))
    unique_tp_times = np.unique(times[is_tp])
    G_lookup = {t: float(max(kmf_cens.predict(t), 0.01))
                for t in unique_tp_times}
    for i in np.where(is_tp)[0]:
        tp_weights[i] = 1.0 / G_lookup[times[i]]

    fp_weight = 1.0 / G_t

    return tp_weights, fp_weight, is_tp, is_fp, prevalence_t


def compute_survival_dca(df, followup, n_thresholds=200):
    """Survival-adapted Decision Curve Analysis with IPCW (paper §4).

    Net Benefit(t) = TP_d(t) - FP_d(t) * (p_t / (1 - p_t))

    Uses Kaplan-Meier estimation of the censoring distribution to weight
    subjects via IPCW, properly handling right-censored observations.
    """
    times = df["time_at_event"].values.astype(float)
    events = df["gold"].values.astype(int)
    scores = df["risk_score"].values.astype(float)
    n = len(times)

    if events.sum() == 0:
        return pd.DataFrame()

    tp_weights, fp_weight, is_tp, is_fp, prevalence_t = \
        _ipcw_weights(times, events, followup)

    thresholds = np.linspace(0.005, 0.50, n_thresholds)

    rows = []
    for pt in thresholds:
        pos = scores >= pt
        tp_rate = float(tp_weights[pos & is_tp].sum()) / n
        fp_rate = float((pos & is_fp).sum()) * fp_weight / n
        w = pt / (1 - pt)
        rows.append(dict(
            threshold=pt,
            nb_model=tp_rate - fp_rate * w,
            nb_all=prevalence_t - (1 - prevalence_t) * w,
            nb_none=0.0,
        ))
    return pd.DataFrame(rows)


def compute_dca_complete_case(df, followup, n_thresholds=200):
    """Standard DCA using complete-case analysis (excludes censored subjects).

    Subjects censored before the horizon are excluded entirely.
    Included: cancer within horizon (TP-eligible) + followed past horizon (FP-eligible).
    """
    scores, labels = _filter_for_year(df, followup)
    if labels.sum() == 0:
        return pd.DataFrame()

    n = len(labels)
    prevalence = float(labels.mean())
    thresholds = np.linspace(0.005, 0.50, n_thresholds)

    rows = []
    for pt in thresholds:
        pos = scores >= pt
        tp_rate = float((pos & (labels == 1)).sum()) / n
        fp_rate = float((pos & (labels == 0)).sum()) / n
        w = pt / (1 - pt)
        rows.append(dict(
            threshold=pt,
            nb_model=tp_rate - fp_rate * w,
            nb_all=prevalence - (1 - prevalence) * w,
            nb_none=0.0,
        ))
    return pd.DataFrame(rows)


def compute_dca_complete_case_comparison(comp_df, followup, n_thresholds=200):
    """Standard DCA complete-case for both models on the matched set."""
    our, syb, labels = _filter_for_year_comparison(comp_df, followup)
    if labels.sum() == 0:
        return pd.DataFrame()

    n = len(labels)
    prevalence = float(labels.mean())
    thresholds = np.linspace(0.005, 0.50, n_thresholds)

    rows = []
    for pt in thresholds:
        w = pt / (1 - pt)
        pos_our = our >= pt
        nb_our = float((pos_our & (labels == 1)).sum()) / n - \
                 float((pos_our & (labels == 0)).sum()) / n * w
        pos_syb = syb >= pt
        nb_syb = float((pos_syb & (labels == 1)).sum()) / n - \
                 float((pos_syb & (labels == 0)).sum()) / n * w
        nb_all = prevalence - (1 - prevalence) * w
        rows.append(dict(
            threshold=pt,
            nb_lungevity=nb_our,
            nb_sybil=nb_syb,
            nb_all=nb_all,
            nb_none=0.0,
        ))
    return pd.DataFrame(rows)


def compute_clinical_metrics_comparison_by_year(comp_df):
    """Compute clinical metrics for both models on the matched set."""
    rows = []
    for year in range(1, MAX_FOLLOWUP + 1):
        our, syb, labels = _filter_for_year_comparison(comp_df, year - 1)
        our_m = _compute_clinical_metrics(our, labels)
        syb_m = _compute_clinical_metrics(syb, labels)
        if our_m is None or syb_m is None:
            continue
        row = {"Year": year, "N": len(labels), "N_cancer": int(labels.sum())}
        for k, v in our_m.items():
            row[f"Lungevity_{k}"] = v
        for k, v in syb_m.items():
            row[f"Sybil_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def compute_survival_dca_comparison(comp_df, followup, n_thresholds=200):
    """Survival-adapted DCA with IPCW for both models on the matched set."""
    times = comp_df["time_at_event"].values.astype(float)
    events = comp_df["gold"].values.astype(int)
    our_scores = comp_df["risk_score"].values.astype(float)
    sybil_col = f"sybil_risk_{followup + 1}"
    syb_scores = comp_df[sybil_col].values.astype(float)
    n = len(times)

    if events.sum() == 0:
        return pd.DataFrame()

    tp_weights, fp_weight, is_tp, is_fp, prevalence_t = \
        _ipcw_weights(times, events, followup)

    thresholds = np.linspace(0.005, 0.50, n_thresholds)

    rows = []
    for pt in thresholds:
        w = pt / (1 - pt)

        pos_our = our_scores >= pt
        nb_our = float(tp_weights[pos_our & is_tp].sum()) / n - \
                 float((pos_our & is_fp).sum()) * fp_weight / n * w

        pos_syb = syb_scores >= pt
        nb_syb = float(tp_weights[pos_syb & is_tp].sum()) / n - \
                 float((pos_syb & is_fp).sum()) * fp_weight / n * w

        nb_all = prevalence_t - (1 - prevalence_t) * w
        rows.append(dict(
            threshold=pt,
            nb_lungevity=nb_our,
            nb_sybil=nb_syb,
            nb_all=nb_all,
            nb_none=0.0,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Printing — clinical evaluation metrics
# ---------------------------------------------------------------------------

def print_clinical_metrics(cm_df):
    print("\nCLINICAL EVALUATION METRICS  (Paper Framework)")
    _hr("=")
    print("  §2  Partial AUC (sens ≥ 0.90) & Specificity at Fixed Sensitivity")
    print("  §3  Brier Score (decomposed) & Log Loss")
    _hr("·")
    print(f"  {'Year':<5} {'pAUC≥90':>8} {'Sp@90%':>8} {'Sp@95%':>8}"
          f"  {'Brier':>7} {'LogLoss':>8}"
          f"  {'Reliab':>7} {'Resol':>7} {'Uncert':>7}"
          f"  {'N':>6}")
    _hr("·")
    for _, r in cm_df.iterrows():
        def _f(v):
            return f"{v:.4f}" if not pd.isna(v) else "  N/A "
        print(f"  {int(r['Year']):<5} {_f(r['pAUC_sens90']):>8} "
              f"{_f(r['spec_at_sens90']):>8} {_f(r['spec_at_sens95']):>8}"
              f"  {_f(r['brier_score']):>7} {_f(r['log_loss_val']):>8}"
              f"  {_f(r['brier_reliability']):>7} {_f(r['brier_resolution']):>7} "
              f"{_f(r['brier_uncertainty']):>7}"
              f"  {int(r['N']):>6}")
    _hr("=")


def print_clinical_metrics_comparison(cm_df):
    print("\nCLINICAL METRICS COMPARISON  (matched set)")
    _hr("=")
    print("  §2  Partial AUC & Specificity at Fixed Sensitivity:")
    _hr("·")
    print(f"  {'Year':<5} {'Lung pAUC':>10} {'Syb pAUC':>10}"
          f"  {'Lung Sp@90':>11} {'Syb Sp@90':>11}"
          f"  {'Lung Sp@95':>11} {'Syb Sp@95':>11}")
    _hr("·")
    for _, r in cm_df.iterrows():
        def _f(v):
            return f"{v:.4f}" if not pd.isna(v) else "  N/A "
        print(f"  {int(r['Year']):<5} "
              f"{_f(r['Lungevity_pAUC_sens90']):>10} "
              f"{_f(r['Sybil_pAUC_sens90']):>10}"
              f"  {_f(r['Lungevity_spec_at_sens90']):>11} "
              f"{_f(r['Sybil_spec_at_sens90']):>11}"
              f"  {_f(r['Lungevity_spec_at_sens95']):>11} "
              f"{_f(r['Sybil_spec_at_sens95']):>11}")

    print()
    print("  §3  Proper Scoring Rules:")
    _hr("·")
    print(f"  {'Year':<5} {'Lung Brier':>11} {'Syb Brier':>11}"
          f"  {'Lung LogL':>10} {'Syb LogL':>10}")
    _hr("·")
    for _, r in cm_df.iterrows():
        def _f(v):
            return f"{v:.4f}" if not pd.isna(v) else "  N/A "
        print(f"  {int(r['Year']):<5} "
              f"{_f(r['Lungevity_brier_score']):>11} "
              f"{_f(r['Sybil_brier_score']):>11}"
              f"  {_f(r['Lungevity_log_loss_val']):>10} "
              f"{_f(r['Sybil_log_loss_val']):>10}")
    _hr("=")


# ---------------------------------------------------------------------------
# Plots — clinical evaluation metrics
# ---------------------------------------------------------------------------

def plot_pauc_roc(df, followup, out):
    """ROC with shaded pAUC region (sensitivity >= 0.90) and operating points."""
    year = followup + 1
    scores, labels = _filter_for_year(df, followup)
    if labels.sum() == 0:
        return

    fpr, tpr, _ = roc_curve(labels, scores)
    auc_val = roc_auc_score(labels, scores)
    pauc = _compute_pauc_high_sensitivity(fpr, tpr, 0.90)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"Full AUC = {auc_val:.3f}")

    mask = tpr >= 0.90
    if mask.sum() >= 2:
        fpr_r, tpr_r = fpr[mask], tpr[mask]
        ax.fill_between(fpr_r, 0.90, tpr_r, alpha=0.3, color="orange",
                        label=f"pAUC (sens≥0.90) = {pauc:.4f}")
        ax.axhline(0.90, color="orange", ls="--", lw=1, alpha=0.7)

    spec_90 = _specificity_at_sensitivity(fpr, tpr, 0.90)
    spec_95 = _specificity_at_sensitivity(fpr, tpr, 0.95)
    if not np.isnan(spec_90):
        ax.scatter([1 - spec_90], [0.90], color="red", s=100, zorder=5,
                   marker="o", label=f"Spec@90%sens = {spec_90:.3f}")
    if not np.isnan(spec_95):
        ax.scatter([1 - spec_95], [0.95], color="darkred", s=100, zorder=5,
                   marker="^", label=f"Spec@95%sens = {spec_95:.3f}")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(f"{year}-Year Risk: ROC with Partial AUC (sens ≥ 0.90)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(out, f"pauc_roc_year{year}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_calibration(df, followup, out, n_bins=10):
    """Reliability diagram with prediction histogram (paper §3)."""
    year = followup + 1
    scores, labels = _filter_for_year(df, followup)
    if labels.sum() == 0:
        return

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, obs_freq, bin_counts = [], [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (scores >= lo) & (scores < hi) if i < n_bins - 1 \
            else (scores >= lo) & (scores <= hi)
        n_k = mask.sum()
        if n_k == 0:
            continue
        bin_centers.append(float(np.mean(scores[mask])))
        obs_freq.append(float(np.mean(labels[mask])))
        bin_counts.append(n_k)

    bc = np.array(bin_centers)
    of = np.array(obs_freq)

    brier = float(brier_score_loss(labels, np.clip(scores, 1e-7, 1 - 1e-7)))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 7),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax1.plot(bc, of, "bo-", lw=2, ms=8, label=f"Model (Brier={brier:.4f})")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title(f"{year}-Year Risk: Calibration (Reliability Diagram)")
    ax1.legend(loc="upper left")
    ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.02)

    ax2.hist(scores, bins=bins, color="steelblue", alpha=0.7, edgecolor="navy")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")

    fig.tight_layout()
    path = os.path.join(out, f"calibration_year{year}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_dca(dca_df, year, out, suffix=""):
    """Decision Curve Analysis plot (paper §4)."""
    if dca_df.empty:
        return

    tag = f" ({suffix.upper()})" if suffix else ""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dca_df["threshold"], dca_df["nb_model"], "b-", lw=2, label="Model")
    ax.plot(dca_df["threshold"], dca_df["nb_all"], "r--", lw=1.5, label="Treat all")
    ax.axhline(0, color="grey", ls=":", lw=1, label="Treat none")

    ax.axvspan(0.01, 0.10, alpha=0.08, color="green",
               label="Clinical range (1–10%)")

    ax.set_xlabel("Threshold probability ($p_t$)")
    ax.set_ylabel("Net Benefit")
    ax.set_title(f"{year}-Year Risk: Survival DCA{tag}")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 0.50)
    ymin = min(dca_df["nb_model"].min(), dca_df["nb_all"].min(), -0.01)
    ax.set_ylim(ymin - 0.01, None)
    fig.tight_layout()
    sfx = f"_{suffix}" if suffix else ""
    path = os.path.join(out, f"dca{sfx}_year{year}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_dca_comparison(dca_df, year, out, suffix=""):
    """DCA overlay for Lungevity vs Sybil (paper §4)."""
    if dca_df.empty:
        return

    tag = f" ({suffix})" if suffix else ""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dca_df["threshold"], dca_df["nb_lungevity"], "b-", lw=2,
            label="Lungevity")
    ax.plot(dca_df["threshold"], dca_df["nb_sybil"], "r-", lw=2,
            label="Sybil")
    ax.plot(dca_df["threshold"], dca_df["nb_all"], "k--", lw=1.5,
            label="Treat all")
    ax.axhline(0, color="grey", ls=":", lw=1, label="Treat none")

    ax.axvspan(0.01, 0.10, alpha=0.08, color="green",
               label="Clinical range (1–10%)")

    ax.set_xlabel("Threshold probability ($p_t$)")
    ax.set_ylabel("Net Benefit")
    ax.set_title(f"{year}-Year Risk: Survival DCA{tag} — Lungevity vs Sybil")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 0.50)
    ymin = min(dca_df[["nb_lungevity", "nb_sybil", "nb_all"]].min().min(),
               -0.01)
    ax.set_ylim(ymin - 0.01, None)
    fig.tight_layout()
    sfx = f"_{suffix}" if suffix else ""
    path = os.path.join(out, f"comparison_dca{sfx}_year{year}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")




def plot_calibration_comparison(comp_df, followup, out, n_bins=10):
    """Calibration curves for both models (raw scores)."""
    year = followup + 1
    our, syb, labels = _filter_for_year_comparison(comp_df, followup)
    if labels.sum() == 0:
        return

    s_our = np.clip(our, 1e-7, 1 - 1e-7)
    s_syb = np.clip(syb, 1e-7, 1 - 1e-7)
    brier_our = brier_score_loss(labels, s_our)
    brier_syb = brier_score_loss(labels, s_syb)

    def _bin(scores):
        bins = np.linspace(0, 1, n_bins + 1)
        centers, freqs = [], []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (scores >= lo) & (scores < hi) if i < n_bins - 1 \
                else (scores >= lo) & (scores <= hi)
            if mask.sum() < 2:
                continue
            centers.append(float(np.mean(scores[mask])))
            freqs.append(float(np.mean(labels[mask])))
        return np.array(centers), np.array(freqs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    c, f = _bin(our)
    ax.plot(c, f, "bo-", lw=2, ms=7,
            label=f"Lungevity (Brier={brier_our:.4f})")
    c, f = _bin(syb)
    ax.plot(c, f, "rs-", lw=2, ms=7,
            label=f"Sybil (Brier={brier_syb:.4f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"{year}-Year Risk: Calibration Comparison")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    path = os.path.join(out, f"calibration_comparison_year{year}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# TRIPOD-style calibration belt (smoothed curve with bootstrap CI band)
# ===========================================================================

def _loess_smooth(scores, labels, n_grid=100, frac=0.3):
    """Local linear smoothing (LOWESS-style) for calibration curve.

    Returns (grid_x, smoothed_y). Uses a Gaussian kernel with bandwidth
    proportional to the interquartile range.
    """
    order = np.argsort(scores)
    s_sorted = scores[order]
    l_sorted = labels[order].astype(float)
    if len(s_sorted) < 10:
        return np.array([]), np.array([])

    xmin, xmax = s_sorted[0], s_sorted[-1]
    grid = np.linspace(xmin, xmax, n_grid)
    bandwidth = max(frac * (xmax - xmin), 1e-3)

    smoothed = np.zeros(n_grid)
    for i, x in enumerate(grid):
        w = np.exp(-0.5 * ((s_sorted - x) / bandwidth) ** 2)
        if w.sum() < 1e-9:
            smoothed[i] = np.nan
        else:
            smoothed[i] = np.sum(w * l_sorted) / w.sum()
    return grid, smoothed


def plot_calibration_belt(scores, labels, year, model_name, out,
                          n_bootstrap=200, n_grid=100):
    """TRIPOD-style calibration plot: smoothed curve + bootstrap CI belt.

    Plots the non-parametric local calibration curve (observed vs predicted
    risk) with a 95% bootstrap confidence band. Overlays the perfect
    calibration diagonal and a histogram of prediction density at the bottom.
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    # Point estimate
    grid, smooth = _loess_smooth(scores, labels, n_grid=n_grid)
    if len(grid) == 0:
        return

    # Bootstrap CI band
    rng = np.random.default_rng(RNG_SEED)
    n = len(scores)
    boot_curves = np.zeros((n_bootstrap, n_grid))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bs_grid, bs_smooth = _loess_smooth(scores[idx], labels[idx],
                                           n_grid=n_grid)
        if len(bs_grid) == 0:
            boot_curves[b] = smooth
        else:
            # Reproject to original grid via linear interpolation
            boot_curves[b] = np.interp(grid, bs_grid, bs_smooth)
    lo = np.nanpercentile(boot_curves, 2.5, axis=0)
    hi = np.nanpercentile(boot_curves, 97.5, axis=0)

    brier = float(brier_score_loss(labels, np.clip(scores, 1e-7, 1 - 1e-7)))
    prevalence = float(labels.mean())

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(6.5, 7),
        gridspec_kw={"height_ratios": [3.5, 1]}, sharex=True)

    ax_main.fill_between(grid, lo, hi, alpha=0.25, color="steelblue",
                         label="95% bootstrap CI")
    ax_main.plot(grid, smooth, "b-", lw=2.2,
                 label=f"{model_name} calibration")
    ax_main.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax_main.axhline(prevalence, color="grey", ls=":", lw=1,
                    label=f"Base rate = {prevalence:.3f}")

    ax_main.set_ylabel("Observed event probability")
    ax_main.set_title(
        f"{year}-Year Risk: TRIPOD Calibration Belt — {model_name} "
        f"(Brier={brier:.4f})")
    ax_main.legend(loc="upper left", fontsize=8)
    ax_main.set_xlim(-0.01, max(0.5, float(grid.max()) * 1.05))
    ax_main.set_ylim(-0.01, max(1.02, float(np.nanmax(hi)) * 1.1))

    ax_hist.hist(scores, bins=40, color="steelblue", alpha=0.7,
                 edgecolor="navy")
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Count")

    fig.tight_layout()
    path = os.path.join(
        out, f"calibration_belt_{model_name.lower()}_year{year}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_calibration_belt_comparison(comp_df, followup, out,
                                      n_bootstrap=200, n_grid=100):
    """Overlay calibration belts for Lungevity and Sybil on the matched set."""
    year = followup + 1
    our, syb, labels = _filter_for_year_comparison(comp_df, followup)
    if labels.sum() == 0:
        return

    def _belt(scores):
        grid, smooth = _loess_smooth(scores, labels, n_grid=n_grid)
        if len(grid) == 0:
            return None, None, None
        rng = np.random.default_rng(RNG_SEED)
        n = len(scores)
        boot = np.zeros((n_bootstrap, n_grid))
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            bg, bs = _loess_smooth(scores[idx], labels[idx], n_grid=n_grid)
            boot[b] = np.interp(grid, bg, bs) if len(bg) else smooth
        lo = np.nanpercentile(boot, 2.5, axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
        return grid, smooth, (lo, hi)

    our_g, our_s, our_ci = _belt(our)
    syb_g, syb_s, syb_ci = _belt(syb)
    brier_our = brier_score_loss(labels, np.clip(our, 1e-7, 1 - 1e-7))
    brier_syb = brier_score_loss(labels, np.clip(syb, 1e-7, 1 - 1e-7))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    if our_g is not None:
        ax.fill_between(our_g, our_ci[0], our_ci[1], alpha=0.2, color="blue")
        ax.plot(our_g, our_s, "b-", lw=2.2,
                label=f"Lungevity (Brier={brier_our:.4f})")
    if syb_g is not None:
        ax.fill_between(syb_g, syb_ci[0], syb_ci[1], alpha=0.2, color="red")
        ax.plot(syb_g, syb_s, "r-", lw=2.2,
                label=f"Sybil (Brier={brier_syb:.4f})")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed event probability")
    ax.set_title(f"{year}-Year Risk: Calibration Belt — Lungevity vs Sybil")
    ax.legend(loc="upper left", fontsize=9)
    xmax = max(0.5,
               float(np.max([our_g.max() if our_g is not None else 0,
                             syb_g.max() if syb_g is not None else 0])) * 1.05)
    ax.set_xlim(-0.01, xmax)
    ax.set_ylim(-0.01, 1.02)
    fig.tight_layout()
    path = os.path.join(out, f"calibration_belt_comparison_year{year}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Stratified Kaplan-Meier curves by model-predicted risk tier
# ===========================================================================

def plot_km_by_risk_tier(df, out, model_name="Lungevity",
                         score_col="risk_score",
                         tiers=(("Low (bottom 50%)", 0.0, 0.50),
                                ("Middle (50–90%)", 0.50, 0.90),
                                ("High (top 10%)", 0.90, 1.01))):
    """Kaplan-Meier curves stratified by model-predicted risk percentile.

    Divides the evaluation cohort into risk tiers based on score percentiles
    and plots each tier's observed survival curve over the full 6-year
    follow-up. This visualizes whether the model's ranking translates into
    distinct survival trajectories — the clinical interpretation of the
    C-index as a discriminatory curve.
    """
    times = df["time_at_event"].values.astype(float)
    events = df["gold"].values.astype(int)
    scores = df[score_col].values.astype(float)

    pct_bounds = np.quantile(
        scores, [t[1] for t in tiers[:-1]] + [0.0, 1.0])
    # Use np.quantile per-tier explicitly
    low_b = np.quantile(scores, [t[1] for t in tiers])
    low_b = np.concatenate([[scores.min() - 1e-6], low_b])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # green, orange, red

    tier_stats = []
    for i, (name, plo, phi) in enumerate(tiers):
        thr_lo = np.quantile(scores, plo) if plo > 0 else -np.inf
        thr_hi = np.quantile(scores, phi) if phi < 1 else np.inf
        if i == 0:
            mask = scores <= thr_hi
        elif i == len(tiers) - 1:
            mask = scores > thr_lo
        else:
            mask = (scores > thr_lo) & (scores <= thr_hi)

        if mask.sum() < 5:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(times[mask], events[mask], label=name)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[i % 3])

        n = int(mask.sum())
        n_events = int(events[mask].sum())
        tier_stats.append((name, n, n_events))

    ax.set_xlabel("Years from screening")
    ax.set_ylabel("Cancer-free probability (KM)")
    ax.set_title(
        f"Kaplan-Meier by {model_name} Risk Tier\n"
        + "  |  ".join([f"{n}: N={tn} ({te} events)"
                        for n, tn, te in tier_stats]))
    ax.set_ylim(0.80, 1.005)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()

    path = os.path.join(out, f"km_by_risk_tier_{model_name.lower()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_km_by_risk_tier_comparison(comp_df, out):
    """Side-by-side KM risk-tier plots for Lungevity and Sybil."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    tiers = (("Low (bot 50%)", 0.0, 0.50),
             ("Mid (50–90%)", 0.50, 0.90),
             ("High (top 10%)", 0.90, 1.01))
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    times = comp_df["time_at_event"].values.astype(float)
    events = comp_df["gold"].values.astype(int)

    # Use year-6 Sybil risk for stratification (longest-horizon signal)
    panels = [
        ("Lungevity", comp_df["risk_score"].values.astype(float), axes[0]),
        ("Sybil (yr-6)", comp_df["sybil_risk_6"].values.astype(float), axes[1]),
    ]

    for model_name, scores, ax in panels:
        for i, (name, plo, phi) in enumerate(tiers):
            thr_lo = np.quantile(scores, plo) if plo > 0 else -np.inf
            thr_hi = np.quantile(scores, phi) if phi < 1 else np.inf
            if i == 0:
                mask = scores <= thr_hi
            elif i == len(tiers) - 1:
                mask = scores > thr_lo
            else:
                mask = (scores > thr_lo) & (scores <= thr_hi)
            if mask.sum() < 5:
                continue
            kmf = KaplanMeierFitter()
            n = int(mask.sum()); ne = int(events[mask].sum())
            kmf.fit(times[mask], events[mask],
                    label=f"{name} (N={n}, e={ne})")
            kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[i % 3])
        ax.set_xlabel("Years from screening")
        ax.set_title(f"KM by {model_name} Risk Tier")
        ax.set_ylim(0.80, 1.005)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)

    axes[0].set_ylabel("Cancer-free probability (KM)")
    fig.suptitle("Risk Stratification: KM Survival by Predicted Risk Tier",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(out, "km_by_risk_tier_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load ----
    print("Loading predictions …")
    preds = load_predictions(PREDS_PATH)
    print("Loading participant data …")
    part  = load_participant(PARTICIPANT_CSV)
    print("Loading lung cancer table …")
    lc    = load_lung_cancer(LUNG_CANCER_CSV)

    print("Merging …")
    df = merge_data(preds, part, lc)
    print(f"  {len(df)} scan records, {df['pid'].nunique()} unique patients, "
          f"{int(df['gold'].sum())} cancer scans")

    # ---- Sybil 1.5 matching via accession CSV (3 strategies) ----
    print("\nBuilding series match maps (accession, exam, pid+yr) …")
    acc_to_series, exam_to_series, pidyr_to_series = \
        load_series_match_maps(SERIES_MATCH_CSV, preds)
    print(f"  Strategy 1 (accession):     {len(acc_to_series)} entries")
    print(f"  Strategy 2 (exact_exam):    {len(exam_to_series)} entries")
    print(f"  Strategy 3 (pid_yr fallback): {len(pidyr_to_series)} entries")

    print("Loading Sybil predictions (3 seeds) …")
    _s0 = pd.read_csv(os.path.join(SYBIL_DIR, "seed0", "checkpoints", "3", "test.csv"))
    _n_total = len(_s0)

    sybil_df = load_sybil_predictions(SYBIL_DIR, SYBIL_SEEDS,
                                      acc_to_series, exam_to_series,
                                      pidyr_to_series)
    comp_df = build_comparison_df(df, sybil_df)

    print("\n" + "=" * 70)
    print(f"  SERIES MATCHED BETWEEN SYBIL AND LUNGEVITY: {len(comp_df)}")
    print("=" * 70)
    print(f"  Sybil seed-0 total rows:    {_n_total}")
    print(f"  Sybil after ensembling:     {len(sybil_df)}")
    print(f"  Dropped (unmatched):        {_n_total - len(sybil_df)}")
    print(f"  Lungevity predictions:      {len(df)}")
    print(f"  Head-to-head (both models): {len(comp_df)}  "
          f"(Lungevity-only: {len(df) - len(comp_df)}, "
          f"Sybil-only: {len(sybil_df) - len(comp_df)})")
    print(f"  Matched patients: {comp_df['pid'].nunique()}, "
          f"cancer scans: {int(comp_df['gold'].sum())}")
    print("=" * 70 + "\n")

    # ---- Population table ----
    pop_df = compute_population_table(df)
    print_population_table(pop_df)
    pop_df.to_csv(os.path.join(OUTPUT_DIR, "population_table.csv"), index=False)

    # ---- AUC by year ----
    print("\nComputing AUC by year …")
    yearly_df = compute_yearly_aucs(df)
    print_yearly_aucs(yearly_df)
    yearly_df.to_csv(os.path.join(OUTPUT_DIR, "auc_by_year.csv"), index=False)

    # ---- Subgroup AUC ----
    sg_all = {}
    for year in range(1, MAX_FOLLOWUP + 1):
        print(f"\nComputing subgroup AUC  (year {year}) …")
        sg_df = compute_subgroup_aucs(df, year=year)
        print_subgroup_table(sg_df, year=year)
        sg_df.to_csv(os.path.join(OUTPUT_DIR, f"subgroup_auc_year{year}.csv"), index=False)
        sg_all[year] = sg_df

    # ---- LungRADS comparison ----
    print("\nComputing LungRADS-like comparison …")
    lr_results = []
    for year in range(1, MAX_FOLLOWUP + 1):
        lr = compute_lungrads_comparison(df, year=year)
        lr_results.append(lr)
    print_lungrads_table(lr_results)

    # Save LungRADS summary (without arrays)
    lr_summary = [
        {k: v for k, v in r.items() if k not in ("fpr", "tpr")}
        for r in lr_results
    ]
    pd.DataFrame(lr_summary).to_csv(
        os.path.join(OUTPUT_DIR, "lungrads_comparison.csv"), index=False
    )

    # ---- Plots ----
    print("\nGenerating plots …")
    plot_auc_by_year(yearly_df, OUTPUT_DIR)
    plot_roc_all_years(df, OUTPUT_DIR)
    for lr in lr_results:
        plot_roc_with_lungrads(lr, OUTPUT_DIR)
    for year, sg_df in sg_all.items():
        plot_subgroup_forest(sg_df, year, OUTPUT_DIR)

    # ==================================================================
    # Sybil comparison (head-to-head on matched set)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SYBIL COMPARISON")
    print("=" * 70)

    # ---- Yearly AUC comparison ----
    print("\nComputing head-to-head AUC by year …")
    comp_yearly = compute_yearly_aucs_comparison(comp_df)
    print_comparison_yearly(comp_yearly)
    comp_yearly.to_csv(
        os.path.join(OUTPUT_DIR, "comparison_auc_by_year.csv"), index=False)

    # ---- Subgroup comparison (years 1, 3, 6) ----
    for year in range(1, MAX_FOLLOWUP + 1):
        print(f"\nComputing subgroup comparison (year {year}) …")
        comp_sg = compute_subgroup_aucs_comparison(comp_df, year=year)
        print_comparison_subgroup(comp_sg, year=year)
        comp_sg.to_csv(
            os.path.join(OUTPUT_DIR, f"comparison_subgroup_year{year}.csv"),
            index=False)

    # ---- Comparison plots ----
    print("\nGenerating comparison plots …")
    plot_comparison_auc_by_year(comp_yearly, OUTPUT_DIR)
    for year in range(1, MAX_FOLLOWUP + 1):
        plot_comparison_roc(comp_df, year, OUTPUT_DIR)

    # ==================================================================
    # Clinical Evaluation Metrics (Paper Framework)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  CLINICAL EVALUATION METRICS  (Paper Framework)")
    print("=" * 70)

    # ---- §2 + §3: Clinical metrics by year (standalone) ----
    print("\nComputing clinical evaluation metrics …")
    cm_df = compute_clinical_metrics_by_year(df)
    print_clinical_metrics(cm_df)
    cm_df.to_csv(os.path.join(OUTPUT_DIR, "clinical_metrics_by_year.csv"),
                 index=False)

    # ---- §4: Survival DCA (standalone) — both IPCW and complete-case ----
    for year in range(1, MAX_FOLLOWUP + 1):
        print(f"\nComputing Survival DCA (year {year}) …")
        dca_ipcw = compute_survival_dca(df, year - 1)
        dca_ipcw.to_csv(os.path.join(OUTPUT_DIR, f"dca_ipcw_year{year}.csv"),
                        index=False)
        plot_dca(dca_ipcw, year, OUTPUT_DIR, suffix="ipcw")

        dca_cc = compute_dca_complete_case(df, year - 1)
        dca_cc.to_csv(os.path.join(OUTPUT_DIR, f"dca_cc_year{year}.csv"),
                      index=False)
        plot_dca(dca_cc, year, OUTPUT_DIR, suffix="cc")

    # ---- Clinical evaluation plots ----
    print("\nGenerating clinical evaluation plots …")
    for year in range(1, MAX_FOLLOWUP + 1):
        plot_pauc_roc(df, year - 1, OUTPUT_DIR)
        plot_calibration(df, year - 1, OUTPUT_DIR)

    # ---- §2 + §3: Comparison clinical metrics (matched set) ----
    print("\nComputing clinical metrics comparison …")
    cm_comp = compute_clinical_metrics_comparison_by_year(comp_df)
    print_clinical_metrics_comparison(cm_comp)
    cm_comp.to_csv(
        os.path.join(OUTPUT_DIR, "comparison_clinical_metrics.csv"),
        index=False)

    # ---- §4: Comparison DCA (matched set) — both IPCW and complete-case ----
    for year in range(1, MAX_FOLLOWUP + 1):
        print(f"\nComputing comparison DCA (year {year}) …")
        dca_ipcw = compute_survival_dca_comparison(comp_df, year - 1)
        dca_ipcw.to_csv(
            os.path.join(OUTPUT_DIR, f"comparison_dca_ipcw_year{year}.csv"),
            index=False)
        plot_dca_comparison(dca_ipcw, year, OUTPUT_DIR, suffix="ipcw")

        dca_cc = compute_dca_complete_case_comparison(comp_df, year - 1)
        dca_cc.to_csv(
            os.path.join(OUTPUT_DIR, f"comparison_dca_cc_year{year}.csv"),
            index=False)
        plot_dca_comparison(dca_cc, year, OUTPUT_DIR, suffix="cc")

    # ---- Calibration comparison plots (raw scores, both models) ----
    print("\nGenerating calibration comparison plots …")
    print("  NOTE: Sybil outputs are sigmoid(logit), not calibrated probabilities.")
    print("        Brier/LogLoss differences reflect this — see interpretation.")
    for year in range(1, MAX_FOLLOWUP + 1):
        plot_calibration_comparison(comp_df, year - 1, OUTPUT_DIR)

    # ---- TRIPOD-style calibration belt plots ----
    print("\nGenerating TRIPOD calibration belt plots …")
    for year in range(1, MAX_FOLLOWUP + 1):
        scores, labels = _filter_for_year(df, year - 1)
        plot_calibration_belt(scores, labels, year, "Lungevity", OUTPUT_DIR)

        our, syb, comp_labels = _filter_for_year_comparison(comp_df, year - 1)
        plot_calibration_belt(syb, comp_labels, year, "Sybil", OUTPUT_DIR)
        plot_calibration_belt_comparison(comp_df, year - 1, OUTPUT_DIR)

    # ---- Stratified Kaplan-Meier curves by model risk tier ----
    print("\nGenerating stratified KM curves …")
    plot_km_by_risk_tier(df, OUTPUT_DIR, model_name="Lungevity")
    plot_km_by_risk_tier_comparison(comp_df, OUTPUT_DIR)

    print(f"\nDone.  All outputs in:  {OUTPUT_DIR}/")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
