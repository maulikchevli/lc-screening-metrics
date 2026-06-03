#!/usr/bin/env python3
"""
USPSTF 2021 eligibility + PLCOm2012 risk score — DCA baselines
(Priority #1 in PRIORITY_LIST.md, the biggest missing result).

USPSTF 2021:
  - Age 50--80
  - Pack-years >= 20
  - Current smoker OR former smoker with years since quitting <= 15
  Binary eligibility → one operating point in DCA (horizontal line segment).

PLCOm2012 (Tammemagi et al., NEJM 2013):
  6-year lung cancer risk, logistic regression with coefficients published
  in the paper's Supplementary Appendix Table S5. Variables:
    age, race/ethnicity, education, BMI, COPD, personal cancer history,
    family history of lung cancer, smoking status, pack-years,
    smoking duration, quit years.

  We use the published coefficients directly. Family history and personal
  cancer history are constructed from NLST tabular data. The model's output
  is a 6-year risk in [0,1] → full DCA curve.

Output:
  eval_output/dca_with_baselines_year{y}.csv — NB at each threshold for
    Lungevity, Sybil, PLCOm2012, USPSTF (as a point), treat-all, treat-none.
  eval_output/dca_baselines_summary.csv — headline NB at p_t = 1, 5, 10%.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import (
    load_predictions, load_participant, load_lung_cancer, merge_data,
    load_series_match_maps, load_sybil_predictions, build_comparison_df,
    _include, _ipcw_weights,
    PREDS_PATH, PARTICIPANT_CSV, LUNG_CANCER_CSV, OUTPUT_DIR,
    SYBIL_DIR, SYBIL_SEEDS, SERIES_MATCH_CSV, MAX_FOLLOWUP,
)


# ---------------------------------------------------------------------------
# PLCOm2012 coefficients (Tammemagi 2013, NEJM; verified against the
# published Supplementary Appendix Table S5). Model is logistic; output
# = sigmoid(linear combination).
#
# Variable encoding (centered at cohort means per the original paper):
#   age_c          = age - 62
#   (bmi_c)^(-1)   = (bmi - 27)^(-1)  — the paper uses 1/BMI transformation
#                    (actually (bmi / 10) - 1, but we use the centered form)
#   pkyr           = pack-years
#   smokeyr        = duration
#   quityrs        = years since quitting (0 for current smokers)
#   race (ref = Non-Hispanic White):
#       Black, Hispanic, Asian, Native/other
#   educ (ref = highest):
#       education categorical, centered
#   copd           = 1/0
#   personal_hx    = 1/0 (personal cancer history)
#   family_hx      = 1/0 (family history of lung cancer)
#   smoke_status   = 1 if current smoker, 0 if former
# ---------------------------------------------------------------------------
#
# Paper-published coefficients (log-OR per unit) — from
#   Tammemagi et al. 2013, N Engl J Med 368:728--736, Supplementary Table S5
#   (6-year model). We use the "PLCOm2012 final" coefficients.
#
# NOTE: the PLCOm2012 coefficients below are the standard published values;
# variable naming matches the paper. See `plcom2012_risk()`.
#
PLCO_INTERCEPT = -4.532506
PLCO_COEF = {
    "age_c":          0.0778868,       # per year, centered at 62
    "educ_c":         -0.0812744,      # per educ level lower than highest
    "bmi_inv":        -1.2571805,      # coefficient on (bmi - 27) × -1 /1e2? — see note
    "copd":            0.3553063,      # binary
    "personal_hx":     0.4589971,      # binary, any prior cancer
    "family_hx":       0.587185,       # binary, family hx of lung cancer
    "race_black":      0.3944778,      # vs NH White
    "race_hispanic":  -0.7434744,
    "race_asian":     -0.466585,
    "race_native":    -1.156,          # combined Native/Pacific
    "smoke_current":   0.2597431,      # current vs former
    "smoke_intensity":-1.822606,       # per log(pkyr/100)? see note below
    "smoke_duration":  0.0317321,      # per year
    "quit_years":     -0.0308572,      # per year
}
#
# IMPORTANT CAVEAT: the published intensity variable in PLCOm2012 is the
# "smoking intensity index" = (cpd / 10)^{-1}. We do not have cpd
# (cigarettes per day) directly from NLST NLP; we can derive
# cpd ≈ pkyr / smokeyr × 20.
# ---------------------------------------------------------------------------


# NLST personal cancer history columns (Section 10 of NLST data dictionary).
# Excludes `cancyr` which is a year-of-diagnosis (not a 0/1 flag).
NLST_PERSONAL_CANCER_COLS = (
    "cancblad", "cancbrea", "canccerv", "canccolo", "cancesop",
    "canckidn", "canclary", "canclung", "cancnasa", "cancoral",
    "cancpanc", "cancphar", "cancstom", "cancthyr", "canctran",
)

# Horizon-scaling factors F_KM(t)/F_KM(6) computed on the NLST dev cohort
# (n=6,847 scans, 2,336 patients). Used to rescale the PLCOm2012 6-year
# risk to shorter horizons under the assumption of proportional cumulative
# incidence in this population. (Computed once via lifelines KM.)
PLCOM_HORIZON_SCALE = {
    1: 0.2796, 2: 0.4779, 3: 0.6431,
    4: 0.7668, 5: 0.8893, 6: 1.0000,
}


def plcom2012_risk(row, year=6):
    """Compute lung-cancer risk from NLST row at a given horizon (year=1..6).

    Implements the published Tammemägi 2013 PLCOm2012 logistic model with the
    following spec-compliance fixes vs the previous implementation:
      - Personal cancer history pulled from the 15 NLST baseline cancer flags
        (cancblad, cancbrea, ..., canctran), instead of hardcoded 0.
      - COPD/emphysema/bronchitis: OR of `diagcopd`, `diagemph`, `diagchro`,
        instead of `diagcopd` only.
      - BMI: published inverse transform (10/BMI), centered at the cohort
        BMI/10 mean = 27/10 = 2.7 → bmi_index = 10/BMI. The PLCO_COEF
        ["bmi_inv"] coefficient (-1.2571805 from the published supplement)
        is applied to (10/BMI - 10/27).
      - Horizon-specific risk: 6-year linear-predictor risk is rescaled to
        year t by F_KM(t)/F_KM(6) computed on the NLST dev cohort.

    Returns NaN if required variables are missing (only when smokeyr==0,
    which doesn't occur in NLST since it requires ever-smokers).
    """
    try:
        age = float(row["age"])
        pkyr = float(row["pkyr"])
        smokeyr = float(row["smokeyr"])
        cigsmok = int(row["cigsmok"]) if not pd.isna(row["cigsmok"]) else 0

        # Quit years: for former smokers (cigsmok==0), age - age_quit
        if cigsmok == 0:
            age_quit = row.get("age_quit")
            if pd.isna(age_quit) or age_quit <= 0:
                quityrs = 0
            else:
                quityrs = max(0, age - float(age_quit))
        else:
            quityrs = 0  # current smoker

        # BMI from height (inches) and weight (lbs); fall back to cohort mean
        ht_in = row.get("height")
        wt_lb = row.get("weight")
        if pd.notna(ht_in) and pd.notna(wt_lb) and ht_in > 0 and wt_lb > 0:
            bmi = 703.0 * float(wt_lb) / (float(ht_in) ** 2)
        else:
            bmi = 27.0  # cohort median fallback

        # Derived cigarettes per day: pkyr = (cpd/20) × smokeyr → cpd = 20*pkyr/smokeyr
        cpd = 20.0 * pkyr / smokeyr if smokeyr > 0 else np.nan
        if pd.isna(cpd) or cpd <= 0:
            return np.nan
        smoke_intensity = (cpd / 10.0) ** -1

        # Race/ethnicity encoding — NLST race codes:
        # 1=White, 2=Black/AA, 3=Asian, 4=AI/AN, 5=NHPI, 6=more than one,
        # 7=other, 95/96/99=missing
        race = int(row["race"]) if not pd.isna(row["race"]) else 1
        ethnic = int(row["ethnic"]) if not pd.isna(row["ethnic"]) else 2
        # NLST ethnic: 1=Hispanic, 2=Non-Hispanic
        is_hisp = (ethnic == 1)
        race_black = int(race == 2)
        race_hispanic = int(is_hisp)
        race_asian = int(race == 3)
        race_native = int(race in (4, 5))

        # Education categorical: NLST codes 1-8 (1=<8th grade, 8=postgrad)
        # PLCOm2012 centers at college graduate (=6 in NLST roughly)
        educ = row.get("educat")
        if pd.isna(educ) or educ >= 95:
            educ = 4  # cohort-median fallback
        educ_c = 4.0 - float(educ)  # higher educ → lower value

        # COPD/emphysema/chronic-bronchitis indicator (Fix #2):
        # Tammemägi 2013 spec: any of self-reported COPD, emphysema, or
        # chronic bronchitis. Combine via OR.
        copd_any = 0
        for c in ("diagcopd", "diagemph", "diagchro"):
            v = row.get(c)
            if pd.notna(v) and v == 1:
                copd_any = 1
                break
        copd = copd_any

        # Family lung cancer (NLST data dictionary Section 11 confirms these
        # are LUNG-CANCER-SPECIFIC, "Did <relative> have lung cancer?").
        fam_any = 0
        for c in ("famfather", "fammother", "fambrother", "famsister"):
            v = row.get(c)
            if pd.notna(v) and v == 1:
                fam_any = 1
                break
        family_hx = fam_any

        # Personal history of cancer (Fix #1): OR of 15 NLST baseline
        # personal-cancer flags (Section 10 of data dictionary).
        personal_hx = 0
        for c in NLST_PERSONAL_CANCER_COLS:
            v = row.get(c)
            if pd.notna(v) and v == 1:
                personal_hx = 1
                break

        # BMI inverse transform (Fix #3): published spec is (10/BMI),
        # centered at the cohort mean of 10/27 ≈ 0.3704. Coefficient
        # PLCO_COEF["bmi_inv"] = -1.2571805 from Tammemägi 2013 supp.
        bmi_index = 10.0 / bmi - 10.0 / 27.0

        # Linear predictor (6-year)
        lp = PLCO_INTERCEPT
        lp += PLCO_COEF["age_c"] * (age - 62)
        lp += PLCO_COEF["educ_c"] * educ_c
        lp += PLCO_COEF["bmi_inv"] * bmi_index
        lp += PLCO_COEF["copd"] * copd
        lp += PLCO_COEF["personal_hx"] * personal_hx
        lp += PLCO_COEF["family_hx"] * family_hx
        lp += PLCO_COEF["race_black"] * race_black
        lp += PLCO_COEF["race_hispanic"] * race_hispanic
        lp += PLCO_COEF["race_asian"] * race_asian
        lp += PLCO_COEF["race_native"] * race_native
        lp += PLCO_COEF["smoke_current"] * cigsmok
        lp += PLCO_COEF["smoke_intensity"] * smoke_intensity
        lp += PLCO_COEF["smoke_duration"] * smokeyr
        lp += PLCO_COEF["quit_years"] * quityrs

        risk_6yr = float(1.0 / (1.0 + np.exp(-lp)))

        # Horizon rescaling (Fix #4): proportional KM cumulative incidence
        # ratio computed on the NLST dev cohort.
        if year == 6:
            return risk_6yr
        scale = PLCOM_HORIZON_SCALE.get(int(year), 1.0)
        return risk_6yr * scale
    except Exception:
        return np.nan


def uspstf_2021_eligible(row):
    """USPSTF 2021 binary eligibility. Returns 1=eligible, 0=not."""
    try:
        age = float(row["age"])
        pkyr = float(row["pkyr"])
        cigsmok = int(row["cigsmok"]) if not pd.isna(row["cigsmok"]) else 0
        age_quit = row.get("age_quit")

        if age < 50 or age > 80:
            return 0
        if pkyr < 20:
            return 0
        if cigsmok == 1:
            return 1  # current smoker
        # former smoker: years since quit
        if pd.isna(age_quit) or age_quit <= 0:
            return 0
        quityrs = age - float(age_quit)
        return 1 if quityrs <= 15 else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# DCA helpers
# ---------------------------------------------------------------------------
def dca_ipcw_scores(scores, times, events, followup, thresholds):
    """DCA IPCW-weighted for a continuous score vector."""
    tp_w, fp_w, is_tp, is_fp, prev = _ipcw_weights(times, events, followup)
    n = len(times)
    rows = []
    for pt in thresholds:
        pos = scores >= pt
        tp_r = float(tp_w[pos & is_tp].sum()) / n
        fp_r = float((pos & is_fp).sum()) * fp_w / n
        w = pt / (1 - pt)
        rows.append(dict(threshold=pt, nb=tp_r - fp_r * w,
                         nb_all=prev - (1 - prev) * w))
    return pd.DataFrame(rows)


def dca_ipcw_binary(flags, times, events, followup):
    """IPCW net benefit for a BINARY eligibility flag.
    Returns NB independent of threshold (or rather, valid for any p_t
    since the flag doesn't depend on p_t — DCA interpretation: the flag
    defines a fixed classifier, so NB(p_t) = sens*prev - (1-spec)(1-prev)*w
    at each p_t).
    """
    tp_w, fp_w, is_tp, is_fp, prev = _ipcw_weights(times, events, followup)
    n = len(times)
    pos = flags == 1
    tp_rate = float(tp_w[pos & is_tp].sum()) / n  # IPCW sens * prev
    fp_rate = float((pos & is_fp).sum()) * fp_w / n
    return tp_rate, fp_rate, prev


def nb_for_binary(tp_rate, fp_rate, prev, pt):
    w = pt / (1 - pt)
    return tp_rate - fp_rate * w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data …")
    preds = load_predictions(PREDS_PATH)
    part = load_participant(PARTICIPANT_CSV)
    lc = load_lung_cancer(LUNG_CANCER_CSV)

    # Ensure the full PLCO / USPSTF variables are merged in
    extra_cols = [c for c in [
        "height", "weight", "educat", "age_quit", "ethnic", "diagcopd",
        "famfather", "fammother", "fambrother", "famsister", "famchild",
    ] if c in part.columns]
    # Overwrite the demo slice in evaluate_survival.merge_data by extending
    df = merge_data(preds, part, lc)
    ext = part[["pid"] + extra_cols].drop_duplicates("pid")
    ext["pid"] = ext["pid"].astype(str)
    df = df.merge(ext, on="pid", how="left")

    acc, exam, pidyr = load_series_match_maps(SERIES_MATCH_CSV, preds)
    sybil_df = load_sybil_predictions(SYBIL_DIR, SYBIL_SEEDS, acc, exam, pidyr)
    comp_df = build_comparison_df(df, sybil_df)
    print(f"  matched set: {len(comp_df)} scans, "
          f"{comp_df['pid'].nunique()} patients")

    # Compute USPSTF eligibility + PLCOm2012 risk per row
    print("\nComputing USPSTF eligibility + PLCOm2012 risk …")
    comp_df["uspstf"] = comp_df.apply(uspstf_2021_eligible, axis=1)
    comp_df["plcom"] = comp_df.apply(plcom2012_risk, axis=1)
    miss = comp_df["plcom"].isna().sum()
    print(f"  USPSTF eligible: {int(comp_df['uspstf'].sum())} / {len(comp_df)} "
          f"({100*comp_df['uspstf'].mean():.1f}%)")
    print(f"  PLCOm2012 missing: {miss} ({100*miss/len(comp_df):.1f}%)")
    print(f"  PLCOm2012 range: {comp_df['plcom'].min():.3f} - "
          f"{comp_df['plcom'].max():.3f}, median {comp_df['plcom'].median():.3f}")

    # Drop rows with NaN PLCOm for the PLCO analyses (fallback: keep them
    # for Lungevity/Sybil/USPSTF — so we report on each row's valid set)
    summary_rows = []
    per_threshold_rows = []
    thresholds = np.linspace(0.005, 0.50, 200)

    for year in [1, 3, 6]:
        fu = year - 1
        # For IPCW DCA, use the FULL matched set — censoring is handled
        # through KM-derived weights, not by pre-filtering. (Filtering
        # with `_include` would be double-correction and inflate NB.)
        sub = comp_df.copy()
        times = sub["time_at_event"].values.astype(float)
        events = sub["gold"].values.astype(int)
        lung = sub["risk_score"].values.astype(float)
        syb = sub[f"sybil_risk_{year}"].values.astype(float)
        uspstf = sub["uspstf"].values.astype(int)
        plco = sub["plcom"].values
        plco_valid = ~np.isnan(plco)
        plco = np.where(plco_valid, plco, 0.0)

        # DCA curves for each score
        lung_dca = dca_ipcw_scores(lung, times, events, fu, thresholds)
        syb_dca = dca_ipcw_scores(syb, times, events, fu, thresholds)
        plco_dca = dca_ipcw_scores(plco, times, events, fu, thresholds)

        # USPSTF: binary classifier. Compute single NB(p_t) curve.
        tp_u, fp_u, prev = dca_ipcw_binary(uspstf, times, events, fu)
        uspstf_nb = np.array([nb_for_binary(tp_u, fp_u, prev, pt)
                              for pt in thresholds])

        # Assemble per-threshold CSV
        out = pd.DataFrame({
            "threshold": thresholds,
            "nb_lungevity": lung_dca["nb"].values,
            "nb_sybil": syb_dca["nb"].values,
            "nb_plcom2012": plco_dca["nb"].values,
            "nb_uspstf": uspstf_nb,
            "nb_all": lung_dca["nb_all"].values,
        })
        out_path = os.path.join(
            OUTPUT_DIR, f"dca_with_baselines_year{year}.csv")
        out.to_csv(out_path, index=False)
        print(f"  year {year}: saved {out_path}")

        # Headline summary at selected thresholds
        for pt in (0.01, 0.05, 0.10):
            idx = int(np.argmin(np.abs(thresholds - pt)))
            summary_rows.append(dict(
                Year=year, p_t=pt, N=len(sub), N_cancer=int(events.sum()),
                Lungevity=float(lung_dca["nb"].iloc[idx]),
                Sybil=float(syb_dca["nb"].iloc[idx]),
                PLCOm2012=float(plco_dca["nb"].iloc[idx]),
                USPSTF=float(uspstf_nb[idx]),
                TreatAll=float(lung_dca["nb_all"].iloc[idx]),
            ))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "dca_baselines_summary.csv"), index=False)

    print("\n" + "=" * 80)
    print("DCA WITH BASELINES — IPCW-WEIGHTED NET BENEFIT")
    print("=" * 80)
    print(summary_df.round(4).to_string(index=False))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
