#!/usr/bin/env python3
"""
Retroactive Lung-RADS v1.0 categorization for NLST scans, faithful to
Pinsky et al., Ann Intern Med 2015;162(7):485-491 ("Performance of Lung-RADS
in the National Lung Screening Trial: A Retrospective Assessment").

Algorithm follows Pinsky's Table 1 (Lung-RADS v1.0 criteria) and the
"Applying Lung-RADS to the NLST" Methods section:

  * Mean diameter = (long + perp) / 2, rounded to nearest whole mm.
  * Attenuation mapping: soft tissue (1) -> solid; ground glass (2) -> GGN;
    mixed (3) -> part-solid; codes {4, 6, 7, 9} not mapped.
  * Baseline (T0) categories from nodule size + attenuation.
  * Subsequent (T1/T2):
      new solid:   <4 -> 2; 4..<6 -> 3; 6..<8 -> 4A; >=8 -> 4B
      new GGN:     <20 -> 2; >=20 -> 3
      new part-solid: <6 -> 3; >=6 -> 4A/4B (range, use 4B as upper bound)
      preexisting + growing solid: <8 -> 4A; >=8 -> 4B
      preexisting + growing part-solid: <6 -> 2; >=6 -> 4B (upper bound)
      preexisting + growing GGN: Cat 2 ("unchanged/slowly growing")
      preexisting + not growing: Cat 2 (regardless of size)
  * Solid-component growth assumed if nodule grew OR suspicious attenuation
    change (sct_ab_attn=2).
  * Code-52 micronodules (<4mm) -> Cat 2 contribution.
  * 4X assigned only when scan has no measured (code-51) nodule but the
    NLST scr_res was positive (other-finding positive).

Output: one row per (pid, study_yr) NLST CT screening, with Cat 1 default
for scans that have no entry in the abnormality table.
"""

import os
import numpy as np
import pandas as pd

ABN_CSV  = os.path.join(os.environ.get("NLST_TAB_DIR", "/tank-1/data/NLST/tab-data"), "sct_abnormalities_d040722.csv")
CMP_CSV  = os.path.join(os.environ.get("NLST_TAB_DIR", "/tank-1/data/NLST/tab-data"), "sct_comparison_abnorm_d040722.csv")
SCR_CSV  = os.path.join(os.environ.get("NLST_TAB_DIR", "/tank-1/data/NLST/tab-data"), "sct_screening_d040722.csv")
PART_CSV = os.path.join(os.environ.get("NLST_TAB_DIR", "/tank-1/data/NLST/tab-data"), "participant_d040722.csv")
OUT_CSV  = os.path.join(os.environ.get("SCREENING_METRICS_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_output")), "nlst_lungrads_pinsky.csv")

# sct_ab_desc codes
NODULE_MEASURED = 51  # non-calcified nodule/mass >=4mm (only code with measurements)
NODULE_MICRO    = 52  # non-calcified micronodule <4mm
NODULE_BENIGN   = 53  # benign calcified

# sct_pre_att codes
ATT_SOLID, ATT_GGN, ATT_PARTSOLID = 1, 2, 3

# NLST scr_res: positive screening result codes (NLST's own definition,
# per participant data dictionary §4).
# Negatives: 1, 2, 3   (yes, 3 is "Negative screen, significant abnormalities
#                       not suspicious for lung cancer").
# Positives: 4, 5, 6.
# Codes >=10 (inadequate / non-compliant / not-expected) are NOT valid screens
# and are excluded from the universe entirely.
NLST_SCR_POS   = {4, 5, 6}
NLST_SCR_VALID = {1, 2, 3, 4, 5, 6}

CAT_ORDER = {"1": 0, "2": 1, "3": 2, "4A": 3, "4B": 4, "4X": 5}
CAT_BACK  = {v: k for k, v in CAT_ORDER.items()}


def _round_int(x: float) -> int:
    """Round-half-up to nearest whole mm, per ACR Lung-RADS spec."""
    return int(np.floor(x + 0.5))


# --- Per-nodule classifiers (Pinsky 2015 Table 1) ---------------------------

def _cat_baseline_solid(d: int) -> str:
    if d < 6:  return "2"
    if d < 8:  return "3"
    if d < 15: return "4A"
    return "4B"

def _cat_baseline_ggn(d: int) -> str:
    return "2" if d < 20 else "3"

def _cat_baseline_partsolid(d: int) -> str:
    # NLST lacks solid-component diameter -> Pinsky uses range; we encode
    # the upper bound for max-across-nodules aggregation.
    if d < 6:  return "2"
    return "4B"  # any part-solid >=6mm could be 3, 4A, or 4B -> upper bound

def _cat_new_solid(d: int) -> str:
    if d < 4:  return "2"
    if d < 6:  return "3"
    if d < 8:  return "4A"
    return "4B"

def _cat_new_partsolid(d: int) -> str:
    if d < 6:  return "3"
    return "4B"  # upper bound

def _cat_growing_solid(d: int) -> str:
    if d < 8:  return "4A"
    return "4B"

def _cat_growing_partsolid(d: int) -> str:
    if d < 6:  return "2"
    return "4B"


def categorize_nodule(d: int, att: int, study_yr: int,
                      preexist, growth, attn_change) -> str:
    """Single-nodule Lung-RADS category per Pinsky 2015.

    preexist: sct_ab_preexist  (1=No/new, 2=Yes, 9=Unable, NaN if T0)
    growth:   sct_ab_gwth      (1=No, 2=Yes, 9, NaN)
    attn_change: sct_ab_attn   (1=No, 2=Yes suspicious change, 9, NaN)
    """
    if study_yr == 0:
        if att == ATT_SOLID:     return _cat_baseline_solid(d)
        if att == ATT_GGN:       return _cat_baseline_ggn(d)
        if att == ATT_PARTSOLID: return _cat_baseline_partsolid(d)
        return "1"
    # T1/T2
    is_new      = (preexist == 1)
    is_growing  = (growth == 2) or (att == ATT_PARTSOLID and attn_change == 2)
    if is_new:
        if att == ATT_SOLID:     return _cat_new_solid(d)
        if att == ATT_GGN:       return "2" if d < 20 else "3"
        if att == ATT_PARTSOLID: return _cat_new_partsolid(d)
        return "1"
    if is_growing:
        if att == ATT_SOLID:     return _cat_growing_solid(d)
        if att == ATT_GGN:       return "2"  # GGN unchanged/slowly growing
        if att == ATT_PARTSOLID: return _cat_growing_partsolid(d)
        return "1"
    # Preexisting and not growing: Cat 2 regardless of size
    return "2"


# --- Pipeline ---------------------------------------------------------------

def derive() -> pd.DataFrame:
    abn  = pd.read_csv(ABN_CSV, low_memory=False)
    cmp_ = pd.read_csv(CMP_CSV, low_memory=False)
    part = pd.read_csv(PART_CSV, low_memory=False,
                       usecols=["pid", "scr_res0", "scr_res1", "scr_res2"])
    scr_table = pd.read_csv(SCR_CSV, low_memory=False,
                            usecols=["pid", "study_yr"])

    # Universe: one row per actually-performed CT screen (sct_screening).
    universe = scr_table.drop_duplicates(subset=["pid", "study_yr"]).copy()

    # Per-screen NLST scr_res from participant (long form), joined to universe
    scr = part.melt(id_vars="pid",
                    value_vars=["scr_res0", "scr_res1", "scr_res2"],
                    var_name="study_yr", value_name="scr_res")
    scr["study_yr"] = scr["study_yr"].str[-1].astype(int)
    scr = scr.dropna(subset=["scr_res"])
    scr["scr_res"] = scr["scr_res"].astype(int)
    scr = universe.merge(scr, on=["pid", "study_yr"], how="left")
    # Restrict universe to valid screens (scr_res in 1..6, per Pinsky's
    # num_screens definition). Drops inadequate/non-compliant rows.
    scr = scr[scr["scr_res"].isin(NLST_SCR_VALID)].copy()
    scr["nlst_pos"] = scr["scr_res"].isin(NLST_SCR_POS).astype(int)

    # Join comparison-read fields onto abnormality rows
    cmp_keep = cmp_[["pid", "study_yr", "sct_ab_num",
                     "sct_ab_preexist", "sct_ab_gwth", "sct_ab_attn"]].copy()
    abn = abn.merge(cmp_keep, on=["pid", "study_yr", "sct_ab_num"], how="left")

    # Per-nodule classification for measured (code 51) nodules
    meas = abn[abn["sct_ab_desc"] == NODULE_MEASURED].copy()
    meas["sct_long_dia"] = pd.to_numeric(meas["sct_long_dia"], errors="coerce")
    meas["sct_perp_dia"] = pd.to_numeric(meas["sct_perp_dia"], errors="coerce")
    meas["sct_pre_att"]  = pd.to_numeric(meas["sct_pre_att"],  errors="coerce")
    meas["mean_diam"]    = (meas["sct_long_dia"] + meas["sct_perp_dia"]) / 2.0
    valid = (meas["mean_diam"].notna() &
             meas["sct_pre_att"].isin([ATT_SOLID, ATT_GGN, ATT_PARTSOLID]))
    meas_valid = meas[valid].copy()
    meas_valid["d_round"] = meas_valid["mean_diam"].apply(_round_int)

    meas_valid["cat"] = meas_valid.apply(
        lambda r: categorize_nodule(
            d=r["d_round"],
            att=int(r["sct_pre_att"]),
            study_yr=int(r["study_yr"]),
            preexist=r["sct_ab_preexist"],
            growth=r["sct_ab_gwth"],
            attn_change=r["sct_ab_attn"],
        ), axis=1)
    meas_valid["rank"] = meas_valid["cat"].map(CAT_ORDER)

    # Per-scan: max rank from measured nodules
    rank_meas = (meas_valid.groupby(["pid", "study_yr"])["rank"]
                            .max().reset_index()
                            .rename(columns={"rank": "rank_meas"}))
    n_meas = (meas[valid].groupby(["pid", "study_yr"]).size()
                          .reset_index(name="n_meas_nodules"))

    # Code-52 micronodules contribute at least Cat 2
    has_micro = (abn[abn["sct_ab_desc"] == NODULE_MICRO]
                 [["pid", "study_yr"]].drop_duplicates())
    has_micro["rank_micro"] = CAT_ORDER["2"]

    # Code-51 presence indicator (any measured nodule, valid or not) — used
    # for the 4X "other-finding positive in absence of measured nodule" rule
    has_meas51 = (abn[abn["sct_ab_desc"] == NODULE_MEASURED]
                  [["pid", "study_yr"]].drop_duplicates())
    has_meas51["has_51"] = 1

    # Universe: every NLST screen recorded in participant scr_res (one row per
    # actually-performed screen). Scans not present in abn -> default Cat 1.
    out = scr.copy()
    out = out.merge(rank_meas, on=["pid", "study_yr"], how="left")
    out = out.merge(n_meas,    on=["pid", "study_yr"], how="left")
    out = out.merge(has_micro, on=["pid", "study_yr"], how="left")
    out = out.merge(has_meas51, on=["pid", "study_yr"], how="left")

    out["n_meas_nodules"] = out["n_meas_nodules"].fillna(0).astype(int)
    out["has_51"] = out["has_51"].fillna(0).astype(int)

    # Max(nodule, micronodule) rank
    out["rank_final"] = out[["rank_meas", "rank_micro"]].max(axis=1)
    out["rank_final"] = out["rank_final"].fillna(CAT_ORDER["1"]).astype(int)

    # 4X: NLST-positive scr_res + no measured (code-51) nodule
    is_4x = (out["nlst_pos"] == 1) & (out["has_51"] == 0)
    out.loc[is_4x, "rank_final"] = CAT_ORDER["4X"]

    out["lungrads_v10"]  = out["rank_final"].map(CAT_BACK)
    out["lungrads_pos3"] = (out["rank_final"] >= CAT_ORDER["3"]).astype(int)
    out["lungrads_pos4"] = (out["rank_final"] >= CAT_ORDER["4A"]).astype(int)

    return out[["pid", "study_yr", "lungrads_v10",
                "lungrads_pos3", "lungrads_pos4",
                "n_meas_nodules", "scr_res", "nlst_pos"]]


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df = derive()
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}  ({len(df):,} screening rows)")
    print()
    print("Screenings per study_yr (cf. Pinsky: 26,309 T0 / 25,097 T1 / 23,574 T2):")
    print(df.groupby("study_yr").size())
    print()
    print("Lung-RADS category distribution (% within study_yr):")
    dist = pd.crosstab(df["study_yr"], df["lungrads_v10"], normalize="index").round(3) * 100
    print(dist)
    print()
    print("Positive (>=Cat 3) rate by study_yr  (Pinsky 2015: 13.6% T0, ~5.5% T1/T2):")
    print((df.groupby("study_yr")["lungrads_pos3"].mean() * 100).round(2))
    print()
    print("Positive (>=Cat 4A) rate by study_yr:")
    print((df.groupby("study_yr")["lungrads_pos4"].mean() * 100).round(2))


if __name__ == "__main__":
    main()
