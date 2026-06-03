#!/usr/bin/env python3
"""
Best-effort Lung-RADS v2022 reproduction on NLST, sharing the Pinsky 2015
framing for everything NLST data supports. v2022 changes implemented:

  * GGN baseline: <30mm -> Cat 2 (was <20 in v1.0); >=30 -> Cat 3.
  * New GGN:      <30mm -> Cat 2; >=30 -> Cat 3.

v2022 changes NOT implementable on NLST and how we handle them:

  * Part-solid: v2022 categorizes by solid-component diameter; NLST records
    only total long/perp diameter. We keep the v1.0 upper-bound treatment.
    Strict v2022 would downgrade many of these to Cat 3 when solid <6mm.
  * Slow-growth downgrade (Cat 4A -> 3 if stable at 3-mo follow-up): skipped.
    Not in NLST pipeline.
  * S modifier (incidental findings): skipped. Orthogonal to risk tier.
  * Atypical pulmonary cyst categories: skipped. NLST attenuation codes
    don't differentiate cyst morphology.

Everything else follows derive_lungrads.py (Pinsky 2015).
"""

import os
import numpy as np
import pandas as pd

ABN_CSV  = os.path.join(os.environ.get("NLST_TAB_DIR", "/tank-1/data/NLST/tab-data"), "sct_abnormalities_d040722.csv")
CMP_CSV  = os.path.join(os.environ.get("NLST_TAB_DIR", "/tank-1/data/NLST/tab-data"), "sct_comparison_abnorm_d040722.csv")
SCR_CSV  = os.path.join(os.environ.get("NLST_TAB_DIR", "/tank-1/data/NLST/tab-data"), "sct_screening_d040722.csv")
PART_CSV = os.path.join(os.environ.get("NLST_TAB_DIR", "/tank-1/data/NLST/tab-data"), "participant_d040722.csv")
OUT_CSV  = os.path.join(os.environ.get("SCREENING_METRICS_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_output")), "nlst_lungrads_v2022.csv")

NODULE_MEASURED = 51
NODULE_MICRO    = 52
ATT_SOLID, ATT_GGN, ATT_PARTSOLID = 1, 2, 3

NLST_SCR_POS   = {4, 5, 6}
NLST_SCR_VALID = {1, 2, 3, 4, 5, 6}

CAT_ORDER = {"1": 0, "2": 1, "3": 2, "4A": 3, "4B": 4, "4X": 5}
CAT_BACK  = {v: k for k, v in CAT_ORDER.items()}


def _round_int(x: float) -> int:
    return int(np.floor(x + 0.5))


# v2022: GGN cutoff changes from <20 to <30
GGN_CUT_V2022 = 30


def _cat_baseline_solid(d: int) -> str:
    if d < 6:  return "2"
    if d < 8:  return "3"
    if d < 15: return "4A"
    return "4B"

def _cat_baseline_ggn(d: int) -> str:
    return "2" if d < GGN_CUT_V2022 else "3"

def _cat_baseline_partsolid(d: int) -> str:
    # v2022 would use solid component; NLST lacks it -> v1.0 upper bound.
    if d < 6:  return "2"
    return "4B"

def _cat_new_solid(d: int) -> str:
    if d < 4:  return "2"
    if d < 6:  return "3"
    if d < 8:  return "4A"
    return "4B"

def _cat_new_partsolid(d: int) -> str:
    if d < 6:  return "3"
    return "4B"

def _cat_growing_solid(d: int) -> str:
    if d < 8:  return "4A"
    return "4B"

def _cat_growing_partsolid(d: int) -> str:
    if d < 6:  return "2"
    return "4B"


def categorize_nodule(d, att, study_yr, preexist, growth, attn_change) -> str:
    if study_yr == 0:
        if att == ATT_SOLID:     return _cat_baseline_solid(d)
        if att == ATT_GGN:       return _cat_baseline_ggn(d)
        if att == ATT_PARTSOLID: return _cat_baseline_partsolid(d)
        return "1"
    is_new     = (preexist == 1)
    is_growing = (growth == 2) or (att == ATT_PARTSOLID and attn_change == 2)
    if is_new:
        if att == ATT_SOLID:     return _cat_new_solid(d)
        if att == ATT_GGN:       return "2" if d < GGN_CUT_V2022 else "3"
        if att == ATT_PARTSOLID: return _cat_new_partsolid(d)
        return "1"
    if is_growing:
        if att == ATT_SOLID:     return _cat_growing_solid(d)
        if att == ATT_GGN:       return "2"
        if att == ATT_PARTSOLID: return _cat_growing_partsolid(d)
        return "1"
    return "2"


def derive() -> pd.DataFrame:
    abn  = pd.read_csv(ABN_CSV, low_memory=False)
    cmp_ = pd.read_csv(CMP_CSV, low_memory=False)
    part = pd.read_csv(PART_CSV, low_memory=False,
                       usecols=["pid", "scr_res0", "scr_res1", "scr_res2"])
    scr_table = pd.read_csv(SCR_CSV, low_memory=False,
                            usecols=["pid", "study_yr"])

    universe = scr_table.drop_duplicates(subset=["pid", "study_yr"]).copy()

    scr = part.melt(id_vars="pid",
                    value_vars=["scr_res0", "scr_res1", "scr_res2"],
                    var_name="study_yr", value_name="scr_res")
    scr["study_yr"] = scr["study_yr"].str[-1].astype(int)
    scr = scr.dropna(subset=["scr_res"])
    scr["scr_res"] = scr["scr_res"].astype(int)
    scr = universe.merge(scr, on=["pid", "study_yr"], how="left")
    scr = scr[scr["scr_res"].isin(NLST_SCR_VALID)].copy()
    scr["nlst_pos"] = scr["scr_res"].isin(NLST_SCR_POS).astype(int)

    cmp_keep = cmp_[["pid", "study_yr", "sct_ab_num",
                     "sct_ab_preexist", "sct_ab_gwth", "sct_ab_attn"]].copy()
    abn = abn.merge(cmp_keep, on=["pid", "study_yr", "sct_ab_num"], how="left")

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

    rank_meas = (meas_valid.groupby(["pid", "study_yr"])["rank"]
                            .max().reset_index()
                            .rename(columns={"rank": "rank_meas"}))
    n_meas = (meas[valid].groupby(["pid", "study_yr"]).size()
                          .reset_index(name="n_meas_nodules"))

    has_micro = (abn[abn["sct_ab_desc"] == NODULE_MICRO]
                 [["pid", "study_yr"]].drop_duplicates())
    has_micro["rank_micro"] = CAT_ORDER["2"]

    has_meas51 = (abn[abn["sct_ab_desc"] == NODULE_MEASURED]
                  [["pid", "study_yr"]].drop_duplicates())
    has_meas51["has_51"] = 1

    out = scr.copy()
    out = out.merge(rank_meas, on=["pid", "study_yr"], how="left")
    out = out.merge(n_meas,    on=["pid", "study_yr"], how="left")
    out = out.merge(has_micro, on=["pid", "study_yr"], how="left")
    out = out.merge(has_meas51, on=["pid", "study_yr"], how="left")

    out["n_meas_nodules"] = out["n_meas_nodules"].fillna(0).astype(int)
    out["has_51"] = out["has_51"].fillna(0).astype(int)

    out["rank_final"] = out[["rank_meas", "rank_micro"]].max(axis=1)
    out["rank_final"] = out["rank_final"].fillna(CAT_ORDER["1"]).astype(int)

    is_4x = (out["nlst_pos"] == 1) & (out["has_51"] == 0)
    out.loc[is_4x, "rank_final"] = CAT_ORDER["4X"]

    out["lungrads_v2022"] = out["rank_final"].map(CAT_BACK)
    out["lungrads_pos3"]  = (out["rank_final"] >= CAT_ORDER["3"]).astype(int)
    out["lungrads_pos4"]  = (out["rank_final"] >= CAT_ORDER["4A"]).astype(int)
    out["lungrads_pos4b"] = (out["rank_final"] >= CAT_ORDER["4B"]).astype(int)

    return out[["pid", "study_yr", "lungrads_v2022",
                "lungrads_pos3", "lungrads_pos4", "lungrads_pos4b",
                "n_meas_nodules", "scr_res", "nlst_pos"]]


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df = derive()
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}  ({len(df):,} screening rows)")
    print()
    print("Lung-RADS v2022 category distribution (% within study_yr):")
    dist = pd.crosstab(df["study_yr"], df["lungrads_v2022"],
                       normalize="index").round(4) * 100
    print(dist)
    print()
    print("Positive (>=Cat 3) rate by study_yr:")
    print((df.groupby("study_yr")["lungrads_pos3"].mean() * 100).round(2))
    print()
    print("Positive (>=Cat 4A) rate by study_yr:")
    print((df.groupby("study_yr")["lungrads_pos4"].mean() * 100).round(2))


if __name__ == "__main__":
    main()
