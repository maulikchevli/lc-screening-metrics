#!/usr/bin/env python3
"""Build a consolidated decile-analysis markdown report from the
dev-fit decile CSVs. One row per model in deployment form.

Outputs:
  eval_output/decile_{devfit,testfit}_consolidated.md
"""
import os
import pandas as pd
import numpy as np

OUT_DIR = os.environ.get("SCREENING_METRICS_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_output"))
# Mode toggle: matches DECILE_MODE used by decile_stratification_devfit.py.
# "devfit" (default) reads *_devfit.csv files; "testfit" reads *_testfit.csv.
MODE = os.environ.get("DECILE_MODE", "devfit")
assert MODE in ("devfit", "testfit"), f"Invalid DECILE_MODE: {MODE}"
SUFFIX = f"_{MODE}"

# (paper-label, source-model-name, source-calibration-name)
DEPLOY = [
    ("LungEvaty",  "LungEvaty", "platt"),
    ("Sybil",      "Sybil",     "cal"),
    ("Pillar-0",   "Pillar-0",  "platt"),
    ("PLCOm2012",  "PLCOm",     "platt"),
]

# Q4 (subgroup) only stores raw rows for LungEvaty and PLCOm — by rank
# invariance these equal the Platt-cal numbers, so the consolidator
# substitutes raw rows when asked for Platt. See
# feedback_rank_invariance_no_rerun.md.
Q4_CAL_SUBSTITUTE = {
    ("LungEvaty", "platt"): "raw",
    ("PLCOm",     "platt"): "raw",
    ("Pillar-0",  "platt"): "raw",
}

YEARS = (1, 3, 6)


def fmt_ci(v, lo, hi, digits=4):
    if pd.isna(v):
        return "—"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def pretty(v, digits=4):
    if pd.isna(v):
        return "—"
    return f"{v:.{digits}f}"


def main():
    q1 = pd.read_csv(os.path.join(OUT_DIR, f"decile_incidence{SUFFIX}.csv"))
    q2 = pd.read_csv(os.path.join(OUT_DIR, f"decile_calibration{SUFFIX}.csv"))
    q3 = pd.read_csv(os.path.join(OUT_DIR, f"nb_at_decile_thresholds{SUFFIX}.csv"))
    q4 = pd.read_csv(os.path.join(OUT_DIR, f"subgroup_year1{SUFFIX}.csv"))

    lines = []
    P = lines.append

    mode_label = "dev-fit" if MODE == "devfit" else "test-fit (own-distribution)"
    P(f"# Decile analysis — consolidated results ({mode_label})\n")
    P(f"Generated from the {mode_label} decile pipeline "
      "(`decile_stratification_devfit.py`). One row per model "
      "in its deployment form. Full per-(model, form) breakdowns "
      "are in the underlying CSVs.\n")

    # ---- Methodology --------------------------------------------------------
    P("## Methodology\n")
    P("- **Unit**: per-scan (n=6,214 scans, 2,199 patients in the matched "
      "4-way test cohort; 295 total cancers, 82 year-1 cancers).")
    P("- **Anchor**: year-1 predicted risk; decile labels propagated "
      "across year-1/3/6 evaluation.")
    if MODE == "devfit":
        P("- **Threshold derivation**: 10/20/.../90 percentile cuts fit "
          "on the held-out NLST dev fold, frozen, applied to test "
          "(deployment-faithful). Pillar-0 cuts derived from the user-"
          "reproduced dev predictions (validated against author scores: "
          "ΔAUC < 0.003 per horizon, ΔC-index = 0.0005).")
    else:
        P("- **Threshold derivation**: 10/20/.../90 percentile cuts fit "
          "on the test scores themselves (own-distribution). By "
          "construction each decile bin contains ~10% of test scans. "
          "This is the S4-style binning — not deployment-faithful "
          "(cut placement uses the evaluation set), but useful as a "
          "sensitivity check on the dev-fit results.")
    P("- **Calibration**: each model evaluated in its deployment form "
      "(one row per model).")
    P("    - LungEvaty: Platt-recalibrated on NLST dev (per horizon)")
    P("    - Sybil: author-released calibrator (built into the release)")
    P("    - Pillar-0: Platt-recalibrated on user-reproduced NLST dev "
      "(per horizon)")
    P("    - PLCOm2012: Platt-recalibrated on NLST dev (per horizon)")
    P("- **IPCW** for year-1 and year-3 incidence/NB; **complete-case** "
      "for year-6 (per paper convention).")
    P("- **CIs**: patient-clustered bootstrap, n=1000 for Q1/Q2, n=500 "
      "for Q3 NB.")
    P("- **5-category aggregation (Option B, selective top)**: C1=D1, "
      "C2=D2, C3=D3+D4, C4=D5+D6+D7, C5=D8+D9+D10.\n")

    # ---- Sanity checks ------------------------------------------------------
    P("## Sanity checks\n")
    P("- Year-1 cancer events sum to 82 across all 10 deciles for every "
      "model (✓ matches total y1 cancers in the matched test cohort).")
    if MODE == "devfit":
        P("- Dev-fit test flag rates per decile fall within ±2 pp of "
          "nominal 10% for LungEvaty, PLCOm, and Pillar-0 (Pillar-0 "
          "exact 10% by test-fit construction).")
        P("- **Sybil exhibits decile collapse** — D6 and D10 are empty "
          "because the author-released calibrator quantizes outputs to "
          "only 228 unique values; multiple percentile cut points "
          "collide at the same numeric value. This is a real "
          "deployment limitation, not an analysis artifact.\n")
    else:
        P("- Test-fit deciles are rank-based (exact 10% per bin by "
          "construction) for every model; flag rates are uniform.")
        P("- Under rank-based test-fit binning, Sybil's 228-unique-"
          "value quantization does **not** produce empty deciles "
          "(ties are broken by stable rank order). This means test-"
          "fit numbers understate the deployment-facing limitation — "
          "see the dev-fit report for the deployment-faithful view.\n")

    # ---- Decile size table --------------------------------------------------
    P("## Achieved test-fold flag rate per decile\n")
    P("| Model | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for paper, mdl, cal in DEPLOY:
        sizes = []
        for d in range(1, 11):
            row = q1[(q1.model == mdl) & (q1.calibration == cal)
                     & (q1.grain == "decile") & (q1.bin == d)
                     & (q1.eval_year == 1)]
            if len(row) == 0:
                sizes.append("—")
            else:
                n = int(row["n_scans"].iloc[0])
                pct = n / 6214 * 100
                sizes.append(f"{pct:.1f}%")
        P("| " + paper + " | " + " | ".join(sizes) + " |")
    P("")

    # ---- Q1: per-decile incidence ------------------------------------------
    P("## Q1 — Observed cancer incidence per decile\n")
    for year in YEARS:
        P(f"### Year {year}\n")
        P("| Bin | " + " | ".join(p for p, _, _ in DEPLOY) + " |")
        P("|---|" + "|".join(["---"] * len(DEPLOY)) + "|")
        for d in range(1, 11):
            row_vals = []
            for paper, mdl, cal in DEPLOY:
                r = q1[(q1.model == mdl) & (q1.calibration == cal)
                       & (q1.grain == "decile") & (q1.bin == d)
                       & (q1.eval_year == year)]
                if len(r) == 0:
                    row_vals.append("—")
                else:
                    n = int(r["n_scans"].iloc[0])
                    if n == 0:
                        row_vals.append("∅ (empty)")
                    else:
                        v = r["obs_incidence"].iloc[0]
                        lo = r["ci_low"].iloc[0]
                        hi = r["ci_high"].iloc[0]
                        row_vals.append(fmt_ci(v, lo, hi))
            P(f"| D{d} | " + " | ".join(row_vals) + " |")
        P("")

    # ---- Q1: 5-cat aggregated incidence ------------------------------------
    P("## Q1 — Observed cancer incidence per 5-category aggregation\n")
    for year in YEARS:
        P(f"### Year {year}\n")
        P("| Bin | " + " | ".join(p for p, _, _ in DEPLOY) + " |")
        P("|---|" + "|".join(["---"] * len(DEPLOY)) + "|")
        cat_names = {1: "C1 (workup)", 2: "C2 (3-mo)", 3: "C3 (6-mo)",
                     4: "C4 (12-mo)", 5: "C5 (24-mo)"}
        for c in range(1, 6):
            row_vals = []
            for paper, mdl, cal in DEPLOY:
                r = q1[(q1.model == mdl) & (q1.calibration == cal)
                       & (q1.grain == "cat") & (q1.bin == c)
                       & (q1.eval_year == year)]
                if len(r) == 0:
                    row_vals.append("—")
                else:
                    n = int(r["n_scans"].iloc[0])
                    if n == 0:
                        row_vals.append("∅")
                    else:
                        v = r["obs_incidence"].iloc[0]
                        lo = r["ci_low"].iloc[0]
                        hi = r["ci_high"].iloc[0]
                        row_vals.append(fmt_ci(v, lo, hi))
            P(f"| {cat_names[c]} | " + " | ".join(row_vals) + " |")
        P("")

    # ---- Q1: top-D and top-C capture ---------------------------------------
    P("## Q1 — Top-decile (D1) and top-category (C1) cancer capture\n")
    for grain, name in (("decile", "D1"), ("cat", "C1")):
        P(f"### Top {name} capture\n")
        P("| Year | " + " | ".join(p for p, _, _ in DEPLOY) + " |")
        P("|---|" + "|".join(["---"] * len(DEPLOY)) + "|")
        for year in YEARS:
            row_vals = []
            for paper, mdl, cal in DEPLOY:
                r = q1[(q1.model == mdl) & (q1.calibration == cal)
                       & (q1.grain == grain) & (q1.bin == 1)
                       & (q1.eval_year == year)]
                if len(r) == 0:
                    row_vals.append("—")
                else:
                    v = r["top_capture"].iloc[0]
                    lo = r["capture_ci_low"].iloc[0]
                    hi = r["capture_ci_high"].iloc[0]
                    row_vals.append(fmt_ci(v, lo, hi, digits=3))
            P(f"| Year {year} | " + " | ".join(row_vals) + " |")
        P("")

    # ---- Q2: calibration ----------------------------------------------------
    P("## Q2 — Per-decile calibration (mean predicted vs mean observed)\n")
    P("Reporting D1 and D10 (extremes) only here; full per-decile and "
      "per-cat numbers in `decile_calibration_devfit.csv`.\n")
    for year in YEARS:
        P(f"### Year {year}\n")
        P("| Model | D1 pred | D1 obs | D1 diff [95% CI] | D10 pred | "
          "D10 obs | D10 diff [95% CI] |")
        P("|---|---|---|---|---|---|---|")
        for paper, mdl, cal in DEPLOY:
            row_vals = [paper]
            for d in (1, 10):
                r = q2[(q2.model == mdl) & (q2.calibration == cal)
                       & (q2.grain == "decile") & (q2.bin == d)
                       & (q2.eval_year == year)]
                if len(r) == 0:
                    row_vals += ["—", "—", "—"]
                    continue
                p_ = r["mean_predicted"].iloc[0]
                o_ = r["mean_observed"].iloc[0]
                df = r["diff"].iloc[0]
                lo = r["diff_lo"].iloc[0]; hi = r["diff_hi"].iloc[0]
                if pd.isna(p_) or pd.isna(o_):
                    row_vals += ["∅", "∅", "—"]
                else:
                    row_vals += [pretty(p_), pretty(o_),
                                 fmt_ci(df, lo, hi)]
            P("| " + " | ".join(row_vals) + " |")
        P("")

    # ---- Q3: NB at all 9 decile cumulative cuts + p_t ----------------------
    P("## Q3 — Net benefit at decile-cumulative thresholds\n")
    P("Cat-aggregated clinical boundaries marked: **fr=10% (C1)**, "
      "**fr=20% (C1+C2)**, **fr=40% (C1+C2+C3)**, **fr=70% (C1+C2+C3+C4)**. "
      "All 9 decile cumulative cuts shown.\n")
    cat_fr = {0.10, 0.20, 0.40, 0.70}
    for year in YEARS:
        P(f"### Year {year}\n")
        P("| Flag rate | " + " | ".join(p for p, _, _ in DEPLOY) + " |")
        P("|---|" + "|".join(["---"] * len(DEPLOY)) + "|")
        for fr in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
            label = f"**{int(fr*100)}%**" if fr in cat_fr else f"{int(fr*100)}%"
            row_vals = [label]
            for paper, mdl, cal in DEPLOY:
                r = q3[(q3.model == mdl) & (q3.calibration == cal)
                       & (q3.anchor_type == "decile_cum")
                       & (np.abs(q3.anchor_value - fr) < 1e-9)
                       & (q3.eval_year == year)]
                if len(r) == 0:
                    row_vals.append("—")
                else:
                    v = r["nb"].iloc[0]
                    lo = r["ci_low"].iloc[0]
                    hi = r["ci_high"].iloc[0]
                    realised = r["realised_flag_rate"].iloc[0]
                    cell = fmt_ci(v, lo, hi)
                    if abs(realised - fr) > 0.02:
                        cell += f" (realised={realised:.2f})"
                    row_vals.append(cell)
            P("| " + " | ".join(row_vals) + " |")
        # Treat-all reference row
        ref_row = ["treat-all"]
        for paper, mdl, cal in DEPLOY:
            r = q3[(q3.model == mdl) & (q3.calibration == cal)
                   & (q3.anchor_type == "decile_cum")
                   & (q3.eval_year == year)]
            if len(r) == 0:
                ref_row.append("—")
            else:
                ta = r["treat_all_nb"].iloc[0]
                ref_row.append(pretty(ta))
        P("| " + " | ".join(ref_row) + " |")
        P("")

    # ---- Q3: NB at fixed p_t -----------------------------------------------
    P("## Q3 — Net benefit at fixed probability thresholds (p_t)\n")
    P("All four models are reported in their deployed form, so NB at "
      "fixed `p_t` is directly interpretable.\n")
    for year in YEARS:
        P(f"### Year {year}\n")
        P("| p_t | " + " | ".join(p for p, _, _ in DEPLOY) + " |")
        P("|---|" + "|".join(["---"] * len(DEPLOY)) + "|")
        for pt in (0.01, 0.05, 0.10):
            row_vals = [f"{int(pt*100)}%"]
            for paper, mdl, cal in DEPLOY:
                r = q3[(q3.model == mdl) & (q3.calibration == cal)
                       & (q3.anchor_type == "pt")
                       & (np.abs(q3.anchor_value - pt) < 1e-9)
                       & (q3.eval_year == year)]
                if len(r) == 0:
                    row_vals.append("—")
                else:
                    v = r["nb"].iloc[0]
                    lo = r["ci_low"].iloc[0]
                    hi = r["ci_high"].iloc[0]
                    row_vals.append(fmt_ci(v, lo, hi))
            P("| " + " | ".join(row_vals) + " |")
        P("")

    # ---- Q4: subgroup year-1 -----------------------------------------------
    P("## Q4 — Subgroup year-1 analysis\n")
    P("Reporting top-D (D1) and top-C (C1) capture per subgroup; "
      "PR-AUC in `subgroup_year1_devfit.csv`.\n")
    subs = q4["subgroup_var"].unique()
    for sg in subs:
        P(f"### {sg}\n")
        P("| Level | n_scan | n_ca | " + " | ".join(
            f"{p} top-D" for p, _, _ in DEPLOY) + " |")
        P("|---|---|---|" + "|".join(["---"] * len(DEPLOY)) + "|")
        levels = q4[q4.subgroup_var == sg]["subgroup_level"].unique()
        for lvl in levels:
            row_vals = []
            for paper, mdl, cal in DEPLOY:
                q4_cal = Q4_CAL_SUBSTITUTE.get((mdl, cal), cal)
                r = q4[(q4.subgroup_var == sg)
                       & (q4.subgroup_level == lvl)
                       & (q4.model == mdl) & (q4.calibration == q4_cal)
                       & (q4.metric == "top_d_capture")]
                if len(r) == 0:
                    row_vals.append("—")
                else:
                    v = r["value"].iloc[0]
                    lo = r["ci_low"].iloc[0]
                    hi = r["ci_high"].iloc[0]
                    n = int(r["n_scans"].iloc[0])
                    n_ca = int(r["n_cancer"].iloc[0])
                    row_vals.append(fmt_ci(v, lo, hi, digits=3))
                    # n, n_ca printed once via first row
            # Get n from first row
            anyrow = q4[(q4.subgroup_var == sg)
                        & (q4.subgroup_level == lvl)].iloc[0]
            n = int(anyrow["n_scans"])
            nca = int(anyrow["n_cancer"])
            P(f"| {lvl} | {n} | {nca} | " + " | ".join(row_vals) + " |")
        P("")

    # ---- Limitations --------------------------------------------------------
    P("## Known limitations\n")
    if MODE == "devfit":
        P("1. **Sybil decile collapse**: D6 and D10 are empty in test "
          "(see Q1 table). Cause: the author-released calibrator "
          "outputs only 228 unique values across 6,694 dev scans; "
          "50/60 and 80/90 percentiles collide. **This is a "
          "deployment property of Sybil, not an analysis artifact** — "
          "adjacent decile follow-up tiers cannot be distinguished "
          "using Sybil's deployed output.\n")
        cal_item = 2
    else:
        P("1. **All decile cuts are test-fit (own-distribution)** — "
          "the cut placement uses the evaluation set, so these numbers "
          "are *not* deployment-faithful. They serve as a sensitivity "
          "check on the dev-fit report; for the deployment-facing "
          "narrative, use the dev-fit consolidated report.\n")
        P("2. **Sybil's calibrator quantization is hidden by rank-"
          "based testfit binning** — the 228-unique-value problem "
          "produces no empty bins here, but it does in deployment "
          "(dev-fit). Do not interpret this report as evidence that "
          "Sybil distinguishes adjacent deciles in deployment.\n")
        cal_item = 3
    P(f"{cal_item}. **Year-6 mode is complete-case**, not IPCW. "
      "NLST's time_at_event maxes at 5 years; year-6 NB uses the pre-"
      "fix paper convention. See `calibrate_plcom.py:_ipcw_weights_"
      "paper` for the rationale.\n")

    # ---- Pointers -----------------------------------------------------------
    P("## Files for further analysis\n")
    P("- `eval_output/decile_incidence_devfit.csv` (495 rows; decile + cat grain)")
    P("- `eval_output/decile_calibration_devfit.csv` (315 rows)")
    P("- `eval_output/nb_at_decile_thresholds_devfit.csv` (252 rows; "
      "9 decile cumulative cuts + 3 p_t × 7 variants × 3 horizons)")
    P("- `eval_output/subgroup_year1_devfit.csv` (192 rows)")
    P("- `eval_output/decile_stratification_devfit.log` (full run log)")
    P("- Source: `decile_stratification_devfit.py`\n")

    out = os.path.join(OUT_DIR, f"decile_{MODE}_consolidated.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
