#!/usr/bin/env bash
# Regenerate every paper table CSV and every paper figure in dependency order.
#
# The pipeline is organised around the paper's four-component evaluation
# framework (Framework §):
#   (i)   Global discrimination
#   (ii)  Calibration — cohort-level and per Lung-RADS category
#   (iii) Operating-point performance — cancer capture and net benefit
#   (iv)  Subgroup equity
#
# Prereqs:
#   Python environment with: pandas, numpy, scipy, scikit-learn, matplotlib,
#   lifelines, pyyaml.
#
# Run from this scripts/ directory; all paths resolve relative to the bundle
# root (parent of scripts/). The master four_way_comp_df.csv and the
# pre-computed Lung-RADS labels ship in data/; the optional BUILD_COMPDF=1
# rebuild reconstructs the master dataframe from the raw model predictions.
# Override file locations via env vars (see evaluate_survival.py).
set -euo pipefail

# Resolve bundle layout so individual scripts find data/ and eval_output/.
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SCREENING_METRICS_OUTPUT_DIR="${SCREENING_METRICS_OUTPUT_DIR:-$BUNDLE_ROOT/eval_output}"
mkdir -p "$SCREENING_METRICS_OUTPUT_DIR"
# Seed the master comp_df from data/ so downstream scripts can read it from
# OUTPUT_DIR. evaluate_survival.py does the same on import, but seeding here
# means run_all.sh's BUILD_COMPDF path also has a clean target dir.
if [ ! -e "$SCREENING_METRICS_OUTPUT_DIR/four_way_comp_df.csv" ] \
   && [ -e "$BUNDLE_ROOT/data/four_way_comp_df.csv" ]; then
  cp "$BUNDLE_ROOT/data/four_way_comp_df.csv" \
     "$SCREENING_METRICS_OUTPUT_DIR/four_way_comp_df.csv"
fi
cd "$BUNDLE_ROOT/scripts"

# ---------------------------------------------------------------------------
# 0. (optional) Rebuild four_way_comp_df.csv from raw model predictions
# ---------------------------------------------------------------------------
# BUILD_COMPDF=1
if [ "${BUILD_COMPDF:-0}" = "1" ]; then
  python four_way_compare.py             # builds comp_df with author Pillar-0
  python patch_comp_df_pillar_user.py    # patches in user-reproduced Pillar-0
fi

# ---------------------------------------------------------------------------
# 1. Derive Lung-RADS labels for the NLST test cohort
# ---------------------------------------------------------------------------
python derive_lungrads.py          # nlst_lungrads_pinsky.csv  (v1.0, Pinsky 2015)
python derive_lungrads_v2022.py    # nlst_lungrads_v2022.csv   (v2022)
python lungrads_v10_vs_v2022.py    # transition / op-point comparison v1.0 vs v2022
                                   # (sensitivity check for the v1.0 choice)

# ---------------------------------------------------------------------------
# 2. Component (i) — Global discrimination
# ---------------------------------------------------------------------------
python layer1_layer3_full.py       # tab:auc (AUC, PR-AUC) + tab:brier_decomp
python cindex_with_ci.py           # tab:auc (Uno C-index, IPCW)
python lr_auc_prauc_cindex.py      # Lung-RADS baseline AUC/PR-AUC/C-index
python paired_diffs_auc_prauc.py   # tab:supp-paired-diffs

# ---------------------------------------------------------------------------
# 3. Component (ii) — Calibration (cohort-level + per Lung-RADS category)
# ---------------------------------------------------------------------------
python calib_lr_anchored.py        # tab:lr_cat_calib (per-LR-category cal y1)
# (Brier decomposition for the cohort-level calibration view ships from
#  layer1_layer3_full.py above.)
python decile_stratification_devfit.py     # decile_incidence_devfit.csv,
                                           # decile_calibration_devfit.csv,
                                           # nb_at_decile_thresholds_devfit.csv,
                                           # subgroup_year1_devfit.csv
python build_consolidated_decile_report.py # decile_devfit_consolidated.md
                                           # (covers tab:decile-cal)

# ---------------------------------------------------------------------------
# 4. Component (iii) — Operating-point performance
#    (a) cancer capture (rank-based)
#    (b) net benefit (calibration-sensitive)
# ---------------------------------------------------------------------------
# Cancer capture
python lungrads_bins_and_lorenz.py  # tab:lr_neg_concentration, tab:lr_anchored_caps,
                                    # tab:supp-lr-multihorizon, tab:supp-tandem-decomp,
                                    # lorenz_gini_multihorizon.csv, operating_points_*.csv
python task1_sp_prec_recalls.py     # task1_sp_prec_recalls.csv (FR10/20/40/70/R95)
python task3_pr_curves.py           # task3_pr_curves.json — input to PR figure
python pr_op_lr_anchored.py         # tab:lr_anchored_capture
python lungrads_comparison_v2.py    # tab:tandem-decomp-y1, tab:supp-lr-discordance,
                                    # tab:supp-lr-stage
python bootstrap_lr_own_t0_recall.py    # tab:lr_anchored_caps (full vs T0)
python paired_diffs_lr_capture.py       # tab:supp-paired-capture
python paired_diffs_vs_lr_auc_cindex.py # AUC/C-index paired-diff vs Lung-RADS

# Net benefit
python nb_lr_anchored.py            # tab:dca_lr_anchored, tab:supp-dca-multihorizon
python compute_tp_fp_decomp.py      # tab:tp_fp_decomp
python dca_continuous.py            # dca_continuous_y1y3y6.csv (continuous DCA grid)
python lungrads_dca.py              # tab:supp-dca-paired-diffs
python audit_plco_no_rescale.py     # tab:supp-norescale (PLCOm rescaling sensitivity)
python audit_completecase_y6_ci.py  # year-6 IPCW-vs-CC sensitivity check

# ---------------------------------------------------------------------------
# 5. Component (iv) — Subgroup equity
# ---------------------------------------------------------------------------
python subgroup_lr_anchored.py     # tab:subgroup_yr1
python subgroup_3way.py            # tab:supp-race
python bootstrap_equity_ranges.py  # tab:subgroup_equity_headline

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
python lorenz_with_ci.py            # fig:lorenz_headline, fig:lorenz_multihorizon
python dca_with_ci.py               # fig:dca_t0,         fig:dca_multihorizon
python make_pr_figure.py            # fig:supp-pr-curves
python make_reliability_figure.py   # reliability_diagrams (commented out in current draft)

echo "Done. CSV outputs are in eval_output/; figure PDF/PNGs are in session_figures/ and eval_output/."
