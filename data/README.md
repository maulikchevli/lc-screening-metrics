# Data files

Model prediction scores and NLST-derived tabular inputs consumed by every analysis in the paper. NLST cohort.

## Cohort

- **Test fold (four-way matched)**: 6,223 scans / 2,199 patients / 295 cancers (82 at year 1).
- **Dev fold**: 6,847 scans for LungEvaty, 6,694 scans for Sybil v1.0, 6,845 for Pillar-0. Used only for fitting per-horizon Platt recalibrators.

NLST `time_at_event` is capped at 5 years.

## Model predictions

### LungEvaty (citation withheld for blind review)

| File | Split | Format | Schema |
|---|---|---|---|
| `predictions_lungevaty_test.json` | Test | JSON, 6,277 records | per-record `pid`, `series`, `cancer_risk` (6-year array), `screen_timepoint`, `institution`, `cancer_laterality`, `y`, `time_at_event`, `y_seq`, `y_mask` |
| `predictions_lungevaty_dev.json` | Dev | JSON, 6,847 records | same schema; used to fit the per-horizon Platt recalibrator |

LungEvaty test predictions are also accessible via `four_way_comp_df.csv` columns `cancer_risk` / `risk_score`. The two JSONs above are needed only for (a) the optional `BUILD_COMPDF=1` rebuild and (b) Platt fitting in `decile_stratification_devfit.py` / `layer1_layer3_full.py` / `audit_completecase_y6_ci.py` / `compute_tp_fp_decomp.py`.

### Pillar-0 (= Sybil 1.5 architecture, our 3-seed reproduction)

| File | Split | Format | Schema |
|---|---|---|---|
| `predictions_pillar_test.csv` | Test | CSV, 6,449 rows | `series_uid`, `patient_id`, `timepoint`, `y`, `time_at_event`, `seed{0,1,2}_prob_y{1..6}`, ensemble `prob_y{1..6}` |
| `predictions_pillar_dev.csv` | Dev | CSV, 6,845 rows | same schema as test |

Reproduction faithfulness on test vs author scores: ΔAUC < 0.003, ΔC-index = 0.0005, per-scan corr ≥ 0.986. The paper uses these reproduced scores throughout, exposed in `four_way_comp_df.csv` as `sybil_risk_1`..`sybil_risk_6`.

`sybil1.5/seed{0,1,2}/test.csv` ships the author per-seed predictions (only used by `BUILD_COMPDF=1` for the optional rebuild against author scores).

### Sybil v1.0 (author-released)

| File | Split | Format | Description |
|---|---|---|---|
| `sybil/sybil_predictions.json` | Test | JSON | Author-calibrated predictions; `cancer_risk` is the 6-year array. Exposed in `four_way_comp_df.csv` as `sv1c_risk_*`. |
| `sybil/sybil_predictions_raw.json` | Test | JSON | Raw (pre-calibration) predictions. Exposed as `sv1r_risk_*`. |
| `predictions_sybil_v1_dev.json` | Dev | JSON, 6,694 records | Author-calibrated dev predictions; used by `decile_stratification_devfit.py` to expose v1.0 calibrator behaviour. Symlink to `predictions_dev.json`. |

Note: the calibrator collapses to 228 unique values on the dev fold. Decile D6 and D10 are empty on test.

### PLCOm2012 (Tammemägi 2013)

Computed at runtime from NLST tabular features (age, smoking history, pack-years, race, COPD, family history, education, BMI). No separate predictions file. Covariates come from `nlst_derived/participant_paper_subset.csv`; the Platt recalibrator is fit on the dev rows at runtime. Implementation: `scripts/uspstf_plcom.py:plcom2012_risk`.

## Master dataframe

| File | Rows | Description |
|---|---|---|
| `four_way_comp_df.csv` | 6,223 (test) | One row per scan, columns for every model's per-horizon predictions side-by-side with demographics, Lung-RADS, and outcomes. Read by almost every table-producing script. |

To regenerate from the raw per-model prediction files:

```bash
BUILD_COMPDF=1 bash ../scripts/run_all.sh
# or, manually:
python ../scripts/four_way_compare.py             # builds four_way_comp_df with author Pillar-0
python ../scripts/patch_comp_df_pillar_user.py    # swaps in user-reproduced Pillar-0 (used in the paper)
```

Verified bit-identical to the shipped file (zero numerical diff across all 46 columns).

## NLST-derived subsets (`nlst_derived/`)

Restricted NLST tabular data isn't shipped in raw form, but the scripts only need specific columns / rows. The subsets here ship exactly what's used.

| Subset | Replaces | What it includes |
|---|---|---|
| `nlst_derived/participant_paper_subset.csv` | `participant_d040722.csv` (~53k rows × ~200 cols, 29 MB) | 4,535 paper pids × 38 cols. Covers every column `merge_data`, `_plcom_extra_cols`, and `plcom2012_risk` reference: age, race, ethnicity, education, height/weight, smoking history, COPD/emphysema, family history, 15 personal-cancer flags; per-screen result code `scr_res0`/`scr_res1`/`scr_res2`. |
| `nlst_derived/lung_cancer_paper_subset.csv` | `lung_cancer_d040722.csv` | 319 cancer events with `pid`, `first_lc`, `candx_days`, `de_stag`, `de_stag_7thed`, `clinical_stag`, `lc_morph`, `lc_topog`, `lc_grade`, `lc_behav`. Used by `lungrads_comparison_v2.py` for the AJCC-stage discordance table. |
| `nlst_derived/nlst_lungrads_pinsky.csv` | output of `derive_lungrads.py` (which reads the restricted `sct_abnormalities_*` and `sct_comparison_abnorm_*` raw tables) | Retrospective Lung-RADS v1.0 per-scan category for every NLST scan. |
| `nlst_derived/nlst_lungrads_v2022.csv` | output of `derive_lungrads_v2022.py` | Lung-RADS v2022 per-scan category. Used for the v1.0-vs-v2022 sensitivity check. |

The pid universe of the participant / lung_cancer subsets is the union of the test cohort (n=2,199) and per-model dev folds (n=2,336), total 4,535 pids — two orders of magnitude smaller than the raw NLST tables. This is the standard NLST-derived publication form and is permitted under the NCI data-use agreement.

If you want to re-run `derive_lungrads*.py` from scratch (rather than using the pre-computed labels), you still need `sct_abnormalities_d040722.csv` and `sct_comparison_abnorm_d040722.csv` from CDAS — we don't ship those because the *outputs* are what every downstream script actually consumes.

## Mapping files

`test_mikhael_with_exam.json` — NLST exam-ID map used by `four_way_compare.py` when building the comp_df. Needed only for the `BUILD_COMPDF=1` rebuild.

`../scripts/nlst_accession_match_results.csv` — series-UID-to-accession map used to align Sybil and Pillar-0 predictions.
