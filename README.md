# Bridging the Gap to Clinical Impact

Code and intermediate data accompanying:

> *Aligning Evaluation to Deployment: Measuring the Clinical Utility of End-to-End Deep Learning for Lung Cancer Screening.*

This release contains everything needed to reproduce the paper's tables and figures from the NLST four-way-matched test cohort (6,223 scans, 2,199 patients, 295 cancers).

## Quick start

```bash
cd scripts
bash run_all.sh
```

Outputs land in `eval_output/`. End-to-end runtime on a single machine is on the order of an hour, dominated by patient-clustered bootstrap (n = 500–1000 resamples).

The fastest sanity check is to verify the year-1 LungEvaty AUC directly against the master dataframe:

```python
import pandas as pd
from sklearn.metrics import roc_auc_score
df = pd.read_csv("data/four_way_comp_df.csv")
roc_auc_score(df["y"], df["risk_score"])   # ≈ 0.950
```

Dependencies: `pandas numpy scipy scikit-learn matplotlib lifelines pyyaml`. Developed against Python 3.10.

## Evaluation framework

We compare four lung-cancer-risk models on the same patients and the same horizons:

- **LungEvaty** - author realeased weights from [here](https://maulikchevli.github.io/lungevaty/) 
- **Pillar-0** — author released weights from [here](https://huggingface.co/YalaLab/Pillar0-Sybil-1.5)
- **Sybil v1.0** — the original author-released calibrated model
- **PLCOm2012** (Tammemägi 2013) — the clinical-logistic baseline

The analysis is organised along four axes that together describe how a screening risk score behaves at deployment, rather than at the level of a single summary statistic:

1. **Global discrimination** — AUC, PR-AUC, and Uno's IPCW C-index.
2. **Calibration** — cohort-level Brier decomposition and per-Lung-RADS-category predicted-vs-observed agreement.
3. **Operating-point performance** — cancer capture (rank-based, via Lorenz / cumulative-gains curves) and net benefit (calibration-sensitive, via decision-curve analysis), anchored both to Lung-RADS workup volumes and to risk-of-malignancy thresholds.
4. **Subgroup equity** — recall spread at the Lung-RADS ≥ 4B-matched flag rate across sex, age tertile, and pack-year quartile.

Every script in `scripts/` slots into one of these four components, or into the foundation layer that builds the master dataframe and the retrospective Lung-RADS labels.

## Data

The master dataframe `data/four_way_comp_df.csv` carries one row per scan, with each model's per-horizon predictions alongside the NLST demographics, retrospective Lung-RADS categories, and outcome columns. It is the only input most table scripts consume.

For Platt recalibration on the dev fold we additionally ship LungEvaty and Sybil v1.0 dev-fold predictions (`predictions_lungevaty_dev.json`, `predictions_sybil_v1_dev.json`) and the Pillar-0 dev CSV (`predictions_pillar_dev.csv`). The raw LungEvaty test JSON, the NLST exam-ID map, and the per-seed Sybil-1.5 author predictions are needed only for the optional rebuild of the master dataframe (`BUILD_COMPDF=1 bash scripts/run_all.sh`); rebuilding produces a file numerically identical to the shipped one.

The NLST tabular tables (`participant_d040722.csv`, `lung_cancer_d040722.csv`, and the per-nodule `sct_*` tables) are restricted under the NCI data-use agreement and cannot be redistributed in full. `data/nlst_derived/` ships paper-specific subsets — exactly the columns the scripts read (PLCOm2012 covariates, per-screen result code, AJCC stage and histology), restricted to the 4,535 patients in the test ∪ dev union — together with pre-computed Lung-RADS v1.0 (Pinsky 2015) and v2022 labels for every scan. This is the standard NLST-derived publication form and is permitted under the data-use agreement. Researchers wishing to recompute Lung-RADS from raw nodules or use columns we do not ship should apply via [CDAS](https://cdas.cancer.gov/nlst/).

Model weights for LungEvaty and Sybil are released by their respective teams; training code is out of scope here. File-level schemas are documented in `data/README.md`.

## Pipeline

`run_all.sh` executes the 36 Python scripts in dependency order: foundation → discrimination → calibration → operating-point capture and net benefit → subgroup equity → figures. Paths inside the scripts resolve relative to the bundle, so the pipeline runs from any clone without staging. Most users will not need to set anything; for unusual setups, every external file location is also reachable through an environment variable (`SCREENING_METRICS_OUTPUT_DIR`, `LUNGEVATY_{TEST,DEV}_JSON`, `NLST_{PARTICIPANT,LUNG_CANCER}_CSV`, `NLST_TAB_DIR`, `NLST_EXAM_MAP_JSON`, `SYBIL15_DIR`) — see the constants block at the top of `scripts/evaluate_survival.py`.

The mapping from each paper table or figure to the script that produces it follows. Numbered axes in the third column refer back to the framework above.

### Main-text tables

| Table | Description | Component / script | Output |
|---|---|---|---|
| `tab:cohort` | Cohort characteristics | population summary in `evaluate_survival.py` | stdout |
| `tab:auc` | AUC, PR-AUC, C-index, years 1–6 | (i) `layer1_layer3_full.py` + `cindex_with_ci.py` + `lr_auc_prauc_cindex.py` | `layer1_layer3_full.csv`, `cindex_with_ci.csv` |
| `tab:brier_decomp` | Brier decomposition at y1 / y3 / y6 | (ii) `layer1_layer3_full.py` | `layer1_layer3_full.csv` |
| `tab:lr_cat_calib` | Per-Lung-RADS-category calibration, year 1 | (ii) `calib_lr_anchored.py` | `calib_lr_anchored_y1.csv` |
| `tab:lr_anchored_capture` | Year-1 precision / recall at LR-anchored flag rates | (iii) `pr_op_lr_anchored.py` | `pr_op_lr_anchored.csv` |
| `tab:dca_lr_anchored` | Year-1 NB at LR-anchored thresholds | (iii) `nb_lr_anchored.py` | `nb_lr_anchored.csv` |
| `tab:tp_fp_decomp` | Year-1 TP / FP decomposition at p_t = 0.10 / 0.15 | (iii) `compute_tp_fp_decomp.py` | included in `nb_lr_anchored.csv` |
| `tab:tandem-decomp-y1` | Year-1 captures decomposed by LR category | (iii) `lungrads_bins_and_lorenz.py` | `lungrads_reclass_y1_wide.csv` |
| `tab:lr_neg_concentration` | Within-LR-negative concentration | (iii) `lungrads_bins_and_lorenz.py` | `lungrads_reclass_y1_wide.csv` |
| `tab:subgroup_equity_headline` | Recall range across subgroups at Cat ≥ 4B | (iv) `bootstrap_equity_ranges.py` | `subgroup_y1_lr_anchored.csv` |

### Supplementary tables

| Table | Description | Component / script | Output |
|---|---|---|---|
| `tab:supp-norescale` | PLCOm net-benefit no-rescale sensitivity | (iii) `audit_plco_no_rescale.py` | `audit_plco_no_rescale.csv` |
| `tab:supp-paired-diffs` | Paired AUC / PR-AUC 95% CIs | (i) `paired_diffs_auc_prauc.py` | `auc_prauc_paired_diffs.csv` |
| `tab:supp-paired-capture` | Paired imaging − LR recall differences | (iii) `paired_diffs_lr_capture.py` | `paired_diffs_lr_capture_diffs.csv` |
| `tab:decile-cal` | D1 / D10 calibration, deployment form | (ii) `decile_stratification_devfit.py` + `build_consolidated_decile_report.py` | `decile_calibration_devfit.csv`, `decile_devfit_consolidated.md` |
| `tab:supp-lr-discordance` | LR ≥ 3 vs AI top-decile contingency | (iii) `lungrads_comparison_v2.py` | stdout |
| `tab:supp-lr-stage` | Discordance by AJCC stage | (iii) `lungrads_comparison_v2.py` | stdout |
| `tab:supp-dca-multihorizon` | Multi-horizon NB at LR-anchored thresholds | (iii) `nb_lr_anchored.py` | `nb_lr_anchored.csv` |
| `tab:supp-lr-multihorizon` | Multi-horizon precision / recall at LR-anchored | (iii) `pr_op_lr_anchored.py` + `lungrads_bins_and_lorenz.py` | `pr_op_lr_anchored.csv`, `operating_points_multihorizon.csv` |
| `tab:supp-pr-op-multihorizon` | Multi-horizon ultra-selective op-points | (iii) `task1_sp_prec_recalls.py` + `lungrads_bins_and_lorenz.py` | `task1_sp_prec_recalls.csv`, `operating_points_multihorizon.csv` |
| `tab:lr_anchored_caps` | Recall at Cat ≥ 4B flag rate, full vs T0 | (iii) `bootstrap_lr_own_t0_recall.py` + `lungrads_bins_and_lorenz.py` | `operating_points_y1_all_TP.csv`, `operating_points_y1_T0.csv` |
| `tab:supp-tandem-decomp` | Multi-horizon tandem decomposition | (iii) `lungrads_bins_and_lorenz.py` | `lungrads_reclass_y1_T0_wide.csv` |
| `tab:supp-dca-paired-diffs` | Paired imaging − LR NB differences | (iii) `lungrads_dca.py` + `paired_diffs_vs_lr_auc_cindex.py` | `lungrads_dca_paired_diffs.csv` |
| `tab:supp-race` | Race-stratified PR-AUC and recall | (iv) `subgroup_3way.py` | `subgroup_3way.csv` |
| `tab:subgroup_yr1` | Subgroup equity at Cat ≥ 4B-matched flag rate | (iv) `subgroup_lr_anchored.py` | `subgroup_y1_lr_anchored.csv` |

`audit_completecase_y6_ci.py` supplies the year-6 IPCW-vs-CC sensitivity check that several supplementary tables draw on. `dca_continuous.py` produces `dca_continuous_y1y3y6.csv`, the continuous-DCA grid underlying the DCA-figure annotations. `lungrads_comparison.py` is a library imported by `lungrads_comparison_v2.py`.

### Figures

| Figure | Description | Script | Output (in `figures/`) |
|---|---|---|---|
| `fig:lorenz_headline` | Year-1 Lorenz, full cohort vs T0 | `lorenz_with_ci.py` | `lorenz_y1_alltp_vs_T0_semilog_ci.{pdf,png}` |
| `fig:lorenz_multihorizon` | Lorenz curves at years 1 / 3 / 6 | `lorenz_with_ci.py` | `lorenz_multihorizon_ci.{pdf,png}` |
| `fig:dca_t0` | Year-1 DCA, full vs T0 | `dca_with_ci.py` | `dca_y1_alltp_vs_T0_zoomed_ci.{pdf,png}` |
| `fig:dca_multihorizon` | DCA at years 1 / 3 / 6 | `dca_with_ci.py` | `dca_multihorizon_zoomed_ci.{pdf,png}` |
| `fig:supp-pr-curves` | PR curves at years 1 / 3 / 6 | `task3_pr_curves.py` → `make_pr_figure.py` | `pr_curves.{pdf,png}` |

`make_reliability_figure.py` produces `reliability_diagrams.{pdf,png}`; the figure is currently commented out in `results.tex` but the script is retained for transparency.

## Conventions

The following hold throughout the pipeline:

- **Horizon mode.** IPCW (Uno-style weights, KM-estimated censoring distribution) at years 1 and 3; complete-case at year 6, since NLST `time_at_event` caps at 5.
- **Bootstrap.** Patient-clustered. n = 1,000 for discrimination, calibration, and per-decile incidence; n = 500 for net benefit. Differences across models are reported on paired resamples (shared indices).
- **Lung-RADS anchors.** Flag rates of 4.11% (Cat ≥ 4B workup volume), 7.62% (Cat ≥ 4A), and 11.81% (Cat ≥ 3). Lung-RADS v1.0 (Pinsky 2015) is primary; v2022 is computed by `lungrads_v10_vs_v2022.py` as a sensitivity check. The per-category risk-of-malignancy thresholds that anchor the DCA p_t grid are unchanged across versions.

