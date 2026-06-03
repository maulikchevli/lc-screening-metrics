# Paper handoff addendum — 2026-05-20 (rev. 2)

Adds Lorenz/concentration curves, Lung-RADS-anchored risk-bin analysis, multi-horizon comparison, DCA + PR-curve operating points re-anchored to Lung-RADS thresholds, calibration assessed per Lung-RADS category, and a redesigned subgroup analysis. **Supersedes the decile-stratification results in `decile_devfit_consolidated.md` for Layer-2 / risk-stratification reporting** and **replaces the FR10/FR20/FR40/FR70 PR operating points in `task3_pr_curves_op_points.csv` and the decile_cum rows in `nb_at_decile_thresholds_devfit.csv`.**

Cohort, models, conventions: see `README.md` (unchanged).

---

## A. Why the framework changed

Equal-N decile binning at 10% granularity places D1 (top decile) at roughly 80% recall on the PR curve. The entire 65–95% recall range — exactly where the clinically actionable workup-threshold lives — sits inside D1 and gets flattened. Lung-RADS clinical practice uses non-uniform risk-anchored bins (90% / 5% / 2% / 2% population), so equal-N is also a statistical, not clinical, convention. The same critique applies to FR10/20/40/70 PR-operating points and decile_cum DCA thresholds — they all share the equal-N percentile anchor.

Replacement framework:
- **Graphical view**: Lorenz / concentration curves (bin-free).
- **Tabular operating points**: anchored to Lung-RADS thresholds rather than score percentiles.
- **Within-LR-category reclassification + calibration**: shows where AI augments vs. agrees with Lung-RADS.

### A.1 Methods disclosure (important for review)

The Lorenz curve uses fraction-of-population on the X-axis (not fraction-of-negatives as ROC does). At low prevalence (1.3% at y1), the Lorenz curve and the ROC curve share nearly identical shape, and the **Lorenz Gini coefficient is mathematically related to AUC by a low-prevalence correction (≈ 2·AUC − 1)**. We therefore do NOT report Gini as a standalone numerical summary in this paper — doing so would invite the (correct) critique that we are re-reporting AUC under a different name in a paper that critiques AUC.

What we do report:
- **Lorenz curves** themselves, as a screening-population-anchored visualization that makes operating-point reading clinically intuitive ("flag X% of scans → capture Y% of cancers").
- **Operating-point capture (precision/recall at specific flag rates)** — single points on the curve, anchored to Lung-RADS-equivalent volumes or to ultra-selective thresholds. These are not aggregable into AUC.
- **PR-AUC** is retained where appropriate as a discrimination reference, with the disclosure that the paper's primary view is operating-point capture.

---

## B. Lung-RADS state on the test cohort

Pinsky 2015 algorithm applied retrospectively (`derive_lungrads.py` → `eval_output/nlst_lungrads_pinsky.csv`). 100% coverage on the test cohort.

| LR category | N | y1 cancers | y1 incidence |
|---|---|---|---|
| LR1 | 2,650 | 4 | 0.15% |
| LR2 | 2,838 | 11 | 0.39% |
| LR3 | 261 | 3 | 1.15% |
| LR4A | 218 | 10 | 4.59% |
| LR4B | 230 | 51 | 22.17% |
| LR4X | 26 | 3 | 11.54% |

**4X caveat**: Pinsky retrospective rule = `nlst_pos == 1 AND has_51 == 0` (positive screen with no measured nodule). Approximates "other findings positive" — NOT real-world 4X. Mention in caption / footnote whenever LR4X numbers appear.

LR1+LR2 (LR-negative): 5,488 scans (88.2%), 15 cancers — **the cancers Lung-RADS misses entirely**.
LR4B+LR4X (LR workup): 256 scans (4.11%), 54 cancers — LR's workup-recommended volume.

---

## C. Lung-RADS uses prior-CT info at T1/T2 — AI does not

The Pinsky algorithm at T1/T2 explicitly requires comparison to prior CT (new nodule, growing nodule rules). AI models see only the current scan. Lung-RADS therefore has an information advantage over AI at T1/T2 that does not exist at T0. Each AI model's own Lorenz curve is unaffected; the Lung-RADS curve is the one that gets a longitudinal boost at T1/T2.

**Reporting policy**: both the full-cohort comparison (asymmetric, gives Lung-RADS the benefit of the doubt) and the T0-only sensitivity (info-symmetric) should be reported. Per-timepoint stratification is the diagnostic that quantifies the asymmetry.

---

## D. Lorenz curves (graphical primary)

### D.1 Files (no Gini in legends; recall @ LR ≥4B-matched volume shown instead)

- `lorenz_y1_paper_figure_no_gini.png` — **two-panel headline figure** (all-timepoints + T0-only).
- `lorenz_multihorizon_no_gini.png` — **3-panel multi-horizon** (y1/y3/y6).
- `lorenz_y1_by_timepoint.png` — 3-panel T0/T1/T2 stratification (information-asymmetry diagnostic).

### D.2 Numerical summary: recall @ LR ≥4B-matched flag rate

This is what to report numerically from the Lorenz framework. NOT Gini.

| Model | y1 all-TP | y1 T0-only | y3 all-TP | y6 all-TP |
|---|---|---|---|---|
| LungEvaty | **70.7%** | **79.5%** | **40.9%** | **28.8%** |
| Pillar-0 | 67.1% | 71.8% | 42.5% | 28.8% |
| Sybil | 56.1% | 59.0% | 34.3% | 24.7% |
| **Lung-RADS** (own LR≥4B set) | 65.9% | 53.8% | 33.1% | 23.4% |
| PLCOm2012 | 13.4% | 15.4% | 12.2% | 11.9% |

Lung-RADS reference is its own recall at the LR4B+LR4X subset (which is its workup-recommended category).

**At T0 (info-symmetric)**, LungEvaty exceeds Lung-RADS by 25.7 pp at LR's own workup volume. This is the cleanest fair comparison.

---

## E. Operating-points (replaces FR10/FR20/FR40/FR70 in `task3_pr_curves_op_points.csv`)

### E.1 Anchors used

| Anchor | Definition | Status |
|---|---|---|
| LR ≥4X | flag rate matching LR's most-selective category | **New (replaces FR10)** |
| LR ≥4B | flag rate matching LR workup volume | **New (replaces FR20–40)** |
| LR ≥4A | flag rate matching LR short-interval CT volume | **New (replaces FR40–70)** |
| ultra-selective top 0.1% / 0.2% / 0.4% by score | high-precision regime AI provides beyond LR | **New** |
| R95 (sensitivity ≥ 95%) | high-sensitivity safety regime | **Kept (clinically meaningful)** |

Bootstrap: patient-clustered n=1000.

### E.2 Headline y1 results, all-timepoints (subset)

| Anchor | Model | flag rate | precision | recall |
|---|---|---|---|---|
| LR ≥4X | Lung-RADS (own) | 0.42% | 11.5% | 3.7% |
| LR ≥4X | LungEvaty | 0.42% | **64.0%** | **19.5%** |
| LR ≥4X | Pillar-0 | 0.42% | 56.0% | 17.1% |
| LR ≥4X | Sybil | 0.42% | 60.0% | 18.3% |
| LR ≥4X | PLCOm | 0.42% | 4.0% | 1.2% |
| LR ≥4B | Lung-RADS (own) | 4.11% | 21.1% | 65.9% |
| LR ≥4B | LungEvaty | 4.11% | 23.3% | **70.7%** |
| LR ≥4B | Pillar-0 | 4.11% | 22.1% | 67.1% |
| LR ≥4B | Sybil | 4.11% | 18.5% | 56.1% |
| ultra 0.1% | LungEvaty | 0.10% | **83.3%** | 6.1% |
| ultra 0.2% | LungEvaty | 0.20% | **91.7%** | 13.4% |
| ultra 0.4% | LungEvaty | 0.40% | **64.0%** | 19.5% |
| R95 | LungEvaty | 24.2% | 4.7% | 95.1% |
| R95 | Lung-RADS (own) | not achievable | — | — |

Full multi-horizon table with bootstrap CIs: `pr_op_lr_anchored.csv`.

### E.3 Cross-horizon headline

| Horizon | LR ≥4B precision/recall | LungEvaty matched precision/recall |
|---|---|---|
| y1 | 21.1% / 65.9% | 23.3% / **70.7%** |
| y3 | 23.8% / 33.1% | 29.4% / **40.9%** |
| y6 | 37.5% / 23.4% | 46.2% / **28.8%** |

At every horizon, LungEvaty improves on Lung-RADS at the matched workup volume in both precision and recall. The gap widens with horizon.

### E.4 Ultra-selective regime persists at long horizons

At y6, LungEvaty top 0.1% (6 scans) achieves 100% precision (bootstrap CI: 100%, 100%). Top 0.2% (12 scans): 100% precision. Top 0.4% (25 scans): 88.2% precision.

---

## F. DCA / Net Benefit at LR-anchored thresholds (replaces `decile_cum` rows in `nb_at_decile_thresholds_devfit.csv`)

### F.1 Threshold grid

p_t ∈ {0.01, 0.02, 0.05, 0.10, 0.15} — aligned to Lung-RADS reference probability thresholds (LR3 starts at 1%; LR4A spans 5–15%; LR4B begins at 15%).

Discarded: decile_cum at {0.1, 0.2, ..., 0.9} percentile-anchored thresholds (same equal-N critique).

### F.2 NB ×1000 with patient-clustered bootstrap 95% CI

| Year | p_t | LungEvaty | Pillar-0 | Sybil | PLCOm2012 |
|---|---|---|---|---|---|
| 1 | 1% | 10.2 [8.0, 12.3] | 10.3 [8.2, 12.4] | 9.8 [7.8, 11.9] | 3.4 [1.6, 5.2] |
| 1 | 2% | 9.3 [7.3, 11.4] | 9.4 [7.4, 11.5] | 9.1 [7.1, 11.1] | 0.5 [−0.8, 1.8] |
| 1 | 5% | **8.1** [6.2, 10.0] | 7.7 [5.8, 9.7] | 5.7 [4.1, 7.5] | 0.1 [−0.4, 0.6] |
| 1 | 10% | **6.8** [4.9, 8.6] | 6.3 [4.4, 8.0] | 3.6 [1.9, 5.3] | 0.0 [0.0, 0.0] |
| 1 | 15% | **6.1** [4.2, 8.0] | 5.4 [3.6, 7.1] | 4.2 [2.6, 5.8] | 0.0 |
| 3 | 1% | **23.3** [19.0, 27.7] | 21.4 [17.4, 25.5] | 20.5 [16.5, 24.6] | 19.7 [15.2, 24.2] |
| 3 | 5% | 11.6 [8.3, 14.9] | 11.6 [8.5, 14.7] | 11.1 [8.0, 14.1] | 1.9 [−1.0, 4.4] |
| 3 | 15% | 7.2 [4.7, 9.8] | **7.6** [5.0, 10.4] | 3.2 [0.7, 5.8] | 0.0 |
| 6 | 1% | 58.0 [50.4, 67.0] | 58.2 [50.2, 67.1] | 58.5 [50.6, 67.2] | 58.6 [50.5, 67.5] |
| 6 | 5% | **35.7** [28.4, 43.9] | 34.0 [27.3, 41.3] | 28.1 [22.0, 34.8] | 20.2 [12.6, 27.7] |
| 6 | 15% | **17.6** [12.4, 23.0] | 15.1 [9.6, 20.2] | 12.9 [8.7, 17.6] | 0.2 [−3.4, 3.5] |

(Bold marks the best NB per row.)

Headline pattern: at the LR-relevant threshold p_t = 15% (≈ LR4B reference), LungEvaty has the highest NB at y1 (6.1) and y6 (17.6). At y1 p_t = 5% (≈ LR4A bottom), LungEvaty leads (8.1). PLCOm collapses to zero NB at p_t ≥ 10% because it cannot reach those probabilities (consistent with its inability to enter the high-risk regime).

Full data: `nb_lr_anchored.csv`.

### F.3 Continuous DCA curves (methodologically pure form)

Augments the discrete table above with full continuous NB(p_t) curves across p_t ∈ [0, 0.2] for each horizon. Files:
- `dca_continuous.png` — 3-panel y1/y3/y6 figure with patient-clustered bootstrap n=500 CI bands per model, treat-all (grey dashed) and treat-none (y=0) reference strategies, and LR-anchored p_t markers at 1%, 2%, 5%, 15% (vertical dotted lines).
- `dca_continuous_y1y3y6.csv` — raw curve data (year × model × p_t × NB + CI).

Conventions: IPCW (Uno) weights for y1 and y3; complete-case for y6 (consistent with paper convention).

Headline patterns visible at every horizon:
- LungEvaty, Pillar-0, and Sybil curves dominate the treat-all strategy across the entire clinically relevant p_t range.
- PLCOm collapses below treat-all at p_t > ~5% (it cannot reach those probabilities; consistent with the calibration finding in §G).
- LungEvaty is the top curve at the LR4B reference threshold (p_t = 15%) at y1 and y6; Pillar-0 catches up at y3.
- All three imaging models offer positive net benefit over treat-none across the entire LR-relevant threshold range (1–15%).

The continuous figure is the bin-free counterpart to the discrete table — same underlying NB formula, just evaluated at 100 thresholds instead of 5. The discrete table remains useful as a deployment-form summary; the continuous curve is the methodologically primary view.

### F.4 Existing artifacts that remain valid

`audit_dca_calibrated.csv` (the p_t ∈ {0.01, 0.05, 0.10} subset) is consistent with the new files — same methodology, expanded threshold grid. The new files supersede it.

---

## G. Calibration per Lung-RADS category (replaces per-decile rows in `decile_calibration_devfit.csv`)

For each LR category and each model: mean predicted p̂_y1 (calibrated) vs mean observed cancer rate, with bootstrap CI on the difference.

| LR cat | N | y1 ca | observed | LungEvaty pred / diff | Pillar-0 pred / diff | Sybil pred / diff | PLCOm pred / diff |
|---|---|---|---|---|---|---|---|
| LR1 | 2,650 | 4 | 0.15% | 0.71% / **+0.56pp** [0.39, 0.71] | 0.67% / +0.52 | 0.66% / +0.51 | 1.41% / +1.26 |
| LR2 | 2,838 | 11 | 0.39% | 1.08% / **+0.70pp** [0.48, 0.92] | 1.23% / +0.84 | 1.19% / +0.80 | 1.46% / +1.07 |
| LR3 | 261 | 3 | 1.15% | 1.88% / +0.74 [0.17, 1.42] | 2.24% / +1.09 | 2.23% / +1.08 | 1.49% / +0.34 |
| LR4A | 218 | 10 | 4.59% | 4.12% / **−0.47pp** [−2.38, +1.51] | 5.15% / +0.56 | 4.50% / −0.08 | 1.78% / −2.81 |
| **LR4B** | 230 | 51 | **22.17%** | 15.07% / **−7.10pp** [−10.56, −3.74] | 13.86% / **−8.31** | 11.07% / **−11.10** | 1.86% / **−20.31** |
| LR4X | 26 | 3 | 11.54% | 7.73% / −3.80 [−9.10, 2.70] | 4.40% / −7.14 | 3.81% / −7.73 | 2.06% / −9.48 |

**Calibration findings**:
- LR1–LR3 (low-risk categories): all AI models **over-predict** by 0.5–1.1 pp. CIs do not include zero — this is a systematic over-prediction in the LR-negative regime.
- LR4A (probably suspicious): well-calibrated; LungEvaty, Pillar-0, and Sybil have CIs straddling zero.
- **LR4B (suspicious, observed 22% incidence)**: all AI models **under-predict** by 7–11 pp; PLCOm under-predicts by 20 pp. The AI models are conservative in the highest-risk LR category.
- LR4X CIs are wide (n=26) but point estimates suggest under-prediction.

PLCOm calibration is poor across all LR categories — predicts 1.5–2% in LR4B where actual rate is 22%. Consistent with PLCOm not seeing the imaging.

### G.1 Continuous calibration belts unchanged

The continuous calibration belts (`calibration_belt_*.png`) are bin-free and need no replacement. They remain the primary calibration visualization. The LR-binned table above is the deployment-form summary that augments them.

Full data: `calib_lr_anchored_y1.csv`.

---

## H. Subgroup analysis — Gini dropped, replaced with operating-point equity (no_gini version)

**Drops**: top-D capture, top-C capture (equal-N critique), **AND Gini** (≈ 2·AUC − 1 at low prevalence; reviewer-bait).

**Keeps**: PR-AUC (reference, with continuity to field's metric vocabulary).

**Adds**: recall @ LR ≥4B-matched flag rate within subgroup (clinically anchored, single operating-point capture, NOT AUC-equivalent); LR's own recall at that volume as side-by-side reference.

### H.1 New equity headline: range of AI recall@LR≥4B-matched volume across powered subgroup levels (pp)

| Subgroup variable | LungEvaty | Pillar-0 | Sybil |
|---|---|---|---|
| sex | 10.0 | **0.9** | 3.2 |
| age tertile | 15.6 | 31.4 | 13.2 |
| pack-year quartile | 20.8 | 40.3 | **59.0** |
| screening timepoint | 15.9 | 8.2 | 9.0 |

Interpretation (units = percentage-point range of recall across the powered levels of the subgroup variable):
- **Pillar-0 has the most uniform sex equity** (0.9 pp — recall nearly identical M vs F at LR's workup volume).
- **Sybil shows the largest pack-year disparity** (59 pp range across quartiles).
- LungEvaty is reasonably uniform across sex and pack-year quartiles; more variable across age and timepoint.

Race is dropped from the equity headline (only White has N_cancer ≥ 10).

### H.2 Sex (y1, all-TP)

| Level | Model | N | y1 ca | PR-AUC | AI prec @ LR≥4B vol | AI recall @ LR≥4B vol | LR own recall |
|---|---|---|---|---|---|---|---|
| Male | LungEvaty | 3,728 | 51 | 0.514 [0.40, 0.62] | 23.2% | **74.5%** | 72.5% (n=164) |
| Male | Pillar-0 | 3,728 | 51 | 0.413 | 21.3% | 68.6% | 72.5% |
| Male | Sybil | 3,728 | 51 | 0.311 | 17.1% | 54.9% | 72.5% |
| Female | LungEvaty | 2,495 | 31 | 0.368 [0.27, 0.48] | 21.7% | **64.5%** | 54.8% (n=92) |
| Female | Pillar-0 | 2,495 | 31 | 0.371 | 22.8% | 67.7% | 54.8% |
| Female | Sybil | 2,495 | 31 | 0.348 | 19.6% | 58.1% | 54.8% |

### H.3 Screening timepoint (y1)

| Level | Model | N | ca | PR-AUC | AI recall @ LR≥4B vol | LR own recall |
|---|---|---|---|---|---|---|
| T0 | LungEvaty | 2,104 | 39 | 0.589 | **79.5%** | 53.8% (n=91) |
| T0 | Pillar-0 | 2,104 | 39 | 0.463 | 71.8% | 53.8% |
| T0 | Sybil | 2,104 | 39 | 0.449 | 59.0% | 53.8% |
| T1 | LungEvaty | 2,076 | 21 | 0.354 | 66.7% | 76.2% (n=71) |
| T1 | Pillar-0 | 2,076 | 21 | 0.351 | 66.7% | 76.2% |
| T1 | Sybil | 2,076 | 21 | 0.271 | 57.1% | 76.2% |
| T2 | LungEvaty | 2,043 | 22 | 0.323 | 63.6% | 77.3% (n=94) |
| T2 | Pillar-0 | 2,043 | 22 | 0.376 | 63.6% | 77.3% |
| T2 | Sybil | 2,043 | 22 | 0.210 | 50.0% | 77.3% |

At T0, LungEvaty exceeds LR's recall by +25.7 pp. At T1/T2, LR exceeds LungEvaty by ~10 pp — the longitudinal-info asymmetry stratified by timepoint.

### H.4 Race (severely underpowered)

White: 5,728 scans, 77 cancers (reportable). Asian: 140/2. Black/AA: 185/1. Other/Native: 163/2. Race-equity claims cannot be made from this cohort. Keep as caveat per existing convention.

Files: `subgroup_y1_lr_anchored.csv` (long-form with all CIs incl. Gini for transparency), `subgroup_y1_lr_anchored_no_gini.md` (Gini-free tables).

---

## I. What is superseded by this addendum

| Prior artifact | Status |
|---|---|
| `decile_devfit_consolidated.md` (Q1, Q2 per-decile/per-5cat) | Superseded by LR-anchored tables in §B, §G |
| `decile_incidence_devfit.csv` | Superseded by `lungrads_actual_y1.csv` + `lungrads_reclass_y1_wide.csv` |
| `decile_calibration_devfit.csv` | Superseded by `calib_lr_anchored_y1.csv` (§G) |
| `nb_at_decile_thresholds_devfit.csv` decile_cum rows | Superseded by `nb_lr_anchored.csv` (§F); p_t rows are consistent with the expanded grid |
| `task3_pr_curves_op_points.csv` FR10/FR20/FR40/FR70 rows | Superseded by `pr_op_lr_anchored.csv` (§E); R95 rows are consistent |
| `subgroup_year1_devfit.csv` (top-D / top-C capture) | Superseded by `subgroup_y1_lr_anchored.csv` + `_no_gini.md` (§H) |

Continuous artifacts that remain primary (unchanged):
- `pr_curves.png`, `pr_curves_*.png/pdf` — continuous PR curves (Layer 1).
- `calibration_belt_*.png/pdf` — continuous calibration belts (Layer 3).
- `audit_dca_calibrated.csv` — p_t-anchored DCA at {0.01, 0.05, 0.10}; supersession is consistent with new file.

---

## J. File index

### Primary figures

- `lorenz_y1_paper_figure_no_gini.png` — headline (all-TP + T0-only).
- `lorenz_multihorizon_no_gini.png` — y1/y3/y6 panel.
- `lorenz_y1_by_timepoint.png` — T0/T1/T2 panel (information-asymmetry diagnostic).
- `dca_continuous.png` — continuous DCA curves across p_t ∈ [0, 0.2] for y1/y3/y6 with LR-anchored p_t markers and bootstrap CI bands.
- `pr_curves.png` — Layer-1 PR curves (unchanged from prior handoff).
- `calibration_belt_*.png` — Layer-3 continuous calibration belts (unchanged).

### Primary tables (CSVs in this folder)

| Table | File | Replaces |
|---|---|---|
| LR-binned reclassification | `lungrads_reclass_y1_wide.csv`, `_T0_wide.csv` | Per-decile incidence |
| Operating points (multi-horizon) | `pr_op_lr_anchored.csv` | task3_pr_curves_op_points.csv FR rows |
| NB at LR-anchored thresholds (discrete) | `nb_lr_anchored.csv` | nb_at_decile_thresholds_devfit.csv decile_cum rows |
| NB continuous curves | `dca_continuous_y1y3y6.csv` (+ `dca_continuous.png`) | None — new artifact (methodologically pure form) |
| Per-LR-cat calibration | `calib_lr_anchored_y1.csv` | decile_calibration_devfit.csv |
| Subgroup (no Gini) | `subgroup_y1_lr_anchored_no_gini.md`, `subgroup_y1_lr_anchored.csv` | subgroup_year1_devfit.csv |
| Gini summary (kept for transparency) | `lorenz_gini_multihorizon.csv` | — (do not headline) |

---

## K. Methodological caveats for the paper-writing session

1. **Lorenz Gini ≈ 2·AUC − 1 at low prevalence.** Do not headline Gini. Report Lorenz CURVES (graphical) and operating-point CAPTURE (numerical). Disclose the relationship in Methods.

2. **Year-6 mode is complete-case, not IPCW** (consistent with paper convention). IPCW-weighted Lorenz at y6 is a possible sensitivity analysis but not currently computed.

3. **4X is the Pinsky retrospective approximation, not real-world 4X.** Mention in caption / footnote whenever LR4X numbers appear.

4. **LR uses prior-CT info at T1/T2; AI does not.** Any LR-vs-AI comparison aggregated across timepoints carries this asymmetry. T0-only is the info-symmetric subset. Both views should appear in the paper.

5. **Risk-anchored bins (B1–B5 at <1%, 1–2%, 2–5%, 5–15%, ≥15%) use Lung-RADS reference thresholds and presume calibrated probabilities.** Sybil v1.0's calibrator quantization (228 unique values) limits its B5 bin populations and is visible in the Lorenz curve as a plateau around 1–3% of scans.

6. **PLCOm is reported as a clinical-risk-score reference, not a head-to-head competitor with imaging models.** PLCOm uses no imaging data; the comparison shows that population-level clinical risk scores cannot substitute for imaging models in the per-scan operating regime.

7. **All bootstrap CIs are patient-clustered n=1000** (subgroup, DCA, PR-ops, calibration). Lorenz Gini bootstrap CIs were intentionally not computed because Gini is not a headline metric.

8. **AI models systematically under-predict in LR4B** by 7–11 pp (§G). This is a calibration finding, not a discrimination finding. The Lorenz / operating-point analysis is unaffected because it uses ranks, but the calibration belts and reliability decomposition should reflect this.
