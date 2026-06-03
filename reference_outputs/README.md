# Paper handoff — 2026-05-19

Self-contained results bundle for the paper-writing session, which has no file access. All numbers it needs are reproduced as markdown tables in this document. Detailed per-decile / per-subgroup tables are in the two consolidated markdown files in this folder.

---

## 1. Cohort

NLST test, four-way matched (LungEvaty ∩ Pillar-0 ∩ Sybil cal ∩ Sybil raw):

| Statistic | Value |
|---|---|
| Scans | 6,223 |
| Patients | 2,199 (3 screening timepoints each: T0 / T1 / T2) |
| Total cancers (any horizon) | 295 |
| Year-1 cancers | 82 (prev 1.32%) |
| Year-6 cancers | 295 (prev 6.80%, complete-case n=4,339) |

Per-horizon complete-case cohort (after `_include` filter):

| Year | N | Cancers | Prev |
|---|---|---|---|
| 1 | 6,223 | 82 | 1.32% |
| 2 | 6,206 | 130 | 2.10% |
| 3 | 6,173 | 181 | 2.93% |
| 4 | 6,114 | 222 | 3.63% |
| 5 | 5,866 | 271 | 4.62% |
| 6 | 4,339 | 295 | 6.80% |

Year-6 caveat: NLST `time_at_event` caps at 5; year-6 evaluation is complete-case.

---

## 2. Models

Four paper models. One row per model, deployment form throughout.

| Model | Architecture | Source / form |
|---|---|---|
| **LungEvaty** | citation withheld for blind review | LungEvaty test predictions, Platt-recalibrated on NLST dev per horizon |
| **Pillar-0** | Sybil 1.5 architecture | User-reproduced 3-seed ensemble, Platt-recalibrated on user-reproduced NLST dev per horizon. Reproduction validated against author scores (ΔAUC < 0.003, ΔC-index = 0.0005, per-scan corr 0.986+) |
| **Sybil** | Sybil v1.0 | Author-released calibrated outputs |
| **PLCOm2012** | Clinical logistic (Tammemägi 2013) | Computed at runtime from NLST tabular features, Platt-recalibrated on NLST dev per horizon |

M3FM is excluded.

---

## 3. Conventions

- **Horizon mode**: IPCW (Uno-style weights, KM censoring distribution) for Year 1 and Year 3. Complete-case for Year 6.
- **Bootstrap**: patient-clustered, n=1000 for incidence / calibration / Layer 1 / Layer 2 metrics; n=500 for NB. Paired across models (shared resample indices).
- **Decile binning**: 90th / 80th / ... / 10th percentile cuts computed on the held-out NLST dev fold per model, frozen, applied to test. Year-1 anchor: each scan receives one decile label D1–D10 based on its year-1 score; label propagates to year-3 / year-6 evaluation.
- **5-category aggregation (Option B, selective top)**: C1=D1, C2=D2, C3=D3+D4, C4=D5+D6+D7, C5=D8+D9+D10. Clinical reading: workup / 3-mo CT / 6-mo CT / 12-mo CT / 24-mo CT.
- **Operating-point anchors** (Layer 2):

| Anchor | Definition | Clinical reading |
|---|---|---|
| FR10 | Top 10% by score (C1 boundary) | Workup tier |
| FR20 | Top 20% (C1+C2) | Through 3-mo CT tier |
| FR40 | Top 40% (C1+C2+C3) | Through 6-mo CT tier |
| FR70 | Top 70% (C1+C2+C3+C4) | Through 12-mo CT tier |
| R95 | Threshold whose recall ≥ 95% (test-fit) | High-sensitivity safety regime |

C5 (D8+D9+D10, the lowest-risk 30%, 24-mo CT tier) is the implicit complement of FR70 — scans not flagged at FR70 fall into C5 by construction. No explicit FR anchor is reported for C5 since it corresponds to flagging the bottom rather than the top.

FR-anchors are rank-based (top-k by score, stable tiebreak).

---

## 4. Layer 1 — Global discrimination

### 4.1 Uno's C-index (IPCW, full follow-up, patient-clustered bootstrap n=1000)

| Model | C-index | 95% CI |
|---|---|---|
| LungEvaty | 0.802 | [0.767, 0.837] |
| Pillar-0 | 0.774 | [0.731, 0.817] |
| Sybil | 0.743 | [0.700, 0.788] |
| PLCOm2012 | 0.617 | [0.561, 0.670] |

### 4.2 AUC by year (deployment form, paired patient-clustered bootstrap n=1000)

| Year | LungEvaty | Pillar-0 | Sybil | PLCOm2012 |
|---|---|---|---|---|
| 1 | 0.950 [0.928, 0.969] | 0.942 [0.915, 0.964] | 0.914 [0.871, 0.947] | 0.589 [0.517, 0.661] |
| 2 | 0.905 [0.878, 0.928] | 0.882 [0.847, 0.914] | 0.858 [0.818, 0.893] | 0.607 [0.538, 0.673] |
| 3 | 0.858 [0.828, 0.885] | 0.832 [0.791, 0.871] | 0.800 [0.755, 0.839] | 0.600 [0.533, 0.661] |
| 4 | 0.823 [0.788, 0.856] | 0.804 [0.761, 0.842] | 0.767 [0.721, 0.810] | 0.604 [0.548, 0.661] |
| 5 | 0.810 [0.773, 0.845] | 0.780 [0.739, 0.822] | 0.752 [0.710, 0.796] | 0.620 [0.563, 0.673] |
| 6 | 0.805 [0.766, 0.839] | 0.776 [0.737, 0.814] | 0.743 [0.697, 0.787] | 0.625 [0.568, 0.679] |

### 4.3 PR-AUC by year (deployment form, paired patient-clustered bootstrap n=1000)

| Year | LungEvaty | Pillar-0 | Sybil | PLCOm2012 |
|---|---|---|---|---|
| 1 | 0.451 [0.338, 0.568] | 0.391 [0.273, 0.518] | 0.325 [0.223, 0.443] | 0.026 [0.016, 0.052] |
| 2 | 0.360 [0.276, 0.449] | 0.334 [0.243, 0.434] | 0.267 [0.187, 0.352] | 0.040 [0.027, 0.068] |
| 3 | 0.314 [0.246, 0.387] | 0.296 [0.219, 0.386] | 0.244 [0.174, 0.324] | 0.056 [0.039, 0.087] |
| 4 | 0.288 [0.225, 0.364] | 0.276 [0.205, 0.356] | 0.227 [0.165, 0.300] | 0.073 [0.050, 0.110] |
| 5 | 0.291 [0.230, 0.365] | 0.274 [0.210, 0.342] | 0.237 [0.178, 0.313] | 0.098 [0.065, 0.145] |
| 6 | 0.348 [0.287, 0.414] | 0.324 [0.260, 0.394] | 0.290 [0.228, 0.362] | 0.133 [0.097, 0.191] |

---

## 5. Layer 2 — Risk gradient (decile / category)

Year-1-anchored decile assignment. Each scan placed in D1–D10 by its own year-1 score (dev-fit cuts), aggregated to C1–C5 under Option B. Year-1 labels propagated to year-3 / year-6 evaluation.

### 5.1 Per-category cancer incidence (year-1 anchored, evaluated at each horizon)

#### Year 1 (IPCW, N=6,223, 82 cancers)

| Bin | LungEvaty | Pillar-0 | Sybil | PLCOm2012 |
|---|---|---|---|---|
| C1 (workup) | 0.103 [0.081, 0.128] | 0.101 [0.079, 0.126] | 0.105 [0.080, 0.130] | 0.032 [0.019, 0.047] |
| C2 (3-mo) | 0.009 [0.003, 0.017] | 0.007 [0.001, 0.015] | 0.008 [0.002, 0.016] | 0.011 [0.003, 0.019] |
| C3 (6-mo) | 0.003 [0.001, 0.007] | 0.003 [0.001, 0.007] | 0.004 [0.001, 0.007] | 0.014 [0.008, 0.021] |
| C4 (12-mo) | 0.001 [0.000, 0.003] | 0.002 [0.000, 0.004] | 0.001 [0.000, 0.003] | 0.009 [0.005, 0.014] |
| C5 (24-mo) | 0.000 | 0.000 | 0.001 [0.000, 0.003] | 0.011 [0.006, 0.017] |

#### Year 3 (IPCW)

| Bin | LungEvaty | Pillar-0 | Sybil | PLCOm2012 |
|---|---|---|---|---|
| C1 (workup) | 0.154 [0.122, 0.188] | 0.146 [0.113, 0.181] | 0.145 [0.113, 0.181] | 0.068 [0.043, 0.097] |
| C2 (3-mo) | 0.050 [0.033, 0.070] | 0.043 [0.027, 0.061] | 0.043 [0.026, 0.060] | 0.031 [0.014, 0.052] |
| C3 (6-mo) | 0.019 [0.011, 0.029] | 0.016 [0.008, 0.025] | 0.019 [0.012, 0.027] | 0.028 [0.017, 0.040] |
| C4 (12-mo) | 0.009 [0.005, 0.014] | 0.010 [0.006, 0.015] | 0.014 [0.008, 0.020] | 0.024 [0.014, 0.036] |
| C5 (24-mo) | 0.003 [0.001, 0.006] | 0.004 [0.002, 0.007] | 0.005 [0.001, 0.010] | 0.022 [0.012, 0.031] |

#### Year 6 (complete-case, N=4,339, 295 cancers)

| Bin | LungEvaty | Pillar-0 | Sybil | PLCOm2012 |
|---|---|---|---|---|
| C1 (workup) | 0.283 [0.232, 0.333] | 0.258 [0.209, 0.306] | 0.247 [0.198, 0.298] | 0.157 [0.105, 0.209] |
| C2 (3-mo) | 0.104 [0.071, 0.140] | 0.108 [0.074, 0.146] | 0.114 [0.075, 0.154] | 0.102 [0.061, 0.149] |
| C3 (6-mo) | 0.064 [0.045, 0.085] | 0.061 [0.043, 0.083] | 0.052 [0.036, 0.069] | 0.065 [0.041, 0.091] |
| C4 (12-mo) | 0.035 [0.021, 0.050] | 0.030 [0.018, 0.044] | 0.042 [0.029, 0.057] | 0.050 [0.031, 0.071] |
| C5 (24-mo) | 0.011 [0.005, 0.019] | 0.019 [0.010, 0.028] | 0.019 [0.008, 0.032] | 0.046 [0.028, 0.065] |

### 5.2 Top-C1 (= top-D1) cancer capture by horizon

Fraction of all year-$t$ cancers in the cohort that fall in C1 = D1 = the top 10% by year-1 score.

| Year | LungEvaty | Pillar-0 | Sybil | PLCOm2012 |
|---|---|---|---|---|
| 1 | 0.854 [0.772, 0.924] | 0.854 [0.771, 0.928] | 0.817 [0.730, 0.893] | 0.244 [0.148, 0.350] |
| 3 | 0.579 [0.499, 0.663] | 0.562 [0.486, 0.639] | 0.507 [0.425, 0.588] | 0.237 [0.153, 0.322] |
| 6 | 0.468 [0.400, 0.541] | 0.427 [0.356, 0.500] | 0.390 [0.317, 0.466] | 0.234 [0.158, 0.312] |

Full per-decile breakdown (D1–D10 at year 1, plus per-category at years 1/3/6) in `decile_devfit_consolidated.md` §Q1.

---

## 6. Operating points (Sp / Prec / Recall)

Five anchors × 6 horizons × 4 models. Paired patient-clustered bootstrap n=1000. Sample sizes: see cohort table in §1.

### 6.1 FR10 (top 10% — workup tier C1)

| Model | Year | Recall | Specificity | Precision |
|---|---|---|---|---|
| LungEvaty | 1 | 0.829 [0.750, 0.909] | 0.910 [0.908, 0.912] | 0.109 [0.086, 0.136] |
| Pillar-0 | 1 | 0.854 [0.771, 0.929] | 0.910 [0.908, 0.913] | 0.112 [0.088, 0.140] |
| Sybil | 1 | 0.805 [0.726, 0.888] | 0.909 [0.907, 0.912] | 0.106 [0.082, 0.134] |
| PLCOm2012 | 1 | 0.244 [0.155, 0.345] | 0.902 [0.901, 0.903] | 0.032 [0.019, 0.048] |
| LungEvaty | 2 | 0.669 [0.585, 0.743] | 0.912 [0.909, 0.915] | 0.140 [0.108, 0.174] |
| Pillar-0 | 2 | 0.669 [0.580, 0.746] | 0.912 [0.909, 0.915] | 0.140 [0.106, 0.176] |
| Sybil | 2 | 0.615 [0.520, 0.695] | 0.911 [0.908, 0.914] | 0.129 [0.099, 0.160] |
| PLCOm2012 | 2 | 0.246 [0.162, 0.349] | 0.903 [0.901, 0.905] | 0.052 [0.032, 0.076] |
| LungEvaty | 3 | 0.558 [0.489, 0.640] | 0.914 [0.911, 0.917] | 0.163 [0.127, 0.203] |
| Pillar-0 | 3 | 0.547 [0.468, 0.631] | 0.913 [0.910, 0.917] | 0.160 [0.124, 0.202] |
| Sybil | 3 | 0.503 [0.429, 0.592] | 0.912 [0.909, 0.916] | 0.147 [0.114, 0.187] |
| PLCOm2012 | 3 | 0.232 [0.158, 0.324] | 0.904 [0.902, 0.907] | 0.068 [0.045, 0.098] |
| LungEvaty | 4 | 0.491 [0.420, 0.564] | 0.915 [0.911, 0.918] | 0.178 [0.140, 0.220] |
| Pillar-0 | 4 | 0.482 [0.415, 0.556] | 0.914 [0.911, 0.918] | 0.175 [0.140, 0.216] |
| Sybil | 4 | 0.432 [0.356, 0.505] | 0.912 [0.909, 0.916] | 0.157 [0.122, 0.195] |
| PLCOm2012 | 4 | 0.230 [0.150, 0.305] | 0.905 [0.902, 0.908] | 0.083 [0.052, 0.117] |
| LungEvaty | 5 | 0.465 [0.396, 0.543] | 0.918 [0.914, 0.922] | 0.215 [0.173, 0.259] |
| Pillar-0 | 5 | 0.432 [0.363, 0.508] | 0.916 [0.912, 0.920] | 0.199 [0.156, 0.242] |
| Sybil | 5 | 0.410 [0.337, 0.475] | 0.915 [0.911, 0.919] | 0.189 [0.147, 0.229] |
| PLCOm2012 | 5 | 0.236 [0.160, 0.306] | 0.907 [0.903, 0.910] | 0.109 [0.071, 0.145] |
| LungEvaty | 6 | 0.441 [0.377, 0.508] | 0.925 [0.919, 0.931] | 0.300 [0.244, 0.363] |
| Pillar-0 | 6 | 0.414 [0.350, 0.491] | 0.923 [0.918, 0.929] | 0.281 [0.227, 0.344] |
| Sybil | 6 | 0.383 [0.318, 0.450] | 0.921 [0.915, 0.926] | 0.260 [0.207, 0.316] |
| PLCOm2012 | 6 | 0.217 [0.158, 0.297] | 0.909 [0.904, 0.915] | 0.147 [0.105, 0.209] |

### 6.2 FR20 (top 20% — through 3-mo CT tier C1+C2)

| Model | Year | Recall | Specificity | Precision |
|---|---|---|---|---|
| LungEvaty | 1 | 0.927 [0.850, 0.974] | 0.810 [0.807, 0.812] | 0.061 [0.047, 0.075] |
| Pillar-0 | 1 | 0.902 [0.836, 0.968] | 0.809 [0.807, 0.812] | 0.059 [0.047, 0.073] |
| Sybil | 1 | 0.878 [0.796, 0.942] | 0.809 [0.807, 0.811] | 0.058 [0.045, 0.071] |
| PLCOm2012 | 1 | 0.329 [0.221, 0.430] | 0.802 [0.800, 0.803] | 0.022 [0.013, 0.031] |
| LungEvaty | 2 | 0.815 [0.747, 0.882] | 0.813 [0.810, 0.816] | 0.085 [0.067, 0.105] |
| Pillar-0 | 2 | 0.792 [0.718, 0.861] | 0.813 [0.810, 0.816] | 0.083 [0.064, 0.102] |
| Sybil | 2 | 0.754 [0.673, 0.827] | 0.812 [0.809, 0.815] | 0.079 [0.061, 0.099] |
| PLCOm2012 | 2 | 0.346 [0.248, 0.456] | 0.803 [0.801, 0.806] | 0.036 [0.024, 0.051] |
| LungEvaty | 3 | 0.724 [0.655, 0.799] | 0.816 [0.812, 0.820] | 0.106 [0.085, 0.130] |
| Pillar-0 | 3 | 0.696 [0.615, 0.767] | 0.815 [0.811, 0.819] | 0.102 [0.080, 0.125] |
| Sybil | 3 | 0.657 [0.577, 0.740] | 0.814 [0.810, 0.818] | 0.096 [0.076, 0.119] |
| PLCOm2012 | 3 | 0.337 [0.250, 0.435] | 0.804 [0.801, 0.807] | 0.049 [0.035, 0.067] |
| LungEvaty | 4 | 0.653 [0.587, 0.728] | 0.817 [0.813, 0.821] | 0.119 [0.095, 0.143] |
| Pillar-0 | 4 | 0.644 [0.572, 0.708] | 0.817 [0.813, 0.820] | 0.117 [0.093, 0.139] |
| Sybil | 4 | 0.590 [0.519, 0.667] | 0.815 [0.811, 0.818] | 0.107 [0.086, 0.130] |
| PLCOm2012 | 4 | 0.333 [0.254, 0.425] | 0.805 [0.802, 0.809] | 0.061 [0.044, 0.082] |
| LungEvaty | 5 | 0.616 [0.546, 0.690] | 0.820 [0.816, 0.825] | 0.142 [0.116, 0.170] |
| Pillar-0 | 5 | 0.594 [0.520, 0.663] | 0.819 [0.814, 0.823] | 0.137 [0.109, 0.163] |
| Sybil | 5 | 0.572 [0.495, 0.644] | 0.818 [0.813, 0.822] | 0.132 [0.104, 0.157] |
| PLCOm2012 | 5 | 0.376 [0.282, 0.447] | 0.808 [0.804, 0.812] | 0.087 [0.062, 0.108] |
| LungEvaty | 6 | 0.590 [0.526, 0.664] | 0.828 [0.822, 0.835] | 0.200 [0.165, 0.240] |
| Pillar-0 | 6 | 0.573 [0.492, 0.646] | 0.827 [0.821, 0.833] | 0.195 [0.158, 0.233] |
| Sybil | 6 | 0.546 [0.476, 0.624] | 0.825 [0.819, 0.832] | 0.185 [0.151, 0.223] |
| PLCOm2012 | 6 | 0.369 [0.295, 0.456] | 0.812 [0.807, 0.819] | 0.126 [0.096, 0.162] |

### 6.3 FR40 (top 40% — through 6-mo CT tier C1+C2+C3)

| Model | Year | Recall | Specificity | Precision |
|---|---|---|---|---|
| LungEvaty | 1 | 0.976 [0.937, 1.000] | 0.608 [0.606, 0.609] | 0.032 [0.025, 0.040] |
| Pillar-0 | 1 | 0.951 [0.901, 1.000] | 0.607 [0.606, 0.609] | 0.031 [0.024, 0.039] |
| Sybil | 1 | 0.939 [0.886, 0.988] | 0.607 [0.605, 0.609] | 0.031 [0.024, 0.038] |
| PLCOm2012 | 1 | 0.537 [0.414, 0.644] | 0.602 [0.600, 0.603] | 0.018 [0.012, 0.023] |
| LungEvaty | 2 | 0.946 [0.904, 0.984] | 0.612 [0.609, 0.614] | 0.050 [0.039, 0.061] |
| Pillar-0 | 2 | 0.877 [0.817, 0.936] | 0.610 [0.608, 0.613] | 0.046 [0.036, 0.057] |
| Sybil | 2 | 0.869 [0.792, 0.928] | 0.610 [0.607, 0.613] | 0.046 [0.036, 0.057] |
| PLCOm2012 | 2 | 0.554 [0.442, 0.664] | 0.603 [0.601, 0.606] | 0.029 [0.021, 0.037] |
| LungEvaty | 3 | 0.878 [0.826, 0.925] | 0.614 [0.611, 0.618] | 0.064 [0.051, 0.078] |
| Pillar-0 | 3 | 0.818 [0.756, 0.884] | 0.612 [0.610, 0.616] | 0.060 [0.048, 0.074] |
| Sybil | 3 | 0.779 [0.709, 0.849] | 0.611 [0.609, 0.614] | 0.057 [0.046, 0.070] |
| PLCOm2012 | 3 | 0.536 [0.442, 0.637] | 0.604 [0.601, 0.607] | 0.039 [0.029, 0.050] |
| LungEvaty | 4 | 0.833 [0.775, 0.884] | 0.616 [0.613, 0.620] | 0.076 [0.061, 0.091] |
| Pillar-0 | 4 | 0.788 [0.723, 0.849] | 0.615 [0.611, 0.618] | 0.072 [0.058, 0.086] |
| Sybil | 4 | 0.739 [0.663, 0.794] | 0.613 [0.609, 0.616] | 0.067 [0.054, 0.079] |
| PLCOm2012 | 4 | 0.550 [0.452, 0.643] | 0.606 [0.602, 0.609] | 0.050 [0.039, 0.063] |
| LungEvaty | 5 | 0.815 [0.751, 0.874] | 0.620 [0.615, 0.625] | 0.094 [0.077, 0.111] |
| Pillar-0 | 5 | 0.764 [0.699, 0.826] | 0.618 [0.613, 0.622] | 0.088 [0.072, 0.105] |
| Sybil | 5 | 0.712 [0.640, 0.779] | 0.615 [0.611, 0.619] | 0.082 [0.066, 0.097] |
| PLCOm2012 | 5 | 0.583 [0.480, 0.661] | 0.609 [0.604, 0.613] | 0.067 [0.051, 0.081] |
| LungEvaty | 6 | 0.800 [0.739, 0.860] | 0.629 [0.623, 0.636] | 0.136 [0.111, 0.161] |
| Pillar-0 | 6 | 0.759 [0.700, 0.824] | 0.626 [0.620, 0.633] | 0.129 [0.107, 0.154] |
| Sybil | 6 | 0.688 [0.623, 0.761] | 0.621 [0.615, 0.627] | 0.117 [0.096, 0.138] |
| PLCOm2012 | 6 | 0.580 [0.496, 0.672] | 0.613 [0.607, 0.620] | 0.099 [0.078, 0.121] |

### 6.4 FR70 (top 70% — through 12-mo CT tier C1+C2+C3+C4)

| Model | Year | Recall | Specificity | Precision |
|---|---|---|---|---|
| LungEvaty | 1 | 1.000 [1.000, 1.000] | 0.304 [0.303, 0.305] | 0.019 [0.015, 0.023] |
| Pillar-0 | 1 | 1.000 [1.000, 1.000] | 0.304 [0.303, 0.305] | 0.019 [0.015, 0.023] |
| Sybil | 1 | 0.963 [0.932, 1.000] | 0.303 [0.303, 0.305] | 0.018 [0.014, 0.023] |
| PLCOm2012 | 1 | 0.756 [0.657, 0.851] | 0.301 [0.299, 0.302] | 0.014 [0.011, 0.018] |
| LungEvaty | 2 | 0.992 [0.975, 1.000] | 0.306 [0.305, 0.308] | 0.030 [0.023, 0.037] |
| Pillar-0 | 2 | 0.985 [0.956, 1.000] | 0.306 [0.305, 0.307] | 0.029 [0.023, 0.036] |
| Sybil | 2 | 0.969 [0.925, 1.000] | 0.306 [0.304, 0.307] | 0.029 [0.023, 0.036] |
| PLCOm2012 | 2 | 0.792 [0.703, 0.872] | 0.302 [0.300, 0.304] | 0.024 [0.018, 0.030] |
| LungEvaty | 3 | 0.972 [0.946, 0.994] | 0.308 [0.306, 0.310] | 0.041 [0.033, 0.049] |
| Pillar-0 | 3 | 0.961 [0.917, 0.994] | 0.308 [0.306, 0.310] | 0.040 [0.032, 0.049] |
| Sybil | 3 | 0.945 [0.886, 0.978] | 0.307 [0.305, 0.309] | 0.040 [0.031, 0.048] |
| PLCOm2012 | 3 | 0.790 [0.699, 0.871] | 0.303 [0.300, 0.305] | 0.033 [0.026, 0.041] |
| LungEvaty | 4 | 0.955 [0.922, 0.983] | 0.310 [0.308, 0.312] | 0.050 [0.040, 0.059] |
| Pillar-0 | 4 | 0.941 [0.894, 0.978] | 0.309 [0.307, 0.311] | 0.049 [0.040, 0.059] |
| Sybil | 4 | 0.928 [0.874, 0.961] | 0.309 [0.306, 0.310] | 0.048 [0.039, 0.057] |
| PLCOm2012 | 4 | 0.793 [0.708, 0.864] | 0.303 [0.300, 0.306] | 0.041 [0.033, 0.050] |
| LungEvaty | 5 | 0.952 [0.918, 0.983] | 0.312 [0.309, 0.315] | 0.063 [0.052, 0.074] |
| Pillar-0 | 5 | 0.923 [0.876, 0.963] | 0.311 [0.308, 0.314] | 0.061 [0.050, 0.072] |
| Sybil | 5 | 0.911 [0.858, 0.953] | 0.310 [0.307, 0.313] | 0.060 [0.049, 0.070] |
| PLCOm2012 | 5 | 0.797 [0.725, 0.871] | 0.305 [0.301, 0.308] | 0.053 [0.042, 0.064] |
| LungEvaty | 6 | 0.949 [0.909, 0.981] | 0.318 [0.314, 0.322] | 0.092 [0.077, 0.108] |
| Pillar-0 | 6 | 0.915 [0.870, 0.957] | 0.316 [0.312, 0.320] | 0.089 [0.075, 0.105] |
| Sybil | 6 | 0.898 [0.849, 0.949] | 0.314 [0.310, 0.319] | 0.087 [0.073, 0.103] |
| PLCOm2012 | 6 | 0.797 [0.729, 0.867] | 0.307 [0.302, 0.312] | 0.077 [0.063, 0.094] |

### 6.5 R95 (recall ≥ 95% — high-sensitivity safety regime)

| Model | Year | Recall | Specificity | Precision |
|---|---|---|---|---|
| LungEvaty | 1 | 0.951 [0.950, 0.987] | 0.723 [0.573, 0.842] | 0.044 [0.027, 0.078] |
| Pillar-0 | 1 | 0.951 [0.950, 0.989] | 0.660 [0.527, 0.875] | 0.036 [0.023, 0.091] |
| Sybil | 1 | 0.951 [0.950, 1.000] | 0.536 [0.000, 0.783] | 0.027 [0.012, 0.055] |
| PLCOm2012 | 1 | 0.951 [0.950, 1.000] | 0.035 [0.009, 0.183] | 0.013 [0.011, 0.017] |
| LungEvaty | 2 | 0.954 [0.950, 0.974] | 0.610 [0.494, 0.704] | 0.050 [0.035, 0.068] |
| Pillar-0 | 2 | 0.954 [0.950, 0.973] | 0.445 [0.311, 0.576] | 0.035 [0.025, 0.049] |
| Sybil | 2 | 0.969 [0.951, 1.000] | 0.392 [0.064, 0.573] | 0.033 [0.020, 0.045] |
| PLCOm2012 | 2 | 0.954 [0.950, 1.000] | 0.057 [0.010, 0.200] | 0.021 [0.017, 0.027] |
| LungEvaty | 3 | 0.950 [0.950, 0.966] | 0.519 [0.285, 0.546] | 0.056 [0.037, 0.067] |
| Pillar-0 | 3 | 0.950 [0.950, 0.967] | 0.326 [0.102, 0.491] | 0.041 [0.030, 0.055] |
| Sybil | 3 | 0.950 [0.950, 0.984] | 0.261 [0.129, 0.399] | 0.037 [0.029, 0.047] |
| PLCOm2012 | 3 | 0.956 [0.950, 1.000] | 0.057 [0.010, 0.202] | 0.030 [0.024, 0.037] |
| LungEvaty | 4 | 0.950 [0.950, 0.965] | 0.386 [0.225, 0.502] | 0.055 [0.039, 0.073] |
| Pillar-0 | 4 | 0.950 [0.950, 0.963] | 0.296 [0.129, 0.425] | 0.048 [0.036, 0.061] |
| Sybil | 4 | 0.959 [0.950, 0.992] | 0.192 [0.106, 0.333] | 0.043 [0.035, 0.054] |
| PLCOm2012 | 4 | 0.955 [0.950, 0.981] | 0.058 [0.011, 0.199] | 0.037 [0.030, 0.046] |
| LungEvaty | 5 | 0.952 [0.950, 0.962] | 0.340 [0.216, 0.489] | 0.065 [0.050, 0.086] |
| Pillar-0 | 5 | 0.952 [0.950, 0.961] | 0.215 [0.116, 0.343] | 0.055 [0.044, 0.070] |
| Sybil | 5 | 0.978 [0.950, 0.991] | 0.115 [0.104, 0.277] | 0.051 [0.043, 0.065] |
| PLCOm2012 | 5 | 0.956 [0.950, 0.979] | 0.058 [0.012, 0.201] | 0.047 [0.039, 0.058] |
| LungEvaty | 6 | 0.953 [0.950, 0.961] | 0.305 [0.212, 0.474] | 0.091 [0.072, 0.122] |
| Pillar-0 | 6 | 0.953 [0.950, 0.961] | 0.205 [0.118, 0.337] | 0.080 [0.066, 0.102] |
| Sybil | 6 | 0.963 [0.950, 0.977] | 0.142 [0.088, 0.296] | 0.076 [0.064, 0.096] |
| PLCOm2012 | 6 | 0.956 [0.950, 0.978] | 0.059 [0.013, 0.212] | 0.069 [0.059, 0.085] |

---

## 7. Layer 3 — Calibration (Brier decomposition, deployment form)

`Brier = Reliability − Resolution + Uncertainty`. Lower reliability = better calibration; higher resolution = better discrimination. Patient-clustered bootstrap n=1000.

### 7.1 Brier score

| Year | LungEvaty | Pillar-0 | Sybil | PLCOm2012 |
|---|---|---|---|---|
| 1 | 0.0095 [0.0074, 0.0118] | 0.0100 [0.0080, 0.0122] | 0.0107 [0.0087, 0.0130] | 0.0130 [0.0102, 0.0162] |
| 2 | 0.0168 [0.0137, 0.0203] | 0.0168 [0.0137, 0.0203] | 0.0181 [0.0147, 0.0217] | 0.0204 [0.0166, 0.0246] |
| 3 | 0.0242 [0.0198, 0.0286] | 0.0243 [0.0197, 0.0289] | 0.0255 [0.0208, 0.0304] | 0.0283 [0.0228, 0.0339] |
| 4 | 0.0306 [0.0255, 0.0358] | 0.0306 [0.0258, 0.0359] | 0.0320 [0.0267, 0.0376] | 0.0347 [0.0292, 0.0415] |
| 5 | 0.0384 [0.0324, 0.0453] | 0.0389 [0.0329, 0.0457] | 0.0401 [0.0341, 0.0473] | 0.0435 [0.0371, 0.0511] |
| 6 | 0.0536 [0.0456, 0.0617] | 0.0546 [0.0462, 0.0631] | 0.0564 [0.0475, 0.0652] | 0.0622 [0.0525, 0.0719] |

### 7.2 Reliability

| Year | LungEvaty | Pillar-0 | Sybil | PLCOm2012 |
|---|---|---|---|---|
| 1 | 0.0002 [0.0001, 0.0010] | 0.0006 [0.0003, 0.0016] | 0.0004 [0.0003, 0.0010] | 0.0000 [0.0000, 0.0000] |
| 2 | 0.0006 [0.0003, 0.0017] | 0.0006 [0.0003, 0.0017] | 0.0009 [0.0006, 0.0018] | 0.0000 [0.0000, 0.0001] |
| 3 | 0.0005 [0.0002, 0.0016] | 0.0007 [0.0004, 0.0018] | 0.0009 [0.0006, 0.0019] | 0.0000 [0.0000, 0.0002] |
| 4 | 0.0007 [0.0003, 0.0018] | 0.0007 [0.0004, 0.0019] | 0.0009 [0.0005, 0.0019] | 0.0000 [0.0000, 0.0002] |
| 5 | 0.0004 [0.0002, 0.0015] | 0.0006 [0.0004, 0.0017] | 0.0008 [0.0005, 0.0019] | 0.0002 [0.0000, 0.0010] |
| 6 | 0.0006 [0.0003, 0.0020] | 0.0009 [0.0005, 0.0025] | 0.0010 [0.0006, 0.0025] | 0.0002 [0.0001, 0.0014] |

### 7.3 Resolution

| Year | LungEvaty | Pillar-0 | Sybil | PLCOm2012 |
|---|---|---|---|---|
| 1 | 0.0036 [0.0023, 0.0056] | 0.0035 [0.0022, 0.0055] | 0.0026 [0.0015, 0.0042] | 0.0000 [0.0000, 0.0000] |
| 2 | 0.0042 [0.0028, 0.0062] | 0.0043 [0.0027, 0.0064] | 0.0031 [0.0018, 0.0048] | 0.0000 [0.0000, 0.0000] |
| 3 | 0.0046 [0.0031, 0.0068] | 0.0048 [0.0031, 0.0073] | 0.0035 [0.0020, 0.0055] | 0.0001 [0.0000, 0.0004] |
| 4 | 0.0047 [0.0031, 0.0072] | 0.0049 [0.0031, 0.0075] | 0.0036 [0.0023, 0.0059] | 0.0002 [0.0000, 0.0006] |
| 5 | 0.0057 [0.0040, 0.0085] | 0.0054 [0.0035, 0.0081] | 0.0045 [0.0029, 0.0070] | 0.0005 [0.0000, 0.0017] |
| 6 | 0.0096 [0.0067, 0.0135] | 0.0093 [0.0062, 0.0131] | 0.0075 [0.0049, 0.0112] | 0.0015 [0.0005, 0.0038] |

### 7.4 Per-decile calibration (top and bottom)

Top-decile (D1) predicted-vs-observed at year 1, deployment form:

| Model | D1 pred | D1 obs | D1 diff |
|---|---|---|---|
| LungEvaty | 0.122 | 0.103 | −0.019 [−0.041, 0.004] |
| Pillar-0 | 0.122 | 0.101 | −0.020 [−0.042, 0.005] |
| Sybil | 0.120 | 0.105 | −0.015 [−0.040, 0.012] |
| PLCOm2012 | 0.037 | 0.032 | −0.005 [−0.018, 0.012] |

Full per-decile / per-horizon calibration in `decile_devfit_consolidated.md` §Q2.

---

## 8. Layer 4 — Decision-theoretic value (NB at flag-rate and p_t anchors)

NB tables at all six anchors (flag-rate 10/20/30/.../90 + p_t 1/5/10) × 3 horizons in `decile_devfit_consolidated.md` §Q3. Headline NB at FR10 (workup tier) by horizon:

| Year | LungEvaty | Pillar-0 | Sybil | PLCOm2012 | Treat-all (ref) |
|---|---|---|---|---|---|
| 1 | 0.0096 [0.007, 0.012] | 0.0094 [0.007, 0.012] | 0.0089 [0.007, 0.012] | 0.0007 [−0.001, 0.002] | +0.0040 |
| 3 | 0.0188 [0.014, 0.025] | 0.0173 [0.013, 0.022] | 0.0146 [0.010, 0.020] | 0.0050 [0.000, 0.010] | +0.0201 |
| 6 | 0.0532 [0.041, 0.064] | 0.0481 [0.036, 0.060] | 0.0481 [0.036, 0.060] | 0.0406 [0.029, 0.053] | +0.0594 |

NB at p_t=5% by horizon:

| Year | LungEvaty | Pillar-0 | Sybil | PLCOm2012 |
|---|---|---|---|---|
| 1 | 0.0081 [0.006, 0.011] | 0.0077 [0.005, 0.011] | 0.0057 [0.004, 0.008] | 0.0001 [−0.001, 0.001] |
| 3 | 0.0117 [0.008, 0.016] | 0.0116 [0.008, 0.016] | 0.0111 [0.007, 0.016] | 0.0019 [−0.002, 0.005] |
| 6 | 0.0357 [0.026, 0.046] | 0.0335 [0.024, 0.044] | 0.0281 [0.020, 0.038] | 0.0203 [0.010, 0.031] |

---

## 9. Layer 5 — Subgroups

Full subgroup table in `decile_devfit_consolidated.md` §Q4. Headline year-1 top-D1 capture:

| Subgroup level | LungEvaty | Pillar-0 | Sybil | N_ca |
|---|---|---|---|---|
| Male | 0.902 [0.81, 0.98] | 0.902 [0.79, 0.98] | 0.843 [0.74, 0.94] | 51 |
| Female | 0.742 [0.57, 0.88] | 0.742 [0.59, 0.90] | 0.774 [0.61, 0.92] | 31 |
| Age ≤58 | 0.789 [0.59, 0.95] | 0.789 [0.61, 1.00] | 0.684 [0.47, 0.89] | 19 |
| Age 58–63 | 0.840 [0.68, 0.97] | 0.920 [0.79, 1.00] | 0.800 [0.62, 0.95] | 25 |
| Age >63 | 0.868 [0.76, 0.96] | 0.789 [0.67, 0.93] | 0.895 [0.78, 0.97] | 38 |
| PY Q1 (≤40) | 0.875 [0.67, 1.00] | 0.875 [0.65, 1.00] | 0.875 [0.67, 1.00] | 16 |
| PY Q2 (40–50) | 0.833 [0.63, 1.00] | 0.833 [0.64, 1.00] | 0.778 [0.57, 0.95] | 18 |
| PY Q3 (50–68) | 0.750 [0.54, 0.94] | 0.625 [0.42, 0.90] | 0.750 [0.53, 0.93] | 16 |
| PY Q4 (>68) | 0.875 [0.76, 0.97] | 0.875 [0.74, 1.00] | 0.844 [0.70, 0.96] | 32 |
| TP 0 | 0.846 [0.73, 0.95] | 0.897 [0.77, 0.97] | 0.846 [0.72, 0.94] | 39 |
| TP 1 | 0.857 [0.68, 1.00] | 0.810 [0.64, 0.96] | 0.810 [0.63, 0.96] | 21 |
| TP 2 | 0.818 [0.65, 0.96] | 0.818 [0.63, 0.95] | 0.773 [0.58, 0.93] | 22 |

Race subgroups: White n=5,720 / 77 cancers (capture 0.844 / 0.844 / 0.818); Black/AA n=185 / 1; Asian n=139 / 2; Other/Native n=163 / 2 — three latter degenerate.

PLCOm2012 top-D1 capture in well-powered subgroups ranges 0.06–0.47 — consistently lowest of the four models.

---

## 10. Figures

**`pr_curves.{pdf, png}`** — 2 × 3 panel PR figure. Top row: linear-scale PR curves. Bottom row: log–log magnification of the high-recall regime. Columns: Year 1 / Year 3 / Year 6. Curves: LungEvaty, Pillar-0, Sybil, PLCOm2012. Operating points marked: flag 10 / 20 / 40 / 70 % plus recall ≥ 95 %. Prevalence dotted reference per panel.

**`reliability_diagrams.{pdf, png}`** — 1 × 3 panel reliability figure for the three imaging models on their un-calibrated sigmoid outputs. Equal-width predicted-probability bins (10 bins on [0, 1]), the canonical calibration-plot convention — bins span the full predicted-probability range so miscalibration in the high-predicted tail is directly visible (predicted values up to ~0.85 for Pillar-0 / Sybil). Each panel: mean predicted (x) vs observed event rate (y) per bin, dashed diagonal = perfect calibration. Columns: Year 1 / 3 / 6. Brier per model in legend. LungEvaty closest to diagonal across all horizons; Pillar-0 and Sybil over-predict the high-risk tail. PLCOm omitted.

---

## 11. Detail markdown reports

For full per-decile / per-category / per-horizon tables (Q1, Q2, Q3, Q4) — paste these into the paper-writing session alongside this README:

- **`decile_devfit_consolidated.md`** — deployment-faithful (dev-fit cuts). The master reference.
- **`decile_testfit_consolidated.md`** — own-distribution (test-fit) sensitivity check.

---

## 12. Caveats to surface in prose

1. **Sybil decile collapse**: the author-released calibrator outputs only 228 unique values across the 6,694-scan dev fold. Decile D6 and D10 are empty on test (50/60 and 80/90 percentile cut points collide). This is a deployment property of Sybil's released calibrator, not an analysis artifact.

2. **Year-6 evaluation is complete-case**: NLST `time_at_event` caps at 5; IPCW would degenerate at year 6 with the current rule. n=4,339 at year 6.

3. **Pillar-0 calibration is fit on user-reproduced dev predictions** (not author dev predictions, which are not released). Reproduction faithfulness on test (ΔAUC < 0.003, ΔC-index = 0.0005) supports that the calibrator transfers comparably.

4. **Race subgroup analysis is structurally limited** — N_ca = 1, 2, 2 for Black/AA, Asian, Other/Native respectively. Inability to evaluate equity in non-White subgroups is itself a finding and a structural limit of NLST.

5. **Subgroup cells with N_ca < 5** have degenerate CIs; do not over-interpret. Reported because inability to evaluate is informative.

---

## 13. Settled decisions (do not relitigate)

- IPCW DCA is primary; complete-case is the secondary sensitivity check.
- IBS is deliberately excluded.
- One row per model in deployment form (no raw/calibrated split shown anywhere).
- M3FM is excluded.
- Bootstrap n=1000 (Layer 1, Layer 2, incidence, calibration); n=500 (NB).
- The four paper models are LungEvaty, Pillar-0, Sybil, PLCOm2012. (Pillar-0 and Sybil are different models — Pillar-0 is built on the Sybil 1.5 architecture; Sybil is the v1.0 release.)
