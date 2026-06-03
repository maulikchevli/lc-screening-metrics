#!/usr/bin/env python3
"""Four-way comparison: Lungevity, Sybil 1.5, Sybil (v1.0) raw, Sybil (v1.0) calibrated.

Now that we have BOTH the author-calibrated and the raw (pre-cal) Sybil v1.0
predictions, we can:
  (a) compare raw v1.0 to raw v1.5 — a fair model-vs-model contrast that
      is not confounded by the authors' shipping decision to drop calibration
  (b) compare raw v1.0 to calibrated v1.0 — quantify what the post-hoc
      calibration step actually contributes

Output: eval_output/four_way_metrics.csv  (per-year metrics for all 4 series)
        eval_output/four_way_comp_df.csv  (merged DataFrame for downstream use)
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import (
    load_predictions, load_participant, load_lung_cancer, merge_data,
    load_series_match_maps, load_sybil_predictions, build_comparison_df,
    _include, _brier_decomposition,
    PREDS_PATH, PARTICIPANT_CSV, LUNG_CANCER_CSV, OUTPUT_DIR,
    SYBIL_DIR, SYBIL_SEEDS, SERIES_MATCH_CSV, MAX_FOLLOWUP,
)
from vital.metrics import concordance_index, get_censoring_dist


SYBIL_V1_CAL_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sybil", "sybil_predictions.json")
SYBIL_V1_RAW_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sybil", "sybil_predictions_raw.json")
PILLAR_USER_TEST_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions_pillar_test.csv")


def _load_user_pillar():
    """User-reproduced Pillar-0 (3-seed ensemble). Schema matches what
    `load_sybil_predictions` used to return so downstream code consuming
    `sybil_risk_{1..6}` works unchanged. See `pillar0_reproduction` memory."""
    df = pd.read_csv(PILLAR_USER_TEST_CSV)
    df["series"] = df["series_uid"].astype(str)
    df["pid"] = df["patient_id"].astype(str)
    df["screen_timepoint"] = df["timepoint"].astype(int)
    df["pid_tp"] = df["pid"] + "_" + df["screen_timepoint"].astype(str)
    for yr in range(1, 7):
        df[f"sybil_risk_{yr}"] = df[f"prob_y{yr}"].astype(float)
    df = df.drop_duplicates("series").copy()
    keep = ["series", "pid_tp"] + [f"sybil_risk_{yr}" for yr in range(1, 7)]
    return df[keep]


def _load_sv1(path, prefix):
    """Load a Sybil v1.0 JSON and expand cancer_risk into per-year columns
    with the given prefix (e.g., 'sv1c_' for calibrated, 'sv1r_' for raw)."""
    with open(path) as f:
        df = pd.DataFrame(json.load(f))
    df["pid"] = df["pid"].astype(str)
    df["screen_timepoint"] = df["screen_timepoint"].astype(int)
    for yr in range(1, 7):
        df[f"{prefix}risk_{yr}"] = df["cancer_risk"].apply(
            lambda x: float(x[yr - 1]))
    df = df.drop_duplicates("series").copy()
    df["pid_tp"] = df["pid"] + "_" + df["screen_timepoint"].astype(str)
    return df


def build_four_way():
    print("Loading Lungevity test …")
    preds = load_predictions(PREDS_PATH)
    part = load_participant(PARTICIPANT_CSV)
    lc = load_lung_cancer(LUNG_CANCER_CSV)
    df_lung = merge_data(preds, part, lc)
    df_lung["pid_tp"] = df_lung["pid"].astype(str) + "_" + \
                       df_lung["screen_timepoint"].astype(int).astype(str)

    print("Loading Pillar-0 (user-reproduced 3-seed ensemble) …")
    pillar = _load_user_pillar()
    print(f"  user-Pillar-0: {len(pillar)} rows")

    print("Loading Sybil v1.0 (calibrated and raw) …")
    sv1c = _load_sv1(SYBIL_V1_CAL_JSON, "sv1c_")
    sv1r = _load_sv1(SYBIL_V1_RAW_JSON, "sv1r_")
    print(f"  v1.0 calibrated: {len(sv1c)} rows, v1.0 raw: {len(sv1r)} rows")

    def _attach(base, src_df, prefix):
        """Two-strategy attach (series → pid_tp fallback) of risk columns."""
        cols = ["series"] + [f"{prefix}risk_{yr}" for yr in range(1, 7)]
        a2 = base.merge(src_df[cols], on="series", how="left")
        miss = a2[f"{prefix}risk_1"].isna()
        by_pidtp = src_df[["pid_tp"]
                          + [f"{prefix}risk_{yr}" for yr in range(1, 7)]
                         ].drop_duplicates("pid_tp")
        fb = a2.loc[miss, ["pid_tp"]].merge(by_pidtp, on="pid_tp", how="left")
        for yr in range(1, 7):
            col = f"{prefix}risk_{yr}"
            a2.loc[miss, col] = fb[col].values
        return a2

    print("  Attaching Pillar-0 …")
    a = _attach(df_lung, pillar, "sybil_")
    print("  Attaching Sybil v1.0 calibrated …")
    a = _attach(a, sv1c, "sv1c_")
    print("  Attaching Sybil v1.0 raw …")
    a = _attach(a, sv1r, "sv1r_")

    # Drop any rows that don't have all sets of predictions
    final = a.dropna(subset=["sybil_risk_1", "sv1c_risk_1", "sv1r_risk_1"]).copy()
    print(f"\n  Four-way matched: {len(final)} scans, "
          f"{final['pid'].nunique()} pids, "
          f"{int(final['gold'].sum())} cancers")
    return final


def compute_metrics(comp):
    rows = []
    for year in range(1, MAX_FOLLOWUP + 1):
        fu = year - 1
        s_l, s_15, s_v1c, s_v1r, labels = [], [], [], [], []
        for _, r in comp.iterrows():
            inc, lbl = _include(r["time_at_event"], r["gold"], fu)
            if inc:
                s_l.append(r["risk_score"])
                s_15.append(r[f"sybil_risk_{year}"])
                s_v1c.append(r[f"sv1c_risk_{year}"])
                s_v1r.append(r[f"sv1r_risk_{year}"])
                labels.append(lbl)
        if sum(labels) == 0:
            continue
        sl, s15, svc, svr = map(np.array, (s_l, s_15, s_v1c, s_v1r))
        y = np.array(labels)
        # Drop rows where any model is missing a prediction for this year
        # (raw v1.0 has 21 records with NaN at year 5/6).
        mask = ~(np.isnan(sl) | np.isnan(s15) | np.isnan(svc) | np.isnan(svr))
        sl, s15, svc, svr, y = sl[mask], s15[mask], svc[mask], svr[mask], y[mask]
        if y.sum() == 0:
            continue
        row = {"Year": year, "N": len(y), "N_cancer": int(y.sum())}
        for name, s in [("Lung", sl), ("Sybil15", s15),
                        ("Sybilv1_cal", svc), ("Sybilv1_raw", svr)]:
            sc = np.clip(s, 1e-7, 1 - 1e-7)
            row[f"{name}_AUC"] = roc_auc_score(y, s)
            row[f"{name}_PRAUC"] = average_precision_score(y, s)
            row[f"{name}_Brier"] = brier_score_loss(y, sc)
            rel, res, _ = _brier_decomposition(y, sc)
            row[f"{name}_Reliab"] = rel
            row[f"{name}_Resol"] = res
        rows.append(row)
    return pd.DataFrame(rows)


def compute_cindex(comp):
    cens_dist = get_censoring_dist([
        {"time_at_event": r["time_at_event"], "y": r["gold"]}
        for _, r in comp.iterrows()])
    times = comp["time_at_event"].values
    events = comp["gold"].values.astype(int)
    out = {}
    for name, build in [
        ("Lung", lambda: np.tile(comp["risk_score"].values[:, None], (1, MAX_FOLLOWUP))),
        ("Sybil15", lambda: np.column_stack([comp[f"sybil_risk_{yr}"].values for yr in range(1, 7)])),
        ("Sybilv1_cal", lambda: np.column_stack([comp[f"sv1c_risk_{yr}"].values for yr in range(1, 7)])),
        ("Sybilv1_raw", lambda: np.column_stack([comp[f"sv1r_risk_{yr}"].values for yr in range(1, 7)])),
    ]:
        out[name] = float(concordance_index(times, build(), events, cens_dist))
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    comp = build_four_way()
    comp.to_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"), index=False)

    df = compute_metrics(comp)
    df.to_csv(os.path.join(OUTPUT_DIR, "four_way_metrics.csv"), index=False)

    print("\n" + "=" * 100)
    print("FOUR-WAY METRIC COMPARISON")
    print("=" * 100)
    print(f"{'Year':<4} {'AUC':>32} {'PR-AUC':>32} {'Reliability':>32}")
    print(f"{'':4} {'Lung':>7} {'S1.5':>7} {'S1.0c':>7} {'S1.0r':>7} "
          f"{'Lung':>7} {'S1.5':>7} {'S1.0c':>7} {'S1.0r':>7} "
          f"{'Lung':>7} {'S1.5':>7} {'S1.0c':>7} {'S1.0r':>7}")
    print("-" * 100)
    for _, r in df.iterrows():
        print(f"{int(r['Year']):<4}"
              f" {r['Lung_AUC']:>7.3f} {r['Sybil15_AUC']:>7.3f} {r['Sybilv1_cal_AUC']:>7.3f} {r['Sybilv1_raw_AUC']:>7.3f}"
              f" {r['Lung_PRAUC']:>7.3f} {r['Sybil15_PRAUC']:>7.3f} {r['Sybilv1_cal_PRAUC']:>7.3f} {r['Sybilv1_raw_PRAUC']:>7.3f}"
              f" {r['Lung_Reliab']:>7.4f} {r['Sybil15_Reliab']:>7.4f} {r['Sybilv1_cal_Reliab']:>7.4f} {r['Sybilv1_raw_Reliab']:>7.4f}")

    print("\nC-INDEX (Uno's, IPCW):")
    cidx = compute_cindex(comp)
    for k, v in cidx.items():
        print(f"  {k:<14} {v:.3f}")

    print("\nKey question: does v1.0 raw look more like v1.5 raw (fair comparison)?")
    print(f"  Year 1 AUC: v1.5={df.iloc[0]['Sybil15_AUC']:.3f}  v1.0_raw={df.iloc[0]['Sybilv1_raw_AUC']:.3f}  v1.0_cal={df.iloc[0]['Sybilv1_cal_AUC']:.3f}")
    print(f"  Year 1 Reliab: v1.5={df.iloc[0]['Sybil15_Reliab']:.4f}  v1.0_raw={df.iloc[0]['Sybilv1_raw_Reliab']:.4f}  v1.0_cal={df.iloc[0]['Sybilv1_cal_Reliab']:.4f}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
