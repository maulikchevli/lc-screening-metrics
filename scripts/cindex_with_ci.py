#!/usr/bin/env python3
"""Uno's C-index with 95% bootstrap CIs for all four paper models.

Models reported in deployment form (per feedback-table-design):
  LungEvaty   — Platt-calibrated (== raw for C-index; rank invariance)
  Pillar-0    — raw (no calibrator)
  Sybil v1.0  — author-calibrated
  PLCOm2012   — Platt-recalibrated (== raw for C-index; rank invariance)

C-index is rank-based and invariant under any strictly monotone calibrator,
so the LungEvaty and PLCOm cells are identical for raw vs Platt forms. The
Sybil v1.0 calibrator has ties (228 unique values across 6,694 dev scans),
so cal/raw can differ marginally; deployment form is the cal output.

Bootstrap: n=1000 patient-clustered resamples. Each replicate re-fits the
KM censoring distribution on its own resample (Uno's IPCW form).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_survival import (
    load_participant, PARTICIPANT_CSV, OUTPUT_DIR, MAX_FOLLOWUP,
)
from uspstf_plcom import plcom2012_risk
from vital.metrics import concordance_index, get_censoring_dist


COMP_DF = os.environ.get("FOUR_WAY_COMP_DF", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "four_way_comp_df.csv"))
PILLAR_USER_TEST_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions_pillar_test.csv")
N_BOOT = 1000
RNG_SEED = 20260518

PLCOM_INPUT_COLS = [
    "height", "weight", "educat", "age_quit", "ethnic",
    "diagcopd", "diagemph", "diagchro",
    "famfather", "fammother", "fambrother", "famsister",
    "cancblad", "cancbrea", "canccerv", "canccolo", "cancesop",
    "canckidn", "canclary", "canclung", "cancnasa", "cancoral",
    "cancpanc", "cancphar", "cancstom", "cancthyr", "canctran",
]


def attach_user_pillar(comp):
    """Two-strategy match (series_uid → pid_tp fallback) user-reproduced
    Pillar-0 test scores into comp_df. Adds pillar_user_y{1..6}."""
    user = pd.read_csv(PILLAR_USER_TEST_CSV)
    user["series_uid"] = user["series_uid"].astype(str)
    user["patient_id"] = user["patient_id"].astype(str)
    user["timepoint"] = user["timepoint"].astype(int)
    user["pid_tp"] = user["patient_id"] + "_" + user["timepoint"].astype(str)
    cols = [f"prob_y{y}" for y in range(1, 7)]
    rename = {f"prob_y{y}": f"pillar_user_y{y}" for y in range(1, 7)}

    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    comp["screen_timepoint"] = comp["screen_timepoint"].astype(int)
    comp["pid_tp"] = comp["pid"] + "_" + comp["screen_timepoint"].astype(str)

    u1 = user.drop_duplicates("series_uid").rename(
        columns={"series_uid": "series", **rename})[
        ["series"] + [rename[c] for c in cols]]
    out = comp.merge(u1, on="series", how="left")
    miss = out["pillar_user_y1"].isna()
    u2 = user.drop_duplicates("pid_tp").rename(columns=rename)[
        ["pid_tp"] + [rename[c] for c in cols]]
    fb = out.loc[miss, ["pid_tp"]].merge(u2, on="pid_tp", how="left")
    for c in cols:
        out.loc[miss, rename[c]] = fb[rename[c]].values
    return out


def attach_plcom_inputs(comp):
    part = load_participant(PARTICIPANT_CSV)
    extra = [c for c in PLCOM_INPUT_COLS if c in part.columns]
    ext = part[["pid"] + extra].drop_duplicates("pid")
    ext["pid"] = ext["pid"].astype(str)
    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    return comp.merge(ext, on="pid", how="left")


def compute_plcom_horizons(comp):
    cols = []
    for y in range(1, MAX_FOLLOWUP + 1):
        cols.append(comp.apply(lambda r: plcom2012_risk(r, year=y), axis=1).values)
    return np.column_stack(cols)


def cindex_on_rows(scores_2d, times, events, idx):
    sub_t = times[idx]
    sub_e = events[idx]
    sub_s = scores_2d[idx]
    cens = get_censoring_dist([
        {"time_at_event": int(t), "y": int(e)} for t, e in zip(sub_t, sub_e)
    ])
    return float(concordance_index(sub_t, sub_s, sub_e, cens))


def patient_bootstrap(comp, scores_dict, times, events, n_boot, seed):
    rng = np.random.default_rng(seed)
    pids = comp["pid"].values
    unique_pids = np.unique(pids)
    pid_to_rows = {p: np.where(pids == p)[0] for p in unique_pids}
    n_pids = len(unique_pids)

    full_idx = np.arange(len(times))
    point = {m: cindex_on_rows(s, times, events, full_idx)
             for m, s in scores_dict.items()}

    boot = {m: np.full(n_boot, np.nan) for m in scores_dict}
    for b in range(n_boot):
        sampled = rng.choice(unique_pids, size=n_pids, replace=True)
        idx = np.concatenate([pid_to_rows[p] for p in sampled])
        for m, s in scores_dict.items():
            try:
                boot[m][b] = cindex_on_rows(s, times, events, idx)
            except Exception:
                pass
        if (b + 1) % 100 == 0:
            print(f"    bootstrap {b+1}/{n_boot}")
    return point, boot


def main():
    print(f"Loading {COMP_DF} …")
    comp = pd.read_csv(COMP_DF)
    print(f"  raw four-way comp_df: {len(comp)} scans, "
          f"{comp['pid'].nunique()} pids, {int(comp['gold'].sum())} cancers")

    print("Attaching user-reproduced Pillar-0 …")
    comp = attach_user_pillar(comp)
    n_pillar_matched = int((~comp["pillar_user_y1"].isna()).sum())
    print(f"  user-Pillar-0 matched: {n_pillar_matched}/{len(comp)} scans")

    print("Attaching PLCOm inputs and computing per-horizon PLCOm risk …")
    comp = attach_plcom_inputs(comp)
    plcom_mat = compute_plcom_horizons(comp)
    nan_mask = np.isnan(plcom_mat).any(axis=1)
    print(f"  PLCOm NaN rows: {int(nan_mask.sum())} "
          f"(dropped for matched-set C-index)")

    keep = ~nan_mask & (~comp["pillar_user_y1"].isna())
    comp = comp.loc[keep].reset_index(drop=True)
    plcom_mat = plcom_mat[keep.values]

    print(f"  matched set: {len(comp)} scans, "
          f"{comp['pid'].nunique()} pids, {int(comp['gold'].sum())} cancers")

    times = comp["time_at_event"].values.astype(float)
    events = comp["gold"].values.astype(int)

    scores = {
        "LungEvaty":  np.tile(comp["risk_score"].values[:, None], (1, MAX_FOLLOWUP)),
        "Pillar-0":   np.column_stack([comp[f"pillar_user_y{y}"].values for y in range(1, 7)]),
        "Sybil v1.0": np.column_stack([comp[f"sv1c_risk_{y}"].values for y in range(1, 7)]),
        "PLCOm2012":  plcom_mat,
    }

    print(f"Bootstrapping (n={N_BOOT}, patient-clustered, seed={RNG_SEED}) …")
    point, boot = patient_bootstrap(comp, scores, times, events, N_BOOT, RNG_SEED)

    rows = []
    for m in scores:
        arr = boot[m]
        arr = arr[~np.isnan(arr)]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        rows.append(dict(
            model=m,
            c_index=point[m],
            ci_lo=float(lo),
            ci_hi=float(hi),
            ci_width=float(hi - lo),
            n_boot=int(len(arr)),
        ))
    out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "cindex_with_ci.csv")
    out.to_csv(out_path, index=False)

    print("\n" + "=" * 64)
    print(f"UNO'S C-INDEX (IPCW), n={len(comp)} scans, "
          f"{comp['pid'].nunique()} pids, {int(events.sum())} cancers")
    print(f"95% patient-clustered bootstrap CI (n_boot={N_BOOT})")
    print("=" * 64)
    print(f"{'Model':<12} {'C-index':>8}  {'95% CI':<22}")
    print("-" * 64)
    for r in rows:
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        print(f"{r['model']:<12} {r['c_index']:>8.3f}  {ci:<22}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
