#!/usr/bin/env python3
"""Replace `sybil_risk_{1..6}` (author Pillar-0 ensemble) in
four_way_comp_df.csv with user-reproduced Pillar-0 predictions via the
two-strategy match (series_uid → pid_tp fallback).

Why: user said "use our reproduction for everything." Author scores
and user reproduction agree to ΔAUC<0.003 per horizon (validated in
pillar0_repro_compare.py), so this is a consistency/clarity update, not
a numerical re-finding. Downstream scripts (`extensions.py`,
`four_way_compare.py`, etc.) that read `sybil_risk_*` will now consume
user reproduction without further edits.

Saves a backup at `four_way_comp_df.author_pillar.csv` so the original
author scores remain recoverable.

Idempotent: detects whether comp_df already contains user-reproduced
Pillar-0 (via correlation check) and refuses to double-patch.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_survival import OUTPUT_DIR

COMP_DF = os.path.join(OUTPUT_DIR, "four_way_comp_df.csv")
BACKUP  = os.path.join(OUTPUT_DIR, "four_way_comp_df.author_pillar.csv")
USER_TEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions_pillar_test.csv")


def main():
    print(f"Loading {COMP_DF} …")
    comp = pd.read_csv(COMP_DF)
    print(f"  comp_df: {len(comp)} rows, {comp['pid'].nunique()} pids")

    if not os.path.exists(BACKUP):
        comp.to_csv(BACKUP, index=False)
        print(f"  Backed up original to {BACKUP}")
    else:
        print(f"  Backup already exists at {BACKUP} (not overwriting)")

    print(f"Loading user reproduction {USER_TEST} …")
    user = pd.read_csv(USER_TEST)
    user["series_uid"] = user["series_uid"].astype(str)
    user["patient_id"] = user["patient_id"].astype(str)
    user["timepoint"] = user["timepoint"].astype(int)
    user["pid_tp"] = user["patient_id"] + "_" + user["timepoint"].astype(str)

    comp["pid"] = comp["pid"].astype(str)
    comp["screen_timepoint"] = comp["screen_timepoint"].astype(int)
    comp["pid_tp"] = comp["pid"] + "_" + comp["screen_timepoint"].astype(str)

    # Two-strategy match (series_uid then pid_tp)
    cols = [f"prob_y{y}" for y in range(1, 7)]
    rename = {f"prob_y{y}": f"sybil_risk_{y}_NEW" for y in range(1, 7)}

    u1 = user.drop_duplicates("series_uid").rename(
        columns={"series_uid": "series", **rename})[
        ["series"] + [rename[c] for c in cols]]
    out = comp.merge(u1, on="series", how="left")
    miss = out["sybil_risk_1_NEW"].isna()
    u2 = user.drop_duplicates("pid_tp").rename(columns=rename)[
        ["pid_tp"] + [rename[c] for c in cols]]
    fb = out.loc[miss, ["pid_tp"]].merge(u2, on="pid_tp", how="left")
    for c in cols:
        out.loc[miss, rename[c]] = fb[rename[c]].values

    n_matched = int((~out["sybil_risk_1_NEW"].isna()).sum())
    print(f"  user-Pillar-0 matched: {n_matched}/{len(out)} scans "
          f"({int(out['sybil_risk_1_NEW'].isna().sum())} NaN)")

    # Sanity: correlation between old (author) and new (user) sybil_risk_1
    a = comp["sybil_risk_1"].values.astype(float)
    u = out["sybil_risk_1_NEW"].values.astype(float)
    mask = ~np.isnan(a) & ~np.isnan(u)
    corr = float(np.corrcoef(a[mask], u[mask])[0, 1])
    print(f"  author vs user sybil_risk_1 corr: {corr:.4f}")
    if corr > 0.9999:
        print("  Looks like comp_df already contains user reproduction — "
              "no patch applied.")
        # Drop the temp _NEW columns we just merged
        for y in range(1, 7):
            if f"sybil_risk_{y}_NEW" in out.columns:
                del out[f"sybil_risk_{y}_NEW"]
        if "pid_tp" in out.columns and "pid_tp" not in comp.columns:
            del out["pid_tp"]
        return

    # Replace sybil_risk_{1..6} with user reproduction
    for y in range(1, 7):
        out[f"sybil_risk_{y}"] = out[f"sybil_risk_{y}_NEW"]
        del out[f"sybil_risk_{y}_NEW"]

    # Tidy: drop pid_tp helper col if it wasn't there originally
    if "pid_tp" not in comp.columns and "pid_tp" in out.columns:
        # Keep it — already in comp_df from earlier scripts, harmless.
        pass

    out.to_csv(COMP_DF, index=False)
    print(f"\nSaved patched comp_df → {COMP_DF}")
    print(f"  sybil_risk_{{1..6}} now contains user-reproduced Pillar-0")
    print(f"  Original author scores preserved at {BACKUP}")
    print(f"  {int(out['sybil_risk_1'].isna().sum())} scans have NaN "
          f"Pillar-0 (unmatchable)")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
