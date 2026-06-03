#!/usr/bin/env python3
"""
Compare best-effort Lung-RADS v2022 against the v1.0 (Pinsky 2015)
reproduction on the matched four-way NLST test cohort.

Outputs:
  * Per-scan transition table (how many scans change category v1.0 -> v2022)
  * Per-category distribution side-by-side
  * Within-category y1 cancer rate side-by-side
  * Operating-point comparison: flag rate, sensitivity, PPV at Cat>=3 /
    Cat>=4A / Cat>=4B for both versions (no bootstrap; exploratory).
  * T0-only subset of the same.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_survival as _es  # noqa: F401  side-effect: seed OUTPUT_DIR
OUT_DIR = os.environ.get("SCREENING_METRICS_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_output"))
COMP_CSV = f"{OUT_DIR}/four_way_comp_df.csv"
LR_V10   = f"{OUT_DIR}/nlst_lungrads_pinsky.csv"
LR_V2022 = f"{OUT_DIR}/nlst_lungrads_v2022.csv"

CAT_ORDER = {"1": 0, "2": 1, "3": 2, "4A": 3, "4B": 4, "4X": 5}
CAT_LIST  = ["1", "2", "3", "4A", "4B", "4X"]


def load_matched() -> pd.DataFrame:
    comp = pd.read_csv(COMP_CSV, low_memory=False)
    v10  = pd.read_csv(LR_V10,  low_memory=False)
    v22  = pd.read_csv(LR_V2022, low_memory=False)

    for d in (comp, v10, v22):
        d["pid"] = d["pid"].astype(int)

    v10 = v10.rename(columns={"study_yr": "screen_timepoint",
                              "lungrads_v10": "lr_v10"})
    v22 = v22.rename(columns={"study_yr": "screen_timepoint",
                              "lungrads_v2022": "lr_v22"})

    m = comp.merge(v10[["pid", "screen_timepoint", "lr_v10"]],
                   on=["pid", "screen_timepoint"], how="left")
    m = m.merge(v22[["pid", "screen_timepoint", "lr_v22"]],
                on=["pid", "screen_timepoint"], how="left")

    import ast
    y_seq = m["y_seq"].apply(ast.literal_eval)
    m["y1"] = y_seq.apply(lambda x: int(x[0]))
    m["y3"] = y_seq.apply(lambda x: int(x[2]))
    m["y6"] = y_seq.apply(lambda x: int(x[5]))
    return m


def transition_table(m: pd.DataFrame) -> pd.DataFrame:
    tab = pd.crosstab(m["lr_v10"], m["lr_v22"]).reindex(
        index=CAT_LIST, columns=CAT_LIST, fill_value=0)
    return tab


def per_category_view(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [("lr_v10", "v1.0"), ("lr_v22", "v2022")]:
        for c in CAT_LIST:
            sub = m[m[col] == c]
            n = len(sub)
            n_cancer = int(sub["y1"].sum()) if n else 0
            rate = (100.0 * n_cancer / n) if n else np.nan
            rows.append({"version": label, "cat": c, "n": n,
                         "pct_cohort": 100.0 * n / len(m),
                         "y1_cancer": n_cancer,
                         "y1_rate_%": rate})
    return pd.DataFrame(rows)


def op_point(sub: pd.DataFrame, col: str, cut: str) -> dict:
    rank = sub[col].map(CAT_ORDER)
    flagged = rank >= CAT_ORDER[cut]
    n = len(sub)
    n_pos = int(sub["y1"].sum())
    n_neg = n - n_pos
    tp = int(((flagged) & (sub["y1"] == 1)).sum())
    fp = int(((flagged) & (sub["y1"] == 0)).sum())
    fr = 100.0 * flagged.sum() / n
    sens = 100.0 * tp / n_pos if n_pos else np.nan
    fpr  = 100.0 * fp / n_neg if n_neg else np.nan
    ppv  = 100.0 * tp / (tp + fp) if (tp + fp) else np.nan
    return {"flag_rate_%": fr, "sens_%": sens, "fpr_%": fpr, "ppv_%": ppv,
            "tp": tp, "fp": fp, "n_pos": n_pos, "n": n}


def op_points_table(m: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for cut in ("3", "4A", "4B"):
        for col, ver in [("lr_v10", "v1.0"), ("lr_v22", "v2022")]:
            r = op_point(m, col, cut)
            r["cohort"] = label
            r["version"] = ver
            r["cut"] = f">={cut}"
            rows.append(r)
    return pd.DataFrame(rows)[
        ["cohort", "cut", "version", "n", "n_pos",
         "flag_rate_%", "sens_%", "fpr_%", "ppv_%", "tp", "fp"]
    ]


def main():
    m = load_matched()
    # Drop scans without LR (shouldn't happen post-merge but be safe)
    m = m.dropna(subset=["lr_v10", "lr_v22"]).copy()
    print(f"Matched cohort: n={len(m):,} scans, "
          f"{m['pid'].nunique():,} pids, "
          f"{int(m['y1'].sum())} y1 cancers")
    print()

    tt = transition_table(m)
    print("=== Transition table: v1.0 (row) -> v2022 (col) ===")
    print(tt.to_string())
    n_moved = int(tt.values.sum() - np.trace(tt.values))
    print(f"Scans that change category: {n_moved} "
          f"({100*n_moved/len(m):.2f}%)")
    print()

    pcv = per_category_view(m)
    print("=== Per-category distribution and y1 cancer rate ===")
    piv_n = pcv.pivot(index="cat", columns="version", values="n").reindex(CAT_LIST)
    piv_pct = pcv.pivot(index="cat", columns="version", values="pct_cohort").reindex(CAT_LIST)
    piv_can = pcv.pivot(index="cat", columns="version", values="y1_cancer").reindex(CAT_LIST)
    piv_rate = pcv.pivot(index="cat", columns="version", values="y1_rate_%").reindex(CAT_LIST)
    side = pd.concat({
        "n":          piv_n.astype("Int64"),
        "% cohort":   piv_pct.round(2),
        "y1 cancers": piv_can.astype("Int64"),
        "y1 rate %":  piv_rate.round(2),
    }, axis=1)
    print(side.to_string())
    print()

    ops_all = op_points_table(m, "full")
    t0 = m[m["screen_timepoint"] == 0].copy()
    ops_t0 = op_points_table(t0, "T0")
    ops = pd.concat([ops_all, ops_t0], ignore_index=True)
    print("=== Operating-point comparison (>=Cat 3 / 4A / 4B) ===")
    print(ops.to_string(index=False,
                       formatters={"flag_rate_%": "{:.2f}".format,
                                   "sens_%": "{:.2f}".format,
                                   "fpr_%":  "{:.2f}".format,
                                   "ppv_%":  "{:.2f}".format}))
    print()

    # Within-LR-negative (Cat 1+2) granularity check: how do the two
    # versions compare on the "lumped negative" population?
    for col, ver in [("lr_v10", "v1.0"), ("lr_v22", "v2022")]:
        neg = m[m[col].isin(["1", "2"])]
        n = len(neg)
        c = int(neg["y1"].sum())
        print(f"Cat 1+2 ({ver}): n={n:,}, y1 cancers={c}, "
              f"rate={100*c/n:.3f}%")

    # Save
    tt.to_csv(f"{OUT_DIR}/lr_v10_vs_v2022_transition.csv")
    side.to_csv(f"{OUT_DIR}/lr_v10_vs_v2022_per_category.csv")
    ops.to_csv(f"{OUT_DIR}/lr_v10_vs_v2022_ops.csv", index=False)
    print()
    print(f"Saved: lr_v10_vs_v2022_{{transition,per_category,ops}}.csv "
          f"in {OUT_DIR}")


if __name__ == "__main__":
    main()
