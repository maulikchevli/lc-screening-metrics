#!/usr/bin/env python3
"""
Pinsky LungRADS vs AI risk models — year-1 head-to-head comparison.

Outputs (in eval_output/):
  - lungrads_classification_y1.csv        : sens/FPR/PPV/NPV at matched OPs
  - lungrads_category_crosstab_y1.csv     : within-tier cancer rate
  - lungrads_subcat_y1.csv                : AI granularity within LR 1+2
  - lungrads_nb_points_y1.csv             : NB(LungRADS) reference points

All metrics at year 1 with patient-clustered bootstrap CIs (n=1000).
Year-1 is the only fair horizon: LungRADS is a next-scan management
system, not a multi-year predictor.

Score conventions:
  - LungEvaty raw y1 = comp.cancer_risk[0]  (rank-equivalent to Platt y1)
  - Sybil v1.0 cal y1 = sv1c_risk_1
  - Pillar-0 raw y1 = pillar_user_y1 (loaded from test CSV)
  Rank-based metrics (sens, FPR, PPV at matched flag rate) are invariant
  to monotonic calibration, so raw scores suffice here.

For NB we use the calibrated columns where available; for LungRADS the
NB depends on pt only (no calibration needed for a categorical rule).
"""

import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_survival as _es  # noqa: F401  side-effect: seed OUTPUT_DIR
OUT_DIR = os.environ.get("SCREENING_METRICS_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_output"))
COMP_CSV = f"{OUT_DIR}/four_way_comp_df.csv"
LR_CSV   = f"{OUT_DIR}/nlst_lungrads_pinsky.csv"
PILLAR_TEST_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions_pillar_test.csv")

RNG_SEED  = 42
N_BOOT    = 1000
PT_LIST   = (0.01, 0.05, 0.10)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_data():
    comp = pd.read_csv(COMP_CSV, low_memory=False)
    lr   = pd.read_csv(LR_CSV)
    comp["pid"] = comp["pid"].astype(int)
    lr["pid"]   = lr["pid"].astype(int)
    m = comp.merge(lr.rename(columns={"study_yr": "screen_timepoint"}),
                   on=["pid", "screen_timepoint"], how="left")
    # LungEvaty raw y1 from cancer_risk list
    m["le_y1"] = m["cancer_risk"].apply(lambda s: float(json.loads(s.replace("'", '"'))[0])
                                        if isinstance(s, str) else float(s[0]))
    # Sybil v1.0 cal y1
    m["sybil_y1"] = m["sv1c_risk_1"].astype(float)
    # Attach Pillar-0 test (CSV uses patient_id, timepoint)
    pill = pd.read_csv(PILLAR_TEST_CSV)
    pill["pid_tp"] = (pill["patient_id"].astype(str) + "_"
                      + pill["timepoint"].astype(str))
    pill_y1 = pill.rename(columns={"prob_y1": "pillar_y1"})[["pid_tp", "pillar_y1"]]
    # comp may already have pid_tp; recompute defensively
    m["pid_tp"] = m["pid"].astype(str) + "_" + m["screen_timepoint"].astype(str)
    # Some pills may have multiple series per pid_tp -> take mean
    pill_y1 = pill_y1.groupby("pid_tp", as_index=False)["pillar_y1"].mean()
    m = m.merge(pill_y1, on="pid_tp", how="left")
    # LungRADS flags
    m["lr_pos3"] = m["lungrads_v10"].isin(["3", "4A", "4B", "4X"]).astype(int)
    m["lr_pos4"] = m["lungrads_v10"].isin(["4A", "4B", "4X"]).astype(int)
    # Year-1 cancer label per existing pipeline convention
    m["y1"] = ((m["y"] == 1) & (m["time_at_event"] <= 0)).astype(int)
    return m


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def classif(y, flag):
    y = np.asarray(y, dtype=int)
    flag = np.asarray(flag, dtype=int)
    tp = int(((y == 1) & (flag == 1)).sum())
    fp = int(((y == 0) & (flag == 1)).sum())
    fn = int(((y == 1) & (flag == 0)).sum())
    tn = int(((y == 0) & (flag == 0)).sum())
    P = tp + fn
    N = fp + tn
    sens = tp / P if P else np.nan
    fpr  = fp / N if N else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan
    return dict(sens=sens, fpr=fpr, ppv=ppv, npv=npv, tp=tp, fp=fp, fn=fn, tn=tn,
                flag_rate=(tp + fp) / len(y))


def threshold_at_flag_rate(scores, fr):
    """Return threshold s.t. exactly fr fraction of scans flagged."""
    scores = np.asarray(scores, dtype=float)
    k = int(round(fr * len(scores)))
    if k <= 0:
        return float(np.max(scores) + 1)
    if k >= len(scores):
        return float(-np.inf)
    return float(np.partition(scores, -k)[-k])


def threshold_at_sens(scores, y, target_sens):
    """Largest threshold such that recall(thr) >= target_sens."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    order = np.argsort(-scores, kind="stable")
    ys = y[order]
    ss = scores[order]
    cum_tp = np.cumsum(ys)
    total_pos = int(y.sum())
    if total_pos == 0:
        return float("nan")
    need = int(np.ceil(target_sens * total_pos))
    idx = np.searchsorted(cum_tp, need)
    if idx >= len(ss):
        return float(ss[-1])
    return float(ss[idx])


def threshold_at_fpr(scores, y, target_fpr):
    """Smallest threshold such that FPR(thr) <= target_fpr."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    order = np.argsort(-scores, kind="stable")
    ys = y[order]
    ss = scores[order]
    cum_fp = np.cumsum(ys == 0)
    total_neg = int((y == 0).sum())
    if total_neg == 0:
        return float("nan")
    allow = int(np.floor(target_fpr * total_neg))
    if allow <= 0:
        return float(ss[0] + 1)
    idx = min(int(np.searchsorted(cum_fp, allow, side="right")) - 1, len(ss) - 1)
    return float(ss[max(idx, 0)])


# ---------------------------------------------------------------------------
# Bootstrap (patient-clustered)
# ---------------------------------------------------------------------------
def cluster_boot_indices(pids, rng):
    unique = np.array(sorted(set(pids.tolist())))
    samp = rng.choice(unique, size=len(unique), replace=True)
    # Build idx list
    pid_to_idx = {}
    for i, p in enumerate(pids):
        pid_to_idx.setdefault(p, []).append(i)
    out = []
    for p in samp:
        out.extend(pid_to_idx[p])
    return np.array(out)


def boot_ci(values):
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def fmt_ci(point, lo, hi, pct=True, dec=1):
    if pct:
        f = lambda x: f"{x*100:.{dec}f}"
    else:
        f = lambda x: f"{x:.{dec}f}"
    return f"{f(point)} [{f(lo)}, {f(hi)}]"


# ---------------------------------------------------------------------------
# Build the main table: classification metrics for each model at matched OPs
# ---------------------------------------------------------------------------
def main_classification_table(m):
    rows = []
    pids = m["pid"].values
    y    = m["y1"].values
    rng  = np.random.default_rng(RNG_SEED)

    # ---- LungRADS reference rows (3+ and 4A+) -----------------------------
    for label, flag_col in [("LungRADS 3+", "lr_pos3"),
                            ("LungRADS 4A+", "lr_pos4")]:
        c = classif(y, m[flag_col].values)
        # Bootstrap CIs for the LungRADS-fixed-rule row
        b_sens, b_fpr, b_ppv, b_npv, b_fr = [], [], [], [], []
        for _ in range(N_BOOT):
            idx = cluster_boot_indices(pids, rng)
            ck = classif(y[idx], m[flag_col].values[idx])
            b_sens.append(ck["sens"]); b_fpr.append(ck["fpr"])
            b_ppv.append(ck["ppv"]);   b_npv.append(ck["npv"])
            b_fr.append(ck["flag_rate"])
        rows.append({
            "model": label, "anchor": "—",
            "flag_rate": fmt_ci(c["flag_rate"], *boot_ci(b_fr)),
            "sens":      fmt_ci(c["sens"],      *boot_ci(b_sens)),
            "fpr":       fmt_ci(c["fpr"],       *boot_ci(b_fpr)),
            "ppv":       fmt_ci(c["ppv"],       *boot_ci(b_ppv)),
            "npv":       fmt_ci(c["npv"],       *boot_ci(b_npv)),
        })

    # ---- AI matched-OP rows ----------------------------------------------
    ai_models = [
        ("LungEvaty",  m["le_y1"].values),
        ("Pillar-0",   m["pillar_y1"].values),
        ("Sybil v1.0", m["sybil_y1"].values),
    ]
    # Two LungRADS anchors:
    lr3_sens = classif(y, m["lr_pos3"].values)["sens"]
    lr3_fpr  = classif(y, m["lr_pos3"].values)["fpr"]
    lr3_fr   = classif(y, m["lr_pos3"].values)["flag_rate"]
    lr4_sens = classif(y, m["lr_pos4"].values)["sens"]
    lr4_fpr  = classif(y, m["lr_pos4"].values)["fpr"]
    lr4_fr   = classif(y, m["lr_pos4"].values)["flag_rate"]

    anchors = [
        ("vs LR 3+, matched FR",   lr3_fr,   "flag_rate"),
        ("vs LR 3+, matched sens", lr3_sens, "sens"),
        ("vs LR 3+, matched FPR",  lr3_fpr,  "fpr"),
        ("vs LR 4A+, matched FR",  lr4_fr,   "flag_rate"),
        ("vs LR 4A+, matched sens",lr4_sens, "sens"),
        ("vs LR 4A+, matched FPR", lr4_fpr,  "fpr"),
    ]

    for label, scores in ai_models:
        valid = ~np.isnan(scores)
        sc = scores[valid]
        yv = y[valid]
        pv = pids[valid]
        for anchor_label, target, anchor_type in anchors:
            if anchor_type == "flag_rate":
                thr = threshold_at_flag_rate(sc, target)
            elif anchor_type == "sens":
                thr = threshold_at_sens(sc, yv, target)
            else:
                thr = threshold_at_fpr(sc, yv, target)
            flag = (sc >= thr).astype(int)
            c = classif(yv, flag)

            # Bootstrap
            b_sens, b_fpr, b_ppv, b_npv, b_fr = [], [], [], [], []
            for _ in range(N_BOOT):
                idx = cluster_boot_indices(pv, rng)
                sc_b, y_b = sc[idx], yv[idx]
                if anchor_type == "flag_rate":
                    thr_b = threshold_at_flag_rate(sc_b, target)
                elif anchor_type == "sens":
                    thr_b = threshold_at_sens(sc_b, y_b, target)
                else:
                    thr_b = threshold_at_fpr(sc_b, y_b, target)
                flag_b = (sc_b >= thr_b).astype(int)
                ck = classif(y_b, flag_b)
                b_sens.append(ck["sens"]); b_fpr.append(ck["fpr"])
                b_ppv.append(ck["ppv"]);   b_npv.append(ck["npv"])
                b_fr.append(ck["flag_rate"])
            rows.append({
                "model": label, "anchor": anchor_label,
                "flag_rate": fmt_ci(c["flag_rate"], *boot_ci(b_fr)),
                "sens":      fmt_ci(c["sens"],      *boot_ci(b_sens)),
                "fpr":       fmt_ci(c["fpr"],       *boot_ci(b_fpr)),
                "ppv":       fmt_ci(c["ppv"],       *boot_ci(b_ppv)),
                "npv":       fmt_ci(c["npv"],       *boot_ci(b_npv)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Category cross-tab: LR 5-tier vs AI Option B 5-tier at year 1
# ---------------------------------------------------------------------------
def category_crosstab(m):
    """For each tier, report N scans, N year-1 cancers, observed rate."""
    rows = []
    # LungRADS native
    lr_order = ["4B", "4X", "4A", "3", "2", "1"]  # high → low
    lr_label = {"4B": "Cat 4B", "4X": "Cat 4X", "4A": "Cat 4A",
                "3": "Cat 3", "2": "Cat 2", "1": "Cat 1"}
    for cat in lr_order:
        mask = m["lungrads_v10"] == cat
        n = int(mask.sum())
        nca = int(m.loc[mask, "y1"].sum())
        rate = nca / n if n else np.nan
        rows.append({"system": "LungRADS", "tier": lr_label[cat],
                     "n_scans": n, "n_y1_cancers": nca,
                     "y1_rate": f"{rate*100:.2f}%" if not np.isnan(rate) else "—",
                     "pct_of_pop": f"{n/len(m)*100:.1f}%"})

    # AI Option B (use rank on year-1 score → deciles → C1..C5)
    ai_models = [("LungEvaty",  m["le_y1"].values),
                 ("Pillar-0",   m["pillar_y1"].values),
                 ("Sybil v1.0", m["sybil_y1"].values)]
    DECILE_TO_CAT = {1: "C1", 2: "C2", 3: "C3", 4: "C3",
                     5: "C4", 6: "C4", 7: "C4",
                     8: "C5", 9: "C5", 10: "C5"}
    for label, scores in ai_models:
        valid = ~np.isnan(scores)
        sc = scores[valid]
        df_v = m.loc[valid].copy()
        # Test-fit decile via rank (deployment-form scores would give nearly
        # identical decile labels in the test cohort)
        n = len(sc)
        order = np.argsort(-sc, kind="stable")
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n)
        decile = np.ceil((rank + 1) / (n / 10)).astype(int).clip(1, 10)
        df_v["cat"] = np.array([DECILE_TO_CAT[d] for d in decile])
        for cat in ["C1", "C2", "C3", "C4", "C5"]:
            mask = df_v["cat"] == cat
            nn = int(mask.sum())
            nca = int(df_v.loc[mask, "y1"].sum())
            rate = nca / nn if nn else np.nan
            rows.append({"system": label, "tier": cat,
                         "n_scans": nn, "n_y1_cancers": nca,
                         "y1_rate": f"{rate*100:.2f}%" if not np.isnan(rate) else "—",
                         "pct_of_pop": f"{nn/len(df_v)*100:.1f}%"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sub-categorization: AI granularity within LungRADS 1+2 (the "value-add")
# ---------------------------------------------------------------------------
def subcat_within_lr12(m):
    """Among LR Cat 1+2 patients (LR-negative), how do AI tiers split them?"""
    lr_neg = m["lungrads_v10"].isin(["1", "2"])
    sub = m.loc[lr_neg].copy()
    rows = []
    DECILE_TO_CAT = {1: "C1", 2: "C2", 3: "C3", 4: "C3",
                     5: "C4", 6: "C4", 7: "C4",
                     8: "C5", 9: "C5", 10: "C5"}
    for label, score_col in [("LungEvaty",  "le_y1"),
                             ("Pillar-0",   "pillar_y1"),
                             ("Sybil v1.0", "sybil_y1")]:
        sc = sub[score_col].values
        valid = ~np.isnan(sc)
        scv = sc[valid]
        df_v = sub.loc[valid].copy()
        n = len(scv)
        order = np.argsort(-scv, kind="stable")
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n)
        decile = np.ceil((rank + 1) / (n / 10)).astype(int).clip(1, 10)
        df_v["cat"] = np.array([DECILE_TO_CAT[d] for d in decile])
        for cat in ["C1", "C2", "C3", "C4", "C5"]:
            mask = df_v["cat"] == cat
            nn = int(mask.sum())
            nca = int(df_v.loc[mask, "y1"].sum())
            rate = nca / nn if nn else np.nan
            rows.append({
                "ai_model": label,
                "ai_tier_within_LR12": cat,
                "n_scans": nn,
                "pct_of_LR12": f"{nn/len(df_v)*100:.1f}%",
                "n_y1_cancers": nca,
                "y1_rate": f"{rate*100:.2f}%" if not np.isnan(rate) else "—",
            })
    # Reference: LR 1+2 overall
    n12 = int(lr_neg.sum())
    nca12 = int(sub["y1"].sum())
    rows.insert(0, {"ai_model": "(reference)",
                    "ai_tier_within_LR12": "LR Cat 1+2 overall",
                    "n_scans": n12,
                    "pct_of_LR12": "100.0%",
                    "n_y1_cancers": nca12,
                    "y1_rate": f"{nca12/n12*100:.2f}%" if n12 else "—"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NB(LungRADS) at pt = 1, 5, 10 %
# ---------------------------------------------------------------------------
def nb_at_pt(flag, y, pt):
    n = len(y)
    tp = int(((y == 1) & (flag == 1)).sum())
    fp = int(((y == 0) & (flag == 1)).sum())
    return (tp / n) - (fp / n) * (pt / (1 - pt))


def lungrads_nb_points(m):
    pids = m["pid"].values
    y = m["y1"].values
    rows = []
    rng = np.random.default_rng(RNG_SEED + 1)
    for label, flag_col in [("LungRADS 3+", "lr_pos3"),
                            ("LungRADS 4A+", "lr_pos4")]:
        flag = m[flag_col].values
        for pt in PT_LIST:
            point = nb_at_pt(flag, y, pt)
            boot = []
            for _ in range(N_BOOT):
                idx = cluster_boot_indices(pids, rng)
                boot.append(nb_at_pt(flag[idx], y[idx], pt))
            lo, hi = boot_ci(boot)
            rows.append({"strategy": label, "pt": f"{int(pt*100)}%",
                         "NB": fmt_ci(point, lo, hi, pct=False, dec=4)})
    # Treat-all reference (positive scaled by prevalence)
    pi = y.mean()
    for pt in PT_LIST:
        nb_all = pi - (1 - pi) * pt / (1 - pt)
        rows.append({"strategy": "Treat-all (ref)", "pt": f"{int(pt*100)}%",
                     "NB": f"{nb_all:.4f}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data ...")
    m = load_data()
    print(f"Loaded {len(m):,} scans, {m['y1'].sum()} year-1 cancers, "
          f"{m['pid'].nunique():,} pids")
    print(f"Missing Pillar-0 y1: {m['pillar_y1'].isna().sum()}, "
          f"missing Sybil y1: {m['sybil_y1'].isna().sum()}, "
          f"missing LE y1: {m['le_y1'].isna().sum()}")

    # 1. Classification table
    print("\n[1/4] Classification table at matched operating points ...")
    tab1 = main_classification_table(m)
    tab1.to_csv(f"{OUT_DIR}/lungrads_classification_y1.csv", index=False)
    print(tab1.to_string(index=False))

    # 2. Category cross-tab
    print("\n[2/4] Category cross-tabulation ...")
    tab2 = category_crosstab(m)
    tab2.to_csv(f"{OUT_DIR}/lungrads_category_crosstab_y1.csv", index=False)
    print(tab2.to_string(index=False))

    # 3. Sub-categorization within LR 1+2
    print("\n[3/4] AI granularity within LR Cat 1+2 ...")
    tab3 = subcat_within_lr12(m)
    tab3.to_csv(f"{OUT_DIR}/lungrads_subcat_y1.csv", index=False)
    print(tab3.to_string(index=False))

    # 4. NB reference points
    print("\n[4/4] NB(LungRADS) reference points at pt=1/5/10% ...")
    tab4 = lungrads_nb_points(m)
    tab4.to_csv(f"{OUT_DIR}/lungrads_nb_points_y1.csv", index=False)
    print(tab4.to_string(index=False))

    print("\nDone. Wrote 4 CSVs to eval_output/.")


if __name__ == "__main__":
    main()
