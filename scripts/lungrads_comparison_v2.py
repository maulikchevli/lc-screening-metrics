#!/usr/bin/env python3
"""
LungRADS vs AI — extended analyses (v2).

Adds to v1:
  - Per-timepoint stratification (T0 prevalence vs T1/T2 incidence)
  - Discordance 2x2 (LR 3+ vs AI top-decile) on year-1 cancers
  - Rank-position of LR-missed cancers in AI distribution (and vice versa)
  - Stage-stratified caught/missed analysis (using de_stag)
  - AI top 3.7% vs LR Cat 4B (matching the most selective LR tier)

All metrics at year 1; patient-clustered bootstrap (n=1000) where appropriate.
Categorical contingency tables use exact counts (no bootstrap needed).
"""

import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lungrads_comparison import (
    load_data, classif, threshold_at_flag_rate, threshold_at_sens,
    cluster_boot_indices, boot_ci, fmt_ci, OUT_DIR, RNG_SEED, N_BOOT,
)

# Stage code -> readable
STAGE_MAP = {
    110: "IA", 120: "IB",
    210: "IIA", 220: "IIB",
    310: "IIIA", 320: "IIIB",
    400: "IV",
    900: "Occult", 994: "Carcinoid", 999: "Unknown",
}
EARLY_STAGE  = {110, 120, 210, 220}   # I-II
LATE_STAGE   = {310, 320, 400}        # III-IV


# ---------------------------------------------------------------------------
# 1. Per-timepoint stratification
# ---------------------------------------------------------------------------
def per_timepoint_table(m):
    rows = []
    rng = np.random.default_rng(RNG_SEED + 10)
    for tp, label in [(0, "T0 (prevalence)"), (1, "T1"), (2, "T2")]:
        sub = m[m["screen_timepoint"] == tp].copy()
        y = sub["y1"].values
        pids = sub["pid"].values
        n = len(sub)
        ncase = int(y.sum())

        # LR 3+
        lr_flag = sub["lr_pos3"].values
        c = classif(y, lr_flag)
        b_fr, b_sens, b_fpr, b_ppv = [], [], [], []
        for _ in range(N_BOOT):
            idx = cluster_boot_indices(pids, rng)
            ck = classif(y[idx], lr_flag[idx])
            b_fr.append(ck["flag_rate"]); b_sens.append(ck["sens"])
            b_fpr.append(ck["fpr"]); b_ppv.append(ck["ppv"])
        rows.append({
            "timepoint": label, "n_scans": n, "n_y1_cancers": ncase,
            "model": "LungRADS 3+", "anchor": "—",
            "FR": fmt_ci(c["flag_rate"], *boot_ci(b_fr)),
            "Sens": fmt_ci(c["sens"], *boot_ci(b_sens)),
            "FPR": fmt_ci(c["fpr"], *boot_ci(b_fpr)),
            "PPV": fmt_ci(c["ppv"], *boot_ci(b_ppv)),
        })

        # Each AI at matched-flag-rate and matched-sens
        target_fr = c["flag_rate"]
        target_sens = c["sens"]
        for ai_label, col in [("LungEvaty", "le_y1"), ("Pillar-0", "pillar_y1"),
                              ("Sybil v1.0", "sybil_y1")]:
            scores = sub[col].values
            valid = ~np.isnan(scores)
            sc = scores[valid]; yv = y[valid]; pv = pids[valid]

            # Matched FR
            thr = threshold_at_flag_rate(sc, target_fr)
            flag = (sc >= thr).astype(int)
            cm = classif(yv, flag)
            b_sens, b_fpr, b_ppv, b_fr = [], [], [], []
            for _ in range(N_BOOT):
                idx = cluster_boot_indices(pv, rng)
                scb, yb = sc[idx], yv[idx]
                thrb = threshold_at_flag_rate(scb, target_fr)
                fb = (scb >= thrb).astype(int)
                ck = classif(yb, fb)
                b_sens.append(ck["sens"]); b_fpr.append(ck["fpr"])
                b_ppv.append(ck["ppv"]);  b_fr.append(ck["flag_rate"])
            rows.append({
                "timepoint": label, "n_scans": n, "n_y1_cancers": ncase,
                "model": ai_label, "anchor": f"match FR ({target_fr*100:.1f}%)",
                "FR": fmt_ci(cm["flag_rate"], *boot_ci(b_fr)),
                "Sens": fmt_ci(cm["sens"], *boot_ci(b_sens)),
                "FPR": fmt_ci(cm["fpr"], *boot_ci(b_fpr)),
                "PPV": fmt_ci(cm["ppv"], *boot_ci(b_ppv)),
            })

            # Matched Sens
            thr = threshold_at_sens(sc, yv, target_sens)
            flag = (sc >= thr).astype(int)
            cm = classif(yv, flag)
            b_sens, b_fpr, b_ppv, b_fr = [], [], [], []
            for _ in range(N_BOOT):
                idx = cluster_boot_indices(pv, rng)
                scb, yb = sc[idx], yv[idx]
                thrb = threshold_at_sens(scb, yb, target_sens)
                fb = (scb >= thrb).astype(int)
                ck = classif(yb, fb)
                b_sens.append(ck["sens"]); b_fpr.append(ck["fpr"])
                b_ppv.append(ck["ppv"]);  b_fr.append(ck["flag_rate"])
            rows.append({
                "timepoint": label, "n_scans": n, "n_y1_cancers": ncase,
                "model": ai_label, "anchor": f"match sens ({target_sens*100:.1f}%)",
                "FR": fmt_ci(cm["flag_rate"], *boot_ci(b_fr)),
                "Sens": fmt_ci(cm["sens"], *boot_ci(b_sens)),
                "FPR": fmt_ci(cm["fpr"], *boot_ci(b_fpr)),
                "PPV": fmt_ci(cm["ppv"], *boot_ci(b_ppv)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Discordance 2x2 on year-1 cancers
# ---------------------------------------------------------------------------
def discordance_table(m):
    """For each AI model, compute LR 3+ x AI-top-decile contingency on cancers
    AND on non-cancers. Then summarize: agreement, LR-only, AI-only, both-miss."""
    rows = []
    cancers = m[m["y1"] == 1].copy()
    n_cases = len(cancers)

    for ai_label, col in [("LungEvaty", "le_y1"), ("Pillar-0", "pillar_y1"),
                          ("Sybil v1.0", "sybil_y1")]:
        # Compute AI top-decile assignment on whole cohort
        scores = m[col].values
        n_all = len(scores)
        order = np.argsort(-scores, kind="stable")
        rank = np.empty(n_all, dtype=int)
        rank[order] = np.arange(n_all)
        in_top_decile = (rank < int(round(0.1 * n_all))).astype(int)
        m[f"{ai_label}_top10"] = in_top_decile

        # Among cancers only:
        c = m[m["y1"] == 1]
        lr_pos = c["lr_pos3"].values
        ai_pos = c[f"{ai_label}_top10"].values

        both    = int(((lr_pos == 1) & (ai_pos == 1)).sum())
        lr_only = int(((lr_pos == 1) & (ai_pos == 0)).sum())
        ai_only = int(((lr_pos == 0) & (ai_pos == 1)).sum())
        neither = int(((lr_pos == 0) & (ai_pos == 0)).sum())

        rows.append({
            "AI model": ai_label,
            "year-1 cancers": n_cases,
            "Both flag": f"{both} ({both/n_cases*100:.1f}%)",
            "LR only": f"{lr_only} ({lr_only/n_cases*100:.1f}%)",
            "AI only": f"{ai_only} ({ai_only/n_cases*100:.1f}%)",
            "Neither": f"{neither} ({neither/n_cases*100:.1f}%)",
            "LR total": f"{both + lr_only} ({(both+lr_only)/n_cases*100:.1f}%)",
            "AI total": f"{both + ai_only} ({(both+ai_only)/n_cases*100:.1f}%)",
        })
    return pd.DataFrame(rows), m


# ---------------------------------------------------------------------------
# 3. Rank position of LR-missed cancers in AI distribution
# ---------------------------------------------------------------------------
def missed_cancer_ranks(m):
    """For LR-missed cancers (LR Cat 1+2, year-1 cancer): what AI percentile?
    For AI-missed cancers (AI C4+C5 = bottom 70%, year-1 cancer): what LR cat?
    """
    rows = []

    # LR-missed cancers: 15 cancers in LR Cat 1+2
    lr_missed = m[(m["y1"] == 1) & (m["lr_pos3"] == 0)].copy()
    n_lr_missed = len(lr_missed)

    for ai_label, col in [("LungEvaty", "le_y1"), ("Pillar-0", "pillar_y1"),
                          ("Sybil v1.0", "sybil_y1")]:
        # Compute AI percentile rank on full cohort
        scores = m[col].values
        n_all = len(scores)
        # percentile = (rank_from_top / n) * 100 -- but report as "AI percentile from top"
        order = np.argsort(-scores, kind="stable")
        rank = np.empty(n_all, dtype=int)
        rank[order] = np.arange(n_all)
        pct_from_top = (rank + 1) / n_all * 100
        m[f"{ai_label}_pct"] = pct_from_top

        # For each LR-missed cancer, where in AI?
        ai_pcts = m.loc[(m["y1"] == 1) & (m["lr_pos3"] == 0),
                        f"{ai_label}_pct"].values
        in_top10 = (ai_pcts <= 10).sum()
        in_top20 = (ai_pcts <= 20).sum()
        in_top40 = (ai_pcts <= 40).sum()
        in_bottom30 = (ai_pcts > 70).sum()
        rows.append({
            "AI model": ai_label,
            "Scenario": f"LR-missed cancers (n={n_lr_missed}) — AI rank",
            "In AI top 10% (C1)": f"{in_top10} ({in_top10/n_lr_missed*100:.0f}%)",
            "In AI top 20% (C1+C2)": f"{in_top20} ({in_top20/n_lr_missed*100:.0f}%)",
            "In AI top 40% (C1-C3)": f"{in_top40} ({in_top40/n_lr_missed*100:.0f}%)",
            "In AI bottom 30% (C5)": f"{in_bottom30} ({in_bottom30/n_lr_missed*100:.0f}%)",
        })

    # AI-missed cancers: cancers in AI bottom 70% (C3-C5)
    for ai_label, col in [("LungEvaty", "le_y1"), ("Pillar-0", "pillar_y1"),
                          ("Sybil v1.0", "sybil_y1")]:
        # In each AI's bottom 70% (= C3 + C4 + C5)
        miss_mask = (m["y1"] == 1) & (m[f"{ai_label}_pct"] > 20)
        ai_missed = m[miss_mask].copy()
        n_ai_missed = len(ai_missed)
        # Where in LR?
        lr_dist = ai_missed["lungrads_v10"].value_counts()
        lr_pos_in_missed = int(ai_missed["lr_pos3"].sum())
        rows.append({
            "AI model": ai_label,
            "Scenario": f"{ai_label}-missed cancers (n={n_ai_missed}, C3-C5) — LR cat",
            "LR Cat 1": int((ai_missed["lungrads_v10"] == "1").sum()),
            "LR Cat 2": int((ai_missed["lungrads_v10"] == "2").sum()),
            "LR Cat 3": int((ai_missed["lungrads_v10"] == "3").sum()),
            "LR Cat 4A/4B/4X": int(ai_missed["lr_pos4"].sum() + (ai_missed["lungrads_v10"] == "4X").sum() if False else (ai_missed["lungrads_v10"].isin(["4A","4B","4X"])).sum()),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Stage-stratified discordance
# ---------------------------------------------------------------------------
def stage_stratified(m):
    """Among year-1 cancers, stratify by stage:
      - LR-only caught (LR 3+ AND AI C2+) — early vs late?
      - AI-only caught (AI C1 AND LR 1+2) — early vs late?
      - Both caught
      - Both missed"""
    cancers = m[m["y1"] == 1].copy()
    cancers["stage_grp"] = cancers["de_stag"].apply(
        lambda s: "Early (I-II)" if s in EARLY_STAGE
                  else ("Late (III-IV)" if s in LATE_STAGE
                  else "Other/Unk"))
    rows = []
    for ai_label in ["LungEvaty", "Pillar-0", "Sybil v1.0"]:
        col_top10 = f"{ai_label}_top10"
        if col_top10 not in cancers.columns:
            # Recompute on cancer subset if discordance hasn't run
            scores = m[{"LungEvaty": "le_y1", "Pillar-0": "pillar_y1",
                        "Sybil v1.0": "sybil_y1"}[ai_label]].values
            n_all = len(scores)
            order = np.argsort(-scores, kind="stable")
            rank = np.empty(n_all, dtype=int)
            rank[order] = np.arange(n_all)
            cancers[col_top10] = (rank[m["y1"] == 1] < int(round(0.1*n_all))).astype(int)

        # 2x2 by stage group
        for sg in ["Early (I-II)", "Late (III-IV)", "Other/Unk"]:
            s = cancers[cancers["stage_grp"] == sg]
            ntot = len(s)
            if ntot == 0:
                continue
            both    = int(((s["lr_pos3"] == 1) & (s[col_top10] == 1)).sum())
            lr_only = int(((s["lr_pos3"] == 1) & (s[col_top10] == 0)).sum())
            ai_only = int(((s["lr_pos3"] == 0) & (s[col_top10] == 1)).sum())
            neither = int(((s["lr_pos3"] == 0) & (s[col_top10] == 0)).sum())
            rows.append({
                "AI model": ai_label,
                "Stage": sg,
                "n_cancers": ntot,
                "Both": both, "LR only": lr_only,
                "AI only": ai_only, "Neither": neither,
                "LR catch %": f"{(both+lr_only)/ntot*100:.0f}%",
                "AI catch %": f"{(both+ai_only)/ntot*100:.0f}%",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. AI top 3.7% vs LR Cat 4B (most selective)
# ---------------------------------------------------------------------------
def match_lr4b(m):
    """LR Cat 4B alone: 230 scans (3.7%), 51 y1 cancers, 22.2% PPV.
    What does AI top 3.7% look like?"""
    rows = []
    rng = np.random.default_rng(RNG_SEED + 20)
    pids = m["pid"].values
    y = m["y1"].values

    # LR 4B alone (NOT 4A+)
    lr4b_flag = (m["lungrads_v10"] == "4B").astype(int).values
    c = classif(y, lr4b_flag)
    b_sens, b_fpr, b_ppv, b_fr = [], [], [], []
    for _ in range(N_BOOT):
        idx = cluster_boot_indices(pids, rng)
        ck = classif(y[idx], lr4b_flag[idx])
        b_sens.append(ck["sens"]); b_fpr.append(ck["fpr"])
        b_ppv.append(ck["ppv"]);  b_fr.append(ck["flag_rate"])
    rows.append({
        "Strategy": "LungRADS Cat 4B alone",
        "Flag rate": fmt_ci(c["flag_rate"], *boot_ci(b_fr)),
        "Sens": fmt_ci(c["sens"], *boot_ci(b_sens)),
        "FPR": fmt_ci(c["fpr"], *boot_ci(b_fpr)),
        "PPV": fmt_ci(c["ppv"], *boot_ci(b_ppv)),
    })

    target_fr = c["flag_rate"]
    for ai_label, col in [("LungEvaty", "le_y1"), ("Pillar-0", "pillar_y1"),
                          ("Sybil v1.0", "sybil_y1")]:
        scores = m[col].values
        valid = ~np.isnan(scores)
        sc = scores[valid]; yv = y[valid]; pv = pids[valid]
        thr = threshold_at_flag_rate(sc, target_fr)
        flag = (sc >= thr).astype(int)
        cm = classif(yv, flag)
        b_sens, b_fpr, b_ppv, b_fr = [], [], [], []
        for _ in range(N_BOOT):
            idx = cluster_boot_indices(pv, rng)
            scb, yb = sc[idx], yv[idx]
            thrb = threshold_at_flag_rate(scb, target_fr)
            fb = (scb >= thrb).astype(int)
            ck = classif(yb, fb)
            b_sens.append(ck["sens"]); b_fpr.append(ck["fpr"])
            b_ppv.append(ck["ppv"]);  b_fr.append(ck["flag_rate"])
        rows.append({
            "Strategy": f"{ai_label} top {target_fr*100:.1f}%",
            "Flag rate": fmt_ci(cm["flag_rate"], *boot_ci(b_fr)),
            "Sens": fmt_ci(cm["sens"], *boot_ci(b_sens)),
            "FPR": fmt_ci(cm["fpr"], *boot_ci(b_fpr)),
            "PPV": fmt_ci(cm["ppv"], *boot_ci(b_ppv)),
        })

    # Also: AI matched to LR 4B PPV (22.2%) — what's the flag rate?
    target_ppv = c["ppv"]
    for ai_label, col in [("LungEvaty", "le_y1"), ("Pillar-0", "pillar_y1"),
                          ("Sybil v1.0", "sybil_y1")]:
        scores = m[col].values
        valid = ~np.isnan(scores)
        sc = scores[valid]; yv = y[valid]
        order = np.argsort(-sc, kind="stable")
        sc_sorted = sc[order]; y_sorted = yv[order]
        # Walk down; for each prefix, compute PPV
        cum_y = np.cumsum(y_sorted)
        prefix_n = np.arange(1, len(sc_sorted) + 1)
        prefix_ppv = cum_y / prefix_n
        # Find largest k with PPV >= target_ppv
        cand = np.where(prefix_ppv >= target_ppv)[0]
        if len(cand) == 0:
            rows.append({"Strategy": f"{ai_label} matched PPV (LR 4B)",
                         "Flag rate": "n/a (PPV unreachable)",
                         "Sens": "—", "FPR": "—", "PPV": "—"})
            continue
        k = cand[-1] + 1
        flag = np.zeros_like(yv)
        flag[order[:k]] = 1
        cm = classif(yv, flag)
        rows.append({
            "Strategy": f"{ai_label} matched PPV (>={target_ppv*100:.1f}%)",
            "Flag rate": f"{cm['flag_rate']*100:.2f}%",
            "Sens": f"{cm['sens']*100:.1f}%",
            "FPR": f"{cm['fpr']*100:.2f}%",
            "PPV": f"{cm['ppv']*100:.1f}%",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data ...")
    m = load_data()
    print(f"Loaded {len(m):,} scans, {m['y1'].sum()} year-1 cancers")

    print("\n[1/5] Per-timepoint stratification (T0 vs T1 vs T2) ...")
    tab1 = per_timepoint_table(m)
    tab1.to_csv(f"{OUT_DIR}/lungrads_per_timepoint_y1.csv", index=False)
    print(tab1.to_string(index=False))

    print("\n[2/5] Discordance 2x2 on year-1 cancers ...")
    tab2, m = discordance_table(m)
    tab2.to_csv(f"{OUT_DIR}/lungrads_discordance_y1.csv", index=False)
    print(tab2.to_string(index=False))

    print("\n[3/5] Missed-cancer rank position ...")
    tab3 = missed_cancer_ranks(m)
    tab3.to_csv(f"{OUT_DIR}/lungrads_missed_ranks_y1.csv", index=False)
    print(tab3.to_string(index=False))

    print("\n[4/5] Stage-stratified discordance ...")
    tab4 = stage_stratified(m)
    tab4.to_csv(f"{OUT_DIR}/lungrads_stage_stratified_y1.csv", index=False)
    print(tab4.to_string(index=False))

    print("\n[5/5] AI vs LR Cat 4B (most selective) ...")
    tab5 = match_lr4b(m)
    tab5.to_csv(f"{OUT_DIR}/lungrads_vs_lr4b_y1.csv", index=False)
    print(tab5.to_string(index=False))

    print("\nDone. Wrote 5 CSVs to eval_output/.")


if __name__ == "__main__":
    main()
