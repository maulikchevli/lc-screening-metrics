#!/usr/bin/env python3
"""Task 3 — PR curves for the Layer-2 figure, with new operating-point
anchors aligned to the decile/NB tables.

Curves for {LungEvaty raw, Pillar-0, Sybil v1.0 raw, PLCOm2012} ×
{Year 1, 3, 6} = 12.

Estimator conventions (per task spec):
  Year 1 and 3 : IPCW-weighted precision/recall
  Year 6       : complete-case  precision/recall

Operating points marked on each curve:
  flag rate 10% (C1)
  flag rate 20% (C1+C2)
  flag rate 40% (C1+C2+C3)
  flag rate 70% (C1+C2+C3+C4)
  recall    95% (high-sensitivity safety regime)

Flag-rate anchors are rank-based (top-k by score, stable tiebreak) —
calibration-invariant under monotone Platt. Recall-95% anchor preserves
the previous Layer-2 convention.

Pillar-0 scores are the user reproduction (sybil_risk_* in comp_df is
patched post-`patch_comp_df_pillar_user.py`).

Output:
  eval_output/task3_pr_curves.json
  eval_output/task3_pr_curves_long.csv (per-threshold)
  eval_output/task3_pr_curves_op_points.csv (operating points)
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import _include, _ipcw_weights, load_participant, \
    PARTICIPANT_CSV, OUTPUT_DIR
from uspstf_plcom import plcom2012_risk, NLST_PERSONAL_CANCER_COLS

N_THRESH = 400
# (kind, value, label)
ANCHORS = [
    ("flag_rate", 0.10, "FR10"),
    ("flag_rate", 0.20, "FR20"),
    ("flag_rate", 0.40, "FR40"),
    ("flag_rate", 0.70, "FR70"),
    ("recall",    0.95, "R95"),
]


def ipcw_pr_curve(scores, times, events, fu, n_thresh=N_THRESH):
    """IPCW-weighted (recall, precision) over n_thresh threshold values."""
    scores = np.asarray(scores, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    valid = ~np.isnan(scores)
    scores = scores[valid]; times = times[valid]; events = events[valid]
    tp_w, fp_w, is_tp, is_fp, prev = _ipcw_weights(times, events, fu)
    sum_tp_w = float(tp_w.sum())
    smax = float(np.nanmax(scores))
    thresholds = np.linspace(0.0, smax, n_thresh)
    rows = []
    for t in thresholds:
        pos = scores >= t
        tp_contrib = float(tp_w[pos & is_tp].sum())
        fp_count = int(np.sum(pos & is_fp))
        recall = tp_contrib / sum_tp_w if sum_tp_w > 0 else np.nan
        denom = tp_contrib + fp_count * fp_w
        precision = tp_contrib / denom if denom > 0 else np.nan
        rows.append((float(t), float(recall), float(precision)))
    return rows, dict(tp_w=tp_w, fp_w=fp_w, is_tp=is_tp, is_fp=is_fp,
                      sum_tp_w=sum_tp_w, scores=scores)


def cc_pr_curve(scores, labels, n_thresh=N_THRESH):
    """Complete-case (recall, precision) over n_thresh threshold values."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    valid = ~np.isnan(scores)
    scores = scores[valid]; labels = labels[valid]
    pos_total = int(labels.sum())
    smax = float(np.nanmax(scores))
    thresholds = np.linspace(0.0, smax, n_thresh)
    rows = []
    for t in thresholds:
        pos = scores >= t
        tp = int(np.sum(pos & (labels == 1)))
        fp = int(np.sum(pos & (labels == 0)))
        recall = tp / pos_total if pos_total > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rows.append((float(t), float(recall), float(precision)))
    return rows, dict(labels=labels, scores=scores, pos_total=pos_total)


# ---------------------------------------------------------------------------
# Operating-point logic
# ---------------------------------------------------------------------------
def op_flag_rate_ipcw(target_fr, aux):
    """Top-X% by rank on IPCW-evaluation cohort scores. Returns the
    realised (threshold, recall, precision) under IPCW weighting."""
    scores = aux["scores"]
    tp_w = aux["tp_w"]; fp_w = aux["fp_w"]
    is_tp = aux["is_tp"]; is_fp = aux["is_fp"]
    sum_tp_w = aux["sum_tp_w"]
    n = len(scores)
    if n == 0 or sum_tp_w == 0:
        return dict(achievable=False, threshold=None, recall=None,
                    precision=None, realised_flag_rate=None)
    k = max(1, int(np.ceil(n * target_fr)))
    order = np.argsort(-scores, kind="stable")
    pos = np.zeros(n, dtype=bool)
    pos[order[:k]] = True
    tp_contrib = float(tp_w[pos & is_tp].sum())
    fp_count = int(np.sum(pos & is_fp))
    recall = tp_contrib / sum_tp_w
    denom = tp_contrib + fp_count * fp_w
    precision = float(tp_contrib / denom) if denom > 0 else float("nan")
    threshold = float(scores[order[k - 1]])
    return dict(achievable=True, threshold=threshold,
                recall=float(recall), precision=precision,
                realised_flag_rate=k / n)


def op_flag_rate_cc(target_fr, aux):
    scores = aux["scores"]; labels = aux["labels"]
    n = len(scores); pos_total = aux["pos_total"]
    if n == 0 or pos_total == 0:
        return dict(achievable=False, threshold=None, recall=None,
                    precision=None, realised_flag_rate=None)
    k = max(1, int(np.ceil(n * target_fr)))
    order = np.argsort(-scores, kind="stable")
    pos = np.zeros(n, dtype=bool)
    pos[order[:k]] = True
    tp = int((pos & (labels == 1)).sum())
    fp = int((pos & (labels == 0)).sum())
    threshold = float(scores[order[k - 1]])
    recall = tp / pos_total
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return dict(achievable=True, threshold=threshold,
                recall=float(recall), precision=float(precision),
                realised_flag_rate=k / n)


def op_recall_from_curve(curve, target):
    """Highest threshold whose recall is still ≥ target."""
    valid = [(t, r, p) for (t, r, p) in curve
             if not (np.isnan(r) or np.isnan(p)) and r >= target]
    if not valid:
        return dict(achievable=False, threshold=None, recall=None,
                    precision=None, realised_flag_rate=None)
    chosen = max(valid, key=lambda x: x[0])
    return dict(achievable=True,
                threshold=chosen[0],
                recall=chosen[1],
                precision=chosen[2],
                realised_flag_rate=None)


def compute_op_points(curve, aux, mode):
    out = {}
    for kind, val, lbl in ANCHORS:
        if kind == "flag_rate":
            op = (op_flag_rate_ipcw(val, aux) if mode == "ipcw"
                  else op_flag_rate_cc(val, aux))
        else:
            op = op_recall_from_curve(curve, val)
        out[lbl] = op
    return out


# ---------------------------------------------------------------------------
def main():
    print("Loading 4-way matched cohort (Pillar-0 = user reproduction) + PLCOm …")
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    part = load_participant(PARTICIPANT_CSV)
    part["pid"] = part["pid"].astype(str)
    extra_cols = ["height", "weight", "educat", "age_quit", "ethnic",
                  "diagcopd", "diagemph", "diagchro",
                  "famfather", "fammother", "fambrother",
                  "famsister", "famchild"]
    extra_cols += list(NLST_PERSONAL_CANCER_COLS)
    extra_cols = [c for c in extra_cols if c in part.columns]
    extra = part[["pid"] + extra_cols].drop_duplicates("pid")
    comp = comp.merge(extra, on="pid", how="left")
    for y in (1, 3, 6):
        comp[f"plcom_y{y}"] = comp.apply(
            lambda r, _y=y: plcom2012_risk(r, year=_y), axis=1)

    out = {}
    flags = []
    for year in (1, 3, 6):
        fu = year - 1
        if year in (1, 3):
            mode = "ipcw"
            sub = comp
            times = sub["time_at_event"].values.astype(float)
            events = sub["gold"].values.astype(int)
            labels_y6 = None
        else:
            mode = "cc"
            mask = comp.apply(
                lambda r, _fu=fu: _include(r["time_at_event"],
                                            r["gold"], _fu)[0],
                axis=1).values
            sub = comp[mask].reset_index(drop=True)
            labels_y6 = sub.apply(
                lambda r, _fu=fu: _include(r["time_at_event"],
                                            r["gold"], _fu)[1],
                axis=1).values
            times = sub["time_at_event"].values.astype(float)
            events = sub["gold"].values.astype(int)
        print(f"\nYear {year}  ({mode.upper()}, N={len(sub)})")

        for mname, key in [
            ("lungevaty", "risk_score"),
            ("pillar0",   f"sybil_risk_{year}"),
            ("sybil",     f"sv1c_risk_{year}"),
            ("plcom",     f"plcom_y{year}"),
        ]:
            scores = sub[key].values.astype(float)
            if mode == "ipcw":
                curve, aux = ipcw_pr_curve(scores, times, events, fu)
            else:
                curve, aux = cc_pr_curve(scores, labels_y6)
            op_points = compute_op_points(curve, aux, mode)
            for lbl, op in op_points.items():
                if not op["achievable"]:
                    flags.append(f"{mname}_year{year} {lbl}: not achievable")

            key_out = f"{mname}_year{year}"
            out[key_out] = dict(
                model=mname, year=year, mode=mode,
                n=int(len(sub)),
                curve=curve, operating_points=op_points,
            )
            ops_str = "  ".join(
                f"{lbl}: r={op['recall']:.2f}/p={op['precision']:.3f}"
                if op["achievable"] else f"{lbl}: n/a"
                for lbl, op in op_points.items()
            )
            print(f"  {mname:<14}  {ops_str}")

    out_path = os.path.join(OUTPUT_DIR, "task3_pr_curves.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    if flags:
        print("\nFLAGS"); print("-----")
        for f in flags:
            print(f"  - {f}")

    # Long-form curves CSV
    long_rows = []
    for key, val in out.items():
        for (t, r, p) in val["curve"]:
            long_rows.append(dict(
                key=key, model=val["model"], year=val["year"],
                mode=val["mode"],
                threshold=t, recall=r, precision=p,
            ))
    pd.DataFrame(long_rows).to_csv(
        os.path.join(OUTPUT_DIR, "task3_pr_curves_long.csv"), index=False)

    # Operating-points CSV
    op_rows = []
    for key, val in out.items():
        for lbl, op in val["operating_points"].items():
            op_rows.append(dict(
                key=key, model=val["model"], year=val["year"],
                anchor=lbl, achievable=op["achievable"],
                threshold=op["threshold"], recall=op["recall"],
                precision=op["precision"],
                realised_flag_rate=op.get("realised_flag_rate"),
            ))
    pd.DataFrame(op_rows).to_csv(
        os.path.join(OUTPUT_DIR, "task3_pr_curves_op_points.csv"), index=False)
    print(f"Saved {os.path.join(OUTPUT_DIR, 'task3_pr_curves_op_points.csv')}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
