#!/usr/bin/env python3
"""Task 1 — operating-point Sp / Prec / Recall at four flag-rate anchors
(10/20/40/70%) plus recall ≥ 95%, with paired patient-clustered bootstrap
95% CIs (n=1000).

Anchors:
  flag-rate 10% (C1, "workup")
  flag-rate 20% (C1+C2, through 3-month CT)
  flag-rate 40% (C1+C2+C3, through 6-month CT)
  flag-rate 70% (C1+C2+C3+C4, through 12-month CT)
  recall    95% (high-sensitivity safety regime)

The first four match the cat-aggregated decile boundaries used in the
decile/NB tables. Threshold-finding for flag-rate anchors is rank-based
(top-k by score, stable sort) — calibration-invariant under monotone
Platt. The recall@95% anchor preserves the previous Layer-2 convention.

Models (deployment form, per feedback_table_design):
  LungEvaty       (raw, rank-invariant under Platt)
  Pillar-0        (user-reproduced; sybil_risk_* in comp_df post-patch)
  Sybil v1.0 raw  (sv1r_risk_*)
  PLCOm2012       (raw, rank-invariant under Platt)

Estimator: complete-case via `_include` (matches Layer 2 of the paper).

Outputs:
  eval_output/task1_sp_prec_recalls.csv  (machine-readable; old name kept
                                          to minimize downstream churn)
  printed markdown summary
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_survival import _include, load_participant, OUTPUT_DIR, PARTICIPANT_CSV
from uspstf_plcom import plcom2012_risk
from decile_stratification_devfit import _plcom_extra_cols

RNG_SEED = 42
N_BOOT = 1000

# (kind, value, short_label)
ANCHORS = [
    ("flag_rate", 0.10, "FR10"),
    ("flag_rate", 0.20, "FR20"),
    ("flag_rate", 0.40, "FR40"),
    ("flag_rate", 0.70, "FR70"),
    ("recall",    0.95, "R95"),
]

# (display name, internal key, score-column template; {y} = horizon)
MODELS = [
    ("LungEvaty",       "lung",  "risk_score"),
    ("Pillar-0",        "p0",    "sybil_risk_{y}"),
    ("Sybil",           "sybil", "sv1c_risk_{y}"),
    ("PLCOm2012",       "plcom", "plcom_y{y}"),
]


# ---------------------------------------------------------------------------
def _attach_plcom(comp):
    part = load_participant(PARTICIPANT_CSV)
    part["pid"] = part["pid"].astype(str)
    extras = _plcom_extra_cols(part)
    ext = part[["pid"] + extras].drop_duplicates("pid").copy()
    comp = comp.copy()
    comp["pid"] = comp["pid"].astype(str)
    comp = comp.merge(ext, on="pid", how="left")
    for y in range(1, 7):
        comp[f"plcom_y{y}"] = comp.apply(
            lambda r, _y=y: plcom2012_risk(r, year=_y), axis=1)
    return comp


def _filter_year(comp, year):
    """Complete-case filter with all four model scores valid this horizon."""
    fu = year - 1
    rows = []
    for _, r in comp.iterrows():
        inc, lbl = _include(r["time_at_event"], r["gold"], fu)
        if not inc:
            continue
        lung   = r["risk_score"]
        p0     = r[f"sybil_risk_{year}"]
        sybil  = r[f"sv1c_risk_{year}"]
        plcom  = r[f"plcom_y{year}"]
        if any(pd.isna(v) for v in (lung, p0, sybil, plcom)):
            continue
        rows.append((lung, p0, sybil, plcom, lbl, r["pid"]))
    if not rows:
        return None
    a = list(zip(*rows))
    return dict(
        lung=np.array(a[0], dtype=float),
        p0=np.array(a[1], dtype=float),
        sybil=np.array(a[2], dtype=float),
        plcom=np.array(a[3], dtype=float),
        labels=np.array(a[4], dtype=int),
        pids=np.array(a[5]),
    )


# ---------------------------------------------------------------------------
def metrics_at_flag_rate(scores, labels, target_fr):
    """Top-k by rank where k = ceil(n * target_fr). Rank-based, stable
    tiebreak → invariant under monotone calibration."""
    n = len(scores)
    pos_total = int(labels.sum())
    neg_total = n - pos_total
    if n == 0 or pos_total == 0 or neg_total == 0:
        return dict(achievable=False, threshold=np.nan, recall=np.nan,
                    specificity=np.nan, precision=np.nan,
                    realised_flag_rate=np.nan)
    k = int(np.ceil(n * target_fr))
    if k < 1:
        k = 1
    order = np.argsort(-scores, kind="stable")
    pos = np.zeros(n, dtype=bool)
    pos[order[:k]] = True
    tp = int((pos & (labels == 1)).sum())
    fp = int((pos & (labels == 0)).sum())
    tn = int((~pos & (labels == 0)).sum())
    threshold = float(scores[order[k - 1]])
    return dict(
        achievable=True,
        threshold=threshold,
        recall=tp / pos_total,
        specificity=tn / neg_total,
        precision=tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        realised_flag_rate=int(pos.sum()) / n,
    )


def metrics_at_recall(scores, labels, target):
    """Most-aggressive threshold (highest score) whose recall is ≥ target.
    Preserves prior task1 logic."""
    pos_total = int(labels.sum())
    neg_total = int(len(labels) - pos_total)
    if pos_total == 0 or neg_total == 0:
        return dict(achievable=False, threshold=np.nan, recall=np.nan,
                    specificity=np.nan, precision=np.nan,
                    realised_flag_rate=np.nan)
    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    y_sorted = labels[order]
    tp_cum = np.cumsum(y_sorted == 1)
    recall_cum = tp_cum / pos_total
    valid = recall_cum >= target
    if not valid.any():
        return dict(achievable=False, threshold=np.nan, recall=np.nan,
                    specificity=np.nan, precision=np.nan,
                    realised_flag_rate=np.nan)
    k = int(np.argmax(valid))
    threshold = float(s_sorted[k])
    pos = scores >= threshold
    tp = int(np.sum(pos & (labels == 1)))
    fp = int(np.sum(pos & (labels == 0)))
    tn = int(np.sum(~pos & (labels == 0)))
    return dict(
        achievable=True,
        threshold=threshold,
        recall=tp / pos_total,
        specificity=tn / neg_total,
        precision=tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        realised_flag_rate=int(pos.sum()) / len(scores),
    )


def metrics_at_anchor(scores, labels, anchor_kind, anchor_value):
    if anchor_kind == "flag_rate":
        return metrics_at_flag_rate(scores, labels, anchor_value)
    elif anchor_kind == "recall":
        return metrics_at_recall(scores, labels, anchor_value)
    raise ValueError(anchor_kind)


def _ci(arr):
    arr = np.array([v for v in arr if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)))


def paired_bootstrap(bundle, year):
    """One shared resample per draw; compute metrics for every (model,
    anchor) on the same indices to get paired CIs."""
    rng = np.random.default_rng(RNG_SEED + year * 1000)
    pids = bundle["pids"]
    unique_pids = np.array(sorted(set(pids.tolist())))
    pid_to_rows = {p: np.where(pids == p)[0] for p in unique_pids}
    labels = bundle["labels"]

    # Storage: boot[(model_key, anchor_label, metric)] = list of values
    keys = [k for _, k, _ in MODELS]
    boot = {(m, a[2], met): []
            for m in keys for a in ANCHORS
            for met in ("recall", "specificity", "precision",
                        "realised_flag_rate")}
    achievable_count = {(m, a[2]): 0 for m in keys for a in ANCHORS}

    for _ in range(N_BOOT):
        sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = np.concatenate([pid_to_rows[p] for p in sampled])
        yb = labels[idx]
        if yb.sum() == 0:
            continue
        for m_key in keys:
            sb = bundle[m_key][idx]
            for kind, val, lbl in ANCHORS:
                r = metrics_at_anchor(sb, yb, kind, val)
                if not r["achievable"]:
                    continue
                achievable_count[(m_key, lbl)] += 1
                for met in ("recall", "specificity", "precision",
                            "realised_flag_rate"):
                    boot[(m_key, lbl, met)].append(r[met])
    return boot, achievable_count


def main():
    print("Loading 4-way matched cohort (Pillar-0 = user reproduction) …")
    comp = pd.read_csv(os.path.join(OUTPUT_DIR, "four_way_comp_df.csv"))
    comp["pid"] = comp["pid"].astype(str)
    print(f"  comp_df: {len(comp)} rows, {comp['pid'].nunique()} pids")

    print("Attaching PLCOm …")
    comp = _attach_plcom(comp)

    rows = []
    flags = []
    for year in range(1, 7):
        bundle = _filter_year(comp, year)
        if bundle is None:
            flags.append(f"Year {year}: no complete-case rows")
            continue
        n = len(bundle["labels"])
        n_cancer = int(bundle["labels"].sum())
        n_pid = len(set(bundle["pids"].tolist()))
        prev = n_cancer / n
        print(f"\nYear {year}: N={n}, patients={n_pid}, cancers={n_cancer} "
              f"(prev={prev:.4f})")

        boot, achiev = paired_bootstrap(bundle, year)

        for mname, key, _tmpl in MODELS:
            scores = bundle[key]
            labels = bundle["labels"]
            row = dict(Model=mname, Year=year, N=n,
                       N_patients=n_pid, N_cancer=n_cancer, prev=prev)
            for kind, val, lbl in ANCHORS:
                p = metrics_at_anchor(scores, labels, kind, val)
                if not p["achievable"]:
                    flags.append(
                        f"{mname} year {year} {lbl}: not achievable")
                    for met in ("threshold", "recall", "specificity",
                                "precision", "realised_flag_rate"):
                        row[f"{met}@{lbl}"] = np.nan
                    for met in ("recall", "specificity", "precision"):
                        row[f"{met}@{lbl}_lo"] = np.nan
                        row[f"{met}@{lbl}_hi"] = np.nan
                    row[f"boot_n@{lbl}"] = 0
                    continue
                for met in ("threshold", "recall", "specificity",
                            "precision", "realised_flag_rate"):
                    row[f"{met}@{lbl}"] = p[met]
                for met in ("recall", "specificity", "precision"):
                    lo, hi = _ci(boot[(key, lbl, met)])
                    row[f"{met}@{lbl}_lo"] = lo
                    row[f"{met}@{lbl}_hi"] = hi
                row[f"boot_n@{lbl}"] = achiev[(key, lbl)]
            rows.append(row)
            # Compact console summary
            ss = ", ".join(
                f"{lbl}: Sp={row[f'specificity@{lbl}']:.3f}/Pr={row[f'precision@{lbl}']:.3f}/R={row[f'recall@{lbl}']:.3f}"
                for _, _, lbl in ANCHORS)
            print(f"  {mname:<16}  {ss}")

    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, "task1_sp_prec_recalls.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    if flags:
        print("\nFLAGS")
        print("-----")
        for f in flags:
            print(f"  - {f}")

    # Markdown summary — one section per anchor
    print("\n" + "=" * 110)
    print("TASK 1 — Sp / Prec / Recall at flag-rate and recall anchors")
    print("(point [lo, hi]; paired patient-clustered bootstrap n=1000)")
    print("=" * 110)
    for kind, val, lbl in ANCHORS:
        if kind == "flag_rate":
            header = f"### Anchor: flag rate ≈ {int(val*100)}% ({lbl})"
        else:
            header = f"### Anchor: recall ≥ {int(val*100)}% ({lbl})"
        print(f"\n{header}")
        print("| Model | Year | N (cancers) | Recall | Specificity | "
              "Precision |")
        print("|-------|------|-------------|--------|-------------|"
              "-----------|")

        def f(p, lo, hi):
            if pd.isna(p):
                return "n/a"
            return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"

        for r in rows:
            print(
                f"| {r['Model']} | {r['Year']} | {r['N']} "
                f"({r['N_cancer']}) | "
                f"{f(r[f'recall@{lbl}'], r[f'recall@{lbl}_lo'], r[f'recall@{lbl}_hi'])} | "
                f"{f(r[f'specificity@{lbl}'], r[f'specificity@{lbl}_lo'], r[f'specificity@{lbl}_hi'])} | "
                f"{f(r[f'precision@{lbl}'], r[f'precision@{lbl}_lo'], r[f'precision@{lbl}_hi'])} |"
            )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
