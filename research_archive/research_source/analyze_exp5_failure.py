"""Exp5 failure analysis — A (precision ceiling), B (same-day spatial confusion), C (domain-day).

Analysis only. Trains nothing. Reads recovered Exp5 artifacts from
results/firecube_direct_10km_artifact_rerun/ (methodology identical to Exp5).

Never uses fire-derived fields as predictors: scores come from the Exp5 model
(predictors t-10..t-1 only) or from FWI(t-1) unit max. burned_areas is used only
as the label y_true.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
RERUN = ROOT / "results" / "firecube_direct_10km_artifact_rerun"
ORIG = ROOT / "results" / "firecube_direct_10km"
OUT = ROOT / "results" / "firecube_exp5_failure_analysis"

RECALL_TARGETS = (0.20, 0.30, 0.50)
PRECISION_GOAL = 0.05
TOPK = (10, 25, 50, 100)
SEED = 1234

SPLITS = {
    "2020": {"pred": "test_predictions_2020.npz", "feat": "test2020"},
    "2021": {"pred": "test_predictions_2021.npz", "feat": "test2021"},
}


# ---------------------------------------------------------------- helpers

def quantiles(x, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"q{int(q * 100)}": None for q in qs}
    return {f"q{int(q * 100)}": float(v) for q, v in zip(qs, np.quantile(x, qs))}


def score_summary(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    out = {
        "n_pos": int(pos.size),
        "n_neg": int(neg.size),
        "pos_mean": float(pos.mean()) if pos.size else None,
        "neg_mean": float(neg.mean()) if neg.size else None,
        "pos_median": float(np.median(pos)) if pos.size else None,
        "neg_median": float(np.median(neg)) if neg.size else None,
        "pos_quantiles": quantiles(pos),
        "neg_quantiles": quantiles(neg),
    }
    if pos.size and neg.size:
        out["mean_diff_pos_minus_neg"] = float(pos.mean() - neg.mean())
        out["median_diff_pos_minus_neg"] = float(np.median(pos) - np.median(neg))
    else:
        out["mean_diff_pos_minus_neg"] = None
        out["median_diff_pos_minus_neg"] = None
    return out


def auroc(y, s):
    y = np.asarray(y).astype(int)
    if y.size == 0 or y.min() == y.max():
        return None
    return float(roc_auc_score(y, np.asarray(s, dtype=np.float64)))


def auroc_from_ranks(y, s):
    """Mann-Whitney AUROC with tie-average ranks (fast for per-day loops)."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(np.asarray(s, dtype=np.float64))
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_at_recalls(y_true, p, targets=RECALL_TARGETS):
    """Exact empirical operating curve (same construction as Exp5)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n_pos = y.sum()
    n_neg = float(y.size - n_pos)
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    rec = tp / max(n_pos, 1.0)
    prec = tp / np.maximum(tp + fp, 1.0)
    out = {}
    for t in targets:
        idx = int(np.searchsorted(rec, t, side="left"))
        if n_pos == 0 or idx >= len(rec):
            out[str(t)] = {"target_recall": t, "precision": None}
            continue
        out[str(t)] = {
            "target_recall": t,
            "recall_achieved": float(rec[idx]),
            "true_positive_rate": float(rec[idx]),
            "min_fpr_required": float(fp[idx] / max(n_neg, 1.0)),
            "precision_empirical": float(prec[idx]),
            "threshold_on_exp5_score": float(p[order[idx]]),
            "n_alerts_at_target": int(idx + 1),
        }
    return out


def load_split(year: str):
    cfg = SPLITS[year]
    zp = np.load(RERUN / cfg["pred"], allow_pickle=False)
    zf = np.load(RERUN / "unit_day_features.npz", allow_pickle=False)
    pref = cfg["feat"]

    unit_id = zp["unit_id"].astype(np.int64)
    t = zp["t"].astype(np.int64)
    date = zp["date"].astype(str)
    y = zp["y_true"].astype(np.float64)
    s = zp["y_score"].astype(np.float64)

    fy = zf[f"{pref}__y"].astype(np.float64)
    ft = zf[f"{pref}__t"].astype(np.int64)
    clc = zf[f"{pref}__clc"].astype(np.float32)
    fwi = np.nan_to_num(zf[f"{pref}__fwi"], nan=-1.0).astype(np.float64)

    # Row alignment guard between prediction export and feature cache
    if fy.shape[0] != y.shape[0]:
        raise RuntimeError(f"{year}: pred rows {y.shape[0]} != feature rows {fy.shape[0]}")
    if not np.array_equal(ft, t):
        raise RuntimeError(f"{year}: day index mismatch between predictions and features")
    if not np.array_equal(fy, y):
        raise RuntimeError(f"{year}: label mismatch between predictions and features")

    return {
        "unit_id": unit_id,
        "t": t,
        "date": date,
        "y": y,
        "score": s,
        "clc": clc,
        "fwi": fwi,
    }


# ---------------------------------------------------------------- part A

def part_a(d: dict, year: str, metrics: dict) -> dict:
    y, s = d["y"], d["score"]
    pi = float(y.mean())
    pr = precision_at_recalls(y, s)
    block = {
        "n": int(y.size),
        "n_pos": int(y.sum()),
        "n_neg": int(y.size - y.sum()),
        "base_rate": pi,
        "roc_auc": auroc(y, s),
        "pr_auc": float(average_precision_score(y, s)),
        "recall_targets": {},
    }
    any_ok = False
    for key, r in pr.items():
        prec = r.get("precision_empirical")
        r = dict(r)
        r["lift_vs_base_rate"] = (prec / pi) if (prec is not None and pi > 0) else None
        r["meets_5pct_precision"] = bool(prec is not None and prec >= PRECISION_GOAL)
        if key in ("0.2", "0.3") and r["meets_5pct_precision"]:
            any_ok = True
        block["recall_targets"][key] = r
    block["can_reach_5pct_precision_at_20_or_30_recall"] = any_ok

    # cross-check against stored metrics.json for the same split
    m = metrics["models"]["XGBoost_direct_10km"][year]
    block["cross_check_vs_metrics_json"] = {
        "n_match": int(m["n"]) == block["n"],
        "n_pos_match": int(m["n_pos"]) == block["n_pos"],
        "pr_auc_metrics": m["pr_auc"],
        "pr_auc_recomputed": block["pr_auc"],
        "pr_auc_abs_diff": abs(m["pr_auc"] - block["pr_auc"]),
        "roc_auc_metrics": m["roc_auc"],
        "roc_auc_recomputed": block["roc_auc"],
        "roc_auc_abs_diff": abs(m["roc_auc"] - block["roc_auc"]),
    }
    return block


# ---------------------------------------------------------------- part B

def clc_matched_negatives(dom_clc, y, rng):
    """Sample negatives so their dominant-CLC counts mirror the positives."""
    pos_idx = np.flatnonzero(y > 0)
    neg_idx = np.flatnonzero(y == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return np.array([], dtype=np.int64)
    pos_cls = dom_clc[pos_idx]
    neg_cls = dom_clc[neg_idx]
    keep = []
    for c in np.unique(pos_cls):
        need = int((pos_cls == c).sum())
        cand = neg_idx[neg_cls == c]
        if cand.size == 0 or need == 0:
            continue
        keep.append(rng.choice(cand, size=min(need, cand.size), replace=False))
    if not keep:
        return np.array([], dtype=np.int64)
    return np.concatenate(keep)


def per_day_stats(t, y, s):
    """Within-day discrimination: AUROC per fire day, positive percentile ranks, top-k hits."""
    uniq, inv = np.unique(t, return_inverse=True)
    aurocs = []
    pos_pct = []
    topk_hits = {k: 0 for k in TOPK}
    n_eval_days = 0
    for gi in range(len(uniq)):
        m = inv == gi
        yy = y[m]
        if yy.sum() == 0 or yy.sum() == yy.size:
            continue
        ss = s[m]
        a = auroc_from_ranks(yy > 0, ss)
        if a is not None:
            aurocs.append(a)
        n_eval_days += 1
        r = rankdata(ss)  # 1 = lowest score
        pct = (r - 1.0) / max(len(ss) - 1, 1)  # 1.0 = highest scored unit that day
        pos_pct.extend(pct[yy > 0].tolist())
        order = np.argsort(-ss, kind="mergesort")
        ranked_y = yy[order]
        for k in TOPK:
            if ranked_y[:k].sum() > 0:
                topk_hits[k] += 1
    aurocs = np.asarray(aurocs, dtype=np.float64)
    pos_pct = np.asarray(pos_pct, dtype=np.float64)
    return {
        "n_days_evaluable": int(n_eval_days),
        "auroc_mean": float(aurocs.mean()) if aurocs.size else None,
        "auroc_median": float(np.median(aurocs)) if aurocs.size else None,
        "auroc_quantiles": quantiles(aurocs),
        "frac_days_auroc_above_0.5": float((aurocs > 0.5).mean()) if aurocs.size else None,
        "frac_days_auroc_above_0.7": float((aurocs > 0.7).mean()) if aurocs.size else None,
        "positive_unit_percentile_rank": {
            "n_positives": int(pos_pct.size),
            "mean": float(pos_pct.mean()) if pos_pct.size else None,
            "median": float(np.median(pos_pct)) if pos_pct.size else None,
            "quantiles": quantiles(pos_pct),
            "note": "1.0 = highest-scored unit that day; 0.5 = no within-day skill",
        },
        "topk_day_hit_rate": {
            str(k): (topk_hits[k] / n_eval_days if n_eval_days else None) for k in TOPK
        },
    }


def part_b(d: dict, rng) -> dict:
    y, s, t, fwi, clc = d["y"], d["score"], d["t"], d["fwi"], d["clc"]

    fire_days = np.unique(t[y > 0])
    fd = np.isin(t, fire_days)
    y_fd, s_fd, t_fd, fwi_fd, clc_fd = y[fd], s[fd], t[fd], fwi[fd], clc[fd]
    dom = np.argmax(clc_fd, axis=1)

    pos = y_fd > 0
    block = {
        "n_unit_days_total": int(y.size),
        "n_fire_days": int(fire_days.size),
        "n_unit_days_on_fire_days": int(y_fd.size),
        "n_pos_on_fire_days": int(pos.sum()),
        "n_neg_on_fire_days": int((~pos).sum()),
        "base_rate_on_fire_days": float(pos.mean()),
        "xgb_score": {
            "pooled_within_fire_day_auroc": auroc(pos, s_fd),
            "per_day": per_day_stats(t_fd, y_fd, s_fd),
            "scores": score_summary(s_fd[pos], s_fd[~pos]),
        },
        "fwi_tminus1_unit_max": {
            "pooled_within_fire_day_auroc": auroc(pos, fwi_fd),
            "per_day": per_day_stats(t_fd, y_fd, fwi_fd),
            "scores": score_summary(fwi_fd[pos], fwi_fd[~pos]),
        },
    }

    matched = clc_matched_negatives(dom, y_fd, rng)
    pos_idx = np.flatnonzero(pos)
    if matched.size and pos_idx.size:
        ym = np.concatenate([np.ones(pos_idx.size), np.zeros(matched.size)])
        sm = np.concatenate([s_fd[pos_idx], s_fd[matched]])
        fm = np.concatenate([fwi_fd[pos_idx], fwi_fd[matched]])
        block["clc_matched"] = {
            "n_matched_negatives": int(matched.size),
            "matching": "dominant CLC class (argmax of 10 CLC_2018 fractions) count-matched to positives",
            "xgb_auroc": auroc(ym, sm),
            "fwi_auroc": auroc(ym, fm),
            "xgb_scores": score_summary(s_fd[pos_idx], s_fd[matched]),
        }
    else:
        block["clc_matched"] = {"n_matched_negatives": 0, "xgb_auroc": None, "fwi_auroc": None}

    block["interpretation_key"] = (
        "pooled AUROC mixes day effects; per_day AUROC and positive percentile rank isolate "
        "'can the model pick WHICH unit ignites on a day it already knows is a fire day'."
    )
    return block


# ---------------------------------------------------------------- part C

def part_c(d: dict) -> dict:
    y, s, t, fwi, date = d["y"], d["score"], d["t"], d["fwi"], d["date"]
    uniq, inv = np.unique(t, return_inverse=True)
    n_days = len(uniq)

    y_day = np.zeros(n_days)
    fwi_mean = np.zeros(n_days)
    fwi_max = np.zeros(n_days)
    sc_mean = np.zeros(n_days)
    sc_max = np.zeros(n_days)
    sc_q99 = np.zeros(n_days)
    n_pos_units = np.zeros(n_days)
    day_dates = []
    for gi in range(n_days):
        m = inv == gi
        y_day[gi] = 1.0 if y[m].sum() > 0 else 0.0
        n_pos_units[gi] = float(y[m].sum())
        fwi_mean[gi] = float(np.nanmean(fwi[m]))
        fwi_max[gi] = float(np.nanmax(fwi[m]))
        sc_mean[gi] = float(np.mean(s[m]))
        sc_max[gi] = float(np.max(s[m]))
        sc_q99[gi] = float(np.quantile(s[m], 0.99))
        day_dates.append(str(date[m][0]))

    return {
        "definition": (
            "Day-level fire-conduciveness: one row per summer day, label = 1 if ANY unit "
            "has a fire start that day. Scores use only t-10..t-1 predictors aggregated "
            "over the whole domain. This is NOT local spatial discrimination."
        ),
        "n_days": int(n_days),
        "n_fire_days": int(y_day.sum()),
        "fire_day_base_rate": float(y_day.mean()),
        "auroc": {
            "domain_mean_fwi_tminus1": auroc(y_day, fwi_mean),
            "domain_max_fwi_tminus1": auroc(y_day, fwi_max),
            "domain_mean_xgb_score": auroc(y_day, sc_mean),
            "domain_max_xgb_score": auroc(y_day, sc_max),
            "domain_q99_xgb_score": auroc(y_day, sc_q99),
        },
        "score_summaries": {
            "domain_mean_fwi": score_summary(fwi_mean[y_day == 1], fwi_mean[y_day == 0]),
            "domain_mean_xgb": score_summary(sc_mean[y_day == 1], sc_mean[y_day == 0]),
        },
        "n_pos_units_on_fire_days": {
            "mean": float(n_pos_units[y_day == 1].mean()) if y_day.sum() else None,
            "median": float(np.median(n_pos_units[y_day == 1])) if y_day.sum() else None,
            "max": float(n_pos_units.max()),
        },
        "date_range": [day_dates[0], day_dates[-1]] if day_dates else None,
    }


# ---------------------------------------------------------------- outputs

def write_tables(out: dict):
    tdir = OUT / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    rows = ["year,target_recall,recall_achieved,min_fpr,precision,lift_vs_base,n_alerts,meets_5pct"]
    for yr, b in out["A_prevalence_precision_ceiling"]["years"].items():
        for r in b["recall_targets"].values():
            if r.get("precision_empirical") is None:
                continue
            rows.append(
                f"{yr},{r['target_recall']},{r['recall_achieved']:.6f},{r['min_fpr_required']:.6f},"
                f"{r['precision_empirical']:.6f},{r['lift_vs_base_rate']:.2f},"
                f"{r['n_alerts_at_target']},{r['meets_5pct_precision']}"
            )
    (tdir / "precision_ceiling.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = [
        "year,pooled_within_fire_day_auroc_xgb,per_day_auroc_mean_xgb,per_day_auroc_median_xgb,"
        "pos_pct_rank_median_xgb,clc_matched_auroc_xgb,pooled_auroc_fwi,per_day_auroc_mean_fwi,"
        "domain_day_auroc_mean_fwi,domain_day_auroc_max_xgb"
    ]
    for yr in SPLITS:
        b = out["B_same_day_spatial_confusion"]["years"][yr]
        c = out["C_domain_day_baseline"]["years"][yr]

        def f(v):
            return "" if v is None else f"{v:.4f}"

        rows.append(
            f"{yr},{f(b['xgb_score']['pooled_within_fire_day_auroc'])},"
            f"{f(b['xgb_score']['per_day']['auroc_mean'])},"
            f"{f(b['xgb_score']['per_day']['auroc_median'])},"
            f"{f(b['xgb_score']['per_day']['positive_unit_percentile_rank']['median'])},"
            f"{f(b['clc_matched']['xgb_auroc'])},"
            f"{f(b['fwi_tminus1_unit_max']['pooled_within_fire_day_auroc'])},"
            f"{f(b['fwi_tminus1_unit_max']['per_day']['auroc_mean'])},"
            f"{f(c['auroc']['domain_mean_fwi_tminus1'])},"
            f"{f(c['auroc']['domain_max_xgb_score'])}"
        )
    (tdir / "spatial_vs_domain_day.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_summary(out: dict):
    L = ["# Exp5 failure analysis — A / B / C", ""]
    L.append(f"Artifacts: `{RERUN.name}` (Exp5-identical methodology; predictors t-10..t-1 only)")
    L.append("")

    L.append("## A — Prevalence / precision ceiling")
    L.append("")
    L.append(out["A_prevalence_precision_ceiling"]["overall_verdict"])
    L.append("")
    for yr, b in out["A_prevalence_precision_ceiling"]["years"].items():
        L.append(
            f"- **{yr}**: base_rate={b['base_rate']:.3e}, ROC={b['roc_auc']:.4f}, PR={b['pr_auc']:.5f}"
        )
        for r in b["recall_targets"].values():
            if r.get("precision_empirical") is None:
                continue
            L.append(
                f"  - R={r['target_recall']:.0%}: minFPR={r['min_fpr_required']:.5f}, "
                f"P={r['precision_empirical']:.5f} (x{r['lift_vs_base_rate']:.1f} base), "
                f"alerts={r['n_alerts_at_target']}, >=5%? {r['meets_5pct_precision']}"
            )
    L.append("")

    L.append("## B — Same-day spatial confusion (fire days only)")
    L.append("")
    for yr in SPLITS:
        b = out["B_same_day_spatial_confusion"]["years"][yr]
        x = b["xgb_score"]
        L.append(
            f"- **{yr}**: {b['n_fire_days']} fire days, {b['n_pos_on_fire_days']} burned units, "
            f"{b['n_neg_on_fire_days']} unburned"
        )
        L.append(
            f"  - XGB pooled AUROC={x['pooled_within_fire_day_auroc']:.4f}; "
            f"per-day AUROC mean={x['per_day']['auroc_mean']:.4f}, "
            f"median={x['per_day']['auroc_median']:.4f}"
        )
        L.append(
            f"  - burned-unit percentile rank within day: median="
            f"{x['per_day']['positive_unit_percentile_rank']['median']:.4f}"
        )
        L.append(
            f"  - CLC-matched AUROC={b['clc_matched']['xgb_auroc']:.4f} "
            f"(n_matched={b['clc_matched']['n_matched_negatives']})"
        )
        L.append(
            f"  - FWI(t-1) per-day AUROC mean="
            f"{b['fwi_tminus1_unit_max']['per_day']['auroc_mean']:.4f}"
        )
        L.append(
            f"  - top-k day hit rate: "
            + ", ".join(f"top{k}={v:.3f}" for k, v in x["per_day"]["topk_day_hit_rate"].items())
        )
    L.append("")

    L.append("## C — Domain-day baseline (day-level, not local)")
    L.append("")
    for yr in SPLITS:
        c = out["C_domain_day_baseline"]["years"][yr]
        a = c["auroc"]
        L.append(
            f"- **{yr}**: {c['n_days']} days, fire-day rate={c['fire_day_base_rate']:.3f}"
        )
        L.append(
            f"  - AUROC domain mean FWI={a['domain_mean_fwi_tminus1']:.4f}, "
            f"domain max FWI={a['domain_max_fwi_tminus1']:.4f}"
        )
        L.append(
            f"  - AUROC domain mean XGB={a['domain_mean_xgb_score']:.4f}, "
            f"domain max XGB={a['domain_max_xgb_score']:.4f}"
        )
    L.append("")
    L.append("## Verdict")
    L.append("")
    for line in out["comparison_verdict"]["statements"]:
        L.append(f"- {line}")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def build_verdict(out: dict) -> dict:
    st = []
    per_day = []
    domain = []
    for yr in SPLITS:
        b = out["B_same_day_spatial_confusion"]["years"][yr]
        c = out["C_domain_day_baseline"]["years"][yr]
        per_day.append(b["xgb_score"]["per_day"]["auroc_mean"])
        domain.append(c["auroc"]["domain_mean_xgb_score"])
        st.append(
            f"{yr}: within-fire-day per-day AUROC (XGB) = "
            f"{b['xgb_score']['per_day']['auroc_mean']:.4f}; "
            f"domain-day AUROC (mean XGB) = {c['auroc']['domain_mean_xgb_score']:.4f}; "
            f"domain-day AUROC (mean FWI) = {c['auroc']['domain_mean_fwi_tminus1']:.4f}"
        )
    pd_mean = float(np.mean([v for v in per_day if v is not None]))
    dm_mean = float(np.mean([v for v in domain if v is not None]))
    gap = dm_mean - pd_mean
    if pd_mean < 0.65 and dm_mean > 0.75:
        mode = (
            "DAY-LEVEL SKILL, PLACE-LEVEL CONFUSION: the model separates fire days from "
            "non-fire days well but cannot say which ~10 km unit ignites on a given day."
        )
    elif pd_mean >= 0.65 and dm_mean > 0.75:
        mode = (
            "BOTH LEVELS CARRY SIGNAL: local ranking is informative on fire days; the "
            "precision ceiling is then driven mainly by prevalence, not by spatial confusion."
        )
    elif pd_mean >= 0.65:
        mode = "LOCAL SIGNAL WITHOUT STRONG DAY SIGNAL (unusual; inspect day aggregation)."
    else:
        mode = "WEAK AT BOTH LEVELS: current predictors carry little usable ignition signal."
    return {
        "statements": st,
        "mean_within_fire_day_per_day_auroc": pd_mean,
        "mean_domain_day_auroc": dm_mean,
        "domain_minus_local_auroc_gap": gap,
        "failure_mode": mode,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)

    for req in ("unit_day_features.npz", "test_predictions_2020.npz", "test_predictions_2021.npz", "metrics.json"):
        if not (RERUN / req).exists():
            raise RuntimeError(f"missing recovered artifact: {RERUN / req}")

    metrics = json.loads((RERUN / "metrics.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)

    cfg = {
        "experiment": "exp5_failure_analysis",
        "trains_model": False,
        "modifies_prior_results": False,
        "artifact_source": str(RERUN),
        "original_exp5_metrics": str(ORIG / "metrics.json"),
        "predictor_policy": "scores from Exp5 model (t-10..t-1 predictors) or FWI(t-1) unit max; burned_areas used only as label",
        "recall_targets": list(RECALL_TARGETS),
        "precision_goal": PRECISION_GOAL,
        "topk": list(TOPK),
        "seed": SEED,
        "parts": {
            "A": "empirical precision/FPR at recall targets on full 10 km test grids",
            "B": "fire-days-only burned vs unburned unit discrimination (pooled, per-day, CLC-matched)",
            "C": "day-level fire-conduciveness from domain-aggregated predictors",
        },
    }
    (OUT / "analysis_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    A = {
        "source": f"{RERUN.name} prediction exports (recomputed from y_score) + metrics.json cross-check",
        "precision_goal": PRECISION_GOAL,
        "years": {},
    }
    B = {"years": {}, "score_source": "Exp5 XGB y_score from recovered predictions"}
    C = {"years": {}}

    for yr in SPLITS:
        print(f"[{yr}] loading recovered artifacts...", flush=True)
        d = load_split(yr)
        print(f"[{yr}] n={d['y'].size} pos={int(d['y'].sum())}", flush=True)

        A["years"][yr] = part_a(d, yr, metrics)
        print(f"[{yr}] part A done", flush=True)
        B["years"][yr] = part_b(d, rng)
        print(f"[{yr}] part B done", flush=True)
        C["years"][yr] = part_c(d)
        print(f"[{yr}] part C done", flush=True)

    ok = all(not b["can_reach_5pct_precision_at_20_or_30_recall"] for b in A["years"].values())
    A["overall_verdict"] = (
        "Observed Exp5 ranking cannot reach >=5% precision at 20-30% recall on full 10 km "
        "test grids (2020 and 2021)."
        if ok else
        ">=5% precision at 20-30% recall IS reachable under Exp5 scores for at least one test year."
    )

    out = {
        "A_prevalence_precision_ceiling": A,
        "B_same_day_spatial_confusion": B,
        "C_domain_day_baseline": C,
    }
    out["comparison_verdict"] = build_verdict(out)

    (OUT / "analysis_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_tables(out)
    write_summary(out)
    print("Wrote", OUT / "analysis_metrics.json", flush=True)
    print("FAILURE MODE:", out["comparison_verdict"]["failure_mode"], flush=True)


if __name__ == "__main__":
    main()
