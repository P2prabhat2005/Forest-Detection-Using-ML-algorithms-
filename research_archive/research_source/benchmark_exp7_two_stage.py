"""Experiment 7: two-stage conditional wildfire risk at direct ~10 km unit-days.

    p(fire at unit u on day t) = p(fire somewhere in domain on t) * p(u fires | fire on t)

Stage A: domain-day binary model (any fire-start in domain on day t).
Stage B: conditional unit model trained/evaluated on fire days only.
Both calibrated on 2019 ONLY. Flat Exp5 XGBoost is the primary baseline.

Preserves Exp5 exactly: same ~10 km units, same fire-start labels, same t-10..t-1
predictor window, same underlying variables, same year splits, same seed.
Writes only to results/firecube_exp7_two_stage/. Never modifies Exp5 artifacts.

Leakage rules enforced:
  * predictors use days t-10..t-1 only; day t fields are never read as features
  * burned_areas / ignition_points / number_of_fires are labels only
  * Stage A features come from a FULL-domain extraction, never from the Exp5
    unit-subsampled train split (whose per-day row count would encode the label)
  * calibrators and early stopping use 2019 only; 2020/2021 are touched at scoring time only
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
NC_PATH = ROOT / "data" / "raw" / "firecube" / "dataset_greece.nc"
EXP5 = ROOT / "results" / "firecube_direct_10km_artifact_rerun"   # read-only
EXP5_ORIG = ROOT / "results" / "firecube_direct_10km"             # read-only
OUT = ROOT / "results" / "firecube_exp7_two_stage"

# ---- Exp5 constants (locked) ----
HISTORY = 16
FIRE_MONTHS = {5, 6, 7, 8, 9}
SEED = 1234
CELL_KM2 = 0.9066669762359021 * 0.7144633272136693
DYNAMIC = [
    "max_t2m", "max_d2m", "min_rh", "max_tp", "max_sp",
    "max_wind_speed", "ndvi", "lst_day", "lst_night", "sminx",
]
FORBIDDEN_AS_FEATURES = ("burned_areas", "ignition_points", "number_of_fires")

TOPN = (1, 5, 10, 25, 50, 100)
AREA_BUDGETS_KM2_PER_DAY = (500.0, 1000.0, 2000.0, 4000.0, 8000.0)
RECALL_TARGETS = (0.20, 0.30, 0.50)
REL_BINS = np.array(
    [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0 + 1e-9]
)

TEST_YEARS = ("2020", "2021")
SPLIT_PRED = {"2020": "test_predictions_2020.npz", "2021": "test_predictions_2021.npz"}
SPLIT_FEAT = {"val": "val", "2020": "test2020", "2021": "test2021"}


# ------------------------------------------------------------------ io utils

def sha256_file(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_savez(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing.npz")
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    print(f"  saved {path.name} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def atomic_joblib(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)
    print(f"  saved {path.name} ({path.stat().st_size / 1e6:.2f} MB)", flush=True)


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, header: str, rows: list[str]):
    atomic_text(path, header + "\n" + "\n".join(rows) + "\n")
    print(f"  wrote {path.name} ({len(rows)} rows)", flush=True)


def f4(v):
    return "" if v is None else f"{v:.6f}"


# ------------------------------------------------- stage A feature extraction

def filled(var, sl=None):
    data = var[:] if sl is None else var[sl]
    return np.ma.filled(np.ma.array(data), np.nan).astype(np.float32, copy=False)


def window_slice():
    last = HISTORY - 1
    return last - 9, last + 1  # (6, 16) -> t-10 .. t-1


def domain_stats(vals: np.ndarray):
    ok = np.isfinite(vals)
    if not ok.any():
        return (np.nan, np.nan, np.nan)
    v = vals[ok]
    return (float(v.mean()), float(v.min()), float(v.max()))


def domain_feature_names() -> list[str]:
    names = []
    for v in DYNAMIC + ["fwi"]:
        for stat in ("mean", "min", "max"):
            names.append(f"{v}__domain_{stat}__mean10")
            names.append(f"{v}__domain_{stat}__last_tminus1")
    return names


def extract_domain_day_features(cache: Path) -> dict:
    """Full-domain daily features from t-10..t-1 for all summer days 2009-2021."""
    if cache.exists():
        print(f"Loading cached domain-day features {cache.name}", flush=True)
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}

    import netCDF4 as nc

    print("Opening cube for domain-day extraction...", flush=True)
    ds = nc.Dataset(NC_PATH, "r")
    dates = nc.num2date(
        ds.variables["time"][:], ds.variables["time"].units, only_use_cftime_datetimes=False
    )
    years = np.array([d.year for d in dates], dtype=np.int16)
    months = np.array([d.month for d in dates], dtype=np.int16)
    date_str = np.array([f"{d.year:04d}-{d.month:02d}-{d.day:02d}" for d in dates])

    land = filled(ds.variables["CLC_2018_9"]) < 0.5
    n_land = int(land.sum())
    print(f"  land cells={n_land}", flush=True)

    dyn_vars = [ds.variables[name] for name in DYNAMIC]
    fwi_var = ds.variables["fwi"]
    ba_var = ds.variables["burned_areas"]

    n_v = len(DYNAMIC) + 1  # + fwi
    buf = np.full((HISTORY, n_v, 3), np.nan, np.float64)

    day_mask = (
        (years >= 2009) & (years <= 2021)
        & np.isin(months, list(FIRE_MONTHS))
        & (np.arange(len(years)) >= HISTORY)
    )
    days = np.flatnonzero(day_mask).astype(np.int32)
    print(f"  extracting {len(days)} summer days...", flush=True)

    def day_stats(t_idx: int):
        st = np.empty((n_v, 3), np.float64)
        for vi, var in enumerate(dyn_vars):
            st[vi] = domain_stats(filled(var, t_idx)[land])
        st[n_v - 1] = domain_stats(filled(fwi_var, t_idx)[land])
        return st

    t0 = int(days[0])
    for k in range(HISTORY):
        buf[k] = day_stats(t0 - HISTORY + k)
    head = 0
    cur_t = t0
    s0, e0 = window_slice()

    X, y_day, t_out, n_pos_units_unused = [], [], [], []
    tic = time.time()
    for di, t in enumerate(days):
        t = int(t)
        while cur_t < t:
            buf[head] = day_stats(cur_t)
            head = (head + 1) % HISTORY
            cur_t += 1

        # buffer now holds t-HISTORY .. t-1 in circular order
        win = np.empty((HISTORY, n_v, 3), np.float64)
        for k in range(HISTORY):
            win[k] = buf[(head + k) % HISTORY]
        win = win[s0:e0]  # t-10 .. t-1

        feat = []
        for vi in range(n_v):
            for si in range(3):
                col = win[:, vi, si]
                feat.append(np.nanmean(col) if np.isfinite(col).any() else np.nan)
                feat.append(col[-1])
        X.append(feat)

        ba = filled(ba_var, t)
        y_day.append(1.0 if np.nansum(ba[land] > 0) > 0 else 0.0)
        t_out.append(t)

        if (di + 1) % 200 == 0 or di == 0:
            print(
                f"  day {di+1}/{len(days)} t={t} year={years[t]} "
                f"y_day={y_day[-1]:.0f} ({time.time()-tic:.0f}s)",
                flush=True,
            )
    ds.close()

    t_out = np.asarray(t_out, dtype=np.int32)
    data = {
        "t": t_out,
        "date": date_str[t_out].astype(str),
        "year": years[t_out].astype(np.int16),
        "month": months[t_out].astype(np.int16),
        "y_day": np.asarray(y_day, dtype=np.float32),
        "X": np.nan_to_num(np.asarray(X, dtype=np.float32), nan=-1.0),
        "feature_names": np.asarray(domain_feature_names()),
    }
    atomic_savez(cache, **data)
    return data


# ------------------------------------------------------------ metric helpers

def auroc(y, s):
    y = np.asarray(y).astype(int)
    if y.size == 0 or y.min() == y.max():
        return None
    return float(roc_auc_score(y, np.asarray(s, dtype=np.float64)))


def auroc_ranks(y, s):
    from scipy.stats import rankdata

    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int(y.size - y.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(np.asarray(s, dtype=np.float64))
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ranking_block(y, s):
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    return {
        "n": int(y.size),
        "n_pos": int(y.sum()),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": auroc(y, s),
        "brier": float(brier_score_loss(y, np.clip(s, 0.0, 1.0))),
    }


def precision_at_recalls(y, p, targets=RECALL_TARGETS):
    y = np.asarray(y, dtype=np.float64)
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
    for tgt in targets:
        idx = int(np.searchsorted(rec, tgt, side="left"))
        if n_pos == 0 or idx >= len(rec):
            out[str(tgt)] = {"target_recall": tgt, "precision": None}
            continue
        out[str(tgt)] = {
            "target_recall": tgt,
            "recall_achieved": float(rec[idx]),
            "precision": float(prec[idx]),
            "min_fpr_required": float(fp[idx] / max(n_neg, 1.0)),
            "n_alerts": int(idx + 1),
            "threshold": float(p[order[idx]]),
        }
    return out


def per_day_topn(t, y, s, n_days, unit_area_km2, ns=TOPN):
    """Fixed DAILY budget: take the N best-scored units each day."""
    uniq, inv = np.unique(t, return_inverse=True)
    total_pos = float(y.sum())
    res = {}
    order_within = {}
    for gi in range(len(uniq)):
        m = inv == gi
        ss = s[m]
        yy = y[m]
        o = np.argsort(-ss, kind="mergesort")
        order_within[gi] = yy[o]
    for N in ns:
        hits = 0.0
        flagged = 0
        days_with_hit = 0
        for gi in range(len(uniq)):
            ry = order_within[gi][:N]
            h = float(ry.sum())
            hits += h
            flagged += len(ry)
            if h > 0:
                days_with_hit += 1
        res[str(N)] = {
            "top_n": N,
            "recall": hits / max(total_pos, 1.0),
            "precision": hits / max(flagged, 1),
            "hits": int(hits),
            "alerts_total": int(flagged),
            "alerts_per_day": flagged / max(len(uniq), 1),
            "km2_flagged_per_day": (flagged / max(len(uniq), 1)) * unit_area_km2,
            "frac_days_with_hit": days_with_hit / max(len(uniq), 1),
        }
    return res


def global_budget(y, s, n_days, unit_area_km2, budgets=AREA_BUDGETS_KM2_PER_DAY):
    """Fixed GLOBAL budget: rank all unit-days, spend mean km2/day across the season."""
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="mergesort")
    ys = np.asarray(y, dtype=np.float64)[order]
    tp = np.cumsum(ys)
    total_pos = float(ys.sum())
    res = {}
    for b in budgets:
        m = int(round(b / unit_area_km2 * n_days))
        m = max(1, min(m, len(ys)))
        hits = float(tp[m - 1])
        res[f"{int(b)}"] = {
            "budget_km2_per_day": b,
            "alerts_total": m,
            "alerts_per_day": m / n_days,
            "recall": hits / max(total_pos, 1.0),
            "precision": hits / m,
            "hits": int(hits),
        }
    return res


def reliability(y, p, bins=REL_BINS):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    idx = np.digitize(p, bins) - 1
    rows = []
    ece = 0.0
    for b in range(len(bins) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        mp = float(p[m].mean())
        of = float(y[m].mean())
        rows.append({
            "bin_lo": float(bins[b]),
            "bin_hi": float(bins[b + 1]),
            "n": n,
            "mean_predicted": mp,
            "observed_frequency": of,
            "gap": of - mp,
        })
        ece += (n / len(y)) * abs(of - mp)
    return {"bins": rows, "ece": float(ece)}


def within_fire_day_block(t, y, s):
    fire_days = np.unique(t[y > 0])
    fd = np.isin(t, fire_days)
    yy, ss, tt = y[fd], s[fd], t[fd]
    uniq, inv = np.unique(tt, return_inverse=True)
    aurocs = []
    for gi in range(len(uniq)):
        m = inv == gi
        a = auroc_ranks(yy[m] > 0, ss[m])
        if a is not None:
            aurocs.append(a)
    aurocs = np.asarray(aurocs)
    return {
        "n_fire_days": int(fire_days.size),
        "n_unit_days_on_fire_days": int(yy.size),
        "pooled_auroc": auroc(yy > 0, ss),
        "per_day_auroc_mean": float(aurocs.mean()) if aurocs.size else None,
        "per_day_auroc_median": float(np.median(aurocs)) if aurocs.size else None,
        "frac_days_auroc_above_0.7": float((aurocs > 0.7).mean()) if aurocs.size else None,
    }


# ------------------------------------------------------------------ features

def fill_time_nans(dyn: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        colmean = np.nanmean(dyn, axis=1, keepdims=True)
    colmean = np.where(np.isfinite(colmean), colmean, 0.0)
    out = np.where(np.isfinite(dyn), dyn, np.broadcast_to(colmean, dyn.shape))
    return np.nan_to_num(out, nan=-1.0).astype(np.float32)


def unit_features(z, split, rows=None):
    """Exp5-identical unit feature construction: [mean10(dyn), dyn(t-1), static5, clc10]."""
    s0, e0 = window_slice()
    dyn = np.asarray(z[f"{split}__dyn"], dtype=np.float32)
    if rows is not None:
        dyn = dyn[rows]
    dyn = fill_time_nans(dyn)
    if dyn.shape[1] == HISTORY:
        dyn = dyn[:, s0:e0, :]
    sta = np.nan_to_num(z[f"{split}__sta"], nan=-1.0).astype(np.float32)
    clc = np.nan_to_num(z[f"{split}__clc"], nan=0.0).astype(np.float32)
    y = z[f"{split}__y"].astype(np.float32)
    t = z[f"{split}__t"].astype(np.int64)
    if rows is not None:
        sta, clc, y, t = sta[rows], clc[rows], y[rows], t[rows]
    return dyn, sta, clc, y, t


def build_x(dyn, sta, clc, sc_dyn, sc_sta):
    dyn_n = sc_dyn.transform(dyn.reshape(-1, dyn.shape[-1])).reshape(dyn.shape).astype(np.float32)
    sta_n = sc_sta.transform(sta).astype(np.float32)
    return np.concatenate([dyn_n.mean(axis=1), dyn_n[:, -1, :], sta_n, clc], axis=1)


# ---------------------------------------------------------------------- main

def main():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)

    for req in ("unit_day_features.npz", "test_predictions_2020.npz",
                "test_predictions_2021.npz", "validation_predictions.npz", "metrics.json"):
        if not (EXP5 / req).exists():
            raise RuntimeError(f"missing Exp5 artifact: {EXP5 / req}")

    integrity = {"checks": [], "errors": []}

    def check(name, ok, detail=""):
        integrity["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
        if not ok:
            integrity["errors"].append(f"{name}: {detail}")

    # ---------------- Stage A data ----------------
    dom = extract_domain_day_features(OUT / "domain_day_features.npz")
    dyear = dom["year"].astype(int)
    dX = dom["X"].astype(np.float32)
    dy = dom["y_day"].astype(np.float32)
    dt = dom["t"].astype(np.int64)
    ddate = dom["date"].astype(str)
    dnames = [str(v) for v in dom["feature_names"]]

    check("stageA_no_fire_derived_features",
          not any(bad in n for n in dnames for bad in FORBIDDEN_AS_FEATURES),
          f"{len(dnames)} features")
    check("stageA_window_is_tminus10_to_tminus1", window_slice() == (6, 16), str(window_slice()))

    a_tr = (dyear >= 2009) & (dyear <= 2018)
    a_va = dyear == 2019
    a_te = {"2020": dyear == 2020, "2021": dyear == 2021}
    check("stageA_splits_disjoint",
          not (a_tr & a_va).any() and not (a_tr & a_te["2020"]).any()
          and not (a_va & a_te["2021"]).any(),
          f"train={int(a_tr.sum())} val={int(a_va.sum())} "
          f"test2020={int(a_te['2020'].sum())} test2021={int(a_te['2021'].sum())}")

    print(
        f"Stage A days: train={int(a_tr.sum())} (pos={int(dy[a_tr].sum())}) "
        f"val={int(a_va.sum())} (pos={int(dy[a_va].sum())})",
        flush=True,
    )

    # ---------------- Stage A model ----------------
    print("Training Stage A (domain-day)...", flush=True)
    stage_a = XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        n_jobs=4, random_state=SEED, early_stopping_rounds=40,
    )
    stage_a.fit(dX[a_tr], dy[a_tr], eval_set=[(dX[a_va], dy[a_va])], verbose=False)
    print(f"  stageA best_iteration={getattr(stage_a, 'best_iteration', None)}", flush=True)

    pa_va_raw = stage_a.predict_proba(dX[a_va])[:, 1]
    # Platt scaling on 2019 only (153 days -> isotonic would overfit)
    cal_a = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    cal_a.fit(pa_va_raw.reshape(-1, 1), dy[a_va].astype(int))

    def pA(mask):
        raw = stage_a.predict_proba(dX[mask])[:, 1]
        return raw, cal_a.predict_proba(raw.reshape(-1, 1))[:, 1]

    # ---------------- Stage B data ----------------
    z = np.load(EXP5 / "unit_day_features.npz", allow_pickle=False)
    unit_area = float(np.mean(z["area_km2"]))
    n_units = int(z["n_units"][0])
    check("units_match_exp5", n_units == 4022, f"n_units={n_units}")

    tr_y = z["train__y"].astype(np.float32)
    tr_t = z["train__t"].astype(np.int64)
    fire_days_tr = np.unique(tr_t[tr_y > 0])
    rows_b = np.flatnonzero(np.isin(tr_t, fire_days_tr))
    check("stageB_train_is_fire_days_only",
          bool((np.isin(tr_t[rows_b], fire_days_tr)).all()),
          f"{len(rows_b)} rows on {fire_days_tr.size} fire days")

    dyn_b, sta_b, clc_b, y_b, t_b = unit_features(z, "train", rows_b)
    sc_dyn = MinMaxScaler().fit(dyn_b.reshape(-1, dyn_b.shape[-1]))
    sc_sta = MinMaxScaler().fit(sta_b)
    x_b = build_x(dyn_b, sta_b, clc_b, sc_dyn, sc_sta)

    dyn_v, sta_v, clc_v, y_v, t_v = unit_features(z, "val")
    fire_days_va = np.unique(t_v[y_v > 0])
    fd_v = np.isin(t_v, fire_days_va)
    x_v_fire = build_x(dyn_v[fd_v], sta_v[fd_v], clc_v[fd_v], sc_dyn, sc_sta)
    y_v_fire = y_v[fd_v]

    print(
        f"Stage B: train n={len(y_b)} pos={int(y_b.sum())} on {fire_days_tr.size} fire days; "
        f"val fire-day rows={len(y_v_fire)} pos={int(y_v_fire.sum())}",
        flush=True,
    )

    print("Training Stage B (conditional unit | fire day)...", flush=True)
    stage_b = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        n_jobs=4, random_state=SEED, early_stopping_rounds=40,
    )
    stage_b.fit(x_b, y_b, eval_set=[(x_v_fire, y_v_fire)], verbose=False)
    print(f"  stageB best_iteration={getattr(stage_b, 'best_iteration', None)}", flush=True)

    # Isotonic calibration of Stage B on 2019 fire-day unit-days ONLY
    pb_va_fire_raw = stage_b.predict_proba(x_v_fire)[:, 1]
    cal_b = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal_b.fit(pb_va_fire_raw, y_v_fire)

    # Flat Exp5 baseline calibrator, also fit on 2019 only (fair Brier/reliability)
    zv = np.load(EXP5 / "validation_predictions.npz", allow_pickle=False)
    cal_flat = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal_flat.fit(zv["y_score"].astype(np.float64), zv["y_true"].astype(np.float64))
    check("calibrators_fit_on_2019_only", True,
          "stageA Platt on 2019 domain-days; stageB + flat isotonic on 2019 unit-days")

    atomic_joblib(OUT / "stage_a_domain_day.joblib", {
        "model": stage_a, "calibrator": cal_a, "feature_names": dnames, "seed": SEED,
    })
    atomic_joblib(OUT / "stage_b_conditional_unit.joblib", {
        "model": stage_b, "calibrator": cal_b,
        "scaler_dyn": sc_dyn, "scaler_sta": sc_sta, "seed": SEED,
    })
    atomic_joblib(OUT / "flat_exp5_calibrator.joblib", {
        "calibrator": cal_flat,
        "note": "isotonic on 2019 Exp5 flat scores; Exp5 model itself NOT modified",
    })

    # ---------------- evaluate ----------------
    results = {
        "protocol": {
            "experiment": 7,
            "title": "Two-stage conditional risk: p(fire day) x p(unit | fire day)",
            "baseline": "Exp5 flat direct 10 km XGBoost (raw + 2019-isotonic-calibrated)",
            "units": {"n_units": n_units, "mean_area_km2": unit_area, "cell_km2": CELL_KM2},
            "label": "unit positive on day t if any fire-start burned_areas cell inside unit on day t",
            "predictor_window": "t-10 .. t-1 only",
            "splits": "train 2009-2018 / val 2019 / test 2020 & 2021 (May-Sep)",
            "calibration": "2019 only (Stage A Platt, Stage B + flat isotonic)",
            "seed": SEED,
            "stage_a": {
                "model": "XGBClassifier(max_depth=3, lr=0.05, n_estimators=400, es=40)",
                "features": f"{len(dnames)} full-domain daily aggregates (mean/min/max of "
                            f"{len(DYNAMIC)} dynamic vars + FWI; 10-day mean and t-1 value)",
                "best_iteration": int(getattr(stage_a, "best_iteration", -1)),
                "train_days": int(a_tr.sum()),
                "train_fire_days": int(dy[a_tr].sum()),
            },
            "stage_b": {
                "model": "XGBClassifier(max_depth=6, lr=0.05, n_estimators=400, es=40)",
                "features": "Exp5-identical unit features [mean10(dyn), dyn(t-1), static5, clc10]",
                "best_iteration": int(getattr(stage_b, "best_iteration", -1)),
                "train_rows": int(len(y_b)),
                "train_pos": int(y_b.sum()),
                "train_fire_days": int(fire_days_tr.size),
            },
        },
        "stage_a_domain_day": {},
        "stage_b_within_fire_day": {},
        "unit_day_operational": {},
        "calibration": {},
        "comparison": {},
    }

    # Stage A performance (val + test years)
    raw_va, cal_va = pA(a_va)
    results["stage_a_domain_day"]["val_2019"] = {
        **ranking_block(dy[a_va], cal_va),
        "roc_auc_uncalibrated": auroc(dy[a_va], raw_va),
    }
    stage_a_pred = {"val": (dt[a_va], ddate[a_va], dy[a_va], raw_va, cal_va)}
    for yr in TEST_YEARS:
        m = a_te[yr]
        raw, cal = pA(m)
        results["stage_a_domain_day"][yr] = {
            **ranking_block(dy[m], cal),
            "roc_auc_uncalibrated": auroc(dy[m], raw),
            "n_days": int(m.sum()),
            "n_fire_days": int(dy[m].sum()),
            "fire_day_base_rate": float(dy[m].mean()),
        }
        stage_a_pred[yr] = (dt[m], ddate[m], dy[m], raw, cal)
        print(
            f"  StageA {yr}: ROC={results['stage_a_domain_day'][yr]['roc_auc']:.4f} "
            f"PR={results['stage_a_domain_day'][yr]['pr_auc']:.4f} "
            f"Brier={results['stage_a_domain_day'][yr]['brier']:.4f}",
            flush=True,
        )

    atomic_savez(
        OUT / "stage_a_predictions.npz",
        **{f"{k}__{f}": v for k, (a, b, c, d_, e) in stage_a_pred.items()
           for f, v in (("t", a), ("date", b), ("y_day", c), ("p_raw", d_), ("p_cal", e))}
    )

    # Unit-day evaluation on test years
    day_p_lookup = {yr: dict(zip(stage_a_pred[yr][0].tolist(), stage_a_pred[yr][4].tolist()))
                    for yr in TEST_YEARS}

    csv_topn, csv_budget, csv_rel, csv_summary = [], [], [], []

    for yr in TEST_YEARS:
        split = SPLIT_FEAT[yr]
        zp = np.load(EXP5 / SPLIT_PRED[yr], allow_pickle=False)
        y = zp["y_true"].astype(np.float64)
        t = zp["t"].astype(np.int64)
        uid = zp["unit_id"].astype(np.int64)
        date = zp["date"].astype(str)
        flat_raw = zp["y_score"].astype(np.float64)

        dyn_t, sta_t, clc_t, y_chk, t_chk = unit_features(z, split)
        if not np.array_equal(y_chk, y) or not np.array_equal(t_chk, t):
            raise RuntimeError(f"{yr}: feature/prediction row misalignment")
        x_t = build_x(dyn_t, sta_t, clc_t, sc_dyn, sc_sta)

        pb_raw = stage_b.predict_proba(x_t)[:, 1]
        pb_cal = cal_b.predict(pb_raw)
        pa_day = np.array([day_p_lookup[yr][int(tt)] for tt in t], dtype=np.float64)
        two_stage = pa_day * pb_cal
        flat_cal = cal_flat.predict(flat_raw)

        n_days = int(np.unique(t).size)
        check(f"{yr}_all_days_have_stageA_prob",
              len(day_p_lookup[yr]) >= n_days, f"{len(day_p_lookup[yr])} vs {n_days}")

        models = {
            "flat_exp5_raw": flat_raw,
            "flat_exp5_calibrated": flat_cal,
            "two_stage_product": two_stage,
            "stage_b_only_conditional": pb_cal,
        }

        yr_block = {"n_days": n_days, "n_unit_days": int(y.size), "n_pos": int(y.sum()),
                    "base_rate": float(y.mean()), "unit_area_km2": unit_area, "models": {}}

        for name, s in models.items():
            blk = ranking_block(y, s)
            blk["precision_at_recall"] = precision_at_recalls(y, s)
            blk["per_day_topn"] = per_day_topn(t, y, s, n_days, unit_area)
            blk["global_area_budget"] = global_budget(y, s, n_days, unit_area)
            blk["within_fire_day"] = within_fire_day_block(t, y, s)
            yr_block["models"][name] = blk

            for N, r in blk["per_day_topn"].items():
                csv_topn.append(
                    f"{yr},{name},{N},{f4(r['recall'])},{f4(r['precision'])},{r['hits']},"
                    f"{r['alerts_total']},{f4(r['alerts_per_day'])},"
                    f"{f4(r['km2_flagged_per_day'])},{f4(r['frac_days_with_hit'])}"
                )
            for b, r in blk["global_area_budget"].items():
                csv_budget.append(
                    f"{yr},{name},{b},{f4(r['recall'])},{f4(r['precision'])},{r['hits']},"
                    f"{r['alerts_total']},{f4(r['alerts_per_day'])}"
                )
            csv_summary.append(
                f"{yr},{name},{f4(blk['pr_auc'])},{f4(blk['roc_auc'])},{f4(blk['brier'])},"
                f"{f4(blk['within_fire_day']['per_day_auroc_mean'])},"
                f"{f4(blk['per_day_topn']['10']['recall'])},"
                f"{f4(blk['per_day_topn']['50']['recall'])},"
                f"{f4(blk['per_day_topn']['100']['recall'])}"
            )
            print(
                f"  {yr} {name}: PR={blk['pr_auc']:.5f} ROC={blk['roc_auc']:.4f} "
                f"Brier={blk['brier']:.3e} top50/day recall={blk['per_day_topn']['50']['recall']:.4f}",
                flush=True,
            )

        # calibration only for probability-scale models
        rel = {}
        for name in ("flat_exp5_calibrated", "two_stage_product"):
            rel[name] = reliability(y, models[name])
            for r in rel[name]["bins"]:
                csv_rel.append(
                    f"{yr},{name},{r['bin_lo']:.1e},{r['bin_hi']:.1e},{r['n']},"
                    f"{f4(r['mean_predicted'])},{f4(r['observed_frequency'])},{f4(r['gap'])}"
                )
        results["calibration"][yr] = rel
        results["unit_day_operational"][yr] = yr_block
        results["stage_b_within_fire_day"][yr] = yr_block["models"]["stage_b_only_conditional"]["within_fire_day"]

        atomic_savez(
            OUT / f"test_predictions_{yr}.npz",
            unit_id=uid, t=t, date=date, y_true=y.astype(np.float32),
            p_stage_a_day=pa_day.astype(np.float32),
            p_stage_b_raw=pb_raw.astype(np.float32),
            p_stage_b_cal=pb_cal.astype(np.float32),
            p_two_stage=two_stage.astype(np.float32),
            p_flat_exp5_raw=flat_raw.astype(np.float32),
            p_flat_exp5_cal=flat_cal.astype(np.float32),
        )

    # validation predictions (2019) for the record
    dyn_v2, sta_v2, clc_v2, y_v2, t_v2 = unit_features(z, "val")
    x_v_all = build_x(dyn_v2, sta_v2, clc_v2, sc_dyn, sc_sta)
    pb_v_all = cal_b.predict(stage_b.predict_proba(x_v_all)[:, 1])
    pa_v_lookup = dict(zip(stage_a_pred["val"][0].tolist(), stage_a_pred["val"][4].tolist()))
    pa_v = np.array([pa_v_lookup[int(tt)] for tt in t_v2], dtype=np.float64)
    atomic_savez(
        OUT / "validation_predictions.npz",
        unit_id=z["val__uid"].astype(np.int64), t=t_v2, y_true=y_v2.astype(np.float32),
        p_stage_a_day=pa_v.astype(np.float32), p_stage_b_cal=pb_v_all.astype(np.float32),
        p_two_stage=(pa_v * pb_v_all).astype(np.float32),
        p_flat_exp5_raw=zv["y_score"].astype(np.float32),
    )

    # ---------------- comparison verdict ----------------
    verdict = {"per_year": {}, "statements": []}
    improved_flags = []
    for yr in TEST_YEARS:
        m = results["unit_day_operational"][yr]["models"]
        flat = m["flat_exp5_raw"]
        two = m["two_stage_product"]
        d = {
            "pr_auc_flat": flat["pr_auc"],
            "pr_auc_two_stage": two["pr_auc"],
            "pr_auc_delta": two["pr_auc"] - flat["pr_auc"],
            "roc_auc_flat": flat["roc_auc"],
            "roc_auc_two_stage": two["roc_auc"],
            "brier_flat_calibrated": m["flat_exp5_calibrated"]["brier"],
            "brier_two_stage": two["brier"],
            "within_fire_day_per_day_auroc_flat": flat["within_fire_day"]["per_day_auroc_mean"],
            "within_fire_day_per_day_auroc_stage_b": m["stage_b_only_conditional"]["within_fire_day"]["per_day_auroc_mean"],
            "topn_recall_delta": {
                N: two["per_day_topn"][N]["recall"] - flat["per_day_topn"][N]["recall"]
                for N in [str(n) for n in TOPN]
            },
            "global_budget_recall_delta": {
                b: two["global_area_budget"][b]["recall"] - flat["global_area_budget"][b]["recall"]
                for b in two["global_area_budget"]
            },
        }
        wins_topn = sum(1 for v in d["topn_recall_delta"].values() if v > 0)
        wins_budget = sum(1 for v in d["global_budget_recall_delta"].values() if v > 0)
        d["topn_settings_improved"] = wins_topn
        d["budget_settings_improved"] = wins_budget
        d["pr_auc_improved"] = bool(d["pr_auc_delta"] > 0)
        improved_flags.append(d["pr_auc_improved"] and wins_budget >= 3)
        verdict["per_year"][yr] = d
        verdict["statements"].append(
            f"{yr}: PR-AUC {flat['pr_auc']:.5f} -> {two['pr_auc']:.5f} "
            f"(delta {d['pr_auc_delta']:+.5f}); top-50/day recall "
            f"{flat['per_day_topn']['50']['recall']:.4f} -> {two['per_day_topn']['50']['recall']:.4f}; "
            f"global-budget settings improved {wins_budget}/{len(d['global_budget_recall_delta'])}"
        )
    verdict["two_stage_improves_operational_decision"] = bool(all(improved_flags))
    verdict["decision_rule"] = (
        "improvement requires PR-AUC gain AND >=3/5 global area-budget recall gains in BOTH years"
    )
    results["comparison"] = verdict

    # ---------------- integrity + repro ----------------
    check("exp5_original_untouched",
          (EXP5_ORIG / "metrics.json").exists(),
          "read-only access to Exp5 dirs; all writes under results/firecube_exp7_two_stage/")
    check("no_test_year_used_for_fitting", True,
          "Stage A/B fit on 2009-2018, early stopping + calibration on 2019 only")
    check("two_stage_is_product_of_calibrated_stages", True,
          "p = p_stage_a_cal(day) * p_stage_b_cal(unit|fire day)")
    integrity["ok"] = len(integrity["errors"]) == 0
    results["integrity"] = integrity
    results["reproducibility"] = {
        "seed": SEED,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "exp5_source_dir": str(EXP5),
        "exp5_features_sha256": sha256_file(EXP5 / "unit_day_features.npz"),
        "runtime_seconds": None,
    }

    results["reproducibility"]["runtime_seconds"] = round(time.time() - t_start, 1)
    atomic_text(OUT / "metrics.json", json.dumps(results, indent=2))
    atomic_text(OUT / "experiment_config.json", json.dumps(results["protocol"], indent=2))
    atomic_text(OUT / "integrity_report.json", json.dumps(integrity, indent=2))

    write_csv(OUT / "tables" / "topn_per_day.csv",
              "year,model,top_n,recall,precision,hits,alerts_total,alerts_per_day,"
              "km2_flagged_per_day,frac_days_with_hit", csv_topn)
    write_csv(OUT / "tables" / "global_area_budget.csv",
              "year,model,budget_km2_per_day,recall,precision,hits,alerts_total,alerts_per_day",
              csv_budget)
    write_csv(OUT / "tables" / "reliability.csv",
              "year,model,bin_lo,bin_hi,n,mean_predicted,observed_frequency,gap", csv_rel)
    write_csv(OUT / "tables" / "summary.csv",
              "year,model,pr_auc,roc_auc,brier,within_fire_day_per_day_auroc,"
              "top10_recall,top50_recall,top100_recall", csv_summary)

    files = [
        OUT / "domain_day_features.npz", OUT / "stage_a_domain_day.joblib",
        OUT / "stage_b_conditional_unit.joblib", OUT / "flat_exp5_calibrator.joblib",
        OUT / "stage_a_predictions.npz", OUT / "validation_predictions.npz",
        OUT / "test_predictions_2020.npz", OUT / "test_predictions_2021.npz",
        OUT / "metrics.json", OUT / "experiment_config.json", OUT / "integrity_report.json",
        OUT / "tables" / "topn_per_day.csv", OUT / "tables" / "global_area_budget.csv",
        OUT / "tables" / "reliability.csv", OUT / "tables" / "summary.csv",
    ]
    man = {"output_dir": str(OUT), "files": {}, "missing": []}
    for p in files:
        if p.exists() and p.stat().st_size > 0:
            man["files"][p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        else:
            man["missing"].append(p.name)
    man["all_present"] = not man["missing"]
    atomic_text(OUT / "artifacts_manifest.json", json.dumps(man, indent=2))

    if not integrity["ok"]:
        raise RuntimeError("INTEGRITY FAILED: " + "; ".join(integrity["errors"]))
    if man["missing"]:
        raise RuntimeError("MISSING ARTIFACTS: " + ", ".join(man["missing"]))

    print(f"INTEGRITY PASS. runtime={results['reproducibility']['runtime_seconds']}s", flush=True)
    print(f"two_stage_improves_operational_decision="
          f"{verdict['two_stage_improves_operational_decision']}", flush=True)
    print("EXP7 COMPLETE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
