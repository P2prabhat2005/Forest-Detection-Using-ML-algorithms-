"""Experiment 5: direct ~10 km unit-day XGBoost vs Exp4 post-hoc 10 km max pooling.

Leakage-safe: predictors use only t-10 ... t-1. Does not modify dataset_greece.nc.
No LSTM / Transformer / CNN / GNN / ensembles.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import netCDF4 as nc
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
NC_PATH = ROOT / "data" / "raw" / "firecube" / "dataset_greece.nc"
EXP4_METRICS = ROOT / "results" / "firecube_spatial_agg" / "metrics.json"
EXP4_DIR = ROOT / "results" / "firecube_spatial_agg"
OUT = ROOT / "results" / "firecube_direct_10km"

HISTORY = 16
FIRE_MONTHS = {5, 6, 7, 8, 9}
SEED = 1234
# Train keeps all positives; negatives subsampled (eval uses full summer grid).
TRAIN_NEG_POS_RATIO = 20
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
RECALL_TARGETS = (0.20, 0.30, 0.40, 0.50)
CELL_KM2 = 0.9066669762359021 * 0.7144633272136693
DY_KM = 0.9066669762359021
DX_KM = 0.7144633272136693

DYNAMIC = [
    "max_t2m", "max_d2m", "min_rh", "max_tp", "max_sp",
    "max_wind_speed", "ndvi", "lst_day", "lst_night", "sminx",
]
# Spatial aggregation of 1 km cells inside each ~10 km unit (documented in metrics).
DYNAMIC_AGG = {
    "max_t2m": "max",          # hottest cell drives danger
    "max_d2m": "max",
    "min_rh": "min",           # driest cell
    "max_tp": "mean",          # areal precipitation
    "max_sp": "mean",
    "max_wind_speed": "max",
    "ndvi": "mean",
    "lst_day": "max",
    "lst_night": "max",
    "sminx": "mean",           # soil moisture mean
}
STATIC = ["DEM", "SLOPE", "ROAD_DISTANCE", "WATERWAY_DISTANCE"]
STATIC_AGG = {
    "DEM": "mean",
    "SLOPE": "mean",
    "ROAD_DISTANCE": "min",       # closest road
    "WATERWAY_DISTANCE": "min",   # closest waterway
    "POP_DENS": "mean",
}
# CLC 10-class fractions: spatial mean → unit composition
# FWI baseline: spatial max of previous-day FWI (matches Exp4 FWI__max pooling intent)


def filled(var, sl=None):
    data = var[:] if sl is None else var[sl]
    return np.ma.filled(np.ma.array(data), np.nan).astype(np.float32, copy=False)


def window_slice():
    last = HISTORY - 1
    return last - 9, last + 1


def fill_time_nans(dyn: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        colmean = np.nanmean(dyn, axis=1, keepdims=True)
    colmean = np.where(np.isfinite(colmean), colmean, 0.0)
    out = np.where(np.isfinite(dyn), dyn, np.broadcast_to(colmean, dyn.shape))
    return np.nan_to_num(out, nan=-1.0).astype(np.float32)


def xgb_features(dyn10, static, clc):
    return np.concatenate([dyn10.mean(axis=1), dyn10[:, -1, :], static, clc], axis=1)


def build_block_units(land_y, land_x, stride_y, stride_x, label):
    by = land_y // stride_y
    bx = land_x // stride_x
    keys = by.astype(np.int64) * (int(land_x.max()) + 1 + stride_x) + bx.astype(np.int64)
    uniq, inv = np.unique(keys, return_inverse=True)
    n_units = len(uniq)
    counts = np.bincount(inv, minlength=n_units)
    areas = counts.astype(np.float64) * CELL_KM2
    return {
        "name": label,
        "kind": "block",
        "unit_id": inv.astype(np.int32),
        "n_units": int(n_units),
        "area_km2": areas,
        "mean_extent_km": {"y": float(stride_y * DY_KM), "x": float(stride_x * DX_KM)},
        "stride_cells": {"y": int(stride_y), "x": int(stride_x)},
        "n_cells": counts.astype(np.int32),
    }


def agg_land_to_units(values, unit_id, n_units, how: str) -> np.ndarray:
    """Aggregate land-cell values to units. NaNs ignored for mean; max/min skip NaN."""
    uid = unit_id
    v = np.asarray(values, dtype=np.float64)
    if how == "mean":
        ok = np.isfinite(v)
        sums = np.bincount(uid[ok], weights=v[ok], minlength=n_units)
        counts = np.bincount(uid[ok], minlength=n_units)
        out = sums / np.maximum(counts, 1)
        out[counts == 0] = np.nan
        return out.astype(np.float32)
    if how == "max":
        out = np.full(n_units, -np.inf, dtype=np.float64)
        ok = np.isfinite(v)
        if ok.any():
            np.maximum.at(out, uid[ok], v[ok])
        out[~np.isfinite(out)] = np.nan
        return out.astype(np.float32)
    if how == "min":
        out = np.full(n_units, np.inf, dtype=np.float64)
        ok = np.isfinite(v)
        if ok.any():
            np.minimum.at(out, uid[ok], v[ok])
        out[~np.isfinite(out)] = np.nan
        return out.astype(np.float32)
    raise ValueError(how)


def far_parts(y_true, y_hat):
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    far_al = fp / (fp + tp) if (fp + tp) else 0.0
    return fpr, far_al, int(tn), int(fp), int(fn), int(tp)


def metrics_at(y_true, p, thr, unit_area_km2=None, n_days=None, total_land_km2=None):
    y_hat = (np.asarray(p) >= thr).astype(np.int32)
    fpr, far_al, tn, fp, fn, tp = far_parts(y_true, y_hat)
    out = {
        "threshold": float(thr),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "fpr": float(fpr),
        "false_alarm_among_alerts": float(far_al),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    if unit_area_km2 is not None and n_days and total_land_km2:
        out["fa_units_per_1000km2_per_day"] = float(fp / n_days / total_land_km2 * 1000.0)
        fa_area = float(np.asarray(unit_area_km2)[(y_hat == 1) & (np.asarray(y_true) == 0)].sum()) if fp else 0.0
        out["fa_area_km2_per_day"] = fa_area / n_days
        out["fa_area_fraction_per_day"] = (fa_area / n_days) / total_land_km2
        out["fa_cell_equiv_per_1000km2_per_day"] = float(
            (fa_area / CELL_KM2) / n_days / total_land_km2 * 1000.0
        )
    return out


def ranking_metrics(y_true, p):
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n_pos = int(y_true.sum())
    return {
        "pr_auc": float(average_precision_score(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "n": int(len(y_true)),
        "n_pos": n_pos,
        "n_neg": int(len(y_true) - n_pos),
        "base_rate": float(y_true.mean()),
    }


def precision_at_recalls(y_true, p, targets=RECALL_TARGETS):
    y = np.asarray(y_true).astype(np.float64)
    p = np.asarray(p)
    n_pos = y.sum()
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    rec = tp / max(n_pos, 1.0)
    prec = tp / np.maximum(tp + fp, 1.0)
    out = {}
    for t in targets:
        idx = int(np.searchsorted(rec, t, side="left"))
        if n_pos == 0 or rec[-1] < t - 1e-12 or idx >= len(rec):
            out[str(t)] = {
                "target_recall": t, "precision": None, "threshold": None,
                "max_recall": float(rec[-1]) if len(rec) else 0.0,
            }
        else:
            out[str(t)] = {
                "target_recall": t,
                "precision": float(prec[idx]),
                "threshold": float(p[order[idx]]),
                "recall_achieved": float(rec[idx]),
                "fpr": float(fp[idx] / max(len(y) - n_pos, 1.0)),
            }
    return out


def best_f1_threshold(y_true, p):
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        f = f1_score(y_true, p >= t, zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, float(best_f)


def pack_eval(y, p, thr, unit_area_km2, n_days, total_land_km2):
    block = ranking_metrics(y, p)
    block["at_val_tuned"] = metrics_at(y, p, thr, unit_area_km2, n_days, total_land_km2)
    block["at_0.5"] = metrics_at(y, p, 0.5, unit_area_km2, n_days, total_land_km2)
    block["threshold_sweep"] = [
        metrics_at(y, p, t, unit_area_km2, n_days, total_land_km2) for t in THRESHOLDS
    ]
    block["precision_at_recall"] = precision_at_recalls(y, p)
    return block


def split_name(year: int) -> str | None:
    if 2009 <= year <= 2018:
        return "train"
    if year == 2019:
        return "val"
    if year == 2020:
        return "test2020"
    if year == 2021:
        return "test2021"
    return None


def extract_slice_from_exp4(metrics: dict, model_key: str) -> dict:
    m = metrics["by_scale"]["10km"]["models"][model_key]
    out = {"val_threshold": m["val_threshold"], "source": "exp4_10km_max"}
    for yr in ("2020", "2021"):
        b = m[yr]
        out[yr] = {
            "pr_auc": b["pr_auc"],
            "roc_auc": b["roc_auc"],
            "brier": b["brier"],
            "n": b["n"],
            "n_pos": b["n_pos"],
            "n_neg": b["n_neg"],
            "base_rate": b["base_rate"],
            "at_val_tuned": b["at_val_tuned"],
            "precision_at_recall": b["precision_at_recall"],
            "n_days": b.get("n_days"),
            "n_units": b.get("n_units"),
            "n_pos_unit_days": b.get("n_pos_unit_days"),
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feat_path = OUT / "unit_day_features.npz"
    meta_path = OUT / "feature_aggregation.json"

    print("Opening cube...", flush=True)
    ds = nc.Dataset(NC_PATH, "r")
    dates = nc.num2date(ds.variables["time"][:], ds.variables["time"].units, only_use_cftime_datetimes=False)
    years = np.array([d.year for d in dates], dtype=np.int16)
    months = np.array([d.month for d in dates], dtype=np.int16)
    land = filled(ds.variables["CLC_2018_9"]) < 0.5
    land_y, land_x = np.where(land)
    n_land = int(land.sum())
    total_land_km2 = n_land * CELL_KM2
    ny, nx = land.shape

    sy10, sx10 = max(1, int(round(10.0 / DY_KM))), max(1, int(round(10.0 / DX_KM)))
    units = build_block_units(land_y, land_x, sy10, sx10, "10km")
    uid = units["unit_id"]
    n_u = units["n_units"]
    areas = units["area_km2"]
    print(
        f"10km units={n_u} stride=({sy10},{sx10}) mean_area={areas.mean():.2f} km2 "
        f"extent≈({units['mean_extent_km']['y']:.2f}×{units['mean_extent_km']['x']:.2f}) km",
        flush=True,
    )

    # Static + CLC unit features (year-specific POP)
    static_maps = {name: filled(ds.variables[name]) for name in STATIC}
    clc_maps = np.stack([filled(ds.variables[f"CLC_2018_{k}"]) for k in range(10)], axis=0)
    pop_maps = {yr: filled(ds.variables[f"POP_DENS_{yr}"]) for yr in range(2009, 2022)}

    sta_unit_base = np.stack([
        agg_land_to_units(static_maps["DEM"][land], uid, n_u, STATIC_AGG["DEM"]),
        agg_land_to_units(static_maps["SLOPE"][land], uid, n_u, STATIC_AGG["SLOPE"]),
        agg_land_to_units(static_maps["ROAD_DISTANCE"][land], uid, n_u, STATIC_AGG["ROAD_DISTANCE"]),
        agg_land_to_units(static_maps["WATERWAY_DISTANCE"][land], uid, n_u, STATIC_AGG["WATERWAY_DISTANCE"]),
    ], axis=1).astype(np.float32)
    clc_unit = np.stack([
        agg_land_to_units(clc_maps[k][land], uid, n_u, "mean") for k in range(10)
    ], axis=1).astype(np.float32)
    pop_unit = {
        yr: agg_land_to_units(pop_maps[yr][land], uid, n_u, STATIC_AGG["POP_DENS"])
        for yr in range(2009, 2022)
    }

    agg_doc = {
        "spatial_units": {
            "scheme": "same as Experiment 4 ~10 km rectangular blocks",
            "stride_cells": {"y": sy10, "x": sx10},
            "mean_extent_km": units["mean_extent_km"],
            "n_units": n_u,
            "mean_area_km2": float(areas.mean()),
            "cell_km2": CELL_KM2,
        },
        "label": "unit positive on day t if any fire-start burned_areas cell inside unit on day t",
        "temporal": "dynamic predictors only from t-10 ... t-1 (HISTORY buffer includes earlier days for continuity)",
        "dynamic_spatial_agg": DYNAMIC_AGG,
        "static_spatial_agg": STATIC_AGG,
        "clc_spatial_agg": "mean of CLC_2018_{0..9} fractions over land cells in unit",
        "fwi_baseline_spatial_agg": "max of previous-day (t-1) FWI over land cells in unit, then min-max to [0,1] on train",
        "xgb_feature_construction": (
            "For each unit-day: dyn10 shape (10 days × 10 vars) after spatial agg; "
            "features = [mean_over_10_days, last_day (t-1), static(DEM,SLOPE,road_min,water_min,pop_mean), clc10]"
        ),
        "train_years": (
            f"2009-2018 May-Sep: ALL positive unit-days + {TRAIN_NEG_POS_RATIO}:1 random negatives"
        ),
        "val_year": "2019 May-Sep FULL unit-day grid",
        "test_years": "2020 and 2021 May-Sep FULL unit-day grid",
        "note": "Training subsamples negatives for memory; evaluation is full-grid and fair vs Exp4",
    }
    meta_path.write_text(json.dumps(agg_doc, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}", flush=True)

    if feat_path.exists():
        print(f"Loading cached features {feat_path}", flush=True)
        z = np.load(feat_path)
        data = {k: z[k] for k in z.files}
    else:
        rng = np.random.default_rng(SEED)
        dyn_vars = [ds.variables[name] for name in DYNAMIC]
        fwi_var = ds.variables["fwi"]
        ba_var = ds.variables["burned_areas"]
        buf_dyn = np.full((HISTORY, len(DYNAMIC), n_u), np.nan, np.float32)
        buf_fwi = np.full((HISTORY, n_u), np.nan, np.float32)

        day_mask = (
            (years >= 2009) & (years <= 2021)
            & np.isin(months, list(FIRE_MONTHS))
            & (np.arange(len(years)) >= HISTORY)
        )
        days = np.flatnonzero(day_mask).astype(np.int32)
        print(f"Building unit-day features for {len(days)} summer days...", flush=True)

        splits = {
            k: {"dyn": [], "sta": [], "clc": [], "fwi": [], "y": [], "uid": [], "area": [], "t": []}
            for k in ("train", "val", "test2020", "test2021")
        }

        t0 = int(days[0])
        for k in range(HISTORY):
            tt = t0 - HISTORY + k
            for v, var in enumerate(dyn_vars):
                buf_dyn[k, v] = agg_land_to_units(
                    filled(var, tt)[land], uid, n_u, DYNAMIC_AGG[DYNAMIC[v]]
                )
            buf_fwi[k] = agg_land_to_units(filled(fwi_var, tt)[land], uid, n_u, "max")
        head = 0
        cur_t = t0

        for di, t in enumerate(days):
            t = int(t)
            while cur_t < t:
                for v, var in enumerate(dyn_vars):
                    buf_dyn[head, v] = agg_land_to_units(
                        filled(var, cur_t)[land], uid, n_u, DYNAMIC_AGG[DYNAMIC[v]]
                    )
                buf_fwi[head] = agg_land_to_units(filled(fwi_var, cur_t)[land], uid, n_u, "max")
                head = (head + 1) % HISTORY
                cur_t += 1

            # Buffer holds t-HISTORY ... t-1 when cur_t == t (leakage-safe).
            dyn_u = np.empty((n_u, HISTORY, len(DYNAMIC)), np.float32)
            fwi_u = np.empty((n_u, HISTORY), np.float32)
            for k in range(HISTORY):
                idx = (head + k) % HISTORY
                dyn_u[:, k, :] = buf_dyn[idx].T
                fwi_u[:, k] = buf_fwi[idx]

            ba = filled(ba_var, t)
            y_cell = (ba[land] > 0).astype(np.float32)
            y_u = np.zeros(n_u, dtype=np.float32)
            np.maximum.at(y_u, uid, y_cell)

            yr = int(years[t])
            sp = split_name(yr)
            if sp is None:
                continue
            s0, e0 = window_slice()
            dyn10 = dyn_u[:, s0:e0, :]
            fwi_last = fwi_u[:, s0:e0][:, -1]
            sta = np.concatenate([sta_unit_base, pop_unit[yr][:, None]], axis=1)

            if sp == "train":
                pos_idx = np.flatnonzero(y_u > 0)
                neg_idx = np.flatnonzero(y_u == 0)
                n_pos = len(pos_idx)
                n_neg_keep = min(len(neg_idx), n_pos * TRAIN_NEG_POS_RATIO)
                if n_pos == 0:
                    # still sample a few negatives so empty fire days contribute context
                    n_neg_keep = min(len(neg_idx), 50)
                    keep = rng.choice(neg_idx, size=n_neg_keep, replace=False) if n_neg_keep else np.array([], dtype=int)
                else:
                    neg_keep = rng.choice(neg_idx, size=n_neg_keep, replace=False) if n_neg_keep else np.array([], dtype=int)
                    keep = np.concatenate([pos_idx, neg_keep])
                if keep.size == 0:
                    continue
            else:
                keep = np.arange(n_u, dtype=np.int32)

            splits[sp]["dyn"].append(dyn10[keep].astype(np.float16))
            splits[sp]["sta"].append(sta[keep].astype(np.float32))
            splits[sp]["clc"].append(clc_unit[keep].astype(np.float32))
            splits[sp]["fwi"].append(fwi_last[keep].astype(np.float32))
            splits[sp]["y"].append(y_u[keep].astype(np.float32))
            splits[sp]["uid"].append(keep.astype(np.int32))
            splits[sp]["area"].append(areas[keep].astype(np.float32))
            splits[sp]["t"].append(np.full(len(keep), t, dtype=np.int32))

            if (di + 1) % 20 == 0 or di == 0:
                print(
                    f"  day {di+1}/{len(days)} t={t} year={yr} pos_units={int(y_u.sum())} "
                    f"kept={len(keep)}",
                    flush=True,
                )

        data = {}
        for sp, bag in splits.items():
            if not bag["y"]:
                continue
            data[f"{sp}__dyn"] = np.concatenate(bag["dyn"], axis=0)
            data[f"{sp}__sta"] = np.concatenate(bag["sta"], axis=0)
            data[f"{sp}__clc"] = np.concatenate(bag["clc"], axis=0)
            data[f"{sp}__fwi"] = np.concatenate(bag["fwi"], axis=0)
            data[f"{sp}__y"] = np.concatenate(bag["y"], axis=0)
            data[f"{sp}__uid"] = np.concatenate(bag["uid"], axis=0)
            data[f"{sp}__area"] = np.concatenate(bag["area"], axis=0)
            data[f"{sp}__t"] = np.concatenate(bag["t"], axis=0)
            print(
                f"  {sp}: n={len(data[f'{sp}__y'])} pos={int(data[f'{sp}__y'].sum())} "
                f"base={float(data[f'{sp}__y'].mean()):.6f}",
                flush=True,
            )
        data["n_units"] = np.array([n_u], dtype=np.int32)
        data["area_km2"] = areas.astype(np.float32)
        data["stride"] = np.array([sy10, sx10], dtype=np.int32)
        data["train_neg_pos_ratio"] = np.array([TRAIN_NEG_POS_RATIO], dtype=np.int32)
        np.savez_compressed(feat_path, **data)
        print(f"Saved {feat_path}", flush=True)

    ds.close()

    # ---- Train direct 10 km XGBoost ----
    print("Training direct 10 km XGBoost...", flush=True)
    s0, e0 = window_slice()
    # dyn already stored as 10-day window in cache; if rebuilt above, same

    def prep(split, scaler_dyn=None, scaler_sta=None):
        dyn = np.asarray(data[f"{split}__dyn"], dtype=np.float32)
        dyn = fill_time_nans(dyn)
        if dyn.shape[1] == HISTORY:
            dyn = dyn[:, s0:e0, :]
        sta = np.nan_to_num(data[f"{split}__sta"], nan=-1.0).astype(np.float32)
        clc = np.nan_to_num(data[f"{split}__clc"], nan=0.0).astype(np.float32)
        fwi = np.nan_to_num(data[f"{split}__fwi"], nan=-1.0).astype(np.float32)
        y = data[f"{split}__y"].astype(np.float32)
        if scaler_dyn is None:
            scaler_dyn = MinMaxScaler().fit(dyn.reshape(-1, dyn.shape[-1]))
            scaler_sta = MinMaxScaler().fit(sta)
        dyn_n = scaler_dyn.transform(dyn.reshape(-1, dyn.shape[-1])).reshape(dyn.shape).astype(np.float32)
        sta_n = scaler_sta.transform(sta).astype(np.float32)
        x = xgb_features(dyn_n, sta_n, clc)
        return x, y, fwi, scaler_dyn, scaler_sta

    x_tr, y_tr, fwi_tr, scaler_dyn, scaler_sta = prep("train")
    x_va, y_va, fwi_va, _, _ = prep("val", scaler_dyn, scaler_sta)
    fwi_min, fwi_max = np.percentile(fwi_tr, [1, 99])
    # Train already uses TRAIN_NEG_POS_RATIO negatives; no extra scale_pos_weight.
    print(f"  train n={len(y_tr)} pos={int(y_tr.sum())} neg={int((y_tr==0).sum())}", flush=True)

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=4,
        random_state=SEED,
        early_stopping_rounds=40,
    )
    xgb.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    print(f"  best_iteration={getattr(xgb, 'best_iteration', None)}", flush=True)

    art = {
        "xgb": xgb,
        "scaler_dyn": scaler_dyn,
        "scaler_sta": scaler_sta,
        "fwi_min": float(fwi_min),
        "fwi_max": float(fwi_max),
        "train_neg_pos_ratio": TRAIN_NEG_POS_RATIO,
        "feature_agg": json.loads(meta_path.read_text(encoding="utf-8")),
    }
    joblib.dump(art, OUT / "xgb_direct_10km.joblib")

    def fwi_p(arr):
        return np.clip((arr - fwi_min) / max(fwi_max - fwi_min, 1e-6), 0, 1)

    p_va = xgb.predict_proba(x_va)[:, 1]
    thr_xgb, f1_va = best_f1_threshold(y_va, p_va)
    thr_fwi, _ = best_f1_threshold(y_va, fwi_p(fwi_va))
    print(f"  2019-tuned thr XGB={thr_xgb:.3f} (F1={f1_va:.4f}) FWI={thr_fwi:.3f}", flush=True)

    n_units = int(data["n_units"][0]) if "n_units" in data else n_u
    area_lookup = data["area_km2"] if "area_km2" in data else areas
    land_km2_eval = float(np.sum(area_lookup))

    results = {
        "protocol": {
            "experiment": 5,
            "model": "XGBoost trained directly on ~10 km unit-days",
            "baselines": [
                "A: Exp1/2 1km XGB easy + Exp4 10km max pooling",
                "B: Exp3 hard-trained 1km XGB + Exp4 10km max pooling",
                "C: FWI aggregated to 10km (Exp4 FWI__max; also direct unit-max FWI for sanity)",
            ],
            "spatial_units": art["feature_agg"]["spatial_units"],
            "feature_aggregation": {
                "dynamic": DYNAMIC_AGG,
                "static": STATIC_AGG,
                "clc": "mean fractions",
                "fwi": "max of t-1",
            },
            "years": (
                f"train 2009-2018 May-Sep (all pos + {TRAIN_NEG_POS_RATIO}:1 neg); "
                "val 2019 / test 2020-2021 May-Sep FULL grids"
            ),
            "threshold": "F1-max on 2019 unit-days only",
            "total_land_km2": land_km2_eval,
            "n_units": n_units,
        },
        "models": {},
        "comparison": {},
        "verdict_notes": {},
    }

    # Direct XGB + direct FWI on same unit-days
    for split, yr_key, n_days_hint in (
        ("val", "val", None),
        ("test2020", "2020", None),
        ("test2021", "2021", None),
    ):
        x, y, fwi, _, _ = prep(split, scaler_dyn, scaler_sta)
        px = xgb.predict_proba(x)[:, 1]
        pf = fwi_p(fwi)
        # n_days from unique t
        n_days = int(len(np.unique(data[f"{split}__t"])))
        unit_areas = data[f"{split}__area"]
        block_x = pack_eval(y, px, thr_xgb, unit_areas, n_days, land_km2_eval)
        block_f = pack_eval(y, pf, thr_fwi, unit_areas, n_days, land_km2_eval)
        block_x["n_days"] = n_days
        block_f["n_days"] = n_days
        block_x["n_units"] = n_units
        block_f["n_units"] = n_units
        block_x["n_pos_unit_days"] = int(y.sum())
        block_f["n_pos_unit_days"] = int(y.sum())
        if "XGBoost_direct_10km" not in results["models"]:
            results["models"]["XGBoost_direct_10km"] = {"val_threshold": thr_xgb}
            results["models"]["FWI_direct_10km_max"] = {"val_threshold": thr_fwi, "note": "unit max of t-1 FWI"}
        results["models"]["XGBoost_direct_10km"][yr_key] = block_x
        results["models"]["FWI_direct_10km_max"][yr_key] = block_f
        print(
            f"  {split} directXGB PR={block_x['pr_auc']:.5f} ROC={block_x['roc_auc']:.3f} "
            f"P={block_x['at_val_tuned']['precision']:.5f} R={block_x['at_val_tuned']['recall']:.3f} "
            f"F1={block_x['at_val_tuned']['f1']:.5f}",
            flush=True,
        )
        print(
            f"  {split} directFWI PR={block_f['pr_auc']:.5f} ROC={block_f['roc_auc']:.3f} "
            f"P={block_f['at_val_tuned']['precision']:.5f} R={block_f['at_val_tuned']['recall']:.3f}",
            flush=True,
        )

    # Baselines from Exp4 (same units / dates / labels / max pooling)
    print("Loading Exp4 10 km baselines...", flush=True)
    exp4 = json.loads(EXP4_METRICS.read_text(encoding="utf-8"))
    results["models"]["A_XGB_easy_1km_then_10km_max"] = extract_slice_from_exp4(exp4, "XGBoost_easy__max")
    results["models"]["B_XGB_hard_1km_then_10km_max"] = extract_slice_from_exp4(exp4, "XGBoost_hard__max")
    results["models"]["C_FWI_1km_then_10km_max"] = extract_slice_from_exp4(exp4, "FWI__max")

    # Fairness check: same n / n_pos as Exp4
    for yr in ("2020", "2021"):
        d = results["models"]["XGBoost_direct_10km"][yr]
        a = results["models"]["A_XGB_easy_1km_then_10km_max"][yr]
        results["comparison"][yr] = {
            "n_match": d["n"] == a["n"],
            "n_pos_match": d["n_pos"] == a["n_pos"],
            "direct_n": d["n"],
            "exp4_n": a["n"],
            "direct_n_pos": d["n_pos"],
            "exp4_n_pos": a["n_pos"],
            "pr_auc": {
                "direct_xgb": d["pr_auc"],
                "A_easy_pool": a["pr_auc"],
                "B_hard_pool": results["models"]["B_XGB_hard_1km_then_10km_max"][yr]["pr_auc"],
                "C_fwi_pool": results["models"]["C_FWI_1km_then_10km_max"][yr]["pr_auc"],
                "direct_fwi": results["models"]["FWI_direct_10km_max"][yr]["pr_auc"],
            },
            "delta_pr_auc_direct_minus_A": d["pr_auc"] - a["pr_auc"],
            "delta_pr_auc_direct_minus_C": d["pr_auc"] - results["models"]["C_FWI_1km_then_10km_max"][yr]["pr_auc"],
            "precision_at_val_thr": {
                "direct_xgb": d["at_val_tuned"]["precision"],
                "direct_xgb_recall": d["at_val_tuned"]["recall"],
                "A_easy_pool": a["at_val_tuned"]["precision"],
                "A_easy_pool_recall": a["at_val_tuned"]["recall"],
                "C_fwi_pool": results["models"]["C_FWI_1km_then_10km_max"][yr]["at_val_tuned"]["precision"],
                "C_fwi_pool_recall": results["models"]["C_FWI_1km_then_10km_max"][yr]["at_val_tuned"]["recall"],
            },
            "precision_at_recall": {
                "direct_xgb": d["precision_at_recall"],
                "A_easy_pool": a["precision_at_recall"],
                "C_fwi_pool": results["models"]["C_FWI_1km_then_10km_max"][yr]["precision_at_recall"],
            },
        }
        print(
            f"  compare {yr}: directPR={d['pr_auc']:.5f} A={a['pr_auc']:.5f} "
            f"dPR={d['pr_auc']-a['pr_auc']:+.5f} n_match={d['n']==a['n']} pos_match={d['n_pos']==a['n_pos']}",
            flush=True,
        )

    out_json = OUT / "metrics.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
