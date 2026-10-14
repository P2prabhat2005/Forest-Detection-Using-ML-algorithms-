"""Experiment 6: hard-negative XGBoost at direct ~10 km unit-day scale.

Same units/labels/lags/splits as Exp5. Does not modify Exp1–Exp5 outputs
or dataset_greece.nc. XGBoost only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
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
EXP5_METRICS = ROOT / "results" / "firecube_direct_10km" / "metrics.json"
OUT = ROOT / "results" / "firecube_hardneg_10km"
CONFIG_PATH = OUT / "experiment_config.json"

HISTORY = 16
FIRE_MONTHS = {5, 6, 7, 8, 9}
SEED = 1234
NEG_POS_RATIO = 2
FIRE_FRAC = 0.7
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
RECALL_TARGETS = (0.20, 0.30, 0.40, 0.50)
CELL_KM2 = 0.9066669762359021 * 0.7144633272136693
DY_KM = 0.9066669762359021
DX_KM = 0.7144633272136693

DYNAMIC = [
    "max_t2m", "max_d2m", "min_rh", "max_tp", "max_sp",
    "max_wind_speed", "ndvi", "lst_day", "lst_night", "sminx",
]
DYNAMIC_AGG = {
    "max_t2m": "max", "max_d2m": "max", "min_rh": "min", "max_tp": "mean",
    "max_sp": "mean", "max_wind_speed": "max", "ndvi": "mean",
    "lst_day": "max", "lst_night": "max", "sminx": "mean",
}
STATIC = ["DEM", "SLOPE", "ROAD_DISTANCE", "WATERWAY_DISTANCE"]
STATIC_AGG = {
    "DEM": "mean", "SLOPE": "mean",
    "ROAD_DISTANCE": "min", "WATERWAY_DISTANCE": "min", "POP_DENS": "mean",
}


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
        "unit_id": inv.astype(np.int32),
        "n_units": int(n_units),
        "area_km2": areas,
        "mean_extent_km": {"y": float(stride_y * DY_KM), "x": float(stride_x * DX_KM)},
        "stride_cells": {"y": int(stride_y), "x": int(stride_x)},
        "n_cells": counts.astype(np.int32),
    }


def agg_land_to_units(values, unit_id, n_units, how: str) -> np.ndarray:
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


def sha256_file(path: Path, chunk=1024 * 1024) -> str:
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"atomic_savez failed verification: {path}")
    print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def atomic_joblib(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    joblib.dump(obj, tmp)
    os.replace(tmp, path)
    # round-trip keys check
    loaded = joblib.load(path)
    if "xgb" not in loaded:
        raise RuntimeError(f"joblib missing xgb after write: {path}")
    print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def write_manifest(out_dir: Path, files: list[Path]):
    man = {"files": {}}
    for p in files:
        if p.exists():
            man["files"][p.name] = {
                "path": str(p),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        else:
            man["files"][p.name] = {"path": str(p), "missing": True}
    dest = out_dir / "artifacts_manifest.json"
    dest.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"  wrote {dest}", flush=True)
    return man


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config (write before run): {CONFIG_PATH}")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    print(f"Exp6 config loaded from {CONFIG_PATH}", flush=True)

    feat_path = OUT / "unit_day_features.npz"
    model_path = OUT / "xgb_hardneg_10km.joblib"
    meta_path = OUT / "feature_aggregation.json"
    rng = np.random.default_rng(SEED)

    print("Opening cube...", flush=True)
    ds = nc.Dataset(NC_PATH, "r")
    dates = nc.num2date(ds.variables["time"][:], ds.variables["time"].units, only_use_cftime_datetimes=False)
    years = np.array([d.year for d in dates], dtype=np.int16)
    months = np.array([d.month for d in dates], dtype=np.int16)
    land = filled(ds.variables["CLC_2018_9"]) < 0.5
    land_y, land_x = np.where(land)
    n_land = int(land.sum())
    ny, nx = land.shape

    sy10, sx10 = max(1, int(round(10.0 / DY_KM))), max(1, int(round(10.0 / DX_KM)))
    units = build_block_units(land_y, land_x, sy10, sx10, "10km")
    uid = units["unit_id"]
    n_u = units["n_units"]
    areas = units["area_km2"]
    print(
        f"10km units={n_u} stride=({sy10},{sx10}) mean_area={areas.mean():.2f} km2",
        flush=True,
    )

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
    unit_dom_clc = np.argmax(clc_unit, axis=1).astype(np.int16)
    pop_unit = {
        yr: agg_land_to_units(pop_maps[yr][land], uid, n_u, STATIC_AGG["POP_DENS"])
        for yr in range(2009, 2022)
    }

    agg_doc = {
        "experiment": 6,
        "spatial_units": {
            "scheme": "same as Experiment 4/5 ~10 km rectangular blocks",
            "stride_cells": {"y": sy10, "x": sx10},
            "mean_extent_km": units["mean_extent_km"],
            "n_units": n_u,
            "mean_area_km2": float(areas.mean()),
            "cell_km2": CELL_KM2,
        },
        "label": cfg["label"],
        "temporal": cfg["temporal_protocol"],
        "dynamic_spatial_agg": DYNAMIC_AGG,
        "static_spatial_agg": STATIC_AGG,
        "clc_spatial_agg": cfg["feature_aggregation"]["clc"],
        "hard_negative_protocol": cfg["training_sample"],
        "xgb_feature_construction": cfg["xgb_features"],
    }
    meta_path.write_text(json.dumps(agg_doc, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}", flush=True)

    if feat_path.exists():
        print(f"Loading cached features {feat_path}", flush=True)
        z = np.load(feat_path)
        data = {k: z[k] for k in z.files}
    else:
        dyn_vars = [ds.variables[name] for name in DYNAMIC]
        fwi_var = ds.variables["fwi"]
        ba_var = ds.variables["burned_areas"]
        nof = filled(ds.variables["number_of_fires"]).astype(np.int16)

        # ---- Pass 1: train hard-neg plan (labels + FWI day stats only) ----
        print("Pass 1: scanning summer days for hard-neg plan...", flush=True)
        day_mask = (
            (years >= 2009) & (years <= 2021)
            & np.isin(months, list(FIRE_MONTHS))
            & (np.arange(len(years)) >= HISTORY)
        )
        days = np.flatnonzero(day_mask).astype(np.int32)

        # Precompute train-summer domain mean FWI on t-1 for high-FWI day definition
        # Need FWI at t-1 for each candidate day — scan sequentially with buffer
        buf_fwi = np.full((HISTORY, n_u), np.nan, np.float32)
        t0 = int(days[0])
        for k in range(HISTORY):
            tt = t0 - HISTORY + k
            buf_fwi[k] = agg_land_to_units(filled(fwi_var, tt)[land], uid, n_u, "max")
        head = 0
        cur_t = t0

        train_pos: list[tuple[int, int]] = []  # (t, unit)
        train_day_meta = {}  # t -> dict
        train_fwi_means = []

        for di, t in enumerate(days):
            t = int(t)
            while cur_t < t:
                buf_fwi[head] = agg_land_to_units(filled(fwi_var, cur_t)[land], uid, n_u, "max")
                head = (head + 1) % HISTORY
                cur_t += 1
            # unit FWI for t-1 = last day in window
            fwi_hist = np.empty((n_u, HISTORY), np.float32)
            for k in range(HISTORY):
                fwi_hist[:, k] = buf_fwi[(head + k) % HISTORY]
            s0, e0 = window_slice()
            fwi_last = fwi_hist[:, s0:e0][:, -1]
            domain_fwi = float(np.nanmean(fwi_last))

            ba = filled(ba_var, t)
            y_cell = (ba[land] > 0).astype(np.float32)
            y_u = np.zeros(n_u, dtype=np.float32)
            np.maximum.at(y_u, uid, y_cell)
            n_pos_u = int(y_u.sum())
            yr = int(years[t])
            sp = split_name(yr)
            is_fire_day = (nof[t] > 0) or (n_pos_u > 0)
            if sp == "train":
                train_fwi_means.append(domain_fwi)
                for u in np.flatnonzero(y_u > 0):
                    train_pos.append((t, int(u)))
                train_day_meta[t] = {
                    "is_fire_day": bool(is_fire_day),
                    "domain_fwi": domain_fwi,
                    "fwi_last": fwi_last.astype(np.float32),
                    "y_u": y_u.astype(np.float32),
                    "n_pos": n_pos_u,
                }
            if (di + 1) % 100 == 0 or di == 0:
                print(f"  scan {di+1}/{len(days)} t={t} yr={yr} pos_units={n_pos_u}", flush=True)

        fwi_day_p75 = float(np.nanpercentile(np.asarray(train_fwi_means, dtype=np.float64), 75))
        n_pos_train = len(train_pos)
        n_neg_need = n_pos_train * NEG_POS_RATIO
        n_from_fire = int(round(FIRE_FRAC * n_neg_need))
        n_from_high = n_neg_need - n_from_fire
        print(
            f"  train pos={n_pos_train} need_neg={n_neg_need} "
            f"(fire_elsewhere={n_from_fire}, high_fwi={n_from_high}) fwi_p75={fwi_day_p75:.3f}",
            flush=True,
        )

        # CLC counts from positives
        pos_clc = [int(unit_dom_clc[u]) for _, u in train_pos]
        clc_classes, clc_counts = np.unique(pos_clc, return_counts=True)
        clc_quota = {int(c): int(cnt) * NEG_POS_RATIO for c, cnt in zip(clc_classes, clc_counts)}

        fire_days = [t for t, m in train_day_meta.items() if m["is_fire_day"]]
        high_days = [
            t for t, m in train_day_meta.items()
            if (not m["is_fire_day"]) and m["domain_fwi"] >= fwi_day_p75
        ]
        print(f"  fire_elsewhere_days={len(fire_days)} high_fwi_zero_fire_days={len(high_days)}", flush=True)

        def sample_hard(days_list, n_want, tag, clc_quota_left):
            picked = []
            if n_want <= 0 or not days_list:
                return picked
            # allocate by remaining CLC quotas proportionally to available
            guard = 0
            while len(picked) < n_want and guard < n_want * 50:
                guard += 1
                t = int(rng.choice(days_list))
                meta = train_day_meta[t]
                y_u = meta["y_u"]
                fwi_last = meta["fwi_last"]
                unburned = np.flatnonzero(y_u == 0)
                if unburned.size == 0:
                    continue
                if tag == "high_fwi_day":
                    med = float(np.nanmedian(fwi_last[unburned]))
                    pool = unburned[np.isfinite(fwi_last[unburned]) & (fwi_last[unburned] >= med)]
                    if pool.size == 0:
                        pool = unburned
                else:
                    pool = unburned
                # prefer CLC with remaining quota
                active = [c for c, q in clc_quota_left.items() if q > 0]
                if active:
                    c = int(rng.choice(active))
                    pool_c = pool[unit_dom_clc[pool] == c]
                    if pool_c.size == 0:
                        pool_c = pool
                else:
                    c = None
                    pool_c = pool
                u = int(rng.choice(pool_c))
                picked.append((t, u, tag))
                if c is not None and clc_quota_left.get(c, 0) > 0:
                    clc_quota_left[c] -= 1
            return picked

        clc_left = dict(clc_quota)
        neg_fire = sample_hard(fire_days, n_from_fire, "fire_elsewhere", clc_left)
        neg_high = sample_hard(high_days, n_from_high, "high_fwi_day", clc_left)
        train_negs = neg_fire + neg_high
        if len(train_negs) < n_neg_need:
            extra_days = list(train_day_meta.keys())
            train_negs.extend(
                sample_hard(extra_days, n_neg_need - len(train_negs), "season_fallback", clc_left)
            )
        train_negs = train_negs[:n_neg_need]
        train_keep = {(t, u) for t, u in train_pos}
        train_keep |= {(t, u) for t, u, _ in train_negs}
        neg_tag_count = {}
        for _, _, tag in train_negs:
            neg_tag_count[tag] = neg_tag_count.get(tag, 0) + 1
        print(f"  hard-neg tags: {neg_tag_count}", flush=True)

        # ---- Pass 2: extract features ----
        print("Pass 2: extracting features...", flush=True)
        buf_dyn = np.full((HISTORY, len(DYNAMIC), n_u), np.nan, np.float32)
        buf_fwi = np.full((HISTORY, n_u), np.nan, np.float32)
        for k in range(HISTORY):
            tt = t0 - HISTORY + k
            for v, var in enumerate(dyn_vars):
                buf_dyn[k, v] = agg_land_to_units(
                    filled(var, tt)[land], uid, n_u, DYNAMIC_AGG[DYNAMIC[v]]
                )
            buf_fwi[k] = agg_land_to_units(filled(fwi_var, tt)[land], uid, n_u, "max")
        head = 0
        cur_t = t0

        splits = {
            k: {"dyn": [], "sta": [], "clc": [], "fwi": [], "y": [], "uid": [], "area": [], "t": [], "tag": []}
            for k in ("train", "val", "test2020", "test2021")
        }

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
                keep = np.array([u for u in range(n_u) if (t, u) in train_keep], dtype=np.int32)
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
            if (di + 1) % 50 == 0 or di == 0:
                print(
                    f"  extract {di+1}/{len(days)} t={t} yr={yr} kept={len(keep)} pos={int(y_u.sum())}",
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
        data["neg_pos_ratio"] = np.array([NEG_POS_RATIO], dtype=np.int32)
        data["fire_frac"] = np.array([FIRE_FRAC], dtype=np.float32)
        data["fwi_day_p75"] = np.array([fwi_day_p75], dtype=np.float32)
        atomic_savez(feat_path, **data)

    ds.close()

    # ---- Train ----
    print("Training hard-neg 10 km XGBoost...", flush=True)
    s0, e0 = window_slice()

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
        "neg_pos_ratio": NEG_POS_RATIO,
        "fire_frac": FIRE_FRAC,
        "feature_agg": json.loads(meta_path.read_text(encoding="utf-8")),
        "experiment": 6,
    }
    atomic_joblib(model_path, art)
    write_manifest(OUT, [feat_path, model_path, meta_path, CONFIG_PATH])

    def fwi_p(arr):
        return np.clip((arr - fwi_min) / max(fwi_max - fwi_min, 1e-6), 0, 1)

    p_va = xgb.predict_proba(x_va)[:, 1]
    thr_xgb, f1_va = best_f1_threshold(y_va, p_va)
    print(f"  2019-tuned thr XGB={thr_xgb:.3f} (F1={f1_va:.4f})", flush=True)

    n_units = int(data["n_units"][0])
    area_lookup = data["area_km2"]
    land_km2_eval = float(np.sum(area_lookup))

    results = {
        "protocol": {
            "experiment": 6,
            "model": "XGBoost hard-neg trained on ~10 km unit-days",
            "baseline": "Exp5 XGBoost_direct_10km",
            "spatial_units": art["feature_agg"]["spatial_units"],
            "feature_aggregation": {
                "dynamic": DYNAMIC_AGG,
                "static": STATIC_AGG,
                "clc": "mean fractions + dominant-class matching for hard negs",
                "fwi": "max of t-1",
            },
            "hard_negatives": {
                "neg_pos_ratio": NEG_POS_RATIO,
                "fire_elsewhere_frac": FIRE_FRAC,
                "high_fwi_frac": 1.0 - FIRE_FRAC,
            },
            "years": "train 2009-2018 May-Sep hard-neg; val/test FULL grids 2019/2020/2021",
            "threshold": "F1-max on 2019 FULL unit-days only",
            "total_land_km2": land_km2_eval,
            "n_units": n_units,
        },
        "models": {
            "XGBoost_hardneg_10km": {"val_threshold": thr_xgb},
        },
        "comparison_vs_exp5": {},
    }

    for split, yr_key in (("val", "val"), ("test2020", "2020"), ("test2021", "2021")):
        x, y, fwi, _, _ = prep(split, scaler_dyn, scaler_sta)
        px = xgb.predict_proba(x)[:, 1]
        n_days = int(len(np.unique(data[f"{split}__t"])))
        unit_areas = data[f"{split}__area"]
        block = pack_eval(y, px, thr_xgb, unit_areas, n_days, land_km2_eval)
        block["n_days"] = n_days
        block["n_units"] = n_units
        block["n_pos_unit_days"] = int(y.sum())
        results["models"]["XGBoost_hardneg_10km"][yr_key] = block
        print(
            f"  {split} hardXGB PR={block['pr_auc']:.5f} ROC={block['roc_auc']:.3f} "
            f"P={block['at_val_tuned']['precision']:.5f} R={block['at_val_tuned']['recall']:.3f} "
            f"F1={block['at_val_tuned']['f1']:.5f}",
            flush=True,
        )

    print("Comparing to Exp5 direct 10 km...", flush=True)
    exp5 = json.loads(EXP5_METRICS.read_text(encoding="utf-8"))
    e5 = exp5["models"]["XGBoost_direct_10km"]
    for yr in ("2020", "2021"):
        h = results["models"]["XGBoost_hardneg_10km"][yr]
        d = e5[yr]
        results["comparison_vs_exp5"][yr] = {
            "n_match": h["n"] == d["n"],
            "n_pos_match": h["n_pos"] == d["n_pos"],
            "hard_n": h["n"],
            "exp5_n": d["n"],
            "hard_n_pos": h["n_pos"],
            "exp5_n_pos": d["n_pos"],
            "pr_auc": {"hardneg_10km": h["pr_auc"], "exp5_direct": d["pr_auc"]},
            "roc_auc": {"hardneg_10km": h["roc_auc"], "exp5_direct": d["roc_auc"]},
            "brier": {"hardneg_10km": h["brier"], "exp5_direct": d["brier"]},
            "delta_pr_auc_hard_minus_exp5": h["pr_auc"] - d["pr_auc"],
            "precision_at_recall": {
                "hardneg_10km": h["precision_at_recall"],
                "exp5_direct": d["precision_at_recall"],
            },
            "at_val_tuned": {
                "hardneg_10km": h["at_val_tuned"],
                "exp5_direct": d["at_val_tuned"],
            },
        }
        print(
            f"  {yr}: hardPR={h['pr_auc']:.5f} exp5PR={d['pr_auc']:.5f} "
            f"dPR={h['pr_auc']-d['pr_auc']:+.5f} n_match={h['n']==d['n']}",
            flush=True,
        )

    out_json = OUT / "metrics.json"
    tmp_json = OUT / "metrics.json.tmp"
    tmp_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    os.replace(tmp_json, out_json)
    write_manifest(
        OUT,
        [feat_path, model_path, meta_path, CONFIG_PATH, out_json],
    )
    # Final existence gate
    for p in (feat_path, model_path, out_json):
        if not p.exists():
            raise RuntimeError(f"Required artifact missing after Exp6: {p}")
    print(f"Wrote {out_json}", flush=True)
    print("Exp6 complete.", flush=True)


if __name__ == "__main__":
    main()
