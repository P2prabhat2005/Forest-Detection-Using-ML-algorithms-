"""Experiment 4: spatial aggregation of existing 1 km XGBoost + FWI scores.

No new model training. Thresholds tuned on 2019 May-Sep only.
Does not modify dataset_greece.nc.
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

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
NC_PATH = ROOT / "data" / "raw" / "firecube" / "dataset_greece.nc"
EASY_MODEL = ROOT / "results" / "firecube_hardneg" / "xgb_easytrain.joblib"
HARD_MODEL = ROOT / "results" / "firecube_hardneg_train" / "xgb_hardtrain.joblib"
NUTS_PATH = ROOT / "data" / "external" / "nuts" / "NUTS_RG_10M_2021_4326_LEVL_3.geojson"
OUT = ROOT / "results" / "firecube_spatial_agg"

HISTORY = 16
FIRE_MONTHS = {5, 6, 7, 8, 9}
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
RECALL_TARGETS = (0.20, 0.30, 0.40, 0.50)
CELL_KM2 = 0.9066669762359021 * 0.7144633272136693
DY_KM = 0.9066669762359021
DX_KM = 0.7144633272136693

DYNAMIC = [
    "max_t2m", "max_d2m", "min_rh", "max_tp", "max_sp",
    "max_wind_speed", "ndvi", "lst_day", "lst_night", "sminx",
]
STATIC = ["DEM", "SLOPE", "ROAD_DISTANCE", "WATERWAY_DISTANCE"]


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
        # FA units per 1000 km² per day (count density, Exp2/3 analog)
        out["fa_units_per_1000km2_per_day"] = float(fp / n_days / total_land_km2 * 1000.0)
        # Area falsely alarmed (km²) per day
        fa_area = float(np.asarray(unit_area_km2)[(y_hat == 1) & (np.asarray(y_true) == 0)].sum()) if fp else 0.0
        out["fa_area_km2_per_day"] = fa_area / n_days
        out["fa_area_fraction_per_day"] = (fa_area / n_days) / total_land_km2
        # Comparable to Exp2 cell metric: FA land-km2 equivalent / 1000km2 / day * (1/cell)?
        # Report FA cells-equivalent: fa_area / CELL_KM2 / n_days / total_land_km2 * 1000
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
            out[str(t)] = {"target_recall": t, "precision": None, "threshold": None,
                           "max_recall": float(rec[-1]) if len(rec) else 0.0}
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


def build_block_units(land_y, land_x, stride_y, stride_x, label):
    """Assign land cells to rectangular blocks; return unit_id per land cell and areas."""
    by = land_y // stride_y
    bx = land_x // stride_x
    # compress ids
    keys = by.astype(np.int64) * (int(land_x.max()) + 1 + stride_x) + bx.astype(np.int64)
    uniq, inv = np.unique(keys, return_inverse=True)
    n_units = len(uniq)
    counts = np.bincount(inv, minlength=n_units)
    areas = counts.astype(np.float64) * CELL_KM2
    mean_dy = stride_y * DY_KM
    mean_dx = stride_x * DX_KM
    return {
        "name": label,
        "kind": "block",
        "unit_id": inv.astype(np.int32),
        "n_units": int(n_units),
        "area_km2": areas,
        "mean_extent_km": {"y": float(mean_dy), "x": float(mean_dx)},
        "stride_cells": {"y": int(stride_y), "x": int(stride_x)},
        "land_mask_for_units": np.ones(len(land_y), dtype=bool),  # all land cells participate
    }


def build_nuts_units(lon, lat, land_y, land_x, nuts_path: Path):
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree

    geo = json.loads(nuts_path.read_text(encoding="utf-8"))
    feats = [f for f in geo["features"] if str(f["properties"].get("CNTR_CODE", "")).startswith("EL")
             or str(f["properties"].get("NUTS_ID", "")).startswith("EL")]
    if not feats:
        # fallback: NUTS_ID starts with EL
        feats = [f for f in geo["features"] if str(f.get("id", "")).startswith("EL")
                 or str(f["properties"].get("NUTS_ID", "")).startswith("EL")]
    polys = []
    names = []
    for f in feats:
        geom = shape(f["geometry"])
        if geom.is_empty:
            continue
        polys.append(geom)
        names.append(f["properties"].get("NUTS_ID", f.get("id")))
    print(f"  NUTS-3 EL polygons: {len(polys)}", flush=True)
    tree = STRtree(polys)
    # map land cells
    unit_raw = np.full(len(land_y), -1, dtype=np.int32)
    # query in batches
    batch = 5000
    for i0 in range(0, len(land_y), batch):
        i1 = min(i0 + batch, len(land_y))
        pts = [Point(float(lon[land_x[i]]), float(lat[land_y[i]])) for i in range(i0, i1)]
        # STRtree.query with predicate 'intersects' returns candidates
        for j, pt in enumerate(pts):
            cands = tree.query(pt)
            # shapely 2 returns ndarray of indices
            hit = -1
            for ci in np.atleast_1d(cands):
                if polys[int(ci)].covers(pt) or polys[int(ci)].intersects(pt):
                    hit = int(ci)
                    break
            unit_raw[i0 + j] = hit
        if (i0 // batch) % 20 == 0:
            print(f"    NUTS assign {i1}/{len(land_y)}", flush=True)
    covered = unit_raw >= 0
    n_covered = int(covered.sum())
    print(f"  land cells in EL NUTS-3: {n_covered}/{len(land_y)} ({100*n_covered/len(land_y):.1f}%)", flush=True)
    if n_covered == 0:
        return None
    # reindex to dense ids over used regions
    used = np.unique(unit_raw[covered])
    remap = {int(u): i for i, u in enumerate(used)}
    unit_id = np.full(len(land_y), -1, dtype=np.int32)
    unit_id[covered] = np.array([remap[int(u)] for u in unit_raw[covered]], dtype=np.int32)
    n_units = len(used)
    counts = np.bincount(unit_id[covered], minlength=n_units)
    areas = counts.astype(np.float64) * CELL_KM2
    id_names = [names[int(u)] for u in used]
    return {
        "name": "nuts3_EL",
        "kind": "nuts3",
        "unit_id": unit_id,
        "n_units": int(n_units),
        "area_km2": areas,
        "unit_names": id_names,
        "land_mask_for_units": covered,
        "note": "Only Greek (EL) NUTS-3 regions; Balkan/Turkey land cells excluded",
    }


def aggregate_day(scores, labels, unit_id, mask, n_units):
    """Return y_unit, score_max, score_q90 for units present in mask."""
    uid = unit_id[mask]
    sc = scores[mask]
    lab = labels[mask]
    y = np.zeros(n_units, dtype=np.float32)
    mx = np.full(n_units, -np.inf, dtype=np.float64)
    # max + positive label
    np.maximum.at(mx, uid, sc)
    np.maximum.at(y, uid, lab)  # any fire-start => 1
    # q90 via sorting groups — use accumulate counts
    order = np.argsort(uid, kind="mergesort")
    uid_s = uid[order]
    sc_s = sc[order]
    q90 = np.full(n_units, np.nan, dtype=np.float64)
    # find group boundaries
    if len(uid_s) == 0:
        return y, np.zeros(n_units), np.zeros(n_units), np.zeros(n_units, dtype=bool)
    changes = np.flatnonzero(np.r_[True, uid_s[1:] != uid_s[:-1], True])
    present = np.zeros(n_units, dtype=bool)
    for a, b in zip(changes[:-1], changes[1:]):
        u = int(uid_s[a])
        present[u] = True
        chunk = sc_s[a:b]
        q90[u] = float(np.quantile(chunk, 0.90))
        if not np.isfinite(mx[u]):
            mx[u] = float(chunk.max())
    # units with no cells today shouldn't happen if mask is static land subset
    present[:] = True  # all units with area>0 always have cells; mask is static
    # for units not in this mask (NUTS uncovered), leave as absent
    # present based on counts
    counts = np.bincount(uid, minlength=n_units)
    present = counts > 0
    mx = np.where(present, mx, np.nan)
    return y, mx.astype(np.float32), q90.astype(np.float32), present


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading models (no retraining)...", flush=True)
    easy = joblib.load(EASY_MODEL)
    hard = joblib.load(HARD_MODEL)
    models = {
        "XGBoost_easy": easy,
        "XGBoost_hard": hard,
    }
    # FWI uses easy train percentiles (same as Exp2) as primary operational baseline
    fwi_min, fwi_max = float(easy["fwi_min"]), float(easy["fwi_max"])

    print("Opening cube...", flush=True)
    ds = nc.Dataset(NC_PATH, "r")
    dates = nc.num2date(ds.variables["time"][:], ds.variables["time"].units, only_use_cftime_datetimes=False)
    years = np.array([d.year for d in dates], dtype=np.int16)
    months = np.array([d.month for d in dates], dtype=np.int16)
    lat = np.asarray(ds.variables["y"][:], dtype=np.float64)
    lon = np.asarray(ds.variables["x"][:], dtype=np.float64)
    land = filled(ds.variables["CLC_2018_9"]) < 0.5
    land_y, land_x = np.where(land)
    n_land = int(land.sum())
    total_land_km2 = n_land * CELL_KM2
    clc_land = np.stack([filled(ds.variables[f"CLC_2018_{k}"]) for k in range(10)], axis=0)[:, land].T.astype(np.float32)
    static_maps = {name: filled(ds.variables[name]) for name in STATIC}
    pop = {yr: filled(ds.variables[f"POP_DENS_{yr}"]) for yr in (2019, 2020, 2021)}
    sta_land = {
        yr: np.stack([
            static_maps["DEM"][land], static_maps["SLOPE"][land],
            static_maps["ROAD_DISTANCE"][land], static_maps["WATERWAY_DISTANCE"][land],
            pop[yr][land],
        ], axis=1).astype(np.float32)
        for yr in (2019, 2020, 2021)
    }

    # ~5 km and ~10 km strides
    sy5, sx5 = max(1, int(round(5.0 / DY_KM))), max(1, int(round(5.0 / DX_KM)))
    sy10, sx10 = max(1, int(round(10.0 / DY_KM))), max(1, int(round(10.0 / DX_KM)))
    print(f"Block strides: 5km -> ({sy5},{sx5}) cells; 10km -> ({sy10},{sx10}) cells", flush=True)

    scales = {
        "5km": build_block_units(land_y, land_x, sy5, sx5, "5km"),
        "10km": build_block_units(land_y, land_x, sy10, sx10, "10km"),
    }
    print("Building NUTS-3 assignment...", flush=True)
    nuts_cache = OUT / "nuts3_land_assignment.npz"
    if nuts_cache.exists():
        z = np.load(nuts_cache, allow_pickle=True)
        nuts = {
            "name": "nuts3_EL",
            "kind": "nuts3",
            "unit_id": z["unit_id"],
            "n_units": int(z["n_units"]),
            "area_km2": z["area_km2"],
            "unit_names": z["unit_names"].tolist(),
            "land_mask_for_units": z["land_mask_for_units"].astype(bool),
            "note": "Only Greek (EL) NUTS-3 regions; Balkan/Turkey land cells excluded",
        }
        print(f"  loaded cached NUTS assignment, units={nuts['n_units']}", flush=True)
    else:
        nuts = build_nuts_units(lon, lat, land_y, land_x, NUTS_PATH)
        if nuts is not None:
            np.savez_compressed(
                nuts_cache,
                unit_id=nuts["unit_id"],
                n_units=nuts["n_units"],
                area_km2=nuts["area_km2"],
                unit_names=np.array(nuts["unit_names"], dtype=object),
                land_mask_for_units=nuts["land_mask_for_units"],
            )
    if nuts is not None:
        scales["nuts3"] = nuts
        print(f"  NUTS-3 units={nuts['n_units']}", flush=True)

    for k, sc in scales.items():
        print(f"  scale {k}: n_units={sc['n_units']}, mean area={sc['area_km2'].mean():.2f} km2", flush=True)

    # Accumulators: year -> scale -> list of daily (y, scores_by_model_method)
    # Store compact: for each year/scale/method/model concatenate later
    # Structure: store_day[year][scale] = list of dicts with y and preds
    store = {2019: {}, 2020: {}, 2021: {}}
    for yr in store:
        for sk, sc in scales.items():
            store[yr][sk] = {
                "n_days": 0,
                "y": [],
                "present": [],
                "preds": {
                    "XGBoost_easy_max": [], "XGBoost_easy_q90": [],
                    "XGBoost_hard_max": [], "XGBoost_hard_q90": [],
                    "FWI_max": [], "FWI_q90": [],
                },
            }

    dyn_vars = [ds.variables[name] for name in DYNAMIC]
    fwi_var = ds.variables["fwi"]
    ba_var = ds.variables["burned_areas"]
    buf_dyn = np.full((HISTORY, len(DYNAMIC), lat.size, lon.size), np.nan, np.float32)
    buf_fwi = np.full((HISTORY, lat.size, lon.size), np.nan, np.float32)
    s0, e0 = window_slice()

    def score_land(head, yr):
        dyn = np.empty((n_land, HISTORY, len(DYNAMIC)), np.float32)
        fw = np.empty((n_land, HISTORY), np.float32)
        for k in range(HISTORY):
            idx = (head + k) % HISTORY
            dyn[:, k, :] = buf_dyn[idx][:, land_y, land_x].T
            fw[:, k] = buf_fwi[idx][land_y, land_x]
        dyn10 = fill_time_nans(dyn[:, s0:e0, :])
        fl = np.nan_to_num(fw[:, s0:e0], nan=-1.0)[:, -1]
        p_fwi = np.clip((fl - fwi_min) / max(fwi_max - fwi_min, 1e-6), 0, 1).astype(np.float32)
        out = {"FWI": p_fwi}
        for name, art in models.items():
            dyn_n = art["scaler_dyn"].transform(dyn10.reshape(-1, dyn10.shape[-1])).reshape(dyn10.shape).astype(np.float32)
            sta_n = art["scaler_sta"].transform(np.nan_to_num(sta_land[yr], nan=-1.0)).astype(np.float32)
            out[name] = art["xgb"].predict_proba(xgb_features(dyn_n, sta_n, clc_land))[:, 1].astype(np.float32)
        return out

    # Process years sequentially (skip year if checkpoint exists)
    for yr in (2019, 2020, 2021):
        ckpt = OUT / f"agg_store_{yr}.npz"
        if ckpt.exists():
            print(f"Loading checkpoint {ckpt}", flush=True)
            z = np.load(ckpt)
            for sk in scales:
                store[yr][sk]["n_days"] = int(z[f"{sk}__n_days"])
                store[yr][sk]["y"] = [z[f"{sk}__y"]]
                store[yr][sk]["present"] = [z[f"{sk}__present"]]
                for pk in store[yr][sk]["preds"]:
                    store[yr][sk]["preds"][pk] = [z[f"{sk}__{pk}"]]
            continue
        if yr == 2021:
            days = np.where((years == yr) & np.isin(months, list(FIRE_MONTHS)) & (np.arange(len(years)) >= HISTORY))[0]
        else:
            days = np.where((years == yr) & np.isin(months, list(FIRE_MONTHS)) & (np.arange(len(years)) >= HISTORY))[0]
        days = [int(t) for t in days]
        if not days:
            continue
        t0 = days[0]
        for k in range(HISTORY):
            tt = t0 - HISTORY + k
            for v, var in enumerate(dyn_vars):
                buf_dyn[k, v] = filled(var, tt)
            buf_fwi[k] = filled(fwi_var, tt)
        head = 0
        cur_t = t0
        print(f"Scoring {yr}: {len(days)} days...", flush=True)
        for di, t in enumerate(days):
            while cur_t < t:
                for v, var in enumerate(dyn_vars):
                    buf_dyn[head, v] = filled(var, cur_t)
                buf_fwi[head] = filled(fwi_var, cur_t)
                head = (head + 1) % HISTORY
                cur_t += 1
            ba = filled(ba_var, t)
            y_cell = (ba[land] > 0).astype(np.float32)
            scores = score_land(head, yr)
            for sk, scfg in scales.items():
                mask = scfg["land_mask_for_units"]
                n_u = scfg["n_units"]
                uid = scfg["unit_id"]
                y_u, mx_e, q_e, present = aggregate_day(scores["XGBoost_easy"], y_cell, uid, mask, n_u)
                _, mx_h, q_h, _ = aggregate_day(scores["XGBoost_hard"], y_cell, uid, mask, n_u)
                _, mx_f, q_f, _ = aggregate_day(scores["FWI"], y_cell, uid, mask, n_u)
                store[yr][sk]["y"].append(y_u[present].astype(np.float32))
                store[yr][sk]["present"].append(np.flatnonzero(present).astype(np.int32))
                store[yr][sk]["preds"]["XGBoost_easy_max"].append(mx_e[present].astype(np.float32))
                store[yr][sk]["preds"]["XGBoost_easy_q90"].append(q_e[present].astype(np.float32))
                store[yr][sk]["preds"]["XGBoost_hard_max"].append(mx_h[present].astype(np.float32))
                store[yr][sk]["preds"]["XGBoost_hard_q90"].append(q_h[present].astype(np.float32))
                store[yr][sk]["preds"]["FWI_max"].append(mx_f[present].astype(np.float32))
                store[yr][sk]["preds"]["FWI_q90"].append(q_f[present].astype(np.float32))
                store[yr][sk]["n_days"] += 1
            if (di + 1) % 10 == 0 or di == 0:
                print(f"  {yr} {di+1}/{len(days)} pos_cells={int(y_cell.sum())}", flush=True)
        # checkpoint year aggregates (small)
        ckpt = OUT / f"agg_store_{yr}.npz"
        payload = {}
        for sk in scales:
            payload[f"{sk}__n_days"] = store[yr][sk]["n_days"]
            payload[f"{sk}__y"] = np.concatenate(store[yr][sk]["y"]) if store[yr][sk]["y"] else np.array([])
            payload[f"{sk}__present"] = np.concatenate(store[yr][sk]["present"]) if store[yr][sk]["present"] else np.array([])
            for pk in store[yr][sk]["preds"]:
                payload[f"{sk}__{pk}"] = (
                    np.concatenate(store[yr][sk]["preds"][pk]) if store[yr][sk]["preds"][pk] else np.array([])
                )
        np.savez_compressed(ckpt, **payload)
        print(f"  checkpoint {ckpt}", flush=True)
        print(f"  done {yr}", flush=True)

    ds.close()

    print("Evaluating with 2019-tuned thresholds...", flush=True)
    results = {"protocol": {
        "models": "XGBoost_easy (Exp1/2), XGBoost_hard (Exp3), FWI — no retraining",
        "years": "val 2019 May-Sep; test 2020 May-Sep; test 2021 May-endAug",
        "scales": {k: {
            "n_units": v["n_units"],
            "mean_area_km2": float(v["area_km2"].mean()),
            "kind": v["kind"],
            "mean_extent_km": v.get("mean_extent_km"),
            "stride_cells": v.get("stride_cells"),
            "note": v.get("note"),
        } for k, v in scales.items()},
        "aggregation": ["max", "q90"],
        "event_def": "unit positive on day t if any fire-start burned cell inside unit on day t",
        "threshold": "F1-max on 2019 unit-days only, per model x scale x aggregation",
        "total_land_km2": total_land_km2,
        "cell_km2": CELL_KM2,
        "block_strides": {"5km": {"y": sy5, "x": sx5}, "10km": {"y": sy10, "x": sx10}},
    }, "by_scale": {}}

    pred_keys = [
        ("XGBoost_easy", "max", "XGBoost_easy_max"),
        ("XGBoost_easy", "q90", "XGBoost_easy_q90"),
        ("XGBoost_hard", "max", "XGBoost_hard_max"),
        ("XGBoost_hard", "q90", "XGBoost_hard_q90"),
        ("FWI", "max", "FWI_max"),
        ("FWI", "q90", "FWI_q90"),
    ]

    for sk, scfg in scales.items():
        print(f"Scale {sk}...", flush=True)
        scale_out = {
            "n_units": scfg["n_units"],
            "mean_unit_area_km2": float(scfg["area_km2"].mean()),
            "total_unit_land_km2": float(scfg["area_km2"].sum()),
            "models": {},
        }
        # Build year arrays with matching unit areas for each day-row
        year_data = {}
        for yr in (2019, 2020, 2021):
            if store[yr][sk]["n_days"] == 0:
                continue
            y = np.concatenate(store[yr][sk]["y"])
            present_ids = np.concatenate(store[yr][sk]["present"])
            areas = scfg["area_km2"][present_ids]
            preds = {pk: np.concatenate(store[yr][sk]["preds"][pk]) for _, _, pk in pred_keys}
            year_data[yr] = {
                "y": y, "areas": areas, "preds": preds,
                "n_days": store[yr][sk]["n_days"],
                "n_pos_unit_days": int(y.sum()),
                "n_unit_days": int(len(y)),
            }
            print(f"  {yr}: unit-days={len(y)} pos={int(y.sum())} base={y.mean():.6f} days={store[yr][sk]['n_days']}", flush=True)

        land_km2_eval = float(scfg["area_km2"].sum())
        for model, agg, pk in pred_keys:
            key = f"{model}__{agg}"
            y_val = year_data[2019]["y"]
            p_val = year_data[2019]["preds"][pk]
            thr, _ = best_f1_threshold(y_val, p_val)
            block = {
                "val_threshold": thr,
                "val": pack_eval(y_val, p_val, thr, year_data[2019]["areas"], year_data[2019]["n_days"], land_km2_eval),
            }
            for yr in (2020, 2021):
                if yr not in year_data:
                    continue
                yd = year_data[yr]
                block[str(yr)] = pack_eval(yd["y"], yd["preds"][pk], thr, yd["areas"], yd["n_days"], land_km2_eval)
                block[str(yr)]["n_days"] = yd["n_days"]
                block[str(yr)]["n_units"] = scfg["n_units"]
                block[str(yr)]["n_pos_unit_days"] = yd["n_pos_unit_days"]
            scale_out["models"][key] = block
            b20 = block.get("2020", {})
            print(
                f"  {key}: thr={thr:.3f} 2020 PR={b20.get('pr_auc', float('nan')):.4f} "
                f"ROC={b20.get('roc_auc', float('nan')):.3f} "
                f"P={b20.get('at_val_tuned', {}).get('precision', float('nan')):.4f} "
                f"R={b20.get('at_val_tuned', {}).get('recall', float('nan')):.3f}",
                flush=True,
            )
        # prevalence summary
        scale_out["prevalence"] = {
            str(yr): {
                "n_days": year_data[yr]["n_days"],
                "n_unit_days": year_data[yr]["n_unit_days"],
                "n_pos_unit_days": year_data[yr]["n_pos_unit_days"],
                "base_rate": float(year_data[yr]["y"].mean()),
            }
            for yr in year_data
        }
        results["by_scale"][sk] = scale_out

    # Attach 1 km full-grid baselines from Experiments 2/3 (same models, already computed)
    exp2 = json.loads((ROOT / "results" / "firecube_hardneg" / "metrics.json").read_text(encoding="utf-8"))
    exp3 = json.loads((ROOT / "results" / "firecube_hardneg_train" / "metrics.json").read_text(encoding="utf-8"))
    results["by_scale"]["1km"] = {
        "n_units": n_land,
        "mean_unit_area_km2": CELL_KM2,
        "total_unit_land_km2": total_land_km2,
        "note": "1km metrics reused from Exp2 (easy XGB+FWI) and Exp3 (hard XGB); not recomputed here",
        "prevalence": {
            "2020": {"n_days": exp2["full_summer_grid"]["2020"]["n_days"], "n_pos_unit_days": exp2["full_summer_grid"]["2020"]["XGBoost"]["n_pos"], "base_rate": exp2["full_summer_grid"]["2020"]["XGBoost"]["base_rate"]},
            "2021": {"n_days": exp2["full_summer_grid"]["2021"]["n_days"], "n_pos_unit_days": exp2["full_summer_grid"]["2021"]["XGBoost"]["n_pos"], "base_rate": exp2["full_summer_grid"]["2021"]["XGBoost"]["base_rate"]},
        },
        "models": {
            "XGBoost_easy__max": {
                "source": "exp2_full_summer_grid",
                "2020": exp2["full_summer_grid"]["2020"]["XGBoost"],
                "2021": exp2["full_summer_grid"]["2021"]["XGBoost"],
            },
            "XGBoost_hard__max": {
                "source": "exp3_full_summer_grid",
                "2020": exp3["full_summer_grid"]["2020"]["XGBoost_hardtrain"],
                "2021": exp3["full_summer_grid"]["2021"]["XGBoost_hardtrain"],
            },
            "FWI__max": {
                "source": "exp2_full_summer_grid",
                "2020": exp2["full_summer_grid"]["2020"]["FWI"],
                "2021": exp2["full_summer_grid"]["2021"]["FWI"],
            },
        },
    }
    # For 1km, max == q90
    results["by_scale"]["1km"]["models"]["XGBoost_easy__q90"] = results["by_scale"]["1km"]["models"]["XGBoost_easy__max"]
    results["by_scale"]["1km"]["models"]["XGBoost_hard__q90"] = results["by_scale"]["1km"]["models"]["XGBoost_hard__max"]
    results["by_scale"]["1km"]["models"]["FWI__q90"] = results["by_scale"]["1km"]["models"]["FWI__max"]

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote", OUT / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
