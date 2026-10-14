"""Experiment 3: train XGBoost on hard negatives; evaluate vs Exp 1 XGB and FWI.

Leakage-safe: fire-start positives at day t; predictors t-10..t-1 only.
Does not modify dataset_greece.nc. No new architectures.
"""
from __future__ import annotations

import json
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

ROOT = Path(__file__).resolve().parents[1]
NC_PATH = ROOT / "data" / "raw" / "firecube" / "dataset_greece.nc"
NPZ = ROOT / "data" / "processed" / "firecube_samples.npz"
EXP1 = ROOT / "results" / "firecube_benchmark" / "metrics.json"
EXP2 = ROOT / "results" / "firecube_hardneg" / "metrics.json"
OUT = ROOT / "results" / "firecube_hardneg_train"

HISTORY = 16
SEED = 1234
NEG_POS_RATIO = 2
FIRE_FRAC = 0.7
FIRE_MONTHS = {5, 6, 7, 8, 9}
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
RECALL_TARGETS = (0.10, 0.20, 0.30, 0.50)
CELL_KM2 = 0.9066669762359021 * 0.7144633272136693

DYNAMIC = [
    "max_t2m", "max_d2m", "min_rh", "max_tp", "max_sp",
    "max_wind_speed", "ndvi", "lst_day", "lst_night", "sminx",
]
STATIC = ["DEM", "SLOPE", "ROAD_DISTANCE", "WATERWAY_DISTANCE"]


def filled(var, sl=None):
    data = var[:] if sl is None else var[sl]
    return np.ma.filled(np.ma.array(data), np.nan).astype(np.float32, copy=False)


def clc_year(year: int) -> int:
    if year <= 2011:
        return 2006
    if year <= 2017:
        return 2012
    return 2018


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


def split_of(year: int) -> str:
    if year <= 2018:
        return "train"
    if year == 2019:
        return "val"
    if year == 2020:
        return "test2020"
    return "test2021"


def far_parts(y_true, y_hat):
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    far_al = fp / (fp + tp) if (fp + tp) else 0.0
    return fpr, far_al, int(tn), int(fp), int(fn), int(tp)


def metrics_at(y_true, p, thr, n_days=None, n_land=None):
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
    if n_land and n_days:
        land_km2 = n_land * CELL_KM2
        out["fa_per_1000km2_per_day"] = float(fp / max(n_days, 1) / land_km2 * 1000.0)
        out["alert_km2_per_day"] = float((tp + fp) * CELL_KM2 / max(n_days, 1))
    return out


def ranking_metrics(y_true, p):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    return {
        "pr_auc": float(average_precision_score(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int(len(y_true) - y_true.sum()),
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
                "target_recall": t,
                "precision": None,
                "threshold": None,
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


def pack_eval(y, p, thr, n_days=None, n_land=None):
    block = ranking_metrics(y, p)
    block["at_val_tuned"] = metrics_at(y, p, thr, n_days, n_land)
    block["at_0.5"] = metrics_at(y, p, 0.5, n_days, n_land)
    block["threshold_sweep"] = [metrics_at(y, p, t, n_days, n_land) for t in THRESHOLDS]
    block["precision_at_recall"] = precision_at_recalls(y, p)
    return block


def main():
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading Experiment 1 positives (raw cube windows)...", flush=True)
    data = np.load(NPZ, allow_pickle=True)
    pos_mask = data["label"] == 1
    pos = {
        "t": data["time_index"][pos_mask].astype(np.int32),
        "y": data["y"][pos_mask].astype(np.int32),
        "x": data["x"][pos_mask].astype(np.int32),
        "year": data["year"][pos_mask].astype(np.int16),
        "dyn": data["dynamic"][pos_mask],
        "fwi": data["fwi"][pos_mask],
        "sta": data["static"][pos_mask],
        "clc": data["clc"][pos_mask],
    }
    for name, yrtest in (("train", lambda y: y <= 2018), ("val", lambda y: y == 2019),
                         ("test2020", lambda y: y == 2020), ("test2021", lambda y: y == 2021)):
        print(f"  positives {name}: {int(yrtest(pos['year']).sum())}", flush=True)

    print("Opening cube read-only...", flush=True)
    ds = nc.Dataset(NC_PATH, "r")
    dates = nc.num2date(ds.variables["time"][:], ds.variables["time"].units, only_use_cftime_datetimes=False)
    years = np.array([d.year for d in dates], dtype=np.int16)
    months = np.array([d.month for d in dates], dtype=np.int16)
    nof = filled(ds.variables["number_of_fires"]).astype(np.int16)
    ny, nx = ds.variables["DEM"].shape
    land = filled(ds.variables["CLC_2018_9"]) < 0.5
    land_y, land_x = np.where(land)
    n_land = int(land.sum())
    clc_mode = {
        2006: filled(ds.variables["CLC_2006"]),
        2012: filled(ds.variables["CLC_2012"]),
        2018: filled(ds.variables["CLC_2018"]),
    }
    clc_vec = {
        y: np.stack([filled(ds.variables[f"CLC_{y}_{k}"]) for k in range(10)], axis=0)
        for y in (2006, 2012, 2018)
    }
    static_maps = {name: filled(ds.variables[name]) for name in STATIC}
    pop_maps = {y: filled(ds.variables[f"POP_DENS_{y}"]) for y in range(2009, 2022)}
    fwi_var = ds.variables["fwi"]
    ba_var = ds.variables["burned_areas"]
    season = np.isin(months, list(FIRE_MONTHS)) & (np.arange(len(years)) >= HISTORY)

    print("Train May-Sep land-mean FWI p75...", flush=True)
    train_means = []
    daily_mean_fwi = {}
    for t in np.where(season)[0]:
        m = float(np.nanmean(filled(fwi_var, t)[land]))
        daily_mean_fwi[int(t)] = m
        if years[t] <= 2018:
            train_means.append(m)
    fwi_day_p75 = float(np.nanpercentile(train_means, 75))
    print(f"  p75={fwi_day_p75:.3f}", flush=True)

    def candidate_days(year, kind):
        ts = [int(t) for t in np.where((years == year) & season)[0]]
        if kind == "fire":
            return [t for t in ts if nof[t] > 0]
        return [t for t in ts if nof[t] == 0 and daily_mean_fwi.get(t, -1) >= fwi_day_p75]

    def sample_split(split, pos_idx):
        n_pos = int(pos_idx.sum())
        n_neg = n_pos * NEG_POS_RATIO
        yrs = pos["year"][pos_idx]
        ys = pos["y"][pos_idx]
        xs = pos["x"][pos_idx]
        pos_clc = [int(clc_mode[clc_year(int(yr))][int(yy), int(xx)]) for yr, yy, xx in zip(yrs, ys, xs)]
        classes, counts = np.unique(pos_clc, return_counts=True)
        year_list = sorted(set(int(y) for y in yrs))
        fire_days, high_days = [], []
        for yr in year_list:
            fire_days.extend(candidate_days(yr, "fire"))
            high_days.extend(candidate_days(yr, "high"))
        n_from_fire = int(round(FIRE_FRAC * n_neg)) if fire_days else 0
        n_from_high = n_neg - n_from_fire
        if not high_days:
            n_from_fire, n_from_high = n_neg, 0

        def take(days, n_want, tag):
            got = []
            if n_want <= 0 or not days:
                return got
            quotas = {int(c): int(np.ceil(n_want * cnt / n_pos)) for c, cnt in zip(classes, counts)}
            filled_q = {int(c): 0 for c in quotas}
            for t in rng.permutation(np.array(days)):
                if sum(filled_q.values()) >= n_want:
                    break
                yr = int(years[int(t)])
                mode = clc_mode[clc_year(yr)]
                ba = filled(ba_var, int(t))
                fwi_prev = filled(fwi_var, int(t - 1))
                land_med = float(np.nanmedian(fwi_prev[land]))
                unburned = land & ~(ba > 0)
                high_cell = unburned & np.isfinite(fwi_prev) & (fwi_prev >= land_med)
                for c, q in quotas.items():
                    if filled_q[c] >= q:
                        continue
                    cy, cx = np.where(high_cell & (mode == c))
                    if cy.size == 0:
                        cy, cx = np.where(unburned & (mode == c))
                    if cy.size == 0:
                        continue
                    need = min(q - filled_q[c], 40, cy.size)
                    idx = rng.choice(cy.size, size=need, replace=False)
                    for j in idx:
                        got.append((int(t), int(cy[j]), int(cx[j]), tag))
                        filled_q[c] += 1
            return got[:n_want]

        negs = take(fire_days, n_from_fire, "fire_day") + take(high_days, n_from_high, "high_fwi_day")
        if len(negs) < n_neg:
            extra_days = [int(t) for t in np.where(np.isin(years, year_list) & season)[0]]
            negs.extend(take(extra_days, n_neg - len(negs), "season_fallback"))
        print(
            f"  {split}: {n_pos} pos, {len(negs[:n_neg])} hard negs "
            f"(fire-days {sum(1 for r in negs[:n_neg] if r[3]=='fire_day')})",
            flush=True,
        )
        return negs[:n_neg]

    print("Sampling hard negatives (2:1, 70% fire-day / 30% high-FWI zero-fire)...", flush=True)
    # val/test first so RNG matches Experiment 2 order as closely as possible
    hard_neg = {}
    for split, pred in (
        ("val", lambda y: y == 2019),
        ("test2020", lambda y: y == 2020),
        ("test2021", lambda y: y == 2021),
        ("train", lambda y: y <= 2018),
    ):
        hard_neg[split] = sample_split(split, pred(pos["year"]))

    neg_feat = {}
    for split, rows in hard_neg.items():
        n = len(rows)
        neg_feat[split] = {
            "dyn": np.full((n, HISTORY, len(DYNAMIC)), np.nan, np.float32),
            "fwi": np.full((n, HISTORY), np.nan, np.float32),
            "sta": np.full((n, 5), np.nan, np.float32),
            "clc": np.full((n, 10), np.nan, np.float32),
            "tag": [r[3] for r in rows],
        }
        for i, (t, y, x, _) in enumerate(rows):
            yr = int(np.clip(years[t], 2009, 2021))
            cy = clc_year(yr)
            neg_feat[split]["sta"][i] = [
                static_maps["DEM"][y, x], static_maps["SLOPE"][y, x],
                static_maps["ROAD_DISTANCE"][y, x], static_maps["WATERWAY_DISTANCE"][y, x],
                pop_maps[yr][y, x],
            ]
            neg_feat[split]["clc"][i] = clc_vec[cy][:, y, x]

    by_t = {}
    for split, rows in hard_neg.items():
        for i, (t, y, x, _) in enumerate(rows):
            by_t.setdefault(int(t), []).append((split, i, int(y), int(x)))

    dyn_vars = [ds.variables[name] for name in DYNAMIC]
    buf_dyn = np.full((HISTORY, len(DYNAMIC), ny, nx), np.nan, np.float32)
    buf_fwi = np.full((HISTORY, ny, nx), np.nan, np.float32)
    needed = sorted(by_t)
    print(f"Extracting {len(needed)} unique hard-neg days with jump buffer...", flush=True)

    def reload(label_t):
        for k in range(HISTORY):
            tt = label_t - HISTORY + k
            for v, var in enumerate(dyn_vars):
                buf_dyn[k, v] = filled(var, tt)
            buf_fwi[k] = filled(fwi_var, tt)
        return 0

    def gather(head, ys, xs):
        dyn = np.empty((len(ys), HISTORY, len(DYNAMIC)), np.float32)
        fw = np.empty((len(ys), HISTORY), np.float32)
        for k in range(HISTORY):
            idx = (head + k) % HISTORY
            dyn[:, k, :] = buf_dyn[idx][:, ys, xs].T
            fw[:, k] = buf_fwi[idx][ys, xs]
        return dyn, fw

    head = reload(needed[0])
    cur = needed[0]
    for t in needed:
        if t - cur >= HISTORY:
            head = reload(t)
            cur = t
        else:
            while cur < t:
                for v, var in enumerate(dyn_vars):
                    buf_dyn[head, v] = filled(var, cur)
                buf_fwi[head] = filled(fwi_var, cur)
                head = (head + 1) % HISTORY
                cur += 1
        for split, i, yy, xx in by_t[t]:
            d, f = gather(head, np.array([yy]), np.array([xx]))
            neg_feat[split]["dyn"][i] = d[0]
            neg_feat[split]["fwi"][i] = f[0]
        if t % 200 == 0:
            print(f"  extracted label-day t={t} {dates[t]}", flush=True)

    ds.close()

    def pos_split(pred):
        m = pred(pos["year"])
        return {k: pos[k][m] for k in ("dyn", "fwi", "sta", "clc")}

    raw = {}
    for split, pred in (
        ("train", lambda y: y <= 2018),
        ("val", lambda y: y == 2019),
        ("test2020", lambda y: y == 2020),
        ("test2021", lambda y: y == 2021),
    ):
        p = pos_split(pred)
        n = neg_feat[split]
        raw[split] = {
            "dyn": np.concatenate([p["dyn"], n["dyn"]], axis=0),
            "fwi": np.concatenate([p["fwi"], n["fwi"]], axis=0),
            "sta": np.concatenate([np.nan_to_num(p["sta"], nan=-1.0), n["sta"]], axis=0),
            "clc": np.concatenate([np.nan_to_num(p["clc"], nan=0.0), n["clc"]], axis=0),
            "y": np.concatenate([np.ones(len(p["dyn"])), np.zeros(len(n["dyn"]))]).astype(np.float32),
        }
        print(f"  packed {split}: n={len(raw[split]['y'])} pos={int(raw[split]['y'].sum())}", flush=True)

    s, e = window_slice()
    def prep(split, scaler_dyn=None, scaler_sta=None):
        dyn = fill_time_nans(raw[split]["dyn"][:, s:e, :])
        sta = np.nan_to_num(raw[split]["sta"], nan=-1.0).astype(np.float32)
        clc = np.nan_to_num(raw[split]["clc"], nan=0.0).astype(np.float32)
        fwi_last = np.nan_to_num(raw[split]["fwi"][:, s:e], nan=-1.0)[:, -1]
        if scaler_dyn is None:
            scaler_dyn = MinMaxScaler().fit(dyn.reshape(-1, dyn.shape[-1]))
            scaler_sta = MinMaxScaler().fit(sta)
        dyn_n = scaler_dyn.transform(dyn.reshape(-1, dyn.shape[-1])).reshape(dyn.shape).astype(np.float32)
        sta_n = scaler_sta.transform(sta).astype(np.float32)
        x = xgb_features(dyn_n, sta_n, clc)
        return x, raw[split]["y"], fwi_last, scaler_dyn, scaler_sta, dyn_n

    print("Training XGBoost from scratch on hard-negative train set...", flush=True)
    x_tr, y_tr, fwi_tr, scaler_dyn, scaler_sta, _ = prep("train")
    x_va, y_va, fwi_va, _, _, _ = prep("val", scaler_dyn, scaler_sta)
    fwi_min, fwi_max = np.percentile(fwi_tr, [1, 99])
    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic",
        eval_metric="aucpr", n_jobs=4, random_state=SEED, early_stopping_rounds=40,
    )
    xgb.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    joblib.dump(
        {"xgb": xgb, "scaler_dyn": scaler_dyn, "scaler_sta": scaler_sta,
         "fwi_min": float(fwi_min), "fwi_max": float(fwi_max)},
        OUT / "xgb_hardtrain.joblib",
    )
    print(f"  best_iteration={getattr(xgb, 'best_iteration', None)}", flush=True)

    def fwi_p(arr):
        return np.clip((arr - fwi_min) / max(fwi_max - fwi_min, 1e-6), 0, 1)

    p_va = xgb.predict_proba(x_va)[:, 1]
    thr_xgb, _ = best_f1_threshold(y_va, p_va)
    thr_fwi, _ = best_f1_threshold(y_va, fwi_p(fwi_va))
    print(f"  val-tuned thresholds XGB={thr_xgb:.3f} FWI={thr_fwi:.3f}", flush=True)

    sampled = {}
    for split in ("val", "test2020", "test2021"):
        x, y, fwi_last, _, _, _ = prep(split, scaler_dyn, scaler_sta)
        px = xgb.predict_proba(x)[:, 1]
        pf = fwi_p(fwi_last)
        sampled[split] = {"XGBoost_hardtrain": pack_eval(y, px, thr_xgb), "FWI": pack_eval(y, pf, thr_fwi)}
        b = sampled[split]["XGBoost_hardtrain"]
        print(
            f"  {split} hardtrain PR={b['pr_auc']:.3f} ROC={b['roc_auc']:.3f} "
            f"F1={b['at_val_tuned']['f1']:.3f} P={b['at_val_tuned']['precision']:.3f}",
            flush=True,
        )

    print("Full summer grid with hard-trained XGBoost...", flush=True)
    ds = nc.Dataset(NC_PATH, "r")
    dyn_vars = [ds.variables[name] for name in DYNAMIC]
    fwi_var = ds.variables["fwi"]
    ba_var = ds.variables["burned_areas"]
    clc_land = clc_vec[2018][:, land].T.astype(np.float32)
    sta_land = {
        yr: np.stack([
            static_maps["DEM"][land], static_maps["SLOPE"][land],
            static_maps["ROAD_DISTANCE"][land], static_maps["WATERWAY_DISTANCE"][land],
            pop_maps[yr][land],
        ], axis=1).astype(np.float32)
        for yr in (2020, 2021)
    }
    buf_dyn = np.full((HISTORY, len(DYNAMIC), ny, nx), np.nan, np.float32)
    buf_fwi = np.full((HISTORY, ny, nx), np.nan, np.float32)

    def score_land(head, yr):
        dyn = np.empty((n_land, HISTORY, len(DYNAMIC)), np.float32)
        fw = np.empty((n_land, HISTORY), np.float32)
        for k in range(HISTORY):
            idx = (head + k) % HISTORY
            dyn[:, k, :] = buf_dyn[idx][:, land_y, land_x].T
            fw[:, k] = buf_fwi[idx][land_y, land_x]
        s0, e0 = window_slice()
        dyn10 = fill_time_nans(dyn[:, s0:e0, :])
        dyn_n = scaler_dyn.transform(dyn10.reshape(-1, dyn10.shape[-1])).reshape(dyn10.shape).astype(np.float32)
        sta_n = scaler_sta.transform(np.nan_to_num(sta_land[yr], nan=-1.0)).astype(np.float32)
        px = xgb.predict_proba(xgb_features(dyn_n, sta_n, clc_land))[:, 1]
        fl = np.nan_to_num(fw[:, s0:e0], nan=-1.0)[:, -1]
        pf = fwi_p(fl)
        return px, pf

    grid_report = {}
    for yr in (2020, 2021):
        days = np.where((years == yr) & np.isin(months, list(FIRE_MONTHS)) & (np.arange(len(years)) >= HISTORY))[0]
        if len(days) == 0:
            continue
        t0 = int(days[0])
        for k in range(HISTORY):
            tt = t0 - HISTORY + k
            for v, var in enumerate(dyn_vars):
                buf_dyn[k, v] = filled(var, tt)
            buf_fwi[k] = filled(fwi_var, tt)
        head = 0
        ys, pxs, pfs = [], [], []
        cur_t = t0
        for t in days:
            t = int(t)
            while cur_t < t:
                for v, var in enumerate(dyn_vars):
                    buf_dyn[head, v] = filled(var, cur_t)
                buf_fwi[head] = filled(fwi_var, cur_t)
                head = (head + 1) % HISTORY
                cur_t += 1
            ba = filled(ba_var, t)
            px, pf = score_land(head, yr)
            ys.append((ba[land] > 0).astype(np.uint8))
            pxs.append(px.astype(np.float16))
            pfs.append(pf.astype(np.float16))
            if len(ys) % 20 == 0:
                print(f"  grid {yr} {len(ys)}/{len(days)} pos_today={int(ys[-1].sum())}", flush=True)
        y = np.concatenate(ys).astype(np.float32)
        px = np.concatenate(pxs).astype(np.float32)
        pf = np.concatenate(pfs).astype(np.float32)
        print(f"  {yr} cell-days={len(y)} pos={int(y.sum())}", flush=True)
        grid_report[str(yr)] = {
            "n_days": int(len(days)),
            "n_land": n_land,
            "land_km2": n_land * CELL_KM2,
            "XGBoost_hardtrain": pack_eval(y, px, thr_xgb, len(days), n_land),
            "FWI": pack_eval(y, pf, thr_fwi, len(days), n_land),
        }
        del y, px, pf, ys, pxs, pfs
    ds.close()

    exp1 = json.loads(EXP1.read_text(encoding="utf-8"))
    exp2 = json.loads(EXP2.read_text(encoding="utf-8"))
    comparison = {}
    for split in ("val", "test2020", "test2021"):
        e1e = exp1["XGBoost"][split]
        e2h = exp2["sampled_hard"][split]["XGBoost"]
        e3 = sampled[split]["XGBoost_hardtrain"]
        fwi3 = sampled[split]["FWI"]
        comparison[split] = {
            "exp1_easytrain_easyeval_pr_auc": e1e["pr_auc"],
            "exp2_easytrain_hardeval_pr_auc": e2h["pr_auc"],
            "exp3_hardtrain_hardeval_pr_auc": e3["pr_auc"],
            "d_pr_auc_vs_exp2": e3["pr_auc"] - e2h["pr_auc"],
            "d_roc_auc_vs_exp2": e3["roc_auc"] - e2h["roc_auc"],
            "d_brier_vs_exp2": e3["brier"] - e2h["brier"],
            "exp2_roc": e2h["roc_auc"],
            "exp3_roc": e3["roc_auc"],
            "fwi_pr_auc": fwi3["pr_auc"],
            "fwi_roc": fwi3["roc_auc"],
        }
    grid_vs = {}
    for yr in ("2020", "2021"):
        if yr not in grid_report:
            continue
        e2 = exp2["full_summer_grid"][yr]
        e3 = grid_report[yr]
        grid_vs[yr] = {}
        for key2, key3 in (("XGBoost", "XGBoost_hardtrain"), ("FWI", "FWI")):
            a, b = e2[key2], e3[key3]
            op2 = a["at_exp1_val_threshold"] if "at_exp1_val_threshold" in a else a["at_hard_or_primary"]
            op3 = b["at_val_tuned"]
            grid_vs[yr][key3] = {
                "exp2_pr_auc": a["pr_auc"],
                "exp3_pr_auc": b["pr_auc"],
                "d_pr_auc": b["pr_auc"] - a["pr_auc"],
                "exp2_roc": a["roc_auc"],
                "exp3_roc": b["roc_auc"],
                "d_roc": b["roc_auc"] - a["roc_auc"],
                "exp2_fa_exp1cut": op2.get("fa_per_1000km2_per_day"),
                "exp3_fa_valtuned": op3.get("fa_per_1000km2_per_day"),
                "exp2_prec_exp1cut": op2["precision"],
                "exp3_prec_valtuned": op3["precision"],
                "exp2_fpr_exp1cut": op2["fpr"],
                "exp3_fpr_valtuned": op3["fpr"],
            }

    report = {
        "protocol": {
            "positives": "fire-start burned land cells (same as Exp 1/2)",
            "hard_negatives": (
                "May-Sep unburned land; CLC-matched to positives; 2:1 ratio; "
                "70% on domain fire days with cell FWI(t-1)>=daily land median; "
                "30% on zero-fire days with land-mean FWI>=train May-Sep p75"
            ),
            "train": "2009-2018 hard set; val 2019 hard set for early stop + F1 threshold; test 2020/2021",
            "features": "t-10..t-1 dynamic mean+last + DEM/slope/roads/water/pop + CLC 10-frac",
            "fwi_day_p75_train": fwi_day_p75,
            "xgb_best_iteration": int(getattr(xgb, "best_iteration", -1)),
            "thr_xgb_2019": thr_xgb,
            "thr_fwi_2019": thr_fwi,
            "cell_km2": CELL_KM2,
        },
        "sampled_hard": sampled,
        "full_summer_grid": grid_report,
        "vs_exp1_exp2_sampled": comparison,
        "vs_exp2_grid": grid_vs,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", OUT / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
