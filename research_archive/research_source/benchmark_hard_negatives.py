"""Experiment 2: hard-negative + full-summer-grid eval of the current FWI/XGBoost system.

Retrains XGBoost on Experiment 1 (easy-negative) training data, then evaluates
on fire-season / high-FWI unburned cells and on all land cells in May-Sep.
Does not modify dataset_greece.nc. No LSTM/CNN/Transformer.
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
OUT = ROOT / "results" / "firecube_hardneg"

HISTORY = 16
H = 1
SEED = 1234
NEG_POS_RATIO = 2
FIRE_MONTHS = {5, 6, 7, 8, 9}
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
DY_KM = 0.9066669762359021
DX_KM = 0.7144633272136693
CELL_KM2 = DY_KM * DX_KM

DYNAMIC = [
    "max_t2m",
    "max_d2m",
    "min_rh",
    "max_tp",
    "max_sp",
    "max_wind_speed",
    "ndvi",
    "lst_day",
    "lst_night",
    "sminx",
]
STATIC = ["DEM", "SLOPE", "ROAD_DISTANCE", "WATERWAY_DISTANCE"]


def filled(var, sl=None):
    data = var[:] if sl is None else var[sl]
    return np.ma.filled(np.ma.array(data), np.nan).astype(np.float32, copy=False)


def window_slice(history=HISTORY, h=H):
    last = history - h
    return last - 9, last + 1


def fill_time_nans(dyn: np.ndarray) -> np.ndarray:
    colmean = np.nanmean(dyn, axis=1, keepdims=True)
    colmean = np.where(np.isfinite(colmean), colmean, 0.0)
    out = np.where(np.isfinite(dyn), dyn, np.broadcast_to(colmean, dyn.shape))
    return np.nan_to_num(out, nan=-1.0).astype(np.float32)


def xgb_features(dyn10, static, clc):
    return np.concatenate([dyn10.mean(axis=1), dyn10[:, -1, :], static, clc], axis=1)


def split_mask(year, name):
    if name == "train":
        return year <= 2018
    if name == "val":
        return year == 2019
    if name == "test2020":
        return year == 2020
    if name == "test2021":
        return year == 2021
    raise ValueError(name)


def far_parts(y_true, y_hat):
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    far_al = fp / (fp + tp) if (fp + tp) else 0.0
    return fpr, far_al, int(tn), int(fp), int(fn), int(tp)


def metrics_at(y_true, p, thr, n_days=None, n_land=None):
    y_hat = (p >= thr).astype(np.int32)
    fpr, far_al, tn, fp, fn, tp = far_parts(y_true, y_hat)
    out = {
        "threshold": float(thr),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "fpr": float(fpr),
        "false_alarm_among_alerts": float(far_al),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if n_land and n_days:
        land_km2 = n_land * CELL_KM2
        # mean daily false-alarm cells per 1000 km²
        out["fa_per_1000km2_per_day"] = float(fp / max(n_days, 1) / land_km2 * 1000.0)
        out["alert_km2_per_day"] = float((tp + fp) * CELL_KM2 / max(n_days, 1))
    return out


def ranking_metrics(y_true, p):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    if y_true.min() == y_true.max():
        pr = roc = float("nan")
    else:
        pr = float(average_precision_score(y_true, p))
        roc = float(roc_auc_score(y_true, p))
    return {
        "pr_auc": pr,
        "roc_auc": roc,
        "brier": float(brier_score_loss(y_true, p)),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int(len(y_true) - y_true.sum()),
        "base_rate": float(y_true.mean()),
    }


def best_f1_threshold(y_true, p, grid=None):
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 37)
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(y_true, p >= t, zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, float(best_f)


def pack_eval(y, p, thr_primary, thr_exp1, n_days=None, n_land=None):
    block = ranking_metrics(y, p)
    block["at_hard_or_primary"] = metrics_at(y, p, thr_primary, n_days, n_land)
    block["at_exp1_val_threshold"] = metrics_at(y, p, thr_exp1, n_days, n_land)
    block["at_0.5"] = metrics_at(y, p, 0.5, n_days, n_land)
    block["threshold_sweep"] = [metrics_at(y, p, t, n_days, n_land) for t in THRESHOLDS]
    return block


def train_current_xgb():
    print("Loading Experiment 1 samples and training current XGBoost...", flush=True)
    data = np.load(NPZ, allow_pickle=True)
    dyn_raw = data["dynamic"]
    fwi_raw = data["fwi"]
    static = np.nan_to_num(data["static"], nan=-1.0).astype(np.float32)
    clc = np.nan_to_num(data["clc"], nan=0.0).astype(np.float32)
    label = data["label"].astype(np.float32)
    year = data["year"]
    s, e = window_slice()
    dyn = fill_time_nans(dyn_raw[:, s:e, :])
    fwi_last = np.nan_to_num(fwi_raw[:, s:e], nan=-1.0)[:, -1]
    train_m = split_mask(year, "train")
    val_m = split_mask(year, "val")
    scaler_dyn = MinMaxScaler().fit(dyn[train_m].reshape(-1, dyn.shape[-1]))
    dyn_n = scaler_dyn.transform(dyn.reshape(-1, dyn.shape[-1])).reshape(dyn.shape).astype(np.float32)
    scaler_sta = MinMaxScaler().fit(static[train_m])
    sta_n = scaler_sta.transform(static).astype(np.float32)
    xgb_x = xgb_features(dyn_n, sta_n, clc)
    fwi_min, fwi_max = np.percentile(fwi_last[train_m], [1, 99])
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=4,
        random_state=SEED,
        early_stopping_rounds=40,
    )
    xgb.fit(xgb_x[train_m], label[train_m], eval_set=[(xgb_x[val_m], label[val_m])], verbose=False)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "xgb": xgb,
            "scaler_dyn": scaler_dyn,
            "scaler_sta": scaler_sta,
            "fwi_min": float(fwi_min),
            "fwi_max": float(fwi_max),
        },
        OUT / "xgb_easytrain.joblib",
    )
    pos_xy = {}
    for name in ("val", "test2020", "test2021"):
        m = split_mask(year, name) & (label == 1)
        pos_xy[name] = {
            "t": data["time_index"][m].astype(np.int32),
            "y": data["y"][m].astype(np.int32),
            "x": data["x"][m].astype(np.int32),
            "dyn": dyn_n[m],
            "sta": sta_n[m],
            "clc": clc[m],
            "fwi": fwi_last[m],
        }
        print(f"  reused positives {name}: {int(m.sum())}", flush=True)
    return xgb, scaler_dyn, scaler_sta, fwi_min, fwi_max, pos_xy, int(getattr(xgb, "best_iteration", -1))


def score_arrays(xgb, scaler_dyn, scaler_sta, fwi_min, fwi_max, dyn_raw, fwi_seq, sta, clc):
    s, e = window_slice()
    dyn = fill_time_nans(dyn_raw[:, s:e, :])
    dyn_n = scaler_dyn.transform(dyn.reshape(-1, dyn.shape[-1])).reshape(dyn.shape).astype(np.float32)
    sta_n = scaler_sta.transform(np.nan_to_num(sta, nan=-1.0)).astype(np.float32)
    clc_n = np.nan_to_num(clc, nan=0.0).astype(np.float32)
    p_xgb = xgb.predict_proba(xgb_features(dyn_n, sta_n, clc_n))[:, 1]
    fwi_last = np.nan_to_num(fwi_seq[:, s:e], nan=-1.0)[:, -1]
    p_fwi = np.clip((fwi_last - fwi_min) / max(fwi_max - fwi_min, 1e-6), 0, 1)
    return p_xgb, p_fwi


def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    xgb, scaler_dyn, scaler_sta, fwi_min, fwi_max, pos_xy, best_iter = train_current_xgb()

    print("Opening cube (read-only)...", flush=True)
    ds = nc.Dataset(NC_PATH, "r")
    dates = nc.num2date(ds.variables["time"][:], ds.variables["time"].units, only_use_cftime_datetimes=False)
    years = np.array([d.year for d in dates], dtype=np.int16)
    months = np.array([d.month for d in dates], dtype=np.int16)
    nof = filled(ds.variables["number_of_fires"]).astype(np.int16)
    ny, nx = ds.variables["DEM"].shape
    land = filled(ds.variables["CLC_2018_9"]) < 0.5
    land_y, land_x = np.where(land)
    n_land = int(land.sum())
    clc_mode = filled(ds.variables["CLC_2018"])
    clc_vec = np.stack([filled(ds.variables[f"CLC_2018_{k}"]) for k in range(10)], axis=0)
    static_maps = {name: filled(ds.variables[name]) for name in STATIC}
    pop = {
        2019: filled(ds.variables["POP_DENS_2019"]),
        2020: filled(ds.variables["POP_DENS_2020"]),
        2021: filled(ds.variables["POP_DENS_2021"]),
    }
    sta_land = {
        yr: np.stack(
            [
                static_maps["DEM"][land],
                static_maps["SLOPE"][land],
                static_maps["ROAD_DISTANCE"][land],
                static_maps["WATERWAY_DISTANCE"][land],
                pop[yr][land],
            ],
            axis=1,
        ).astype(np.float32)
        for yr in (2020, 2021)
    }
    clc_land = clc_vec[:, land].T.astype(np.float32)

    season = np.isin(months, list(FIRE_MONTHS)) & (np.arange(len(years)) >= HISTORY)

    print("Computing train-season land-mean FWI for high-FWI day threshold...", flush=True)
    fwi_var = ds.variables["fwi"]
    ba_var = ds.variables["burned_areas"]
    train_means = []
    for t in np.where(season & (years <= 2018))[0]:
        sl = filled(fwi_var, t)[land]
        train_means.append(float(np.nanmean(sl)))
    fwi_day_p75 = float(np.nanpercentile(train_means, 75))
    print(f"  train May-Sep land-mean FWI p75 = {fwi_day_p75:.3f}", flush=True)

    # candidate days 2019-2021
    daily_mean_fwi = {}
    for t in np.where(season & (years >= 2019))[0]:
        sl = filled(fwi_var, t)[land]
        daily_mean_fwi[int(t)] = float(np.nanmean(sl))

    def candidate_days(year, kind):
        ts = [t for t, y, m in zip(range(len(years)), years, months) if y == year and m in FIRE_MONTHS and t >= HISTORY]
        if kind == "fire":
            return [t for t in ts if nof[t] > 0]
        return [t for t in ts if nof[t] == 0 and daily_mean_fwi.get(t, -1) >= fwi_day_p75]

    hard_neg = {"val": [], "test2020": [], "test2021": []}
    year_of = {"val": 2019, "test2020": 2020, "test2021": 2021}
    print("Sampling CLC-aware hard negatives (2:1)...", flush=True)
    for split, bundle in pos_xy.items():
        yr = year_of[split]
        n_pos = len(bundle["t"])
        n_neg = n_pos * NEG_POS_RATIO
        pos_clc = [int(clc_mode[int(yy), int(xx)]) for yy, xx in zip(bundle["y"], bundle["x"])]
        classes, counts = np.unique(pos_clc, return_counts=True)
        fire_days = candidate_days(yr, "fire")
        high_days = candidate_days(yr, "high")
        if not fire_days and not high_days:
            raise RuntimeError(f"No hard-neg days for {split}")
        # 70% same-day-as-fire spatial hard negs, 30% high-FWI no-fire days
        n_from_fire = int(round(0.7 * n_neg)) if fire_days else 0
        n_from_high = n_neg - n_from_fire
        if not high_days:
            n_from_fire, n_from_high = n_neg, 0

        def take(days, n_want, tag):
            got = []
            if n_want <= 0 or not days:
                return got
            days_arr = np.array(days)
            # CLC quotas proportional to positives
            quotas = {int(c): int(np.ceil(n_want * cnt / n_pos)) for c, cnt in zip(classes, counts)}
            filled_q = {int(c): 0 for c in quotas}
            order = rng.permutation(days_arr)
            for t in order:
                if sum(filled_q.values()) >= n_want:
                    break
                ba = filled(ba_var, int(t))
                fwi_prev = filled(fwi_var, int(t - 1))
                land_med = float(np.nanmedian(fwi_prev[land]))
                unburned = land & ~(ba > 0)
                high_cell = unburned & np.isfinite(fwi_prev) & (fwi_prev >= land_med)
                for c, q in quotas.items():
                    if filled_q[c] >= q:
                        continue
                    ys, xs = np.where(high_cell & (clc_mode == c))
                    if ys.size == 0:
                        ys, xs = np.where(unburned & (clc_mode == c))
                    if ys.size == 0:
                        continue
                    need = min(q - filled_q[c], 40, ys.size)
                    idx = rng.choice(ys.size, size=need, replace=False)
                    for j in idx:
                        got.append((int(t), int(ys[j]), int(xs[j]), tag))
                        filled_q[c] += 1
                        if sum(filled_q.values()) >= n_want:
                            break
            return got[:n_want]

        negs = take(fire_days, n_from_fire, "fire_day") + take(high_days, n_from_high, "high_fwi_day")
        if len(negs) < n_neg:
            # fallback: any season unburned land
            extra = take(
                [t for t in range(len(years)) if years[t] == yr and months[t] in FIRE_MONTHS and t >= HISTORY],
                n_neg - len(negs),
                "season_fallback",
            )
            negs.extend(extra)
        hard_neg[split] = negs[:n_neg]
        n_fd = sum(1 for *_, tag in hard_neg[split] if tag == "fire_day")
        print(f"  {split}: {len(hard_neg[split])} negs ({n_fd} on fire days)", flush=True)

    needed_t = set()
    for split, rows in hard_neg.items():
        needed_t.update(t for t, _, _, _ in rows)

    # allocate hard-neg feature arrays
    neg_feat = {}
    for split, rows in hard_neg.items():
        n = len(rows)
        neg_feat[split] = {
            "t": np.array([r[0] for r in rows], dtype=np.int32),
            "y": np.array([r[1] for r in rows], dtype=np.int32),
            "x": np.array([r[2] for r in rows], dtype=np.int32),
            "tag": [r[3] for r in rows],
            "dyn": np.full((n, HISTORY, len(DYNAMIC)), np.nan, np.float32),
            "fwi": np.full((n, HISTORY), np.nan, np.float32),
            "sta": np.full((n, 5), np.nan, np.float32),
            "clc": np.full((n, 10), np.nan, np.float32),
        }
        yr = year_of[split]
        for i, (t, y, x, _) in enumerate(rows):
            neg_feat[split]["sta"][i] = [
                static_maps["DEM"][y, x],
                static_maps["SLOPE"][y, x],
                static_maps["ROAD_DISTANCE"][y, x],
                static_maps["WATERWAY_DISTANCE"][y, x],
                pop[yr][y, x],
            ]
            neg_feat[split]["clc"][i] = clc_vec[:, y, x]

    by_t = {}
    for split, rows in hard_neg.items():
        for i, (t, y, x, _) in enumerate(rows):
            by_t.setdefault(int(t), []).append((split, i))

    dyn_vars = [ds.variables[name] for name in DYNAMIC]
    buf_dyn = np.full((HISTORY, len(DYNAMIC), ny, nx), np.nan, np.float32)
    buf_fwi = np.full((HISTORY, ny, nx), np.nan, np.float32)

    start_t = int(np.where((years == 2019) & (months == 5))[0][0])
    # need buffer filled with days start_t-HISTORY ... start_t-1
    fill_from = start_t - HISTORY
    print(f"Filling buffer from t={fill_from} then scanning {fill_from}..end...", flush=True)
    for k, t in enumerate(range(fill_from, start_t)):
        for v, var in enumerate(dyn_vars):
            buf_dyn[k, v] = filled(var, t)
        buf_fwi[k] = filled(fwi_var, t)
    head = 0

    grid_pred = {
        2020: {"y": [], "p_xgb": [], "p_fwi": [], "fp_xgb_05": [], "fp_xgb_50": [], "fp_fwi_10": [], "fp_fwi_50": [], "n_pos_day": []},
        2021: {"y": [], "p_xgb": [], "p_fwi": [], "fp_xgb_05": [], "fp_xgb_50": [], "fp_fwi_10": [], "fp_fwi_50": [], "n_pos_day": []},
    }
    # store grid as running lists of day-level arrays then concatenate — memory heavy.
    # Instead accumulate day metrics and reservoir of scores? We need PR-AUC so store compressed:
    # keep all y (bit) and p float16.
    grid_y = {2020: [], 2021: []}
    grid_px = {2020: [], 2021: []}
    grid_pf = {2020: [], 2021: []}
    grid_days = {2020: 0, 2021: 0}

    max_t = len(years) - 1
    s_win, e_win = window_slice()

    def gather_window(ys, xs):
        dyn = np.empty((len(ys), HISTORY, len(DYNAMIC)), np.float32)
        fw = np.empty((len(ys), HISTORY), np.float32)
        for k in range(HISTORY):
            idx = (head + k) % HISTORY
            dyn[:, k, :] = buf_dyn[idx][:, ys, xs].T
            fw[:, k] = buf_fwi[idx][ys, xs]
        return dyn, fw

    for t in range(start_t, max_t + 1):
        yr = int(years[t])
        mo = int(months[t])
        if t in by_t:
            for split, i in by_t[t]:
                yy = int(neg_feat[split]["y"][i])
                xx = int(neg_feat[split]["x"][i])
                d, f = gather_window(np.array([yy]), np.array([xx]))
                neg_feat[split]["dyn"][i] = d[0]
                neg_feat[split]["fwi"][i] = f[0]

        if yr in (2020, 2021) and mo in FIRE_MONTHS:
            ba = filled(ba_var, t)
            y_land = (ba[land] > 0).astype(np.float32)
            dyn, fw = gather_window(land_y, land_x)
            p_xgb, p_fwi = score_arrays(xgb, scaler_dyn, scaler_sta, fwi_min, fwi_max, dyn, fw, sta_land[yr], clc_land)
            grid_y[yr].append(y_land.astype(np.uint8))
            grid_px[yr].append(p_xgb.astype(np.float16))
            grid_pf[yr].append(p_fwi.astype(np.float16))
            grid_days[yr] += 1
            if grid_days[yr] % 20 == 0:
                print(f"  grid {yr} day {grid_days[yr]} t={t} pos={int(y_land.sum())}", flush=True)

        if t == max_t:
            break
        for v, var in enumerate(dyn_vars):
            buf_dyn[head, v] = filled(var, t)
        buf_fwi[head] = filled(fwi_var, t)
        head = (head + 1) % HISTORY
        if (t - start_t) % 50 == 0:
            print(f"  scan t={t}/{max_t} {dates[t]}", flush=True)

    ds.close()

    exp1 = json.loads(EXP1.read_text(encoding="utf-8"))
    exp1_thr_xgb = float(exp1["XGBoost"]["val_best_f1_threshold"])
    exp1_thr_fwi = float(exp1["FWI"]["val_best_f1_threshold"])

    print("Scoring sampled hard-negative sets...", flush=True)
    hard_report = {}
    hard_val_scores = {}
    for split in ("val", "test2020", "test2021"):
        pos = pos_xy[split]
        neg = neg_feat[split]
        p_xgb_pos = xgb.predict_proba(xgb_features(pos["dyn"], pos["sta"], pos["clc"]))[:, 1]
        p_fwi_pos = np.clip((pos["fwi"] - fwi_min) / max(fwi_max - fwi_min, 1e-6), 0, 1)
        p_xgb_neg, p_fwi_neg = score_arrays(
            xgb, scaler_dyn, scaler_sta, fwi_min, fwi_max, neg["dyn"], neg["fwi"], neg["sta"], neg["clc"]
        )
        y = np.concatenate([np.ones(len(p_xgb_pos)), np.zeros(len(p_xgb_neg))])
        px = np.concatenate([p_xgb_pos, p_xgb_neg])
        pf = np.concatenate([p_fwi_pos, p_fwi_neg])
        hard_val_scores[split] = (y, px, pf)
        n_fire_day = sum(1 for tag in neg["tag"] if tag == "fire_day")
        print(
            f"  {split} n={len(y)} pos={int(y.sum())} neg_fire_days={n_fire_day}/{len(neg['tag'])}",
            flush=True,
        )

    thr_xgb_hard, _ = best_f1_threshold(*hard_val_scores["val"][:2])
    thr_fwi_hard, _ = best_f1_threshold(hard_val_scores["val"][0], hard_val_scores["val"][2])

    sampled = {}
    for split, (y, px, pf) in hard_val_scores.items():
        sampled[split] = {
            "XGBoost": pack_eval(y, px, thr_xgb_hard, exp1_thr_xgb),
            "FWI": pack_eval(y, pf, thr_fwi_hard, exp1_thr_fwi),
            "n_hard_negs_on_fire_days": int(sum(1 for tag in neg_feat[split]["tag"] if tag == "fire_day")),
            "n_hard_negs_on_high_fwi_days": int(sum(1 for tag in neg_feat[split]["tag"] if tag == "high_fwi_day")),
        }

    print("Scoring full summer grids...", flush=True)
    grid_report = {}
    for yr in (2020, 2021):
        if not grid_y[yr]:
            continue
        y = np.concatenate(grid_y[yr]).astype(np.float32)
        px = np.concatenate(grid_px[yr]).astype(np.float32)
        pf = np.concatenate(grid_pf[yr]).astype(np.float32)
        n_days = grid_days[yr]
        print(f"  {yr}: {n_days} days, {len(y)} cell-days, pos={int(y.sum())}", flush=True)
        grid_report[str(yr)] = {
            "n_days": n_days,
            "n_land": n_land,
            "cell_km2": CELL_KM2,
            "land_km2": n_land * CELL_KM2,
            "XGBoost": pack_eval(y, px, thr_xgb_hard, exp1_thr_xgb, n_days, n_land),
            "FWI": pack_eval(y, pf, thr_fwi_hard, exp1_thr_fwi, n_days, n_land),
        }
        del y, px, pf, grid_y[yr], grid_px[yr], grid_pf[yr]

    # Experiment 1 vs 2 deltas (sampled, same 1:2 ratio, same positives)
    def grab(exp, model, split, key):
        block = exp[model][split]
        if key in block:
            return block[key]
        return block["at_val_best_f1"][key] if key in block.get("at_val_best_f1", {}) else None

    comparison = {}
    for split, exp1_split in (("test2020", "test2020"), ("test2021", "test2021"), ("val", "val")):
        comparison[split] = {}
        for model in ("XGBoost", "FWI"):
            e1 = exp1[model][exp1_split]
            e2 = sampled[split][model]
            comparison[split][model] = {
                "exp1_pr_auc": e1["pr_auc"],
                "exp2_pr_auc": e2["pr_auc"],
                "d_pr_auc": e2["pr_auc"] - e1["pr_auc"],
                "exp1_roc_auc": e1["roc_auc"],
                "exp2_roc_auc": e2["roc_auc"],
                "d_roc_auc": e2["roc_auc"] - e1["roc_auc"],
                "exp1_brier": e1["brier"],
                "exp2_brier": e2["brier"],
                "d_brier": e2["brier"] - e1["brier"],
            }

    report = {
        "protocol": {
            "models": "FWI (t-1) and XGBoost trained on Experiment 1 easy negatives",
            "positives": "same fire-start burned cells as Experiment 1",
            "hard_negatives": (
                "May-Sep land cells that did not burn; 70% on days with fires in the domain "
                "(same-day spatial hard negs, cell FWI(t-1) >= that day's land median, CLC-matched); "
                "30% on zero-fire days whose land-mean FWI >= train May-Sep p75"
            ),
            "ratio": "2:1 sampled hard set (comparable to Exp 1); plus full land grid May-Sep",
            "features": "dynamic t-10..t-1; no burned_areas/ignition_points/number_of_fires; no same-day fields",
            "cell_km2": CELL_KM2,
            "fwi_day_p75_train": fwi_day_p75,
            "xgb_best_iteration": best_iter,
            "thr_xgb_exp1": exp1_thr_xgb,
            "thr_fwi_exp1": exp1_thr_fwi,
            "thr_xgb_hard_val": thr_xgb_hard,
            "thr_fwi_hard_val": thr_fwi_hard,
        },
        "sampled_hard": sampled,
        "full_summer_grid": grid_report,
        "vs_experiment1": comparison,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", OUT / "metrics.json", flush=True)
    for split in ("test2020", "test2021"):
        for model in ("XGBoost", "FWI"):
            b = sampled[split][model]
            d = comparison[split][model]
            print(
                f"{model:8} {split} hard PR={b['pr_auc']:.3f} (Δ {d['d_pr_auc']:+.3f}) "
                f"ROC={b['roc_auc']:.3f} (Δ {d['d_roc_auc']:+.3f}) Brier={b['brier']:.3f}",
                flush=True,
            )
    for yr, blk in grid_report.items():
        for model in ("XGBoost", "FWI"):
            b = blk[model]
            op = b["at_exp1_val_threshold"]
            print(
                f"GRID {yr} {model:8} PR={b['pr_auc']:.4f} ROC={b['roc_auc']:.3f} "
                f"P={op['precision']:.4f} R={op['recall']:.3f} FPR={op['fpr']:.3f} "
                f"FA/1000km2/day={op.get('fa_per_1000km2_per_day')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
