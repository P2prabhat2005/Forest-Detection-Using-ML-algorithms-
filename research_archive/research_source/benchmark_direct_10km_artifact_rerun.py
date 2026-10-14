"""Exp5 artifact-recovery rerun: identical methodology, durable persistence only.

Does NOT write to results/firecube_direct_10km/ (original Exp5 results untouched).
Does NOT change units, labels, lags, features, splits, thresholding, or eval protocol.
Re-uses helpers and constants from src/benchmark_direct_10km.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import joblib
import netCDF4 as nc
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

import benchmark_direct_10km as e5

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "firecube_direct_10km_artifact_rerun"
ORIG_METRICS = ROOT / "results" / "firecube_direct_10km" / "metrics.json"
CONFIG_PATH = OUT / "experiment_config.json"

# Re-export Exp5 constants (methodology lock)
HISTORY = e5.HISTORY
FIRE_MONTHS = e5.FIRE_MONTHS
SEED = e5.SEED
TRAIN_NEG_POS_RATIO = e5.TRAIN_NEG_POS_RATIO
DYNAMIC = e5.DYNAMIC
DYNAMIC_AGG = e5.DYNAMIC_AGG
STATIC = e5.STATIC
STATIC_AGG = e5.STATIC_AGG
NC_PATH = e5.NC_PATH
EXP4_METRICS = e5.EXP4_METRICS


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
    """Write npz atomically. Temp name must end in .npz so numpy does not append another .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing.npz")
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, **arrays)
    if not tmp.exists() or tmp.stat().st_size <= 0:
        raise RuntimeError(f"atomic_savez temp missing after write: {tmp}")
    os.replace(tmp, path)
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"atomic_savez failed: {path}")
    print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def atomic_joblib(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    joblib.dump(obj, tmp)
    os.replace(tmp, path)
    loaded = joblib.load(path)
    if "xgb" not in loaded:
        raise RuntimeError(f"joblib missing xgb after write: {path}")
    print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_predictions(path_npz: Path, path_csv: Path, unit_id, t_idx, y_true, y_score, dates_str):
    unit_id = np.asarray(unit_id, dtype=np.int32)
    t_idx = np.asarray(t_idx, dtype=np.int32)
    y_true = np.asarray(y_true, dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float64)
    dates_str = np.asarray(dates_str)
    n = len(y_true)
    if not (len(unit_id) == len(t_idx) == len(y_score) == len(dates_str) == n):
        raise RuntimeError(
            f"prediction length mismatch: uid={len(unit_id)} t={len(t_idx)} "
            f"y={n} score={len(y_score)} date={len(dates_str)}"
        )
    atomic_savez(
        path_npz,
        unit_id=unit_id,
        t=t_idx,
        date=dates_str.astype(str),
        y_true=y_true,
        y_score=y_score.astype(np.float32),
    )
    # CSV for B/C convenience
    tmp = path_csv.with_suffix(path_csv.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write("unit_id,date,t,y_true,y_score\n")
        for i in range(n):
            f.write(
                f"{int(unit_id[i])},{dates_str[i]},{int(t_idx[i])},"
                f"{int(y_true[i])},{float(y_score[i]):.8g}\n"
            )
    os.replace(tmp, path_csv)
    print(f"  saved {path_csv} rows={n}", flush=True)


def write_manifest(files: list[Path]) -> dict:
    man = {"output_dir": str(OUT), "files": {}}
    missing = []
    for p in files:
        if p.exists() and p.stat().st_size > 0:
            man["files"][p.name] = {
                "path": str(p),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        else:
            man["files"][p.name] = {"path": str(p), "missing": True}
            missing.append(p.name)
    man["all_present"] = len(missing) == 0
    man["missing"] = missing
    dest = OUT / "artifacts_manifest.json"
    atomic_write_text(dest, json.dumps(man, indent=2))
    print(f"  wrote {dest}", flush=True)
    return man


def assert_no_same_day_leakage():
    """Documented + structural check: dyn window is t-10..t-1 via HISTORY buffer."""
    s0, e0 = e5.window_slice()
    # HISTORY-1 is last buffer slot = t-1 when cur_t==t; window uses last 10 of HISTORY
    assert HISTORY == 16
    assert s0 == 6 and e0 == 16, (s0, e0)
    # last day in dyn10 corresponds to buffer index HISTORY-1 => day t-1
    return {
        "history_buffer_days": HISTORY,
        "dyn_window_slice": [s0, e0],
        "dyn_window_means": "t-10 ... t-1 relative to label day t",
        "label_uses_same_day_burned_areas": True,
        "predictors_use_same_day_fields": False,
        "check": "PASS — feature window excludes day t (matches Exp5)",
    }


def integrity_check(
    feat_path: Path,
    model_path: Path,
    pred_paths: dict[str, Path],
    metrics: dict,
    data: dict,
) -> dict:
    errors = []
    required = [
        feat_path,
        model_path,
        OUT / "metrics.json",
        OUT / "experiment_config.json",
        OUT / "feature_aggregation.json",
        OUT / "artifacts_manifest.json",
        pred_paths["val"],
        pred_paths["2020"],
        pred_paths["2021"],
        pred_paths["val"].with_suffix(".csv"),
        pred_paths["2020"].with_suffix(".csv"),
        pred_paths["2021"].with_suffix(".csv"),
    ]
    for p in required:
        if not p.exists() or p.stat().st_size <= 0:
            errors.append(f"missing or empty: {p.name}")

    # Row counts vs metrics
    m = metrics["models"]["XGBoost_direct_10km"]
    for split, key in (("val", "val"), ("test2020", "2020"), ("test2021", "2021")):
        n_feat = int(len(data[f"{split}__y"]))
        n_met = int(m[key]["n"])
        n_pos_feat = int(data[f"{split}__y"].sum())
        n_pos_met = int(m[key]["n_pos"])
        if n_feat != n_met:
            errors.append(f"{split}: feature n={n_feat} != metrics n={n_met}")
        if n_pos_feat != n_pos_met:
            errors.append(f"{split}: feature n_pos={n_pos_feat} != metrics n_pos={n_pos_met}")
        z = np.load(pred_paths[key if key != "val" else "val"])
        for col in ("unit_id", "t", "date", "y_true", "y_score"):
            if col not in z.files:
                errors.append(f"{pred_paths[key if key != 'val' else 'val'].name} missing {col}")
        if "y_true" in z.files and len(z["y_true"]) != n_met:
            errors.append(
                f"{split}: pred rows={len(z['y_true'])} != metrics n={n_met}"
            )
        if "y_true" in z.files and int(z["y_true"].sum()) != n_pos_met:
            errors.append(
                f"{split}: pred n_pos={int(z['y_true'].sum())} != metrics n_pos={n_pos_met}"
            )

    # Optional: compare to original Exp5 metrics n / n_pos (reproducibility target)
    orig_cmp = {}
    if ORIG_METRICS.exists():
        orig = json.loads(ORIG_METRICS.read_text(encoding="utf-8"))
        om = orig["models"]["XGBoost_direct_10km"]
        for key in ("val", "2020", "2021"):
            orig_cmp[key] = {
                "orig_n": om[key]["n"],
                "rerun_n": m[key]["n"],
                "n_match": om[key]["n"] == m[key]["n"],
                "orig_n_pos": om[key]["n_pos"],
                "rerun_n_pos": m[key]["n_pos"],
                "n_pos_match": om[key]["n_pos"] == m[key]["n_pos"],
            }
            if not orig_cmp[key]["n_match"] or not orig_cmp[key]["n_pos_match"]:
                errors.append(
                    f"vs original Exp5 {key}: n/n_pos mismatch "
                    f"(orig {om[key]['n']}/{om[key]['n_pos']} vs "
                    f"rerun {m[key]['n']}/{m[key]['n_pos']})"
                )

    leak = assert_no_same_day_leakage()
    report = {
        "ok": len(errors) == 0,
        "errors": errors,
        "leakage_check": leak,
        "vs_original_exp5_metrics": orig_cmp,
    }
    atomic_write_text(OUT / "integrity_report.json", json.dumps(report, indent=2))
    if errors:
        raise RuntimeError("INTEGRITY FAILED:\n  - " + "\n  - ".join(errors))
    print("INTEGRITY PASS", flush=True)
    return report


def write_experiment_config(units_meta: dict):
    cfg = {
        "experiment": "5_artifact_rerun",
        "purpose": "Recover Exp5 artifacts for failure-analysis B+C; methodology identical to Exp5",
        "does_not_modify": [
            "results/firecube_direct_10km/",
            "Exp1-Exp4 result folders",
            "dataset_greece.nc",
        ],
        "output_dir": str(OUT),
        "parent_script": "src/benchmark_direct_10km.py",
        "methodology_locked": {
            "spatial_units": "~10 km rectangular blocks; stride from 10km/DY_KM, 10km/DX_KM",
            "label": "unit positive on day t if any fire-start burned_areas cell inside unit on day t",
            "temporal_predictors": "dynamic t-10 ... t-1 only; no same-day predictors",
            "features": "mean_over_10d(dyn)+last_day(t-1)+static5+clc10; FWI baseline = unit max t-1",
            "train": "2009-2018 May-Sep; all positives + 20:1 random negatives; SEED=1234",
            "validation": "2019 May-Sep FULL unit-day grid",
            "test": "2020 and 2021 May-Sep FULL unit-day grids",
            "threshold": "F1-max on 2019 unit-days only (linspace 0.05..0.95, 37 steps)",
            "model": "XGBClassifier identical hyperparams to Exp5",
            "seed": SEED,
            "train_neg_pos_ratio": TRAIN_NEG_POS_RATIO,
            "history": HISTORY,
            "fire_months": sorted(FIRE_MONTHS),
        },
        "spatial_units_resolved": units_meta,
        "required_artifacts": [
            "unit_day_features.npz",
            "xgb_direct_10km.joblib",
            "validation_predictions.npz",
            "validation_predictions.csv",
            "test_predictions_2020.npz",
            "test_predictions_2020.csv",
            "test_predictions_2021.npz",
            "test_predictions_2021.csv",
            "artifacts_manifest.json",
            "metrics.json",
            "experiment_config.json",
            "feature_aggregation.json",
            "integrity_report.json",
            "run.log",
        ],
        "changes_vs_exp5": [
            "output directory only",
            "atomic save + SHA-256 manifest",
            "explicit prediction exports (unit_id, date, y_true, y_score)",
            "hard integrity checks before success",
        ],
    }
    atomic_write_text(CONFIG_PATH, json.dumps(cfg, indent=2))
    return cfg


def main():
    if OUT.resolve() == e5.OUT.resolve():
        raise RuntimeError("Refusing to write into original Exp5 OUT path")

    OUT.mkdir(parents=True, exist_ok=True)
    # Never touch original Exp5 folder
    assert not str(OUT).endswith("firecube_direct_10km") or "artifact_rerun" in str(OUT)

    feat_path = OUT / "unit_day_features.npz"
    meta_path = OUT / "feature_aggregation.json"
    model_path = OUT / "xgb_direct_10km.joblib"
    pred_val = OUT / "validation_predictions.npz"
    pred_2020 = OUT / "test_predictions_2020.npz"
    pred_2021 = OUT / "test_predictions_2021.npz"

    print("Opening cube...", flush=True)
    ds = nc.Dataset(NC_PATH, "r")
    dates = nc.num2date(
        ds.variables["time"][:], ds.variables["time"].units, only_use_cftime_datetimes=False
    )
    years = np.array([d.year for d in dates], dtype=np.int16)
    months = np.array([d.month for d in dates], dtype=np.int16)
    date_str = np.array([f"{d.year:04d}-{d.month:02d}-{d.day:02d}" for d in dates])

    land = e5.filled(ds.variables["CLC_2018_9"]) < 0.5
    land_y, land_x = np.where(land)
    sy10 = max(1, int(round(10.0 / e5.DY_KM)))
    sx10 = max(1, int(round(10.0 / e5.DX_KM)))
    units = e5.build_block_units(land_y, land_x, sy10, sx10, "10km")
    uid = units["unit_id"]
    n_u = units["n_units"]
    areas = units["area_km2"]
    print(
        f"10km units={n_u} stride=({sy10},{sx10}) mean_area={areas.mean():.2f} km2 "
        f"extent~({units['mean_extent_km']['y']:.2f}x{units['mean_extent_km']['x']:.2f}) km",
        flush=True,
    )

    write_experiment_config({
        "stride_cells": {"y": sy10, "x": sx10},
        "n_units": n_u,
        "mean_extent_km": units["mean_extent_km"],
        "mean_area_km2": float(areas.mean()),
        "cell_km2": e5.CELL_KM2,
    })

    static_maps = {name: e5.filled(ds.variables[name]) for name in STATIC}
    clc_maps = np.stack([e5.filled(ds.variables[f"CLC_2018_{k}"]) for k in range(10)], axis=0)
    pop_maps = {yr: e5.filled(ds.variables[f"POP_DENS_{yr}"]) for yr in range(2009, 2022)}

    sta_unit_base = np.stack([
        e5.agg_land_to_units(static_maps["DEM"][land], uid, n_u, STATIC_AGG["DEM"]),
        e5.agg_land_to_units(static_maps["SLOPE"][land], uid, n_u, STATIC_AGG["SLOPE"]),
        e5.agg_land_to_units(static_maps["ROAD_DISTANCE"][land], uid, n_u, STATIC_AGG["ROAD_DISTANCE"]),
        e5.agg_land_to_units(static_maps["WATERWAY_DISTANCE"][land], uid, n_u, STATIC_AGG["WATERWAY_DISTANCE"]),
    ], axis=1).astype(np.float32)
    clc_unit = np.stack([
        e5.agg_land_to_units(clc_maps[k][land], uid, n_u, "mean") for k in range(10)
    ], axis=1).astype(np.float32)
    pop_unit = {
        yr: e5.agg_land_to_units(pop_maps[yr][land], uid, n_u, STATIC_AGG["POP_DENS"])
        for yr in range(2009, 2022)
    }

    agg_doc = {
        "spatial_units": {
            "scheme": "same as Experiment 4 ~10 km rectangular blocks",
            "stride_cells": {"y": sy10, "x": sx10},
            "mean_extent_km": units["mean_extent_km"],
            "n_units": n_u,
            "mean_area_km2": float(areas.mean()),
            "cell_km2": e5.CELL_KM2,
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
        "note": "Artifact rerun — identical to Exp5 methodology; writes only to artifact_rerun folder",
        "artifact_rerun": True,
    }
    atomic_write_text(meta_path, json.dumps(agg_doc, indent=2))
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
                buf_dyn[k, v] = e5.agg_land_to_units(
                    e5.filled(var, tt)[land], uid, n_u, DYNAMIC_AGG[DYNAMIC[v]]
                )
            buf_fwi[k] = e5.agg_land_to_units(e5.filled(fwi_var, tt)[land], uid, n_u, "max")
        head = 0
        cur_t = t0

        for di, t in enumerate(days):
            t = int(t)
            while cur_t < t:
                for v, var in enumerate(dyn_vars):
                    buf_dyn[head, v] = e5.agg_land_to_units(
                        e5.filled(var, cur_t)[land], uid, n_u, DYNAMIC_AGG[DYNAMIC[v]]
                    )
                buf_fwi[head] = e5.agg_land_to_units(
                    e5.filled(fwi_var, cur_t)[land], uid, n_u, "max"
                )
                head = (head + 1) % HISTORY
                cur_t += 1

            # Buffer holds t-HISTORY ... t-1 when cur_t == t (leakage-safe).
            dyn_u = np.empty((n_u, HISTORY, len(DYNAMIC)), np.float32)
            fwi_u = np.empty((n_u, HISTORY), np.float32)
            for k in range(HISTORY):
                idx = (head + k) % HISTORY
                dyn_u[:, k, :] = buf_dyn[idx].T
                fwi_u[:, k] = buf_fwi[idx]

            ba = e5.filled(ba_var, t)
            y_cell = (ba[land] > 0).astype(np.float32)
            y_u = np.zeros(n_u, dtype=np.float32)
            np.maximum.at(y_u, uid, y_cell)

            yr = int(years[t])
            sp = e5.split_name(yr)
            if sp is None:
                continue
            s0, e0 = e5.window_slice()
            dyn10 = dyn_u[:, s0:e0, :]
            fwi_last = fwi_u[:, s0:e0][:, -1]
            sta = np.concatenate([sta_unit_base, pop_unit[yr][:, None]], axis=1)

            if sp == "train":
                pos_idx = np.flatnonzero(y_u > 0)
                neg_idx = np.flatnonzero(y_u == 0)
                n_pos = len(pos_idx)
                n_neg_keep = min(len(neg_idx), n_pos * TRAIN_NEG_POS_RATIO)
                if n_pos == 0:
                    n_neg_keep = min(len(neg_idx), 50)
                    keep = (
                        rng.choice(neg_idx, size=n_neg_keep, replace=False)
                        if n_neg_keep else np.array([], dtype=int)
                    )
                else:
                    neg_keep = (
                        rng.choice(neg_idx, size=n_neg_keep, replace=False)
                        if n_neg_keep else np.array([], dtype=int)
                    )
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
        atomic_savez(feat_path, **data)

    ds.close()

    print("Training direct 10 km XGBoost (Exp5-identical)...", flush=True)
    s0, e0 = e5.window_slice()

    def prep(split, scaler_dyn=None, scaler_sta=None):
        dyn = np.asarray(data[f"{split}__dyn"], dtype=np.float32)
        dyn = e5.fill_time_nans(dyn)
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
        x = e5.xgb_features(dyn_n, sta_n, clc)
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
        "train_neg_pos_ratio": TRAIN_NEG_POS_RATIO,
        "feature_agg": json.loads(meta_path.read_text(encoding="utf-8")),
        "artifact_rerun": True,
    }
    atomic_joblib(model_path, art)

    def fwi_p(arr):
        return np.clip((arr - fwi_min) / max(fwi_max - fwi_min, 1e-6), 0, 1)

    p_va = xgb.predict_proba(x_va)[:, 1]
    thr_xgb, f1_va = e5.best_f1_threshold(y_va, p_va)
    thr_fwi, _ = e5.best_f1_threshold(y_va, fwi_p(fwi_va))
    print(f"  2019-tuned thr XGB={thr_xgb:.3f} (F1={f1_va:.4f}) FWI={thr_fwi:.3f}", flush=True)

    n_units = int(data["n_units"][0]) if "n_units" in data else n_u
    area_lookup = data["area_km2"] if "area_km2" in data else areas
    land_km2_eval = float(np.sum(area_lookup))

    results = {
        "protocol": {
            "experiment": "5_artifact_rerun",
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
            "artifact_rerun": True,
            "does_not_modify_original_exp5_dir": True,
        },
        "models": {},
        "comparison": {},
        "verdict_notes": {},
    }

    pred_map = {"val": pred_val, "2020": pred_2020, "2021": pred_2021}

    for split, yr_key in (
        ("val", "val"),
        ("test2020", "2020"),
        ("test2021", "2021"),
    ):
        x, y, fwi, _, _ = prep(split, scaler_dyn, scaler_sta)
        px = xgb.predict_proba(x)[:, 1]
        pf = fwi_p(fwi)
        n_days = int(len(np.unique(data[f"{split}__t"])))
        unit_areas = data[f"{split}__area"]
        block_x = e5.pack_eval(y, px, thr_xgb, unit_areas, n_days, land_km2_eval)
        block_f = e5.pack_eval(y, pf, thr_fwi, unit_areas, n_days, land_km2_eval)
        block_x["n_days"] = n_days
        block_f["n_days"] = n_days
        block_x["n_units"] = n_units
        block_f["n_units"] = n_units
        block_x["n_pos_unit_days"] = int(y.sum())
        block_f["n_pos_unit_days"] = int(y.sum())
        if "XGBoost_direct_10km" not in results["models"]:
            results["models"]["XGBoost_direct_10km"] = {"val_threshold": thr_xgb}
            results["models"]["FWI_direct_10km_max"] = {
                "val_threshold": thr_fwi,
                "note": "unit max of t-1 FWI",
            }
        results["models"]["XGBoost_direct_10km"][yr_key] = block_x
        results["models"]["FWI_direct_10km_max"][yr_key] = block_f

        t_idx = data[f"{split}__t"]
        write_predictions(
            pred_map[yr_key],
            pred_map[yr_key].with_suffix(".csv"),
            data[f"{split}__uid"],
            t_idx,
            y,
            px,
            date_str[t_idx],
        )
        print(
            f"  {split} directXGB PR={block_x['pr_auc']:.5f} ROC={block_x['roc_auc']:.3f} "
            f"P={block_x['at_val_tuned']['precision']:.5f} R={block_x['at_val_tuned']['recall']:.3f}",
            flush=True,
        )

    print("Loading Exp4 10 km baselines...", flush=True)
    exp4 = json.loads(EXP4_METRICS.read_text(encoding="utf-8"))
    results["models"]["A_XGB_easy_1km_then_10km_max"] = e5.extract_slice_from_exp4(
        exp4, "XGBoost_easy__max"
    )
    results["models"]["B_XGB_hard_1km_then_10km_max"] = e5.extract_slice_from_exp4(
        exp4, "XGBoost_hard__max"
    )
    results["models"]["C_FWI_1km_then_10km_max"] = e5.extract_slice_from_exp4(exp4, "FWI__max")

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

    out_json = OUT / "metrics.json"
    atomic_write_text(out_json, json.dumps(results, indent=2))
    print(f"Wrote {out_json}", flush=True)

    required_for_manifest = [
        feat_path,
        model_path,
        meta_path,
        CONFIG_PATH,
        out_json,
        pred_val,
        pred_val.with_suffix(".csv"),
        pred_2020,
        pred_2020.with_suffix(".csv"),
        pred_2021,
        pred_2021.with_suffix(".csv"),
    ]
    write_manifest(required_for_manifest)

    integrity_check(
        feat_path,
        model_path,
        {"val": pred_val, "2020": pred_2020, "2021": pred_2021},
        results,
        data,
    )

    # Final manifest including integrity report
    write_manifest(required_for_manifest + [OUT / "integrity_report.json", OUT / "artifacts_manifest.json"])
    print(f"Artifact rerun COMPLETE → {OUT}", flush=True)


if __name__ == "__main__":
    main()
