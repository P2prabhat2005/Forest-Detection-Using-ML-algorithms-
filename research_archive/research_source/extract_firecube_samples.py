"""Leakage-safe FireCube sample extraction. Does not modify dataset_greece.nc."""
from __future__ import annotations

import json
from pathlib import Path

import netCDF4 as nc
import numpy as np

NC_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "firecube" / "dataset_greece.nc"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "firecube_samples.npz"
META_PATH = OUT_PATH.with_suffix(".json")

HISTORY = 16  # days t-16 ... t-1, so lead times 1,3,5,7 are possible
NEG_POS_RATIO = 2
SEED = 1234
PIXELS_PER_NEG_DAY = 80

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


def clc_year(year: int) -> int:
    if year <= 2011:
        return 2006
    if year <= 2017:
        return 2012
    return 2018


def pop_year(year: int) -> int:
    return int(np.clip(year, 2009, 2021))


def split_name(year: int) -> str:
    if year <= 2018:
        return "train"
    if year == 2019:
        return "val"
    if year == 2020:
        return "test2020"
    return "test2021"


def filled(var, sl=None):
    data = var[:] if sl is None else var[sl]
    return np.ma.filled(np.ma.array(data), np.nan).astype(np.float32, copy=False)


def main():
    rng = np.random.default_rng(SEED)
    print(f"Opening {NC_PATH}", flush=True)
    ds = nc.Dataset(NC_PATH, "r")
    times = np.array(ds.variables["time"][:])
    # CF days since 1970 or datetime already numeric; use netCDF4 num2date
    dates = nc.num2date(ds.variables["time"][:], ds.variables["time"].units, only_use_cftime_datetimes=False)
    years = np.array([d.year for d in dates], dtype=np.int16)
    nof = filled(ds.variables["number_of_fires"]).astype(np.int16)
    ny, nx = ds.variables["DEM"].shape
    land = filled(ds.variables["CLC_2018_9"]) < 0.5
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

    print("Scanning burned cells...", flush=True)
    positives = []  # (t, y, x)
    ba_var = ds.variables["burned_areas"]
    fire_days = np.where(nof > 0)[0]
    for t in fire_days:
        if t < HISTORY:
            continue
        ba = filled(ba_var, t)
        yy, xx = np.where((ba > 0) & land)
        for y, x in zip(yy.tolist(), xx.tolist()):
            positives.append((int(t), int(y), int(x)))
    positives = np.array(positives, dtype=np.int32)
    print(f"positives: {len(positives)}", flush=True)

    pos_by_split: dict[str, list] = {"train": [], "val": [], "test2020": [], "test2021": []}
    for t, y, x in positives:
        pos_by_split[split_name(int(years[t]))].append((t, y, x))

    nofire = np.where((nof == 0) & (np.arange(len(nof)) >= HISTORY))[0]
    nofire_by_split = {k: [] for k in pos_by_split}
    for t in nofire:
        nofire_by_split[split_name(int(years[t]))].append(int(t))

    samples = []  # t,y,x,label
    for split, pos in pos_by_split.items():
        n_pos = len(pos)
        n_neg = n_pos * NEG_POS_RATIO
        print(f"{split}: {n_pos} pos, sampling {n_neg} neg", flush=True)
        for item in pos:
            samples.append((*item, 1))
        if n_pos == 0:
            continue
        # CLC-stratified negatives
        clc_ids = []
        for t, y, x in pos:
            clc_ids.append(int(clc_mode[clc_year(int(years[t]))][y, x]))
        classes, counts = np.unique(clc_ids, return_counts=True)
        days = np.array(nofire_by_split[split], dtype=np.int32)
        if len(days) == 0:
            raise RuntimeError(f"No negative days for {split}")
        for c, cnt in zip(classes, counts):
            need = int(cnt) * NEG_POS_RATIO
            # cells of this CLC that are land
            # use CLC_2018 mode for cell eligibility (stable); value match on year-specific later
            cells_y, cells_x = np.where(land & (clc_mode[2018] == c))
            if cells_y.size == 0:
                cells_y, cells_x = np.where(land)
            picked = 0
            guard = 0
            while picked < need and guard < need * 20:
                guard += 1
                t = int(rng.choice(days))
                n_pix = min(PIXELS_PER_NEG_DAY, need - picked)
                idx = rng.integers(0, cells_y.size, size=n_pix)
                for j in idx:
                    samples.append((t, int(cells_y[j]), int(cells_x[j]), 0))
                    picked += 1
                    if picked >= need:
                        break

    samples = np.array(samples, dtype=np.int32)
    rng.shuffle(samples)
    print(f"total samples {len(samples)} unique days {len(np.unique(samples[:, 0]))}", flush=True)

    by_t: dict[int, list[int]] = {}
    for i, (t, y, x, lab) in enumerate(samples):
        by_t.setdefault(int(t), []).append(i)

    n = len(samples)
    dyn = np.full((n, HISTORY, len(DYNAMIC)), np.nan, dtype=np.float32)
    fwi = np.full((n, HISTORY), np.nan, dtype=np.float32)
    sta = np.full((n, 5), np.nan, dtype=np.float32)
    clc = np.full((n, 10), np.nan, dtype=np.float32)
    y_lab = samples[:, 3].astype(np.int8)
    year_arr = years[samples[:, 0]]
    t_arr = samples[:, 0]
    y_arr = samples[:, 1]
    x_arr = samples[:, 2]

    # static / clc / pop (no time lag issues)
    print("Filling static/CLC/population...", flush=True)
    for i, (t, y, x, lab) in enumerate(samples):
        yr = int(years[t])
        sta[i, 0] = static_maps["DEM"][y, x]
        sta[i, 1] = static_maps["SLOPE"][y, x]
        sta[i, 2] = static_maps["ROAD_DISTANCE"][y, x]
        sta[i, 3] = static_maps["WATERWAY_DISTANCE"][y, x]
        sta[i, 4] = pop_maps[pop_year(yr)][y, x]
        clc[i] = clc_vec[clc_year(yr)][:, y, x]

    dyn_vars = [ds.variables[name] for name in DYNAMIC]
    fwi_var = ds.variables["fwi"]
    buf_dyn = np.full((HISTORY, len(DYNAMIC), ny, nx), np.nan, dtype=np.float32)
    buf_fwi = np.full((HISTORY, ny, nx), np.nan, dtype=np.float32)

    print("Reading circular 16-day windows...", flush=True)
    for t in range(HISTORY):
        for v, var in enumerate(dyn_vars):
            buf_dyn[t, v] = filled(var, t)
        buf_fwi[t] = filled(fwi_var, t)
    head = 0  # index of oldest day (t-HISTORY)

    needed = set(by_t)
    max_t = max(needed)
    for t in range(HISTORY, max_t + 1):
        if t in by_t:
            for i in by_t[t]:
                yy = int(samples[i, 1])
                xx = int(samples[i, 2])
                for k in range(HISTORY):
                    idx = (head + k) % HISTORY
                    dyn[i, k] = buf_dyn[idx, :, yy, xx]
                    fwi[i, k] = buf_fwi[idx, yy, xx]
        if t == max_t:
            break
        for v, var in enumerate(dyn_vars):
            buf_dyn[head, v] = filled(var, t)
        buf_fwi[head] = filled(fwi_var, t)
        head = (head + 1) % HISTORY
        if (t + 1) % 250 == 0:
            print(f"  time {t}/{max_t}", flush=True)

    ds.close()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        dynamic=dyn,
        fwi=fwi,
        static=sta,
        clc=clc,
        label=y_lab,
        year=year_arr,
        time_index=t_arr,
        y=y_arr,
        x=x_arr,
        dynamic_names=np.array(DYNAMIC),
        static_names=np.array(STATIC + ["population_density"]),
    )
    counts = {s: int((np.array([split_name(int(y)) for y in year_arr]) == s).sum()) for s in pos_by_split}
    pos_counts = {
        s: int(((np.array([split_name(int(y)) for y in year_arr]) == s) & (y_lab == 1)).sum())
        for s in pos_by_split
    }
    meta = {
        "n_samples": int(n),
        "history": HISTORY,
        "neg_pos_ratio": NEG_POS_RATIO,
        "seed": SEED,
        "dynamic": DYNAMIC,
        "protocol": "label day t; features t-16..t-1; H=1 uses t-10..t-1",
        "splits": {"train": "2009-2018", "val": "2019", "test2020": "2020", "test2021": "2021"},
        "counts": counts,
        "positive_counts": pos_counts,
        "nan_dynamic_pct": float(np.isnan(dyn).mean() * 100),
        "nan_fwi_pct": float(np.isnan(fwi).mean() * 100),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote", OUT_PATH, meta, flush=True)


if __name__ == "__main__":
    main()
