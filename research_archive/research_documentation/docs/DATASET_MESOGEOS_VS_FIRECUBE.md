# Mesogeos vs FireCube — acquisition and starting-benchmark decision

**Date:** 2026-09-12  
**Action taken:** cloned official code; probed downloads; **did not train models**; **did not pull the 648 GB cube or the 250 GB npy dump**.  
**Recommendation:** start with **FireCube** (Zenodo NetCDF + GRL protocol). Treat **Mesogeos Track A** as the second, larger Mediterranean benchmark once Google Drive access is set up.

---

## What we actually acquired locally

| Item | Path | Status |
| --- | --- | --- |
| Mesogeos **code** (NeurIPS 2023) | `data/external/mesogeos-repo/` | Cloned |
| FireCube / GRL **training code** | `data/external/wildfire_forecasting-repo/` | Cloned |
| Mesogeos **full zarr cube** (~648 GB) | Google Drive only | **Not downloaded** (size + Drive 403 from this machine) |
| Mesogeos **Track A CSVs** | Same Drive `ml_tracks/` | **Not downloaded** (need rclone / your Google account) |
| FireCube **dataset_greece.nc** | Zenodo 6475592 | **Available** HTTP 200, `content-length` = **23 501 672 194** bytes (~21.9 GiB) |
| GRL extracted **npy samples** | Zenodo 6528394 `datasets.tar.gz` | **Available** HTTP 200, **17 023 818 559** bytes compressed; README says **≥250 GB** uncompressed |

Mesogeos public S3/OVH is **dead** ([issue #4](https://github.com/Orion-AI-Lab/mesogeos/issues/4), Jan 2025). Maintainers say use **rclone** on [this Drive folder](https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9). Automated fetch from this environment returned **403**.

---

## Mesogeos (inspected from paper + repo)

- **What it is:** Mediterranean 1 km × 1 day **zarr datacube**, 2006-04-01 → 2022-09-29, bbox `(-10.72, 30.07, 36.74, 47.7)`, EPSG:4326. ~4714 × 1753 km, 6026 days, **27 driver variables + burned areas + ignitions**.
- **On disk:** 648 GB compressed zarr; ~3.2 TB if fully loaded as float32 (paper). Daily chunks — xarray can lazy-load, but you still need the files.
- **Sources:** MODIS LST/NDVI/LAI, ERA5-Land (t2m, d2m, rh, wind, sp, ssrd, tp), JRC EDO soil moisture index, WorldPop population + distance to roads, C3S land-cover fractions, EU-DEM (elev, slope, aspect, curvature), **EFFIS** burned areas and ignitions.
- **No FWI variable** in the cube (confirmed in README table and Track A `config_train.json`).
- **Track A (danger) — the only Mesogeos track that matches our objective:**
  - Binary classification: fire that becomes **≥ 30 ha**.
  - Positive = **ignition point** of that fire (not every burned cell).
  - Features: **30 days** before ignition; ignition date shifted **−1 day** so inputs are strictly pre-event.
  - Restricted to WWF biome 12, Palearctic (Mediterranean forests/woodlands).
  - Negatives: 2× positives, land-cover-matched.
  - Loss: CE × log(burned area ha).
  - Loader: `positives.csv` / `negatives.csv`; 12 dynamic + 13 static features.
  - **Split in code** (`datasets/dataset.py`): train years except 2020–2022; **val = 2020**; **test = 2021–2022**. (Docstring still says 2009–2018 / 2019 / 2020 — stale.)
  - Leaderboard (site): LSTM / Transformer / GTN **AUPRC ~0.85–0.87** (near tie).
  - Hardware they used: Ubuntu 20.04, **RTX 3080**.
- **Track B** is **final burned area given ignition** — spread-like, **not** pre-ignition EWS.
- **License:** CC BY 4.0.

## FireCube (inspected from Zenodo + GRL repo)

- **What it is:** **Greece-centred** 1 km × 1 day NetCDF, 2009–2021, 1253 × 983 km (Greece + parts of Balkans / W. Turkey). **~90 variables**, including ERA5-Land, MODIS, soil moisture, topography, land cover, population, roads, **historical fire layers, and FWI**.
- **On disk:** one file `dataset_greece.nc` **23.5 GB** (Zenodo [6475592](https://doi.org/10.5281/zenodo.6475592)). Older 2009–2020 cut: 10.5 GB ([4943354](https://doi.org/10.5281/zenodo.4943354)) — note that version **shifts fire, meteo, and FWI one day back** for “perfect-prog” convenience.
- **GRL task (Kondylatos 2022):** next-day danger; positive = **cells that burn in a fire that starts that day and becomes >30 ha**; 10-day history; ConvLSTM uses **25 km patches**; negatives from **days with no fire in the whole domain**, land-cover stratified, ~2:1.
- **Published split:** train 2009–2018, val 2019, test **2020 (typical)** and **2021 (extreme)**. Code uses val 2020 / test 2021 for the npy dump (check when using).
- **Published comparison we need:** RF, XGBoost, LSTM, ConvLSTM, **and FWI**. FWI is **in the cube** but **commented out** of the LSTM feature list (`configs/config.yaml`) — used as an **index baseline**, not an ML input. That is the right design for our H1.
- **Extracted GRL samples:** [Zenodo 6528394](https://doi.org/10.5281/zenodo.6528394), 17 GB `.tar.gz` → **≥250 GB** npy. Code: [wildfire_forecasting](https://github.com/Orion-AI-Lab/wildfire_forecasting). Authors: 128 GB RAM, RTX 3080, GPU recommended.
- **License:** CC BY 4.0.

Same lab (Orion / NOA). Mesogeos is the **Mediterranean generalization** of the FireCube idea; FireCube is the **GRL paper’s dataset**.

---

## Comparison against our project

| Criterion | Mesogeos | FireCube | Winner for *first* experiments |
| --- | --- | --- | --- |
| **Objective (pre-ignition danger)** | Track A yes; Track B no | GRL task is exactly next-day large-fire danger | Tie (both valid) |
| **Labels** | EFFIS ignition **points** ≥30 ha | EFFIS burned **cells** of large fires, ignition date from FIRMS | FireCube closer to “which cells are dangerous”; Mesogeos closer to “will a large fire start here” |
| **Horizon** | Implicit 1-day (day-of ignition, features t−1…t−30) | Explicit **next day** | FireCube slightly cleaner for H∈{1,3,5} ablation on a cube |
| **Features** | 27 drivers; **no FWI** | ~90 including **FWI** | **FireCube** (required FWI baseline) |
| **Geography** | Whole Mediterranean (geo holdout possible) | Greece-centred only | **Mesogeos** for later H4 |
| **Years** | 2006–2022 | 2009–2021 | Mesogeos slightly longer |
| **History length in official track** | **30 days** | **10 days** | Mesogeos better for 7/10/14 ablation without re-extracting |
| **Access** | Drive + rclone; S3 dead; 403 here | Zenodo HTTP 200, stable | **FireCube** |
| **Size to start** | 648 GB cube, or unknown CSV track | **23.5 GB** nc (lazy) or 250 GB npy | **FireCube nc** |
| **Reproducibility** | Official Track A train/test scripts + leaderboard | GRL numbers + `wildfire_forecasting` + RF notebook | **FireCube** for “beat published LSTM/XGB/FWI” |
| **Compute (official DL)** | LSTM/Transformer, batch 256, RTX 3080 | LSTM/ConvLSTM, same class GPU; npy dump huge | FireCube **nc + our own 1D samples** cheaper than 250 GB npy |
| **Planned experiments** | Strong for geo holdout + 30-d lag + leaderboard | Strong for FWI vs ML, extreme 2021, ConvLSTM spatial | FireCube first; Mesogeos second |
| **Ops / user AOI** | Still Mediterranean-only; not global | Even narrower | Neither is a global EWS cube |

---

## Recommendation

**Do not start with the full Mesogeos zarr.** It is the right *eventual* Mediterranean benchmark, not the right *first* download.

**Use FireCube first**, because:

1. Our first scientific tests are **FWI vs XGBoost vs LSTM** on the **same labels/split**. FWI is in FireCube and already compared in GRL. Mesogeos would require computing FWI ourselves before that test is fair.
2. Download is **one 23.5 GB Zenodo file** we already confirmed is live — unlike Mesogeos Drive.
3. Published **2021 extreme-year** holdout matches our temporal-generalisation plan.
4. We can **lazy-open the NetCDF** and extract 1 km time series for H=1/3 without the 250 GB npy ConvLSTM dump.
5. Mesogeos Track A is still **ignition-point classification** on a **2:1 sampled** set — same balanced-F1 trap as GRL. FireCube at least lets us later score a **full Greece grid** from the nc.

**Then Mesogeos Track A** (Drive `ml_tracks/` CSVs + official configs) when you can rclone:

- official NeurIPS leaderboard;
- multi-country **geographic holdout**;
- **30-day** sequences for history ablation;
- still Mediterranean, still ≥30 ha, still sampled negatives.

**Do not** decompress `datasets.tar.gz` unless you have **~250 GB free** and need bit-for-bit ConvLSTM reproduction.

---

## Next acquisition steps (no training)

1. Download FireCube cube (when you confirm disk):  
   `https://zenodo.org/records/6475592/files/dataset_greece.nc`
2. Open with xarray (`chunks={}`) and list variables, time, lat/lon, missingness — **inspect only**.
3. On your Google account: rclone the Drive folder; **prefer `ml_tracks/` over `mesogeos_cube.zarr/`** first.
4. Only after inspection: freeze H and extract samples.

Same authors, same 1 km daily EFFIS-based danger idea. FireCube is the **reproducible, FWI-ready, downloadable** starting cube. Mesogeos is the **wider benchmark** to move to once Drive works.
