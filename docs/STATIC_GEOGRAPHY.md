# Greece Track-B compact static geography v1

Live Track-B can run from the project source and small inference artifacts
without opening `dataset_greece.nc`. Historical Track-B automatically retains
the original cube-backed static and dynamic providers when constructed with
`HistoricalCubeDynamicProvider`. Exp5 replay also retains its original cube
dependency. No historical experiment, model, calibrator or result was changed.

## Provenance and integrity

`models/production/greece_static_v1.npz` is **320,498 bytes** (about 313 KiB).
Its adjacent JSON manifest records source attributes, all variable units,
array shapes/dtypes/missing counts, generation code hashes and library versions.
Both files belong in a source checkout; only this small NPZ is excepted from
the dataset ignore rules. Runtime verifies the manifest schema, feature order,
population years, payload SHA-256, dtypes/shapes, coordinates and land mapping.
It fails if either file is missing or incompatible; it never opens the cube
or downloads alternative geography as a fallback.

Artifact SHA-256:
`66b45961241a602f5427d4b3e490feb897b3b6ada5942ad67c181e2fcc0dc574`

Source cube: `dataset_greece.nc`, **23,501,672,194 bytes**, SHA-256:
`500aec8ed808e6b1a8615cbc7300881d5c4a981a79a5f6853ee36d5985e7f9e6`

Source creators: Ioannis Prapas, Spyros Kondylatos and Ioannis Papoutsis.
Source title: *FireCube: A Daily Datacube for the Modeling and Analysis of
Wildfires in Greece*. The cube declares Creative Commons Attribution v4;
this derived extract retains attribution and documents its aggregation changes.
The local research notes identify Zenodo DOI `10.5281/zenodo.6475592`, while
the cube's embedded citation refers to the earlier `10.5281/zenodo.4943354`.
Both are preserved in the manifest. The source hash identifies the actual
local file; no publisher-checksum or citation-version verification is implied.
Embedded temporal coverage is 2009-03-06 through 2021-08-29.

## Grid, coverage and schema

Schema: `greece_track_b_static_v1`. Coordinates are EPSG:4326, longitude in
degrees east and latitude in degrees north. The grid is **983 × 1,253**, with
axis-centre bounds longitude **18.70211622302644–28.899269063907617** and latitude
**34.30110854569159–42.299194879353976**. Bounds are inclusive, followed by
nearest-centre `argmin` selection (first array index wins a tie), exactly as in
the original provider. This bounding rectangle includes excluded cells; it
does not mean all locations or all Greek territory are supported.

There are **495,032 eligible land cells**, **736,667 excluded cells**, and
**4,022 units**. Eligibility is precisely `filled(CLC_2018_9) < 0.5`.
Water and unknown/masked land-cover cells are excluded. Units use 11 rows ×
14 columns in the source grid (approximately 10 km), the existing sorted
block-key IDs, and original per-cell-area constants. They are not a new
equal-area or global grid. Nonfinite/out-of-bounds coordinates and excluded
cells produce explicit errors before external provider requests.

NPZ arrays (no Python pickles):

- `x`, `y`: float64 longitude/latitude axes, lengths 1,253 and 983.
- `land_mask`: bool array `[983,1253]`.
- `grid_unit_id`: int32 `[983,1253]`; `-1` means excluded, otherwise 0–4,021.
- `static`: float32 `[4022,4]`, in DEM, SLOPE, ROAD_DISTANCE, WATERWAY_DISTANCE order.
- `population`: float32 `[13,4022]`, years 2009 through 2021 inclusive.
- `clc`: float32 `[4022,10]`, CLC_2018 classes 0 through 9.
- `unit_area_km2`: float64 `[4022]`, preserving the original area calculation.

## Features, units and missing values

The 35-column production schema is unchanged. Columns 0–9 are dynamic
ten-day means and 10–19 are the corresponding t−1 values. Columns 20–34 are:

1. `DEM__mean`: elevation, metres.
2. `SLOPE__mean`: slope, degrees.
3. `ROAD_DISTANCE__min`: distance, metres.
4. `WATERWAY_DISTANCE__min`: distance, metres.
5. `POP_DENS__mean`: humans/km²; requested year clamped to 2009–2021.
6. `CLC_2018_0__mean` through `CLC_2018_9__mean`: dimensionless class fractions,
   preserving the source's `/100 %` unit notation and numeric values.

Means use eligible source land cells with equal cell weights, exactly as in
`agg_land_to_units`; distances use minima. No resampling, coordinate reprojection,
area reweighting or unit conversion is introduced. Source masks become float32
NaNs. Aggregation ignores nonfinite values and leaves all-missing units NaN.
The compact provider preserves these NaNs. The existing saved production
median imputer handles them and records missing features; no new imputation
policy was added. `grid_unit_id=-1` is a mapping sentinel, never a static value.

Population after 2021 and CLC from 2018 remain historical proxies. DEM/static
capture dates are unspecified where the source does not provide them. Runtime
provenance includes population year, land-cover year, schema and artifact hash.

## Reproduction and verification

Use the Python 3.12 environment installed by `setup.ps1`, constrained by
`requirements-lock.txt`. The following commands require the full authorized
local cube; they never train or modify it:

```powershell
.\.venv\Scripts\python.exe -m scripts.export_static_geography
.\.venv\Scripts\python.exe -m scripts.verify_static_geography
```

Export refuses to overwrite existing outputs unless `--overwrite` is supplied.
For independent regeneration use `--output .cache/greece_static_v1.npz` and
compare payload SHA-256. JSON creation timestamps/output names may differ.
The verifier writes `docs/static_geography_parity.json`, separate from research
results. It checks the entire coordinate grid, mask, unit IDs/areas, all unit
static/CLC aggregates and all 13 population years. It also compares the original
provider with the compact provider at representative units, historical/clamped
years and excluded water cells, then checks raw/normalized feature order and
existing-model prediction equality with identical frozen dynamic fixtures.
These fixture comparisons isolate the static substitution; they do not test
live provider accuracy or scientific validity.

Without the cube, run `python -m src.check_setup`, `python -m pytest -q -m
"not integration"`, and optionally `python -m src.verify_predictions --live`.
Real weather/MODIS/place requests need network access. Add `--historical` to
the setup checker/verifier to explicitly require historical replay.

The model remains a Greece-domain experimental prototype. Current weather/EO
providers include documented approximations and imputation; no global
validation, calibration or operational alert performance follows from static
or prediction parity. The global label/calendar gate remains unresolved.

## Verification performed (25 October 2026, IST)

- Exhaustive cube comparison: all 1,231,699 mapping cells, 4,022 units,
  four fixed static aggregates, ten CLC aggregates and all 13 population
  years matched exactly, including missing-value locations.
- Original-provider comparison: nine locations across nine units, six requested
  years (2008, 2009, 2015, 2020, 2021, 2026), and six excluded cells.
  All 54 raw/normalized feature vectors and existing-model scores matched
  exactly (maximum absolute difference zero). Water cells were rejected by
  both providers. Details: `static_geography_parity.json`.
- Full project tests: **40 passed** with the local cube available. An initial
  attempt hit sandbox temp-directory permissions; rerunning with TEMP/TMP
  inside `.cache/test-temp` passed without changing application behavior.
- `python -m scripts.smoke_without_cube` staged current tracked and pending
  source/artifacts under `.cache`, excluding the cube, environments and bulk
  research data. Setup checks passed; tests reported **34 passed, 6 skipped**
  (historical cube tests); offline verification passed. This reused the installed
  Python environment, so it is not a fresh dependency-installation claim.
- Actual `src.verify_predictions --live` in that cube-free staging directory
  passed place search, weather/current context, ORNL EO and Track-B inference.
  The 35-feature score was finite with no missing raw features. EO source
  composites were still dated September 2026 for the October history window,
  and proxy variables remained marked approximate. Availability does not mean
  fresh/source-equivalent EO or calibrated fire probability. UTC target date
  was 24 October (the run occurred just after midnight IST on 25 October).
- `pip check`, constrained `pip install --dry-run --no-index`, Python
  compilation, PowerShell setup/launcher syntax and `git diff --check` passed.
  No new environment was downloaded/installed and the interactive launcher was
  not started. Existing model and evaluation SHA-256 values were unchanged.

Local verification evidence is retained at `.cache/no_cube_smoke.json` and
`.cache/no-cube-checkout-ca36e212/.cache/live_verification.json`. To repeat from
current source use `python -m scripts.smoke_without_cube --live`; a new staging
directory will be created. No commit or push was performed.

## Pending work preserved

Before this migration, five tracked modifications and eleven untracked files
were present. Their original bytes were copied to
`.cache/before_static_migration/`. The pending launcher, setup constraints, CI,
dashboard/error handling, tests and evaluation work were reviewed and retained.
Necessary follow-up edits update README/setup checks for optional historical
data, distinguish compact-artifact errors, and make the existing verifier skip
optional historical replay when the cube is absent (or require it explicitly).
The pre-existing sampled temporal evaluation and research outputs are untouched.

Files added by this migration:

- `src/static_geography.py`
- `scripts/export_static_geography.py`
- `scripts/verify_static_geography.py`
- `scripts/smoke_without_cube.py`
- `tests/test_static_geography.py`
- `models/production/greece_static_v1.npz`
- `models/production/greece_static_v1.json`
- `docs/STATIC_GEOGRAPHY.md`
- `docs/static_geography_parity.json`

Files edited by this migration:

- `.gitignore`
- `src/production_features.py`
- `src/check_setup.py` (existing pending file)
- `src/verify_predictions.py` (existing pending file)
- `app.py` (retained pending changes; updated asset-error message only)
- `README.md` (retained pending setup documentation; corrected data requirement)
- `CLEANUP_MANIFEST.md`
- `docs/FACULTY_VIVA_GUIDE.md`

The other twelve pre-existing pending files retain their baseline bytes,
including `run_fire_warning.bat`, `setup.ps1`, `scripts/start_demo.ps1`, the
dependency constraints, CI configuration, pytest configuration, existing tests,
`src/evaluate_production.py` and `sampled_temporal_evaluation.json`.
