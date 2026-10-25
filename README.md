# Forest Fire Detection — Fire Early Warning Demo

This repository contains the current Streamlit faculty demo. Start it with
`run_fire_warning.bat` on Windows after setup, or run
`streamlit run app.py` from a configured Python environment.

The primary experience is **Fire Early Warning**: place search, selected
location, current/recent environmental inputs, Track-B XGBoost model score,
map, and data-quality explanation. The image, weather, and combined MVP tabs
remain available as supporting demonstrations.

Completed experiments and training inputs are preserved locally in
`research_archive/`; Git includes only selected small documentation, scripts,
and inference artifacts. See `CLEANUP_MANIFEST.md` for the local inventory.

## Greece live inference from a checkout

Live Track-B uses the compact `models/production/greece_static_v1.npz` and its
matching `.json` provenance/checksum manifest. These small files must accompany
the model and preprocessor in a checkout. The 22 GiB FireCube cube at
`data/raw/firecube/dataset_greece.nc` is deliberately excluded from Git and is
required only for historical replay and optional cube-parity checks. No
runtime geography is fabricated or downloaded as a fallback. Missing, corrupt,
or incompatible compact artifacts stop live prediction with a setup error.

The artifact preserves the existing Greece grid, land eligibility and 15
static features, including all population years used by the current provider.
See `docs/STATIC_GEOGRAPHY.md` for provenance, units, schema, limitations and
reproduction commands. This change does not extend prediction coverage.

The small production XGBoost model, preprocessor, and schema are included.
The legacy image/weather demo artifacts are also included. The Windows launcher
uses `.venv/Scripts/python.exe` relative to this checkout. Set
`FIRE_WARNING_PYTHON` to use another configured executable. Model scores are experimental and
are not calibrated operational wildfire probabilities.

## Original MVP scope

College / portfolio project that flags forest-fire risk from:

1. **Photos** — MobileNetV2 transfer learning (`fire` vs `no_fire`)
2. **Weather / FWI sensors** — Random Forest on the [Algerian Forest Fires](https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset) dataset
3. **Late fusion** — weighted mix of the two `P(fire)` scores when both inputs are present

This is a demo system, not an operational satellite wildfire product.

## Run the current demo

Double-click:

```text
run_fire_warning.bat
```

It starts the local Streamlit app at `http://127.0.0.1:8502`.

## Environment setup (if rebuilding the environment)

Install Python 3.12, then run from the checkout:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

For a first checkout on Windows, install Git and 64-bit Python 3.12 (enable
Python on PATH), then open PowerShell and run:

```powershell
git clone https://github.com/P2prabhat2005/Forest-Detection-Using-ML-algorithms-.git Forest-fire-Detection
Set-Location .\Forest-fire-Detection
python --version
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
.\run_fire_warning.bat
```

`python --version` must report 3.12.x. If Windows opens the Store or selects
another Python version, pass your installed Python 3.12 executable to
`setup.ps1 -Python "C:\path\to\python.exe"`. No API keys or FireCube download
are needed for the live prototype. Setup downloads Python packages, including
PyTorch; allow network access and enough disk space for a separate environment.

For a Python executable outside PATH, pass `-Python "C:\path\to\python.exe"`.
Setup creates or repairs `.venv`, installs dependencies constrained to the
verified environment in `requirements-lock.txt`, and validates the compact
artifact and required assets. It does not download FireCube. Use 64-bit Python
3.12; the saved models were verified with the exact package versions in the
constraints file (including numpy, scikit-learn and XGBoost). Other platforms
can create a Python 3.12 venv, install `requirements-dev.txt` with
`-c requirements-lock.txt`, and run `python -m streamlit run app.py`.
The verified environment/CI target is Windows x64; other platforms need their
own installation check. Existing `.venv` paths are local and never versioned.

## Verify the demo

```powershell
.\.venv\Scripts\python.exe -m src.check_setup
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.verify_predictions --live
```

Tests are restricted to this project's `tests/` folder. Archived third-party
research tests are excluded. Dataset-dependent checks skip when the local cube
is absent. Tests use stubs for external services; live API availability must be
checked separately. `src.verify_predictions --live` checks the compact
geography, legacy model fusion, live place search and live Track-B inference, and
saves diagnostic results/provenance in `.cache/verification.json`. Omit `--live`
for offline model checks. A passing live inference may still contain explicitly
flagged approximate/imputed inputs; inspect the report. CI runs the offline
suite on Windows/Python 3.12. Historical checks run when the cube is present;
use `--historical` to require them. `python -m src.check_setup --historical`
also requires the full cube. Missing historical data never falls back to live
inputs or changes the historical model's behavior.

## Provisioning and deployment

For optional historical replay, obtain the authorized Greece FireCube file and copy it to
`data/raw/firecube/dataset_greece.nc` (23,501,672,194 bytes in the reviewed local
copy). Keep it outside Git. Run `python -m src.check_setup` and the full tests
before demonstrating historical predictions. A matching filename or size alone does not
establish dataset authenticity; use the original research source and its terms.

The included launcher binds to localhost and writes logs under `.cache/`.
It rejects an occupied port rather than opening an unrelated service. Stop the
server before launching it again. Live Greece inference requires the compact
artifact pair, saved model/preprocessor and network access to Open-Meteo and
ORNL MODIS; place search uses Nominatim. A hosted installation must include
those assets and configure its own hosting/network environment. The live input
providers retain their existing approximation and missing-data policies.

## Remaining research scope

The runnable demo is separate from an operational warning service. Remaining
research includes real-image evaluation, source-aligned environmental inputs,
realistic-prevalence and geographic holdouts, calibration, prospective testing,
and reviewed alert thresholds. Global/India coverage and public alerts are not
implemented. Legacy image scores use synthetic validation data; the combined
score has no independently evaluated accuracy. See `docs/FACULTY_VIVA_GUIDE.md`
for experiment evidence and limitations.

For a repeatable evaluation of the unchanged Track-B artifacts on the archived
2019 validation and 2020/2021 temporal test samples, run
`python -m src.evaluate_production`. Pass `--samples` if the archive is elsewhere.
It writes `models/production/sampled_temporal_evaluation.json`. These samples have
artificial prevalence and historical inputs, so their metrics do not measure
live alert precision or regional generalization.

## Archived training workflow

The original data-download and model-training scripts were intentionally moved
to `research_archive/research_source/` with their associated training data.
They are not needed to run the faculty demonstration or its current models.

## Original MVP data notes

Download the UCI CSV and write bundled sample images:

```powershell
python research_archive/research_source/scripts/download_data.py
```

For a local image-model smoke test (synthetic orange “fire” vs green “forest” photos):

```powershell
python research_archive/research_source/scripts/download_data.py --synthetic-images
```

For real camera/drone images, unzip a public fire dataset into:

```
data/images/train/fire/
data/images/train/no_fire/
data/images/val/fire/
data/images/val/no_fire/
```

Sources: [FIRE Dataset on Kaggle](https://www.kaggle.com/datasets/phylake1337/fire-dataset), other forest-fire image collections, or the FLAME aerial dataset.

## Original MVP training notes

The original training scripts are preserved under
`research_archive/research_source/`.

Models are written to `models/tabular_rf.joblib` and `models/image_mobilenet.pt`.

## Legacy MVP tabs

```powershell
streamlit run app.py
```

- **Image** — upload a photo or pick a bundled sample
- **Weather** — enter temperature, humidity, wind, rain, and FWI components
- **Combined** — late fusion (default 60% image / 40% weather)
- **Training metrics** — accuracy, F1, confusion matrix

## Project layout

| Path | Role |
| --- | --- |
| `src/predict.py` | Inference + fusion helper |
| `app.py` | Streamlit UI |
| `research_archive/` | Preserved research, training, and completed experiments |
