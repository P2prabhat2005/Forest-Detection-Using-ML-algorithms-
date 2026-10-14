# Forest Fire Detection — Fire Early Warning Demo

This repository contains the current Streamlit faculty demo. Start it with
`run_fire_warning.bat` on the original Windows machine, or run
`streamlit run app.py` from a configured Python environment.

The primary experience is **Fire Early Warning**: place search, selected
location, current/recent environmental inputs, Track-B XGBoost model score,
map, and data-quality explanation. The image, weather, and combined MVP tabs
remain available as supporting demonstrations.

Completed experiments and training inputs are preserved locally in
`research_archive/`; Git includes only selected small documentation, scripts,
and inference artifacts. See `CLEANUP_MANIFEST.md` for the local inventory.

## GitHub checkout limitations

The 22 GiB FireCube cube at `data/raw/firecube/dataset_greece.nc` is deliberately
not in Git. The current Track-B live and historical prediction pipelines use
that cube for Greece static features and domain mapping, so a fresh clone can
show the dashboard but cannot produce those predictions until the authorized
FireCube data is provisioned at that path. This is a known deployment blocker,
not a missing model file. Do not substitute unrelated static data silently.

The small production XGBoost model, preprocessor, and schema are included.
The legacy image/weather demo artifacts are also included. The Windows launcher
currently references a Python runtime installed at a user-specific path;
other machines should create a local environment and use the Streamlit command
above until the launcher is made portable. Model scores are experimental and
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

Python 3.12 recommended.

```powershell
cd C:\Users\prabh\Forest-fire-Detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

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
