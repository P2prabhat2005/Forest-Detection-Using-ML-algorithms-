# Faculty Review Cleanup Manifest

This repository was simplified without permanently deleting any project material.
Items outside the current application were moved into `research_archive/` and
remain available for inspection or future research work.

## Required for the current application

- `app.py` — Streamlit Fire Early Warning application.
- `run_fire_warning.bat` — supported local launcher.
- `src/` — only runtime modules retained: FireCube inference/data adapters,
  production prediction pipeline, place search, and legacy MVP inference.
- `models/` — production XGBoost artifacts and the legacy image/weather MVP
  artifacts used by the application tabs.
- `data/raw/firecube/dataset_greece.nc` — required read-only FireCube source
  data for location mapping, static features, and historical mode.
- `data/images/samples/` — bundled images used by the Image MVP tab.
- `results/firecube_direct_10km_artifact_rerun/` — retained only
  `xgb_direct_10km.joblib` and `feature_aggregation.json`, which are loaded
  by the historical/runtime adapter.
- `results/firecube_exp7_two_stage/flat_exp5_calibrator.joblib` — retained
  because the historical view explicitly uses this optional calibration
  artifact.
- `requirements.txt`, `README.md`, and `tests/` — run guidance and focused
  application checks.
- `.venv/` — retained because the existing launcher explicitly uses its
  `Lib/site-packages` directory. It is a local dependency, not a research
  artifact.

## Required for research / reproducibility (archived)

- `research_archive/results/` — completed benchmark outputs, prediction
  exports, analysis reports, experiment manifests, and superseded model
  artifacts.
- `research_archive/research_source/` — training, extraction, benchmark,
  rerun, and analysis scripts that are not imported by the live application.
- `research_archive/research_data/` — training/validation images, tabular
  training data, extracted samples, reference repositories, and FireCube
  inspection material.
- `research_archive/research_documentation/` — experiment and dataset notes.

## Generated / cache / local only (archived where possible)

- `research_archive/local_runtime/` — Python bytecode and prior runtime
  artifacts that were not locked by another Windows process.
- `research_archive/local_editor/` — Cursor and VS Code workspace settings.
- `research_archive/local_diagnostics/` — prior environment diagnosis output.

`/.cache` remains at the root because Windows reported an active file handle
when it was moved. It contains disposable application caches/logs only; the
launcher recreates it automatically. No files were deleted to work around that
lock.

## Obsolete / unused (archived, not deleted)

- `research_archive/legacy_launchers/` — completed experiment launchers and
  environment-repair launchers, superseded by `run_fire_warning.bat`.
- `research_archive/empty_root_folders/` — the now-empty legacy `scripts/`
  folder after its data-download helper was archived with the training source.

## Deliberately not removed

- `.git/` was untouched.
- No trained model contents were altered.
- No current application code paths, models, sample images, production
  configuration, or FireCube runtime data were moved.
- No project file was permanently deleted.
