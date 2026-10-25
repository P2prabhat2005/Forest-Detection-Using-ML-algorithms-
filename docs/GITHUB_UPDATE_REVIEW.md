# GitHub update review — 25 October 2026 (IST)

**Verdict: ready for human review and a normal commit/push for the existing
Greece-domain live prototype.** No commit, staging or push was performed.
No model was retrained and no research results were changed. HEAD remains
`da6dd78c87c71c94cfef0efc5aa6451228db17ad`.

## Fresh Windows environment validation

The review staged only Git-tracked and explicitly reviewed project-distributable
pending files: 91 files, 12,097,837 bytes before documentation additions. It
included the compact geography and existing small models, with no NetCDF cube,
bulk research samples, existing environment, application caches or Git state.
The checkout path deliberately contained spaces.

A brand-new Python 3.12.14 venv initially contained only pip. It had
`include-system-site-packages = false`, disabled user-site packages and an empty
inherited PYTHONPATH. Only the base interpreter and standard library were reused.
No packages were copied from the original environment or global site-packages.
Pip bootstrap used Python's bundled ensurepip; a sandbox temporary-directory
permission issue was resolved by directing TEMP/TMP inside the isolated checkout.

The documented `setup.ps1 -Python <Python-3.12-executable>` then completed:

- Public index: `https://pypi.org/simple`; pip configuration files disabled.
- `PIP_NO_CACHE_DIR=true`; no extra index or find-links source.
- 64 package-wheel downloads recorded; zero `Using cached` records.
- Exact documented requirements/constraints installed successfully.
- `pip check` and `src.check_setup` passed.
- Imported numpy, scikit-learn, XGBoost, Streamlit, PyTorch, torchvision and
  netCDF4 were verified to reside inside the new venv, not the original one.

The following checks ran in that fresh environment without the full cube:

- **Pytest: 34 passed, 6 skipped**, with zero failures/errors.
- **Live Nominatim place search:** returned Athens.
- **Live Open-Meteo weather/current context and ORNL MODIS:** returned values;
  all ten dynamic days were available, with no missing raw features.
- **Track-B feature construction/prediction:** 35 ordered features and a
  finite experimental score from the existing model. The static provenance
  hash matched the reviewed compact artifact.
- **Streamlit startup:** localhost root and `/_stcore/health` returned HTTP 200;
  the temporary server was stopped afterward. AppTest also exercised the app.
- **Portable PowerShell helper:** passed using the fresh `.venv` and project
  path with spaces; HTTP/health checks passed and its captured server was stopped.
  The batch wrapper was reviewed; its interactive browser-opening step was not
  invoked. Initial test-harness process-capture/logging errors were corrected
  before the passing helper test; no project launcher edit was needed.
- **Historical setup check:** explicitly returned exit code 1 with
  `FAIL historical cube: ...dataset_greece.nc`, as expected.
- **Historical dashboard:** showed the dataset requirement and setup command,
  with no uncaught exception. Historical data was not replaced with live inputs.

The six skipped tests were exclusively full-cube checks:

- `test_historical_track_b_dashboard_prediction`
- `test_historical_inference_artifacts_exist`
- `test_historical_prediction_is_exp5_compatible`
- `test_outside_domain_rejected_without_model_call`
- `test_historical_provider_reproduces_exp5_feature_vector`
- `test_cube_provider_static_and_prediction_parity`

The existing 4,022-unit/13-population-year parity report and 54 zero-difference
comparisons were inspected and preserved. They were not recomputed in this
cube-free environment. Historical checks skipped by the optional live verifier
were also reported separately, rather than counted as successful predictions.

Full local evidence (deliberately ignored, not for commit):

- `.cache/fresh review/distribution_inventory.json`
- `.cache/fresh review/project/.cache/fresh_setup.log`
- `.cache/fresh review/project/.cache/fresh_validation_report.json`
- `.cache/fresh review/project/.cache/fresh-tests.xml`
- `.cache/fresh review/project/.cache/fresh_live.json`
- `.cache/fresh review/project/.cache/launcher_retry.log`

## Publication and artifact review

The complete tracked diff and all 20 initially untracked files were reviewed.
All 29 pre-existing pending files were copied to
`.cache/before_github_review/` before this review's edits. Their useful work
remains present. This review additionally changed only `.gitignore`, README
instructions, and this report.

The compact payload remains 320,498 bytes with SHA-256:
`66b45961241a602f5427d4b3e490feb897b3b6ada5942ad67c181e2fcc0dc574`.
Its v1 schema, 983×1,253 grid, 4,022 units, feature order, population-year
policy, numeric-only arrays, source checksum, attribution, units and missing
values agree with the loader and existing parity evidence. The payload checksum
was recomputed. The manifest contains no absolute local paths or credentials;
its dataset authors/contact field is preserved public source attribution.

The source citation discrepancy (older embedded Zenodo citation versus the
record named by local research notes) is explicitly documented. The local
source checksum identifies the exact cube; publisher-checksum verification is
not claimed. The existing model-training manifest already contains original
local training paths; it is unchanged provenance and does not select runtime
paths. New runtime loading and launcher paths are portable.

Ignore-rule probes confirmed exclusion of the original cube, `.cache`, `.venv`,
other configured environments, `.env` files, Streamlit secrets, keys,
credential JSON, and bulk `results/` / `research_archive/` outputs. Rules were
extended for `.aws`, `.ssh`, credential dotfiles and private certificate bundles.
The compact NPZ is the single explicit NPZ exception. Existing deliberately
tracked small models/research documentation remain tracked; ignore rules do
not remove them. No dataset, cache, environment or credential file is proposed
for addition. A credential-pattern scan of distributable text found no matches;
this is a scoped repository check, not a comprehensive security certification.

README now includes Git clone, project-directory selection, Python-version
verification, setup, launcher and alternate-interpreter instructions for a
student on Windows. Dependency versions were validated by actual installation.

## Exact files proposed for commit

Review and stage only these **30 files**, including all intended earlier
pending setup/verification work and this review report:

```text
.github/workflows/tests.yml
.gitignore
CLEANUP_MANIFEST.md
README.md
app.py
docs/FACULTY_VIVA_GUIDE.md
docs/GITHUB_UPDATE_REVIEW.md
docs/STATIC_GEOGRAPHY.md
docs/static_geography_parity.json
models/production/greece_static_v1.json
models/production/greece_static_v1.npz
models/production/sampled_temporal_evaluation.json
pytest.ini
requirements-dev.txt
requirements-lock.txt
run_fire_warning.bat
scripts/export_static_geography.py
scripts/smoke_without_cube.py
scripts/start_demo.ps1
scripts/verify_static_geography.py
setup.ps1
src/check_setup.py
src/evaluate_production.py
src/production_features.py
src/static_geography.py
src/verify_predictions.py
tests/test_dashboard.py
tests/test_firecube_live.py
tests/test_production_pipeline.py
tests/test_static_geography.py
```

No other files are required for this update. Existing unchanged tracked files,
including the saved model/preprocessor and sample images, are already in Git.
The earlier sampled temporal evaluation and its utility are retained because
the pending README documents them; neither was rerun or altered in this review.

## Remaining limits and blockers

There is no identified software blocker to publishing this prototype update.
It is ready for human review; GitHub remains unchanged until a human commits
and pushes the listed files. Fresh installation and testing were on Windows
x64/Python 3.12 only, using an existing CPython interpreter installation.

Historical replay still needs the original cube, by design. Live services need
network access and retain their approximation/imputation policies. During this
fresh run, NDVI was sourced from 14 September and LST from 22 September for an
October history window. Successful retrieval does not establish EO freshness.
The score remains experimental, uncalibrated and Greece-domain only; the
global label/calendar gate and prospective research evidence are unchanged.
