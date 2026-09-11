# Prediction Module

**Status:** Implemented (no Flask / no web UI yet)

## Purpose

`src/predict.py` loads the **already-trained** sklearn Pipeline from:

`models/random_forest_pipeline.joblib`

and predicts **Fire / No Fire** for new feature rows.

## How the saved model is loaded

- `load_pipeline()` uses `joblib.load(...)` on the saved Pipeline.
- The Pipeline contains:
  1. `preprocess` — `ColumnTransformer` **fitted during training**
  2. `model` — `RandomForestClassifier` fitted on transformed training features
- The Pipeline object is cached in memory after first load.
- Prediction **never retrains** the model.

## How preprocessing is handled

Because the saved artifact is a full Pipeline, calling:

```python
pipeline.predict(X)
pipeline.predict_proba(X)
```

automatically applies the **same** scaling / one-hot encoding used at training time.
No separate manual preprocessing step is required (or allowed) in the prediction module.

## Expected input schema

Required features (exact names / approved set):

| Feature | Type | Notes |
|---------|------|-------|
| Temperature | numeric | °C |
| RH | numeric | relative humidity % |
| Ws | numeric | wind speed |
| Rain | numeric | mm |
| FFMC | numeric | Fine Fuel Moisture Code |
| DMC | numeric | Duff Moisture Code |
| DC | numeric | Drought Code |
| ISI | numeric | Initial Spread Index |
| Region | categorical | `"Bejaia"` or `"Sidi-Bel-Abbes"` |
| month | integer | calendar month `1..12` (model trained mainly on 6–9) |

Accepted Python input forms:

- one `dict`
- `list[dict]`
- `pandas.Series`
- `pandas.DataFrame`

## Returned prediction format

Single-row example shape:

```json
{
  "prediction": "Fire",
  "prediction_encoded": 1,
  "probability_fire": 0.93,
  "probability_no_fire": 0.07,
  "class_names": {"0": "No Fire", "1": "Fire"},
  "target_mapping": {"not fire": 0, "fire": 1}
}
```

- `prediction`: `"Fire"` or `"No Fire"`
- `prediction_encoded`: `1` or `0`
- probabilities come from `predict_proba` when available

## Validation rules

`validate_prediction_input()` rejects invalid inputs with clear errors if:

1. any required feature is missing
2. a numeric feature cannot be converted to a number / is missing
3. `Region` is not one of `Bejaia`, `Sidi-Bel-Abbes`
4. `month` is not an integer in `1..12`
5. input type is unsupported

## Public API

```python
from src.predict import predict_fire, load_pipeline, validate_prediction_input

result = predict_fire({...feature dict...})
```

## CLI demo (integration only)

```bash
PYTHONPATH=. python -m src.predict --demo-from-processed 3
```

This runs the real saved Pipeline on the first N rows of the cleaned processed CSV.
It is an **integration/smoke demo**, not a new accuracy report.

JSON file input:

```bash
PYTHONPATH=. python -m src.predict --json path/to/input.json
```

## Out of scope (not built yet)

- Flask / HTML / CSS / JS
- REST API
- Live weather / satellite / IoT feeds
