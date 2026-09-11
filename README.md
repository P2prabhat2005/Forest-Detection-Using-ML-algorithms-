# Forest Fire Detection Using ML Algorithms

BCA PBL academic prototype: predict **Fire** vs **Not Fire** from environmental / fire-weather features using a **Random Forest** classifier.

**Scope:** supervised learning prototype with CLI + simple Flask demo later. Not a real-time emergency system.

## Current status

| Step | Status |
|------|--------|
| Project structure | Done |
| Official UCI dataset in `data/raw/` | Done (unchanged) |
| Cleaning + validation modules | Done |
| Processed dataset in `data/processed/` | Done (243 rows) |
| EDA notebook + figures | Done |
| Random Forest training + evaluation | **Done** |
| Saved model Pipeline | **Done** (`models/random_forest_pipeline.joblib`) |
| Prediction module (`src/predict.py`) | **Done** |
| Flask app | **Not started** |

## Dataset

| Item | Detail |
|------|--------|
| Name | Algerian Forest Fires |
| Source | [UCI ML Repository — Dataset 547](https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset) |
| DOI | [10.24432/C5KW4N](https://doi.org/10.24432/C5KW4N) |
| Raw file | `data/raw/Algerian_forest_fires_dataset_UPDATE.csv` |
| Processed file | `data/processed/algerian_forest_fires_cleaned.csv` (243 rows after dropping 1 malformed row) |

### Citation

Abid, Faroudja. (2019). *Algerian Forest Fires* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5KW4N

See also `data/raw/SOURCE.md`.

## Project pipeline

```text
Dataset (UCI raw)
→ Data cleaning / validation
→ EDA
→ Feature preparation (shared sklearn preprocessor)
→ Stratified train/test split
→ Random Forest (Pipeline: preprocess + model)
→ Held-out evaluation
→ Saved model artifacts
→ Prediction module         [done]
→ Flask interface           [next]
```

## Preprocessing decisions (approved)

**Keep for primary model:**  
`Temperature`, `RH`, `Ws`, `Rain`, `FFMC`, `DMC`, `DC`, `ISI`, `Region`, `month`

**Exclude from primary model:**  
`FWI`, `BUI`, `day`, `year`

- FWI/BUI excluded as composite/redundant with other fire-weather variables (modeling clarity, not a claim they are useless).
- `day` excluded to reduce calendar memorization on a tiny single-year sample.
- `year` excluded because it is constant (2012).
- Malformed row **dropped** (not repaired).
- Target encoding: `not fire → 0`, `fire → 1`.

Details: `reports/data_preparation_report.md`

## Model training summary

| Item | Value |
|------|-------|
| Algorithm | RandomForestClassifier |
| Split | Stratified 80/20 (`test_size=0.20`, `random_state=42`) |
| Train / test rows | 194 / 49 |
| Preprocessing | Fitted **only on train** inside sklearn Pipeline |
| Held-out accuracy / precision / recall / F1 | 1.00 / 1.00 / 1.00 / 1.00 |
| Train-only 5-fold CV accuracy (mean ± std) | 0.9794 ± 0.0192 |

**Important limitation:** The dataset is small (243 cleaned rows, one country, one year). Perfect held-out scores on 49 test rows do **not** prove real-world or production performance.

Full write-up: `reports/model_training_report.md`

## Project structure

```text
├── data/
│   ├── raw/                 # Original UCI CSV (do not modify)
│   └── processed/           # Cleaned modeling table
├── notebooks/
│   └── eda_and_training.ipynb
├── src/
│   ├── config.py
│   ├── validation.py
│   ├── data_cleaning.py
│   ├── preprocessing.py
│   ├── prepare_data.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── reports/
│   ├── figures/
│   ├── metrics/
│   ├── dataset_inspection.md
│   ├── data_preparation_report.md
│   ├── model_training_report.md
│   └── prediction_module.md
├── models/                  # Saved Pipeline + training metadata
├── app/                     # Future Flask UI
├── requirements.txt
└── README.md
```

## Prediction module

Loads the saved Pipeline (`models/random_forest_pipeline.joblib`) and applies the **same** training-time preprocessing automatically.

```python
from src.predict import predict_fire

result = predict_fire({
    "Temperature": 32,
    "RH": 55,
    "Ws": 15,
    "Rain": 0.0,
    "FFMC": 86.0,
    "DMC": 20.0,
    "DC": 50.0,
    "ISI": 6.0,
    "Region": "Bejaia",
    "month": 8,
})
# result["prediction"] -> "Fire" or "No Fire"
# result["probability_fire"] -> float in [0, 1]
```

CLI integration demo (uses cleaned processed rows; not a new evaluation metric):

```bash
PYTHONPATH=. python -m src.predict --demo-from-processed 3
```

Details: `reports/prediction_module.md`

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Clean raw → processed
PYTHONPATH=. python -m src.prepare_data

# 2) EDA figures
PYTHONPATH=. python -m src.eda

# 3) Train Random Forest + evaluate + save artifacts
PYTHONPATH=. python -m src.train

# 4) Reload and print saved metrics
PYTHONPATH=. python -m src.evaluate

# 5) Prediction module integration demo
PYTHONPATH=. python -m src.predict --demo-from-processed 3
```

Then open `notebooks/eda_and_training.ipynb` for interactive EDA.
