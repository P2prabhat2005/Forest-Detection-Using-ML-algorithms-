# Forest Fire Detection Using ML Algorithms

BCA PBL academic prototype: predict **Fire** vs **Not Fire** from environmental / fire-weather features using a **Random Forest** classifier (training not started yet).

**Scope:** supervised learning prototype with CLI + simple Flask demo later. Not a real-time emergency system.

## Current status

| Step | Status |
|------|--------|
| Project structure | Done |
| Official UCI dataset in `data/raw/` | Done (unchanged) |
| Cleaning + validation modules | Done |
| Processed dataset in `data/processed/` | Done |
| EDA notebook + figures | Done |
| Random Forest training | **Not started** |
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
→ Train/Test split          [next]
→ Random Forest             [next]
→ Evaluation                [next]
→ Saved model               [next]
→ CLI + Flask interface     [next]
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
│   └── eda.py
├── reports/
│   ├── figures/
│   ├── dataset_inspection.md
│   └── data_preparation_report.md
├── models/                  # Future saved model artifacts
├── app/                     # Future Flask UI
├── requirements.txt
└── README.md
```

## Setup and run (this phase)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Clean raw → processed
PYTHONPATH=. python -m src.prepare_data

# Generate EDA figures
PYTHONPATH=. python -m src.eda
```

Then open `notebooks/eda_and_training.ipynb` for interactive EDA (no RF training in this stage).
