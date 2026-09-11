# Forest Fire Detection Using ML Algorithms

Academic BCA PBL prototype: predict **Fire** vs **Not Fire** from environmental / Fire Weather Index (FWI) features using a **Random Forest** classifier.

**Scope:** supervised learning prototype with CLI + simple Flask demo. Not a real-time emergency system.

## Current status

- Project folder structure created
- Official UCI dataset downloaded into `data/raw/` (unchanged)
- Dataset inspected; training / Flask app **not started yet** (awaiting preprocessing approval)

## Dataset

| Item | Detail |
|------|--------|
| Name | Algerian Forest Fires |
| Source | [UCI Machine Learning Repository — Dataset 547](https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset) |
| DOI | [10.24432/C5KW4N](https://doi.org/10.24432/C5KW4N) |
| Download used | Official UCI zip: `https://archive.ics.uci.edu/static/public/547/algerian+forest+fires+dataset.zip` |
| Raw file | `data/raw/Algerian_forest_fires_dataset_UPDATE.csv` |
| Task | Binary classification: Fire / Not Fire |

### Citation

Abid, Faroudja. (2019). *Algerian Forest Fires* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5KW4N

See also `data/raw/SOURCE.md`.

## Planned project structure

```text
├── data/
│   ├── raw/           # Original UCI CSV only (do not modify)
│   └── processed/     # Cleaned outputs after approved preprocessing
├── notebooks/         # EDA + training notebook (planned)
├── src/               # Modular Python pipeline (planned)
├── models/            # Saved model + preprocessing artifacts (planned)
├── reports/figures/   # EDA / evaluation plots (planned)
├── app/               # Simple Flask UI (planned)
└── README.md
```

## Confirmed design decisions

1. Dataset: Algerian Forest Fires (UCI 547)
2. Problem: supervised binary classification — Fire vs Not Fire
3. Algorithm: Random Forest Classifier
4. Interface: CLI + simple Flask web app
5. Notebook: EDA and training notebook
6. Libraries: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (+ Flask for UI)
7. No database unless a later requirement appears
8. Academic prototype only
