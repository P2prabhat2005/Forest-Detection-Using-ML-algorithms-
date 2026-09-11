# Data Preparation & EDA Report

**Project:** Forest Fire Detection Using ML Algorithms (BCA PBL)  
**Stage:** Data preparation + EDA only (Random Forest **not trained**)  
**Dataset:** UCI Algerian Forest Fires Dataset (ID 547)

## 1. Raw dataset issues

Observed in `data/raw/Algerian_forest_fires_dataset_UPDATE.csv` (unchanged):

1. Two region title rows embedded in the file (`Bejaia Region Dataset`, `Sidi-Bel Abbes Region Dataset`).
2. A repeated header row in the middle of the file.
3. One blank line between regions.
4. Whitespace in some column names and in `Classes` values (e.g. `"fire "`, `"not fire   "`).
5. One malformed observation (source line 171): `DC` contains `"14.6 9"`, `FWI` contains `"fire"`, and `Classes` is empty. This is a field-shift defect.

## 2. Exact cleaning decisions (approved)

| Decision | Action taken |
|----------|----------------|
| Region | Derived from the two file sections → `Bejaia` / `Sidi-Bel-Abbes` |
| Column names | Stripped whitespace |
| Target labels | Stripped whitespace only; semantics preserved as `fire` / `not fire` |
| Malformed row | **Dropped** (not repaired) |
| Numeric coercion | Applied only to kept rows |
| Target encoding | `not fire → 0`, `fire → 1` |
| Output location | `data/processed/algerian_forest_fires_cleaned.csv` |
| Raw file | Never modified |

## 3. Dropped row

- **Count:** 1  
- **Source line:** 171  
- **Region:** Sidi-Bel-Abbes  
- **Evidence:** `DC="14.6 9"`, `FWI="fire"`, `Classes=""`  
- **Policy:** Drop only; no reconstructed values were inserted.

## 4. Final feature set (primary PBL model)

**Kept:**

`Temperature`, `RH`, `Ws`, `Rain`, `FFMC`, `DMC`, `DC`, `ISI`, `Region`, `month`

**Dropped from the primary model:**

`FWI`, `BUI`, `day`, `year`

### Why these were excluded

- **FWI / BUI:** Excluded because they are composite/redundant with other fire-weather variables. This gives a clearer, less redundant feature set for the primary PBL model. This is a **modeling choice**, not a claim that FWI/BUI are universally useless.
- **day:** Excluded to reduce calendar memorization risk on a tiny single-year sample.
- **year:** Excluded because it is constant (2012) and adds no information.

## 5. Target encoding

| Original label | Encoded value |
|----------------|---------------|
| `not fire` | 0 |
| `fire` | 1 |

Original label strings are retained in the processed CSV as `Classes`, with `Classes_encoded` for modeling.

## 6. Class distribution after cleaning

| Class | Count | Encoded |
|-------|------:|--------:|
| fire | 137 | 1 |
| not fire | 106 | 0 |
| **Total** | **243** | — |

Region split after cleaning: Bejaia 122, Sidi-Bel-Abbes 121.

## 7. Preprocessing object

`src/preprocessing.py` builds a reusable sklearn `ColumnTransformer`:

- numeric features → `StandardScaler`
- `Region` → `OneHotEncoder`

The same object is intended for training and inference. Scaling is not required for Random Forest, but keeps train/serve transforms identical and reusable.

Validation (`src/validation.py`) fails clearly if:

- expected columns are missing
- target labels are unexpected
- numeric conversion fails
- feature inputs do not match the approved feature set/order

## 8. EDA outputs

Figures saved under `reports/figures/`:

- `class_distribution.png`
- `feature_distributions.png`
- `correlation_heatmap.png`
- `features_by_class_boxplot.png`
- `region_class_counts.png`
- `month_class_counts.png`

Notebook: `notebooks/eda_and_training.ipynb` (EDA + preprocessing smoke test only; **no RF training**).

### Key EDA observations (descriptive only; no model metrics)

1. Mild class imbalance (137 fire vs 106 not fire).
2. Fire days tend toward higher `Temperature`, higher `FFMC` / `ISI`, and lower `RH` / `Rain` (group means).
3. Strong correlation remains between some kept FWI components (notably `DMC`–`DC` ≈ 0.88). Both were **kept** as approved; redundancy is a known limitation, not an extra silent drop.
4. Class rates differ by region and month (useful for stratified evaluation later).

## 9. Remaining limitations / decisions for later

1. Small sample (n=243) from one country and one year (2012) → limited generalization.
2. `DMC` and `DC` remain highly correlated; kept by approval.
3. No model trained yet → no accuracy/precision/recall/F1 to report.
4. Flask interface not built yet.
5. Train/test split strategy will be chosen in the training phase (should be stratified).

## 10. How to reproduce this phase

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m src.prepare_data
PYTHONPATH=. python -m src.eda
```

Then open `notebooks/eda_and_training.ipynb`.
