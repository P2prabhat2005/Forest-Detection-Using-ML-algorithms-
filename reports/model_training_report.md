# Model Training & Evaluation Report

**Stage:** Random Forest training + held-out evaluation  
**Dataset:** UCI Algerian Forest Fires (cleaned processed file, 243 rows)  
**Primary model:** `RandomForestClassifier` inside a sklearn `Pipeline`

## 1. Data used

- Source file: `data/processed/algerian_forest_fires_cleaned.csv`
- Features (approved):  
  `Temperature`, `RH`, `Ws`, `Rain`, `FFMC`, `DMC`, `DC`, `ISI`, `Region`, `month`
- Target: `Classes_encoded` (`0 = not fire`, `1 = fire`)
- Raw file under `data/raw/` was **not** modified.

## 2. Train/test split

| Setting | Value |
|---------|-------|
| Strategy | Stratified `train_test_split` |
| `test_size` | **0.20** |
| `random_state` | **42** |
| Train size | **194** |
| Test size | **49** |

**Why 20% test?**  
The cleaned dataset is small (243 rows). A 20% hold-out keeps most samples for training while still providing a stratified held-out test set for unbiased evaluation. This is a practical academic choice, not a production optimization.

### Class counts

| Split | not fire (0) | fire (1) |
|-------|-------------:|---------:|
| Train | 85 | 109 |
| Test | 21 | 28 |

## 3. Leakage control

- Features and target were separated before splitting.
- A single sklearn `Pipeline` contains:
  1. `preprocess` (`ColumnTransformer`: scale numerics + one-hot `Region`)
  2. `model` (`RandomForestClassifier`)
- The pipeline was **fitted only on the training split**.
- Held-out test data was used **once** for final metrics (no test-set tuning).

## 4. Random Forest hyperparameters (a priori)

```json
{
  "n_estimators": 100,
  "max_depth": null,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": "sqrt",
  "bootstrap": true,
  "class_weight": null,
  "n_jobs": -1,
  "random_state": 42
}
```

These were chosen as standard defaults for a BCA PBL prototype and were **not** searched/optimized against the test set.

## 5. Held-out TEST metrics (primary result)

| Metric | Value |
|--------|------:|
| Accuracy | 1.0000 |
| Precision (fire=1) | 1.0000 |
| Recall (fire=1) | 1.0000 |
| F1-score (fire=1) | 1.0000 |

### Confusion matrix (test set)

|  | Pred not fire | Pred fire |
|--|--------------:|----------:|
| **Actual not fire** | 21 (TN) | 0 (FP) |
| **Actual fire** | 0 (FN) | 28 (TP) |

**Interpretation:** On this 49-row held-out split, every sample was classified correctly. With such a small, strongly separable fire-weather dataset, a perfect score can occur and must **not** be treated as proof of real-world performance.

Figure: `reports/figures/confusion_matrix.png`

## 6. Feature importance (from trained RF)

Importances are Gini/impurity-based on the transformed training features. See:

- `reports/figures/feature_importance.png`
- `reports/metrics/evaluation_metrics.json` → `feature_importance`

Top contributors are typically fire-weather indices such as `FFMC` / `ISI` (exact ranking is in the saved JSON).

## 7. Optional cross-validation (TRAINING split only)

5-fold stratified CV was run **only on the 194 training rows** as a stability check.  
These numbers are **not** the final test result.

| CV metric | Mean | Std |
|-----------|-----:|----:|
| Accuracy | 0.9794 | 0.0192 |
| Precision | 0.9818 | 0.0223 |
| Recall | 0.9818 | 0.0223 |
| F1 | 0.9816 | 0.0171 |

CV means are high and close to the test score, which supports stability on this dataset, but the sample remains tiny.

## 8. Saved artifacts

| Artifact | Path |
|----------|------|
| Full train Pipeline (preprocess + RF) | `models/random_forest_pipeline.joblib` |
| Training metadata | `models/training_metadata.json` |
| Evaluation metrics JSON | `reports/metrics/evaluation_metrics.json` |
| Confusion matrix figure | `reports/figures/confusion_matrix.png` |
| Feature importance figure | `reports/figures/feature_importance.png` |

Prediction code can load `random_forest_pipeline.joblib` and call `.predict(X)` / `.predict_proba(X)` so preprocessing matches training exactly.

## 9. Limitations / concerns (small dataset)

1. Only **243** cleaned rows; test set is only **49** rows.
2. Data covers **two Algerian regions** and **one year (2012)** only.
3. Several FWI-related features are highly informative / correlated, so near-perfect separation is plausible and can look “too good.”
4. Perfect test metrics ≠ production readiness.
5. No external validation set / no deployment monitoring.

## 10. Ready for prediction module?

**Yes, for the next academic step:** the saved Pipeline is ready to be connected to a CLI/Flask prediction interface.

**No, not for real emergency use:** this remains a BCA PBL prototype.

## Reproduce

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m src.prepare_data   # if processed CSV missing
PYTHONPATH=. python -m src.train          # train + evaluate + save artifacts
PYTHONPATH=. python -m src.evaluate       # reload and print saved metrics
```
