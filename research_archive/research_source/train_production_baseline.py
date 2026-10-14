"""Train the first Track-B production baseline from existing labeled samples.

This is an explicit production artifact path, separate from ``results/`` and
from all FireCube experiment outputs.  Running this script is intentionally a
separate action; importing it never trains or writes anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from .production_features import (
    PRODUCTION_MODEL_DIR,
    PRODUCTION_PREPROCESSOR,
    raw_training_features,
    ProductionPreprocessor,
    production_feature_names,
    write_feature_schema,
)
from .production_model import save_model_manifest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "processed" / "firecube_samples.npz"
MODEL_PATH = PRODUCTION_MODEL_DIR / "xgb_baseline.joblib"
METRICS_PATH = PRODUCTION_MODEL_DIR / "metrics.json"
MANIFEST_PATH = PRODUCTION_MODEL_DIR / "manifest.json"


def train(samples_path: Path = SAMPLES) -> dict:
    raw, labels, years = raw_training_features(samples_path)
    train_mask = years <= 2018
    val_mask = years == 2019
    if not train_mask.any() or not val_mask.any():
        raise ValueError("existing samples do not contain both train years <=2018 and 2019 validation")
    preprocessor = ProductionPreprocessor.fit(raw[train_mask])
    x_train, _ = preprocessor.transform(raw[train_mask])
    x_val, _ = preprocessor.transform(raw[val_mask])
    y_train, y_val = labels[train_mask], labels[val_mask]
    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
    )
    model.fit(x_train, y_train)
    pred = model.predict_proba(x_val)[:, 1]
    metrics = {
        "schema": "track_b_production_v1",
        "feature_names": production_feature_names(),
        "train_years": "<=2018",
        "validation_year": 2019,
        "n_train": int(train_mask.sum()),
        "n_validation": int(val_mask.sum()),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_validation": float(y_val.mean()),
        "roc_auc_validation": float(roc_auc_score(y_val, pred)),
        "average_precision_validation": float(average_precision_score(y_val, pred)),
        "source": str(samples_path),
        "operational_validity": False,
    }
    PRODUCTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor.save(PRODUCTION_PREPROCESSOR)
    joblib.dump({"xgb": model, "feature_names": production_feature_names(), "schema": "track_b_production_v1"}, MODEL_PATH)
    write_feature_schema()
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_model_manifest(MANIFEST_PATH, MODEL_PATH, metrics)
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
