"""Train a Random Forest classifier on the cleaned processed dataset.

Pipeline design:
- Stratified train/test split
- Preprocessor fitted ONLY on training data (via sklearn Pipeline)
- RandomForestClassifier trained on transformed training features
- Full Pipeline + metadata saved under models/

Hyperparameters are fixed a priori and are NOT tuned on the held-out test set.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    EVALUATION_METRICS_PATH,
    FEATURE_COLUMNS,
    MODEL_PIPELINE_PATH,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RF_HYPERPARAMETERS,
    TARGET_ENCODED_COLUMN,
    TEST_SIZE,
    TRAINING_METADATA_PATH,
)
from src.evaluate import evaluate_pipeline, save_evaluation_outputs
from src.preprocessing import build_preprocessor, select_features
from src.validation import DataValidationError, require_columns


def load_processed_dataset(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. "
            "Run `python -m src.prepare_data` first."
        )
    df = pd.read_csv(path)
    require_columns(df, FEATURE_COLUMNS + [TARGET_ENCODED_COLUMN], "processed dataset")
    if df.empty:
        raise DataValidationError("Processed dataset is empty.")
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = select_features(df)
    y = df[TARGET_ENCODED_COLUMN].astype(int)
    if set(y.unique()) - {0, 1}:
        raise DataValidationError(
            f"Target must contain only 0/1, found {sorted(y.unique())}."
        )
    return X, y


def build_model_pipeline() -> Pipeline:
    """Return an unfitted preprocess + RandomForest pipeline."""
    rf = RandomForestClassifier(random_state=RANDOM_STATE, **RF_HYPERPARAMETERS)
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", rf),
        ]
    )


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def run_training_cv(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Cross-validation on the TRAINING split only (stability check)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1"]
    results = {}
    for metric in scoring:
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=metric,
            n_jobs=-1,
        )
        results[metric] = {
            "fold_scores": [round(float(s), 4) for s in scores],
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
        }
    return results


def save_training_artifacts(
    pipeline: Pipeline,
    metadata: dict,
    model_path: Path = MODEL_PIPELINE_PATH,
    metadata_path: Path = TRAINING_METADATA_PATH,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_and_evaluate(
    processed_path: Path = PROCESSED_DATA_PATH,
    run_cv: bool = True,
) -> dict:
    """End-to-end training + held-out evaluation."""
    df = load_processed_dataset(processed_path)
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = stratified_split(X, y)

    pipeline = build_model_pipeline()

    # Optional CV on training data only (before final fit), for stability insight.
    cv_results = None
    if run_cv:
        cv_results = run_training_cv(build_model_pipeline(), X_train, y_train)

    # Fit preprocessor + model ONLY on training data.
    pipeline.fit(X_train, y_train)

    # Held-out evaluation (once; no hyperparameter search on this set).
    evaluation = evaluate_pipeline(pipeline, X_test, y_test)
    save_evaluation_outputs(pipeline, evaluation)

    metadata = {
        "dataset_path": str(processed_path),
        "n_total_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "split_strategy": "stratified train_test_split",
        "test_size_rationale": (
            "TEST_SIZE=0.20 keeps most of the small 243-row sample for training "
            "while retaining a stratified held-out test set (~20%) for unbiased "
            "evaluation. This is a practical choice for a tiny academic dataset, "
            "not an optimized production split."
        ),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_ENCODED_COLUMN,
        "target_mapping": {"not fire": 0, "fire": 1},
        "train_class_counts": {
            "0_not_fire": int((y_train == 0).sum()),
            "1_fire": int((y_train == 1).sum()),
        },
        "test_class_counts": {
            "0_not_fire": int((y_test == 0).sum()),
            "1_fire": int((y_test == 1).sum()),
        },
        "random_forest_hyperparameters": dict(RF_HYPERPARAMETERS),
        "pipeline_steps": ["preprocess", "model"],
        "model_path": str(MODEL_PIPELINE_PATH),
        "limitation_note": (
            "The dataset has only 243 cleaned rows from one country and one year "
            "(2012). Held-out metrics are useful for this academic prototype but "
            "must not be interpreted as real-world or production performance."
        ),
        "cross_validation_on_train_only": cv_results,
        "held_out_test_metrics": {
            "accuracy": evaluation["accuracy"],
            "precision": evaluation["precision"],
            "recall": evaluation["recall"],
            "f1": evaluation["f1"],
            "confusion_matrix": evaluation["confusion_matrix"],
        },
    }

    save_training_artifacts(pipeline, metadata)

    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "evaluation": evaluation,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def main() -> None:
    result = train_and_evaluate(run_cv=True)
    meta = result["metadata"]
    ev = result["evaluation"]

    print("=" * 60)
    print("MODEL TRAINING & EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total rows     : {meta['n_total_rows']}")
    print(f"Train / Test   : {meta['n_train']} / {meta['n_test']} (test_size={TEST_SIZE})")
    print(f"Train classes  : {meta['train_class_counts']}")
    print(f"Test classes   : {meta['test_class_counts']}")
    print(f"RF params      : {meta['random_forest_hyperparameters']}")
    print("--- Held-out TEST metrics (primary) ---")
    print(f"Accuracy       : {ev['accuracy']:.4f}")
    print(f"Precision      : {ev['precision']:.4f}")
    print(f"Recall         : {ev['recall']:.4f}")
    print(f"F1-score       : {ev['f1']:.4f}")
    print(f"Confusion mtx  : {ev['confusion_matrix']}")
    if meta.get("cross_validation_on_train_only"):
        print("--- Train-only 5-fold CV means (stability check; NOT the final test result) ---")
        for metric, vals in meta["cross_validation_on_train_only"].items():
            print(f"CV {metric:10s}: mean={vals['mean']:.4f} ± {vals['std']:.4f}")
    print(f"Saved pipeline : {MODEL_PIPELINE_PATH}")
    print(f"Saved metadata : {TRAINING_METADATA_PATH}")
    print(f"Saved metrics  : {EVALUATION_METRICS_PATH}")
    print(
        "NOTE: Small-sample academic prototype only; "
        "not production-ready / not real-world validated."
    )


if __name__ == "__main__":
    main()
