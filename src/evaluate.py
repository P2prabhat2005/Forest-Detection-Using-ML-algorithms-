"""Evaluate a fitted training Pipeline on held-out data and save reports/figures."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    CONFUSION_MATRIX_FIG_PATH,
    EVALUATION_METRICS_PATH,
    FEATURE_IMPORTANCE_FIG_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_PIPELINE_PATH,
)


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Compute held-out classification metrics for the fitted pipeline."""
    y_pred = pipeline.predict(X_test)

    labels = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=["not fire", "fire"],
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=["not fire", "fire"],
        digits=4,
        zero_division=0,
    )

    return {
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": {
            "rows": "true label [not fire=0, fire=1]",
            "columns": "predicted label [not fire=0, fire=1]",
            "layout": [["TN", "FP"], ["FN", "TP"]],
        },
        "classification_report_dict": report,
        "classification_report_text": report_text,
        "y_true": y_test.astype(int).tolist(),
        "y_pred": [int(v) for v in y_pred],
    }


def plot_confusion_matrix(cm: list[list[int]], output_path: Path) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["not fire", "fire"],
        yticklabels=["not fire", "fire"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Held-out Test Set)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_feature_importance(pipeline: Pipeline, output_path: Path) -> Path | None:
    """Plot Random Forest feature importances on transformed feature names."""
    if "model" not in pipeline.named_steps:
        return None
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    if not hasattr(model, "feature_importances_"):
        return None

    names = list(preprocess.get_feature_names_out())
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    names_sorted = [names[i] for i in order]
    vals_sorted = importances[order]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(x=vals_sorted, y=names_sorted, ax=ax, color="#2f6f4e")
    ax.set_xlabel("Importance (Gini / impurity-based)")
    ax.set_ylabel("Feature")
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def extract_feature_importance(pipeline: Pipeline) -> dict[str, float]:
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    names = list(preprocess.get_feature_names_out())
    vals = model.feature_importances_
    paired = sorted(zip(names, vals), key=lambda x: x[1], reverse=True)
    return {name: round(float(val), 6) for name, val in paired}


def save_evaluation_outputs(
    pipeline: Pipeline,
    evaluation: dict,
    metrics_path: Path = EVALUATION_METRICS_PATH,
    cm_fig_path: Path = CONFUSION_MATRIX_FIG_PATH,
    fi_fig_path: Path = FEATURE_IMPORTANCE_FIG_PATH,
) -> dict:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    importance = extract_feature_importance(pipeline)
    evaluation_to_save = {
        k: v
        for k, v in evaluation.items()
        if k not in {"y_true", "y_pred"}
    }
    evaluation_to_save["feature_importance"] = importance
    evaluation_to_save["notes"] = (
        "Metrics are from a single stratified held-out test set on a small "
        "academic dataset (243 cleaned rows). They are not proof of real-world "
        "or production performance."
    )

    metrics_path.write_text(json.dumps(evaluation_to_save, indent=2), encoding="utf-8")
    plot_confusion_matrix(evaluation["confusion_matrix"], cm_fig_path)
    plot_feature_importance(pipeline, fi_fig_path)

    return {
        "metrics_path": str(metrics_path),
        "confusion_matrix_figure": str(cm_fig_path),
        "feature_importance_figure": str(fi_fig_path),
        "feature_importance": importance,
    }


def load_trained_pipeline(model_path: Path = MODEL_PIPELINE_PATH) -> Pipeline:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            "Run `python -m src.train` first."
        )
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(pipeline)}")
    return pipeline


def main() -> None:
    """Print held-out metrics from the saved evaluation file."""
    pipeline = load_trained_pipeline()
    if not EVALUATION_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {EVALUATION_METRICS_PATH}. "
            "Run `python -m src.train`."
        )
    metrics = json.loads(EVALUATION_METRICS_PATH.read_text(encoding="utf-8"))
    print("Loaded pipeline steps:", list(pipeline.named_steps.keys()))
    print("Held-out metrics from", EVALUATION_METRICS_PATH)
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in ["accuracy", "precision", "recall", "f1", "confusion_matrix"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
