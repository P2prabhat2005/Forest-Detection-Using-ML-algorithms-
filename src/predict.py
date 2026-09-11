"""Inference helpers for the saved Random Forest Pipeline.

Loads ``models/random_forest_pipeline.joblib`` (preprocess + model) and applies
the exact fitted preprocessing from training. Does not retrain.

Expected input features (approved training set):
    Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, Region, month

Example (programmatic)::

    from src.predict import predict_fire

    result = predict_fire(
        {
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
        }
    )
    print(result["prediction"], result["probability_fire"])

CLI integration demo::

    PYTHONPATH=. python -m src.predict --demo-from-processed 3
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    EXPECTED_REGIONS,
    FEATURE_COLUMNS,
    MODEL_PIPELINE_PATH,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    TARGET_MAPPING,
)

InputType = Union[Mapping[str, Any], pd.Series, pd.DataFrame, Sequence[Mapping[str, Any]]]

# Human-readable labels for API/CLI output
PREDICTION_LABELS = {
    0: "No Fire",
    1: "Fire",
}

VALID_MONTHS = set(range(1, 13))


class PredictionError(ValueError):
    """Raised when prediction inputs are missing or invalid."""


@lru_cache(maxsize=1)
def load_pipeline(model_path: str | None = None) -> Pipeline:
    """Load the fitted training Pipeline from disk (cached).

    The Pipeline contains:
    - ``preprocess``: ColumnTransformer fitted on training data
    - ``model``: RandomForestClassifier fitted on transformed training data
    """
    path = Path(model_path) if model_path else MODEL_PIPELINE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Trained pipeline not found: {path}. "
            "Run `python -m src.train` before prediction."
        )
    pipeline = joblib.load(path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(pipeline)}")
    expected_steps = {"preprocess", "model"}
    if set(pipeline.named_steps) != expected_steps:
        raise TypeError(
            f"Unexpected pipeline steps {list(pipeline.named_steps)}. "
            f"Expected {sorted(expected_steps)}."
        )
    return pipeline


def clear_pipeline_cache() -> None:
    """Clear the cached pipeline (useful in tests)."""
    load_pipeline.cache_clear()


def _to_dataframe(input_data: InputType) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    if isinstance(input_data, pd.Series):
        return input_data.to_frame().T
    if isinstance(input_data, Mapping):
        return pd.DataFrame([dict(input_data)])
    if isinstance(input_data, Sequence) and not isinstance(input_data, (str, bytes)):
        if len(input_data) == 0:
            raise PredictionError("input_data sequence is empty.")
        if not all(isinstance(row, Mapping) for row in input_data):
            raise PredictionError(
                "Sequence input must contain mappings/dicts of feature values."
            )
        return pd.DataFrame([dict(row) for row in input_data])
    raise PredictionError(
        "input_data must be a dict, list of dicts, pandas Series, or DataFrame."
    )


def validate_prediction_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce prediction inputs to the approved feature schema.

    Rules:
    - All training feature columns must be present
    - Numeric features must be coercible to numbers and non-null
    - Region must be one of the trained categories
    - month must be an integer in 1..12
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise PredictionError(
            f"Missing required features: {missing}. "
            f"Required features: {FEATURE_COLUMNS}."
        )

    out = df.loc[:, FEATURE_COLUMNS].copy()

    for col in NUMERIC_FEATURES:
        coerced = pd.to_numeric(out[col], errors="coerce")
        if coerced.isna().any():
            bad_idx = out.index[coerced.isna()].tolist()[:5]
            raise PredictionError(
                f"Feature '{col}' must be numeric and non-missing. "
                f"Problem at row index/positions: {bad_idx}."
            )
        out[col] = coerced

    regions = out["Region"].astype(str).str.strip()
    invalid_region = ~regions.isin(EXPECTED_REGIONS)
    if invalid_region.any():
        bad_vals = sorted(set(regions[invalid_region].tolist()))
        raise PredictionError(
            f"Invalid Region value(s): {bad_vals}. "
            f"Expected one of {sorted(EXPECTED_REGIONS)}."
        )
    out["Region"] = regions

    month = pd.to_numeric(out["month"], errors="coerce")
    if month.isna().any():
        raise PredictionError("Feature 'month' must be numeric (integer 1..12).")
    if not ((month % 1) == 0).all():
        raise PredictionError("Feature 'month' must be an integer in 1..12.")
    month_int = month.astype(int)
    invalid_month = ~month_int.isin(VALID_MONTHS)
    if invalid_month.any():
        bad_vals = sorted(set(month_int[invalid_month].tolist()))
        raise PredictionError(
            f"Invalid month value(s): {bad_vals}. Expected an integer in 1..12."
        )
    out["month"] = month_int

    if out.isna().any().any():
        na_cols = out.columns[out.isna().any()].tolist()
        raise PredictionError(f"Missing values after validation in columns: {na_cols}.")

    return out


def _format_result(pred_label: int, proba_fire: float | None) -> dict[str, Any]:
    return {
        "prediction": PREDICTION_LABELS[int(pred_label)],
        "prediction_encoded": int(pred_label),
        "probability_fire": None if proba_fire is None else float(proba_fire),
        "probability_no_fire": None
        if proba_fire is None
        else float(1.0 - float(proba_fire)),
        "class_names": {"0": "No Fire", "1": "Fire"},
        "target_mapping": dict(TARGET_MAPPING),
    }


def predict_fire(
    input_data: InputType,
    model_path: str | Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict Fire / No Fire for one or more feature rows.

    Parameters
    ----------
    input_data:
        A single feature dict, a list of dicts, a pandas Series, or a DataFrame
        containing the approved feature columns.
    model_path:
        Optional override path to the saved Pipeline. Defaults to
        ``models/random_forest_pipeline.joblib``.

    Returns
    -------
    dict or list[dict]
        Single-row inputs return one dict. Multi-row inputs return a list.
        Each result includes:
        - ``prediction``: ``"Fire"`` or ``"No Fire"``
        - ``prediction_encoded``: ``1`` or ``0``
        - ``probability_fire`` / ``probability_no_fire`` when available
    """
    pipeline = load_pipeline(str(model_path) if model_path else None)
    features = validate_prediction_input(_to_dataframe(input_data))

    preds = pipeline.predict(features)
    proba_fire_arr = None
    if hasattr(pipeline, "predict_proba"):
        classes = list(pipeline.named_steps["model"].classes_)
        proba = pipeline.predict_proba(features)
        if 1 in classes:
            fire_idx = classes.index(1)
            proba_fire_arr = proba[:, fire_idx]

    results = []
    for i, pred in enumerate(preds):
        p_fire = None if proba_fire_arr is None else float(proba_fire_arr[i])
        results.append(_format_result(int(pred), p_fire))

    single_input = isinstance(input_data, (Mapping, pd.Series))
    if single_input and len(results) == 1:
        return results[0]
    return results


def demo_from_processed(n_rows: int = 3) -> list[dict[str, Any]]:
    """Integration demo: predict a few rows from the cleaned processed CSV.

    This is a smoke/integration check only — not a new model evaluation report.
    """
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_DATA_PATH}. "
            "Run data preparation first."
        )
    if n_rows < 1:
        raise PredictionError("n_rows must be >= 1.")

    df = pd.read_csv(PROCESSED_DATA_PATH)
    sample = df.head(n_rows)
    preds = predict_fire(sample.loc[:, FEATURE_COLUMNS])
    if isinstance(preds, dict):
        preds = [preds]

    demo_rows = []
    for i, (_, row) in enumerate(sample.iterrows()):
        actual_encoded = row.get("Classes_encoded")
        demo_rows.append(
            {
                "source": "processed_csv_integration_demo",
                "row_index": int(i),
                "actual_classes": row.get("Classes"),
                "actual_encoded": int(actual_encoded)
                if pd.notna(actual_encoded)
                else None,
                "prediction_result": preds[i],
            }
        )
    return demo_rows


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict Fire / No Fire using the saved Random Forest Pipeline. "
            "Does not retrain the model."
        )
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Path to a JSON file with one object or a list of feature objects.",
    )
    parser.add_argument(
        "--demo-from-processed",
        type=int,
        nargs="?",
        const=3,
        metavar="N",
        help=(
            "Integration demo: run predictions on the first N rows of the "
            "cleaned processed dataset (default N=3). Not a new evaluation metric."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_cli()
    args = parser.parse_args(argv)

    if args.demo_from_processed is not None:
        demo = demo_from_processed(args.demo_from_processed)
        print("=== Prediction module integration demo ===")
        print("Source: cleaned processed CSV (unchanged)")
        print("Note: demonstration only — not a new evaluation metric report.")
        print(json.dumps(demo, indent=2))
        return

    if args.json:
        path = Path(args.json)
        if not path.exists():
            raise FileNotFoundError(f"JSON input not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = predict_fire(payload)
        print(json.dumps(result, indent=2))
        return

    parser.print_help()
    print(
        "\nExample:\n"
        "  PYTHONPATH=. python -m src.predict --demo-from-processed 3\n"
    )


if __name__ == "__main__":
    main()
