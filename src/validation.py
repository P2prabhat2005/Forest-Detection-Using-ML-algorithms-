"""Fail-fast validation helpers for dataset loading and preprocessing."""

from __future__ import annotations

import pandas as pd

from src.config import (
    EXPECTED_REGIONS,
    EXPECTED_TARGET_LABELS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TARGET_ENCODED_COLUMN,
)


class DataValidationError(ValueError):
    """Raised when dataset structure or values do not match project expectations."""


def require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{context}: missing expected columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_target_labels(series: pd.Series, context: str) -> None:
    labels = set(series.dropna().astype(str).str.strip().unique())
    unexpected = labels - EXPECTED_TARGET_LABELS
    if unexpected:
        raise DataValidationError(
            f"{context}: unexpected target labels {sorted(unexpected)}. "
            f"Expected only {sorted(EXPECTED_TARGET_LABELS)}."
        )
    if series.isna().any():
        raise DataValidationError(f"{context}: target contains missing values.")


def validate_regions(series: pd.Series, context: str) -> None:
    regions = set(series.dropna().astype(str).unique())
    unexpected = regions - EXPECTED_REGIONS
    if unexpected:
        raise DataValidationError(
            f"{context}: unexpected Region values {sorted(unexpected)}. "
            f"Expected only {sorted(EXPECTED_REGIONS)}."
        )


def validate_numeric_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    for col in columns:
        if col not in df.columns:
            raise DataValidationError(f"{context}: numeric column '{col}' is missing.")
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().any():
            bad_idx = df.index[coerced.isna()].tolist()[:5]
            raise DataValidationError(
                f"{context}: column '{col}' has non-numeric values "
                f"(example row index/positions: {bad_idx})."
            )


def validate_cleaned_frame(df: pd.DataFrame) -> None:
    """Validate the cleaned dataset before writing to data/processed/."""
    require_columns(
        df,
        FEATURE_COLUMNS + [TARGET_COLUMN, TARGET_ENCODED_COLUMN],
        "cleaned dataset",
    )
    validate_target_labels(df[TARGET_COLUMN], "cleaned dataset")
    validate_regions(df["Region"], "cleaned dataset")
    validate_numeric_columns(df, NUMERIC_FEATURES, "cleaned dataset")

    encoded = set(df[TARGET_ENCODED_COLUMN].unique())
    if encoded - {0, 1}:
        raise DataValidationError(
            f"cleaned dataset: Classes_encoded must be 0/1 only, found {sorted(encoded)}."
        )

    remapped = df[TARGET_COLUMN].astype(str).str.strip().map({"not fire": 0, "fire": 1})
    if not remapped.equals(df[TARGET_ENCODED_COLUMN]):
        raise DataValidationError(
            "cleaned dataset: Classes_encoded does not match "
            "approved mapping (not fire->0, fire->1)."
        )

    if df.isna().any().any():
        na_cols = df.columns[df.isna().any()].tolist()
        raise DataValidationError(
            f"cleaned dataset: unexpected missing values in columns {na_cols}."
        )


def validate_feature_frame(X: pd.DataFrame, context: str = "feature matrix") -> None:
    """Validate feature inputs against the approved feature set and order."""
    require_columns(X, FEATURE_COLUMNS, context)
    extra = [c for c in X.columns if c not in FEATURE_COLUMNS]
    if extra:
        raise DataValidationError(
            f"{context}: unexpected extra columns {extra}. "
            f"Expected exactly {FEATURE_COLUMNS}."
        )
    if list(X.columns) != FEATURE_COLUMNS:
        raise DataValidationError(
            f"{context}: columns must be in order {FEATURE_COLUMNS}. "
            f"Got {list(X.columns)}."
        )
    validate_regions(X["Region"], context)
    validate_numeric_columns(X, NUMERIC_FEATURES, context)
