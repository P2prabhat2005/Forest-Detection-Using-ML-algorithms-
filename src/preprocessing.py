"""Reusable sklearn preprocessing for training and inference.

Builds a ColumnTransformer that:
- scales approved numeric features
- one-hot encodes Region

Random Forest does not require scaling; StandardScaler is included so the
same fitted transformer can be reused consistently at inference time.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from src.validation import validate_feature_frame


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted preprocessor for the approved feature set."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="error", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline() -> Pipeline:
    """Pipeline wrapper around the shared preprocessor (extendable later)."""
    return Pipeline(steps=[("preprocess", build_preprocessor())])


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order approved model features from a cleaned frame."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot select features; missing columns: {missing}")
    X = df.loc[:, FEATURE_COLUMNS].copy()
    validate_feature_frame(X, "select_features")
    return X


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Feature names after transform (requires a fitted preprocessor)."""
    return list(preprocessor.get_feature_names_out())
