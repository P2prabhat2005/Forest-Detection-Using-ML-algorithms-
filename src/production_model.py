"""Replaceable model interface for the Track-B production path."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np

from .production_features import FEATURE_COUNT, ProductionFeatureRequest, production_feature_names


class ProductionModel(Protocol):
    model_name: str
    feature_names: list[str]

    def predict_probability(self, features: np.ndarray) -> float: ...


@dataclass
class XGBoostProductionModel:
    """Artifact-backed XGBoost model with strict production schema checks."""

    estimator: Any
    feature_names: list[str]
    model_name: str = "Track-B production XGBoost baseline"

    @classmethod
    def load(cls, path: Path) -> "XGBoostProductionModel":
        artifact = joblib.load(path)
        estimator = artifact.get("xgb", artifact.get("model")) if isinstance(artifact, dict) else artifact
        names = artifact.get("feature_names") if isinstance(artifact, dict) else None
        names = names or production_feature_names()
        if list(names) != production_feature_names():
            raise ValueError("production model feature schema does not match track_b_production_v1")
        if estimator is None or not hasattr(estimator, "predict_proba"):
            raise ValueError("production artifact does not contain a predict_proba estimator")
        expected = getattr(estimator, "n_features_in_", FEATURE_COUNT)
        if int(expected) != FEATURE_COUNT:
            raise ValueError(f"production model expects {expected} features, expected {FEATURE_COUNT}")
        return cls(estimator, list(names))

    def predict_probability(self, features: np.ndarray) -> float:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != FEATURE_COUNT:
            raise ValueError(f"production model requires {FEATURE_COUNT} features, got {x.shape[1]}")
        return float(self.estimator.predict_proba(x)[0, 1])


@dataclass
class ProductionPredictionService:
    pipeline: Any
    model: ProductionModel

    def predict_recent(self, latitude: float, longitude: float, weather_provider=None) -> dict[str, Any]:
        """Today's target, ten completed UTC days, and separate current weather."""
        from .firecube_live import OpenMeteoWeatherProvider
        now = datetime.now(timezone.utc)
        result = self.predict(latitude, longitude, now.date().isoformat())
        weather_names = ("max_t2m", "max_d2m", "min_rh", "max_tp", "max_sp", "max_wind_speed")
        if all(f"{name}__mean10" in result["missing_features"] for name in weather_names):
            raise RuntimeError("Recent weather history is unavailable; prediction withheld. Try again later.")
        result["current_weather"] = (weather_provider or OpenMeteoWeatherProvider()).fetch_current(latitude, longitude)
        result["requested_at"] = now.isoformat()
        result["history_window"] = "t-10..t-1 (completed UTC days)"
        return result

    def predict(self, latitude: float, longitude: float, target_date: str) -> dict[str, Any]:
        request: ProductionFeatureRequest = self.pipeline.build(latitude, longitude, target_date)
        probability = self.model.predict_probability(request.model_features)
        result = request.as_dict()
        result.update({
            "model": self.model.model_name,
            "raw_probability": probability,
            "operational_validity": False,
            "validity_note": "Track-B baseline; not yet operationally validated or FireCube-equivalent",
        })
        return result


def save_model_manifest(path: Path, model_path: Path, training_metadata: dict[str, Any]) -> None:
    """Write a small manifest next to a production artifact, never to research results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "track_b_production_v1",
        "model_path": str(model_path),
        "feature_names": production_feature_names(),
        "training": training_metadata,
        "operational_validity": False,
    }, indent=2), encoding="utf-8")
