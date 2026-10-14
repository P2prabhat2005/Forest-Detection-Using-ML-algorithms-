"""Provider-independent Track-B production feature pipeline.

This module deliberately does not import or modify the research experiment
artifacts.  It uses the same historical sample convention (the ten days
immediately before the label day) so the existing labels can support a first
tabular baseline, while recording the source and approximation status of
every live request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .benchmark_direct_10km import DYNAMIC
from .firecube_inference import FireCubeExp5Inference
from .firecube_live import (
    CLC_VARS,
    DailyValues,
    DynamicProvider,
    OpenMeteoWeatherProvider,
    ORNLModisEOProvider,
    ProviderStatus,
)

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_MODEL_DIR = ROOT / "models" / "production"
PRODUCTION_PREPROCESSOR = PRODUCTION_MODEL_DIR / "feature_preprocessor.joblib"
PRODUCTION_FEATURE_CONFIG = PRODUCTION_MODEL_DIR / "feature_schema.json"

TEN_DAY_COUNT = 10
STATIC_COUNT = 5
CLC_COUNT = 10
FEATURE_COUNT = len(DYNAMIC) * 2 + STATIC_COUNT + CLC_COUNT


def production_feature_names() -> list[str]:
    return (
        [f"{name}__mean10" for name in DYNAMIC]
        + [f"{name}__last_tminus1" for name in DYNAMIC]
        + ["DEM__mean", "SLOPE__mean", "ROAD_DISTANCE__min", "WATERWAY_DISTANCE__min", "POP_DENS__mean"]
        + [f"CLC_2018_{i}__mean" for i in range(CLC_COUNT)]
    )


class StaticFeatureProvider(Protocol):
    def fetch_static(self, latitude: float, longitude: float, year: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]: ...


class FireCubeStaticFeatureProvider:
    """Static Greece geography source reused as a read-only production input."""

    def __init__(self, inference: FireCubeExp5Inference | None = None) -> None:
        self.inference = inference or FireCubeExp5Inference()

    def fetch_static(self, latitude: float, longitude: float, year: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        ds, mapping, land, units = self.inference._open_and_mapping(latitude, longitude)
        try:
            unit_ids = units["unit_id"]
            from .benchmark_direct_10km import STATIC, STATIC_AGG, agg_land_to_units, filled

            static = [
                agg_land_to_units(filled(ds.variables[name])[land], unit_ids, units["n_units"], STATIC_AGG[name])[mapping.unit_id]
                for name in STATIC
            ]
            pop_year = min(max(year, 2009), 2021)
            static.append(
                agg_land_to_units(
                    filled(ds.variables[f"POP_DENS_{pop_year}"])[land],
                    unit_ids,
                    units["n_units"],
                    STATIC_AGG["POP_DENS"],
                )[mapping.unit_id]
            )
            clc = [
                agg_land_to_units(filled(ds.variables[name])[land], unit_ids, units["n_units"], "mean")[mapping.unit_id]
                for name in CLC_VARS
            ]
            return (
                np.asarray(static, dtype=np.float32),
                np.asarray(clc, dtype=np.float32),
                {
                    "provider": "FireCube static geography",
                    "data_date": "historical static raster (date unspecified)",
                    "population_year": pop_year,
                    "land_cover_year": 2018,
                    "unit_id": mapping.unit_id,
                    "grid_y": mapping.grid_y,
                    "grid_x": mapping.grid_x,
                    "historical_static_fallback": year > 2021,
                },
            )
        finally:
            ds.close()


@dataclass
class ProductionPreprocessor:
    """Median imputation plus standardisation learned only from training data."""

    feature_names: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler

    @classmethod
    def fit(cls, raw_features: np.ndarray) -> "ProductionPreprocessor":
        names = production_feature_names()
        if raw_features.ndim != 2 or raw_features.shape[1] != len(names):
            raise ValueError(f"expected (n, {len(names)}) raw features, got {raw_features.shape}")
        imputer = SimpleImputer(strategy="median")
        filled = imputer.fit_transform(raw_features)
        scaler = StandardScaler().fit(filled)
        return cls(names, imputer, scaler)

    def transform(self, raw_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(raw_features, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != len(self.feature_names):
            raise ValueError(f"expected (n, {len(self.feature_names)}) raw features, got {raw.shape}")
        missing = ~np.isfinite(raw)
        transformed = self.scaler.transform(self.imputer.transform(raw)).astype(np.float32)
        return transformed, missing

    def save(self, path: Path = PRODUCTION_PREPROCESSOR) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path = PRODUCTION_PREPROCESSOR) -> "ProductionPreprocessor":
        obj = joblib.load(path)
        if not isinstance(obj, cls) or obj.feature_names != production_feature_names():
            raise ValueError("production preprocessor schema does not match current production features")
        return obj


@dataclass
class ProductionFeatureRequest:
    latitude: float
    longitude: float
    target_date: str
    raw_features: np.ndarray
    model_features: np.ndarray
    feature_names: list[str]
    feature_values: dict[str, float]
    missing_features: list[str]
    feature_provenance: dict[str, dict[str, Any]]
    provider_status: dict[str, list[dict[str, Any]]]
    location_metadata: dict[str, Any]
    schema: str = "track_b_production_v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "target_date": self.target_date,
            "schema": self.schema,
            "feature_names": self.feature_names,
            "raw_feature_values": dict(zip(self.feature_names, self.raw_features.tolist())),
            "feature_values": self.feature_values,
            "missing_features": self.missing_features,
            "feature_provenance": self.feature_provenance,
            "provider_status": self.provider_status,
            "location_metadata": self.location_metadata,
        }


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


class ProductionFeaturePipeline:
    """Build a normalized Track-B input without knowing which model consumes it."""

    def __init__(
        self,
        dynamic_provider: DynamicProvider | None = None,
        static_provider: StaticFeatureProvider | None = None,
        preprocessor: ProductionPreprocessor | None = None,
    ) -> None:
        self.dynamic_provider = dynamic_provider or _CombinedLiveProvider()
        self.static_provider = static_provider or FireCubeStaticFeatureProvider()
        self.preprocessor = preprocessor

    def build(self, latitude: float, longitude: float, target_date: str | date | datetime) -> ProductionFeatureRequest:
        target = _as_date(target_date)
        window = [target - timedelta(days=i) for i in range(TEN_DAY_COUNT, 0, -1)]
        # Reject unsupported coordinates before calling any external provider.
        static, clc, location = self.static_provider.fetch_static(latitude, longitude, target.year)
        rows = self.dynamic_provider.fetch_window(latitude, longitude, window)
        by_date = {row.date: row for row in rows}
        raw_dynamic = np.full((TEN_DAY_COUNT, len(DYNAMIC)), np.nan, dtype=np.float32)
        provenance: dict[str, dict[str, Any]] = {}
        statuses: list[dict[str, Any]] = []
        for i, day in enumerate(window):
            row = by_date.get(day)
            if row is None:
                statuses.append(ProviderStatus("combined", False, "missing provider row").__dict__)
                continue
            statuses.append(row.status.__dict__)
            for k, name in enumerate(DYNAMIC):
                value = row.values.get(name, np.nan)
                raw_dynamic[i, k] = float(value) if np.isfinite(value) else np.nan
                provenance.setdefault(name, {"sources": [], "dates": [], "approximate": False})
                source = row.metadata.get("variable_sources", {}).get(name, row.status.provider)
                if source not in provenance[name]["sources"]:
                    provenance[name]["sources"].append(source)
                if np.isfinite(value):
                    source_date = row.metadata.get("variable_dates", {}).get(name, day.isoformat())
                    if source_date:
                        provenance[name]["dates"].append(source_date)
                provenance[name].setdefault("missing_days", [])
                if not np.isfinite(value):
                    provenance[name]["missing_days"].append(day.isoformat())
                if name in row.status.approximate:
                    provenance[name]["approximate"] = True

        counts = np.isfinite(raw_dynamic).sum(axis=0)
        mean = np.divide(np.nansum(raw_dynamic, axis=0), counts, out=np.full(len(DYNAMIC), np.nan), where=counts > 0)
        raw = np.concatenate([mean, raw_dynamic[-1], static, clc]).astype(np.float32)
        if raw.shape != (FEATURE_COUNT,):
            raise ValueError(f"production raw feature shape {raw.shape}, expected {(FEATURE_COUNT,)}")
        if self.preprocessor is None:
            model_input = np.nan_to_num(raw, nan=0.0).reshape(1, -1).astype(np.float32)
            missing_mask = ~np.isfinite(raw)
        else:
            model_input, missing_mask_2d = self.preprocessor.transform(raw.reshape(1, -1))
            missing_mask = missing_mask_2d[0]
        names = production_feature_names()
        dynamic_provenance = dict(provenance)
        provenance = {}
        for name in DYNAMIC:
            base = dynamic_provenance.get(name, {"sources": [], "dates": [], "approximate": False})
            provenance[f"{name}__mean10"] = {**base, "aggregation": "mean10"}
            provenance[f"{name}__last_tminus1"] = {**base, "aggregation": "last_tminus1"}
        for name in names[len(DYNAMIC) * 2:]:
            data_date = str(location.get("data_date", target.year))
            if name.startswith("CLC_"):
                data_date = str(location.get("land_cover_year", data_date))
            elif name.startswith("POP_DENS"):
                data_date = str(location.get("population_year", data_date))
            provenance[name] = {
                "sources": [location.get("provider", "static")],
                "dates": [data_date],
                "approximate": bool(location.get("historical_static_fallback", False)),
            }
        missing = [name for name, absent in zip(names, missing_mask) if absent]
        return ProductionFeatureRequest(
            float(latitude), float(longitude), target.isoformat(), raw, model_input[0], names,
            dict(zip(names, model_input[0].astype(float).tolist())), missing, provenance,
            {"dynamic": statuses}, location,
        )


class _CombinedLiveProvider:
    """Merge weather/soil and optional EO rows without changing providers."""

    def __init__(self) -> None:
        self.weather = OpenMeteoWeatherProvider()
        self.eo = ORNLModisEOProvider()

    def fetch_window(self, latitude: float, longitude: float, dates: list[date]) -> list[DailyValues]:
        weather = self.weather.fetch_window(latitude, longitude, dates)
        eo = self.eo.fetch_window(latitude, longitude, dates)
        by_day = {}
        for left, right in zip(weather, eo):
            values = {**left.values, **right.values}
            approximate = tuple(sorted(set(left.status.approximate + right.status.approximate)))
            provider = f"{left.status.provider}+{right.status.provider}"
            missing = tuple(name for name in DYNAMIC if not np.isfinite(values.get(name, np.nan)))
            status = ProviderStatus(provider, not missing, left.status.message + "; " + right.status.message, tuple(values), approximate)
            metadata = {**right.metadata, "variable_sources": {
                **{name: "Open-Meteo hourly weather / soil moisture" for name in left.values},
                **{name: "ORNL MOD13Q1" if name == "ndvi" else "ORNL MOD11A2" for name in right.values},
            }}
            by_day[left.date] = DailyValues(left.date, values, status, metadata)
        return [by_day.get(day, DailyValues(day, {}, ProviderStatus("combined", False, "missing provider row"))) for day in dates]


def raw_training_features(samples_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Track-B raw features, labels, and years from existing samples."""
    data = np.load(samples_path)
    dynamic = np.asarray(data["dynamic"], dtype=np.float32)[:, -TEN_DAY_COUNT:, :]
    raw = np.concatenate([
        np.nanmean(dynamic, axis=1),
        dynamic[:, -1, :],
        np.asarray(data["static"], dtype=np.float32),
        np.asarray(data["clc"], dtype=np.float32),
    ], axis=1)
    return raw, np.asarray(data["label"], dtype=np.int8), np.asarray(data["year"], dtype=np.int16)


def write_feature_schema(path: Path = PRODUCTION_FEATURE_CONFIG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "track_b_production_v1",
        "feature_names": production_feature_names(),
        "dynamic_window": "t-10..t-1",
        "aggregation": {"dynamic_mean": "mean10", "dynamic_last": "last_tminus1"},
        "research_equivalence": False,
    }, indent=2), encoding="utf-8")
