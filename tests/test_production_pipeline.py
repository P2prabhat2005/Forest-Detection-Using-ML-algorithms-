"""Offline tests for the provider-independent Track-B foundation."""

from datetime import date, timedelta

import numpy as np

from src.firecube_live import DailyValues, ProviderStatus
from src.production_features import (
    FEATURE_COUNT,
    ProductionFeaturePipeline,
    ProductionPreprocessor,
    production_feature_names,
)


class StaticStub:
    def fetch_static(self, latitude, longitude, year):
        return np.arange(5, dtype=np.float32), np.arange(10, dtype=np.float32), {"provider": "stub-static"}


class DynamicStub:
    def fetch_window(self, latitude, longitude, dates):
        rows = []
        for i, day in enumerate(dates):
            values = {name: float(i + k) for k, name in enumerate((
                "max_t2m", "max_d2m", "min_rh", "max_tp", "max_sp", "max_wind_speed",
                "ndvi", "lst_day", "lst_night", "sminx",
            ))}
            if i == 3:
                values["ndvi"] = float("nan")
            rows.append(DailyValues(day, values, ProviderStatus("stub", True, "test", tuple(values), ("ndvi",))))
        return rows


def test_production_pipeline_has_stable_35_feature_schema_and_provenance():
    pipe = ProductionFeaturePipeline(DynamicStub(), StaticStub())
    request = pipe.build(38.0, 23.7, date(2020, 7, 15))
    assert request.raw_features.shape == (FEATURE_COUNT,)
    assert request.model_features.shape == (FEATURE_COUNT,)
    assert request.feature_names == production_feature_names()
    raw_feature_values = request.as_dict()["raw_feature_values"]
    assert raw_feature_values["ndvi__mean10"] != raw_feature_values["ndvi__last_tminus1"]
    assert request.feature_provenance["ndvi__mean10"]["approximate"] is True
    assert request.location_metadata["provider"] == "stub-static"


def test_preprocessor_fits_and_rejects_wrong_schema():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(8, FEATURE_COUNT)).astype(np.float32)
    raw[0, 2] = np.nan
    prep = ProductionPreprocessor.fit(raw)
    transformed, missing = prep.transform(raw[:1])
    assert transformed.shape == (1, FEATURE_COUNT)
    assert bool(missing[0, 2]) is True
