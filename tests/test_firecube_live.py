"""Offline/provider tests for the live FireCube adapter layer."""

from datetime import date

import numpy as np
import pytest

from src.firecube_inference import DEFAULT_CUBE, DEFAULT_MODEL, FireCubeExp5Inference
from src.firecube_live import (
    DailyValues,
    HistoricalCubeDynamicProvider,
    JsonCache,
    LiveFireCubeInference,
    ORNLModisEOProvider,
    ORNLModisLSTProvider,
    ORNLModisNDVIProvider,
    ProviderStatus,
    UnavailableEOProvider,
)


class EmptyEOProvider:
    def fetch_window(self, latitude, longitude, dates):
        return [DailyValues(d, {}, ProviderStatus("empty-eo", False, "test provider")) for d in dates]


def test_cache_round_trip(tmp_path):
    cache = JsonCache(tmp_path, ttl_seconds=60)
    cache.put("example", {"value": 1})
    assert cache.get("example") == {"value": 1}


def test_unavailable_eo_reports_missing_variables():
    rows = UnavailableEOProvider().fetch_window(38.0, 23.7, [date(2020, 7, 5)])
    assert rows[0].status.available is False
    assert set(rows[0].values) == {"ndvi", "lst_day", "lst_night"}


class StubModisLST(ORNLModisLSTProvider):
    def _available_dates(self, latitude, longitude):
        return [(date(2020, 7, 11), "A2020193")]

    def _subset(self, latitude, longitude, modis_date, band):
        return {"LST_Day_1km": 312.5, "LST_Night_1km": 299.0}[band]


class StubModisNDVI(ORNLModisNDVIProvider):
    def _available_dates(self, latitude, longitude):
        return [(date(2020, 7, 12), "A2020194")]

    def _subset(self, latitude, longitude, modis_date, band):
        assert band == "250m_16_days_NDVI"
        return 0.21


def test_modis_adapters_preserve_explicit_approximation_status():
    lst = StubModisLST().fetch_window(38.0, 23.7, [date(2020, 7, 15)])[0]
    ndvi = StubModisNDVI().fetch_window(38.0, 23.7, [date(2020, 7, 15)])[0]
    assert set(lst.values) == {"lst_day", "lst_night"}
    assert set(lst.status.approximate) == {"lst_day", "lst_night"}
    assert ndvi.values["ndvi"] == 0.21
    assert ndvi.status.approximate == ("ndvi",)


def test_combined_modis_provider_reports_missing_variable_without_silent_substitution():
    provider = ORNLModisEOProvider()
    provider.lst = StubModisLST()
    provider.ndvi = StubModisNDVI()
    row = provider.fetch_window(38.0, 23.7, [date(2020, 7, 15)])[0]
    assert row.status.available is True
    assert set(row.values) == {"ndvi", "lst_day", "lst_night"}
    assert set(row.status.approximate) == {"ndvi", "lst_day", "lst_night"}


@pytest.mark.integration
@pytest.mark.skipif(
    not DEFAULT_CUBE.exists() or not DEFAULT_MODEL.exists(),
    reason="local FireCube cube or Exp5 artifact is unavailable",
)
def test_historical_provider_reproduces_exp5_feature_vector():
    base = FireCubeExp5Inference()
    target = date(2020, 7, 15)
    expected = base.predict_historical(38.0, 23.7, target)
    live = LiveFireCubeInference(
        model_inference=base,
        weather_provider=HistoricalCubeDynamicProvider(),
        eo_provider=EmptyEOProvider(),
    )
    built = live.build_live_features(38.0, 23.7, target)
    np.testing.assert_allclose(
        np.asarray(built["feature_vector"]),
        np.asarray(list(expected["feature_values"].values())),
        rtol=0,
        atol=1e-5,
    )
    assert built["feature_count"] == 35
    assert built["window_dates"] == [f"2020-07-{d:02d}" for d in range(5, 15)]
