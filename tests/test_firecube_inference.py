"""Tests for the standalone historical FireCube inference layer."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.firecube_inference import (
    DEFAULT_CUBE,
    DEFAULT_MODEL,
    FEATURE_COUNT,
    FireCubeExp5Inference,
    FireCubeInferenceError,
)
from src.benchmark_direct_10km import fill_time_nans


def test_feature_order_and_dimension():
    names = FireCubeExp5Inference.feature_names()
    assert len(names) == FEATURE_COUNT == 35
    assert names[:3] == ["max_t2m__mean10", "max_d2m__mean10", "min_rh__mean10"]
    assert names[10] == "max_t2m__last_tminus1"
    assert names[-1] == "CLC_2018_9__mean"


def test_missing_dynamic_values_match_exp5_fill_policy():
    values = np.array([[[np.nan, 2.0], [np.nan, np.nan], [4.0, np.nan]]], dtype=np.float32)
    filled_values = fill_time_nans(values)
    assert np.allclose(filled_values[0, :, 0], [4.0, 4.0, 4.0])
    assert np.allclose(filled_values[0, :, 1], [2.0, 2.0, 2.0])


def test_historical_inference_artifacts_exist():
    if not DEFAULT_CUBE.exists() or not DEFAULT_MODEL.exists():
        pytest.skip("local FireCube cube or Exp5 artifact is unavailable")
    assert DEFAULT_CUBE.stat().st_size > 20_000_000_000
    predictor = FireCubeExp5Inference()
    assert predictor.feature_config["spatial_units"]["n_units"] == 4022


@pytest.mark.integration
def test_historical_prediction_is_exp5_compatible():
    if not DEFAULT_CUBE.exists() or not DEFAULT_MODEL.exists():
        pytest.skip("local FireCube cube or Exp5 artifact is unavailable")
    result = FireCubeExp5Inference().predict_historical(38.0, 23.7, date(2020, 7, 15))
    assert result["feature_count"] == FEATURE_COUNT
    assert len(result["feature_values"]) == FEATURE_COUNT
    assert 0.0 <= result["raw_probability"] <= 1.0
    assert result["predictor_window"] == "t-10..t-1"
    assert result["date"] == "2020-07-15"


def test_outside_domain_rejected_without_model_call():
    if not DEFAULT_CUBE.exists() or not DEFAULT_MODEL.exists():
        pytest.skip("local FireCube cube or Exp5 artifact is unavailable")
    with pytest.raises(FireCubeInferenceError, match="outside"):
        FireCubeExp5Inference().predict_historical(0.0, 0.0, "2020-07-15")

