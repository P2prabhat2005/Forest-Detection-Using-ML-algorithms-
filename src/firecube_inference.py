"""Historical, read-only inference for the recovered Exp5 FireCube model.

This module intentionally contains no training, writes, live-data access, or
Streamlit dependencies.  Its feature construction follows
``src/benchmark_direct_10km.py`` exactly for one unit and one historical date.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import netCDF4 as nc
import numpy as np

from .benchmark_direct_10km import (
    CELL_KM2,
    DYNAMIC,
    DYNAMIC_AGG,
    DX_KM,
    DY_KM,
    HISTORY,
    STATIC,
    STATIC_AGG,
    agg_land_to_units,
    build_block_units,
    fill_time_nans,
    filled,
    window_slice,
    xgb_features,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUBE = ROOT / "data" / "raw" / "firecube" / "dataset_greece.nc"
DEFAULT_ARTIFACT_DIR = ROOT / "results" / "firecube_direct_10km_artifact_rerun"
DEFAULT_MODEL = DEFAULT_ARTIFACT_DIR / "xgb_direct_10km.joblib"
DEFAULT_FEATURE_CONFIG = DEFAULT_ARTIFACT_DIR / "feature_aggregation.json"
DEFAULT_EXP7_CALIBRATOR = ROOT / "results" / "firecube_exp7_two_stage" / "flat_exp5_calibrator.joblib"

CLC_VARS = [f"CLC_2018_{i}" for i in range(10)]
STRIDE_Y = 11
STRIDE_X = 14
FEATURE_COUNT = 35


class FireCubeInferenceError(RuntimeError):
    """Base error for invalid or unavailable historical inference inputs."""


@dataclass(frozen=True)
class LocationMapping:
    latitude: float
    longitude: float
    grid_y: int
    grid_x: int
    unit_id: int
    unit_area_km2: float


def _parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise FireCubeInferenceError(f"date must be ISO YYYY-MM-DD, got {value!r}") from exc


def _date_index(ds: Any, target: date) -> tuple[int, list[str]]:
    dates = nc.num2date(
        ds.variables["time"][:],
        ds.variables["time"].units,
        only_use_cftime_datetimes=False,
    )
    labels = [f"{d.year:04d}-{d.month:02d}-{d.day:02d}" for d in dates]
    wanted = target.isoformat()
    try:
        return labels.index(wanted), labels
    except ValueError as exc:
        raise FireCubeInferenceError(
            f"date {wanted} is not present in the FireCube time axis"
        ) from exc


def _load_feature_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


class FireCubeExp5Inference:
    """Construct Exp5-compatible features and predict one historical location."""

    def __init__(
        self,
        cube_path: Path = DEFAULT_CUBE,
        model_path: Path = DEFAULT_MODEL,
        feature_config_path: Path = DEFAULT_FEATURE_CONFIG,
        calibration_path: Path = DEFAULT_EXP7_CALIBRATOR,
    ) -> None:
        self.cube_path = Path(cube_path)
        self.model_path = Path(model_path)
        self.feature_config_path = Path(feature_config_path)
        self.calibration_path = Path(calibration_path)
        if not self.cube_path.exists():
            raise FileNotFoundError(self.cube_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.feature_config = _load_feature_config(self.feature_config_path)
        if self.feature_config["spatial_units"]["stride_cells"] != {"y": STRIDE_Y, "x": STRIDE_X}:
            raise FireCubeInferenceError("feature config stride does not match Exp5 (11, 14)")
        if self.feature_config["dynamic_spatial_agg"] != DYNAMIC_AGG:
            raise FireCubeInferenceError("feature config dynamic aggregation differs from Exp5 source")
        if self.feature_config["static_spatial_agg"] != STATIC_AGG:
            raise FireCubeInferenceError("feature config static aggregation differs from Exp5 source")
        artifact = joblib.load(self.model_path)
        try:
            self.model = artifact["xgb"]
            self.scaler_dyn = artifact["scaler_dyn"]
            self.scaler_sta = artifact["scaler_sta"]
        except KeyError as exc:
            raise FireCubeInferenceError(f"Exp5 artifact missing {exc.args[0]}") from exc
        model_features = getattr(self.model, "n_features_in_", FEATURE_COUNT)
        if int(model_features) != FEATURE_COUNT:
            raise FireCubeInferenceError(
                f"Exp5 model expects {model_features} features, expected {FEATURE_COUNT}"
            )
        self._calibrator = None
        if self.calibration_path.exists():
            calibrator = joblib.load(self.calibration_path)
            # Exp7 stores provenance alongside the isotonic estimator.
            self._calibrator = calibrator.get("calibrator", calibrator) if isinstance(calibrator, dict) else calibrator

    @staticmethod
    def feature_names() -> list[str]:
        return (
            [f"{v}__mean10" for v in DYNAMIC]
            + [f"{v}__last_tminus1" for v in DYNAMIC]
            + ["DEM__mean", "SLOPE__mean", "ROAD_DISTANCE__min", "WATERWAY_DISTANCE__min", "POP_DENS__mean"]
            + [f"CLC_2018_{i}__mean" for i in range(10)]
        )

    def _open_and_mapping(self, latitude: float, longitude: float) -> tuple[Any, LocationMapping, np.ndarray, dict]:
        ds = nc.Dataset(self.cube_path, "r")
        lat = float(latitude)
        lon = float(longitude)
        xs = np.asarray(ds.variables["x"][:], dtype=np.float64)
        ys = np.asarray(ds.variables["y"][:], dtype=np.float64)
        if not (float(xs.min()) <= lon <= float(xs.max()) and float(ys.min()) <= lat <= float(ys.max())):
            ds.close()
            raise FireCubeInferenceError("latitude/longitude is outside the FireCube Greece domain")
        grid_x = int(np.abs(xs - lon).argmin())
        grid_y = int(np.abs(ys - lat).argmin())
        land = filled(ds.variables["CLC_2018_9"]) < 0.5
        if not bool(land[grid_y, grid_x]):
            ds.close()
            raise FireCubeInferenceError("location maps to a non-land FireCube cell")
        land_y, land_x = np.where(land)
        units = build_block_units(land_y, land_x, STRIDE_Y, STRIDE_X, "10km")
        raw_key = (grid_y // STRIDE_Y) * (int(land_x.max()) + 1 + STRIDE_X) + (grid_x // STRIDE_X)
        keys = (land_y // STRIDE_Y) * (int(land_x.max()) + 1 + STRIDE_X) + (land_x // STRIDE_X)
        matching = np.flatnonzero(np.unique(keys) == raw_key)
        if matching.size != 1:
            ds.close()
            raise FireCubeInferenceError("could not map location to a FireCube 10 km unit")
        unit_id = int(matching[0])
        mapping = LocationMapping(lat, lon, grid_y, grid_x, unit_id, float(units["area_km2"][unit_id]))
        return ds, mapping, land, units

    def _features_for(self, ds: Any, target_index: int, unit_id: int, land: np.ndarray, units: dict, year: int) -> tuple[np.ndarray, dict[str, float]]:
        if target_index < HISTORY:
            raise FireCubeInferenceError("date has fewer than 16 preceding cube days; t-10..t-1 is unavailable")
        s0, e0 = window_slice()
        dyn_window = np.empty((1, HISTORY, len(DYNAMIC)), dtype=np.float32)
        land_y, land_x = np.where(land)
        uid = units["unit_id"]
        for k, var_name in enumerate(DYNAMIC):
            values = []
            for t in range(target_index - HISTORY, target_index):
                values.append(agg_land_to_units(
                    filled(ds.variables[var_name], t)[land], uid, units["n_units"], self.feature_config["dynamic_spatial_agg"][var_name]
                )[unit_id])
            dyn_window[0, :, k] = np.asarray(values, dtype=np.float32)
        dyn10 = dyn_window[:, s0:e0, :]
        dyn10 = fill_time_nans(dyn10)

        static = []
        for name in STATIC:
            static.append(agg_land_to_units(
                filled(ds.variables[name])[land], uid, units["n_units"], self.feature_config["static_spatial_agg"][name]
            )[unit_id])
        pop = agg_land_to_units(
            filled(ds.variables[f"POP_DENS_{year}"])[land], uid, units["n_units"], self.feature_config["static_spatial_agg"]["POP_DENS"]
        )[unit_id]
        static = np.asarray(static + [pop], dtype=np.float32).reshape(1, 5)
        clc = np.asarray([
            agg_land_to_units(filled(ds.variables[name])[land], uid, units["n_units"], "mean")[unit_id]
            for name in CLC_VARS
        ], dtype=np.float32).reshape(1, 10)
        sta = np.nan_to_num(static, nan=-1.0)
        clc = np.nan_to_num(clc, nan=0.0)
        dyn = np.asarray(dyn10, dtype=np.float32)
        dyn_n = self.scaler_dyn.transform(dyn.reshape(-1, dyn.shape[-1])).reshape(dyn.shape).astype(np.float32)
        sta_n = self.scaler_sta.transform(sta).astype(np.float32)
        x = xgb_features(dyn_n, sta_n, clc).astype(np.float32)
        if x.shape != (1, FEATURE_COUNT):
            raise FireCubeInferenceError(f"constructed feature shape {x.shape}, expected (1, {FEATURE_COUNT})")
        values = dict(zip(self.feature_names(), x[0].astype(float).tolist()))
        return x, values

    def predict_historical(
        self,
        latitude: float,
        longitude: float,
        target_date: str | date | datetime,
        calibration: str = "none",
    ) -> dict[str, Any]:
        """Predict for a date using only cube variables at dates strictly before it.

        ``calibration='exp7_flat'`` applies the Exp7 isotonic calibrator only to
        the compatible recovered Exp5 model score. Raw Exp5 probability remains
        available in ``raw_probability``.
        """
        target = _parse_date(target_date)
        ds, mapping, land, units = self._open_and_mapping(latitude, longitude)
        try:
            target_index, labels = _date_index(ds, target)
            x, feature_values = self._features_for(ds, target_index, mapping.unit_id, land, units, target.year)
            raw = float(self.model.predict_proba(x)[0, 1])
        finally:
            ds.close()
        calibrated = None
        if calibration == "exp7_flat":
            if self._calibrator is None:
                raise FireCubeInferenceError("Exp7 flat calibrator is not available")
            calibrated = float(self._calibrator.predict([raw])[0])
        elif calibration != "none":
            raise ValueError("calibration must be 'none' or 'exp7_flat'")
        return {
            "latitude": mapping.latitude,
            "longitude": mapping.longitude,
            "date": target.isoformat(),
            "cube_date_index": target_index,
            "grid_y": mapping.grid_y,
            "grid_x": mapping.grid_x,
            "unit_id": mapping.unit_id,
            "unit_area_km2": mapping.unit_area_km2,
            "feature_count": len(feature_values),
            "feature_names": self.feature_names(),
            "feature_values": feature_values,
            "raw_probability": raw,
            "calibrated_probability": calibrated,
            "model": "Exp5 direct 10 km XGBoost",
            "calibration": calibration,
            "predictor_window": "t-10..t-1",
            "data_source": str(self.cube_path),
        }
