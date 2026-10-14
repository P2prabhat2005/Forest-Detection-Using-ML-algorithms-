"""Provider-based live-input layer for the existing Exp5 FireCube model.

The model and its preprocessing remain in ``firecube_inference``.  This module
only acquires/normalises recent inputs and assembles the same 35-column vector.
No result or model artifact is written.  Live EO is deliberately an interface
with an explicit unavailable fallback until a licensed/public EO adapter is
selected and validated.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import netCDF4 as nc

from .benchmark_direct_10km import DYNAMIC, HISTORY, STATIC, STATIC_AGG, agg_land_to_units, build_block_units, fill_time_nans, filled, window_slice, xgb_features
from .firecube_inference import (
    DEFAULT_CUBE,
    DEFAULT_MODEL,
    DEFAULT_FEATURE_CONFIG,
    FEATURE_COUNT,
    FireCubeExp5Inference,
    FireCubeInferenceError,
)

LIVE_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "firecube_live"
STRIDE_Y = 11
STRIDE_X = 14
CLC_VARS = [f"CLC_2018_{i}" for i in range(10)]


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    available: bool
    message: str
    variables: tuple[str, ...] = ()
    approximate: tuple[str, ...] = ()


@dataclass
class DailyValues:
    date: date
    values: dict[str, float]
    status: ProviderStatus
    metadata: dict[str, Any] = field(default_factory=dict)


class DynamicProvider(Protocol):
    def fetch_window(self, latitude: float, longitude: float, dates: list[date]) -> list[DailyValues]: ...


class JsonCache:
    """Small file cache for provider responses; never writes research outputs."""

    def __init__(self, directory: Path = LIVE_CACHE_DIR, ttl_seconds: int = 6 * 3600) -> None:
        self.directory = Path(directory)
        self.ttl_seconds = ttl_seconds

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists() or time.time() - path.stat().st_mtime > self.ttl_seconds:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, payload: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(json.dumps(payload), encoding="utf-8")


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


class OpenMeteoWeatherProvider:
    """Open-Meteo adapter for the weather subset of Exp5.

    Values are point/grid-cell estimates, so they are an operational
    approximation of Exp5's 10 km spatial aggregates. Units are converted to
    FireCube conventions: K, fraction RH, Pa, m/s, and metres precipitation.
    """

    endpoint = "https://archive-api.open-meteo.com/v1/archive"
    forecast_endpoint = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, cache: JsonCache | None = None, timeout_seconds: int = 30) -> None:
        self.cache = cache or JsonCache()
        self.timeout_seconds = timeout_seconds

    def _request(self, latitude: float, longitude: float, start: date, end: date) -> dict[str, Any]:
        endpoint = self.forecast_endpoint if end >= date.today() - timedelta(days=5) else self.endpoint
        params = {
            "latitude": f"{latitude:.5f}", "longitude": f"{longitude:.5f}",
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "hourly": ",".join([
                "temperature_2m", "dew_point_2m", "relative_humidity_2m",
                "precipitation", "surface_pressure", "wind_speed_10m",
                "soil_moisture_0_to_7cm",
            ]),
            "timezone": "UTC",
        }
        key = endpoint + "?" + urlencode(params)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        request = Request(key, headers={"User-Agent": "Forest-fire-Detection/1.0"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.cache.put(key, payload)
        return payload

    def fetch_window(self, latitude: float, longitude: float, dates: list[date]) -> list[DailyValues]:
        if not dates:
            return []
        try:
            payload = self._request(latitude, longitude, min(dates), max(dates))
        except Exception as exc:  # provider errors become status, not model crashes
            return [DailyValues(d, {}, ProviderStatus("open-meteo", False, str(exc))) for d in dates]
        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        by_day: dict[str, list[int]] = {}
        for i, stamp in enumerate(times):
            by_day.setdefault(str(stamp)[:10], []).append(i)
        out = []
        for d in dates:
            idx = by_day.get(d.isoformat(), [])
            def vals(name: str) -> list[float]:
                series = hourly.get(name, [])
                return [_safe_float(series[i]) if i < len(series) else float("nan") for i in idx]
            temp, dew, rh, precip, pressure, wind, soil = [vals(n) for n in (
                "temperature_2m", "dew_point_2m", "relative_humidity_2m",
                "precipitation", "surface_pressure", "wind_speed_10m", "soil_moisture_0_to_7cm")]
            def finite_max(v: list[float]) -> float:
                return float(np.nanmax(v)) if v and np.isfinite(v).any() else float("nan")
            def finite_min(v: list[float]) -> float:
                return float(np.nanmin(v)) if v and np.isfinite(v).any() else float("nan")
            values = {
                "max_t2m": finite_max(temp) + 273.15,
                "max_d2m": finite_max(dew) + 273.15,
                "min_rh": finite_min(rh) / 100.0,
                "max_tp": finite_max(precip) / 1000.0,
                "max_sp": finite_max(pressure) * 100.0,
                "max_wind_speed": finite_max(wind) / 3.6,
                # Open-Meteo soil moisture is not FireCube's EDO SMI. It is
                # retained as an explicitly flagged proxy so the vector can
                # be assembled while the EO provider remains replaceable.
                "sminx": finite_max(soil),
            }
            missing = tuple(k for k, v in values.items() if not math.isfinite(v))
            out.append(DailyValues(
                d, values,
                ProviderStatus(
                    "open-meteo", not missing,
                    "weather values available" if not missing else "missing weather variables: " + ", ".join(missing),
                    tuple(values),
                    ("max_tp", "max_sp", "sminx"),
                ),
            ))
        return out

    def fetch_current(self, latitude: float, longitude: float) -> dict[str, Any]:
        """Display-only current modelled weather; does not replace the history."""
        params = {
            "latitude": f"{latitude:.5f}", "longitude": f"{longitude:.5f}",
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,surface_pressure",
            "timezone": "UTC",
        }
        url = self.forecast_endpoint + "?" + urlencode(params)
        cache = JsonCache(self.cache.directory / "current", ttl_seconds=600)
        try:
            payload = cache.get(url)
            if payload is None:
                with urlopen(Request(url, headers={"User-Agent": "Forest-fire-Detection/1.0"}), timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                payload["retrieved_at"] = datetime.now(timezone.utc).isoformat()
                cache.put(url, payload)
            current = payload.get("current", {})
            return {"available": bool(current.get("time")), "source": self.forecast_endpoint,
                    "timestamp": current.get("time"), "timezone": "UTC",
                    "retrieved_at": payload.get("retrieved_at"), "values": current,
                    "units": payload.get("current_units", {}), "approximate": True,
                    "message": "Current modelled weather from Open-Meteo"}
        except Exception as exc:
            return {"available": False, "message": str(exc), "source": self.forecast_endpoint}


def _modis_date(d: date) -> str:
    return f"A{d.year:04d}{d.timetuple().tm_yday:03d}"


class ORNLModisLSTProvider:
    """Public ORNL TESViS adapter for MOD11A2 day/night LST.

    MOD11A2 is 1 km and 8-day composite, while FireCube stores MOD11A1
    daily 1 km LST. Values are therefore returned with an explicit temporal
    approximation status and are never presented as exact FireCube inputs.
    """

    endpoint = "https://modis.ornl.gov/rst/api/v1"
    product = "MOD11A2"

    def __init__(self, cache: JsonCache | None = None, timeout_seconds: int = 60) -> None:
        self.cache = cache or JsonCache(ttl_seconds=24 * 3600)
        self.timeout_seconds = timeout_seconds

    def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        key = self.endpoint + path + "?" + urlencode(params)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        request = Request(key, headers={"Accept": "application/json", "User-Agent": "Forest-fire-Detection/1.0"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.cache.put(key, payload)
        return payload

    def _available_dates(self, latitude: float, longitude: float) -> list[tuple[date, str]]:
        payload = self._get_json(
            f"/{self.product}/dates",
            {"latitude": f"{latitude:.5f}", "longitude": f"{longitude:.5f}"},
        )
        return [(date.fromisoformat(row["calendar_date"]), row["modis_date"]) for row in payload.get("dates", [])]

    def _subset(self, latitude: float, longitude: float, modis_date: str, band: str) -> float:
        payload = self._get_json(
            f"/{self.product}/subset",
            {
                "latitude": f"{latitude:.5f}", "longitude": f"{longitude:.5f}",
                "startDate": modis_date, "endDate": modis_date,
                "kmAboveBelow": 0, "kmLeftRight": 0, "band": band,
            },
        )
        rows = payload.get("subset", [])
        if not rows or not rows[0].get("data"):
            return float("nan")
        raw = _safe_float(rows[0]["data"][0])
        return raw * _safe_float(payload.get("scale", "0.02")) if math.isfinite(raw) else float("nan")

    def fetch_window(self, latitude: float, longitude: float, dates: list[date]) -> list[DailyValues]:
        try:
            available = self._available_dates(latitude, longitude)
        except Exception as exc:
            return [DailyValues(d, {}, ProviderStatus("ornl-mod11a2", False, str(exc))) for d in dates]
        out = []
        for d in dates:
            prior = [(when, code) for when, code in available if when <= d]
            if not prior:
                out.append(DailyValues(d, {}, ProviderStatus("ornl-mod11a2", False, "no prior MOD11A2 composite")))
                continue
            composite_date, code = prior[-1]
            try:
                values = {
                    "lst_day": self._subset(latitude, longitude, code, "LST_Day_1km"),
                    "lst_night": self._subset(latitude, longitude, code, "LST_Night_1km"),
                }
                missing = tuple(name for name, value in values.items() if not math.isfinite(value))
                message = f"MOD11A2 composite {composite_date.isoformat()} reused for {d.isoformat()}"
                out.append(DailyValues(
                    d, values,
                    ProviderStatus("ornl-mod11a2", not missing, message, tuple(values), ("lst_day", "lst_night")),
                    {"source_date": composite_date.isoformat()},
                ))
            except Exception as exc:
                out.append(DailyValues(d, {}, ProviderStatus("ornl-mod11a2", False, str(exc))))
        return out


class ORNLModisNDVIProvider(ORNLModisLSTProvider):
    """Public ORNL TESViS adapter for MOD13Q1 NDVI.

    MOD13Q1 is a 250 m, 16-day composite.  FireCube's ``ndvi`` is sourced
    from MOD13A2 (1 km, 16-day).  The cadence is comparable but the product
    and native resolution are not identical, so every returned value is
    explicitly marked approximate.
    """

    product = "MOD13Q1"

    def fetch_window(self, latitude: float, longitude: float, dates: list[date]) -> list[DailyValues]:
        try:
            available = self._available_dates(latitude, longitude)
        except Exception as exc:
            return [DailyValues(d, {}, ProviderStatus("ornl-mod13q1", False, str(exc))) for d in dates]
        out = []
        for d in dates:
            prior = [(when, code) for when, code in available if when <= d]
            if not prior:
                out.append(DailyValues(d, {}, ProviderStatus("ornl-mod13q1", False, "no prior MOD13Q1 composite")))
                continue
            composite_date, code = prior[-1]
            try:
                value = self._subset(latitude, longitude, code, "250m_16_days_NDVI")
                values = {"ndvi": value}
                missing = tuple(name for name, current in values.items() if not math.isfinite(current))
                message = f"MOD13Q1 composite {composite_date.isoformat()} reused for {d.isoformat()}"
                out.append(DailyValues(
                    d, values,
                    ProviderStatus("ornl-mod13q1", not missing, message, tuple(values), ("ndvi",)),
                    {"source_date": composite_date.isoformat()},
                ))
            except Exception as exc:
                out.append(DailyValues(d, {}, ProviderStatus("ornl-mod13q1", False, str(exc))))
        return out


class ORNLModisEOProvider:
    """Combined optional ORNL MODIS adapter for NDVI and day/night LST.

    This is intentionally opt-in.  It does not claim exact FireCube
    equivalence and keeps per-variable provider status in the inference
    result.  Weather and soil-moisture inputs remain the responsibility of
    the weather provider.
    """

    def __init__(self, cache: JsonCache | None = None, timeout_seconds: int = 60) -> None:
        cache = cache or JsonCache(ttl_seconds=24 * 3600)
        self.lst = ORNLModisLSTProvider(cache=cache, timeout_seconds=timeout_seconds)
        self.ndvi = ORNLModisNDVIProvider(cache=cache, timeout_seconds=timeout_seconds)

    def fetch_window(self, latitude: float, longitude: float, dates: list[date]) -> list[DailyValues]:
        lst_rows = self.lst.fetch_window(latitude, longitude, dates)
        ndvi_rows = self.ndvi.fetch_window(latitude, longitude, dates)
        out = []
        for lst_row, ndvi_row in zip(lst_rows, ndvi_rows):
            values = {**lst_row.values, **ndvi_row.values}
            statuses = (lst_row.status, ndvi_row.status)
            missing = tuple(name for name in ("ndvi", "lst_day", "lst_night") if not math.isfinite(values.get(name, float("nan"))))
            available = not missing
            message = "; ".join(f"{status.provider}: {status.message}" for status in statuses)
            out.append(DailyValues(
                lst_row.date,
                values,
                ProviderStatus("ornl-modis-eo", available, message, tuple(values), ("ndvi", "lst_day", "lst_night")),
                {"variable_dates": {**{name: lst_row.metadata.get("source_date") for name in lst_row.values},
                                    **{name: ndvi_row.metadata.get("source_date") for name in ndvi_row.values}}},
            ))
        return out


class UnavailableEOProvider:
    """Explicit EO fallback until a validated current EO adapter is selected."""

    def fetch_window(self, latitude: float, longitude: float, dates: list[date]) -> list[DailyValues]:
        missing = ("ndvi", "lst_day", "lst_night")
        status = ProviderStatus(
            "eo-unavailable", False,
            "No live EO adapter configured; Exp5 fill policy will be applied",
            missing,
        )
        return [DailyValues(d, {name: float("nan") for name in missing}, status) for d in dates]


class HistoricalCubeDynamicProvider:
    """Offline provider used for parity checks against the exact Exp5 pipeline."""

    def __init__(self, cube_path: Path = DEFAULT_CUBE) -> None:
        self.cube_path = Path(cube_path)

    def fetch_window(self, latitude: float, longitude: float, dates: list[date]) -> list[DailyValues]:
        base = FireCubeExp5Inference(cube_path=self.cube_path)
        ds, mapping, land, units = base._open_and_mapping(latitude, longitude)
        ds.close()
        return self.fetch_window_for_unit(mapping.unit_id, dates, units, land)

    def fetch_window_for_unit(self, unit_id: int, dates: list[date], units: dict, land: np.ndarray) -> list[DailyValues]:
        ds = nc.Dataset(self.cube_path, "r")
        try:
            times = nc.num2date(ds.variables["time"][:], ds.variables["time"].units, only_use_cftime_datetimes=False)
            labels = [f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in times]
            uid = units["unit_id"]
            out = []
            for d in dates:
                t = labels.index(d.isoformat())
                values: dict[str, float] = {}
                for name in DYNAMIC:
                    values[name] = float(agg_land_to_units(
                        filled(ds.variables[name], t)[land], uid, units["n_units"],
                        {"max_t2m": "max", "max_d2m": "max", "min_rh": "min", "max_tp": "mean",
                         "max_sp": "mean", "max_wind_speed": "max", "ndvi": "mean", "lst_day": "max",
                         "lst_night": "max", "sminx": "mean"}[name]
                    )[unit_id])
                out.append(DailyValues(d, values, ProviderStatus("historical-firecube", True, "exact cube values", tuple(values))))
            return out
        finally:
            ds.close()


class LiveFireCubeInference:
    """Combine providers with the unchanged Exp5 model preprocessing."""

    def __init__(self, model_inference: FireCubeExp5Inference | None = None,
                 weather_provider: DynamicProvider | None = None,
                 eo_provider: DynamicProvider | None = None) -> None:
        self.base = model_inference or FireCubeExp5Inference()
        self.weather_provider = weather_provider or OpenMeteoWeatherProvider()
        self.eo_provider = eo_provider or UnavailableEOProvider()

    def _static_features(self, latitude: float, longitude: float, year: int) -> tuple[Any, Any, Any, dict]:
        ds, mapping, land, units = self.base._open_and_mapping(latitude, longitude)
        try:
            unit_id = mapping.unit_id
            unit_ids = units["unit_id"]
            static = [agg_land_to_units(filled(ds.variables[name])[land], unit_ids, units["n_units"], STATIC_AGG[name])[unit_id] for name in STATIC]
            pop_year = min(max(year, 2009), 2021)
            static.append(agg_land_to_units(filled(ds.variables[f"POP_DENS_{pop_year}"])[land], unit_ids, units["n_units"], STATIC_AGG["POP_DENS"])[unit_id])
            clc = [agg_land_to_units(filled(ds.variables[name])[land], unit_ids, units["n_units"], "mean")[unit_id] for name in CLC_VARS]
            return mapping, np.asarray(static, dtype=np.float32), np.asarray(clc, dtype=np.float32), units
        finally:
            ds.close()

    def build_live_features(self, latitude: float, longitude: float, target_date: str | date | datetime) -> dict[str, Any]:
        target = target_date.date() if isinstance(target_date, datetime) else (target_date if isinstance(target_date, date) else date.fromisoformat(str(target_date)))
        dates = [target - timedelta(days=i) for i in range(10, 0, -1)]
        mapping, static, clc, units = self._static_features(latitude, longitude, target.year)
        weather = self.weather_provider.fetch_window(latitude, longitude, dates)
        eo = self.eo_provider.fetch_window(latitude, longitude, dates)
        by_date = {r.date: r for r in weather + eo}
        dyn = np.full((1, 10, len(DYNAMIC)), np.nan, dtype=np.float32)
        statuses: list[ProviderStatus] = []
        for i, d in enumerate(dates):
            row = by_date.get(d, DailyValues(d, {}, ProviderStatus("combined", False, "no provider row")))
            statuses.append(row.status)
            for k, name in enumerate(DYNAMIC):
                dyn[0, i, k] = _safe_float(row.values.get(name))
            # Merge weather and EO rows when providers return separate records.
            weather_row = next((r for r in weather if r.date == d), None)
            eo_row = next((r for r in eo if r.date == d), None)
            for source in (weather_row, eo_row):
                if source:
                    for k, name in enumerate(DYNAMIC):
                        if name in source.values:
                            dyn[0, i, k] = _safe_float(source.values[name])
        filled_dyn = fill_time_nans(dyn)
        dyn_n = self.base.scaler_dyn.transform(filled_dyn.reshape(-1, len(DYNAMIC))).reshape(filled_dyn.shape).astype(np.float32)
        sta_n = self.base.scaler_sta.transform(np.nan_to_num(static, nan=-1.0).reshape(1, 5)).astype(np.float32)
        x = xgb_features(dyn_n, sta_n, np.nan_to_num(clc, nan=0.0).reshape(1, 10)).astype(np.float32)
        if x.shape != (1, FEATURE_COUNT):
            raise FireCubeInferenceError(f"live feature shape {x.shape}, expected (1, {FEATURE_COUNT})")
        missing = [name for k, name in enumerate(DYNAMIC) if not np.isfinite(dyn[0, :, k]).all()]
        approximate = ["sminx", "max_tp", "max_sp"] if isinstance(self.weather_provider, OpenMeteoWeatherProvider) else []
        if isinstance(self.eo_provider, ORNLModisLSTProvider):
            approximate.extend(["lst_day", "lst_night"])
        return {
            "latitude": mapping.latitude, "longitude": mapping.longitude, "date": target.isoformat(),
            "unit_id": mapping.unit_id, "grid_y": mapping.grid_y, "grid_x": mapping.grid_x,
            "unit_area_km2": mapping.unit_area_km2, "feature_count": FEATURE_COUNT,
            "feature_names": self.base.feature_names(), "feature_vector": x[0].tolist(),
            "feature_values": dict(zip(self.base.feature_names(), x[0].astype(float).tolist())),
            "window_dates": [d.isoformat() for d in dates], "missing_dynamic_variables": sorted(set(missing)),
            "provider_status": {
                "weather": [s.__dict__ for s in weather],
                "eo": [s.__dict__ for s in eo],
            },
            "weather_provider": type(self.weather_provider).__name__, "eo_provider": type(self.eo_provider).__name__,
            "approximated_dynamic_variables": approximate,
            "historical_static_fallback": True,
        }

    def predict_live(self, latitude: float, longitude: float, target_date: str | date | datetime) -> dict[str, Any]:
        built = self.build_live_features(latitude, longitude, target_date)
        x = np.asarray(built["feature_vector"], dtype=np.float32).reshape(1, FEATURE_COUNT)
        built["raw_probability"] = float(self.base.model.predict_proba(x)[0, 1])
        built["model"] = "Exp5 direct 10 km XGBoost"
        built["prediction_status"] = (
            "weather + static inputs; EO unavailable"
            if built["missing_dynamic_variables"]
            else "all dynamic inputs supplied; some variables are provider approximations"
            if built["approximated_dynamic_variables"]
            else "all dynamic inputs supplied"
        )
        return built
