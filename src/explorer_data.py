"""Display data for the map explorer. Satellite observations are independent of Track-B."""

from __future__ import annotations

import csv
import math
from base64 import b64encode
from datetime import datetime, timedelta, timezone
from io import BytesIO, StringIO
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .static_geography import CompactStaticFeatureProvider

FIRMS_PRODUCT = "VIIRS_NOAA20_NRT"
FIRMS_WINDOWS = ("Today UTC", "Last 24 hours", "Last 72 hours")
PUBLIC_EXPLORER_URL = "https://forestsentinels.vercel.app/"


def coverage_overlay(provider: CompactStaticFeatureProvider) -> dict:
    """Render the land mask in the Web Mercator projection used by Leaflet tiles."""
    arrays = provider.arrays
    return _mask_overlay(arrays["x"], arrays["y"], arrays["land_mask"], (43, 155, 118, 104))


def _mask_overlay(x: np.ndarray, y: np.ndarray, land: np.ndarray, color: tuple) -> dict:
    if x[0] > x[-1]:
        x, land = x[::-1], land[:, ::-1]
    if y[0] < y[-1]:
        y, land = y[::-1], land[::-1, :]

    mercator = np.arcsinh(np.tan(np.radians(y)))
    uniform_y = np.linspace(mercator[0], mercator[-1], y.size)
    row_indices = np.rint(np.interp(uniform_y, mercator[::-1], np.arange(y.size)[::-1])).astype(int)
    rgba = np.zeros((*land.shape, 4), dtype=np.uint8)
    rgba[land[row_indices]] = color
    buffer = BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG")
    return {
        "bounds": [[float(y[-1]), float(x[0])], [float(y[0]), float(x[-1])]],
        "image": "data:image/png;base64," + b64encode(buffer.getvalue()).decode("ascii"),
    }


def model_unit_overlay(provider: CompactStaticFeatureProvider, location: dict) -> dict:
    """Highlight the eligible land cells in the selected Track-B grid block."""
    stride = provider.manifest["stride_cells"]
    unit_id = location["unit_id"]
    start_y = location["grid_y"] // stride["y"] * stride["y"]
    start_x = location["grid_x"] // stride["x"] * stride["x"]
    rows = slice(start_y, start_y + stride["y"])
    columns = slice(start_x, start_x + stride["x"])
    arrays = provider.arrays
    x, y = arrays["x"][columns], arrays["y"][rows]
    land = arrays["grid_unit_id"][rows, columns] == unit_id
    layer = _mask_overlay(x, y, land, (96, 81, 187, 180))
    # Use nearest-centre cell edges for the enclosing block, clipped to model bounds.
    dx = abs(float(arrays["x"][1] - arrays["x"][0])) / 2
    dy = abs(float(arrays["y"][1] - arrays["y"][0])) / 2
    layer["bounds"] = [
        [max(float(arrays["y"].min()), float(y.min()) - dy), max(float(arrays["x"].min()), float(x.min()) - dx)],
        [min(float(arrays["y"].max()), float(y.max()) + dy), min(float(arrays["x"].max()), float(x.max()) + dx)],
    ]
    layer["id"] = unit_id
    layer["landAreaKm2"] = float(arrays["unit_area_km2"][unit_id])
    return layer


def parse_shared_location(value: str) -> tuple[float, float]:
    latitude, longitude = (float(part) for part in value.split(","))
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Coordinates are outside valid latitude and longitude limits")
    return latitude, longitude


def share_location_url(app_url: str | None, latitude: float, longitude: float) -> str:
    # Community Cloud reports its internal iframe path. Share the public alias.
    if not app_url or urlsplit(app_url).hostname == "forestsentinels.streamlit.app":
        app_url = PUBLIC_EXPLORER_URL
    return f"{app_url}?{urlencode({'at': f'{latitude},{longitude}'})}"


def parse_firms_csv(payload: str, bounds: list, start: datetime, end: datetime) -> list[dict]:
    """Read in-window NOAA-20 active-fire pixels; preserve source observation times."""
    rows = csv.DictReader(StringIO(payload))
    if not {"latitude", "longitude", "acq_date", "acq_time", "confidence"} <= set(rows.fieldnames or []):
        raise ValueError("FIRMS response did not contain the expected CSV header")
    (south, west), (north, east) = bounds
    detections = {}
    for row in rows:
        lat, lon = float(row["latitude"]), float(row["longitude"])
        date_text, time_text = row["acq_date"], row["acq_time"]
        if not time_text or not time_text.isascii() or not time_text.isdigit() or len(time_text) > 4:
            raise ValueError("Invalid FIRMS acquisition time")
        prefix = time_text.zfill(4)
        stamp = datetime.fromisoformat(f"{date_text}T{prefix[:2]}:{prefix[2:]}+00:00")
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Invalid FIRMS coordinates")
        if start <= stamp <= end and south <= lat <= north and west <= lon <= east:
            confidence = {"l": "Low", "n": "Nominal", "h": "High"}.get(row["confidence"], "Unspecified")
            detections[(lat, lon, stamp)] = {
                "latitude": lat, "longitude": lon, "observed": stamp.strftime("%Y-%m-%d %H:%M UTC"),
                "confidence": confidence,
            }
    return list(detections.values())


def fetch_firms_hotspots(key: str, bounds: list, window: str) -> list[dict]:
    now = datetime.now(timezone.utc)
    days = {"Today UTC": 0, "Last 24 hours": 1, "Last 72 hours": 3}[window]
    start = now - timedelta(days=days) if days else now.replace(hour=0, minute=0, second=0, microsecond=0)
    (south, west), (north, east) = bounds
    box = f"{west:.5f},{south:.5f},{east:.5f},{north:.5f}"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{quote(key, safe='')}/{FIRMS_PRODUCT}/{box}/{days + 1}"
    request = Request(url, headers={"User-Agent": "ForestSentinels/1.0"})
    with urlopen(request, timeout=12) as response:
        payload = response.read(2_000_001)
    if len(payload) > 2_000_000:
        raise ValueError("FIRMS response exceeded the explorer's size limit")
    return parse_firms_csv(payload.decode("utf-8-sig"), bounds, start, now)
