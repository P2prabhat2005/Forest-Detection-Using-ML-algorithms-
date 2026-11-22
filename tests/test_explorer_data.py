from base64 import b64decode
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.explorer_data import coverage_overlay, model_unit_overlay, parse_firms_csv, parse_shared_location, share_location_url
from src.static_geography import CompactStaticFeatureProvider


def test_coverage_overlay_preserves_geographic_bounds_and_land_at_athens():
    provider = CompactStaticFeatureProvider()
    overlay = coverage_overlay(provider)
    (south, west), (north, east) = overlay["bounds"]
    assert [south, north] == provider.manifest["coordinate_bounds"]["latitude"]
    assert [west, east] == provider.manifest["coordinate_bounds"]["longitude"]
    image = Image.open(BytesIO(b64decode(overlay["image"].split(",", 1)[1])))
    assert image.size == tuple(reversed(provider.manifest["grid_shape"]))

    def alpha_at(lat, lon):
        top, bottom, center = np.arcsinh(np.tan(np.radians([north, south, lat])))
        col = round((lon - west) / (east - west) * (image.width - 1))
        row = round((top - center) / (top - bottom) * (image.height - 1))
        return image.getpixel((col, row))[3]

    assert alpha_at(37.9755648, 23.7348324) > 0
    assert alpha_at(36.0, 25.0) == 0


def test_model_unit_overlay_uses_selected_land_cells_and_unit_area():
    provider = CompactStaticFeatureProvider()
    latitude, longitude = 36.42, 25.43
    _, _, location = provider.fetch_static(latitude, longitude, 2026)
    layer = model_unit_overlay(provider, location)
    assert layer["id"] == location["unit_id"]
    assert layer["landAreaKm2"] == provider.arrays["unit_area_km2"][layer["id"]]
    (south, west), (north, east) = layer["bounds"]
    assert south <= latitude <= north and west <= longitude <= east
    image = Image.open(BytesIO(b64decode(layer["image"].split(",", 1)[1])))
    assert image.size == (14, 11)
    assert set(np.unique(np.asarray(image.getchannel("A")))) == {0, 180}


@pytest.mark.parametrize("malformed", ["", "38", "38,24,0", "nan,23", "38,inf", "91,24", "38,-181"])
def test_invalid_shared_coordinates(malformed):
    with pytest.raises(ValueError):
        parse_shared_location(malformed)


def test_cloud_share_link_uses_public_url():
    assert share_location_url("https://forestsentinels.streamlit.app/~/+/", 38.0, 23.5) == (
        "https://forestsentinels.vercel.app/?at=38.0%2C23.5"
    )
    assert share_location_url("http://127.0.0.1:8521", 38.0, 23.5) == "http://127.0.0.1:8521?at=38.0%2C23.5"


def test_firms_filters_and_deduplicates_using_acquisition_time():
    csv = """latitude,longitude,acq_date,acq_time,confidence
37.9,23.7,2026-11-21,2000,h
37.9,23.7,2026-11-22,15,n
37.9,23.7,2026-11-22,15,n
29.9,23.7,2026-11-22,15,n
37.9,23.7,2026-11-22,1800,l
"""
    start = datetime(2026, 11, 21, 21, tzinfo=timezone.utc)
    end = datetime(2026, 11, 22, 12, tzinfo=timezone.utc)
    assert parse_firms_csv(csv, [[34, 18], [43, 29]], start, end) == [
        {"latitude": 37.9, "longitude": 23.7, "observed": "2026-11-22 00:15 UTC", "confidence": "Nominal"},
    ]
    with pytest.raises(ValueError, match="expected CSV header"):
        parse_firms_csv("Invalid MAP_KEY.", [[34, 18], [43, 29]], start, end)
