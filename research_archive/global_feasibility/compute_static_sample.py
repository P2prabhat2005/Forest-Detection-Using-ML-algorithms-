"""Reproduce two small EPSG:6933 static-feature samples from public COG windows.

Dependencies: numpy, rasterio, pyproj. Reads only windows intersecting two 25 km
cells via GDAL /vsicurl/ HTTP range requests; does not download full tiles.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from pyproj import Geod, Transformer
from rasterio.windows import from_bounds, transform as window_transform

HERE = Path(__file__).resolve().parent
OUT = HERE / "static_sample_manifest.json"
STEP_M = 25_000
TO_GRID = Transformer.from_crs(4326, 6933, always_xy=True)
FROM_GRID = Transformer.from_crs(6933, 4326, always_xy=True)
GEOD = Geod(ellps="WGS84")

CLASSES = {
    10: "tree_cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built_up", 60: "bare_sparse_vegetation", 70: "snow_ice",
    80: "permanent_water", 90: "herbaceous_wetland", 95: "mangroves",
    100: "moss_lichen",
}

SITES = [
    {
        "region": "Greece", "anchor_lonlat": [22.5, 39.5],
        "dem_tile": "Copernicus_DSM_COG_10_N39_00_E022_00_DEM",
        "worldcover_tile": "N39E021",
    },
    {
        "region": "British Columbia", "anchor_lonlat": [-120.5, 50.5],
        "dem_tile": "Copernicus_DSM_COG_10_N50_00_W121_00_DEM",
        "worldcover_tile": "N48W123",
    },
]


def grid_cell(lon, lat):
    x, y = TO_GRID.transform(lon, lat)
    ix, iy = math.floor(x / STEP_M), math.floor(y / STEP_M)
    bounds = [ix * STEP_M, iy * STEP_M, (ix + 1) * STEP_M, (iy + 1) * STEP_M]
    corners = [(bounds[0], bounds[1]), (bounds[2], bounds[1]),
               (bounds[2], bounds[3]), (bounds[0], bounds[3]),
               (bounds[0], bounds[1])]
    coords = [list(FROM_GRID.transform(px, py)) for px, py in corners]
    return {
        "id": f"EPSG6933_25km_{ix}_{iy}", "indices": [ix, iy],
        "crs": "EPSG:6933", "bounds_m": bounds,
        "geometry_wgs84": {"type": "Polygon", "coordinates": [coords]},
        "area_km2": 625,
    }


def summarize(url, bounds, categorical=False):
    with urlopen(Request(url, method="HEAD"), timeout=20) as response:
        object_metadata = {
            "content_length_bytes": int(response.headers.get("Content-Length", "0")),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "accept_ranges": response.headers.get("Accept-Ranges"),
        }
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open("/vsicurl/" + url) as src:
            lonlat = [FROM_GRID.transform(bounds[0], bounds[1]),
                      FROM_GRID.transform(bounds[2], bounds[3])]
            west, east = sorted([p[0] for p in lonlat])
            south, north = sorted([p[1] for p in lonlat])
            window = from_bounds(west, south, east, north, src.transform).round_offsets().round_lengths()
            data = src.read(1, window=window, masked=True)
            affine = window_transform(window, src.transform)
            cols = np.arange(data.shape[1], dtype=np.float64) + 0.5
            rows = np.arange(data.shape[0], dtype=np.float64) + 0.5
            lon = affine.c + affine.a * cols
            lat = affine.f + affine.e * rows
            # EPSG:6933 is cylindrical equal-area: projected X depends on lon,
            # projected Y on lat. Pixel-centre inclusion approximates edge area.
            xx, _ = TO_GRID.transform(lon, np.zeros_like(lon))
            _, yy = TO_GRID.transform(np.zeros_like(lat), lat)
            col_inside = (xx >= bounds[0]) & (xx < bounds[2])
            row_inside = (yy >= bounds[1]) & (yy < bounds[3])
            weighted_valid = weighted_total = 0.0
            weighted_sum = weighted_sum_sq = 0.0
            minimum, maximum = float("inf"), -float("inf")
            classes = {code: 0.0 for code in CLASSES}
            unknown_class_area = 0.0
            valid_pixels = candidate_pixels = 0
            left_edge = affine.c + affine.a * np.flatnonzero(col_inside)[0]
            right_edge = affine.c + affine.a * (np.flatnonzero(col_inside)[-1] + 1)
            for ri in np.flatnonzero(row_inside):
                top = affine.f + affine.e * ri
                bottom = top + affine.e
                area, _ = GEOD.polygon_area_perimeter(
                    [left_edge, right_edge, right_edge, left_edge],
                    [top, top, bottom, bottom],
                )
                pixel_area = abs(area) / int(col_inside.sum())
                row = data[ri, col_inside]
                mask = np.ma.getmaskarray(row)
                valid = ~mask
                count = int(valid.sum())
                candidate_pixels += len(row)
                valid_pixels += count
                weighted_total += pixel_area * len(row)
                weighted_valid += pixel_area * count
                if categorical:
                    values = np.asarray(row)[valid]
                    for code in CLASSES:
                        classes[code] += pixel_area * int(np.count_nonzero(values == code))
                    unknown_class_area += pixel_area * int(np.count_nonzero(~np.isin(values, list(CLASSES))))
                elif count:
                    values = np.asarray(row)[valid].astype(np.float64)
                    weighted_sum += pixel_area * float(values.sum())
                    weighted_sum_sq += pixel_area * float(np.square(values).sum())
                    minimum = min(minimum, float(values.min()))
                    maximum = max(maximum, float(values.max()))
            coverage = weighted_valid / weighted_total if weighted_total else None
            common = {
                "url": url, "object_metadata": object_metadata,
                "source_crs": str(src.crs), "native_pixel_size_degrees":
                [abs(src.transform.a), abs(src.transform.e)],
                "source_nodata": src.nodata, "window_shape": list(data.shape),
                "candidate_pixel_centres": candidate_pixels, "valid_pixel_centres": valid_pixels,
                "valid_area_fraction": coverage,
                "nodata_area_fraction": 1 - coverage if coverage is not None else None,
                "sampled_pixel_area_m2": weighted_total,
            }
            if categorical:
                common["class_fractions_of_valid_area"] = {
                    CLASSES[k]: classes[k] / weighted_valid if weighted_valid else None
                    for k in CLASSES
                }
                common["unmapped_valid_class_fraction"] = unknown_class_area / weighted_valid if weighted_valid else None
            else:
                mean = weighted_sum / weighted_valid if weighted_valid else None
                common["elevation_m"] = {
                    "area_weighted_mean": mean,
                    "area_weighted_std": math.sqrt(max(0, weighted_sum_sq / weighted_valid - mean**2)) if weighted_valid else None,
                    "min": minimum if weighted_valid else None,
                    "max": maximum if weighted_valid else None,
                }
            return common


def main():
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Small common-grid static-data feasibility sample; no labels or model",
        "overall_global_feasibility_gate": "FAIL_PENDING_LABEL_CALENDAR_CLARIFICATION",
        "grid": {"crs": "EPSG:6933", "origin_m": [0, 0], "step_m": STEP_M,
                 "cell_rule": "floor(projected_x/25000), floor(projected_y/25000)"},
        "processing": {
            "spatial_selection": "Native WGS84 source pixel centres inside projected cell; no resampling; boundary approximation at native pixel size",
            "weighting": "WGS84 ellipsoidal pixel area, computed by latitude row",
            "elevation": "weighted mean/std; valid-pixel min/max",
            "land_cover": "weighted class fractions over valid source area; NoData separate",
            "software": {"rasterio": rasterio.__version__, "pyproj": __import__("pyproj").__version__, "numpy": np.__version__},
        },
        "sources": {
            "elevation": {"name": "Copernicus DEM GLO-30 Public", "version": "COP-DEM_GLO-30-DGED AWS COG public instance", "native_resolution": "1 arcsecond nominal (~30 m)", "semantics": "digital surface model, not bare-earth terrain"},
            "land_cover": {"name": "ESA WorldCover", "version": "2021 v200", "native_resolution": "10 m nominal (~0.0000833 degree)", "class_codes": CLASSES},
        },
        "samples": [],
    }
    for site in SITES:
        cell = grid_cell(*site["anchor_lonlat"])
        bounds = cell["bounds_m"]
        dt = site["dem_tile"]
        wc = site["worldcover_tile"]
        dem_url = f"https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{dt}/{dt}.tif"
        wc_url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{wc}_Map.tif"
        output["samples"].append({
            "region": site["region"], "anchor_lonlat": site["anchor_lonlat"],
            "grid_cell": cell,
            "elevation": summarize(dem_url, bounds),
            "land_cover": summarize(wc_url, bounds, categorical=True),
        })
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {OUT}")
    for s in output["samples"]:
        print(s["region"], s["grid_cell"]["id"],
              "elev_mean_m=", round(s["elevation"]["elevation_m"]["area_weighted_mean"], 1),
              "landcover_coverage=", round(s["land_cover"]["valid_area_fraction"], 4))


if __name__ == "__main__":
    main()
