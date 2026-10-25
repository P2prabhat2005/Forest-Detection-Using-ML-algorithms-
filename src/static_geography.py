"""Versioned, checksum-verified Greece geography for Track-B live inference.

No NetCDF file or research model is opened here. Historical replay continues
to use FireCubeStaticFeatureProvider in production_features.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .firecube_inference import FireCubeInferenceError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIC_ARTIFACT = ROOT / "models/production/greece_static_v1.npz"
SCHEMA = "greece_track_b_static_v1"
STATIC_NAMES = ["DEM__mean", "SLOPE__mean", "ROAD_DISTANCE__min", "WATERWAY_DISTANCE__min", "POP_DENS__mean"]
CLC_NAMES = [f"CLC_2018_{i}__mean" for i in range(10)]
POPULATION_YEARS = list(range(2009, 2022))


class StaticArtifactError(FireCubeInferenceError):
    """Missing, corrupt or incompatible compact geography; never use a fallback."""


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class CompactStaticFeatureProvider:
    def __init__(self, artifact_path: Path = DEFAULT_STATIC_ARTIFACT) -> None:
        self.path = Path(artifact_path)
        manifest_path = self.path.with_suffix(".json")
        for path in (self.path, manifest_path):
            if not path.is_file():
                raise StaticArtifactError(f"Missing Greece static artifact: {path}. Restore the versioned .npz and .json pair from the project checkout.")
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            m = self.manifest
            if (m["schema"] != SCHEMA or m["static_feature_names"] != STATIC_NAMES
                    or m["clc_feature_names"] != CLC_NAMES or m["population_years"] != POPULATION_YEARS
                    or m["stride_cells"] != {"y": 11, "x": 14}):
                raise ValueError("schema, feature order, population years or grid stride differs from v1")
            self.checksum = sha256_file(self.path)
            if m["artifact_sha256"] != self.checksum or m["artifact_bytes"] != self.path.stat().st_size:
                raise ValueError("SHA-256 checksum mismatch")
            with np.load(self.path, allow_pickle=False) as archive:
                self.arrays = {key: archive[key] for key in archive.files}
            self._validate()
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, EOFError, BadZipFile) as exc:
            raise StaticArtifactError(f"Invalid Greece static artifact {self.path}: {exc}. Restore a matching v1 artifact/manifest pair.") from exc

    def _validate(self) -> None:
        a, m = self.arrays, self.manifest
        n = m["n_units"]
        ny, nx = m["grid_shape"]
        if not isinstance(n, int) or n <= 0 or ny < 2 or nx < 2:
            raise ValueError("invalid unit count or grid shape")
        expected = {
            "x": ((nx,), "float64"), "y": ((ny,), "float64"),
            "land_mask": ((ny, nx), "bool"), "grid_unit_id": ((ny, nx), "int32"),
            "static": ((n, 4), "float32"), "population": ((13, n), "float32"),
            "clc": ((n, 10), "float32"), "unit_area_km2": ((n,), "float64"),
        }
        if set(a) != set(expected):
            raise ValueError("unexpected or missing arrays")
        for name, (shape, dtype) in expected.items():
            if a[name].shape != shape or a[name].dtype != np.dtype(dtype):
                raise ValueError(f"invalid shape/dtype for {name}")
        for name in ("x", "y"):
            axis = a[name]
            diff = np.diff(axis)
            if not np.isfinite(axis).all() or not ((diff > 0).all() or (diff < 0).all()):
                raise ValueError(f"{name} axis must be finite and strictly monotonic")
        land, ids = a["land_mask"], a["grid_unit_id"]
        if not np.all(ids[~land] == -1) or not np.array_equal(np.unique(ids[land]), np.arange(n)):
            raise ValueError("land mask and unit IDs disagree")
        for name in ("static", "population", "clc"):
            if np.isinf(a[name]).any():
                raise ValueError(f"{name} contains infinity; v1 permits finite values or NaN only")
        if not np.isfinite(a["unit_area_km2"]).all() or not (a["unit_area_km2"] > 0).all():
            raise ValueError("invalid unit areas")

    def fetch_static(self, latitude: float, longitude: float, year: int) -> tuple[np.ndarray, np.ndarray, dict]:
        lat, lon = float(latitude), float(longitude)
        x, y = self.arrays["x"], self.arrays["y"]
        if not (np.isfinite(lat) and np.isfinite(lon) and x.min() <= lon <= x.max() and y.min() <= lat <= y.max()):
            raise FireCubeInferenceError("latitude/longitude is outside the FireCube Greece domain (finite coordinates required)")
        # Same inclusive bounds, nearest-centre selection and argmin tie rule as the cube.
        grid_x, grid_y = int(np.abs(x - lon).argmin()), int(np.abs(y - lat).argmin())
        if not self.arrays["land_mask"][grid_y, grid_x]:
            raise FireCubeInferenceError("location maps to a non-land FireCube cell")
        unit_id = int(self.arrays["grid_unit_id"][grid_y, grid_x])
        pop_year = min(max(year, 2009), 2021)
        static = np.concatenate((self.arrays["static"][unit_id], [self.arrays["population"][pop_year - 2009, unit_id]])).astype(np.float32)
        return static, self.arrays["clc"][unit_id].copy(), {
            "provider": "FireCube static geography (compact v1)",
            "data_date": "historical static raster (date unspecified)",
            "population_year": pop_year, "land_cover_year": 2018,
            "unit_id": unit_id, "grid_y": grid_y, "grid_x": grid_x,
            "historical_static_fallback": year > 2021,
            "static_artifact_schema": SCHEMA, "static_artifact_sha256": self.checksum,
        }
