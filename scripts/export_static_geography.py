"""Derive compact v1 geography from the authorized local cube; never train.

Run from the project root: python -m scripts.export_static_geography
Existing outputs are protected unless --overwrite is explicitly supplied.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np

from src.benchmark_direct_10km import STATIC, STATIC_AGG, agg_land_to_units, build_block_units, filled
from src.firecube_inference import DEFAULT_CUBE, CLC_VARS, STRIDE_X, STRIDE_Y
from src.static_geography import DEFAULT_STATIC_ARTIFACT, SCHEMA, STATIC_NAMES, CLC_NAMES, POPULATION_YEARS, sha256_file


def export(cube_path: Path, output: Path, overwrite: bool = False) -> dict:
    manifest_path = output.with_suffix(".json")
    if not overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError("Output exists; use --overwrite only for a deliberate regeneration")
    print("Hashing source cube (read only)...", flush=True)
    source_hash = sha256_file(cube_path)
    with nc.Dataset(cube_path, "r") as ds:
        land = filled(ds.variables["CLC_2018_9"]) < 0.5
        ly, lx = np.where(land)
        units = build_block_units(ly, lx, STRIDE_Y, STRIDE_X, "10km")
        uid, n = units["unit_id"], units["n_units"]
        mapping = np.full(land.shape, -1, dtype=np.int32)
        mapping[land] = uid

        def aggregate(name, how):
            return agg_land_to_units(filled(ds.variables[name])[land], uid, n, how)

        arrays = {
            "x": np.asarray(ds.variables["x"][:], dtype=np.float64),
            "y": np.asarray(ds.variables["y"][:], dtype=np.float64),
            "land_mask": land, "grid_unit_id": mapping,
            "static": np.stack([aggregate(name, STATIC_AGG[name]) for name in STATIC], axis=1),
            "population": np.stack([aggregate(f"POP_DENS_{year}", "mean") for year in POPULATION_YEARS]),
            "clc": np.stack([aggregate(name, "mean") for name in CLC_VARS], axis=1),
            "unit_area_km2": units["area_km2"],
        }
        variables = STATIC + [f"POP_DENS_{year}" for year in POPULATION_YEARS] + CLC_VARS
        # NetCDF attribute values are retained as text; NaN is documented separately.
        attributes = {name: {key: str(value) for key, value in ds.variables[name].__dict__.items()} for name in variables}
        source_attributes = {key: str(value) for key, value in ds.__dict__.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    manifest = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_filename": output.name, "artifact_bytes": output.stat().st_size,
        "artifact_sha256": sha256_file(output),
        "source": {"filename": cube_path.name, "bytes": cube_path.stat().st_size,
                   "sha256": source_hash, "netcdf_attributes": source_attributes,
                   "local_documentation_source_doi": "10.5281/zenodo.6475592",
                   "citation_note": "Embedded citation references older record 4943354; preserved verbatim. Source hash identifies the actual local cube; no claim of publisher checksum verification."},
        "generator": "python -m scripts.export_static_geography",
        "generator_sha256": sha256_file(Path(__file__)),
        "aggregation_source_sha256": sha256_file(Path(__file__).resolve().parents[1] / "src/benchmark_direct_10km.py"),
        "numpy_version": np.__version__, "netcdf4_version": nc.__version__,
        "crs": "EPSG:4326", "grid_shape": list(land.shape), "n_units": n,
        "land_cell_count": int(land.sum()), "non_land_cell_count": int((~land).sum()),
        "coordinate_bounds": {"longitude": [float(arrays["x"].min()), float(arrays["x"].max())],
                              "latitude": [float(arrays["y"].min()), float(arrays["y"].max())]},
        "stride_cells": {"y": STRIDE_Y, "x": STRIDE_X},
        "mapping": "Inclusive axis-centre bounds; nearest axis argmin (first index wins ties); land iff filled CLC_2018_9 < 0.5; non-land/unknown excluded. Sorted unique block keys and IDs exactly follow build_block_units.",
        "static_feature_names": STATIC_NAMES, "clc_feature_names": CLC_NAMES,
        "population_years": POPULATION_YEARS, "population_year_policy": "clamp requested year to [2009,2021] (unchanged)",
        "aggregation": {**STATIC_AGG, "CLC_2018_0..9": "mean"},
        "missing_values": "Source masked values become float32 NaN; aggregation ignores nonfinite values, all-missing units stay NaN. Stored unchanged; existing trained median imputer handles NaNs. grid_unit_id=-1 denotes excluded cells, never a feature value.",
        "variable_attributes": attributes,
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                           "nan_count": int(np.isnan(value).sum()) if value.dtype.kind == "f" else 0}
                   for name, value in arrays.items()},
        "operational_validity": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported {output}: {manifest['artifact_bytes']} bytes, {n} units", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, default=DEFAULT_CUBE)
    parser.add_argument("--output", type=Path, default=DEFAULT_STATIC_ARTIFACT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export(args.cube, args.output, args.overwrite)
