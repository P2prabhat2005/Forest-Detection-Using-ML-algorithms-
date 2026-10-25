"""Read-only cube/compact parity checks; writes a separate production audit.

No network calls, model fitting, research-output changes or source-cube writes.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np

from src.benchmark_direct_10km import STATIC, STATIC_AGG, DYNAMIC, filled, agg_land_to_units, build_block_units
from src.firecube_inference import DEFAULT_CUBE, FireCubeInferenceError
from src.firecube_live import DailyValues, ProviderStatus
from src.production_features import FireCubeStaticFeatureProvider, ProductionFeaturePipeline, ProductionPreprocessor, production_feature_names
from src.production_model import XGBoostProductionModel
from src.static_geography import CompactStaticFeatureProvider, POPULATION_YEARS, CLC_NAMES, ROOT, sha256_file


class FrozenDynamicWindow:
    """Identical deterministic inputs isolate static-source effects on scores."""
    def fetch_window(self, latitude, longitude, dates):
        base = [300., 285., 0.35, 0.001, 100000., 5., 0.4, 310., 290., 0.3]
        return [DailyValues(day, {name: base[k] * (1 + i / 1000) for k, name in enumerate(DYNAMIC)},
                            ProviderStatus("parity-fixture", True, "frozen deterministic test inputs"))
                for i, day in enumerate(dates)]


def verify(output: Path) -> dict:
    compact = CompactStaticFeatureProvider()
    original = FireCubeStaticFeatureProvider()
    a = compact.arrays
    prep = ProductionPreprocessor.load()
    model_path = ROOT / 'models/production/xgb_baseline.joblib'
    model = XGBoostProductionModel.load(model_path)
    assert production_feature_names()[20:] == compact.manifest['static_feature_names'] + CLC_NAMES
    report = {
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'artifact_sha256': compact.checksum,
        'source_cube_sha256_at_export': compact.manifest['source']['sha256'],
        'model_sha256': sha256_file(model_path),
        'preprocessor_sha256': sha256_file(ROOT / 'models/production/feature_preprocessor.joblib'),
        'dynamic_inputs': 'Identical deterministic ten-day fixtures; isolates static-source parity, not live API validation.',
        'operational_validity': False, 'land_checks': [], 'water_checks': [],
    }
    with nc.Dataset(DEFAULT_CUBE, 'r') as ds:
        x, y = np.asarray(ds.variables['x'][:], dtype=np.float64), np.asarray(ds.variables['y'][:], dtype=np.float64)
        land = filled(ds.variables['CLC_2018_9']) < 0.5
        ly, lx = np.where(land)
        units = build_block_units(ly, lx, 11, 14, '10km')
        mapping = np.full(land.shape, -1, dtype=np.int32)
        mapping[land] = units['unit_id']
        for actual, expected in ((a['x'], x), (a['y'], y), (a['land_mask'], land), (a['grid_unit_id'], mapping),
                                 (a['unit_area_km2'], units['area_km2'])):
            np.testing.assert_array_equal(actual, expected)
        def aggregate(name, how):
            return agg_land_to_units(filled(ds.variables[name])[land], units['unit_id'], units['n_units'], how)
        for k, name in enumerate(STATIC):
            np.testing.assert_array_equal(a['static'][:, k], aggregate(name, STATIC_AGG[name]))
        for k, year in enumerate(POPULATION_YEARS):
            np.testing.assert_array_equal(a['population'][k], aggregate(f'POP_DENS_{year}', 'mean'))
        for k in range(10):
            np.testing.assert_array_equal(a['clc'][:, k], aggregate(f'CLC_2018_{k}', 'mean'))
    report['exhaustive_static_arrays'] = {'units': units['n_units'], 'population_years': 13,
                                          'grid_cells': int(land.size), 'exact_equal_including_nan': True}
    # Spread across sorted block units and include the existing Athens demo.
    coordinates = [(38., 23.7)]
    for uid in np.linspace(0, units['n_units'] - 1, 8, dtype=int):
        cells = np.argwhere(mapping == uid)
        gy, gx = cells[len(cells) // 2]
        coordinates.append((float(y[gy]), float(x[gx])))
    dynamic = FrozenDynamicWindow()
    compact_pipeline = ProductionFeaturePipeline(dynamic, compact, prep)
    original_pipeline = ProductionFeaturePipeline(dynamic, original, prep)
    for lat, lon in coordinates:
        for year in (2008, 2009, 2015, 2020, 2021, 2026):
            cs, cc, cm = compact.fetch_static(lat, lon, year)
            os, oc, om = original.fetch_static(lat, lon, year)
            np.testing.assert_array_equal(cs, os)
            np.testing.assert_array_equal(cc, oc)
            for name in ('population_year', 'unit_id', 'grid_y', 'grid_x', 'historical_static_fallback'):
                assert cm[name] == om[name], name
            target = f'{year}-07-15'
            c = compact_pipeline.build(lat, lon, target)
            o = original_pipeline.build(lat, lon, target)
            assert c.feature_names == o.feature_names == production_feature_names() == model.feature_names
            np.testing.assert_array_equal(c.raw_features, o.raw_features)
            np.testing.assert_array_equal(c.model_features, o.model_features)
            cp, op = model.predict_probability(c.model_features), model.predict_probability(o.model_features)
            assert cp == op
            report['land_checks'].append({'latitude': lat, 'longitude': lon, 'year': year, 'unit_id': cm['unit_id'],
                                          'static_max_absolute_difference': 0., 'normalized_max_absolute_difference': 0.,
                                          'score_absolute_difference': abs(cp - op), 'score': cp})
    water = np.argwhere(~land)
    for gy, gx in water[np.linspace(0, len(water) - 1, 6, dtype=int)]:
        lat, lon = float(y[gy]), float(x[gx])
        for provider in (compact, original):
            try:
                provider.fetch_static(lat, lon, 2026)
            except FireCubeInferenceError as exc:
                assert 'non-land' in str(exc)
            else:
                raise AssertionError('Water cell unexpectedly accepted')
        report['water_checks'].append({'latitude': lat, 'longitude': lon, 'both_rejected': True})
    report['passed'] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"PASS: all {units['n_units']} units/all 13 population years; {len(report['land_checks'])} pipeline/prediction pairs; {len(water)} excluded cells ({len(report['water_checks'])} sampled). Report: {output}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/static_geography_parity.json')
    args = parser.parse_args()
    verify(args.output)
