"""Compact production artifact integrity, mapping, parity and model checks."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.firecube_inference import DEFAULT_CUBE, FireCubeInferenceError
from src.firecube_live import HistoricalCubeDynamicProvider
from src.production_features import FireCubeStaticFeatureProvider, ProductionFeaturePipeline, ProductionPreprocessor, production_feature_names
from src.production_model import ProductionPredictionService, XGBoostProductionModel
from src.static_geography import CompactStaticFeatureProvider, DEFAULT_STATIC_ARTIFACT, StaticArtifactError, sha256_file
from scripts.verify_static_geography import FrozenDynamicWindow

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def provider():
    return CompactStaticFeatureProvider()


def variant(tmp_path, mutate_manifest=None, mutate_arrays=None):
    original = CompactStaticFeatureProvider()
    path = tmp_path / 'greece_static_v1.npz'
    arrays = {key: value.copy() for key, value in original.arrays.items()}
    if mutate_arrays:
        mutate_arrays(arrays)
    np.savez_compressed(path, **arrays)
    manifest = dict(original.manifest)
    manifest['artifact_sha256'] = sha256_file(path)
    manifest['artifact_bytes'] = path.stat().st_size
    if mutate_manifest:
        mutate_manifest(manifest)
    path.with_suffix('.json').write_text(json.dumps(manifest), encoding='utf-8')
    return path


def test_load_real_artifact_and_feature_order(provider):
    assert provider.manifest['n_units'] == 4022
    assert production_feature_names()[20:] == provider.manifest['static_feature_names'] + provider.manifest['clc_feature_names']
    static, clc, metadata = provider.fetch_static(38., 23.7, 2026)
    assert static.dtype == clc.dtype == np.float32
    assert static.shape == (5,) and clc.shape == (10,)
    assert metadata['static_artifact_sha256'] == provider.manifest['artifact_sha256']
    assert metadata['population_year'] == 2021


@pytest.mark.parametrize('lat,lon', [(0, 0), (38, 0), (90, 23.7), (np.nan, 23.7), (38, np.inf), (-np.inf, 23.7)])
def test_unsupported_coordinates(provider, lat, lon):
    with pytest.raises(FireCubeInferenceError, match='outside'):
        provider.fetch_static(lat, lon, 2026)


def test_water_and_grid_mapping(provider):
    a = provider.arrays
    gy, gx = np.argwhere(~a['land_mask'])[0]
    with pytest.raises(FireCubeInferenceError, match='non-land'):
        provider.fetch_static(a['y'][gy], a['x'][gx], 2026)
    for uid in (0, 1000, 2000, 4021):
        gy, gx = np.argwhere(a['grid_unit_id'] == uid)[0]
        _, _, m = provider.fetch_static(a['y'][gy], a['x'][gx], 2015)
        assert (m['unit_id'], m['grid_y'], m['grid_x']) == (uid, gy, gx)


def test_inclusive_edges_and_argmin_ties(provider):
    a = provider.arrays
    for lat, lon in [(a['y'][0], a['x'][0]), (a['y'][-1], a['x'][-1]),
                     ((a['y'][400] + a['y'][401]) / 2, (a['x'][600] + a['x'][601]) / 2)]:
        gy = int(np.abs(a['y'] - lat).argmin())
        gx = int(np.abs(a['x'] - lon).argmin())
        if a['land_mask'][gy, gx]:
            _, _, m = provider.fetch_static(lat, lon, 2026)
            assert (m['grid_y'], m['grid_x']) == (gy, gx)
        else:
            with pytest.raises(FireCubeInferenceError, match='non-land'):
                provider.fetch_static(lat, lon, 2026)


def test_population_year_clamping(provider):
    for requested, expected in ((1900, 2009), (2009, 2009), (2015, 2015), (2021, 2021), (2030, 2021)):
        static, _, metadata = provider.fetch_static(38., 23.7, requested)
        assert metadata['population_year'] == expected
        assert static[-1] == provider.arrays['population'][expected - 2009, metadata['unit_id']]


def test_missing_artifact_and_manifest(tmp_path):
    path = tmp_path / 'missing.npz'
    with pytest.raises(StaticArtifactError, match='Missing Greece static artifact'):
        CompactStaticFeatureProvider(path)
    path.write_bytes(DEFAULT_STATIC_ARTIFACT.read_bytes())
    with pytest.raises(StaticArtifactError, match=r'missing.json'):
        CompactStaticFeatureProvider(path)


@pytest.mark.parametrize('field,value', [('schema', 'unknown_v2'), ('static_feature_names', []), ('population_years', [2021]), ('stride_cells', {'y': 1, 'x': 1})])
def test_incompatible_manifest(tmp_path, field, value):
    path = variant(tmp_path, lambda m: m.update({field: value}))
    with pytest.raises(StaticArtifactError, match='schema, feature order'):
        CompactStaticFeatureProvider(path)


def test_checksum_corruption(tmp_path):
    path = variant(tmp_path)
    path.write_bytes(path.read_bytes() + b'corruption')
    with pytest.raises(StaticArtifactError, match='checksum mismatch'):
        CompactStaticFeatureProvider(path)


def test_invalid_array_shape_and_mapping(tmp_path):
    path = variant(tmp_path, mutate_arrays=lambda a: a.update(static=a['static'][:, :3]))
    with pytest.raises(StaticArtifactError, match='shape/dtype'):
        CompactStaticFeatureProvider(path)
    def invalid_map(a):
        a['grid_unit_id'][~a['land_mask']] = 0
    path = variant(tmp_path, mutate_arrays=invalid_map)
    with pytest.raises(StaticArtifactError, match='land mask'):
        CompactStaticFeatureProvider(path)


def test_nan_is_preserved_instead_of_substituted(tmp_path):
    original = CompactStaticFeatureProvider()
    _, _, metadata = original.fetch_static(38, 23.7, 2026)
    uid = metadata['unit_id']
    def missing_value(a):
        a['static'][uid, 0] = np.nan
    provider = CompactStaticFeatureProvider(variant(tmp_path, mutate_arrays=missing_value))
    assert np.isnan(provider.fetch_static(38, 23.7, 2026)[0][0])
    request = ProductionFeaturePipeline(FrozenDynamicWindow(), provider, ProductionPreprocessor.load()).build(38, 23.7, '2026-07-15')
    assert 'DEM__mean' in request.missing_features
    assert np.isfinite(request.model_features).all()  # Existing saved median imputer.


def test_live_pipeline_and_saved_model_never_open_cube():
    with patch('netCDF4.Dataset', side_effect=AssertionError('live inference must not open NetCDF')):
        pipeline = ProductionFeaturePipeline(dynamic_provider=FrozenDynamicWindow(), preprocessor=ProductionPreprocessor.load())
        assert isinstance(pipeline.static_provider, CompactStaticFeatureProvider)
        model = XGBoostProductionModel.load(ROOT / 'models/production/xgb_baseline.joblib')
        service = ProductionPredictionService(pipeline, model)
        class CurrentWeather:
            def fetch_current(self, *args):
                return {'available': True, 'source': 'test fixture'}
        result = service.predict_recent(38, 23.7, weather_provider=CurrentWeather())
    assert 0 <= result['raw_probability'] <= 1
    assert result['operational_validity'] is False
    assert result['feature_names'] == model.feature_names == production_feature_names()


def test_historical_pipeline_keeps_cube_requirement():
    with patch('src.production_features.FireCubeExp5Inference', side_effect=FileNotFoundError('historical cube required')):
        with pytest.raises(FileNotFoundError, match='historical cube required'):
            ProductionFeaturePipeline(dynamic_provider=HistoricalCubeDynamicProvider())


@pytest.mark.integration
def test_cube_provider_static_and_prediction_parity(provider):
    if not DEFAULT_CUBE.exists():
        pytest.skip('optional local cube absent')
    old = FireCubeStaticFeatureProvider()
    prep = ProductionPreprocessor.load()
    model = XGBoostProductionModel.load(ROOT / 'models/production/xgb_baseline.joblib')
    for lat, lon in ((38., 23.7), (float(provider.arrays['y'][0]), float(provider.arrays['x'][0]))):
        try:
            actual = provider.fetch_static(lat, lon, 2015)
        except FireCubeInferenceError:
            with pytest.raises(FireCubeInferenceError):
                old.fetch_static(lat, lon, 2015)
            continue
        expected = old.fetch_static(lat, lon, 2015)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        compact = ProductionFeaturePipeline(FrozenDynamicWindow(), provider, prep).build(lat, lon, '2015-07-15')
        cube = ProductionFeaturePipeline(FrozenDynamicWindow(), old, prep).build(lat, lon, '2015-07-15')
        np.testing.assert_array_equal(compact.raw_features, cube.raw_features)
        np.testing.assert_array_equal(compact.model_features, cube.model_features)
        assert model.predict_probability(compact.model_features) == model.predict_probability(cube.model_features)
