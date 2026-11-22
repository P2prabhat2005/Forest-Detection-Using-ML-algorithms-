"""Render the user-facing pages with Streamlit's test runtime."""
from pathlib import Path

from streamlit.testing.v1 import AppTest
import pytest

from src.firecube_inference import DEFAULT_CUBE

APP = Path(__file__).resolve().parents[1] / 'app.py'


def test_dashboard_navigation_and_legacy_weather():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    for page in ('About & Methodology', 'Historical Research', 'Legacy demos'):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
    app.selectbox[0].set_value('Weather MVP').run()
    app.button[0].click().run()
    assert not app.exception
    messages = list(app.success) + list(app.error)
    assert any('P(fire)' in message.value for message in messages)
    app.selectbox[0].set_value('Image MVP').run()
    samples = app.selectbox[1].options
    assert len(samples) > 1
    app.selectbox[1].set_value(samples[1]).run()
    assert not app.exception
    assert any('P(fire)' in message.value for message in list(app.success) + list(app.error))
    app.selectbox[0].set_value('Combined MVP').run()
    app.selectbox[1].set_value(samples[1]).run()
    app.button[0].click().run()
    assert not app.exception
    assert any(metric.label == 'Fused P(fire)' for metric in app.metric)


@pytest.mark.integration
def test_historical_track_b_dashboard_prediction():
    if not DEFAULT_CUBE.exists():
        pytest.skip('local FireCube cube unavailable')
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value('Historical Research').run()
    app.button(key='production_predict_historical').click().run()
    assert not app.exception
    result = app.session_state['production_result']
    assert result['target_date'] == '2020-07-15'
    assert len(result['feature_names']) == 35
    assert 0 <= result['raw_probability'] <= 1


def test_missing_cube_returns_unavailable_service(monkeypatch):
    import app as dashboard

    dashboard.load_production_service.clear()
    monkeypatch.setattr(dashboard, 'load_production_model', lambda: object())
    monkeypatch.setattr(dashboard, 'load_production_preprocessor', lambda: object())

    def missing_pipeline(**kwargs):
        raise FileNotFoundError('dataset_greece.nc')

    monkeypatch.setattr(dashboard, 'ProductionFeaturePipeline', missing_pipeline)
    try:
        assert dashboard.load_production_service(False) is None
    finally:
        dashboard.load_production_service.clear()


def test_location_summary_report_is_specific_and_preserves_limitations():
    import app as dashboard

    class Estimator:
        feature_importances_ = [0.1, 0.9, 0.2]

    class Model:
        estimator = Estimator()

    class Service:
        model = Model()

    result = {
        'resolved_place': 'Athens, Greece',
        'latitude': 37.9756,
        'longitude': 23.7348,
        'target_date': '2026-11-16',
        'raw_probability': 0.017,
        'feature_names': ['max_t2m__mean10', 'lst_day__mean10', 'min_rh__mean10'],
        'feature_values': {'max_t2m__mean10': -0.2, 'lst_day__mean10': 0.1, 'min_rh__mean10': 0.3},
        'missing_features': [],
        'feature_provenance': {
            'max_t2m__mean10': {'approximate': False},
            'lst_day__mean10': {'approximate': True},
            'min_rh__mean10': {'approximate': False},
        },
        'current_weather': {
            'available': True,
            'values': {'temperature_2m': 18.3, 'relative_humidity_2m': 74},
            'units': {'temperature_2m': '°C', 'relative_humidity_2m': '%'},
        },
    }
    report = dashboard._location_summary_report(result, Service(), historical=False)
    assert 'Athens, Greece' in report
    assert '1.7 / 100' in report
    assert '3 / 35' in report
    assert 'lst_day__mean10' in report
    assert 'not a calibrated real-world fire probability' in report
    assert 'not an operational wildfire-warning service' in report


def test_feature_dashboard_labels_and_quality_counts_are_readable():
    import app as dashboard

    result = {
        'missing_features': ['min_rh__mean10'],
        'feature_provenance': {
            'max_t2m__mean10': {'approximate': False},
            'max_tp__mean10': {'approximate': True},
            'min_rh__mean10': {'approximate': False},
        },
    }
    assert dashboard._feature_display_name('max_t2m__mean10') == '2 m air temperature · 10-day mean'
    assert dashboard._feature_display_name('CLC_2018_2__mean') == '2018 land-cover class 2 share · unit mean'
    assert dashboard._feature_quality_counts(
        ['max_t2m__mean10', 'max_tp__mean10', 'min_rh__mean10'], result
    ) == (1, 1, 1)


def test_recorded_athens_briefing_and_return_to_search():
    from json import loads
    import app as dashboard

    example = loads((APP.parent / "examples/athens-2026-11-21.json").read_text(encoding="utf-8"))
    assert len(example["feature_names"]) == len(example["feature_values"]) == 35
    assert dashboard._feature_quality_counts(example["feature_names"], example) == (8, 27, 6)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.radio(key="explore_mode").set_value("Recorded Athens demo").run()
    assert not app.exception
    assert any("Athens · location briefing" in item.value for item in app.subheader)
    assert any("6 filled · 6 approximate" in item.label for item in app.expander)
    assert len(app.get("download_button")) == 2
    assert "production_result" not in app.session_state
    app.radio(key="explore_mode").set_value("Place search").run()
    assert not app.get("download_button")
    next(b for b in app.button if b.label == "Use Athens").click().run()
    assert not app.exception and not app.number_input
    assert not app.button(key="production_predict_live").disabled
    assert not app.get("download_button")


def test_map_explorer_rejects_water_before_inference_and_shows_missing_key(monkeypatch):
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.secrets = {}
    app.toggle[1].set_value(True).run()
    assert any("FIRMS_MAP_KEY" in note.value for note in app.caption)
    app.radio(key="explore_mode").set_value("Coordinates").run()
    app.number_input[0].set_value(36.0)
    app.number_input[1].set_value(25.0)
    next(b for b in app.button if b.label == "Show on map").click().run()
    assert not app.exception
    assert any("non-land FireCube cell" in warning.value for warning in app.warning)
    assert not any(b.key == "production_predict_live" for b in app.button)


def test_shared_location_reopens_exact_point_without_running_inference():
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.query_params["at"] = "37.9755648,23.7348324"
    app.run()
    assert not app.exception
    point = app.session_state["explorer_selection"]
    assert (point["latitude"], point["longitude"]) == (37.9755648, 23.7348324)
    assert any("Model unit 3,156" in note.value for note in app.caption)
    assert any("37.9755648%2C23.7348324" in element.value for element in app.code)
    assert "explorer_result" not in app.session_state
