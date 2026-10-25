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
