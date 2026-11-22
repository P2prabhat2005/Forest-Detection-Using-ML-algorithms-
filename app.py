"""Fire Early Warning Streamlit dashboard and preserved research demos."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from html import escape
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from src.firecube_inference import FireCubeExp5Inference, FireCubeInferenceError
from src.explorer_data import (
    FIRMS_WINDOWS,
    coverage_overlay,
    fetch_firms_hotspots,
    model_unit_overlay,
    parse_shared_location,
    share_location_url,
)
from src.explorer_map import explorer_map
from src.firecube_live import HistoricalCubeDynamicProvider
from src.production_features import ProductionFeaturePipeline, ProductionPreprocessor
from src.production_model import ProductionPredictionService, XGBoostProductionModel
from src.place_search import NominatimSearch, PlaceSearchError
from src.predict import ImagePredictor, TabularPredictor, default_weather, fuse_probabilities
from src.static_geography import CompactStaticFeatureProvider, StaticArtifactError

SAMPLES = ROOT / "data" / "images" / "samples"
TABULAR_METRICS = ROOT / "models" / "tabular_metrics.json"
IMAGE_METRICS = ROOT / "models" / "image_metrics.json"
CM_PATH = ROOT / "models" / "tabular_confusion_matrix.png"
PRODUCTION_MODEL = ROOT / "models" / "production" / "xgb_baseline.joblib"
PRODUCTION_PREPROCESSOR = ROOT / "models" / "production" / "feature_preprocessor.joblib"


@st.cache_resource(show_spinner=False)
def load_place_search() -> NominatimSearch:
    return NominatimSearch()


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_tabular() -> TabularPredictor | None:
    try:
        return TabularPredictor()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_image() -> ImagePredictor | None:
    try:
        return ImagePredictor()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_firecube() -> FireCubeExp5Inference | None:
    try:
        return FireCubeExp5Inference()
    except (FileNotFoundError, OSError, ImportError):
        return None


@st.cache_resource(show_spinner=False)
def load_production_model() -> XGBoostProductionModel | None:
    try:
        return XGBoostProductionModel.load(PRODUCTION_MODEL)
    except (FileNotFoundError, OSError, ValueError):
        return None


@st.cache_resource(show_spinner=False)
def load_production_preprocessor() -> ProductionPreprocessor | None:
    try:
        return ProductionPreprocessor.load(PRODUCTION_PREPROCESSOR)
    except (FileNotFoundError, OSError, ValueError):
        return None


@st.cache_resource(show_spinner=False)
def load_production_service(historical: bool) -> ProductionPredictionService | None:
    model = load_production_model()
    preprocessor = load_production_preprocessor()
    if model is None or preprocessor is None:
        return None
    try:
        if historical:
            pipeline = ProductionFeaturePipeline(
                dynamic_provider=HistoricalCubeDynamicProvider(),
                preprocessor=preprocessor,
            )
        else:
            pipeline = ProductionFeaturePipeline(preprocessor=preprocessor, static_provider=load_explorer_geography())
    except (FileNotFoundError, OSError, FireCubeInferenceError):
        return None
    return ProductionPredictionService(pipeline, model)


@st.cache_resource(show_spinner=False)
def load_explorer_geography() -> CompactStaticFeatureProvider:
    return CompactStaticFeatureProvider()


@st.cache_data(show_spinner=False)
def _explorer_coverage() -> dict:
    return coverage_overlay(load_explorer_geography())


@st.cache_data(ttl=300, max_entries=24, show_spinner=False)
def _recent_hotspots(key: str, bounds: list, window: str) -> list[dict]:
    return fetch_firms_hotspots(key, bounds, window)


def risk_banner(result: dict) -> None:
    p = result["p_fire"]
    label = result["label"]
    if label == "fire":
        st.error(f"Fire likely  ·  P(fire) = {p:.1%}")
    else:
        st.success(f"No fire likely  ·  P(fire) = {p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))


def weather_form(prefix: str) -> dict:
    defaults = default_weather()
    cols = st.columns(2)
    values = {}
    fields = [
        ("Temperature", "Temperature (°C)", 0.0, 50.0),
        ("RH", "Relative humidity (%)", 0.0, 100.0),
        ("Ws", "Wind speed (km/h)", 0.0, 40.0),
        ("Rain", "Rain (mm)", 0.0, 20.0),
        ("FFMC", "FFMC", 0.0, 100.0),
        ("DMC", "DMC", 0.0, 200.0),
        ("DC", "DC", 0.0, 800.0),
        ("ISI", "ISI", 0.0, 20.0),
        ("BUI", "BUI", 0.0, 200.0),
        ("FWI", "FWI", 0.0, 40.0),
    ]
    for i, (key, label, lo, hi) in enumerate(fields):
        with cols[i % 2]:
            values[key] = st.number_input(
                label,
                min_value=float(lo),
                max_value=float(hi),
                value=float(defaults[key]),
                key=f"{prefix}_{key}",
            )
    return values


def pick_image(key_prefix: str) -> Image.Image | None:
    uploaded = st.file_uploader(
        "Upload a photo", type=["jpg", "jpeg", "png"], key=f"{key_prefix}_upload"
    )
    sample_paths = sorted(SAMPLES.glob("*/*.jpg"))
    sample_names = ["(none)"] + [p.relative_to(SAMPLES).as_posix() for p in sample_paths]
    choice = st.selectbox("Or use a bundled sample", sample_names, key=f"{key_prefix}_sample")
    if uploaded is not None:
        return Image.open(uploaded)
    if choice != "(none)":
        return Image.open(SAMPLES / choice)
    return None


def firecube_section() -> None:
    """Historical FireCube product surface; no live-data or threshold logic."""
    st.subheader("FireCube Early Warning")
    st.warning(
        "HISTORICAL FireCube inference only. This prototype reads the local "
        "2009–2021 FireCube cube; it does not call live APIs."
    )

    predictor = load_firecube()
    if predictor is None:
        st.error("FireCube model or historical cube is unavailable in this environment.")
        return

    st.caption("Validated demo defaults: 38.0°N, 23.7°E, 2020-07-15")
    c1, c2, c3 = st.columns(3)
    with c1:
        latitude = st.number_input(
            "Latitude", min_value=34.0, max_value=43.0, value=38.0, step=0.01, format="%.4f"
        )
    with c2:
        longitude = st.number_input(
            "Longitude", min_value=18.0, max_value=29.5, value=23.7, step=0.01, format="%.4f"
        )
    with c3:
        target_date = st.date_input("Historical date", value=date(2020, 7, 15))

    run_demo = st.button("Run historical FireCube prediction", type="primary")
    selection = (latitude, longitude, target_date)
    if run_demo or st.session_state.get("firecube_selection") != selection:
        st.session_state.pop("firecube_result", None)
    if run_demo:
        try:
            st.session_state["firecube_result"] = predictor.predict_historical(
                latitude, longitude, target_date, calibration="exp7_flat"
            )
            st.session_state["firecube_selection"] = selection
        except (FireCubeInferenceError, ValueError, OSError) as exc:
            st.error(f"Historical inference could not be completed: {exc}")

    result = st.session_state.get("firecube_result")
    if result is None:
        st.info("Choose a historical location/date and run the prediction.")
        return

    st.markdown("**HISTORICAL FireCube inference result**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw model probability", f"{result['raw_probability']:.6f}")
    calibrated = result.get("calibrated_probability")
    m2.metric("Exp7-calibrated probability", "Unavailable" if calibrated is None else f"{calibrated:.6f}")
    m3.metric("Mapped 10 km unit", str(result["unit_id"]))
    m4.metric("Input features", f"{result['feature_count']} / 35")

    st.write(
        f"Selected location: **{result['latitude']:.4f}, {result['longitude']:.4f}**  ·  "
        f"Grid cell: **({result['grid_y']}, {result['grid_x']})**  ·  "
        f"Unit area: **{result['unit_area_km2']:.2f} km²**"
    )
    st.write(
        f"Data date: **{result['date']}**  ·  Predictor window: **{result['predictor_window']}**  ·  "
        f"Source: local historical FireCube cube"
    )

    st.markdown("**Selected model inputs**")
    key_names = [
        "max_t2m__mean10", "min_rh__mean10", "max_wind_speed__mean10",
        "ndvi__mean10", "lst_day__mean10", "sminx__mean10",
        "DEM__mean", "SLOPE__mean", "ROAD_DISTANCE__min", "POP_DENS__mean",
    ]
    key_rows = [
        {"feature": name, "value": result["feature_values"][name]}
        for name in key_names if name in result["feature_values"]
    ]
    st.dataframe(pd.DataFrame(key_rows), hide_index=True, width="stretch")
    with st.expander("Show all 35 transformed feature values"):
        st.dataframe(
            pd.DataFrame(
                [{"feature": name, "value": result["feature_values"][name]} for name in result["feature_names"]]
            ),
            hide_index=True,
            width="stretch",
        )

    st.markdown("**Selected location map**")
    st.map(pd.DataFrame({"latitude": [result["latitude"]], "longitude": [result["longitude"]]}), zoom=7)


def _demo_styles() -> None:
    """Presentation-only design system; all inference stays in src/."""
    st.markdown(
        """
        <style>
        :root { --pine:#123e38; --ink:#18332f; --muted:#49645a; --line:#cbdcd4; --paper:#fff; --mist:#f3f7f5; --amber:#b85d24; }
        .stApp { background: var(--mist); color: var(--ink); }
        .block-container { max-width: 1160px; padding-top:3.5rem; padding-bottom:4rem; }
        section[data-testid="stSidebar"] { background: #102e2b; }
        section[data-testid="stSidebar"] * { color: #eef7f1 !important; }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius: 10px; }
        h1, h2, h3 { letter-spacing: -.03em; }
        .eyebrow, .section-kicker { font-size: .72rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; }
        .section-kicker { color: #39735c; margin-bottom: .2rem; }
        .explorer-head { background:#123e38; border-radius:16px; padding:clamp(1rem,3vw,1.9rem); }
        .explorer-head h1 { color:white; margin:0; padding:0; line-height:1.15; font-size:clamp(1.7rem,4vw,2.65rem); }
        .explorer-head .brand { color:#c1e3cc; font-size:.86rem; margin-bottom:.4rem; }
        .reading-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem; margin:0; }
        .reading-grid > div { background:white; border:1px solid var(--line); border-radius:12px; padding:.8rem 1rem; }
        .reading-grid dt { color:#35554a; font-size:.88rem; font-weight:600; }
        .reading-grid dd { color:#123e38; font-size:clamp(1.25rem,2.3vw,1.9rem); font-weight:750; margin:0; }
        .stMain h2,.stMain h3 { color:#123e38; }
        [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
        [data-testid="stMain"] [role="radiogroup"] label p { color:#35554a; }
        .stMain button:focus-visible { outline:3px solid #2271ad; outline-offset:2px; }
        @media(max-width:800px){.reading-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
        .panel-title { font-size:1.23rem; font-weight:750; margin:.35rem 0 .2rem; color:var(--ink); }
        .location-card { background:#ecf5ef; border:1px solid #d4e8db; border-left:4px solid #31775b; border-radius:12px; padding:.9rem 1rem; margin:.55rem 0; }
        .location-card small { color:#536d61; }
        .score-card { background:linear-gradient(125deg,#173c34,#245b48); color:#fff; padding:1.4rem 1.6rem; border-radius:18px; min-height:180px; box-shadow:0 9px 21px rgba(22,63,48,.14); }
        .score-card .score-number { font-size:clamp(2.6rem,5vw,4.4rem); font-weight:800; line-height:1.1; letter-spacing:-.06em; margin:.45rem 0; }
        .score-card .score-unit { font-size:1.15rem; font-weight:500; color:#bfe3cb; }
        .score-card .score-note { color:#d8eee1; font-size:.85rem; }
        .mini-card { background:white; border:1px solid var(--line); border-radius:14px; padding:.9rem 1rem; min-height:92px; }
        .mini-card strong { display:block; font-size:.85rem; color:#49675a; margin-bottom:.3rem; }
        .mini-card span { font-size:1.05rem; font-weight:700; color:#193c33; }
        div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding:.85rem 1rem; box-shadow:0 3px 12px rgba(24,67,53,.05); }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, div[data-testid="stMetric"] label, div[data-testid="stMetric"] label * { color:#35554a !important; opacity:1 !important; font-weight:750 !important; }
        div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * { color:#102f28 !important; opacity:1 !important; font-weight:800; }
        div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * { color:#1d624b !important; opacity:1 !important; }
        [data-testid="stDownloadButton"] button { background:#225e49 !important; border-color:#225e49 !important; }
        [data-testid="stDownloadButton"] button, [data-testid="stDownloadButton"] button * { color:#fff !important; font-weight:750; }
        [data-testid="stDownloadButton"] button:hover { background:#194b3b !important; }
        .stMain [data-testid="stCaptionContainer"], .stMain [data-testid="stCaptionContainer"] * { color:#405d52 !important; opacity:1 !important; }
        div[data-testid="stAlert"] { background:#eaf2f6; border:1px solid #c4d8e1; box-shadow:none; }
        div[data-testid="stAlert"] > div { background:transparent; }
        div[data-testid="stAlert"] *, div[data-testid="stAlert"] p { color:#163f5f !important; opacity:1 !important; font-weight:550; }
        div[data-testid="stExpander"] details > summary { background:#173c34 !important; border-radius:10px; padding:.55rem .8rem; }
        div[data-testid="stExpander"] details > summary:hover { background:#245746 !important; }
        div[data-testid="stExpander"] details > summary, div[data-testid="stExpander"] details > summary * { color:#fff !important; opacity:1 !important; font-weight:700; }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(>div[data-testid="stVerticalBlock"]) { border-color:var(--line); }
        .stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] { background:#225e49; border-color:#225e49; border-radius:9px; font-weight:700; }
        .stButton>button[kind="primary"]:hover, .stFormSubmitButton>button[kind="primary"]:hover { background:#194b3b; border-color:#194b3b; }
        .st-key-explorer_layout [data-testid="stColumn"] { min-width:0; }
        .st-key-explorer_controls { margin-bottom:.4rem; }
        .explorer-head {padding:.95rem 1.25rem}
        .st-key-explorer_controls button[data-testid^="stBaseButton-secondary"],
        .st-key-explorer_map_pane button[data-testid^="stPopoverButton"] {
          background:#fff; color:#173e36; border:1px solid #91b5a7;
        }
        .st-key-explorer_controls button[data-testid^="stBaseButton-secondary"] p,
        .st-key-explorer_map_pane button[data-testid^="stPopoverButton"] p {color:#173e36}
        .st-key-explorer_controls button[data-testid^="stBaseButton-primary"],
        .st-key-explorer_briefing button[data-testid^="stBaseButton-primary"] {
          background:#22644f; color:#fff; border-color:#22644f;
        }
        .explorer-legend {display:flex; flex-wrap:wrap; gap:.65rem 1rem; color:#24483e; font-size:.83rem}
        .explorer-legend span {display:inline-flex; gap:.35rem; align-items:center}
        .explorer-legend i {width:11px; height:11px; display:inline-block; border:1px solid #314d46}
        .legend-coverage {background:#7dc1a9; border-radius:2px}
        .legend-selection {background:#006857; border-radius:100%}
        .legend-unit {background:#6051bb; border-radius:2px}
        .legend-hotspot {background:#ff9a44; border-color:#9a3400 !important; border-radius:100%}
        @media(min-width:851px) {
          .st-key-explorer_layout [data-testid="stColumn"]:has(.st-key-explorer_map_pane) {position:sticky; top:3.5rem; align-self:flex-start}
          .st-key-explorer_briefing .reading-grid {grid-template-columns:repeat(2,minmax(0,1fr))}
        }
        @media(max-width:850px){
          .st-key-explorer_layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {flex-direction:column}
          .st-key-explorer_layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            width:100% !important; flex:1 1 auto !important;
          }
        }
        @media(max-width:700px){.block-container{padding:3rem 1rem}.score-card{min-height:0}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _reading_grid(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        f"<div><dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>"
        for label, value in items
    )
    st.markdown(f"<dl class='reading-grid'>{cards}</dl>", unsafe_allow_html=True)


def _weather_cards(current: dict) -> None:
    """Show current conditions separately from the historical feature window."""
    values = current.get("values", {})
    units = current.get("units", {})
    labels = [
        ("temperature_2m", "Temperature"),
        ("relative_humidity_2m", "Humidity"),
        ("precipitation", "Precipitation"),
        ("wind_speed_10m", "Wind speed"),
    ]
    available = [(key, label) for key, label in labels if key in values]
    if not available:
        st.warning("Current weather was returned without displayable values. The prediction uses its separate recent-history window.")
        return
    readings = []
    for key, label in available:
        value = values[key]
        unit = units.get(key, "")
        readings.append((label, "—" if value is None else f"{value:g} {unit}".strip()))
    _reading_grid(readings)
    st.caption(
        f"Current modelled weather at {current.get('timestamp', 'unknown time')} UTC · "
        f"retrieved {current.get('retrieved_at', 'during this request')}. "
        "This display is contextual; the model uses the preceding ten completed UTC days."
    )


def _top_global_features(result: dict, service: ProductionPredictionService | None, limit: int = 3) -> list[str]:
    """Return model-wide feature names for a concise dashboard summary.

    These are intentionally labelled model-wide: they are not causal or
    location-specific explanations of the score.
    """
    estimator = getattr(getattr(service, "model", None), "estimator", None)
    importances = getattr(estimator, "feature_importances_", None)
    names = result.get("feature_names", [])
    if importances is None or len(importances) != len(names):
        return []
    indices = sorted(range(len(importances)), key=lambda index: float(importances[index]), reverse=True)
    return [str(names[index]) for index in indices[:limit]]


def _location_summary_report(result: dict, service: ProductionPredictionService | None, historical: bool) -> str:
    """Build a download-ready, location-specific summary without changing inference."""
    feature_count = len(result.get("feature_values", {}))
    missing = list(result.get("missing_features", []))
    approximate = sorted(
        name for name, info in result.get("feature_provenance", {}).items() if info.get("approximate")
    )
    current = result.get("current_weather", {})
    values = current.get("values", {}) if current.get("available") else {}
    units = current.get("units", {})
    weather_summary = ", ".join(
        f"{label}: {values[key]:g} {units.get(key, '')}".strip()
        for key, label in (
            ("temperature_2m", "temperature"),
            ("relative_humidity_2m", "humidity"),
            ("precipitation", "precipitation"),
            ("wind_speed_10m", "wind speed"),
        )
        if values.get(key) is not None
    ) or "Current weather context was unavailable."
    top_features = _top_global_features(result, service)
    score = float(result["raw_probability"]) * 100
    mode = _run_mode(result, historical)
    return "\n".join([
        "# ForestSentinels Location Prediction Summary",
        "",
        "## Requested location",
        f"- Place: {result.get('resolved_place', 'Manual coordinates')}",
        f"- Coordinates: {result.get('latitude', 'unknown')} N, {result.get('longitude', 'unknown')} E",
        f"- Target date: {result.get('target_date', 'unknown')}",
        f"- Mode: {mode}",
        "",
        "## Experimental model output",
        f"- Track-B XGBoost baseline score: {score:.1f} / 100",
        "- This is an experimental model score, not a calibrated real-world fire probability.",
        "- No operational risk category, alert threshold, or active-fire claim is assigned.",
        "",
        "## Input readiness and quality",
        f"- Ordered feature contract: {feature_count} / 35 features prepared.",
        f"- Missing/imputed features: {len(missing)}.",
        f"- Approximate/proxy features: {len(approximate)}.",
        "- Model window: ten preceding completed UTC days.",
        "- Filled and approximate flags can overlap.",
        *([] if historical else [f"- Weather snapshot ({current.get('timestamp', 'time unavailable')} UTC): {weather_summary}"]),
        "",
        "## Model-wide context",
        "- Most-used model-wide features: " + (", ".join(top_features) if top_features else "not available"),
        "- Model-wide feature importance is not a causal or per-location explanation.",
        "",
        "## Interpretation boundary",
        "This Greece-domain research prototype makes its data sources, missingness and proxies visible. "
        "It is not an operational wildfire-warning service and is not globally validated.",
    ])


def _run_mode(result: dict, historical: bool) -> str:
    if result.get("recorded_demo"):
        return "Recorded Athens demo"
    return "Historical replay" if historical else "Live prototype"


def _report_downloads(result: dict, service: ProductionPredictionService | None, historical: bool) -> None:
    report = _location_summary_report(result, service, historical)
    place = result.get("resolved_place", "coordinates").split(",")[0]
    slug = re.sub(r"[^a-z0-9]+", "-", place.lower()).strip("-") or "location"
    stem = f"forestsentinels-{slug}-{result['target_date']}"
    paragraphs = []
    for line in report.splitlines():
        if line.startswith("# "):
            paragraphs.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            paragraphs.append(f"<h2>{escape(line[3:])}</h2>")
        elif line:
            paragraphs.append(f"<p>{escape(line.removeprefix('- '))}</p>")
    printable = """<!doctype html><html lang="en"><meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>ForestSentinels location briefing</title><style>
        body{font:16px/1.45 system-ui;color:#163c34;max-width:46rem;margin:2rem auto;padding:0 1rem}
        h1{line-height:1.12}h2{font-size:1.05rem;border-bottom:1px solid #aac4b6;margin:1rem 0 .2rem}
        p{margin:.2rem 0;overflow-wrap:anywhere}@page{size:A4;margin:15mm}
        @media print{body{font-size:10pt;margin:0;padding:0}h2{break-after:avoid}}
        </style><main>""" + "".join(paragraphs) + "</main></html>"
    st.subheader("Take this briefing with you")
    st.caption("Open the printable report in a browser and choose Print or Save as PDF.")
    with st.container(horizontal=True):
        st.download_button("Download printable report", printable, stem + ".html", "text/html", on_click="ignore")
        st.download_button("Download Markdown", report, stem + ".md", "text/markdown", on_click="ignore")


def _research_snapshot() -> None:
    st.markdown("<div class='section-kicker'>Evidence and limitations</div>", unsafe_allow_html=True)
    st.subheader("Research results in context")
    st.write(
        "FireCube Exp1–Exp7 establish historical Greece benchmarks, including the direct 10 km XGBoost baseline "
        "and calibration experiments. Track-B is a separate production-feature baseline trained on historical "
        "samples. Its validation scores do not establish current-day accuracy, cross-region performance, or "
        "calibrated real-world occurrence probabilities."
    )
    st.warning(
        "The training and validation samples have an artificial event prevalence. The live data pipeline also "
        "contains source mismatches and approximate inputs. No operational alert threshold is defined."
    )


def _limitations_scope() -> None:
    with st.expander("Limitations & future scope", expanded=False):
        st.markdown(
            "- Use more source-aligned environmental data where available.\n"
            "- Validate across global and additional regional settings.\n"
            "- Adapt features and evaluation to regional conditions.\n"
            "- Add FIRMS, contextual and news signals later.\n"
            "- Add reviewed alerting workflows later; this MVP does not issue alerts."
        )


def _feature_group(name: str) -> str:
    if name.startswith(("ndvi", "lst_")):
        return "Vegetation / EO"
    if name.startswith("sminx"):
        return "Soil moisture"
    if name.startswith(("DEM", "SLOPE", "ROAD_", "WATERWAY_", "POP_", "CLC_")):
        return "Terrain / land cover"
    return "Weather"


def _feature_display_name(name: str) -> str:
    """Turn production schema names into concise labels for the faculty view."""
    base, _, suffix = name.partition("__")
    aliases = {
        "max_t2m": "2 m air temperature",
        "max_d2m": "2 m dew point",
        "min_rh": "relative humidity",
        "max_tp": "precipitation",
        "max_sp": "surface pressure",
        "max_wind_speed": "wind speed",
        "ndvi": "vegetation index (NDVI)",
        "lst_day": "day land-surface temperature",
        "lst_night": "night land-surface temperature",
        "sminx": "soil moisture",
        "DEM": "terrain elevation",
        "SLOPE": "terrain slope",
        "ROAD": "road-distance context",
        "WATERWAY": "waterway-distance context",
        "POP_DENS": "population-density context",
    }
    if base.startswith("CLC_2018_"):
        label = f"2018 land-cover class {base.rsplit('_', 1)[-1]} share"
    elif base.startswith("ROAD_"):
        label = aliases["ROAD"]
    elif base.startswith("WATERWAY_"):
        label = aliases["WATERWAY"]
    elif base.startswith("POP_DENS"):
        label = aliases["POP_DENS"]
    else:
        label = aliases.get(base, base.replace("_", " ").lower())
    suffixes = {"mean10": "10-day mean", "last_tminus1": "previous completed day", "mean": "unit mean"}
    return f"{label} · {suffixes[suffix]}" if suffix in suffixes else label


def _feature_quality_counts(names: list[str], result: dict) -> tuple[int, int, int]:
    missing = set(result.get("missing_features", []))
    provenance = result.get("feature_provenance", {})
    missing_count = sum(name in missing for name in names)
    proxy_count = sum(bool(provenance.get(name, {}).get("approximate")) for name in names)
    available_count = sum(name not in missing and not provenance.get(name, {}).get("approximate") for name in names)
    return available_count, proxy_count, missing_count


def _input_evidence_dashboard(result: dict) -> None:
    st.subheader("Input quality")
    names = result["feature_names"]
    _, proxy, filled = _feature_quality_counts(names, result)
    st.write(f"{len(result['feature_values'])} of 35 inputs prepared. The preprocessor filled {filled} missing inputs. "
             f"{proxy} inputs carry an approximation flag. An input can have both flags.")
    table = _feature_rows(result)
    for group in ("Weather", "Vegetation / EO", "Soil moisture", "Terrain / land cover"):
        rows = table[table["Group"] == group]
        _, approximate, missing = _feature_quality_counts(rows["Feature"].tolist(), result)
        with st.expander(f"{group} · {len(rows)} inputs · {missing} filled · {approximate} approximate"):
            sources = " · ".join(rows["Source"].unique())
            st.write(f"Sources · {sources}")
            st.caption("Model inputs are transformed values. The table gives the latest source date for each input.")
            st.dataframe(rows.drop(columns="Group"), hide_index=True, width="stretch")


def _feature_rows(result: dict) -> pd.DataFrame:
    missing = set(result.get("missing_features", []))
    rows = []
    for name in result.get("feature_names", []):
        info = result.get("feature_provenance", {}).get(name, {})
        dates = info.get("dates", [])
        rows.append({
            "Group": _feature_group(name), "Feature": name,
            "Model input": result.get("feature_values", {}).get(name),
            "Source": ", ".join(info.get("sources", [])) or "Unavailable",
            "Latest source date": max(dates) if dates else "—",
            "Filled": name in missing,
            "Approximate": bool(info.get("approximate")),
        })
    return pd.DataFrame(rows)


def _weather_history(result: dict) -> None:
    rows = result.get("weather_history", [])
    with st.expander("Weather history · 10 completed UTC days"):
        if not rows:
            st.caption("Daily weather readings were not saved for this run.")
            return
        labels = {
            "temperature_c": "Daily high (°C)",
            "peak_hourly_precipitation_mm": "Peak hourly precipitation (mm)",
            "wind_kmh": "Daily peak wind (km/h)",
        }
        frame = pd.DataFrame(rows).rename(columns={"date": "UTC day", **labels})
        st.caption("Open-Meteo inputs before standardization and imputation. Blank days indicate missing values. Precipitation is the highest hourly reading for the day.")
        variable = st.selectbox("Weather variable", list(labels.values()))
        chart = alt.Chart(frame).mark_bar(color="#227a62").encode(
            x=alt.X("UTC day:O", sort=frame["UTC day"].tolist(), axis=alt.Axis(labelAngle=-42)),
            y=alt.Y(field=variable, type="quantitative", title=variable),
            tooltip=["UTC day", alt.Tooltip(field=variable, type="quantitative", format=".1f")],
        )
        st.altair_chart(chart, height=190, width="stretch")
        st.dataframe(frame, hide_index=True, width="stretch", column_config={
            label: st.column_config.NumberColumn(format="%.2f") for label in labels.values()
        })
        st.download_button(
            "Download 10-day model weather CSV",
            pd.DataFrame(rows).to_csv(index=False),
            file_name=f"forestsentinels-weather-{result['target_date']}.csv",
            mime="text/csv",
            on_click="ignore",
        )


def _dashboard_result(
    result: dict, service: ProductionPredictionService | None, historical: bool, *, show_map: bool = True
) -> None:
    place = result.get("resolved_place", "Manual coordinates")
    short_name = place.split(",")[0]
    target = date.fromisoformat(result["target_date"])
    mode = _run_mode(result, historical)
    st.subheader(f"{short_name} · location briefing")
    st.caption(f"{mode} · Target {target:%d %b %Y} UTC")
    if show_map:
        summary, map_area = st.columns([1, 1])
        with map_area:
            st.map(pd.DataFrame({"latitude": [result["latitude"]], "longitude": [result["longitude"]]}), zoom=8, height=210, width="stretch")
            st.caption("The marker identifies the selected coordinate.")
    else:
        summary = st.container()
    with summary:
        st.markdown(
            "<div class='score-card'><div class='eyebrow'>Experimental Track-B score</div>"
            f"<div class='score-number'>{result['raw_probability'] * 100:.1f}<span class='score-unit'> / 100</span></div>"
            "<div class='score-note'>Research score on a 0–100 scale. Fire probability and alert thresholds have not been established.</div></div>",
            unsafe_allow_html=True,
        )
        st.write(f"Model input window · {target - timedelta(days=10):%d %b %Y} – {target - timedelta(days=1):%d %b %Y} UTC")
        with st.expander("Location details"):
            st.text(place)
            st.write(f"{result['latitude']:.5f}° N, {result['longitude']:.5f}° E")
    if not historical:
        st.subheader("Weather at the time of the run")
        current = result.get("current_weather", {})
        if current.get("available"):
            _weather_cards(current)
        else:
            st.info(f"Weather snapshot unavailable. {current.get('message', '')}")
        _weather_history(result)
    _input_evidence_dashboard(result)
    with st.expander("How the saved model uses its inputs"):
        st.write("The highest-ranked inputs below describe the fitted model across locations. Individual scores can depend on different inputs.")
        ranked = _top_global_features(result, service, limit=5)
        if ranked:
            weights = dict(zip(result["feature_names"], service.model.estimator.feature_importances_))
            frame = pd.DataFrame({"Input": [_feature_display_name(n) for n in ranked],
                                  "Global importance": [float(weights[n]) for n in ranked]})
            st.bar_chart(frame, x="Input", y="Global importance", horizontal=True, color="#2d715b", sort=False)
            st.dataframe(frame, hide_index=True, width="stretch")
        else:
            st.caption("Model-wide importance values are unavailable for this run.")
    _report_downloads(result, service, historical)


def _hotspots_for_map(bounds: list) -> tuple[list[dict], str]:
    with st.popover("Satellite hotspots"):
        enabled = st.toggle("Show NASA FIRMS hotspots")
        window = st.selectbox("Observation window", FIRMS_WINDOWS, index=1, disabled=not enabled)
        st.caption("NOAA-20 VIIRS thermal detections. Times are UTC. The layer can include non-wildfire heat sources.")
        st.link_button("Open NASA FIRMS", "https://firms.modaps.eosdis.nasa.gov/map/")
    if not enabled:
        return [], ""
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        try:
            key = st.secrets.get("FIRMS_MAP_KEY")
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            key = None
    if not key:
        return [], "NASA FIRMS layer is unavailable until a server-side FIRMS_MAP_KEY is configured."
    try:
        with st.spinner("Loading NASA FIRMS observations…"):
            points = _recent_hotspots(key, bounds, window)
    except (OSError, ValueError, TypeError):
        return [], "NASA FIRMS observations are temporarily unavailable. Map selection and Track-B remain available."
    return points, f"NOAA-20 VIIRS · {window} · {len(points):,} observations in the map region."


def _choose_place() -> None:
    index = st.session_state["production_place_choice"]
    st.session_state["explorer_selection"] = st.session_state["production_places"][index]
    st.session_state["explorer_focus"] = True


def _select_coordinates(latitude: float, longitude: float, *, focus: bool) -> None:
    if math.isfinite(latitude) and math.isfinite(longitude) and -90 <= latitude <= 90 and -180 <= longitude <= 180:
        st.session_state["explorer_selection"] = {
            "name": "Selected location",
            "latitude": latitude,
            "longitude": longitude,
        }
        st.session_state["explorer_focus"] = focus


def _search_explorer() -> None:
    with st.form("production_place_search", border=False):
        search_field, search_action, athens_action = st.columns([5, 1.25, 1.25], vertical_alignment="bottom")
        query = search_field.text_input("City or municipality", placeholder="Try Athens, Greece")
        with search_action:
            submitted = st.form_submit_button("Search place", type="primary", width="stretch")
        with athens_action:
            athens = st.form_submit_button("Use Athens", help="Select Athens city centre for a new live run.", width="stretch")
    if submitted or athens:
        places = []
        if athens:
            places = [{"name": "Athens, Greece", "latitude": 37.9755648, "longitude": 23.7348324}]
        else:
            try:
                with st.spinner("Finding matching places…"):
                    places = load_place_search().search(query)
                if not places:
                    st.info("No match. Try a city and country, pick a point on the map, or enter coordinates.")
            except PlaceSearchError as exc:
                st.error(f"Place search unavailable. Use the map or enter coordinates. {exc}")
        st.session_state["production_places"] = places
        st.session_state["production_place_choice"] = 0
        if places:
            _choose_place()
    places = st.session_state.get("production_places", [])
    if len(places) > 1:
        st.selectbox("Matching places", range(len(places)), format_func=lambda i: places[i]["name"],
                     key="production_place_choice", on_change=_choose_place)


def _run_location_analysis(point: dict | None, unit: dict | None, selection_error: str | None, result_slot) -> None:
    with result_slot:
        if point is None:
            st.subheader("Location briefing")
            st.write("Search for a place or select a point on the map to start.")
            return
        name, lat, lon = point["name"], point["latitude"], point["longitude"]
        previous = st.session_state.get("explorer_result")
        if previous is not None and (previous["latitude"], previous["longitude"], previous["resolved_place"]) != (lat, lon, name):
            st.session_state.pop("explorer_result", None)
        st.caption(f"{name} · {lat:.5f}, {lon:.5f}")
        if unit is None:
            st.warning(selection_error)
            return
        st.caption(f"Eligible FireCube land cell · Model unit {unit['id']:,} · {unit['landAreaKm2']:.1f} km² mapped land")
        with st.expander("Share this location"):
            st.code(share_location_url(st.context.url, lat, lon), language=None)
            st.caption("The link opens these coordinates. Visitors can run their own analysis.")
        service = load_production_service(historical=False)
        if service is None:
            st.error("Prediction assets are unavailable. Run `python -m src.check_setup` for details.")
            return
        if st.button("Get experimental score", type="primary", key="production_predict_live", width="stretch"):
            st.session_state.pop("explorer_result", None)
            try:
                with st.spinner("Collecting recent inputs and running Track-B…"):
                    result = service.predict_recent(lat, lon)
                st.session_state["explorer_result"] = {**result, "resolved_place": name}
            except (FireCubeInferenceError, ValueError, OSError, RuntimeError) as exc:
                st.error(f"Prediction unavailable. {exc}")
                st.info("Retry this location or switch to the Recorded Athens demo.")
        result = st.session_state.get("explorer_result")
        if result is not None:
            _dashboard_result(result, service, historical=False, show_map=False)


def live_explorer() -> None:
    st.markdown("<header class='explorer-head'><div class='brand'>ForestSentinels · Greece</div>"
                "<h1>Wildfire map explorer</h1></header>", unsafe_allow_html=True)
    if "explorer_link_loaded" not in st.session_state:
        st.session_state["explorer_link_loaded"] = True
        if "at" in st.query_params:
            try:
                _select_coordinates(*parse_shared_location(st.query_params["at"]), focus=True)
            except ValueError:
                st.warning("This location link is invalid. Search or select a point on the map.")
    with st.container(key="explorer_controls"):
        mode = st.radio("Choose how to explore", ["Place search", "Coordinates", "Recorded Athens demo"],
                        horizontal=True, key="explore_mode", label_visibility="collapsed")
        recorded = mode == "Recorded Athens demo"
        if recorded:
            example = _load_json(ROOT / "examples" / "athens-2026-11-21.json")
            if example is None:
                st.error("The recorded Athens example is unavailable in this installation.")
                return
            st.info("Saved Athens run from 21 November 2026 UTC. All briefing readings and the score come from that recording.")
        elif mode == "Place search":
            _search_explorer()
        else:
            with st.form("explorer_coordinate_form", border=False):
                c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
                prior = st.session_state.get("explorer_selection", {"latitude": 38.0, "longitude": 23.7})
                lat = c1.number_input("Latitude", min_value=-90.0, max_value=90.0, value=prior["latitude"], format="%.5f")
                lon = c2.number_input("Longitude", min_value=-180.0, max_value=180.0, value=prior["longitude"], format="%.5f")
                with c3:
                    if st.form_submit_button("Show on map"):
                        _select_coordinates(lat, lon, focus=True)
    point = example if recorded else st.session_state.get("explorer_selection")
    try:
        geo = load_explorer_geography()
        layer = _explorer_coverage()
    except StaticArtifactError:
        geo = None
        layer = {"bounds": [[34.3, 18.7], [42.3, 28.9]], "image": None}
    unit, selection_error = None, None
    if point and geo is not None:
        try:
            _, _, location = geo.fetch_static(point["latitude"], point["longitude"], date.today().year)
            unit = model_unit_overlay(geo, location)
        except FireCubeInferenceError as exc:
            selection_error = f"Analysis unavailable at this location. {exc}."
    elif point:
        selection_error = "Greece model coverage is unavailable. Restore the compact geography artifact before running analysis."
    with st.container(key="explorer_layout"):
        left, right = st.columns([2.35, 1.25], vertical_alignment="top")
        with left, st.container(key="explorer_map_pane"):
            controls = st.container(horizontal=True, vertical_alignment="center")
            with controls:
                show_coverage = st.toggle("Model coverage", value=True, disabled=geo is None)
                points, satellite_note = _hotspots_for_map(layer["bounds"]) if not recorded else ([], "")
            map_event = explorer_map(data={
                "bounds": layer["bounds"], "coverage": layer["image"], "showCoverage": show_coverage and geo is not None,
                "selection": [point["latitude"], point["longitude"]] if point else None,
                "focusSelection": recorded or st.session_state.get("explorer_focus", False),
                "allowPick": not recorded, "hotspots": points, "unit": unit,
            })
            legend = ["<span><i class='legend-coverage'></i>Eligible land cells</span>"] if show_coverage and geo else []
            if point:
                legend.append("<span><i class='legend-selection'></i>Selected point</span>")
            if unit:
                legend.append("<span><i class='legend-unit'></i>Selected model unit</span>")
            if points:
                legend.append("<span><i class='legend-hotspot'></i>Satellite thermal detection</span>")
            if legend:
                st.markdown("<div class='explorer-legend'>" + "".join(legend) + "</div>", unsafe_allow_html=True)
            if satellite_note:
                st.caption(satellite_note)
            if not recorded:
                st.caption("Click the map to select a point · Place search by OpenStreetMap Nominatim")
        with right:
            briefing = st.container(key="explorer_briefing", border=True)
        if not recorded and getattr(map_event, "pick", None) is not None:
            _select_coordinates(*map_event.pick, focus=False)
            st.rerun()
        if recorded:
            with briefing:
                _dashboard_result({**example, "recorded_demo": True}, None, historical=False, show_map=False)
        else:
            _run_location_analysis(point, unit, selection_error, briefing)


def production_section(historical: bool = False) -> None:
    if not historical:
        live_explorer()
        return

    st.subheader("Revisit a FireCube-era date")
    st.info("Local historical inference only · Greece FireCube domain · example: 38.0000°N, 23.7000°E on 2020-07-15.")
    c1, c2, c3 = st.columns(3)
    lat = c1.number_input("Latitude", min_value=34.0, max_value=43.0, value=38.0, step=0.01, format="%.4f", key="production_latitude")
    lon = c2.number_input("Longitude", min_value=18.0, max_value=29.5, value=23.7, step=0.01, format="%.4f", key="production_longitude")
    target_date = c3.date_input("Historical target date", value=date(2020, 7, 15), key="production_historical_date").isoformat()
    service = load_production_service(historical=True)
    if service is None:
        st.error("Historical replay needs the local FireCube dataset and saved Track-B assets. Run `python -m src.check_setup --historical` for details.")
        return
    if st.button("Run historical prediction", type="primary", key="production_predict_historical"):
        st.session_state.pop("production_result", None)
        try:
            with st.spinner("Running historical Track-B replay…"):
                st.session_state["production_result"] = service.predict(lat, lon, target_date)
        except (FireCubeInferenceError, ValueError, OSError, RuntimeError) as exc:
            st.error(f"Historical prediction unavailable. {exc}")
    result = st.session_state.get("production_result")
    if result is not None:
        if (result["latitude"], result["longitude"], result["target_date"]) == (lat, lon, target_date):
            _dashboard_result(result, service, historical=True)
        else:
            st.session_state.pop("production_result", None)


def main() -> None:
    st.set_page_config(page_title="ForestSentinels | Wildfire map explorer", page_icon="🌲", layout="wide", initial_sidebar_state="collapsed")
    _demo_styles()
    with st.sidebar:
        st.markdown("### ForestSentinels")
        st.caption("WILDFIRE INTELLIGENCE · RESEARCH PROTOTYPE")
        st.divider()
        page = st.radio("Navigate", ["Live dashboard", "Historical Research", "About & Methodology", "Legacy demos"], label_visibility="collapsed")
        st.divider()
        st.caption("SUPPORTED REGION")
        st.write("Greece · FireCube-domain land cells")
        st.caption("This system is experimental. It does not issue public warnings or show calibrated event probabilities.")
    if page == "Live dashboard":
        production_section(historical=False)
        return
    if page == "Historical Research":
        st.title("Historical replay")
        st.caption("Research-only inference; these outputs are separate from the current-data prototype.")
        track_b_tab, firecube_tab = st.tabs(["Track-B historical", "FireCube Exp5 historical"])
        with track_b_tab:
            production_section(historical=True)
        with firecube_tab:
            firecube_section()
        return
    if page == "About & Methodology":
        st.markdown("<div class='section-kicker'>Methodology</div>", unsafe_allow_html=True)
        st.title("From place to model score")
        st.write(
            "A submitted place search resolves a coordinate. The existing Greece-domain feature pipeline "
            "then collects ten completed UTC days of available weather and environmental data, combines "
            "them with static geographic inputs, applies the saved 35-feature preprocessing, and evaluates "
            "the saved Track-B XGBoost model. Current weather is shown as context, not substituted for the ten-day window."
        )
        st.info("Place search: OpenStreetMap Nominatim (© OpenStreetMap contributors). Weather: Open-Meteo. Other EO and static source details appear with each prediction's provenance.")
        _research_snapshot()
        _limitations_scope()
        st.caption("The historical FireCube and production model paths are separate. Neither is a public alerting service.")
        return

    st.markdown("<div class='section-kicker'>Preserved demonstrations</div>", unsafe_allow_html=True)
    st.title("Legacy demonstrations")
    st.caption("These demonstrations remain available for continuity. Their scores are unrelated to the Track-B location score.")
    demo = st.selectbox("Choose a demo", ["Image MVP", "Weather MVP", "Combined MVP"])
    tab_model = load_tabular()
    img_model = load_image()
    if demo == "Image MVP":
        st.subheader("Image fire-detection MVP")
        if img_model is None:
            st.warning("Image model artifact is unavailable in this environment.")
        image = pick_image("image")
        if image is not None:
            st.image(image, caption="Input", width="stretch")
            if img_model is not None:
                result = img_model.predict_proba(image)
                risk_banner(result)
    elif demo == "Weather MVP":
        st.subheader("Weather-input MVP")
        if tab_model is None:
            st.warning("Weather model artifact is unavailable in this environment.")
        values = weather_form("weather")
        if st.button("Predict weather risk", type="primary"):
            if tab_model is None:
                st.error("Train the tabular model first.")
            else:
                st.session_state["weather_result"] = tab_model.predict_proba(values)
        if "weather_result" in st.session_state:
            risk_banner(st.session_state["weather_result"])
    else:
        st.subheader("Combined-image/weather MVP")
        st.write(
            "When both an image and weather values are available, combine "
            "P(fire) as `w · image + (1 − w) · weather`."
        )
        weight = st.slider("Image weight", 0.0, 1.0, 0.6, 0.05)
        image = pick_image("fusion")
        values = weather_form("fusion")
        if image is not None:
            st.image(image, caption="Input", width=280)
        if st.button("Run combined prediction", type="primary"):
            p_img = img_model.predict_proba(image)["p_fire"] if (img_model and image) else None
            p_tab = tab_model.predict_proba(values)["p_fire"] if tab_model else None
            if p_img is None and p_tab is None:
                st.error("Need a trained model and at least weather values or an image.")
            else:
                st.session_state["fusion_result"] = fuse_probabilities(
                    p_img, p_tab, image_weight=weight
                )
        if "fusion_result" in st.session_state:
            fused = st.session_state["fusion_result"]
            risk_banner(fused)
            c1, c2, c3 = st.columns(3)
            p_img = fused["p_image"]
            p_tab = fused["p_tabular"]
            c1.metric("Image P(fire)", "—" if p_img is None else f"{p_img:.1%}")
            c2.metric("Weather P(fire)", "—" if p_tab is None else f"{p_tab:.1%}")
            c3.metric("Fused P(fire)", f"{fused['p_fire']:.1%}")

if __name__ == "__main__":
    main()
