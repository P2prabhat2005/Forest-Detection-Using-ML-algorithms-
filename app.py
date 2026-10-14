"""Fire Early Warning Streamlit dashboard and preserved research demos."""

from __future__ import annotations

import json
import sys
from html import escape
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from PIL import Image

from src.firecube_inference import FireCubeExp5Inference, FireCubeInferenceError
from src.firecube_live import HistoricalCubeDynamicProvider
from src.production_features import ProductionFeaturePipeline, ProductionPreprocessor
from src.production_model import ProductionPredictionService, XGBoostProductionModel
from src.place_search import NominatimSearch, PlaceSearchError
from src.predict import ImagePredictor, TabularPredictor, default_weather, fuse_probabilities

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
    if historical:
        pipeline = ProductionFeaturePipeline(
            dynamic_provider=HistoricalCubeDynamicProvider(),
            preprocessor=preprocessor,
        )
    else:
        pipeline = ProductionFeaturePipeline(preprocessor=preprocessor)
    return ProductionPredictionService(pipeline, model)


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
    if run_demo:
        try:
            st.session_state["firecube_result"] = predictor.predict_historical(
                latitude, longitude, target_date, calibration="exp7_flat"
            )
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
        :root { --pine:#123e38; --ink:#18332f; --muted:#617770; --line:#dfe9e4; --paper:#fff; --mist:#f3f7f5; --amber:#dc9452; }
        .stApp { background: var(--mist); color: var(--ink); }
        .block-container { max-width: 1320px; padding-top: 1.45rem; padding-bottom: 4rem; }
        section[data-testid="stSidebar"] { background: #102e2b; }
        section[data-testid="stSidebar"] * { color: #eef7f1 !important; }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius: 10px; }
        h1, h2, h3 { letter-spacing: -.03em; }
        .eyebrow, .section-kicker { font-size: .72rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; }
        .section-kicker { color: #39735c; margin-bottom: .2rem; }
        .demo-hero { position:relative; overflow:hidden; background: linear-gradient(118deg,#113a34 0%,#20594a 68%,#426a50 100%); border-radius: 22px; color: #fff; padding: 2.15rem 2.35rem; margin-bottom: 1.35rem; box-shadow:0 14px 30px rgba(23,57,48,.13); animation: rise .5s ease-out both; }
        .demo-hero::after { content:""; position:absolute; width:250px; height:250px; right:-45px; top:-105px; border:1px solid rgba(255,255,255,.16); border-radius:50%; box-shadow:0 0 0 48px rgba(255,255,255,.045),0 0 0 98px rgba(255,255,255,.025); pointer-events:none; }
        .demo-hero .eyebrow { color:#efc18c; }
        .demo-hero h1 { color:#fff; font-size:clamp(2rem,4vw,3.25rem); line-height:1.1; margin:.45rem 0 .65rem; }
        .demo-hero p { max-width:720px; margin:0; color:#e5f2ea; font-size:1.02rem; }
        .hero-meta { margin-top:1.25rem; color:#bfe1d0; font-size:.83rem; }
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
        div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding:.85rem 1rem; box-shadow:0 3px 12px rgba(24,67,53,.03); }
        div[data-testid="stMetricLabel"] { color:#526a60; }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(>div[data-testid="stVerticalBlock"]) { border-color:var(--line); }
        .stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] { background:#225e49; border-color:#225e49; border-radius:9px; font-weight:700; }
        .stButton>button[kind="primary"]:hover, .stFormSubmitButton>button[kind="primary"]:hover { background:#194b3b; border-color:#194b3b; }
        @keyframes rise { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
        @media (max-width:700px) { .block-container { padding:1rem .8rem 3rem; } .demo-hero { padding:1.5rem; } .score-card { min-height:0; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _production_risk_status() -> None:
    st.info(
        "**Interpretation:** this is an XGBoost model score, shown on a 0–100 scale. "
        "It is not a calibrated real-world fire probability and no operational alert threshold is assigned."
    )


def _production_data_status(result: dict) -> None:
    provenance = result.get("feature_provenance", {})
    approximate = sorted({
        name for name, info in provenance.items() if info.get("approximate")
    })
    missing = result.get("missing_features", [])
    if missing:
        st.warning(
            f"Missing source values were imputed for {len(missing)} model features. "
            "This is a prototype result, not an operational warning."
        )
    if approximate:
        st.warning(
            "Approximate/proxy inputs are present: "
            + ", ".join(approximate[:8])
            + (" …" if len(approximate) > 8 else "")
        )
    if not missing and not approximate:
        st.success("Requested inputs were available without approximation flags for this run.")


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
    columns = st.columns(len(available))
    for column, (key, label) in zip(columns, available):
        value = values[key]
        unit = units.get(key, "")
        rendered = "—" if value is None else f"{value:g} {unit}".strip()
        column.metric(label, rendered)
    st.caption(
        f"Current modelled weather at {current.get('timestamp', 'unknown time')} UTC · "
        f"retrieved {current.get('retrieved_at', 'during this request')}. "
        "This display is contextual; the model uses the preceding ten completed UTC days."
    )


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
            "Quality": "Missing / imputed" if name in missing else ("Approximate" if info.get("approximate") else "Available"),
        })
    return pd.DataFrame(rows)


def _dashboard_result(result: dict, service: ProductionPredictionService, historical: bool) -> None:
    st.markdown("---")
    st.markdown("<div class='section-kicker'>Prediction summary</div>", unsafe_allow_html=True)
    st.subheader("Experimental model output")
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        score = result["raw_probability"] * 100
        st.markdown(
            f"<div class='score-card'><div class='eyebrow'>TRACK-B · XGBOOST BASELINE</div>"
            f"<div class='score-number'>{score:.1f}<span class='score-unit'> / 100</span></div>"
            "<div class='score-note'>Experimental model score · not a calibrated real-world fire probability</div></div>",
            unsafe_allow_html=True,
        )
        st.caption("No risk category, operational accuracy claim, or alert threshold is assigned.")
        st.markdown(
            f"<div class='location-card'><b>{escape(result.get('resolved_place', 'Manual coordinates'))}</b><br>"
            f"{result['latitude']:.4f}° N, {result['longitude']:.4f}° E · target {escape(str(result['target_date']))}<br>"
            "<small>Input window: ten preceding completed UTC days</small></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.map(pd.DataFrame({"latitude": [result["latitude"]], "longitude": [result["longitude"]]}), zoom=8, height=300, width="stretch")
        st.caption("Selected coordinate · map marker is not a forecast footprint or active-fire detection.")

    st.markdown("<div class='section-kicker'>Conditions & data quality</div>", unsafe_allow_html=True)
    if historical:
        st.info("Historical Track-B inference uses local FireCube history. It does not fetch current weather.")
    else:
        current = result.get("current_weather", {})
        st.markdown("<div class='panel-title'>Current weather context</div>", unsafe_allow_html=True)
        if current.get("available"):
            _weather_cards(current)
        else:
            st.warning("Current weather unavailable: " + str(current.get("message", "No data")) + ". The model used available recent history.")
        st.caption(f"Request time: {result.get('requested_at', 'not recorded')} · Weather provider: Open-Meteo")
    q1, q2, q3 = st.columns(3)
    feature_count = len(result.get("feature_values", {}))
    missing_count = len(result.get("missing_features", []))
    approx_count = sum(bool(x.get("approximate")) for x in result.get("feature_provenance", {}).values())
    q1.metric("Model inputs", f"{feature_count} / 35")
    q2.metric("Missing / imputed", str(missing_count))
    q3.metric("Approximate inputs", str(approx_count))
    _production_data_status(result)
    feature_table = _feature_rows(result)
    for group in ("Weather", "Vegetation / EO", "Soil moisture", "Terrain / land cover"):
        subset = feature_table[feature_table["Group"] == group]
        with st.expander(f"{group} · {len(subset)} features", expanded=(group == "Weather")):
            st.dataframe(subset.drop(columns="Group"), hide_index=True, width="stretch")

    st.markdown("<div class='section-kicker'>Model interpretation</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>What the model uses most overall</div>", unsafe_allow_html=True)
    importances = getattr(service.model.estimator, "feature_importances_", None)
    if importances is not None:
        rows = []
        for index in sorted(range(len(importances)), key=lambda i: float(importances[i]), reverse=True)[:8]:
            name = result["feature_names"][index]
            rows.append({"Feature": name, "Global importance": float(importances[index]), "This input": result["feature_values"].get(name)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Global XGBoost feature importance is not a causal or per-location explanation.")
    else:
        st.info("This model does not expose feature importances.")
    with st.expander("How this prediction was made"):
        st.write(
            "Location → 10 completed days of weather and available EO/soil inputs → Greece-domain static context "
            "→ saved 35-feature preprocessing → Track-B XGBoost score. Current weather is displayed separately; "
            "it does not replace the historical input window. Source dates, missingness and proxies appear above."
        )


def production_section(historical: bool = False) -> None:
    """Track-B dashboard; unchanged production service handles all prediction logic."""
    if historical:
        st.markdown("<div class='section-kicker'>Historical research</div>", unsafe_allow_html=True)
        st.subheader("Revisit a FireCube-era date")
        st.info("Local historical inference only · Greece FireCube domain · example: 38.0000°N, 23.7000°E on 2020-07-15.")
    else:
        st.markdown("<div class='section-kicker'>01 / Choose a location</div>", unsafe_allow_html=True)
        st.subheader("Explore a location in Greece")
        st.caption("Current prototype coverage: FireCube Greece-domain land cells only. Place search can resolve locations elsewhere, but prediction is unavailable there.")

    selected_place = None
    if not historical:
        with st.form("production_place_search"):
            search_col, submit_col = st.columns([4, 1])
            with search_col:
                query = st.text_input("Search a place", placeholder="Try Athens, Greece", label_visibility="collapsed")
            with submit_col:
                submitted = st.form_submit_button("Search place", type="primary", use_container_width=True)
        st.caption("Search only runs when submitted · OpenStreetMap Nominatim · © OpenStreetMap contributors")
        if submitted:
            st.session_state.pop("production_result", None)
            try:
                with st.spinner("Finding matching places…"):
                    st.session_state["production_places"] = load_place_search().search(query)
                if not st.session_state["production_places"]:
                    st.info("No place found. Try a city plus country, or enter coordinates below.")
            except PlaceSearchError as exc:
                st.session_state["production_places"] = []
                st.error(f"Place search unavailable: {exc}")
        places = st.session_state.get("production_places", [])
        if places:
            index = st.selectbox("Matching places", range(len(places)), format_func=lambda i: places[i]["name"], key="production_place_choice")
            if st.checkbox("Use this place", value=True, key="production_use_place"):
                selected_place = places[index]
                st.markdown(
                    f"<div class='location-card'><b>{escape(selected_place['name'])}</b><br>"
                    f"<small>{selected_place['latitude']:.5f}, {selected_place['longitude']:.5f}</small></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<div class='section-kicker'>02 / Confirm & run</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1.3])
    with col1:
        latitude = st.number_input("Latitude", min_value=34.0, max_value=43.0, value=38.0, step=0.01, format="%.4f", key="production_latitude", disabled=selected_place is not None)
    with col2:
        longitude = st.number_input("Longitude", min_value=18.0, max_value=29.5, value=23.7, step=0.01, format="%.4f", key="production_longitude", disabled=selected_place is not None)
    with col3:
        if historical:
            target_date = st.date_input("Historical target date", value=date(2020, 7, 15), key="production_historical_date")
        else:
            st.markdown("<div class='mini-card'><strong>Prediction window</strong><span>10 completed UTC days</span></div>", unsafe_allow_html=True)
    if selected_place:
        latitude, longitude = selected_place["latitude"], selected_place["longitude"]
        st.caption(f"Using resolved coordinates {latitude:.5f}, {longitude:.5f}.")
    supported_box = 34.0 <= latitude <= 43.0 and 18.0 <= longitude <= 29.5
    if not supported_box:
        st.warning("This place is outside the supported Greece bounding area. Choose a Greece location or enter supported coordinates.")
    elif not historical:
        st.caption("Land-cell eligibility is checked by the existing inference pipeline when you run the prediction.")
    service = load_production_service(historical)
    if service is None:
        st.error("The saved production model or preprocessor is unavailable. The dashboard cannot run predictions here.")
        return
    label = "Run historical prediction" if historical else "Get experimental score"
    if st.button(label, type="primary", key="production_predict_historical" if historical else "production_predict_live", disabled=not supported_box):
        try:
            with st.spinner("Collecting available inputs and evaluating the saved XGBoost model…"):
                result = (service.predict(float(latitude), float(longitude), target_date.isoformat()) if historical
                          else service.predict_recent(float(latitude), float(longitude)))
            result["resolved_place"] = selected_place["name"] if selected_place else "Manual coordinates"
            st.session_state["production_result"] = result
            st.session_state["production_result_mode"] = "historical" if historical else "live"
        except (FireCubeInferenceError, ValueError, OSError, RuntimeError) as exc:
            st.session_state.pop("production_result", None)
            st.error(f"Prediction unavailable for this request: {exc}")
            st.info("Check the location, provider availability, and data coverage, then try again. No score is shown when inference fails.")
    result = st.session_state.get("production_result")
    if result is not None and (
        abs(result.get("latitude", float("nan")) - latitude) > 1e-7
        or abs(result.get("longitude", float("nan")) - longitude) > 1e-7
        or (historical and result.get("target_date") != target_date.isoformat())
        or not supported_box
    ):
        result = None  # Inputs changed; never display a score for a different selection.
    if result is None or st.session_state.get("production_result_mode") != ("historical" if historical else "live"):
        st.info("Select a supported location and run a prediction to see the model score, map and data-quality details.")
        if selected_place and supported_box:
            st.map(pd.DataFrame({"latitude": [latitude], "longitude": [longitude]}), zoom=8, height=280, width="stretch")
        return
    _dashboard_result(result, service, historical)


def main() -> None:
    st.set_page_config(page_title="Fire Early Warning", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")
    _demo_styles()
    with st.sidebar:
        st.markdown("### 🔥 Fire Early Warning")
        st.caption("WILDFIRE INTELLIGENCE · RESEARCH PROTOTYPE")
        st.divider()
        page = st.radio("Navigate", ["Live dashboard", "Historical Research", "About & Methodology", "Legacy demos"], label_visibility="collapsed")
        st.divider()
        st.caption("SUPPORTED REGION")
        st.write("Greece · FireCube-domain land cells")
        st.caption("This system is experimental. It does not issue public warnings or show calibrated event probabilities.")
    st.markdown(
        """
        <div class="demo-hero">
          <div class="eyebrow">LOCATION-LED WILDFIRE RESEARCH · FACULTY DEMONSTRATION</div>
          <h1>Fire Early Warning</h1>
          <p>Explore environmental conditions, a saved experimental model score, and the quality of the data behind it.</p>
          <div class="hero-meta">GREECE FIRST &nbsp;·&nbsp; WEATHER + ENVIRONMENTAL HISTORY &nbsp;·&nbsp; XGBOOST BASELINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if page == "Live dashboard":
        production_section(historical=False)
        return
    if page == "Historical Research":
        st.caption("Research-only inference; these outputs are separate from the current-data prototype.")
        track_b_tab, firecube_tab = st.tabs(["Track-B historical", "FireCube Exp5 historical"])
        with track_b_tab:
            production_section(historical=True)
        with firecube_tab:
            firecube_section()
        return
    if page == "About & Methodology":
        st.markdown("<div class='section-kicker'>Methodology</div>", unsafe_allow_html=True)
        st.subheader("From place to model score")
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
    st.subheader("Legacy MVP demos")
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
