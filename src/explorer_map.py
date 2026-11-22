"""Streamlit bridge for the interactive map."""

from pathlib import Path

import streamlit as st


def explorer_map(*, data: dict):
    return _map_component()(data=data, key="forestsentinels_map", on_pick_change=lambda: None)


def _map_component():
    return st.components.v2.component(
        "forestsentinels_explorer",
        html="""
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="anonymous">
        <div class="map-shell">
          <div class="explorer-map" aria-label="Interactive map of Greece"></div>
          <div class="map-navigation">
            <button class="reset-map" type="button" disabled>Reset to Greece</button>
            <button class="zoom-model-unit" type="button" hidden>Zoom to model unit</button>
          </div>
          <p class="map-feedback" role="status">Loading the interactive map…</p>
        </div>
        """,
        css="""
        .map-shell {position:relative; isolation:isolate; border:1px solid #b8cfc4; border-radius:14px; overflow:hidden}
        .explorer-map {height:clamp(300px,60vh,620px); background:#e0ece8; cursor:crosshair}
        .map-navigation {position:absolute; bottom:25px; left:10px; display:flex; flex-wrap:wrap; gap:.3rem; z-index:900}
        .map-navigation button {padding:.4rem .65rem;
          border:1px solid #56706a; border-radius:7px; background:white; color:#133a32; cursor:pointer; font-weight:650}
        .map-feedback {position:absolute; top:50%; left:50%; margin:0; transform:translate(-50%,-50%);
          width:max-content; max-width:85%; z-index:900; background:white; color:#133a32; padding:.65rem; border-radius:7px}
        @media(max-width:700px){.explorer-map{height:clamp(290px,48vh,500px)}}
        """,
        js=Path(__file__).with_suffix(".js").read_text(encoding="utf-8"),
    )
