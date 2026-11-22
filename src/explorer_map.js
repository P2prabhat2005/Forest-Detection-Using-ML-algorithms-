const leafletVersion = "https://unpkg.com/leaflet@1.9.4/dist/";
let leafletReady;

function loadLeaflet() {
    if (window.L) return Promise.resolve(window.L);
    leafletReady ??= new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = leafletVersion + "leaflet.js";
        script.integrity = "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=";
        script.crossOrigin = "anonymous";
        script.onload = () => resolve(window.L);
        script.onerror = () => {
            leafletReady = undefined;
            script.remove();
            reject(new Error("Leaflet could not load"));
        };
        document.head.append(script);
    });
    return leafletReady;
}

export default function render({ parentElement, data, setTriggerValue }) {
    const node = parentElement.querySelector(".explorer-map");
    const feedback = parentElement.querySelector(".map-feedback");
    const zoomUnit = parentElement.querySelector(".zoom-model-unit");
    let alive = true;
    loadLeaflet().then(L => {
        if (!alive) return;
        let state = node.mapState;
        if (!state) {
            const map = L.map(node, { scrollWheelZoom: false, minZoom: 3, maxZoom: 18 });
            const street = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            }).addTo(map);
            const satellite = L.tileLayer(
                "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg",
                { maxNativeZoom: 8, attribution: '<a href="https://www.earthdata.nasa.gov/data/projects/gibs">NASA GIBS · Blue Marble</a>' },
            );
            L.control.layers({ "Street map": street, "Blue Marble imagery": satellite }, {}, {
                position: "topright", collapsed: true,
            }).addTo(map);
            const reset = parentElement.querySelector(".reset-map");
            reset.onclick = () => map.fitBounds(data.bounds, { padding: [16, 16] });
            reset.disabled = false;
            map.fitBounds(data.bounds, { padding: [16, 16] });
            const coverage = data.coverage ? L.imageOverlay(data.coverage, data.bounds, {interactive: false}) : null;
            const unitPane = map.createPane("selectedUnit");
            unitPane.style.zIndex = "410";
            unitPane.style.pointerEvents = "none";
            const modelUnit = L.layerGroup().addTo(map);
            const selection = L.layerGroup().addTo(map);
            const hotspots = L.layerGroup().addTo(map);
            const observer = new ResizeObserver(() => map.invalidateSize({pan: false}));
            observer.observe(node);
            state = node.mapState = { map, coverage, modelUnit, selection, hotspots, observer };
            map.on("click", event => state.pick?.(event));
        }
        feedback.hidden = true;
        const { map, coverage, modelUnit, selection, hotspots } = state;
        if (coverage) {
            if (data.showCoverage) coverage.addTo(map);
            else coverage.remove();
        }
        const unit = data.unit;
        if (unit?.id !== state.unitId) {
            state.unitId = unit?.id;
            modelUnit.clearLayers();
            if (unit) {
                L.imageOverlay(unit.image, unit.bounds, { pane: "selectedUnit", interactive: false }).addTo(modelUnit);
                L.rectangle(unit.bounds, {
                    pane: "selectedUnit", color: "#4b3e9d", fill: false, weight: 2, dashArray: "5 3", interactive: false,
                }).addTo(modelUnit);
            }
        }
        zoomUnit.hidden = !unit;
        zoomUnit.onclick = unit ? () => map.fitBounds(unit.bounds, { maxZoom: 13, padding: [30, 30] }) : null;
        const selectionKey = JSON.stringify(data.selection);
        if (selectionKey !== state.selectionKey) {
            state.selectionKey = selectionKey;
            selection.clearLayers();
            if (data.selection) {
                L.circleMarker(data.selection, {
                    radius: 8, color: "#fff", weight: 3, fillColor: "#006857",
                    fillOpacity: 1, interactive: false,
                    pane: "markerPane",
                }).addTo(selection);
                if (data.focusSelection) map.setView(data.selection, 8);
            }
        }
        const hotspotKey = JSON.stringify(data.hotspots);
        if (hotspotKey !== state.hotspotKey) {
            state.hotspotKey = hotspotKey;
            hotspots.clearLayers();
            for (const point of data.hotspots) {
                const details = document.createElement("span");
                details.textContent = "NASA FIRMS · NOAA-20 VIIRS · " + point.observed + " · " + point.confidence + " confidence";
                L.circleMarker([point.latitude, point.longitude], {
                    radius: 5, color: "#9a3400", weight: 2, fillColor: "#ff9a44", fillOpacity: 0.9,
                }).bindPopup(details).addTo(hotspots);
            }
        }
        state.pick = data.allowPick ? event => {
            setTriggerValue("pick", [
                Number(event.latlng.lat.toFixed(6)), Number(event.latlng.wrap().lng.toFixed(6)),
            ]);
        } : null;
    }).catch(() => {
        if (!alive) return;
        feedback.hidden = false;
        feedback.textContent = "The map could not load. You can still search by place or enter coordinates.";
    });
    return () => {
        alive = false;
        queueMicrotask(() => {
            if (!node.isConnected && node.mapState) {
                node.mapState.observer.disconnect();
                node.mapState.map.remove();
                delete node.mapState;
            }
        });
    };
}
