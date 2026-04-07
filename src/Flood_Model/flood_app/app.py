"""
Flood Susceptibility Prediction App — Mahanadi Basin
=====================================================
Streamlit application with place search and interactive map click.
"""

import streamlit as st
import folium
import geopandas as gpd
import pandas as pd
from streamlit_folium import st_folium

from config import MAP_CENTER, MAP_ZOOM, BASIN_SHP_PATH, FEATURE_ORDER
from geocode import search_place
from basin_check import is_inside_basin
from raster_extract import extract_features
from preprocess import preprocess_features
from predict import predict


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mahanadi Flood Susceptibility",
    page_icon="🌊",
    layout="wide",
)


# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .main-header h1 {
        color: #1a5276;
        font-size: 2rem;
    }
    .main-header p {
        color: #7f8c8d;
        font-size: 0.95rem;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }
    .risk-low    { background: #27ae60; color: white; }
    .risk-mod    { background: #f39c12; color: white; }
    .risk-high   { background: #e74c3c; color: white; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .metric-card h3 { color: #2c3e50; margin: 0 0 0.3rem; }
    .metric-card p  { font-size: 1.5rem; font-weight: 700; margin: 0; color: #2c3e50; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="main-header">
    <h1>🌊 Flood Susceptibility Prediction</h1>
    <p>Mahanadi Basin — Location-based risk assessment using geospatial raster features and XGBoost</p>
</div>
""",
    unsafe_allow_html=True,
)

st.info(
    "ℹ️ This system supports **only locations within the Mahanadi Basin**. "
    "Search a place name or click on the map to get started."
)


# ── Session state setup ────────────────────────────────────────────────────
for key in ("lat", "lon", "label", "result", "features"):
    if key not in st.session_state:
        st.session_state[key] = None


# ── Helper: run pipeline ───────────────────────────────────────────────────
def run_prediction(lat: float, lon: float, label: str):
    """Execute the full prediction pipeline and store results in session."""
    st.session_state.lat = lat
    st.session_state.lon = lon
    st.session_state.label = label
    st.session_state.result = None
    st.session_state.features = None

    # 1. Basin check
    if not is_inside_basin(lat, lon):
        st.session_state.result = {"status": "outside_basin"}
        return

    # 2. Feature extraction
    raw_features = extract_features(lat, lon)
    if raw_features is None:
        st.session_state.result = {"status": "missing_data"}
        return

    # 3. Preprocess
    try:
        scaled = preprocess_features(raw_features)
    except Exception:
        st.session_state.result = {"status": "preprocess_error"}
        return

    # 4. Predict
    pred = predict(scaled)

    st.session_state.features = raw_features
    st.session_state.result = {
        "status": "success",
        "probability": pred["probability"],
        "risk_class": pred["risk_class"],
    }


# ── Sidebar: input method ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📍 Select Location")
    method = st.radio(
        "Input method", ["🔍 Search by place name", "🗺️ Click on map"], index=0
    )

# ── Layout columns ─────────────────────────────────────────────────────────
col_input, col_result = st.columns([3, 2])


# ═══════════════════════════════════════════════════════════════════════════
# Method A: Search by place name
# ═══════════════════════════════════════════════════════════════════════════
if method == "🔍 Search by place name":
    with col_input:
        st.subheader("🔍 Search Location")
        place_query = st.text_input("Enter a place name", placeholder="e.g. Cuttack")

        if st.button("Search", type="primary", width="stretch"):
            if place_query.strip():
                with st.spinner("Searching…"):
                    candidates = search_place(place_query.strip())
                    st.session_state["candidates"] = candidates
            else:
                st.warning("Please enter a place name.")

        # Show candidate dropdown
        candidates = st.session_state.get("candidates", [])
        if candidates:
            names = [c["name"] for c in candidates]
            selected_idx = st.selectbox(
                "Select a location", range(len(names)), format_func=lambda i: names[i]
            )

            chosen = candidates[selected_idx]
            st.caption(f"📌 Lat: `{chosen['lat']:.5f}`, Lon: `{chosen['lon']:.5f}`")

            if st.button(
                "Predict Flood Susceptibility", type="primary", width="stretch"
            ):
                with st.spinner("Extracting features & predicting…"):
                    run_prediction(chosen["lat"], chosen["lon"], chosen["name"])
        elif (
            "candidates" in st.session_state
            and st.session_state["candidates"] is not None
        ):
            if len(st.session_state.get("candidates", [])) == 0 and place_query.strip():
                st.warning("No matching location found. Try a different name.")


# ═══════════════════════════════════════════════════════════════════════════
# Method B: Click on map
# ═══════════════════════════════════════════════════════════════════════════
else:
    with col_input:
        st.subheader("🗺️ Click on the Map")
        st.caption("Click any point inside the basin boundary to predict flood risk.")

        # Build Folium map
        m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="OpenStreetMap")

        # Overlay basin boundary
        try:
            basin_gdf = gpd.read_file(BASIN_SHP_PATH).to_crs(epsg=4326)
            folium.GeoJson(
                basin_gdf.to_json(),
                name="Mahanadi Basin",
                style_function=lambda _: {
                    "fillColor": "#3498db",
                    "color": "#1a5276",
                    "weight": 2,
                    "fillOpacity": 0.12,
                },
            ).add_to(m)
        except Exception:
            pass

        # If we already have a selected point, add a marker
        if st.session_state.lat is not None:
            folium.Marker(
                [st.session_state.lat, st.session_state.lon],
                popup=st.session_state.label or "Selected point",
                icon=folium.Icon(color="red", icon="info-sign"),
            ).add_to(m)

        map_data = st_folium(m, width=700, height=480, key="map")

        # Capture click
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            st.info(f"📌 Clicked: Lat `{click_lat:.5f}`, Lon `{click_lon:.5f}`")

            if st.button(
                "Predict Flood Susceptibility", type="primary", width="stretch"
            ):
                with st.spinner("Extracting features & predicting…"):
                    run_prediction(
                        click_lat,
                        click_lon,
                        f"Map click ({click_lat:.4f}, {click_lon:.4f})",
                    )


# ═══════════════════════════════════════════════════════════════════════════
# Result display
# ═══════════════════════════════════════════════════════════════════════════
with col_result:
    st.subheader("📊 Prediction Result")

    result = st.session_state.result
    if result is None:
        st.markdown(
            "<p style='color:#95a5a6; text-align:center; padding:3rem 0;'>"
            "Select a location and click <b>Predict</b> to see results.</p>",
            unsafe_allow_html=True,
        )
    elif result["status"] == "outside_basin":
        st.error(
            "🚫 **Selected point is outside the Mahanadi Basin.**\n\n"
            "Prediction is only available for locations within the basin."
        )
    elif result["status"] == "missing_data":
        st.warning(
            "⚠️ **Required feature data is unavailable for this point.**\n\n"
            "The location may be at the edge of the raster coverage."
        )
    elif result["status"] == "preprocess_error":
        st.error(
            "❌ Coordinate transformation or preprocessing failed. Please verify input data."
        )
    elif result["status"] == "success":
        # Location info
        st.markdown(f"**📍 Location:** {st.session_state.label}")
        st.markdown(
            f"**🌐 Coordinates:** `{st.session_state.lat:.5f}°N`, `{st.session_state.lon:.5f}°E`"
        )

        st.divider()

        # Probability & Risk
        prob = result["probability"]
        risk = result["risk_class"]
        css_class = {"Low": "risk-low", "Moderate": "risk-mod", "High": "risk-high"}[
            risk
        ]

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f'<div class="metric-card"><h3>Probability</h3>'
                f"<p>{prob:.2%}</p></div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><h3>Risk Class</h3>'
                f'<p><span class="risk-badge {css_class}">{risk}</span></p></div>',
                unsafe_allow_html=True,
            )

        # Progress bar
        st.progress(prob)

        st.divider()

        # Feature table
        if st.session_state.features:
            st.markdown("**📋 Extracted Features**")
            feat_df = pd.DataFrame(
                [
                    {
                        "Feature": k,
                        "Value": f"{v:.4f}" if isinstance(v, float) else str(v),
                    }
                    for k, v in st.session_state.features.items()
                ]
            )
            st.dataframe(feat_df, width="stretch", hide_index=True)


# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Mahanadi Basin Flood Susceptibility Prediction · "
    "Geospatial raster feature extraction + XGBoost ML model"
)
