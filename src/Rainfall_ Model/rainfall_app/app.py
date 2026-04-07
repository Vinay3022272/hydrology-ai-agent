"""Standalone Streamlit app for Mahanadi rainfall prediction."""

from __future__ import annotations

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from basin_check import is_inside_basin
from config import (
    BASIN_SHP_PATH,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    MAP_CENTER,
    MAP_ZOOM,
)
from geocode import search_place
from predictor import predict_rainfall_for_basin, predict_rainfall_for_point

st.set_page_config(
    page_title="Mahanadi Rainfall Prediction", page_icon="🌧️", layout="wide"
)

st.markdown(
    """
<style>
.main-header { text-align:center; padding:1rem 0 0.5rem; }
.main-header h1 { color:#1f4e79; font-size:2rem; }
.main-header p { color:#5d6d7e; }
.metric-card { background:#f8f9fa; border:1px solid #dfe6e9; border-radius:10px; padding:0.8rem; text-align:center; }
.metric-card h3 { margin:0; color:#2d3436; font-size:0.95rem; }
.metric-card p { margin:0.2rem 0 0; font-weight:700; color:#2d3436; font-size:1.2rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
  <h1>🌧️ Rainfall Prediction</h1>
  <p>Mahanadi Basin · 3-day rainfall forecast using date, latitude, and longitude</p>
</div>
""",
    unsafe_allow_html=True,
)

for key in ("lat", "lon", "label", "result"):
    if key not in st.session_state:
        st.session_state[key] = None


def run_prediction(
    date_value,
    mode: str,
    lat: float | None = None,
    lon: float | None = None,
    label: str | None = None,
):
    st.session_state.lat = lat
    st.session_state.lon = lon
    st.session_state.label = label

    if mode == "basin":
        st.session_state.result = predict_rainfall_for_basin(str(date_value))
    else:
        if lat is None or lon is None or not is_inside_basin(lat, lon):
            st.session_state.result = {"status": "outside_basin"}
            return
        st.session_state.result = predict_rainfall_for_point(lat, lon, str(date_value))


def _render_grid_map(
    grid: list, day_label: str, lat: float | None = None, lon: float | None = None
):
    """Render a small rainfall heatmap for one predicted day."""
    arr = np.array(grid, dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(
        arr,
        cmap="YlGnBu",
        origin="lower",
        extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
    )
    if lat is not None and lon is not None:
        ax.scatter([lon], [lat], c="red", s=32, marker="x", label="Selected point")
    ax.set_title(day_label, fontsize=10)
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    if lat is not None and lon is not None:
        ax.legend(loc="upper right", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mm")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


st.sidebar.header("🌧️ Prediction Options")
method = st.sidebar.radio(
    "Mode",
    [
        "🛰️ Complete Basin Prediction",
        "🔍 Search by place name",
        "🗺️ Click on map",
    ],
    index=0,
)
target_date = st.sidebar.date_input(
    "Target start date", value=pd.Timestamp.today().date()
)

col_input, col_result = st.columns([3, 2])

if method == "🛰️ Complete Basin Prediction":
    with col_input:
        st.subheader("🛰️ Complete Basin Prediction")
        st.caption("Predict full 31x31 basin rainfall maps for 3 days.")
        if st.button("Predict Basin Rainfall", type="primary", width="stretch"):
            with st.spinner("Running prediction..."):
                run_prediction(target_date, mode="basin", label="Complete Basin")

elif method == "🔍 Search by place name":
    with col_input:
        st.subheader("🔍 Search Location")
        place_query = st.text_input("Enter a place name", placeholder="e.g. Sambalpur")

        if st.button("Search", type="primary", width="stretch"):
            if place_query.strip():
                with st.spinner("Searching..."):
                    st.session_state["candidates"] = search_place(place_query.strip())
            else:
                st.warning("Please enter a place name.")

        candidates = st.session_state.get("candidates", [])
        if candidates:
            names = [c["name"] for c in candidates]
            selected_idx = st.selectbox(
                "Select a location", range(len(names)), format_func=lambda i: names[i]
            )
            chosen = candidates[selected_idx]
            st.caption(f"📌 Lat: {chosen['lat']:.5f}, Lon: {chosen['lon']:.5f}")

            if st.button("Predict Rainfall", type="primary", width="stretch"):
                with st.spinner("Running prediction..."):
                    run_prediction(
                        target_date,
                        mode="point",
                        lat=chosen["lat"],
                        lon=chosen["lon"],
                        label=chosen["name"],
                    )

else:
    with col_input:
        st.subheader("🗺️ Click on the Map")
        m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="OpenStreetMap")

        try:
            basin_gdf = gpd.read_file(BASIN_SHP_PATH).to_crs(epsg=4326)
            folium.GeoJson(
                basin_gdf.to_json(),
                style_function=lambda _: {
                    "fillColor": "#5dade2",
                    "color": "#1f4e79",
                    "weight": 2,
                    "fillOpacity": 0.12,
                },
            ).add_to(m)
        except Exception:
            pass

        map_data = st_folium(m, width=700, height=500, key="rainfall_map")
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            st.info(f"📌 Clicked: Lat {click_lat:.5f}, Lon {click_lon:.5f}")
            if st.button("Predict Rainfall", type="primary", width="stretch"):
                with st.spinner("Running prediction..."):
                    run_prediction(
                        target_date,
                        mode="point",
                        lat=click_lat,
                        lon=click_lon,
                        label=f"Map click ({click_lat:.4f}, {click_lon:.4f})",
                    )

with col_result:
    st.subheader("📊 Prediction Result")
    result = st.session_state.result

    if result is None:
        st.info("Select a location and click Predict Rainfall.")
    elif result.get("status") == "outside_basin":
        st.error("Selected point is outside the Mahanadi Basin.")
    elif result.get("status") != "success":
        st.error(result.get("message", "Prediction failed."))
    else:
        if result.get("mode") == "basin":
            st.markdown("**📍 Scope:** Complete Basin")
        else:
            st.markdown(f"**📍 Location:** {st.session_state.label}")
            st.markdown(
                f"**🌐 Input:** {result['input']['lat']:.5f}, {result['input']['lon']:.5f}"
            )
            st.markdown(
                f"**🧭 Nearest Grid Cell:** {result['input']['grid_lat']:.5f}, {result['input']['grid_lon']:.5f}"
            )
        st.markdown(f"**🗓️ Mode:** {result['way']}")
        if result.get("mode") != "basin" and result["input"].get("local_area"):
            area = result["input"]["local_area"]
            st.markdown(
                f"**📦 Local Area Used:** {area['lat_min']:.2f}–{area['lat_max']:.2f}°N, "
                f"{area['lon_min']:.2f}–{area['lon_max']:.2f}°E ({area['n_pixels']} pixels)"
            )

        df = pd.DataFrame(result["days"])
        if result.get("mode") == "basin":
            show_cols = [
                c
                for c in [
                    "date",
                    "pred_basin_mean_mm",
                    "pred_basin_max_mm",
                    "actual_basin_mean_mm",
                    "day_mae_mm",
                ]
                if c in df.columns
            ]
        else:
            show_cols = [
                c
                for c in [
                    "date",
                    "max_coord",
                    "max_pred_mm",
                    "mean_pred_mm",
                    "actual_mm_at_point",
                    "day_mae_mm",
                ]
                if c in df.columns
            ]

        m1, m2 = st.columns(2)
        with m1:
            day1_val = (
                float(df.iloc[0]["pred_basin_max_mm"])
                if result.get("mode") == "basin"
                else float(df.iloc[0]["max_pred_mm"])
            )
            title1 = (
                "Day 1 Basin Max"
                if result.get("mode") == "basin"
                else "Day 1 Local Max"
            )
            st.markdown(
                f'<div class="metric-card"><h3>{title1}</h3><p>{day1_val:.2f} mm</p></div>',
                unsafe_allow_html=True,
            )
        with m2:
            avg_3d = (
                float(df["pred_basin_mean_mm"].mean())
                if result.get("mode") == "basin"
                else float(df["mean_pred_mm"].mean())
            )
            title2 = (
                "3-Day Avg Basin Mean"
                if result.get("mode") == "basin"
                else "3-Day Avg Local Mean"
            )
            st.markdown(
                f'<div class="metric-card"><h3>{title2}</h3><p>{avg_3d:.2f} mm</p></div>',
                unsafe_allow_html=True,
            )

        if result.get("overall_mae_mm") is not None:
            st.caption(f"Overall 3-day MAE: {result['overall_mae_mm']:.2f} mm")

        st.dataframe(df[show_cols], width="stretch", hide_index=True)

        grid_days = [d for d in result.get("days", []) if d.get("pred_grid_31x31")]
        if grid_days:
            st.markdown("### 🗺️ Predicted Rainfall Maps (31x31 Grid)")
            tabs = st.tabs(
                [d.get("date", f"Day {i+1}") for i, d in enumerate(grid_days)]
            )
            for tab, d in zip(tabs, grid_days):
                with tab:
                    _render_grid_map(
                        d["pred_grid_31x31"],
                        f"Predicted Rainfall - {d.get('date', '')}",
                        result["input"].get("lat"),
                        result["input"].get("lon"),
                    )
