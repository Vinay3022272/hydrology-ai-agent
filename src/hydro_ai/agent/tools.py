"""
All tools the agent can invoke, grouped by category:

  1. Trained Model Tools (Flood + Rainfall)
  2. Analysis Tools
  3. Retrieval Tool
  4. Optional Combiner Tool
"""

from __future__ import annotations

import math
import sys
import os
from typing import Any, Dict, List


from langchain_core.tools import tool


def _sanitize_floats(obj):
    """Recursively make a dict JSON-safe: convert numpy types, replace NaN/Inf with None."""
    # Handle numpy scalars first (float64, int64, bool_, etc.)
    try:
        import numpy as np

        if isinstance(obj, (np.floating, np.complexfloating)):
            obj = float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return _sanitize_floats(obj.tolist())
    except ImportError:
        pass

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    return obj


# Ensure the flood_app package is importable
_FLOOD_APP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Flood_Model", "flood_app")
)
if _FLOOD_APP_DIR not in sys.path:
    sys.path.insert(0, _FLOOD_APP_DIR)


# ══════════════════════════════════════════════════════════════════════
# 1. TRAINED MODEL TOOLS
# ══════════════════════════════════════════════════════════════════════

# ── Prediction Cache (in-memory, keyed by rounded lat/lon) ──
_prediction_cache: Dict[tuple, Dict[str, Any]] = {}
_geocode_cache: Dict[str, Dict[str, Any]] = {}
_rainfall_prediction_cache: Dict[tuple, Dict[str, Any]] = {}

# Canonical coordinates for known ambiguous place names.
_PLACE_COORD_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "sambalpur": {
        "name": "Sambalpur, Odisha, India",
        "lat": 21.47064,
        "lon": 83.98369,
    }
}


def _build_flood_report(
    location_label: str, pred: Dict[str, Any], features: Dict[str, Any]
) -> str:
    """Generate a concise but detailed markdown report for flood prediction."""
    prob = float(pred.get("probability", 0.0))
    risk = str(pred.get("risk_class", "Unknown"))
    threshold = float(pred.get("threshold", 0.5))
    decision = (
        "Flood-prone" if int(pred.get("prediction", 0)) == 1 else "Less flood-prone"
    )

    focus_features = [
        "distance_to_river",
        "flow_accumulation",
        "twi",
        "slope",
        "rainfall",
        "drainage_density",
    ]

    detail_lines = []
    for key in focus_features:
        if key in features:
            val = features[key]
            if isinstance(val, (int, float)):
                detail_lines.append(f"- {key}: {float(val):.4f}")
            else:
                detail_lines.append(f"- {key}: {val}")

    return (
        f"### Flood Susceptibility Report\n"
        f"- Location: {location_label}\n"
        f"- Flood Probability: {prob:.2%}\n"
        f"- Risk Class: {risk}\n"
        f"- Decision Threshold: {threshold:.3f}\n"
        f"- Model Decision: {decision}\n\n"
        "#### Key Feature Snapshot\n" + "\n".join(detail_lines) + "\n\n"
        "Interpretation: This prediction is computed from local terrain, hydrology, and rainfall-related geospatial features."
    )


@tool
def predict_flood_susceptibility(
    place_name: str = "",
    lat: float = 0.0,
    lon: float = 0.0,
) -> Dict[str, Any]:
    """
    Predict flood susceptibility for a location using the trained LightGBM model.

    Provide EITHER a place_name OR (lat, lon) coordinates.
    If place_name is given, it will be geocoded to coordinates automatically.

    Parameters
    ----------
    place_name : str
        Name of a place (e.g. "Cuttack", "Sambalpur", "Raipur").
        The tool will geocode it to lat/lon automatically.
    lat : float
        Latitude of the location (WGS84). Used only if place_name is empty.
    lon : float
        Longitude of the location (WGS84). Used only if place_name is empty.

    Returns
    -------
    dict
        Keys: status, probability, risk_class, features, location, report.
        status is one of: success, outside_basin, missing_data, geocode_failed, error.
        The 'report' key contains a pre-formatted markdown report — include it
        VERBATIM in your response.
    """
    try:
        from basin_check import is_inside_basin
        from raster_extract import extract_features
        from preprocess import preprocess_features
        from predict import predict
        from geocode import search_place

        # --- Geocoding step ---
        location_label = f"({lat}, {lon})"
        if place_name:
            place_key = place_name.strip().lower()
            if place_key in _PLACE_COORD_OVERRIDES:
                best = _PLACE_COORD_OVERRIDES[place_key]
                _geocode_cache[place_key] = best
                lat = float(best["lat"])
                lon = float(best["lon"])
                location_label = f"{best['name']} ({lat:.4f}, {lon:.4f})"
            else:
                # Always re-resolve place names so stale geocode cache cannot force old coordinates.
                candidates = search_place(place_name, max_results=5)
                if not candidates:
                    return {
                        "status": "geocode_failed",
                        "tool": "predict_flood_susceptibility",
                        "message": f"Could not find coordinates for '{place_name}'. Try a different name.",
                    }
                best = candidates[0]
                _geocode_cache[place_key] = best

                lat = best["lat"]
                lon = best["lon"]
                location_label = f"{best['name']} ({lat:.4f}, {lon:.4f})"

        if lat == 0.0 and lon == 0.0:
            return {
                "status": "error",
                "tool": "predict_flood_susceptibility",
                "message": "Please provide either a place_name or lat/lon coordinates.",
            }

        # --- Cache check (rounded to 3 decimals ≈ 110m resolution) ---
        cache_key = (round(lat, 3), round(lon, 3))
        if cache_key in _prediction_cache:
            cached = _prediction_cache[cache_key]
            cached["_cached"] = True
            return cached

        # 1. Basin check
        if not is_inside_basin(lat, lon):
            return {
                "status": "outside_basin",
                "tool": "predict_flood_susceptibility",
                "location": location_label,
                "message": (
                    f"{location_label} is outside the Mahanadi Basin. "
                    "Flood predictions are only available within the basin."
                ),
            }

        # 2. Feature extraction
        raw_features = extract_features(lat, lon)
        if raw_features is None:
            return {
                "status": "missing_data",
                "tool": "predict_flood_susceptibility",
                "location": location_label,
                "message": f"Could not extract terrain features for {location_label}.",
            }

        # 3. Preprocess
        scaled = preprocess_features(raw_features)

        # 4. Predict
        pred = predict(scaled)

        # 5. Build compact result (no verbose report — saves tokens)
        # Only include key features the LLM needs for a concise answer
        key_features = {
            k: v for k, v in raw_features.items() if k not in ("easting", "northing")
        }

        result = _sanitize_floats(
            {
                "status": "success",
                "location": location_label,
                "probability": pred["probability"],
                "risk_class": pred["risk_class"],
                "prediction": pred["prediction"],
                "threshold": pred["threshold"],
                "features": key_features,
                "report": _build_flood_report(location_label, pred, key_features),
            }
        )

        # Store in cache for future lookups
        _prediction_cache[cache_key] = result
        return result

    except Exception as e:
        return {
            "status": "error",
            "tool": "predict_flood_susceptibility",
            "message": f"Flood prediction failed: {e}",
        }


@tool
def predict_rainfall(
    place_name: str = "",
    lat: float = 0.0,
    lon: float = 0.0,
    date: str = "",
    mode: str = "point",
) -> Dict[str, Any]:
    """
    Predict rainfall for a location/date or complete basin.

    Parameters
    ----------
    place_name : str
        Optional place name to geocode (e.g., "Cuttack").
    lat : float
        Latitude of the location (WGS84). Used if place_name is not provided.
    lon : float
        Longitude of the location (WGS84). Used if place_name is not provided.
    date : str
        Optional target date in YYYY-MM-DD format.
    mode : str
        "point" for location-based rainfall, "basin" for complete basin prediction.

    Returns
    -------
    dict
        Rainfall prediction result.
    """
    try:
        from geocode import search_place
        from services.rainfall_service import get_rainfall_prediction

        mode = (mode or "point").strip().lower()
        if mode not in ("point", "basin"):
            return {
                "status": "error",
                "tool": "predict_rainfall",
                "message": "mode must be either 'point' or 'basin'.",
            }

        location_label = f"({lat}, {lon})"
        if mode == "point" and place_name:
            place_key = place_name.strip().lower()
            if place_key in _PLACE_COORD_OVERRIDES:
                best = _PLACE_COORD_OVERRIDES[place_key]
                _geocode_cache[place_key] = best
                lat = float(best["lat"])
                lon = float(best["lon"])
                location_label = f"{best['name']} ({lat:.4f}, {lon:.4f})"
            else:
                candidates = search_place(place_name, max_results=5)
                if not candidates:
                    return {
                        "status": "geocode_failed",
                        "tool": "predict_rainfall",
                        "message": f"Could not find coordinates for '{place_name}'.",
                    }
                best = candidates[0]
                _geocode_cache[place_key] = best

                lat = float(best["lat"])
                lon = float(best["lon"])
                location_label = f"{best['name']} ({lat:.4f}, {lon:.4f})"

        if mode == "point" and lat == 0.0 and lon == 0.0:
            return {
                "status": "error",
                "tool": "predict_rainfall",
                "message": "Provide either place_name or lat/lon for point mode.",
            }

        cache_key = (
            mode,
            str(date or ""),
            round(float(lat), 3) if mode == "point" else None,
            round(float(lon), 3) if mode == "point" else None,
        )
        if cache_key in _rainfall_prediction_cache:
            cached = dict(_rainfall_prediction_cache[cache_key])
            cached["_cached"] = True
            return cached

        payload = {"mode": mode, "date": date or None}
        if mode == "point":
            payload.update({"lat": float(lat), "lon": float(lon)})

        result = get_rainfall_prediction(payload)
        result = _sanitize_floats(result)

        if result.get("status") == "success":
            days = result.get("days", [])
            if days:
                if mode == "basin":
                    lines = [
                        "### Rainfall Prediction Report",
                        "- Scope: Complete Basin",
                        f"- Date: {result.get('input', {}).get('date', 'N/A')}",
                        f"- Mode: {result.get('way', 'N/A')}",
                        "",
                        "#### 3-Day Forecast",
                        "| Date | Basin Mean (mm) | Basin Max (mm) |",
                        "|---|---:|---:|",
                    ]
                    for d in days:
                        lines.append(
                            f"| {d.get('date', 'N/A')} | "
                            f"{float(d.get('pred_basin_mean_mm', 0.0)):.2f} | "
                            f"{float(d.get('pred_basin_max_mm', 0.0)):.2f} |"
                        )
                else:
                    lines = [
                        "### Rainfall Prediction Report",
                        f"- Location: {location_label}",
                        f"- Date: {result.get('input', {}).get('date', 'N/A')}",
                        f"- Mode: {result.get('way', 'N/A')}",
                        "",
                        "#### 3-Day Forecast",
                        "| Date | Max Coord | Local Max (mm) | Local Mean (mm) |",
                        "|---|---|---:|---:|",
                    ]
                    for d in days:
                        lines.append(
                            f"| {d.get('date', 'N/A')} | "
                            f"{d.get('max_coord', 'N/A')} | "
                            f"{float(d.get('max_pred_mm', 0.0)):.2f} | "
                            f"{float(d.get('mean_pred_mm', 0.0)):.2f} |"
                        )

                if result.get("overall_mae_mm") is not None:
                    lines.append("")
                    lines.append(
                        f"- Overall 3-day MAE: {float(result['overall_mae_mm']):.2f} mm"
                    )

                result["report"] = "\n".join(lines)

        result["tool"] = "predict_rainfall"
        _rainfall_prediction_cache[cache_key] = result
        return result

    except Exception as e:
        return {
            "status": "error",
            "tool": "predict_rainfall",
            "message": f"Rainfall prediction failed: {e}",
        }


# ══════════════════════════════════════════════════════════════════════
# 2. RETRIEVAL TOOL
# ══════════════════════════════════════════════════════════════════════


@tool
def retrieve_hydrology_context(
    query: str,
    scope: str = "general",
) -> str:
    """
    Search the knowledge base for hydrology information.
    Uses Wikipedia, Arxiv, DuckDuckGo, Tavily, and Exa — automatically
    routes to the best source based on query intent.

    Parameters
    ----------
    query : str
        Search query (e.g. "origin of Mahanadi river", "flood prediction ML papers").
    scope : str
        "general" or "mahanadi" for basin-specific results.

    Returns
    -------
    str
        The raw context text from the best matching source.
    """
    try:
        from agent.retrieval_chain import smart_route

        context = smart_route(query, scope=scope)
        return context
    except Exception as e:
        return f"Retrieval failed: {e}"


# ══════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════


def get_tools() -> list:
    """Return the list of active agent tools."""
    return [
        predict_flood_susceptibility,
        predict_rainfall,
        retrieve_hydrology_context,
    ]
