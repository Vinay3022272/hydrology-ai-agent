import sys
import os
import math

# Ensure the flood_app package is in the path
FLOOD_APP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "Flood_Model", "flood_app")
)
if FLOOD_APP_DIR not in sys.path:
    sys.path.insert(0, FLOOD_APP_DIR)

from basin_check import is_inside_basin
from raster_extract import extract_features
from preprocess import preprocess_features
from predict import predict


def _build_prediction_report(location_label: str, pred: dict, features: dict) -> str:
    """Build a readable, detailed model output summary."""
    prob = float(pred.get("probability", 0.0))
    risk = str(pred.get("risk_class", "Unknown"))
    threshold = float(pred.get("threshold", 0.5))
    decision = (
        "Flood-prone" if int(pred.get("prediction", 0)) == 1 else "Less flood-prone"
    )

    feature_priority = [
        "distance_to_river",
        "flow_accumulation",
        "twi",
        "slope",
        "rainfall",
        "drainage_density",
    ]
    lines = []
    for name in feature_priority:
        if name in features:
            val = features[name]
            if isinstance(val, (int, float)):
                lines.append(f"- {name}: {float(val):.4f}")
            else:
                lines.append(f"- {name}: {val}")

    return (
        f"### Flood Susceptibility Report\n"
        f"- Location: {location_label}\n"
        f"- Flood Probability: {prob:.2%}\n"
        f"- Risk Class: {risk}\n"
        f"- Decision Threshold: {threshold:.3f}\n"
        f"- Model Decision: {decision}\n\n"
        "#### Key Feature Snapshot\n" + "\n".join(lines) + "\n\n"
        "Interpretation: This score is model-driven from geospatial terrain, drainage, and rainfall predictors at the selected location."
    )


def _sanitize_floats(obj):
    """Recursively make a dict JSON-safe: convert numpy types, replace NaN/Inf with None."""
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


def get_flood_prediction(lat: float, lon: float) -> dict:
    """
    Runs the full flood prediction pipeline for a given lat/lon.
    """
    # 1. Basin check
    if not is_inside_basin(lat, lon):
        return {"status": "outside_basin"}

    # 2. Feature extraction
    raw_features = extract_features(lat, lon)
    if raw_features is None:
        return {"status": "missing_data"}

    # 3. Preprocess
    try:
        scaled = preprocess_features(raw_features)
    except Exception as e:
        return {"status": "preprocess_error", "message": str(e)}

    # 4. Predict
    pred = predict(scaled)

    location_label = f"({lat:.5f}, {lon:.5f})"
    report = _build_prediction_report(location_label, pred, raw_features)

    return _sanitize_floats(
        {
            "status": "success",
            "probability": pred["probability"],
            "prediction": pred.get("prediction"),
            "threshold": pred.get("threshold"),
            "risk_class": pred["risk_class"],
            "features": raw_features,
            "report": report,
        }
    )
