import os
import sys

RAINFALL_APP_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "src", "Rainfall_ Model", "rainfall_app"
    )
)
if RAINFALL_APP_DIR not in sys.path:
    sys.path.insert(0, RAINFALL_APP_DIR)

from predictor import predict_rainfall_for_basin, predict_rainfall_for_point


def get_rainfall_prediction(data: dict) -> dict:
    """Run rainfall prediction for a given date/lat/lon request payload."""
    mode = (data.get("mode") or "point").strip().lower()
    lat = data.get("lat")
    lon = data.get("lon")
    date = data.get("date")

    if mode == "basin":
        return predict_rainfall_for_basin(date_str=date)

    if lat is None or lon is None:
        return {
            "status": "invalid_input",
            "message": "Both lat and lon are required.",
        }

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return {
            "status": "invalid_input",
            "message": "lat and lon must be numeric values.",
        }

    return predict_rainfall_for_point(lat=lat, lon=lon, date_str=date)
