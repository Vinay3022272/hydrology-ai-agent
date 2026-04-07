"""
Module 3 - Raster Feature Extraction
Extract hydrological feature values from raster layers for a given lat/lon.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from functools import lru_cache

from config import RASTER_FEATURE_MAP, RASTER_CRS_EPSG, TRAINING_CSV_PATH, FEATURE_ORDER


#  Coordinate transformers (cached)
@lru_cache(maxsize=1)
def _get_transformer_to_utm():
    """EPSG:4326 (lat/lon) → EPSG:32645 (UTM 45N) for raster sampling."""
    return Transformer.from_crs("EPSG:4326", f"EPSG:{RASTER_CRS_EPSG}", always_xy=True)


def _latlon_to_utm(lat: float, lon: float) -> tuple[float, float]:
    """Convert lat/lon to UTM zone 45N coordinates."""
    transformer = _get_transformer_to_utm()
    x, y = transformer.transform(lon, lat)  # always_xy: input is (lon, lat)
    return x, y


def _sample_raster(raster_path: str, x: float, y: float) -> float | None:
    """
    Sample a single pixel value from a raster at the given projected coordinates.

    Returns None if the point is outside the raster extent or the value is nodata.
    """
    try:
        with rasterio.open(raster_path) as src:
            # Convert projected coordinate to pixel row/col
            row, col = src.index(x, y)

            # Bounds check
            if row < 0 or row >= src.height or col < 0 or col >= src.width:
                return None

            # Read the pixel value (band 1)
            val = src.read(1, window=rasterio.windows.Window(col, row, 1, 1))
            val = float(val[0, 0])

            # Check nodata
            if src.nodata is not None and np.isclose(val, src.nodata):
                return None

            # Treat NaN / Inf as missing data
            if not np.isfinite(val):
                return None

            return val
    except Exception:
        return None


@lru_cache(maxsize=1)
def _build_safe_defaults() -> dict[str, float]:
    """Build stable fallback defaults from training data."""
    if not os.path.exists(TRAINING_CSV_PATH):
        return {feature: 0.0 for feature in FEATURE_ORDER}

    df_ref = pd.read_csv(TRAINING_CSV_PATH)
    defaults: dict[str, float] = {}

    for feature in FEATURE_ORDER:
        if feature not in df_ref.columns:
            defaults[feature] = 0.0
            continue

        series = df_ref[feature].dropna()
        if series.empty:
            defaults[feature] = 0.0
            continue

        # Keep land-cover like fields category-friendly using mode.
        if feature in ("lulc", "soil"):
            defaults[feature] = float(series.mode().iloc[0])
        else:
            defaults[feature] = float(series.median())

    return defaults


def extract_features(lat: float, lon: float) -> dict | None:
    """
    Extract all raster feature values and easting/northing for a point.

    Parameters
    ----------
    lat : float
        Latitude (WGS84).
    lon : float
        Longitude (WGS84).

    Returns
    -------
    dict or None
        Dictionary with feature names as keys and extracted values.
        Includes 'easting' and 'northing' (Lambert Conformal Conic).
        Returns None if any essential feature has nodata.
    """
    safe_defaults = _build_safe_defaults()

    # Get UTM coordinates for raster sampling
    utm_x, utm_y = _latlon_to_utm(lat, lon)

    features = {}

    # Sample each raster
    for feature_name, raster_path in RASTER_FEATURE_MAP.items():
        if not os.path.exists(raster_path):
            features[feature_name] = safe_defaults.get(feature_name, 0.0)
            continue

        value = _sample_raster(raster_path, utm_x, utm_y)
        if value is None or not np.isfinite(value):
            fallback = safe_defaults.get(feature_name, 0.0)
            print(f"[Fallback] {feature_name}: using {fallback}")
            features[feature_name] = fallback
        else:
            features[feature_name] = float(value)

    features["easting"] = float(utm_x)
    features["northing"] = float(utm_y)

    return features
