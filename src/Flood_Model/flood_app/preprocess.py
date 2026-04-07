"""
Module 4 - Preprocessing
Build model-ready input DataFrame from extracted features.
"""

import pandas as pd

from config import FEATURE_ORDER


def preprocess_features(features: dict) -> pd.DataFrame:
    """
    Convert extracted feature dict into a model-ready 1-row DataFrame.

    Parameters
    ----------
    features : dict
        Dictionary of feature name → value (from raster_extract.extract_features).

    Returns
    -------
    pd.DataFrame
        One-row DataFrame in model feature order.
    """
    row = pd.DataFrame([{col: features[col] for col in FEATURE_ORDER}])

    # Keep categorical dtype consistent with LightGBM training setup.
    if "lulc" in row.columns:
        row["lulc"] = row["lulc"].astype("category")

    return row
