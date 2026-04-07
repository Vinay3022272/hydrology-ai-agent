"""
Module 5 - Prediction
Load the trained LightGBM model and return threshold-based flood prediction.
"""

import joblib
import pandas as pd
from functools import lru_cache

from config import MODEL_PATH, THRESHOLD_PATH


@lru_cache(maxsize=1)
def _load_model():
    """Load the pre-trained LightGBM model."""
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_threshold() -> float:
    """Load the tuned decision threshold used in notebook experiments."""
    return float(joblib.load(THRESHOLD_PATH))


def classify_risk(probability: float) -> str:
    """Map flood probability to a risk class."""
    if probability <= 0.33:
        return "Low"
    elif probability <= 0.66:
        return "Moderate"
    else:
        return "High"


def predict(model_input: pd.DataFrame) -> dict:
    """
    Run flood susceptibility prediction.

    Parameters
    ----------
    model_input : pd.DataFrame
        One-row model input DataFrame from preprocess.

    Returns
    -------
    dict
        Keys: 'probability' (float), 'risk_class' (str), 'prediction' (int), 'threshold' (float).
    """
    model = _load_model()
    threshold = _load_threshold()

    prob = float(model.predict_proba(model_input)[:, 1][0])
    pred = int(prob >= threshold)

    return {
        "probability": prob,
        "prediction": pred,
        "threshold": threshold,
        "risk_class": classify_risk(prob),
    }
