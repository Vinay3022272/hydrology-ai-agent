"""Configuration for Mahanadi rainfall prediction app."""

from __future__ import annotations

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAINFALL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PICKLE_DIR = os.path.join(RAINFALL_DIR, "pickle_files")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(PICKLE_DIR, "mahanadi_cnnlstm_unet11.keras")
DATASET_PATH = os.path.join(PICKLE_DIR, "Mahanadi_DeepLearning_Data.npz")
Y_CLEAN_PATH = os.path.join(PICKLE_DIR, "y_clean.npy")
SCALER_X_PATH = os.path.join(PICKLE_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(PICKLE_DIR, "scaler_y.pkl")

BASIN_SHP_PATH = os.path.join(DATA_DIR, "basin_boudary", "mahanadi_basin.shp")

MAP_CENTER = [21.5, 83.0]
MAP_ZOOM = 7

DATASET_START_DATE = "2005-01-01"
LOOKBACK = 7
HORIZON = 3
GRID_SIZE = 31
N_FEATURES = 5

LAT_MIN, LAT_MAX = 17.0, 24.5
LON_MIN, LON_MAX = 80.0, 87.5
