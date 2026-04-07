"""
Centralized configuration for the Flood Susceptibility Prediction App.
"""

import os

#  Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "src", "Flood_Model", "model_training", "lgbm")

#  File paths
MODEL_PATH = os.path.join(MODEL_DIR, "best_lgbm_model.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "best_lgbm_threshold.pkl")
BASIN_SHP_PATH = os.path.join(DATA_DIR, "basin_boudary", "mahanadi_basin.shp")
TRAINING_CSV_PATH = os.path.join(
    BASE_DIR,
    "src",
    "Flood_Model",
    "data",
    "proceessed",
    "flood_training_data_10k_clean.csv",
)
RASTER_DIR = os.path.join(DATA_DIR, "Flood_data_final")

#  Feature order (must match training)
FEATURE_ORDER = [
    "distance_to_river",
    "aspect",
    "dem",
    "flow_accumulation",
    "twi",
    "slope",
    "rainfall",
    "drainage_density",
    "ext_rainfall",
    "lulc",
    "soil",
]


#  Raster Feature Map
RASTER_FEATURE_MAP = {
    "distance_to_river": os.path.join(RASTER_DIR, "distance_to_river.tif"),
    "aspect": os.path.join(RASTER_DIR, "aspect.tif"),
    "dem": os.path.join(RASTER_DIR, "dem_30.tif"),
    "flow_accumulation": os.path.join(RASTER_DIR, "flow_acc_30m.tif"),
    "twi": os.path.join(RASTER_DIR, "TWI.tif"),
    "slope": os.path.join(RASTER_DIR, "fixed_slope.tif"),
    "rainfall": os.path.join(RASTER_DIR, "rainfall_30m_f.tif"),
    "drainage_density": os.path.join(RASTER_DIR, "drainage_density_30.tif"),
    "ext_rainfall": os.path.join(RASTER_DIR, "ext_rainfall.tif"),
    "lulc": os.path.join(RASTER_DIR, "lulc_30m.tif"),
    "soil": os.path.join(RASTER_DIR, "soil_clay_30m.tif"),
}

# ── CRS definitions ────────────────────────────────────────────────────────
# Rasters are in UTM zone 45N
RASTER_CRS_EPSG = 32645

# Basin boundary / training easting-northing use India Lambert Conformal Conic
# (WGS_1984_Lambert_Conformal_Conic — custom CRS from the .prj file)
BASIN_LCC_PROJ4 = (
    "+proj=lcc +lat_1=12.4729444 +lat_2=35.17280555 +lat_0=24 "
    "+lon_0=80 +x_0=4000000 +y_0=4000000 +datum=WGS84 +units=m +no_defs"
)

# ── Risk thresholds ─────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "Low": (0.00, 0.33),
    "Moderate": (0.34, 0.66),
    "High": (0.77, 1.00),
}

# ── Map defaults ────────────────────────────────────────────────────────────
# Approximate center of Mahanadi Basin
MAP_CENTER = [21.5, 83.0]
MAP_ZOOM = 7
