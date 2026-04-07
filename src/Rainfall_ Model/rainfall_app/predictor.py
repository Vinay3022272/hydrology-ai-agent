"""Rainfall prediction pipeline for date/lat/lon inputs."""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
import tempfile
import zipfile
from functools import lru_cache

# Keep TensorFlow startup logs minimal in app runtime.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from scipy.interpolate import RegularGridInterpolator
from tensorflow.keras import Model, layers, regularizers

tf.get_logger().setLevel("ERROR")

_MODULE_DIR = os.path.dirname(__file__)


def _load_local_module(module_name: str, file_name: str):
    """Load a sibling module by file path to avoid cross-app name collisions."""
    module_path = os.path.join(_MODULE_DIR, file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cfg = _load_local_module("rainfall_app_config", "config.py")
_basin = _load_local_module("rainfall_app_basin_check", "basin_check.py")

is_inside_basin = _basin.is_inside_basin

DATASET_PATH = _cfg.DATASET_PATH
DATASET_START_DATE = _cfg.DATASET_START_DATE
HORIZON = _cfg.HORIZON
LAT_MAX = _cfg.LAT_MAX
LAT_MIN = _cfg.LAT_MIN
LON_MAX = _cfg.LON_MAX
LON_MIN = _cfg.LON_MIN
LOOKBACK = _cfg.LOOKBACK
MODEL_PATH = _cfg.MODEL_PATH
N_FEATURES = _cfg.N_FEATURES
SCALER_X_PATH = _cfg.SCALER_X_PATH
SCALER_Y_PATH = _cfg.SCALER_Y_PATH
Y_CLEAN_PATH = _cfg.Y_CLEAN_PATH

REG = regularizers.l2(1e-4)


class TileHorizon(layers.Layer):
    """Expand a single 2D spatial map into multiple future frames."""

    def __init__(self, horizon, **kwargs):
        super().__init__(**kwargs)
        self.horizon = horizon

    def call(self, x):
        return tf.tile(tf.expand_dims(x, 1), [1, self.horizon, 1, 1, 1])

    def get_config(self):
        return {**super().get_config(), "horizon": self.horizon}


class StackHorizon(layers.Layer):
    """Stack a 4D spatial tensor across time."""

    def __init__(self, horizon, **kwargs):
        super().__init__(**kwargs)
        self.horizon = horizon

    def call(self, x):
        return tf.stack([x] * self.horizon, axis=1)

    def get_config(self):
        return {**super().get_config(), "horizon": self.horizon}


class ReduceMeanTime(layers.Layer):
    """Mean-reduce along time dimension."""

    def call(self, x):
        return tf.reduce_mean(x, axis=1)

    def get_config(self):
        return super().get_config()


def build_cnn_lstm_unet(
    lookback: int = LOOKBACK,
    h: int = 31,
    w: int = 31,
    n_feats: int = N_FEATURES,
    horizon: int = HORIZON,
) -> Model:
    """Rebuild architecture compatible with notebook-trained weights."""
    inp = layers.Input(shape=(lookback, h, w, n_feats))

    s1 = layers.TimeDistributed(
        layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=REG)
    )(inp)
    s1 = layers.TimeDistributed(layers.BatchNormalization())(s1)
    s1 = layers.TimeDistributed(
        layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=REG)
    )(s1)
    s1 = layers.TimeDistributed(layers.BatchNormalization())(s1)

    x = layers.TimeDistributed(layers.MaxPooling2D(2, padding="same"))(s1)

    s2 = layers.TimeDistributed(
        layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=REG)
    )(x)
    s2 = layers.TimeDistributed(layers.BatchNormalization())(s2)

    x = layers.ConvLSTM2D(
        128,
        kernel_size=3,
        padding="same",
        return_sequences=False,
        recurrent_dropout=0.1,
    )(s2)
    x = layers.Dropout(0.4)(x)

    x = StackHorizon(horizon)(x)
    x = layers.ConvLSTM2D(
        128,
        kernel_size=3,
        padding="same",
        return_sequences=True,
        recurrent_dropout=0.1,
    )(x)

    s2_mean = ReduceMeanTime()(s2)
    s2_exp = TileHorizon(horizon)(s2_mean)
    x = layers.Concatenate(axis=-1)([x, s2_exp])

    x = layers.TimeDistributed(
        layers.Conv2DTranspose(
            64, 3, strides=2, padding="same", activation="relu", kernel_regularizer=REG
        )
    )(x)
    x = layers.TimeDistributed(layers.Cropping2D(((0, 1), (0, 1))))(x)

    s1_mean = ReduceMeanTime()(s1)
    s1_exp = TileHorizon(horizon)(s1_mean)
    x = layers.Concatenate(axis=-1)([x, s1_exp])

    x = layers.TimeDistributed(
        layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=REG)
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    out = layers.TimeDistributed(layers.Conv2D(1, 1, activation="relu"))(x)
    out = layers.Reshape((horizon, h, w))(out)

    return Model(inp, out)


@lru_cache(maxsize=1)
def _load_artifacts():
    """Load dataset tensors and scalers once."""
    import pickle

    dataset = np.load(DATASET_PATH)
    x_features = dataset["X"]

    with open(SCALER_X_PATH, "rb") as f:
        scaler_x = pickle.load(f)
    with open(SCALER_Y_PATH, "rb") as f:
        scaler_y = pickle.load(f)

    y_clean = None
    if os.path.exists(Y_CLEAN_PATH):
        y_clean = np.load(Y_CLEAN_PATH)

    return x_features, y_clean, scaler_x, scaler_y


@lru_cache(maxsize=1)
def _load_model():
    """Load model weights robustly from .keras or .weights.h5 file."""
    model = build_cnn_lstm_unet()

    if MODEL_PATH.lower().endswith(".keras"):
        with zipfile.ZipFile(MODEL_PATH, "r") as zf:
            if "model.weights.h5" not in zf.namelist():
                raise FileNotFoundError(
                    "model.weights.h5 missing inside .keras archive"
                )
            tmp_dir = tempfile.mkdtemp(prefix="rainfall_weights_")
            zf.extract("model.weights.h5", path=tmp_dir)
            model.load_weights(os.path.join(tmp_dir, "model.weights.h5"))
    else:
        model.load_weights(MODEL_PATH)

    return model


def _scale_array(arr, scaler, fit: bool = False):
    if arr.ndim == 4:
        t, h, w, f = arr.shape
        flat = arr.reshape(-1, f)
        if fit:
            scaler.fit(flat)
        return scaler.transform(flat).reshape(t, h, w, f)

    t, h, w = arr.shape
    flat = arr.reshape(-1, 1)
    if fit:
        scaler.fit(flat)
    return scaler.transform(flat).reshape(t, h, w)


def _rh_to_specific_humidity(rh_percent, temp_celsius, p_hpa=850.0):
    es = 6.112 * np.exp(17.67 * temp_celsius / (temp_celsius + 243.5))
    e = (rh_percent / 100.0) * es
    return 0.622 * e / (p_hpa - 0.378 * e)


def _fetch_openmeteo_window(target_date: pd.Timestamp, lookback: int = LOOKBACK):
    """Fetch full-basin lookback weather grids and interpolate to 31x31."""
    start_date = target_date - pd.Timedelta(days=lookback - 1)
    dates = [start_date + pd.Timedelta(days=i) for i in range(lookback)]

    n = 9
    s_lats = np.linspace(LAT_MIN, LAT_MAX, n)
    s_lons = np.linspace(LON_MIN, LON_MAX, n)
    d_lats = np.linspace(LAT_MIN, LAT_MAX, 31)
    d_lons = np.linspace(LON_MIN, LON_MAX, 31)

    lat_list = [round(float(la), 4) for la in s_lats for _ in s_lons]
    lon_list = [round(float(lo), 4) for _ in s_lats for lo in s_lons]

    today = pd.Timestamp(dt.date.today())
    start_date_str = str(start_date.date())
    end_date_str = str(target_date.date())

    forecast_params = {
        "hourly": ",".join(
            [
                "temperature_2m",
                "pressure_msl",
                "windspeed_10m",
                "winddirection_10m",
                "relativehumidity_850hPa",
                "temperature_850hPa",
            ]
        ),
        "past_days": lookback + 2,
        "forecast_days": 1,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    archive_params = {
        "hourly": ",".join(
            [
                "temperature_2m",
                "pressure_msl",
                "windspeed_10m",
                "winddirection_10m",
                "relativehumidity_850hPa",
                "temperature_850hPa",
            ]
        ),
        "start_date": start_date_str,
        "end_date": end_date_str,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    def _fetch_locations_chunked(
        all_lats: list[float], all_lons: list[float], chunk_size: int = 20
    ):
        """Fetch Open-Meteo in chunks to avoid long URL 502/Bad Gateway errors."""
        out = []
        for start in range(0, len(all_lats), chunk_size):
            lats_chunk = all_lats[start : start + chunk_size]
            lons_chunk = all_lons[start : start + chunk_size]
            last_err = None

            for _ in range(3):
                try:
                    use_archive = target_date < today
                    endpoint = (
                        "https://archive-api.open-meteo.com/v1/archive"
                        if use_archive
                        else "https://api.open-meteo.com/v1/forecast"
                    )
                    params = dict(archive_params if use_archive else forecast_params)
                    params["latitude"] = ",".join(map(str, lats_chunk))
                    params["longitude"] = ",".join(map(str, lons_chunk))

                    resp = requests.get(
                        endpoint,
                        params=params,
                        timeout=120,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    if isinstance(payload, dict):
                        payload = [payload]
                    out.extend(payload)
                    last_err = None
                    break
                except Exception as e:
                    # Forecast endpoint can be unstable (502/504). Fallback to archive.
                    try:
                        params_archive = dict(archive_params)
                        params_archive["latitude"] = ",".join(map(str, lats_chunk))
                        params_archive["longitude"] = ",".join(map(str, lons_chunk))
                        resp = requests.get(
                            "https://archive-api.open-meteo.com/v1/archive",
                            params=params_archive,
                            timeout=120,
                        )
                        resp.raise_for_status()
                        payload = resp.json()
                        if isinstance(payload, dict):
                            payload = [payload]
                        out.extend(payload)
                        last_err = None
                        break
                    except Exception:
                        last_err = e

            if last_err is not None:
                raise RuntimeError(
                    f"Open-Meteo chunk request failed ({start}:{start + len(lats_chunk)}): {last_err}"
                )

        return out

    # Notebook-equivalent path: try one-shot multi-location request first.
    try:
        use_archive = target_date < today
        endpoint = (
            "https://archive-api.open-meteo.com/v1/archive"
            if use_archive
            else "https://api.open-meteo.com/v1/forecast"
        )
        params = dict(archive_params if use_archive else forecast_params)
        params["latitude"] = ",".join(map(str, lat_list))
        params["longitude"] = ",".join(map(str, lon_list))
        resp = requests.get(endpoint, params=params, timeout=120)
        resp.raise_for_status()
        results = resp.json()
        if isinstance(results, dict):
            results = [results]
    except Exception:
        # App fallback: chunked requests for reliability during Open-Meteo gateway spikes.
        try:
            results = _fetch_locations_chunked(lat_list, lon_list, chunk_size=20)
        except Exception:
            # Final fallback with single-location calls.
            results = _fetch_locations_chunked(lat_list, lon_list, chunk_size=1)

    sparse = np.zeros((n, n, lookback, 5), dtype=np.float32)

    for idx, res in enumerate(results):
        i = idx // n
        j = idx % n
        hourly = res["hourly"]
        times = pd.DatetimeIndex(hourly["time"])

        for k, d in enumerate(dates):
            target_dt = pd.Timestamp(d.date()).replace(hour=12)
            time_diffs = np.abs((times - target_dt).total_seconds())
            t_idx = int(np.argmin(time_diffs))

            def safe(val, default):
                return float(val) if val is not None else default

            t2m_c = safe(hourly["temperature_2m"][t_idx], 25.0)
            pmsl = safe(hourly["pressure_msl"][t_idx], 1013.0)
            wspd = safe(hourly["windspeed_10m"][t_idx], 0.0)
            wdir = safe(hourly["winddirection_10m"][t_idx], 0.0)
            rh850 = safe(hourly["relativehumidity_850hPa"][t_idx], 60.0)
            t850_c = safe(hourly["temperature_850hPa"][t_idx], 15.0)

            t2m_k = t2m_c + 273.15
            msl_pa = pmsl * 100.0
            q = _rh_to_specific_humidity(rh850, t850_c)
            wdir_rad = np.deg2rad(wdir)
            u10 = -wspd * np.sin(wdir_rad)
            v10 = -wspd * np.cos(wdir_rad)

            sparse[i, j, k] = [t2m_k, msl_pa, q, u10, v10]

    x_window = np.zeros((lookback, 31, 31, 5), dtype=np.float32)
    lat_g, lon_g = np.meshgrid(d_lats, d_lons, indexing="ij")
    pts = np.stack([lat_g.ravel(), lon_g.ravel()], axis=-1)

    for k in range(lookback):
        for v in range(5):
            fn = RegularGridInterpolator(
                (s_lats, s_lons),
                sparse[:, :, k, v],
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            x_window[k, :, :, v] = fn(pts).reshape(31, 31)

    return x_window


def _determine_way(target_start: pd.Timestamp, dataset_len: int):
    dataset_start = pd.Timestamp(DATASET_START_DATE)
    today = pd.Timestamp(dt.date.today())

    target_end = target_start + pd.Timedelta(days=HORIZON - 1)
    target_start_idx = (target_start - dataset_start).days
    target_end_idx = (target_end - dataset_start).days

    train_end = int(dataset_len * 0.70)
    dataset_end = dataset_len - 1
    target_in_dataset = (target_start_idx >= 0) and (target_end_idx <= dataset_end)

    if target_start > today:
        return "future_blocked", target_start_idx, target_end_idx
    if target_in_dataset and target_end_idx < train_end:
        return "way1", target_start_idx, target_end_idx
    if target_in_dataset:
        return "way2", target_start_idx, target_end_idx
    return "way3a", target_start_idx, target_end_idx


def _nearest_cell(lat: float, lon: float):
    lats = np.linspace(LAT_MIN, LAT_MAX, 31)
    lons = np.linspace(LON_MIN, LON_MAX, 31)
    r = int(np.argmin(np.abs(lats - lat)))
    c = int(np.argmin(np.abs(lons - lon)))
    return r, c, float(lats[r]), float(lons[c])


def _prepare_prediction(date_str: str | None = None) -> dict:
    """Run model inference for a target date and return core tensors/metadata."""
    try:
        target_start = (
            pd.Timestamp(date_str) if date_str else pd.Timestamp(dt.date.today())
        )
    except Exception:
        return {"status": "invalid_date", "message": "Date must be YYYY-MM-DD."}

    x_features, y_clean, scaler_x, scaler_y = _load_artifacts()
    model = _load_model()

    way, target_start_idx, _ = _determine_way(target_start, len(x_features))
    if way == "future_blocked":
        return {
            "status": "future_date_blocked",
            "message": "Future dates are not supported. Use today or a past date.",
            "today": str(pd.Timestamp(dt.date.today()).date()),
        }

    lookback_start_idx = target_start_idx - (LOOKBACK - 1)

    if way in ("way1", "way2"):
        if lookback_start_idx < 0:
            return {
                "status": "date_out_of_range",
                "message": "Date is too early for lookback window.",
                "min_date": str(
                    (
                        pd.Timestamp(DATASET_START_DATE)
                        + pd.Timedelta(days=LOOKBACK - 1)
                    ).date()
                ),
            }
        x_window_raw = x_features[lookback_start_idx : lookback_start_idx + LOOKBACK]
    else:
        try:
            x_window_raw = _fetch_openmeteo_window(target_start, LOOKBACK)
        except Exception as e:
            return {
                "status": "live_data_error",
                "message": f"Failed to fetch weather forcing data: {e}",
            }

    x_window_scaled = _scale_array(x_window_raw, scaler_x, fit=False)
    x_input = x_window_scaled[np.newaxis, ...]

    predicted_scaled = model.predict(x_input, verbose=0)
    predicted_scaled = np.clip(predicted_scaled, 0, None)
    pred_mm = scaler_y.inverse_transform(predicted_scaled[0].reshape(-1, 1)).reshape(
        predicted_scaled[0].shape
    )

    has_actual = False
    actual_mm = None
    if way in ("way1", "way2") and y_clean is not None:
        actual_raw = y_clean[target_start_idx : target_start_idx + HORIZON]
        if actual_raw.shape[0] == HORIZON:
            has_actual = True
            actual_mm = actual_raw

    return {
        "status": "ok",
        "target_start": target_start,
        "way": way,
        "pred_mm": pred_mm,
        "has_actual": has_actual,
        "actual_mm": actual_mm,
    }


def predict_rainfall_for_basin(date_str: str | None = None) -> dict:
    """Predict 3-day rainfall maps and basin summary metrics."""
    core = _prepare_prediction(date_str=date_str)
    if core.get("status") != "ok":
        return core

    target_start = core["target_start"]
    way = core["way"]
    pred_mm = core["pred_mm"]
    has_actual = core["has_actual"]
    actual_mm = core["actual_mm"]

    days = []
    for d in range(HORIZON):
        day_date = target_start + pd.Timedelta(days=d)
        day_pred = pred_mm[d]
        entry = {
            "date": str(day_date.date()),
            "pred_basin_mean_mm": float(np.mean(day_pred)),
            "pred_basin_max_mm": float(np.max(day_pred)),
            "pred_grid_31x31": np.round(day_pred, 3).tolist(),
        }
        if has_actual and actual_mm is not None:
            day_actual = actual_mm[d]
            entry["actual_basin_mean_mm"] = float(np.mean(day_actual))
            entry["day_mae_mm"] = float(np.mean(np.abs(day_actual - day_pred)))
        days.append(entry)

    overall_mae = None
    if has_actual and actual_mm is not None:
        overall_mae = float(np.mean(np.abs(actual_mm - pred_mm)))

    return {
        "status": "success",
        "mode": "basin",
        "way": way,
        "input": {
            "date": str(target_start.date()),
            "lat_min": LAT_MIN,
            "lat_max": LAT_MAX,
            "lon_min": LON_MIN,
            "lon_max": LON_MAX,
        },
        "days": days,
        "overall_mae_mm": overall_mae,
        "notes": {
            "lookback_days": LOOKBACK,
            "horizon_days": HORIZON,
            "model": os.path.basename(MODEL_PATH),
        },
    }


def predict_rainfall_for_point(
    lat: float, lon: float, date_str: str | None = None
) -> dict:
    """Predict 3-day rainfall for a location and target start date."""
    if not is_inside_basin(lat, lon):
        return {
            "status": "outside_basin",
            "message": "Point is outside Mahanadi Basin boundary.",
        }

    core = _prepare_prediction(date_str=date_str)
    if core.get("status") != "ok":
        return core

    target_start = core["target_start"]
    way = core["way"]
    pred_mm = core["pred_mm"]
    has_actual = core["has_actual"]
    actual_mm = core["actual_mm"]

    lats = np.linspace(LAT_MIN, LAT_MAX, 31)
    lons = np.linspace(LON_MIN, LON_MAX, 31)
    row, col, grid_lat, grid_lon = _nearest_cell(lat, lon)

    # Match notebook predict_at_location(): analyze local +-1 degree patch.
    row_indices = np.where(np.abs(lats - grid_lat) <= 1.0)[0]
    col_indices = np.where(np.abs(lons - grid_lon) <= 1.0)[0]
    r_min, r_max = int(row_indices[0]), int(row_indices[-1])
    c_min, c_max = int(col_indices[0]), int(col_indices[-1])

    days = []

    for d in range(HORIZON):
        day_date = target_start + pd.Timedelta(days=d)
        day_pred = pred_mm[d]

        pred_patch = day_pred[r_min : r_max + 1, c_min : c_max + 1]
        flat_idx = int(np.argmax(pred_patch))
        local_row, local_col = np.unravel_index(flat_idx, pred_patch.shape)
        global_row = r_min + int(local_row)
        global_col = c_min + int(local_col)

        max_lat = float(lats[global_row])
        max_lon = float(lons[global_col])
        max_pred = float(pred_patch[local_row, local_col])
        mean_pred = float(pred_patch.mean())

        entry = {
            "date": str(day_date.date()),
            # Values below intentionally mirror notebook predict_at_location output.
            "pred_mm_at_point": max_pred,
            "pred_basin_mean_mm": mean_pred,
            "pred_basin_max_mm": max_pred,
            "max_pred_mm": max_pred,
            "mean_pred_mm": mean_pred,
            "max_coord": f"({max_lat:.2f}°N,{max_lon:.2f}°E)",
            "max_lat": max_lat,
            "max_lon": max_lon,
            # Small 31x31 grid for UI map visualization.
            "pred_grid_31x31": np.round(day_pred, 3).tolist(),
        }
        if has_actual and actual_mm is not None:
            day_actual = actual_mm[d]
            entry["actual_mm_at_point"] = float(day_actual[global_row, global_col])
            entry["day_mae_mm"] = float(np.mean(np.abs(day_actual - day_pred)))
        days.append(entry)

    overall_mae = None
    if has_actual and actual_mm is not None:
        overall_mae = float(np.mean(np.abs(actual_mm - pred_mm)))

    return {
        "status": "success",
        "way": way,
        "input": {
            "lat": float(lat),
            "lon": float(lon),
            "date": str(target_start.date()),
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            "local_area": {
                "lat_min": float(lats[r_min]),
                "lat_max": float(lats[r_max]),
                "lon_min": float(lons[c_min]),
                "lon_max": float(lons[c_max]),
                "n_pixels": int(len(row_indices) * len(col_indices)),
            },
        },
        "days": days,
        "overall_mae_mm": overall_mae,
        "notes": {
            "lookback_days": LOOKBACK,
            "horizon_days": HORIZON,
            "model": os.path.basename(MODEL_PATH),
        },
    }
