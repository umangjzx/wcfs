"""Turn the hourly feature matrix into a supervised multi-horizon training table.

For each (station, t0) base row and each forecast ``horizon`` h we emit one example:
    X          = everything knowable at t0 (current state + lags/rollings + ISI/stubble/
                 feedback + calendar-at-t0)
    f_*        = "known future" inputs at t0+h (forecast meteorology, advected fire load,
                 calendar) — at training these are the actual t0+h values (perfect-prog),
                 at serving they come from the GFS forecast frame
    target     = the pollutant value at t0+h
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = [1, 2, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]

CATEGORICAL = ["station_id", "site_type", "city",
               "local_hour", "local_month", "is_weekend", "is_stubble_season"]

# Curated model inputs (a subset of the ~160 columns in the feature matrix). Keeps
# LightGBM fast and interpretable; the full matrix stays available for analysis.
TARGETS = ["pm25", "pm10", "no2"]  # forecast each -> real multi-pollutant AQI

_MODEL_ALLOW_T0 = [
    # recent pollutant state / persistence
    "pm25", "pm10", "no2", "o3", "so2", "co", "aqi", "aqi_inst",
    "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag4", "pm25_lag6", "pm25_lag8",
    "pm25_lag12", "pm25_lag24", "pm25_lag36", "pm25_lag48",
    "pm25_roll6", "pm25_roll24", "pm25_rollstd24", "pm25_rollmax24",
    "pm25_tend_6h", "pm25_tend_24h", "pm25_anom", "pm25_over_pm10",
    "pm10_lag6", "pm10_lag24", "pm10_roll24", "no2_lag6", "no2_lag24", "no2_rollmax24",
    "o3_lag6", "so2_lag6", "co_lag6", "aqi_lag24",
    # coupled features (the point of the project)
    "isi", "isi_pbl", "isi_stagnation", "isi_theta", "isi_radiative",
    "self_trapping", "self_trapping_lag24", "isi_x_pm25", "stubble_x_isi",
    "ventilation_index", "ventilation_index_lag24", "ventilation_index_roll24",
    "pbl_height", "dtheta_surface", "blh_tend_6h", "vent_tend_6h",
    "incoming_stubble_load", "incoming_stubble_load_lag24", "incoming_stubble_load_roll24",
    "stubble_index", "stubble_index_lag24", "plume_from_bearing_deg",
    "fire_frp_active", "fire_frp_active_lag24", "nearest_fire_km",
    "aod_proxy", "aod_proxy_lag24", "radiative_dimming", "pbl_suppression",
    "pm25_x_blh_lag24", "wind_steadiness_6h", "hours_since_rain",
    # meteorology now
    "t2m", "d2m", "rh2m", "blh", "wind_speed10", "wind_dir10", "wind_u10", "wind_v10",
    "wind_u850", "wind_v850", "surface_pressure", "precip", "solar", "cloud",
    "blh_lag24", "wind_speed10_lag24", "wind_speed10_roll24", "t2m_lag24", "solar_lag6",
    "cloud_lag6", "isi_lag24",
    # calendar / static / regime
    "local_hour", "local_dow", "local_month", "is_weekend", "is_stubble_season",
    "is_morning_rush", "is_evening_peak",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
    "days_into_stubble_season", "diwali_proximity",
    "station_id", "site_type", "city", "lat", "lon",
]
_MODEL_ALLOW_FUTURE = [
    "blh", "t2m", "rh2m", "wind_speed10", "wind_dir10", "wind_u10", "wind_v10",
    "wind_u850", "wind_v850", "precip", "solar", "cloud", "surface_pressure",
    "isi", "isi_pbl", "isi_stagnation", "isi_radiative", "ventilation_index",
    "incoming_stubble_load", "stubble_index", "hours_since_rain",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_weekend",
    "is_morning_rush", "is_evening_peak",
    "diwali_proximity", "is_stubble_season", "days_into_stubble_season",
]

# columns never used as model inputs
_EXCLUDE = {"ts", "ts0", "aqi_category", "kind", "source", "lead_h"}

# "known future" columns carried at t0+h
FUTURE_COLS = _MODEL_ALLOW_FUTURE


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """Curated t0 model inputs that exist in this feature matrix."""
    return [c for c in _MODEL_ALLOW_T0 if c in feat.columns]


def make_supervised(
    feat: pd.DataFrame,
    horizons: list[int] | None = None,
    *,
    target: str = "pm25",
    base_stride_h: int = 3,
    require_target: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the (station, t0, horizon) table. ``require_target=False`` for serving, where
    the t0+h rows have no observed target yet."""
    horizons = horizons or DEFAULT_HORIZONS
    feat = feat.sort_values(["station_id", "ts"]).reset_index(drop=True)
    feat["ts"] = pd.to_datetime(feat["ts"], utc=True)

    x_cols = list(feature_columns(feat))  # includes the t0 target column (persistence signal)
    fut_cols = [c for c in FUTURE_COLS if c in feat.columns]
    model_cols = list(dict.fromkeys([*x_cols, *[f"f_{c}" for c in fut_cols], "horizon"]))

    # keep only what we need + downcast floats -> ~half the memory (matters on Colab)
    keep_base = ["station_id", "ts", target, *[c for c in x_cols if c not in ("station_id",)]]
    base = feat.loc[feat["ts"].dt.hour % base_stride_h == 0, list(dict.fromkeys(keep_base))].copy()
    if require_target:
        base = base.dropna(subset=[target] + [c for c in ("blh", "t2m") if c in base])
    for c in base.select_dtypes("float64").columns:
        base[c] = base[c].astype("float32")

    fut_base = feat[["station_id", "ts", target, *fut_cols]].copy()
    for c in fut_base.select_dtypes("float64").columns:
        if c != target:
            fut_base[c] = fut_base[c].astype("float32")

    parts = []
    for h in horizons:
        f = fut_base.copy()
        f["ts0"] = f["ts"] - pd.Timedelta(hours=h)
        f = f.rename(columns={target: "target", **{c: f"f_{c}" for c in fut_cols}})
        f = f.drop(columns="ts")
        m = base.merge(f, left_on=["station_id", "ts"], right_on=["station_id", "ts0"], how="inner")
        parts.append(m.assign(horizon=np.int16(h)))

    sup = pd.concat(parts, ignore_index=True)
    del parts, fut_base, base
    if require_target:
        sup = sup.dropna(subset=["target"])
    return sup, model_cols


def encode_categoricals(df: pd.DataFrame, cols: list[str] | None = None
                        ) -> tuple[pd.DataFrame, list[str]]:
    """Cast the *categorical* columns among ``cols`` (default: all) to pandas 'category'.

    Only names in ``CATEGORICAL`` are ever converted — everything else stays numeric.
    """
    out = df.copy()
    candidates = cols if cols is not None else out.columns
    present = [c for c in candidates if c in out.columns and c in CATEGORICAL]
    for c in present:
        out[c] = out[c].astype("category")
    return out, present


def add_predicted_aqi(pred_pm25: np.ndarray) -> np.ndarray:
    """Single-pollutant AQI proxy from a PM2.5 forecast (PM2.5 dominates Delhi winter AQI)."""
    from aqi.cpcb_aqi import sub_index_series

    return sub_index_series("PM2.5", pred_pm25)
