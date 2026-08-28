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
_MODEL_ALLOW_T0 = [
    # recent pollutant state / persistence
    "pm25", "pm10", "no2", "aqi", "aqi_inst",
    "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag12", "pm25_lag24",
    "pm25_roll6", "pm25_roll24", "pm25_tend_6h", "pm25_tend_24h",
    "pm10_lag6", "no2_lag6", "aqi_lag24",
    # coupled features (the point of the project)
    "isi", "isi_pbl", "isi_stagnation", "isi_theta", "self_trapping",
    "ventilation_index", "pbl_height", "dtheta_surface",
    "incoming_stubble_load", "stubble_index", "plume_from_bearing_deg",
    "fire_frp_active", "nearest_fire_km",
    "aod_proxy", "aod_proxy_lag24", "radiative_dimming", "pbl_suppression",
    "pm25_x_blh_lag24",
    # meteorology now
    "t2m", "d2m", "rh2m", "blh", "wind_speed10", "wind_u10", "wind_v10",
    "wind_u850", "wind_v850", "surface_pressure", "precip", "solar", "cloud",
    "blh_lag24", "wind_speed10_lag24", "t2m_lag24", "isi_lag24",
    # calendar / static
    "local_hour", "local_month", "is_weekend", "is_stubble_season",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "days_into_stubble_season", "diwali_proximity",
    "station_id", "site_type", "city", "lat", "lon",
]
_MODEL_ALLOW_FUTURE = [
    "blh", "t2m", "rh2m", "wind_speed10", "wind_u10", "wind_v10",
    "wind_u850", "wind_v850", "precip", "solar", "cloud",
    "isi", "isi_pbl", "isi_stagnation", "ventilation_index",
    "incoming_stubble_load", "stubble_index",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_weekend",
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
) -> tuple[pd.DataFrame, list[str]]:
    horizons = horizons or DEFAULT_HORIZONS
    feat = feat.sort_values(["station_id", "ts"]).reset_index(drop=True)
    feat["ts"] = pd.to_datetime(feat["ts"], utc=True)

    x_cols = [c for c in feature_columns(feat) if c != target or True]  # keep t0 target (persistence)
    fut_cols = [c for c in FUTURE_COLS if c in feat.columns]

    base = feat[feat["ts"].dt.hour % base_stride_h == 0].copy()
    base = base.dropna(subset=[target] + [c for c in ("blh", "t2m") if c in base])

    fut_base = feat[["station_id", "ts", target, *fut_cols]].copy()
    parts = []
    for h in horizons:
        f = fut_base.copy()
        f["ts0"] = f["ts"] - pd.Timedelta(hours=h)
        f = f.rename(columns={target: "target", **{c: f"f_{c}" for c in fut_cols}})
        f = f.drop(columns="ts")
        m = base.merge(f, left_on=["station_id", "ts"], right_on=["station_id", "ts0"], how="inner")
        m["horizon"] = h
        parts.append(m)

    sup = pd.concat(parts, ignore_index=True)
    sup = sup.dropna(subset=["target"])
    model_cols = [*x_cols, *[f"f_{c}" for c in fut_cols], "horizon"]
    model_cols = list(dict.fromkeys(model_cols))
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
