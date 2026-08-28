"""Aerosol -> radiation -> boundary-layer feedback features.

These don't run radiative transfer; they build the interaction terms that let the ML
emulator *learn* the feedback the problem statement calls out:

  aod_proxy              log1p(PM2.5) — saturating proxy for column aerosol load
  radiative_dimming      lagged aerosol * available clear-sky sun (daytime) -> surface cooling
  expected_solar_reduction_wm2   rough W/m^2 of sunlight the aerosol layer removes
  pbl_suppression        lagged daytime aerosol -> today's mixing depth is capped
  pm25_x_blh_lag24       aerosol * boundary-layer depth interaction
  self_trapping          ISI * aod_proxy — strong inversion AND heavy aerosol => run-away accumulation

Expects a frame already on a regular hourly per-station grid with columns:
station_id, ts, lat, lon, pm25, blh, solar, cloud, and optionally isi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.calendar_feats import clear_sky_solar

_BETA_WM2 = 2.5  # ~ W/m^2 of solar loss per ug/m^3 of PM2.5 at midday (order-of-magnitude)

FEEDBACK_FEATURE_COLUMNS = [
    "aod_proxy", "aod_proxy_lag6", "aod_proxy_lag24",
    "clearness", "radiative_dimming", "expected_solar_reduction_wm2",
    "pbl_suppression", "pm25_x_blh_lag24",
    "pm25_tend_6h", "pm25_tend_24h", "pm25_persist_24h_mean", "pm25_persist_24h_std",
    "self_trapping",
]


def _clear_sky(frame: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(frame), dtype="float64")
    for (lat, lon), sub in frame.groupby(["lat", "lon"], sort=False):
        out[sub["_pos"].to_numpy()] = clear_sky_solar(sub["ts"], float(lat), float(lon))
    return out


def compute_feedback_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return feedback features aligned to ``frame`` (must have the columns in the module docstring)."""
    f = frame.sort_values(["station_id", "ts"]).reset_index(drop=True).copy()
    f["_pos"] = np.arange(len(f))
    g = f.groupby("station_id", sort=False)

    pm = f["pm25"].astype("float64")
    f["aod_proxy"] = np.log1p(pm.clip(lower=0))
    f["aod_proxy_lag6"] = g["aod_proxy"].shift(6)
    f["aod_proxy_lag24"] = g["aod_proxy"].shift(24)

    blh_lag24 = g["blh"].shift(24)
    pm_lag6 = g["pm25"].shift(6)
    pm_lag24 = g["pm25"].shift(24)

    cs = _clear_sky(f)
    f["_clear_sky"] = cs
    daytime = (cs > 20.0).astype("float64")
    f["clearness"] = np.clip(f["solar"].fillna(0.0) / np.where(cs > 1.0, cs, np.nan), 0.0, 1.2)
    f["clearness"] = f["clearness"].fillna(0.0)

    cs_norm = np.clip(cs / 1000.0, 0.0, 1.0)
    f["radiative_dimming"] = f["aod_proxy_lag6"].fillna(f["aod_proxy"]) * cs_norm * daytime
    f["expected_solar_reduction_wm2"] = (
        _BETA_WM2 * pm_lag6.fillna(pm).clip(lower=0) * cs_norm * daytime
    )
    f["pbl_suppression"] = f["aod_proxy_lag24"].fillna(f["aod_proxy"]) * daytime
    f["pm25_x_blh_lag24"] = pm_lag24 * blh_lag24

    f["pm25_tend_6h"] = pm - pm_lag6
    f["pm25_tend_24h"] = pm - pm_lag24
    roll = g["pm25"].rolling(24, min_periods=6)
    f["pm25_persist_24h_mean"] = roll.mean().reset_index(level=0, drop=True)
    f["pm25_persist_24h_std"] = roll.std().reset_index(level=0, drop=True)

    if "isi" in f:
        f["self_trapping"] = f["isi"].astype("float64") * f["aod_proxy"]
    else:
        f["self_trapping"] = np.nan

    return f[["station_id", "ts", *FEEDBACK_FEATURE_COLUMNS]]
