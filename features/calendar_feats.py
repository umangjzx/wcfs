"""Calendar / temporal features (IST-local), plus solar geometry used by the feedback module.

All inputs are timezone-aware UTC timestamps; local-time features use Asia/Kolkata.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from config.settings import STUBBLE_SEASON
from ingest.common import IST

# Diwali dates (the night of the main celebration; firecracker load spikes +-2 days).
DIWALI_DATES = {
    2019: dt.date(2019, 10, 27),
    2020: dt.date(2020, 11, 14),
    2021: dt.date(2021, 11, 4),
    2022: dt.date(2022, 10, 24),
    2023: dt.date(2023, 11, 12),
    2024: dt.date(2024, 11, 1),
    2025: dt.date(2025, 10, 21),
    2026: dt.date(2026, 11, 8),
    2027: dt.date(2027, 10, 29),
}


def _cyc(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    ang = 2 * np.pi * values / period
    return np.sin(ang), np.cos(ang)


def solar_zenith_cos(ts_utc: pd.Series, lat: float, lon: float) -> np.ndarray:
    """Approximate cos(solar zenith). Ignores the equation of time (~<=16 min error)."""
    t = pd.to_datetime(ts_utc, utc=True)
    doy = t.dt.dayofyear.to_numpy()
    frac_hour = (t.dt.hour + t.dt.minute / 60).to_numpy()
    decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    solar_hour = frac_hour + lon / 15.0
    hour_angle = np.deg2rad(15.0 * (solar_hour - 12.0))
    phi = np.deg2rad(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    return np.clip(cosz, -1.0, 1.0)


def clear_sky_solar(ts_utc: pd.Series, lat: float, lon: float, s0: float = 1000.0) -> np.ndarray:
    """Rough clear-sky global horizontal irradiance (W/m^2)."""
    cosz = solar_zenith_cos(ts_utc, lat, lon)
    return s0 * np.clip(cosz, 0.0, None) ** 1.15


def _days_into_season(local_dates: pd.Series) -> np.ndarray:
    (m0, d0), (m1, d1) = STUBBLE_SEASON
    out = np.zeros(len(local_dates), dtype="float64")
    for i, d in enumerate(local_dates):
        if d is None or pd.isna(d):
            continue
        start = dt.date(d.year, m0, d0)
        end = dt.date(d.year, m1, d1)
        if start <= d <= end:
            out[i] = (d - start).days
    return out


def diwali_proximity(local_dates: pd.Series) -> np.ndarray:
    out = np.zeros(len(local_dates), dtype="float64")
    for i, d in enumerate(local_dates):
        if d is None or pd.isna(d):
            continue
        dd = DIWALI_DATES.get(d.year)
        if dd is None:
            continue
        out[i] = np.exp(-abs((d - dd).days) / 3.0)
    return out


def add_calendar_features(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """Append calendar features. ``df`` must have a tz-aware UTC ``ts_col``."""
    out = df.copy()
    local = pd.to_datetime(out[ts_col], utc=True).dt.tz_convert(IST)
    hour = local.dt.hour.to_numpy()
    dow = local.dt.dayofweek.to_numpy()
    month = local.dt.month.to_numpy()
    doy = local.dt.dayofyear.to_numpy()

    out["local_hour"] = hour
    out["local_dow"] = dow
    out["local_month"] = month
    out["is_weekend"] = (dow >= 5).astype("int8")
    out["hour_sin"], out["hour_cos"] = _cyc(hour, 24)
    out["doy_sin"], out["doy_cos"] = _cyc(doy, 365.25)
    out["month_sin"], out["month_cos"] = _cyc(month, 12)

    (m0, d0), (m1, d1) = STUBBLE_SEASON
    md = list(zip(month, local.dt.day.to_numpy(), strict=False))
    in_season = [
        (m, d) >= (m0, d0) and (m, d) <= (m1, d1) for m, d in md
    ]
    out["is_stubble_season"] = np.array(in_season, dtype="int8")

    local_dates = local.dt.date
    out["days_into_stubble_season"] = _days_into_season(local_dates)
    out["diwali_proximity"] = diwali_proximity(local_dates)
    return out


CALENDAR_FEATURE_COLUMNS = [
    "local_hour", "local_dow", "local_month", "is_weekend",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
    "is_stubble_season", "days_into_stubble_season", "diwali_proximity",
]
