"""Indian National Air Quality Index (CPCB, 2014 methodology).

Reference: CPCB "National Air Quality Index" report (2014), IIT Kanpur breakpoint tables.

Overall AQI = max of available pollutant sub-indices, with the CPCB data-sufficiency rule:
at least three pollutants must be present and one of them must be PM2.5 or PM10.

Concentration units expected (same as the CPCB CAAQMS feed):
    PM2.5, PM10, NO2, SO2, NH3, O3  -> microgram / m^3
    CO                              -> milligram / m^3
    Pb                              -> microgram / m^3

Averaging periods (apply BEFORE calling ``compute_aqi``; use
``rolling_average_concentration`` for raw hourly series):
    O3, CO      -> 8-hour rolling mean (take the daily maximum of that mean)
    all others  -> 24-hour mean
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# pollutant -> list of (BP_lo, BP_hi, I_lo, I_hi) segments, ascending.
_SUBINDEX_TABLES: dict[str, list[tuple[float, float, float, float]]] = {
    "PM2.5": [
        (0, 30, 0, 50),
        (30, 60, 50, 100),
        (60, 90, 100, 200),
        (90, 120, 200, 300),
        (120, 250, 300, 400),
        (250, 500, 400, 500),
    ],
    "PM10": [
        (0, 50, 0, 50),
        (50, 100, 50, 100),
        (100, 250, 100, 200),
        (250, 350, 200, 300),
        (350, 430, 300, 400),
        (430, 600, 400, 500),
    ],
    "NO2": [
        (0, 40, 0, 50),
        (40, 80, 50, 100),
        (80, 180, 100, 200),
        (180, 280, 200, 300),
        (280, 400, 300, 400),
        (400, 600, 400, 500),
    ],
    "O3": [
        (0, 50, 0, 50),
        (50, 100, 50, 100),
        (100, 168, 100, 200),
        (168, 208, 200, 300),
        (208, 748, 300, 400),
        (748, 1000, 400, 500),
    ],
    "CO": [  # mg/m^3
        (0, 1.0, 0, 50),
        (1.0, 2.0, 50, 100),
        (2.0, 10, 100, 200),
        (10, 17, 200, 300),
        (17, 34, 300, 400),
        (34, 50, 400, 500),
    ],
    "SO2": [
        (0, 40, 0, 50),
        (40, 80, 50, 100),
        (80, 380, 100, 200),
        (380, 800, 200, 300),
        (800, 1600, 300, 400),
        (1600, 2400, 400, 500),
    ],
    "NH3": [
        (0, 200, 0, 50),
        (200, 400, 50, 100),
        (400, 800, 100, 200),
        (800, 1200, 200, 300),
        (1200, 1800, 300, 400),
        (1800, 2400, 400, 500),
    ],
    "Pb": [
        (0, 0.5, 0, 50),
        (0.5, 1.0, 50, 100),
        (1.0, 2.0, 100, 200),
        (2.0, 3.0, 200, 300),
        (3.0, 3.5, 300, 400),
        (3.5, 5.0, 400, 500),
    ],
}

POLLUTANTS: tuple[str, ...] = tuple(_SUBINDEX_TABLES.keys())

# hours of averaging, and whether to take the max of the rolling mean (8h pollutants)
_AVERAGING: dict[str, tuple[int, bool]] = {
    "PM2.5": (24, False),
    "PM10": (24, False),
    "NO2": (24, False),
    "SO2": (24, False),
    "NH3": (24, False),
    "Pb": (24, False),
    "O3": (8, True),
    "CO": (8, True),
}

_CATEGORIES: list[tuple[float, float, str]] = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]

_ADVISORY: dict[str, str] = {
    "Good": "Air quality is good. No health impact expected.",
    "Satisfactory": "Minor breathing discomfort possible for highly sensitive people.",
    "Moderate": (
        "Breathing discomfort for people with lung/heart disease, children and older adults. "
        "Sensitive groups should limit prolonged outdoor exertion."
    ),
    "Poor": (
        "Breathing discomfort for most on prolonged exposure. Reduce outdoor activity; "
        "sensitive groups should avoid it."
    ),
    "Very Poor": (
        "Respiratory illness on prolonged exposure. Avoid outdoor activity; keep windows "
        "closed; use N95 masks and air purifiers where possible."
    ),
    "Severe": (
        "Serious health impact for everyone; severe for those with existing disease. "
        "Stay indoors, avoid all outdoor exertion, run purifiers, follow GRAP restrictions."
    ),
    "Severe+": (
        "Emergency air pollution. Health warning of emergency conditions for the entire "
        "population. Remain indoors."
    ),
}

_MIN_POLLUTANTS = 3
_PM = {"PM2.5", "PM10"}


def sub_index(pollutant: str, concentration: float | None) -> float | None:
    """CPCB linear sub-index for one pollutant. ``None`` if concentration is missing/negative.

    Values above the top breakpoint are linearly extrapolated on the last segment slope
    (AQI can exceed 500 — the "Severe+ / beyond scale" region).
    """
    if pollutant not in _SUBINDEX_TABLES:
        raise KeyError(f"Unknown pollutant {pollutant!r}; expected one of {POLLUTANTS}")
    if concentration is None:
        return None
    try:
        c = float(concentration)
    except (TypeError, ValueError):
        return None
    if math.isnan(c) or c < 0:
        return None

    table = _SUBINDEX_TABLES[pollutant]
    for bp_lo, bp_hi, i_lo, i_hi in table:
        if c <= bp_hi:
            return round((i_hi - i_lo) / (bp_hi - bp_lo) * (c - bp_lo) + i_lo, 2)

    # above scale: extrapolate on the final segment
    bp_lo, bp_hi, i_lo, i_hi = table[-1]
    return round((i_hi - i_lo) / (bp_hi - bp_lo) * (c - bp_lo) + i_lo, 2)


def sub_index_series(pollutant: str, values) -> np.ndarray:
    """Vectorized :func:`sub_index` for a numpy array / pandas Series of concentrations.

    Negative / NaN concentrations map to NaN. Values above the top breakpoint are
    linearly extrapolated on the last segment's slope.
    """
    table = _SUBINDEX_TABLES[pollutant]
    xp = np.array([table[0][0]] + [seg[1] for seg in table], dtype="float64")
    fp = np.array([table[0][2]] + [seg[3] for seg in table], dtype="float64")
    v = np.asarray(values, dtype="float64")
    out = np.interp(v, xp, fp)

    bp_lo, bp_hi, i_lo, i_hi = table[-1]
    slope = (i_hi - i_lo) / (bp_hi - bp_lo)
    over = v > xp[-1]
    out[over] = i_hi + slope * (v[over] - bp_hi)
    out[~np.isfinite(v) | (v < 0)] = np.nan
    return np.round(out, 2)


def aqi_category(aqi: float | None) -> str:
    """Map an overall AQI value to its CPCB category label."""
    if aqi is None or (isinstance(aqi, float) and math.isnan(aqi)):
        return "Unknown"
    if aqi > 500:
        return "Severe+"
    for _lo, hi, label in _CATEGORIES:
        if aqi <= hi:
            return label
    return "Severe"


def health_advisory(category: str) -> str:
    """Plain-language health advisory for a CPCB AQI category."""
    return _ADVISORY.get(category, "")


@dataclass
class AQIResult:
    """Outcome of an AQI computation."""

    aqi: float | None
    category: str
    dominant_pollutant: str | None
    sub_indices: dict[str, float] = field(default_factory=dict)
    valid: bool = True
    reason: str = ""
    advisory: str = ""

    def as_dict(self) -> dict:
        return {
            "aqi": self.aqi,
            "category": self.category,
            "dominant_pollutant": self.dominant_pollutant,
            "sub_indices": self.sub_indices,
            "valid": self.valid,
            "reason": self.reason,
            "advisory": self.advisory,
        }


def compute_aqi(
    concentrations: dict[str, float | None],
    *,
    enforce_min_pollutants: bool = True,
) -> AQIResult:
    """Compute overall AQI from a dict of *already period-averaged* concentrations.

    Parameters
    ----------
    concentrations
        e.g. ``{"PM2.5": 210.0, "PM10": 300.0, "NO2": 55.0, "CO": 1.4}``.
        Unknown keys are ignored; missing/None/NaN pollutants are skipped.
    enforce_min_pollutants
        Apply the CPCB rule (>=3 pollutants incl. at least one of PM2.5/PM10).
        Set ``False`` for the forecasting pipeline where PM2.5 alone is modelled.
    """
    sub: dict[str, float] = {}
    for pol, conc in concentrations.items():
        if pol not in _SUBINDEX_TABLES:
            continue
        si = sub_index(pol, conc)
        if si is not None:
            sub[pol] = si

    if not sub:
        return AQIResult(None, "Unknown", None, {}, valid=False, reason="no pollutant data")

    if enforce_min_pollutants:
        if len(sub) < _MIN_POLLUTANTS or not (_PM & sub.keys()):
            return AQIResult(
                None,
                "Unknown",
                None,
                sub,
                valid=False,
                reason=(
                    f"insufficient data: need >={_MIN_POLLUTANTS} pollutants including "
                    "PM2.5 or PM10"
                ),
            )

    dominant = max(sub, key=sub.get)
    aqi_val = round(sub[dominant])
    category = aqi_category(aqi_val)
    return AQIResult(
        aqi=aqi_val,
        category=category,
        dominant_pollutant=dominant,
        sub_indices=sub,
        valid=True,
        advisory=health_advisory(category),
    )


def rolling_average_concentration(
    series: pd.Series,
    pollutant: str,
    *,
    min_fraction: float = 0.5,
) -> float | None:
    """Reduce a raw hourly concentration ``series`` to the single CPCB averaging value.

    - 24h pollutants: mean of the last 24 hourly values.
    - 8h pollutants (O3, CO): daily maximum of the 8-hour rolling mean.
    Returns ``None`` if fewer than ``min_fraction`` of the window is present.
    """
    if pollutant not in _AVERAGING:
        raise KeyError(f"Unknown pollutant {pollutant!r}")
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None

    hours, take_max = _AVERAGING[pollutant]
    if not take_max:
        window = s.tail(hours)
        if len(window) < math.ceil(min_fraction * hours):
            return None
        return float(window.mean())

    roll = s.rolling(window=hours, min_periods=math.ceil(min_fraction * hours)).mean().dropna()
    if roll.empty:
        return None
    return float(roll.max())
