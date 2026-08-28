"""Tests for the CPCB National AQI implementation."""

import math

import numpy as np
import pandas as pd
import pytest

from aqi.cpcb_aqi import (
    _SUBINDEX_TABLES,
    POLLUTANTS,
    aqi_category,
    compute_aqi,
    rolling_average_concentration,
    sub_index,
)


# --- hand-computed exact sub-index values ---------------------------------
@pytest.mark.parametrize(
    "pollutant, conc, expected",
    [
        ("PM2.5", 90, 200.0),   # top of the 100-200 band
        ("PM2.5", 45, 75.0),    # midpoint of 30-60 -> 50-100
        ("PM2.5", 0, 0.0),
        ("PM10", 100, 100.0),
        ("PM10", 175, 150.0),
        ("NO2", 80, 100.0),
        ("O3", 100, 100.0),
        ("CO", 1.5, 75.0),      # mg/m^3
        ("SO2", 40, 50.0),
    ],
)
def test_sub_index_known_points(pollutant, conc, expected):
    assert sub_index(pollutant, conc) == pytest.approx(expected, abs=0.01)


def test_sub_index_hits_band_ceilings_exactly():
    for pol, table in _SUBINDEX_TABLES.items():
        for bp_lo, bp_hi, i_lo, i_hi in table:
            assert sub_index(pol, bp_hi) == pytest.approx(i_hi, abs=0.01), (pol, bp_hi)
            assert sub_index(pol, bp_lo) == pytest.approx(i_lo, abs=0.01), (pol, bp_lo)


def test_sub_index_monotonic_non_decreasing():
    for pol in POLLUTANTS:
        top = _SUBINDEX_TABLES[pol][-1][1]
        xs = np.linspace(0, top, 200)
        ys = [sub_index(pol, x) for x in xs]
        assert all(b >= a - 1e-6 for a, b in zip(ys, ys[1:], strict=False)), pol


def test_sub_index_missing_and_bad_values():
    assert sub_index("PM2.5", None) is None
    assert sub_index("PM2.5", float("nan")) is None
    assert sub_index("PM2.5", -5) is None
    with pytest.raises(KeyError):
        sub_index("PM1", 10)


def test_sub_index_above_scale_extrapolates_past_500():
    assert sub_index("PM2.5", 1000) > 500


# --- overall AQI --------------------------------------------------------
def test_compute_aqi_is_max_subindex_and_names_dominant():
    res = compute_aqi({"PM2.5": 90, "PM10": 100, "NO2": 40, "CO": 1.0})
    assert res.valid
    assert res.dominant_pollutant == "PM2.5"
    assert res.aqi == 200
    assert res.category == "Moderate"


def test_compute_aqi_severe_winter_case():
    # A typical bad Delhi November day.
    res = compute_aqi({"PM2.5": 260, "PM10": 420, "NO2": 90, "CO": 2.5, "SO2": 20})
    assert res.valid
    assert res.dominant_pollutant == "PM2.5"
    assert res.category in {"Severe", "Very Poor"}
    assert res.aqi >= 401


def test_data_sufficiency_rule_enforced_by_default():
    # only two pollutants -> invalid
    res = compute_aqi({"PM2.5": 120, "NO2": 55})
    assert not res.valid
    assert "insufficient" in res.reason

    # three pollutants but none is PM -> invalid
    res2 = compute_aqi({"NO2": 55, "O3": 80, "CO": 1.2})
    assert not res2.valid

    # three incl. PM -> valid
    res3 = compute_aqi({"PM2.5": 120, "NO2": 55, "O3": 80})
    assert res3.valid


def test_data_sufficiency_rule_can_be_disabled_for_forecasting():
    res = compute_aqi({"PM2.5": 105}, enforce_min_pollutants=False)  # -> sub-index 250
    assert res.valid
    assert res.dominant_pollutant == "PM2.5"
    assert res.aqi == 250
    assert res.category == "Poor"


def test_compute_aqi_no_data():
    res = compute_aqi({"PM2.5": None, "PM10": float("nan")})
    assert not res.valid
    assert res.aqi is None


@pytest.mark.parametrize(
    "aqi, cat",
    [
        (0, "Good"),
        (50, "Good"),
        (75, "Satisfactory"),
        (150, "Moderate"),
        (250, "Poor"),
        (350, "Very Poor"),
        (450, "Severe"),
        (600, "Severe+"),
    ],
)
def test_aqi_category_boundaries(aqi, cat):
    assert aqi_category(aqi) == cat


def test_aqi_category_unknown():
    assert aqi_category(None) == "Unknown"
    assert aqi_category(float("nan")) == "Unknown"


# --- averaging helper -------------------------------------------------
def test_rolling_average_24h_mean():
    idx = pd.date_range("2024-11-05", periods=24, freq="h")
    s = pd.Series(np.full(24, 200.0), index=idx)
    assert rolling_average_concentration(s, "PM2.5") == pytest.approx(200.0)


def test_rolling_average_requires_enough_coverage():
    idx = pd.date_range("2024-11-05", periods=5, freq="h")
    s = pd.Series([100.0] * 5, index=idx)
    assert rolling_average_concentration(s, "PM2.5") is None  # <50% of 24h


def test_rolling_average_8h_takes_daily_max_of_rolling_mean():
    idx = pd.date_range("2024-11-05", periods=24, freq="h")
    vals = np.concatenate([np.full(12, 10.0), np.full(12, 100.0)])
    s = pd.Series(vals, index=idx)
    out = rolling_average_concentration(s, "O3")
    assert out == pytest.approx(100.0, abs=1e-6)


def test_rolling_average_all_nan():
    idx = pd.date_range("2024-11-05", periods=24, freq="h")
    s = pd.Series([math.nan] * 24, index=idx)
    assert rolling_average_concentration(s, "PM2.5") is None
