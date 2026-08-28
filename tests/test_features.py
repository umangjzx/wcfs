"""Offline tests for the coupled feature modules (geometry, physics sign conventions)."""

import numpy as np
import pandas as pd

from features.calendar_feats import add_calendar_features, clear_sky_solar, diwali_proximity
from features.feedback import compute_feedback_features
from features.inversion import compute_inversion_features
from features.stubble import compute_stubble_features


def _met_row(**over):
    base = dict(
        station_id="DEL-ito", ts=pd.Timestamp("2023-11-05T00:00:00Z"), kind="reanalysis",
        lead_h=np.nan, t2m=15.0, d2m=10.0, rh2m=80.0, wind_speed10=1.0, wind_dir10=315.0,
        wind_u10=0.7, wind_v10=0.7, surface_pressure=1010.0, precip=0.0, solar=0.0,
        cloud=5.0, blh=80.0, t1000=16.0, t925=18.0, t850=14.0, wind_u850=3.0, wind_v850=3.0,
    )
    base.update(over)
    return base


# --- Inversion Strength Index ------------------------------------------
def test_isi_high_for_shallow_calm_clear_night():
    met = pd.DataFrame([_met_row()])
    out = compute_inversion_features(met)
    assert 0.0 <= out["isi"].iloc[0] <= 1.0
    assert out["isi"].iloc[0] > 0.6  # shallow PBL + calm + clear night + warm-above


def test_isi_low_for_deep_windy_daytime():
    met = pd.DataFrame([_met_row(blh=2000.0, wind_speed10=8.0, solar=600.0, cloud=10.0,
                                 t925=8.0, t2m=20.0)])
    out = compute_inversion_features(met)
    assert out["isi"].iloc[0] < 0.2


def test_isi_robust_to_missing_pressure_levels():
    met = pd.DataFrame([_met_row(t925=np.nan, t850=np.nan)])
    out = compute_inversion_features(met)
    assert np.isfinite(out["isi"].iloc[0])          # renormalized over remaining components
    assert np.isnan(out["isi_theta"].iloc[0])
    assert out["isi"].iloc[0] > 0.5


def test_ventilation_index_is_blh_times_wind():
    met = pd.DataFrame([_met_row(blh=500.0, wind_speed10=4.0)])
    out = compute_inversion_features(met)
    assert out["ventilation_index"].iloc[0] == 2000.0


# --- Stubble-plume transport vector -----------------------------------
def test_plume_vector_points_at_station_when_wind_carries_smoke_in():
    # Fire ~NW of Delhi; wind FROM the NW (u,v both +ve -> blowing toward SE / Delhi).
    station = _pick_station("DEL-ito")
    fire_lat, fire_lon = station.lat + 1.0, station.lon - 1.0  # NW of the station
    met = pd.DataFrame([
        _met_row(station_id=station.id, ts=pd.Timestamp("2023-11-05T06:00:00Z"),
                 wind_u850=5.0, wind_v850=-5.0)  # from NW: eastward + southward
    ])
    fires = pd.DataFrame({
        "date": [pd.Timestamp("2023-11-05")], "cluster_id": ["c1"],
        "lat": [fire_lat], "lon": [fire_lon], "frp_sum": [5000.0],
        "count": [50], "confidence_mean": [80.0], "source": ["firms"],
    })
    out = compute_stubble_features(met, fires, [station])
    row = out.iloc[0]
    assert row["incoming_stubble_load"] > 0
    # smoke arrives FROM the north-west -> bearing roughly 270..360
    assert 250 <= row["plume_from_bearing_deg"] <= 360


def test_plume_zero_when_wind_blows_smoke_away():
    station = _pick_station("DEL-ito")
    met = pd.DataFrame([
        _met_row(station_id=station.id, wind_u850=-6.0, wind_v850=6.0)  # toward NW, away from Delhi
    ])
    fires = pd.DataFrame({
        "date": [pd.Timestamp("2023-11-05")], "cluster_id": ["c1"],
        "lat": [station.lat + 1.0], "lon": [station.lon - 1.0], "frp_sum": [5000.0],
        "count": [50], "confidence_mean": [80.0], "source": ["firms"],
    })
    out = compute_stubble_features(met, fires, [station])
    assert out.iloc[0]["incoming_stubble_load"] == 0.0


def test_stubble_all_zero_without_fire_data():
    met = pd.DataFrame([_met_row(), _met_row(ts=pd.Timestamp("2023-11-05T01:00:00Z"))])
    out = compute_stubble_features(met, None)
    assert (out["incoming_stubble_load"] == 0).all()
    assert len(out) == 2


def _pick_station(sid):
    from config.settings import station_index

    return station_index()[sid]


# --- Feedback ---------------------------------------------------------
def test_self_trapping_scales_with_isi_and_aerosol():
    ts = pd.date_range("2023-11-05", periods=30, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "station_id": "DEL-ito", "ts": ts, "lat": 28.63, "lon": 77.24,
        "pm25": np.linspace(50, 300, 30), "blh": 100.0, "solar": 0.0, "cloud": 10.0,
        "isi": np.linspace(0.1, 0.9, 30),
    })
    out = compute_feedback_features(frame)
    assert out["self_trapping"].iloc[-1] > out["self_trapping"].iloc[0]
    assert (out["aod_proxy"] >= 0).all()


# --- Calendar / solar ----------------------------------------------
def test_clear_sky_solar_zero_at_night_positive_at_noon():
    night = pd.Series([pd.Timestamp("2023-11-05T20:00:00Z")])   # ~01:30 IST
    noon = pd.Series([pd.Timestamp("2023-11-05T07:00:00Z")])    # ~12:30 IST
    assert clear_sky_solar(night, 28.6, 77.2)[0] == 0.0
    assert clear_sky_solar(noon, 28.6, 77.2)[0] > 400.0


def test_diwali_proximity_peaks_on_the_day():
    on_day = pd.Series([pd.Timestamp("2023-11-12").date()])
    far = pd.Series([pd.Timestamp("2023-11-30").date()])
    assert diwali_proximity(on_day)[0] == 1.0
    assert diwali_proximity(far)[0] < 0.01


def test_add_calendar_features_stubble_season_flag():
    df = pd.DataFrame({"ts": pd.to_datetime(
        ["2023-11-05T06:00:00Z", "2023-07-01T06:00:00Z"], utc=True)})
    out = add_calendar_features(df)
    assert out["is_stubble_season"].tolist() == [1, 0]
    assert set(out["hour_sin"].round(6).unique()) <= set(np.round(np.sin(
        2 * np.pi * np.arange(24) / 24), 6))
