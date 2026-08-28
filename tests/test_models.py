"""Tests for supervised reshaping and the LightGBM forecaster on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from models.baseline_lgbm import LGBMForecaster, train
from models.dataset import make_supervised


def _synthetic_features(n_stations=3, hours=24 * 30, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-11-01", periods=hours, freq="h", tz="UTC")
    rows = []
    for s in range(n_stations):
        diurnal = 60 + 40 * np.sin(2 * np.pi * (ts.hour - 4) / 24)
        pm = np.clip(diurnal + rng.normal(0, 8, hours) + np.linspace(0, 30, hours), 5, None)
        rows.append(pd.DataFrame({
            "station_id": f"S{s}", "ts": ts, "site_type": "traffic", "city": "Delhi",
            "pm25": pm, "pm10": pm * 1.6, "no2": 30 + rng.normal(0, 5, hours),
            "blh": 300 + 500 * np.clip(np.sin(2 * np.pi * (ts.hour - 6) / 24), 0, 1),
            "t2m": 18 + 6 * np.sin(2 * np.pi * (ts.hour - 6) / 24),
            "wind_speed10": np.abs(rng.normal(2, 1, hours)),
            "wind_u10": rng.normal(0, 2, hours), "wind_v10": rng.normal(0, 2, hours),
            "wind_u850": rng.normal(0, 4, hours), "wind_v850": rng.normal(0, 4, hours),
            "d2m": 12.0, "rh2m": 70.0, "surface_pressure": 1010.0, "precip": 0.0,
            "solar": np.clip(600 * np.sin(2 * np.pi * (ts.hour - 6) / 24), 0, None),
            "cloud": 20.0,
            "isi": np.clip(0.7 - 0.5 * np.sin(2 * np.pi * (ts.hour - 6) / 24), 0, 1),
            "isi_pbl": 0.5, "isi_stagnation": 0.4, "ventilation_index": 1500.0,
            "incoming_stubble_load": np.abs(rng.normal(50, 30, hours)),
            "stubble_index": 0.05, "plume_u": 0.0, "plume_v": 0.0,
            "aqi": pm * 1.5, "aqi_dominant": "pm25",
            "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
            "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
            "doy_sin": 0.1, "doy_cos": 0.9, "month_sin": 0.0, "month_cos": 1.0,
            "local_hour": ts.hour, "local_dow": ts.dayofweek, "local_month": ts.month,
            "is_weekend": (ts.dayofweek >= 5).astype(int), "is_stubble_season": 1,
            "days_into_stubble_season": 30.0, "diwali_proximity": 0.0,
            "pm25_lag1": np.r_[pm[0], pm[:-1]], "pm25_lag24": np.r_[pm[:24], pm[:-24]],
        }))
    return pd.concat(rows, ignore_index=True)


def test_make_supervised_shapes_and_target_alignment():
    feat = _synthetic_features()
    sup, cols = make_supervised(feat, horizons=[1, 6, 24], target="pm25", base_stride_h=3)
    assert "target" in sup and "horizon" in sup
    assert set(sup["horizon"].unique()) == {1, 6, 24}
    # target at horizon h equals the station's pm25 h hours after ts0
    r = sup[sup["horizon"] == 6].iloc[0]
    src = feat[(feat["station_id"] == r["station_id"])
              & (feat["ts"] == r["ts"] + pd.Timedelta(hours=6))]
    assert src.iloc[0]["pm25"] == pytest.approx(r["target"])
    assert "f_blh" in cols and "horizon" in cols


def test_lgbm_forecaster_trains_predicts_and_roundtrips(tmp_path):
    feat = _synthetic_features(hours=24 * 40)
    fc = train(feat, horizons=[1, 6, 12, 24], target="pm25", base_stride_h=3, num_boost=60)
    sup, _ = make_supervised(feat, [1, 6, 12, 24], target="pm25", base_stride_h=6)
    pred = fc.predict(sup)

    assert {"pm25_p10", "pm25_p50", "pm25_p90", "aqi_p50", "valid_ts"} <= set(pred.columns)
    # quantiles ordered
    assert (pred["pm25_p10"] <= pred["pm25_p50"] + 1e-6).all()
    assert (pred["pm25_p50"] <= pred["pm25_p90"] + 1e-6).all()
    # beats a naive "predict the global mean" on MAE
    mae = (pred["pm25_p50"].to_numpy() - sup["target"].to_numpy())
    assert np.mean(np.abs(mae)) < sup["target"].std()

    fc.save(tmp_path)
    fc2 = LGBMForecaster.load(tmp_path)
    pred2 = fc2.predict(sup)
    np.testing.assert_allclose(pred["pm25_p50"].to_numpy(), pred2["pm25_p50"].to_numpy(), rtol=1e-5)


def test_walk_forward_folds_respects_gap():
    from models.backtest import walk_forward_folds

    feat = _synthetic_features(hours=24 * 90)
    folds = walk_forward_folds(feat, n_folds=3, test_days=14, gap_days=2)
    assert len(folds) >= 1
    for f in folds:
        assert f.train_end < f.test_start
        assert (f.test_start - f.train_end).days >= 2


def test_serving_forecast_and_alerts_and_drivers():
    from models.drivers import explain_station_forecast
    from models.serving import forecast, peak_alerts

    feat = _synthetic_features(n_stations=2, hours=24 * 40)
    fc = train(feat, horizons=[1, 6, 12, 24], num_boost=40, base_stride_h=6)

    # serving frame: recent obs rows + 24 future rows (no pm25) carrying f_* covariates
    serv = feat.groupby("station_id").tail(24 * 4).copy()
    fut = serv.groupby("station_id").tail(1).copy()
    future_rows = []
    for _, row in fut.iterrows():
        for h in range(1, 25):
            r = row.copy()
            r["ts"] = row["ts"] + pd.Timedelta(hours=h)
            r["pm25"] = np.nan
            future_rows.append(r)
    serv = pd.concat([serv, pd.DataFrame(future_rows)], ignore_index=True)

    fdf = forecast(fc, serv, horizon_h=24)
    assert not fdf.empty
    assert set(fdf["station_id"].unique()) == {"S0", "S1"}
    assert (fdf.groupby("station_id")["horizon"].max() == 24).all()
    assert {"pm25_p10", "pm25_p50", "pm25_p90", "aqi", "category", "advisory"} <= set(fdf.columns)
    assert (fdf["pm25_p10"] <= fdf["pm25_p90"] + 1e-6).all()

    alerts = peak_alerts(fdf)
    assert isinstance(alerts, list)

    from models.dataset import make_supervised

    sup, _ = make_supervised(feat.iloc[:15000], horizons=[6, 24], base_stride_h=6)
    d = explain_station_forecast(fc, fdf, sup, "S0")
    assert d["station_id"] == "S0"
    assert set(d["groups"]) & {
        "inversion_trapping", "stubble_transport", "local_emissions_persistence",
        "wind_ventilation", "other_meteorology", "time_season",
    }
