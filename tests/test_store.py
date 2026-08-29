"""The Postgres mirror must be a clean no-op when DATABASE_URL is unset, and its
frame-prep helpers must shape the pipeline state into the table schema."""

from __future__ import annotations

import pandas as pd
import pytest

from api.services import store


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    monkeypatch.setattr(store.SETTINGS, "database_url", None, raising=False)
    store._engine.cache_clear()
    store._schema_ready = False
    yield
    store._engine.cache_clear()


class _State:
    observations = pd.DataFrame()
    forecast = pd.DataFrame()
    alerts: list = []
    last_refresh = None
    model_name = "lgbm"

    def latest_obs_wide(self):  # noqa: D401 - matches PipelineState
        return pd.DataFrame()


def test_disabled_without_url():
    assert store.enabled() is False
    assert store.init_schema() is False
    assert store.write_refresh(_State()) is None
    assert store.observation_history("DEL-anand-vihar", 24) is None


def test_prep_forecasts_maps_columns_and_fills_optional():
    fc = pd.DataFrame(
        {
            "station_id": ["DEL-x", "DEL-x"],
            "issued_ts": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T00:00Z"]),
            "valid_ts": pd.to_datetime(["2026-01-01T01:00Z", "2026-01-01T02:00Z"]),
            "horizon": [1, 2],
            "pm25_p10": [10.0, 12.0],
            "pm25_p50": [20.0, 22.0],
            "pm25_p90": [40.0, 44.0],
            "aqi": [80, 95],
            "category": ["Satisfactory", "Satisfactory"],
        }
    )
    out = store._prep_forecasts(fc)
    assert list(out.columns) == [
        "issued_ts", "station_id", "valid_ts", "horizon_h",
        "pm25_p10", "pm25_p50", "pm25_p90", "pm10_p50", "no2_p50",
        "aqi", "category", "dominant_pollutant",
    ]
    assert out["horizon_h"].tolist() == [1, 2]
    assert out["pm10_p50"].isna().all()


def test_prep_alerts_shapes_rows():
    alerts = [
        {"station_id": "DEL-x", "level": "Severe", "lead_hours": 6,
         "valid_ts": "2026-01-01T06:00Z", "aqi": 420, "name": "X Road"},
    ]
    out = store._prep_alerts(alerts, "2026-01-01T00:00:00Z")
    assert out.loc[0, "station_name"] == "X Road"
    assert out.loc[0, "level"] == "Severe"
    assert int(out.loc[0, "aqi"]) == 420


def test_prep_observations_trims_window():
    now = pd.Timestamp.now(tz="UTC")
    obs = pd.DataFrame(
        {
            "station_id": ["DEL-x", "DEL-x"],
            "ts": [now - pd.Timedelta(days=90), now - pd.Timedelta(hours=1)],
            "pollutant": ["PM2.5", "PM2.5"],
            "value": [55.0, 60.0],
            "source": ["cpcb", "cpcb"],
        }
    )
    out = store._prep_observations(obs)
    assert len(out) == 1
    assert out.loc[0, "value"] == 60.0
