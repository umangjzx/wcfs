"""Offline tests for CPCB real-time parsing and station reconciliation."""

import pandas as pd

from config.settings import load_stations
from ingest.cpcb import parse_realtime_records, reconcile_stations

# Minimal data.gov.in-style records (mixed formats / junk values).
RECORDS = [
    {
        "state": "Delhi", "city": "Delhi", "station": "Anand Vihar, Delhi - DPCC",
        "last_update": "05-11-2024 09:00:00", "latitude": "28.6469", "longitude": "77.3160",
        "pollutant_id": "PM2.5", "avg_value": "412",
    },
    {
        "state": "Delhi", "city": "Delhi", "station": "Anand Vihar, Delhi - DPCC",
        "last_update": "05-11-2024 09:00:00", "latitude": "28.6469", "longitude": "77.3160",
        "pollutant_id": "OZONE", "avg_value": "31",
    },
    {
        "state": "Delhi", "city": "Delhi", "station": "Anand Vihar, Delhi - DPCC",
        "last_update": "05-11-2024 09:00:00", "latitude": "28.6469", "longitude": "77.3160",
        "pollutant_id": "CO", "avg_value": "NA",
    },
    {
        "state": "Haryana", "city": "Gurugram", "station": "Vikas Sadan, Gurugram - HSPCB",
        "last_update": "05-11-2024 09:00:00", "latitude": "28.4507", "longitude": "77.0263",
        "pollutant_id": "PM10", "pollutant_avg": "356",
    },
    {
        "state": "Rajasthan", "city": "Jaipur", "station": "Some Station, Jaipur - RSPCB",
        "last_update": "05-11-2024 09:00:00", "latitude": "26.9", "longitude": "75.8",
        "pollutant_id": "PM2.5", "avg_value": "88",
    },
]


def test_parse_maps_pollutants_and_drops_missing_values():
    df = parse_realtime_records(RECORDS)
    assert set(df["pollutant"]) == {"PM2.5", "O3", "PM10"}  # OZONE->O3, CO=NA dropped
    assert (df["ts"].dt.tz is not None) and str(df["ts"].dt.tz) == "UTC"
    # 09:00 IST -> 03:30 UTC
    anand_pm = df[(df["station_name"].str.startswith("Anand")) & (df["pollutant"] == "PM2.5")]
    assert anand_pm.iloc[0]["ts"].hour == 3 and anand_pm.iloc[0]["ts"].minute == 30
    assert anand_pm.iloc[0]["value"] == 412.0


def test_parse_handles_empty():
    assert parse_realtime_records([]).empty


def test_reconcile_matches_registry_names():
    df = parse_realtime_records(RECORDS)
    reconciled, unmatched = reconcile_stations(df, load_stations())
    matched = reconciled.dropna(subset=["station_id"])
    assert set(matched["station_id"]) == {"DEL-anand-vihar", "GGN-vikas-sadan"}
    # the Jaipur station is not in the NCR registry
    assert any("Jaipur" in u for u in unmatched)


def test_reconcile_relaxed_match_on_site_head():
    df = pd.DataFrame(
        {
            "station_name": ["Anand Vihar", "R K Puram, New Delhi - DPCC"],
            "city": ["Delhi", "Delhi"],
            "state": ["Delhi", "Delhi"],
            "lat": [28.64, 28.56],
            "lon": [77.31, 77.16],
            "ts": pd.to_datetime(["2024-11-05T09:00Z", "2024-11-05T09:00Z"], utc=True),
            "pollutant": ["PM2.5", "PM2.5"],
            "value": [400.0, 380.0],
            "source": ["x", "x"],
        }
    )
    reconciled, _ = reconcile_stations(df, load_stations())
    assert reconciled.iloc[0]["station_id"] == "DEL-anand-vihar"
    assert reconciled.iloc[1]["station_id"] == "DEL-r-k-puram"
