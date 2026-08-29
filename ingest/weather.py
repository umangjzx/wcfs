"""Meteorology ingestion via Open-Meteo (genuine ERA5 reanalysis + GFS forecast, keyless).

- ``fetch_reanalysis``  -> ERA5 archive, hourly, for the training history.
- ``fetch_forecast``    -> GFS, hourly, 0..~96 h ahead, for inference-time known-future inputs.

Both return the canonical MET_COLUMNS schema (see ingest.common). Wind is decomposed into
u/v using the meteorological "wind FROM direction" convention.

CLI:
    python -m ingest.weather --forecast
    python -m ingest.weather --history --start 2021-10-01 --end 2024-02-29
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np
import pandas as pd

from config.settings import SETTINGS, Station, load_stations
from ingest.common import (
    MET_COLUMNS,
    SourceResult,
    get_json,
    read_table,
    save_snapshot,
    write_table,
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_CHUNK_PAUSE_S = 1.5  # be gentle on the Open-Meteo free tier

# canonical column -> Open-Meteo hourly variable
_FIELD_MAP = {
    "t2m": "temperature_2m",
    "d2m": "dew_point_2m",
    "rh2m": "relative_humidity_2m",
    "surface_pressure": "surface_pressure",
    "precip": "precipitation",
    "cloud": "cloud_cover",
    "solar": "shortwave_radiation",
    "blh": "boundary_layer_height",
    "t1000": "temperature_1000hPa",
    "t925": "temperature_925hPa",
    "t850": "temperature_850hPa",
}
_HOURLY_VARS = list(dict.fromkeys(
    list(_FIELD_MAP.values())
    + ["wind_speed_10m", "wind_direction_10m", "wind_speed_850hPa", "wind_direction_850hPa"]
))


def _uv(speed: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological wind vector: direction is where the wind blows FROM (degrees)."""
    rad = np.deg2rad(direction.astype("float64"))
    s = speed.astype("float64")
    return -s * np.sin(rad), -s * np.cos(rad)


def _parse_block(block: dict, station_id: str, kind: str, source: str,
                 issue_time: pd.Timestamp | None) -> pd.DataFrame:
    h = block.get("hourly")
    if not h or not h.get("time"):
        return pd.DataFrame(columns=MET_COLUMNS)
    ts = pd.to_datetime(h["time"], utc=True)
    out = pd.DataFrame({"station_id": station_id, "ts": ts, "kind": kind, "source": source})

    for col, var in _FIELD_MAP.items():
        out[col] = pd.to_numeric(pd.Series(h.get(var, [np.nan] * len(ts))), errors="coerce")

    ws10 = pd.to_numeric(pd.Series(h.get("wind_speed_10m", [np.nan] * len(ts))), errors="coerce")
    wd10 = pd.to_numeric(pd.Series(h.get("wind_direction_10m", [np.nan] * len(ts))), errors="coerce")
    out["wind_speed10"] = ws10
    out["wind_dir10"] = wd10
    out["wind_u10"], out["wind_v10"] = _uv(ws10.to_numpy(), wd10.to_numpy())

    ws850 = pd.to_numeric(pd.Series(h.get("wind_speed_850hPa", [np.nan] * len(ts))), errors="coerce")
    wd850 = pd.to_numeric(pd.Series(h.get("wind_direction_850hPa", [np.nan] * len(ts))), errors="coerce")
    out["wind_u850"], out["wind_v850"] = _uv(ws850.to_numpy(), wd850.to_numpy())

    if kind == "forecast" and issue_time is not None:
        out["lead_h"] = ((out["ts"] - issue_time) / pd.Timedelta(hours=1)).round().astype("Int64")
    else:
        out["lead_h"] = pd.NA

    for c in MET_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[MET_COLUMNS]


def _request(url: str, stations: list[Station], extra: dict, *, chunk: int = 20) -> list[dict]:
    """Call Open-Meteo for many stations, chunked. Returns per-station response blocks."""
    blocks: list[dict] = []
    for i in range(0, len(stations), chunk):
        grp = stations[i : i + chunk]
        params = {
            "latitude": ",".join(f"{s.lat:.4f}" for s in grp),
            "longitude": ",".join(f"{s.lon:.4f}" for s in grp),
            "hourly": ",".join(_HOURLY_VARS),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
            **extra,
        }
        payload = get_json(url, params=params)
        group_blocks = payload if isinstance(payload, list) else [payload]
        for st, blk in zip(grp, group_blocks, strict=False):
            blk["_station_id"] = st.id
            blocks.append(blk)
        if i + chunk < len(stations):
            time.sleep(_CHUNK_PAUSE_S)
    return blocks


def fetch_reanalysis(
    stations: list[Station] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> tuple[pd.DataFrame, SourceResult]:
    stations = stations or load_stations()
    start = start or (dt.date.today() - dt.timedelta(days=60))
    end = end or (dt.date.today() - dt.timedelta(days=6))
    try:
        blocks = _request(
            ARCHIVE_URL, stations,
            {"start_date": start.isoformat(), "end_date": end.isoformat(), "models": "era5_seamless"},
        )
    except Exception as exc:  # noqa: BLE001
        snap = _load("met_reanalysis")
        return snap if snap is not None else _empty(), SourceResult(
            "openmeteo:era5", ok=snap is not None, stale=True,
            message=f"fetch failed ({exc}); used snapshot")
    frames = [_parse_block(b, b["_station_id"], "reanalysis", "openmeteo:era5", None) for b in blocks]
    df = pd.concat(frames, ignore_index=True) if frames else _empty()
    save_snapshot("met_reanalysis", df)
    return df, SourceResult("openmeteo:era5", ok=not df.empty, rows=len(df),
                            message=f"{df['station_id'].nunique()} stations {start}..{end}")


def fetch_forecast(
    stations: list[Station] | None = None, *, forecast_days: int = 4, past_days: int = 1
) -> tuple[pd.DataFrame, SourceResult]:
    stations = stations or load_stations()
    issue_time = pd.Timestamp.now(tz="UTC").floor("h")
    try:
        blocks = _request(
            FORECAST_URL, stations,
            {"forecast_days": forecast_days, "past_days": past_days, "models": "gfs_seamless"},
        )
    except Exception as exc:  # noqa: BLE001
        snap = _load("met_forecast")
        return snap if snap is not None else _empty(), SourceResult(
            "openmeteo:gfs", ok=snap is not None, stale=True,
            message=f"fetch failed ({exc}); used snapshot")
    frames = [
        _parse_block(b, b["_station_id"], "forecast", "openmeteo:gfs", issue_time) for b in blocks
    ]
    df = pd.concat(frames, ignore_index=True) if frames else _empty()
    save_snapshot("met_forecast", df)
    horizon = df.loc[df["lead_h"] >= 0, "lead_h"].max() if not df.empty else 0
    return df, SourceResult("openmeteo:gfs", ok=not df.empty, rows=len(df),
                            message=f"{df['station_id'].nunique()} stations, "
                                    f"horizon +{horizon}h from {issue_time.isoformat()}")


def fetch_recent_history(
    stations: list[Station] | None = None, *, past_days: int = 92
) -> tuple[pd.DataFrame, SourceResult]:
    """GFS analysis for the last ``past_days`` (<=92), WITH pressure levels.

    Bridges the gap left by the Open-Meteo ERA5 archive, whose pressure-level fields are
    currently unavailable — so the most recent ~3 months of training history (and, run
    through the stubble season, the live season itself) has a full-fidelity 850 hPa
    transport wind and ISI theta-term.
    """
    stations = stations or load_stations()
    past_days = min(past_days, 92)
    try:
        blocks = _request(
            FORECAST_URL, stations,
            {"past_days": past_days, "forecast_days": 1, "models": "gfs_seamless"},
        )
    except Exception as exc:  # noqa: BLE001
        return _empty(), SourceResult("openmeteo:gfs_hist", ok=False, stale=True,
                                      message=f"fetch failed ({exc})")
    frames = [
        _parse_block(b, b["_station_id"], "reanalysis", "openmeteo:gfs_hist", None)
        for b in blocks
    ]
    df = pd.concat(frames, ignore_index=True) if frames else _empty()
    df = df[df["ts"] < pd.Timestamp.now(tz="UTC").floor("h")]  # keep past only
    return df, SourceResult("openmeteo:gfs_hist", ok=not df.empty, rows=len(df),
                            message=f"{df['station_id'].nunique()} stations, last {past_days}d")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=MET_COLUMNS)


def _load(name: str) -> pd.DataFrame | None:
    from ingest.common import load_snapshot

    return load_snapshot(name)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ingest.weather", description=__doc__)
    ap.add_argument("--forecast", action="store_true", help="fetch GFS forecast (default)")
    ap.add_argument("--history", action="store_true", help="fetch ERA5 reanalysis (surface + BLH)")
    ap.add_argument("--past-days", type=int, metavar="N",
                    help="fetch last N days of GFS analysis WITH pressure levels; "
                         "merge into met_history.parquet")
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--end", default=(dt.date.today() - dt.timedelta(days=6)).isoformat())
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if nothing new was fetched (for pipelines)")
    args = ap.parse_args(argv)

    SETTINGS.ensure_dirs()
    hist_path = SETTINGS.processed_dir / "met_history.parquet"
    fetched_ok = False

    def _merge_write(df: pd.DataFrame) -> None:
        if hist_path.exists():
            prior = read_table(hist_path)
            df = (pd.concat([prior, df], ignore_index=True)
                  .drop_duplicates(["station_id", "ts"], keep="last")
                  .sort_values(["station_id", "ts"]))
        write_table(df, hist_path)

    if args.past_days:
        df, res = fetch_recent_history(past_days=args.past_days)
        print(res.as_dict())
        if not df.empty:
            _merge_write(df)
            fetched_ok = True
            print(f"met_history.parquet now {len(read_table(hist_path))} rows")
    elif args.history:
        s, e = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
        # month-by-month so a 429 on one chunk doesn't lose the rest, and each
        # write is incremental (resumable)
        cursor = s
        while cursor < e:
            nxt = min(e, (cursor.replace(day=28) + dt.timedelta(days=8)).replace(day=1))
            try:
                df, res = fetch_reanalysis(start=cursor, end=nxt)
                print(f"  {cursor}..{nxt}: {res.as_dict()}")
                if not df.empty and res.ok:
                    _merge_write(df)
                    fetched_ok = True
            except Exception as exc:  # noqa: BLE001
                print(f"  {cursor}..{nxt}: FAILED {exc}")
            cursor = nxt
    else:
        df, res = fetch_forecast()
        print(res.as_dict())
        if not df.empty:
            write_table(df, SETTINGS.interim_dir / "weather_forecast.parquet")
            fetched_ok = True

    if args.strict and not fetched_ok:
        raise SystemExit("weather: nothing fetched (rate-limited or upstream down) — retry")


if __name__ == "__main__":
    main()
