"""CPCB air-quality ingestion.

Real-time  : data.gov.in resource "Real time Air Quality Index" (one row per station x pollutant).
Historical : Open-Meteo Air Quality API (CAMS model reanalysis, keyless) by default;
             OpenAQ v3 ground-truth path is used when ``OPENAQ_API_KEY`` is set.

CLI:
    python -m ingest.cpcb --once
    python -m ingest.cpcb --history --start 2021-10-01 --end 2024-02-29
    python -m ingest.cpcb --sync-stations [--write]
    python -m ingest.cpcb --stats
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from config.settings import NCR_CITIES, SETTINGS, Station, load_stations
from ingest.common import (
    IST,
    OBS_COLUMNS,
    SourceResult,
    empty_obs,
    get_json,
    merge_observations,
    normalize_name,
    to_utc,
    write_table,
)

DATA_GOV_RESOURCE = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
DATA_GOV_URL = f"https://api.data.gov.in/resource/{DATA_GOV_RESOURCE}"
OPENMETEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# pollutant_id (as reported) -> canonical name used across VayuCast
_POLLUTANT_MAP = {
    "PM2.5": "PM2.5", "PM25": "PM2.5", "PM 2.5": "PM2.5",
    "PM10": "PM10", "PM 10": "PM10",
    "NO2": "NO2",
    "NH3": "NH3",
    "SO2": "SO2",
    "CO": "CO",
    "OZONE": "O3", "O3": "O3", "OZON": "O3",
}

_OPENMETEO_VARS = {
    "pm2_5": ("PM2.5", 1.0),
    "pm10": ("PM10", 1.0),
    "nitrogen_dioxide": ("NO2", 1.0),
    "ozone": ("O3", 1.0),
    "sulphur_dioxide": ("SO2", 1.0),
    "carbon_monoxide": ("CO", 0.001),  # Open-Meteo AQ returns CO in ug/m3; CPCB AQI wants mg/m3
}


# ---------------------------------------------------------------------------
# Real-time
# ---------------------------------------------------------------------------
def _num(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.upper() in {"NA", "N/A", "NONE", "-", "NAN"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_realtime_records(records: list[dict]) -> pd.DataFrame:
    """Pure parser: data.gov.in records -> tidy long frame (no station reconciliation yet).

    Output columns: station_name, city, state, lat, lon, ts, pollutant, value, source
    """
    rows: list[dict] = []
    for rec in records:
        raw_pol = str(rec.get("pollutant_id") or rec.get("pollutant") or "").strip().upper()
        pol = _POLLUTANT_MAP.get(raw_pol)
        if pol is None:
            continue
        value = _num(
            rec.get("avg_value",
                    rec.get("pollutant_avg", rec.get("pollutant_max", rec.get("max_value"))))
        )
        if value is None:
            continue
        rows.append(
            {
                "station_name": (rec.get("station") or "").strip(),
                "city": (rec.get("city") or "").strip(),
                "state": (rec.get("state") or "").strip(),
                "lat": _num(rec.get("latitude")),
                "lon": _num(rec.get("longitude")),
                "ts_raw": rec.get("last_update"),
                "pollutant": pol,
                "value": value,
                "source": "cpcb:data.gov.in",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = to_utc(
        pd.to_datetime(df["ts_raw"], format="%d-%m-%Y %H:%M:%S", errors="coerce"),
        assume_tz=IST,
    )
    # some rows carry ISO timestamps instead
    missing = df["ts"].isna()
    if missing.any():
        df.loc[missing, "ts"] = to_utc(pd.to_datetime(df.loc[missing, "ts_raw"], errors="coerce"))
    return df.drop(columns=["ts_raw"]).dropna(subset=["ts"])


_AGENCY_TOKENS = {"dpcc", "cpcb", "imd", "hspcb", "uppcb", "rspcb", "mpcb", "pcb", " npl"}
_CITY_TOKENS = [
    "greater noida", "new delhi", "delhi", "gurugram", "gurgaon", "noida",
    "faridabad", "ghaziabad",
]


def _site_head(norm: str) -> str:
    """Reduce a normalized station name to its distinguishing site tokens.

    Drops trailing city names and monitoring-agency codes so
    'r k puram new delhi dpcc' and 'r k puram delhi dpcc' both become 'r k puram'.
    """
    n = f" {norm} "
    for city in _CITY_TOKENS:
        n = n.replace(f" {city} ", " ")
    toks = [t for t in n.split() if t not in _AGENCY_TOKENS]
    return " ".join(toks).strip()


def reconcile_stations(
    df: pd.DataFrame, stations: list[Station] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Attach ``station_id`` by matching the feed's station name to the registry.

    Returns (frame_with_station_id_for_matched_rows, list_of_unmatched_feed_names).
    """
    stations = stations or load_stations()
    by_norm: dict[str, str] = {}
    heads: dict[str, str] = {}
    for s in stations:
        for label in (s.cpcb_name, s.name):
            nm = normalize_name(label)
            by_norm.setdefault(nm, s.id)
            heads.setdefault(_site_head(nm), s.id)

    def _match(name: str) -> str | None:
        n = normalize_name(name)
        if n in by_norm:
            return by_norm[n]
        h = _site_head(n)
        if not h:
            return None
        if h in heads:
            return heads[h]
        for hh, sid in heads.items():
            if hh and (hh.startswith(h + " ") or h.startswith(hh + " ")):
                return sid
        return None

    if df.empty:
        return df.assign(station_id=pd.Series(dtype=str)), []

    df = df.copy()
    df["station_id"] = df["station_name"].map(_match)
    unmatched = sorted(df.loc[df["station_id"].isna(), "station_name"].unique().tolist())
    return df, unmatched


def _fetch_reconciled(
    api_key: str, *, cities: list[str] | None = None, limit: int = 5000
) -> tuple[pd.DataFrame, list[str]]:
    """One live data.gov.in pull -> reconciled rows (station_id + lat/lon) and unmatched names."""
    cities = cities or NCR_CITIES
    city_norms = {normalize_name(c) for c in cities}
    payload = get_json(
        DATA_GOV_URL, params={"api-key": api_key, "format": "json", "limit": limit}
    )
    records = payload.get("records", []) if isinstance(payload, dict) else []
    parsed = parse_realtime_records(records)
    if not parsed.empty:
        parsed = parsed[parsed["city"].map(normalize_name).isin(city_norms)]
    return reconcile_stations(parsed)


def fetch_realtime(
    api_key: str | None = None, *, cities: list[str] | None = None, limit: int = 5000
) -> tuple[pd.DataFrame, SourceResult]:
    """Fetch the current CPCB observations live. No cache: on any failure returns empty."""
    api_key = api_key or SETTINGS.data_gov_in_api_key
    if not api_key:
        return empty_obs(), SourceResult(
            "cpcb:data.gov.in", ok=False, rows=0, message="no DATA_GOV_IN_API_KEY set"
        )

    try:
        reconciled, unmatched = _fetch_reconciled(api_key, cities=cities, limit=limit)
        obs = (
            reconciled.dropna(subset=["station_id"])[["station_id", "ts", "pollutant", "value", "source"]]
            .reset_index(drop=True)
        )
        msg = f"{len(obs)} obs, {obs['station_id'].nunique()} stations"
        if unmatched:
            msg += f"; {len(unmatched)} unmatched feed names"
        return obs[OBS_COLUMNS], SourceResult("cpcb:data.gov.in", ok=not obs.empty, rows=len(obs), message=msg)
    except Exception as exc:  # noqa: BLE001 - report the failure, serve nothing
        return empty_obs(), SourceResult(
            "cpcb:data.gov.in", ok=False, rows=0, message=f"live fetch failed ({exc})"
        )


# ---------------------------------------------------------------------------
# Historical
# ---------------------------------------------------------------------------
def fetch_history_openmeteo(
    stations: list[Station], start: dt.date, end: dt.date
) -> tuple[pd.DataFrame, SourceResult]:
    """Keyless historical hourly pollutant series from Open-Meteo (CAMS model output)."""
    frames: list[pd.DataFrame] = []
    hourly = ",".join(_OPENMETEO_VARS.keys())
    for st in stations:
        try:
            payload = get_json(
                OPENMETEO_AQ_URL,
                params={
                    "latitude": st.lat,
                    "longitude": st.lon,
                    "hourly": hourly,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "timezone": "UTC",
                    "domains": "cams_global",
                },
            )
            h = payload.get("hourly", {}) if isinstance(payload, dict) else {}
            times = pd.to_datetime(h.get("time", []), utc=True)
            if len(times) == 0:
                continue
            for var, (pol, factor) in _OPENMETEO_VARS.items():
                vals = h.get(var)
                if not vals:
                    continue
                s = pd.Series(vals, dtype="float64") * factor
                frames.append(
                    pd.DataFrame(
                        {
                            "station_id": st.id,
                            "ts": times,
                            "pollutant": pol,
                            "value": s.values,
                            "source": "openmeteo:cams",
                        }
                    )
                )
        except Exception:  # noqa: BLE001 - skip a station, keep going
            continue

    if not frames:
        return empty_obs(), SourceResult("openmeteo:cams", ok=False, message="no history returned")
    out = merge_observations(*frames).dropna(subset=["value"])
    return out, SourceResult("openmeteo:cams", ok=True, rows=len(out),
                             message=f"{out['station_id'].nunique()} stations, "
                                     f"{start}..{end}")


def fetch_history(
    stations: list[Station] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> tuple[pd.DataFrame, SourceResult]:
    """Historical pollutant series.

    Uses OpenAQ v3 ground-truth (real CPCB/CAAQMS measurements) when ``OPENAQ_API_KEY`` is
    set; otherwise the Open-Meteo CAMS model-output fallback.
    """
    stations = stations or load_stations()
    start = start or (dt.date.today() - dt.timedelta(days=120))
    end = end or dt.date.today()
    try:
        from ingest.openaq import fetch_history_s3

        obs, res = fetch_history_s3(stations, start, end)
        if res.ok and not obs.empty:
            return obs, res
    except Exception:  # noqa: BLE001 - fall through to the model proxy
        pass
    return fetch_history_openmeteo(stations, start, end)


# ---------------------------------------------------------------------------
# Station coordinate reconciliation
# ---------------------------------------------------------------------------
def propose_station_coords() -> pd.DataFrame:
    """Compare registry coords with the feed's reported lat/lon from a fresh live pull."""
    cols = ["station_id", "reg_lat", "reg_lon", "feed_lat", "feed_lon", "km_off"]
    api_key = SETTINGS.data_gov_in_api_key
    if not api_key:
        return pd.DataFrame(columns=cols)
    try:
        meta, _ = _fetch_reconciled(api_key)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=cols)
    if meta.empty:
        return pd.DataFrame(columns=cols)
    reg = {s.id: s for s in load_stations()}
    feed = (
        meta.dropna(subset=["station_id", "lat", "lon"])
        .groupby("station_id")[["lat", "lon"]]
        .median()
    )
    rows = []
    for sid, r in feed.iterrows():
        st = reg.get(sid)
        if not st:
            continue
        km = _haversine_km(st.lat, st.lon, r["lat"], r["lon"])
        rows.append(
            {"station_id": sid, "reg_lat": st.lat, "reg_lon": st.lon,
             "feed_lat": round(r["lat"], 4), "feed_lon": round(r["lon"], 4), "km_off": round(km, 2)}
        )
    return pd.DataFrame(rows).sort_values("km_off", ascending=False)


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    import math

    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_once() -> None:
    SETTINGS.ensure_dirs()
    obs, res = fetch_realtime()
    print(res.as_dict())
    if not obs.empty:
        write_table(obs, SETTINGS.interim_dir / "cpcb_realtime.parquet")
        prior = None
        proc = SETTINGS.processed_dir / "obs.parquet"
        if proc.exists():
            prior = pd.read_parquet(proc)
        merged = merge_observations(prior, obs)
        write_table(merged, proc)
        print(f"obs.parquet now {len(merged)} rows / {merged['station_id'].nunique()} stations")


def _cmd_history(start: str, end: str, source: str, append: bool) -> None:
    SETTINGS.ensure_dirs()
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    if source == "cams":
        df, res = fetch_history_openmeteo(load_stations(), s, e)
    elif source == "openaq":
        from ingest.openaq import fetch_history_s3

        df, res = fetch_history_s3(load_stations(), s, e)
    else:  # auto
        df, res = fetch_history(start=s, end=e)
    print(res.as_dict())
    if not df.empty:
        out = SETTINGS.processed_dir / "obs_history.parquet"
        if append and out.exists():
            df = merge_observations(pd.read_parquet(out), df)
        write_table(df, out)


def _cmd_sync_stations(write: bool) -> None:
    df = propose_station_coords()
    if df.empty:
        print("No feed metadata — set DATA_GOV_IN_API_KEY (the live pull returned nothing).")
        return
    print(df.to_string(index=False))
    if write:
        print("\n--write not yet implemented for stations.yaml; apply large km_off rows manually.")


def _cmd_stats() -> None:
    proc = SETTINGS.processed_dir / "obs.parquet"
    if not proc.exists():
        print("no data/processed/obs.parquet yet")
        return
    df = pd.read_parquet(proc)
    print(f"rows={len(df)} stations={df['station_id'].nunique()} "
          f"pollutants={sorted(df['pollutant'].unique())}")
    print(df.groupby("station_id").size().sort_values(ascending=False).head(20).to_string())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ingest.cpcb", description=__doc__)
    ap.add_argument("--once", action="store_true", help="fetch current real-time snapshot")
    ap.add_argument("--history", action="store_true", help="fetch historical series")
    ap.add_argument("--source", choices=["auto", "openaq", "cams"], default="auto",
                    help="auto = OpenAQ S3 ground truth, else CAMS model fallback")
    ap.add_argument("--append", action="store_true", help="merge into obs_history.parquet")
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--sync-stations", action="store_true", help="compare registry vs feed coords")
    ap.add_argument("--write", action="store_true", help="apply coord updates (with --sync-stations)")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args(argv)

    if args.history:
        _cmd_history(args.start, args.end, args.source, args.append)
    elif args.sync_stations:
        _cmd_sync_stations(args.write)
    elif args.stats:
        _cmd_stats()
    else:
        _cmd_once()


if __name__ == "__main__":
    main()
