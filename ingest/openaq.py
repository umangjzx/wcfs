"""OpenAQ v3 — historical ground-truth pollutant series from the CPCB / CAAQMS network.

This is the real measured PM2.5/PM10/NO2/O3/SO2/CO that the model is trained to forecast
(the Open-Meteo CAMS path is a keyless model-output fallback only).

Flow:
  1. resolve each VayuCast station to its nearest OpenAQ location (CPCB/CAAQMS provider),
     caching the station -> {location, sensor per parameter} map to data/raw/openaq_map.json
  2. page /v3/sensors/{id}/hours over the requested window
  3. normalize units to canonical (ug/m3 for all; mg/m3 for CO) and emit long OBS rows

    python -m ingest.openaq --map                              # (re)build the station map
    python -m ingest.openaq --start 2024-10-01 --end 2025-03-01 [--stations DEL-ito,DEL-ito]
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from config.settings import SETTINGS, Station, load_stations
from ingest.common import (
    OBS_COLUMNS,
    SourceResult,
    get_json,
    http_get,
    merge_observations,
    write_table,
)

BASE = "https://api.openaq.org/v3"
S3_BASE = "https://openaq-data-archive.s3.amazonaws.com"
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_PROVIDERS = {"cpcb", "caaqm", "caaqms"}
_PARAMS = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "o3": "O3", "so2": "SO2", "co": "CO"}
_MW = {"NO2": 46.0055, "O3": 47.998, "SO2": 64.066, "CO": 28.010, "NH3": 17.031}
_MAP_PATH = SETTINGS.raw_dir / "openaq_map.json"
_MIN_INTERVAL_S = 1.05  # stay under the 60 req/min free-tier limit

_last_call = [0.0]


def _throttle() -> None:
    wait = _MIN_INTERVAL_S - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def _get(path: str, params: dict) -> dict:
    _throttle()
    return get_json(f"{BASE}{path}", params=params,
                    headers={"X-API-Key": SETTINGS.openaq_api_key or ""})


def _haversine_km(a_lat, a_lon, b_lat, b_lon) -> float:
    r = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dphi = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


# ---------------------------------------------------------------------------
def build_station_map(stations: list[Station] | None = None, radius_m: int = 3000) -> dict:
    """station_id -> {location_id, location_name, distance_km, sensors: {PARAM: sensor_id}}."""
    stations = stations or load_stations()
    out: dict[str, dict] = {}
    for st in stations:
        try:
            j = _get("/locations", {"coordinates": f"{st.lat},{st.lon}",
                                    "radius": radius_m, "limit": 100})
        except Exception as exc:  # noqa: BLE001
            out[st.id] = {"error": str(exc)}
            continue
        cands = []
        for loc in j.get("results", []):
            prov = (loc.get("provider", {}) or {}).get("name", "").lower()
            if _PROVIDERS and prov not in _PROVIDERS:
                continue
            c = loc.get("coordinates") or {}
            if c.get("latitude") is None:
                continue
            dist = _haversine_km(st.lat, st.lon, c["latitude"], c["longitude"])
            cands.append((dist, loc))
        if not cands:
            out[st.id] = {"error": "no CPCB/CAAQMS location within radius"}
            continue
        dist, loc = min(cands, key=lambda x: x[0])
        sensors: dict[str, int] = {}
        for s in loc.get("sensors", []):
            pname = s.get("parameter", {}).get("name")
            units = (s.get("parameter", {}) or {}).get("units", "")
            canon = _PARAMS.get(pname)
            if not canon:
                continue
            # prefer a mass-concentration sensor over ppb when both exist
            if canon not in sensors or "g/m" in units:
                sensors[canon] = s["id"]
        out[st.id] = {
            "location_id": loc["id"],
            "location_name": loc.get("name"),
            "distance_km": round(dist, 3),
            "sensors": sensors,
        }
    _MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MAP_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def load_station_map() -> dict:
    if _MAP_PATH.exists():
        return json.loads(_MAP_PATH.read_text(encoding="utf-8"))
    return build_station_map()


# ---------------------------------------------------------------------------
def _to_canonical(value: float, canon: str, units: str) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    u = (units or "").lower().replace("µ", "u").replace("�", "u")
    ugm3 = None
    if "g/m" in u and "mg" not in u:            # ug/m3
        ugm3 = value
    elif "mg/m" in u:
        ugm3 = value * 1000.0
    elif "ppm" in u:
        ugm3 = value * 1000.0 * _MW.get(canon, 1.0) / 24.45
    elif "ppb" in u:
        ugm3 = value * _MW.get(canon, 1.0) / 24.45
    else:                                        # assume already ug/m3
        ugm3 = value
    if canon == "CO":
        return ugm3 / 1000.0                     # CPCB AQI wants CO in mg/m3
    return ugm3


def _sensor_hours(sensor_id: int, start: dt.date, end: dt.date, canon: str) -> pd.DataFrame:
    rows: list[dict] = []
    page = 1
    while True:
        try:
            j = _get(f"/sensors/{sensor_id}/hours", {
                "datetime_from": start.isoformat(),
                "datetime_to": end.isoformat(),
                "limit": 1000,
                "page": page,
            })
        except Exception:  # noqa: BLE001
            break
        res = j.get("results", [])
        if not res:
            break
        for r in res:
            period = r.get("period", {}) or {}
            ts = (period.get("datetimeFrom") or {}).get("utc")
            units = (r.get("parameter", {}) or {}).get("units", "")
            val = _to_canonical(r.get("value"), canon, units)
            if ts and val is not None:
                rows.append({"ts": ts, "pollutant": canon, "value": val})
        if len(res) < 1000:
            break
        page += 1
        if page > 60:  # safety
            break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def fetch_history_openaq(
    stations: list[Station] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    *,
    params: list[str] | None = None,
) -> tuple[pd.DataFrame, SourceResult]:
    if not SETTINGS.openaq_api_key:
        return pd.DataFrame(columns=OBS_COLUMNS), SourceResult(
            "openaq:cpcb", ok=False, message="no OPENAQ_API_KEY")
    stations = stations or load_stations()
    start = start or (dt.date.today() - dt.timedelta(days=120))
    end = end or dt.date.today()
    want = set(params or _PARAMS.values())
    smap = load_station_map()

    frames: list[pd.DataFrame] = []
    n_sensors = 0
    for st in stations:
        entry = smap.get(st.id, {})
        for canon, sid in (entry.get("sensors") or {}).items():
            if canon not in want:
                continue
            d = _sensor_hours(sid, start, end, canon)
            n_sensors += 1
            if not d.empty:
                d["station_id"] = st.id
                d["source"] = "openaq:cpcb"
                frames.append(d[OBS_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=OBS_COLUMNS), SourceResult(
            "openaq:cpcb", ok=False, rows=0,
            message=f"no data ({n_sensors} sensors queried {start}..{end})")
    obs = merge_observations(*frames)
    return obs, SourceResult("openaq:cpcb", ok=True, rows=len(obs),
                             message=f"{obs['station_id'].nunique()} stations, "
                                     f"{n_sensors} sensors, {start}..{end}")


# ---------------------------------------------------------------------------
# S3 open-data archive (no key, no rate limit) — the reliable bulk-history path
# ---------------------------------------------------------------------------
_S3_PARAMS = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "o3": "O3", "so2": "SO2", "co": "CO"}


def _s3_list(prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        root = ET.fromstring(http_get(S3_BASE, params=params).text)
        keys += [e.text for e in root.iter(f"{_S3_NS}Key")]
        trunc = root.findtext(f"{_S3_NS}IsTruncated")
        token = root.findtext(f"{_S3_NS}NextContinuationToken")
        if trunc != "true" or not token:
            break
    return keys


def _s3_day_keys(location_id: int, start: dt.date, end: dt.date) -> list[str]:
    keys: list[str] = []
    y = start.year
    while y <= end.year:
        for m in range(1, 13):
            first = dt.date(y, m, 1)
            if first > end or dt.date(y, m, 28) < start.replace(day=1):
                continue
            keys += _s3_list(
                f"records/csv.gz/locationid={location_id}/year={y}/month={m:02d}/"
            )
        y += 1
    return keys


def _s3_fetch_csv(key: str) -> pd.DataFrame:
    try:
        raw = http_get(f"{S3_BASE}/{key}").content
        return pd.read_csv(io.BytesIO(raw), compression="gzip")
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def fetch_history_s3(
    stations: list[Station] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    *,
    workers: int = 16,
) -> tuple[pd.DataFrame, SourceResult]:
    """Bulk historical ground truth from the OpenAQ S3 archive (15-min raw -> hourly mean)."""
    stations = stations or load_stations()
    start = start or (dt.date.today() - dt.timedelta(days=180))
    end = end or dt.date.today()
    smap = load_station_map()

    frames: list[pd.DataFrame] = []
    n_files = 0
    for st in stations:
        lid = (smap.get(st.id) or {}).get("location_id")
        if not lid:
            continue
        day_keys = _s3_day_keys(lid, start, end)
        n_files += len(day_keys)
        if not day_keys:
            continue
        with ThreadPoolExecutor(max_workers=workers) as ex:
            parts = [d for d in ex.map(_s3_fetch_csv, day_keys) if not d.empty]
        if not parts:
            continue
        raw = pd.concat(parts, ignore_index=True)
        raw = raw[raw["parameter"].isin(_S3_PARAMS)]
        if raw.empty:
            continue
        raw["ts"] = pd.to_datetime(raw["datetime"], utc=True, errors="coerce")
        raw = raw.dropna(subset=["ts", "value"])
        raw["canon"] = raw["parameter"].map(_S3_PARAMS)
        raw["value"] = [
            _to_canonical(v, c, u)
            for v, c, u in zip(raw["value"], raw["canon"], raw["units"], strict=False)
        ]
        raw = raw.dropna(subset=["value"])
        raw["ts"] = raw["ts"].dt.floor("h")
        hourly = (
            raw.groupby(["canon", "ts"], as_index=False)["value"].mean()
            .rename(columns={"canon": "pollutant"})
        )
        hourly["station_id"] = st.id
        hourly["source"] = "openaq:s3"
        frames.append(hourly[OBS_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=OBS_COLUMNS), SourceResult(
            "openaq:s3", ok=False, message=f"no archive data ({n_files} files) {start}..{end}")
    obs = merge_observations(*frames)
    return obs, SourceResult("openaq:s3", ok=True, rows=len(obs),
                             message=f"{obs['station_id'].nunique()} stations, "
                                     f"{n_files} day-files, {start}..{end}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ingest.openaq", description=__doc__)
    ap.add_argument("--map", action="store_true", help="(re)build the station->location map")
    ap.add_argument("--s3", action="store_true", help="use the S3 archive (default; no key/limit)")
    ap.add_argument("--api", action="store_true", help="use the v3 API /hours path instead")
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--end", default="2025-03-01")
    ap.add_argument("--stations", help="comma-separated station ids subset")
    ap.add_argument("--append", action="store_true",
                    help="merge into obs_history.parquet instead of overwriting")
    args = ap.parse_args(argv)
    SETTINGS.ensure_dirs()

    if args.map:
        m = build_station_map()
        ok = sum(1 for v in m.values() if v.get("sensors"))
        print(f"mapped {ok}/{len(m)} stations -> {_MAP_PATH}")
        return

    sts = load_stations()
    if args.stations:
        keep = set(args.stations.split(","))
        sts = [s for s in sts if s.id in keep]
    s, e = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    if args.api:
        df, res = fetch_history_openaq(sts, s, e)
    else:
        df, res = fetch_history_s3(sts, s, e)
    print(res.as_dict())
    if df.empty:
        return
    out = SETTINGS.processed_dir / "obs_history.parquet"
    if args.append and out.exists():
        df = merge_observations(pd.read_parquet(out), df)
    write_table(df, out)
    print(f"obs_history.parquet now {len(df)} rows, {df['station_id'].nunique()} stations, "
          f"pollutants {sorted(df['pollutant'].unique())}")


if __name__ == "__main__":
    main()
