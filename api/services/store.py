"""Optional Postgres mirror of the live pipeline state.

When ``DATABASE_URL`` is set, every successful refresh upserts the current
observations, the 72-hour forecast, the active alerts and a per-station rollup
into Postgres — so the data survives restarts and is queryable from psql,
Grafana or any BI tool. The API keeps serving the map/forecast from its
in-memory cache; Postgres is a mirror, never on that read path. The one reader
is ``GET /api/history/{id}``, which pages back through accumulated observations.

Every function degrades to a no-op (returns ``None`` / ``False``) when the URL
is missing, SQLAlchemy/psycopg is not installed, or the database is unreachable.
A database problem must never break a refresh.
"""

from __future__ import annotations

import datetime as dt
import functools
import logging

import pandas as pd

from config.settings import SETTINGS, station_index

log = logging.getLogger("vayucast.store")

# CREATE ... IF NOT EXISTS — safe to run on every process start.
_DDL = """
CREATE TABLE IF NOT EXISTS observations (
    station_id  TEXT        NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    pollutant   TEXT        NOT NULL,
    value       DOUBLE PRECISION,
    source      TEXT,
    PRIMARY KEY (station_id, ts, pollutant)
);
CREATE INDEX IF NOT EXISTS observations_ts_idx ON observations (ts DESC);

CREATE TABLE IF NOT EXISTS forecasts (
    issued_ts          TIMESTAMPTZ NOT NULL,
    station_id         TEXT        NOT NULL,
    valid_ts           TIMESTAMPTZ NOT NULL,
    horizon_h          INTEGER     NOT NULL,
    pm25_p10           DOUBLE PRECISION,
    pm25_p50           DOUBLE PRECISION,
    pm25_p90           DOUBLE PRECISION,
    pm10_p50           DOUBLE PRECISION,
    no2_p50            DOUBLE PRECISION,
    aqi                INTEGER,
    category           TEXT,
    dominant_pollutant TEXT,
    PRIMARY KEY (issued_ts, station_id, valid_ts)
);
CREATE INDEX IF NOT EXISTS forecasts_station_valid_idx ON forecasts (station_id, valid_ts DESC);

CREATE TABLE IF NOT EXISTS alerts (
    issued_ts    TIMESTAMPTZ NOT NULL,
    station_id   TEXT        NOT NULL,
    level        TEXT        NOT NULL,
    lead_hours   INTEGER,
    valid_ts     TIMESTAMPTZ,
    aqi          INTEGER,
    station_name TEXT,
    PRIMARY KEY (issued_ts, station_id, level)
);

CREATE TABLE IF NOT EXISTS stations (
    station_id      TEXT PRIMARY KEY,
    name            TEXT,
    city            TEXT,
    site_type       TEXT,
    lat             DOUBLE PRECISION,
    lon             DOUBLE PRECISION,
    latest_aqi      INTEGER,
    latest_category TEXT,
    updated_at      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS refresh_log (
    id           BIGSERIAL PRIMARY KEY,
    refreshed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_name   TEXT,
    n_stations   INTEGER,
    n_obs        INTEGER,
    n_forecast   INTEGER,
    n_alerts     INTEGER
);
"""

# Only mirror a trailing window of observations each cycle — older rows are
# already in Postgres from earlier runs.
_OBS_MIRROR_DAYS = 30

_schema_ready = False


@functools.lru_cache(maxsize=1)
def _engine():
    url = SETTINGS.database_url
    if not url:
        return None
    try:
        from sqlalchemy import create_engine

        return create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=2, future=True)
    except Exception as exc:  # noqa: BLE001 - missing driver / bad URL -> disabled
        log.warning("store: engine unavailable, Postgres mirror disabled (%s)", exc)
        return None


def _safe_url() -> str:
    try:
        from sqlalchemy.engine import make_url

        return make_url(SETTINGS.database_url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001
        return "<database>"


def enabled() -> bool:
    """True when a usable engine is configured (not whether the server is up)."""
    return _engine() is not None


def init_schema() -> bool:
    eng = _engine()
    if eng is None:
        return False
    try:
        from sqlalchemy import text

        with eng.begin() as cx:
            for stmt in (s.strip() for s in _DDL.split(";")):
                if stmt:
                    cx.execute(text(stmt))
        log.info("store: schema ready at %s", _safe_url())
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("store: schema init failed (%s)", exc)
        return False


def _ensure_schema() -> bool:
    global _schema_ready
    if not _schema_ready:
        _schema_ready = init_schema()
    return _schema_ready


# --- frame prep ------------------------------------------------------------
def _prep_observations(obs: pd.DataFrame) -> pd.DataFrame:
    if obs is None or obs.empty:
        return pd.DataFrame()
    o = obs.copy()
    o["ts"] = pd.to_datetime(o["ts"], utc=True)
    if "source" not in o:
        o["source"] = None
    cut = o["ts"].max() - pd.Timedelta(days=_OBS_MIRROR_DAYS)
    o = o[o["ts"] >= cut]
    o = o[["station_id", "ts", "pollutant", "value", "source"]]
    o = o.dropna(subset=["station_id", "ts", "pollutant"])
    return o.drop_duplicates(["station_id", "ts", "pollutant"], keep="last").reset_index(drop=True)


def _prep_forecasts(fc: pd.DataFrame) -> pd.DataFrame:
    if fc is None or fc.empty:
        return pd.DataFrame()
    f = fc.rename(columns={"horizon": "horizon_h"}).copy()
    for c in ("issued_ts", "valid_ts"):
        f[c] = pd.to_datetime(f[c], utc=True)
    for c in ("pm10_p50", "no2_p50", "dominant_pollutant"):
        if c not in f:
            f[c] = None
    for c in ("pm25_p10", "pm25_p50", "pm25_p90", "pm10_p50", "no2_p50"):
        f[c] = pd.to_numeric(f[c], errors="coerce").astype("float64")
    f["aqi"] = pd.to_numeric(f["aqi"], errors="coerce").astype("Int64")
    cols = [
        "issued_ts", "station_id", "valid_ts", "horizon_h",
        "pm25_p10", "pm25_p50", "pm25_p90", "pm10_p50", "no2_p50",
        "aqi", "category", "dominant_pollutant",
    ]
    f = f[cols].dropna(subset=["issued_ts", "station_id", "valid_ts"])
    return f.drop_duplicates(["issued_ts", "station_id", "valid_ts"], keep="last").reset_index(drop=True)


def _prep_alerts(alerts: list[dict], issued_ts: str | None) -> pd.DataFrame:
    if not alerts:
        return pd.DataFrame()
    iso = pd.to_datetime(issued_ts, utc=True) if issued_ts else pd.Timestamp.now(tz="UTC")
    rows = [
        {
            "issued_ts": iso,
            "station_id": a["station_id"],
            "level": a["level"],
            "lead_hours": a.get("lead_hours"),
            "valid_ts": pd.to_datetime(a.get("valid_ts"), utc=True) if a.get("valid_ts") else None,
            "aqi": a.get("aqi"),
            "station_name": a.get("name"),
        }
        for a in alerts
    ]
    df = pd.DataFrame(rows)
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").astype("Int64")
    return df.drop_duplicates(["issued_ts", "station_id", "level"], keep="last")


def _prep_stations(state) -> pd.DataFrame:
    idx = station_index()
    latest: dict[str, dict] = {}
    try:
        wide = state.latest_obs_wide()
        if not wide.empty:
            from aqi.cpcb_aqi import compute_aqi

            for _, r in wide.iterrows():
                concs = {
                    p: r[p]
                    for p in ("PM2.5", "PM10", "NO2", "O3", "SO2", "CO")
                    if p in r and pd.notna(r[p])
                }
                res = compute_aqi(concs, enforce_min_pollutants=False)
                latest[r["station_id"]] = {"aqi": res.aqi, "category": res.category}
    except Exception:  # noqa: BLE001
        latest = {}

    now = pd.Timestamp.now(tz="UTC")
    rows = [
        {
            "station_id": s.id, "name": s.name, "city": s.city, "site_type": s.site_type,
            "lat": s.lat, "lon": s.lon,
            "latest_aqi": latest.get(s.id, {}).get("aqi"),
            "latest_category": latest.get(s.id, {}).get("category"),
            "updated_at": now,
        }
        for s in idx.values()
    ]
    df = pd.DataFrame(rows)
    df["latest_aqi"] = pd.to_numeric(df["latest_aqi"], errors="coerce").astype("Int64")
    return df


# --- write path ----------------------------------------------------------
def _merge_via_staging(cx, target: str, df: pd.DataFrame, keys: list[str], *, update: bool = True) -> int:
    """Load ``df`` into a temp table shaped exactly like ``target``, then
    INSERT ... SELECT ... ON CONFLICT.

    Staging (rather than one multi-row INSERT) avoids the 65k parameter bound the
    ~28k-row forecast frame would blow; cloning the target's column types avoids
    the dtype mismatch an all-NULL column gets when pandas creates the table.
    """
    if df is None or df.empty:
        return 0
    stg = f"_stg_{target}"
    cx.exec_driver_sql(f"CREATE TEMP TABLE {stg} (LIKE {target}) ON COMMIT DROP")
    df.to_sql(stg, cx, if_exists="append", index=False)
    cols = ", ".join(df.columns)
    if update:
        sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in df.columns if c not in keys)
        clause = f"DO UPDATE SET {sets}" if sets else "DO NOTHING"
    else:
        clause = "DO NOTHING"
    cx.exec_driver_sql(
        f"INSERT INTO {target} ({cols}) SELECT {cols} FROM {stg} "
        f"ON CONFLICT ({', '.join(keys)}) {clause}"
    )
    return len(df)


def write_refresh(state) -> dict | None:
    """Mirror the current pipeline state into Postgres. No-op / None if disabled."""
    eng = _engine()
    if eng is None or not _ensure_schema():
        return None
    try:
        obs = _prep_observations(state.observations)
        fc = _prep_forecasts(state.forecast)
        al = _prep_alerts(state.alerts, state.last_refresh)
        stn = _prep_stations(state)
        with eng.begin() as cx:
            n_obs = _merge_via_staging(cx, "observations", obs, ["station_id", "ts", "pollutant"])
            n_fc = _merge_via_staging(
                cx, "forecasts", fc, ["issued_ts", "station_id", "valid_ts"], update=False
            )
            n_al = _merge_via_staging(cx, "alerts", al, ["issued_ts", "station_id", "level"])
            n_stn = _merge_via_staging(cx, "stations", stn, ["station_id"])
            cx.exec_driver_sql(
                "INSERT INTO refresh_log (model_name, n_stations, n_obs, n_forecast, n_alerts) "
                "VALUES (%s, %s, %s, %s, %s)",
                (state.model_name, n_stn, n_obs, n_fc, n_al),
            )
        out = {"observations": n_obs, "forecasts": n_fc, "alerts": n_al, "stations": n_stn}
        log.info("store: mirrored refresh -> %s", out)
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("store: write_refresh failed (%s)", exc)
        return None


# --- read path ---------------------------------------------------------
def observation_history(station_id: str, hours: int) -> pd.DataFrame | None:
    """Pull observation history for one station from Postgres. None if disabled/empty."""
    eng = _engine()
    if eng is None:
        return None
    try:
        since = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours)
        from sqlalchemy import text

        with eng.connect() as cx:
            df = pd.read_sql(
                text(
                    "SELECT ts, pollutant, value FROM observations "
                    "WHERE station_id = :sid AND ts >= :since ORDER BY ts"
                ),
                cx,
                params={"sid": station_id, "since": since},
            )
        return df if not df.empty else None
    except Exception as exc:  # noqa: BLE001
        log.warning("store: observation_history failed (%s)", exc)
        return None
