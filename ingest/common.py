"""Shared ingestion utilities: HTTP with retry, canonical schemas, Parquet + snapshot I/O.

All timestamps in stored tables are timezone-aware UTC (``datetime64[ns, UTC]``).
Delhi local time (IST) is used only for display / calendar features downstream.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import SETTINGS

IST = ZoneInfo("Asia/Kolkata")
UTC = dt.UTC

USER_AGENT = "VayuCast/0.1 (SIH2026 PS26082; research/non-commercial)"

# --- canonical table schemas -------------------------------------------------
OBS_COLUMNS = ["station_id", "ts", "pollutant", "value", "source"]

MET_COLUMNS = [
    "station_id", "ts", "kind", "lead_h",
    "t2m", "d2m", "rh2m",
    "wind_speed10", "wind_dir10", "wind_u10", "wind_v10",
    "surface_pressure", "precip", "solar", "cloud", "blh",
    "t1000", "t925", "t850", "wind_u850", "wind_v850",
    "source",
]

FIRE_COLUMNS = [
    "date", "cluster_id", "lat", "lon", "frp_sum", "count", "confidence_mean", "source"
]


@dataclass
class SourceResult:
    """Outcome of one ingestion call, for the orchestrator to log and the API to expose."""

    source: str
    ok: bool
    rows: int = 0
    stale: bool = False
    path: str | None = None
    message: str = ""
    fetched_at: str = field(default_factory=lambda: dt.datetime.now(UTC).isoformat())

    def as_dict(self) -> dict:
        return asdict(self)


# --- HTTP ------------------------------------------------------------------
class HttpError(RuntimeError):
    pass


class RateLimited(HttpError):
    """HTTP 429 — separated so callers can wait much longer for shared-IP throttling."""


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((requests.RequestException, HttpError)),
    reraise=True,
)
def _http_get_inner(url: str, *, params, headers, timeout) -> requests.Response:
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    resp = requests.get(url, params=params, headers=hdrs, timeout=timeout)
    if resp.status_code == 429:
        raise RateLimited(f"429 from {url}", resp.headers.get("Retry-After"))
    if resp.status_code >= 500:
        raise HttpError(f"{resp.status_code} from {url}")
    resp.raise_for_status()
    return resp


def http_get(url: str, *, params: dict | None = None, headers: dict | None = None,
             timeout: int = 45, rate_limit_waits: tuple[int, ...] = (30, 75, 150, 300)) -> requests.Response:
    """GET with exponential-backoff retry on 5xx/network errors, plus a longer,
    separate wait schedule for 429 (Colab and other shared IPs get throttled hard)."""
    for wait in (*rate_limit_waits, 0):
        try:
            return _http_get_inner(url, params=params, headers=headers, timeout=timeout)
        except RateLimited as exc:
            if wait == 0:
                raise
            hint = exc.args[1] if len(exc.args) > 1 and exc.args[1] else None
            delay = int(hint) if hint and str(hint).isdigit() else wait
            time.sleep(delay)
    raise RateLimited(f"429 from {url} after {len(rate_limit_waits)} long waits")  # pragma: no cover


def get_json(url: str, *, params: dict | None = None, headers: dict | None = None,
             timeout: int = 45) -> dict | list:
    resp = http_get(url, params=params, headers=headers, timeout=timeout)
    try:
        return resp.json()
    except json.JSONDecodeError as exc:  # pragma: no cover - upstream returned non-JSON
        raise HttpError(f"non-JSON response from {url}: {exc}") from exc


# --- text / matching -----------------------------------------------------
_WS = re.compile(r"[^a-z0-9]+")


def normalize_name(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace — for fuzzy matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return _WS.sub(" ", text.lower()).strip()


# --- Parquet + snapshot I/O -------------------------------------------------
def write_table(df: pd.DataFrame, path: Path | str) -> Path:
    """Write a DataFrame to Parquet, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_table(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


def snapshot_path(name: str) -> Path:
    return SETTINGS.snapshots_dir / f"{name}.parquet"


def save_snapshot(name: str, df: pd.DataFrame) -> Path:
    """Persist a 'last known good' copy used when a live source is unavailable."""
    return write_table(df, snapshot_path(name))


def load_snapshot(name: str) -> pd.DataFrame | None:
    p = snapshot_path(name)
    if p.exists():
        return pd.read_parquet(p)
    return None


def merge_observations(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate observation frames and de-duplicate on (station_id, ts, pollutant).

    Later frames win on conflict (pass freshest last).
    """
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame(columns=OBS_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["station_id", "ts", "pollutant"])
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = out.drop_duplicates(subset=["station_id", "ts", "pollutant"], keep="last")
    return out.sort_values(["station_id", "ts", "pollutant"])[OBS_COLUMNS].reset_index(drop=True)


def to_utc(series: pd.Series, assume_tz: ZoneInfo = IST) -> pd.Series:
    """Parse a timestamp series to UTC, assuming ``assume_tz`` for naive values."""
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(assume_tz, ambiguous="NaT", nonexistent="shift_forward")
    return s.dt.tz_convert("UTC")


def empty_obs() -> pd.DataFrame:
    return pd.DataFrame(columns=OBS_COLUMNS)


# Physically plausible ranges (ug/m3; CO in mg/m3). CPCB feeds use -999/-9999 as
# missing sentinels and occasionally emit sensor-fault spikes.
PLAUSIBLE_RANGE = {
    "PM2.5": (1.0, 1500.0),
    "PM10": (1.0, 3000.0),
    "NO2": (0.1, 800.0),
    "O3": (0.1, 900.0),
    "SO2": (0.1, 2000.0),
    "CO": (0.01, 60.0),
    "NH3": (0.1, 3000.0),
}


def sanitize_observations(obs: pd.DataFrame) -> pd.DataFrame:
    """Drop out-of-range / sentinel pollutant values (keeps the long OBS schema)."""
    if obs.empty:
        return obs
    o = obs.copy()
    o["value"] = pd.to_numeric(o["value"], errors="coerce")
    lo = o["pollutant"].map(lambda p: PLAUSIBLE_RANGE.get(p, (None, None))[0])
    hi = o["pollutant"].map(lambda p: PLAUSIBLE_RANGE.get(p, (None, None))[1])
    keep = o["value"].notna()
    keep &= lo.isna() | (o["value"] >= lo)
    keep &= hi.isna() | (o["value"] <= hi)
    return o[keep].reset_index(drop=True)
