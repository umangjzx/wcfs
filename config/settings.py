"""Central configuration: paths, geographic constants, physical parameters, secrets.

Import ``SETTINGS`` for runtime config and ``load_stations()`` for the station registry.
Nothing here performs I/O at import time except reading environment variables.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
STATIONS_FILE = CONFIG_DIR / "stations.yaml"


class Settings(BaseSettings):
    """Environment-driven settings. Missing keys are allowed (offline fallback)."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_gov_in_api_key: str | None = None
    firms_map_key: str | None = None
    openaq_api_key: str | None = None
    cdsapi_url: str | None = None
    cdsapi_key: str | None = None
    open_meteo_api_key: str | None = None
    vayucast_data_dir: Path = REPO_ROOT / "data"

    # Optional Postgres mirror of the live pipeline state. Unset -> the API keeps
    # serving from its in-memory cache and DB writes are skipped.
    # e.g. postgresql+psycopg://vayucast:vayucast@db:5432/vayucast
    database_url: str | None = None

    # --- derived paths -------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return Path(self.vayucast_data_dir)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.interim_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()


# ---------------------------------------------------------------------------
# Geographic + physical constants for Delhi NCR
# ---------------------------------------------------------------------------

# Approx geometric centre of the built-up NCR core (near Connaught Place).
DELHI_CENTROID = (28.6139, 77.2090)  # (lat, lon)

# Bounding box for pulling station-level met/air data (lat_min, lon_min, lat_max, lon_max).
NCR_BBOX = (27.9, 76.5, 29.2, 77.9)

# Wider box that captures Punjab + Haryana stubble-burning belt for FIRMS.
STUBBLE_BBOX = (28.0, 73.5, 32.2, 78.2)

# Cities included in "Delhi NCR" for this project.
NCR_CITIES = ["Delhi", "Gurugram", "Faridabad", "Ghaziabad", "Noida", "Greater Noida"]

# Forecast configuration.
FORECAST_HORIZON_HOURS = 72
FORECAST_STEP_HOURS = 1
QUANTILES = (0.1, 0.5, 0.9)

# Stubble-plume transport model.
PLUME_DECAY_TAU_HOURS = 18.0  # e-folding time for advected biomass-burning influence
PLUME_MIN_INBOUND_SPEED_MS = 0.5  # floor on projected wind speed to avoid divide-by-zero

# Indian National AQI category breakpoints (overall AQI value -> label).
AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]

# Thresholds used for alerting / event-detection scoring.
ALERT_THRESHOLDS = {"Very Poor": 301, "Severe": 401}

# Stubble-burning season window (month, day) inclusive.
STUBBLE_SEASON = ((9, 25), (11, 30))


@dataclass(frozen=True)
class Station:
    """One air-quality monitoring station in the NCR registry."""

    id: str
    name: str
    city: str
    agency: str
    lat: float
    lon: float
    site_type: str  # traffic | industrial | residential | background | mixed
    cpcb_name: str  # name as returned by the CPCB / data.gov.in feed, for matching
    coords_verified: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)


@functools.lru_cache(maxsize=1)
def load_stations() -> list[Station]:
    """Load and validate the station registry from ``config/stations.yaml``."""
    with STATIONS_FILE.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    stations: list[Station] = []
    seen_ids: set[str] = set()
    for entry in raw["stations"]:
        st = Station(
            id=entry["id"],
            name=entry["name"],
            city=entry["city"],
            agency=entry.get("agency", "Unknown"),
            lat=float(entry["lat"]),
            lon=float(entry["lon"]),
            site_type=entry.get("site_type", "mixed"),
            cpcb_name=entry.get("cpcb_name", entry["name"]),
            coords_verified=bool(entry.get("coords_verified", False)),
            tags=tuple(entry.get("tags", []) or []),
        )
        if st.id in seen_ids:
            raise ValueError(f"Duplicate station id in stations.yaml: {st.id}")
        seen_ids.add(st.id)
        if st.city not in NCR_CITIES:
            raise ValueError(f"Station {st.id} has city '{st.city}' outside NCR_CITIES")
        stations.append(st)

    if not stations:
        raise ValueError("stations.yaml contains no stations")
    return stations


def station_index() -> dict[str, Station]:
    """Map station id -> Station."""
    return {s.id: s for s in load_stations()}
