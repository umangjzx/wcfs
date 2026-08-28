"""Pydantic response models for the VayuCast API."""

from __future__ import annotations

from pydantic import BaseModel


class StationOut(BaseModel):
    id: str
    name: str
    city: str
    agency: str
    lat: float
    lon: float
    site_type: str
    latest_aqi: int | None = None
    latest_category: str | None = None
    latest_pm25: float | None = None
    latest_ts: str | None = None


class ForecastPoint(BaseModel):
    valid_ts: str
    horizon: int
    pm25_p10: float
    pm25_p50: float
    pm25_p90: float
    aqi: int | None
    category: str


class StationForecast(BaseModel):
    station_id: str
    name: str
    issued_ts: str | None
    dominant_pollutant: str
    advisory: str
    points: list[ForecastPoint]


class ObservationSeries(BaseModel):
    station_id: str
    pollutant: str
    ts: list[str]
    value: list[float | None]


class GridCell(BaseModel):
    lat: float
    lon: float
    aqi: float


class GridOut(BaseModel):
    valid_ts: str
    horizon: int
    bounds: list[float]  # [lat_min, lon_min, lat_max, lon_max]
    cells: list[GridCell]


class DriverGroup(BaseModel):
    group: str
    contribution: float


class DriverFeature(BaseModel):
    feature: str
    group: str
    importance: float


class StationDrivers(BaseModel):
    station_id: str
    isi: float | None = None
    isi_components: dict[str, float] = {}
    incoming_stubble_load: float | None = None
    plume_from_bearing_deg: float | None = None
    ventilation_index: float | None = None
    groups: list[DriverGroup] = []
    top_features: list[DriverFeature] = []


class FireCluster(BaseModel):
    lat: float
    lon: float
    frp_sum: float
    count: int
    date: str


class FiresOut(BaseModel):
    as_of: str
    clusters: list[FireCluster]
    plume_vector: dict[str, float]  # {u, v, from_bearing_deg, incoming_load}


class Alert(BaseModel):
    station_id: str
    name: str
    level: str
    lead_hours: int
    valid_ts: str
    aqi: int


class ModelCard(BaseModel):
    model: str
    trained_on: str | None
    training_rows: int | None
    horizons: list[int]
    backtest: dict
    data_sources: list[str]
    limitations: list[str]
    wrfchem_validation: str


class Health(BaseModel):
    status: str
    model_loaded: bool
    last_refresh: str | None
    sources: list[dict]
    stale: bool
