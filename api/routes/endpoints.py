"""All VayuCast API routes on one router (small surface, keeps the hackathon build lean)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.services.pipeline import STATE, refresh
from config.settings import NCR_BBOX, SETTINGS, load_stations, station_index

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    try:
        from api.services.store import enabled as _store_enabled
        store_on = _store_enabled()
    except Exception:  # noqa: BLE001
        store_on = False
    return {
        "status": "ok",
        "model_loaded": STATE.model_name not in ("none", "naive"),
        "model_name": STATE.model_name,
        "last_refresh": STATE.last_refresh,
        "stale": STATE.stale,
        "sources": STATE.sources,
        "postgres_mirror": store_on,
        "stations_in_forecast": int(STATE.forecast["station_id"].nunique()) if not STATE.forecast.empty else 0,
    }


@router.post("/refresh")
def force_refresh(ingest: bool = True):
    return refresh(do_ingest=ingest)


@router.get("/stations")
def stations():
    wide = STATE.latest_obs_wide()
    latest = {}
    if not wide.empty:
        for _, r in wide.iterrows():
            concs = {p: r[p] for p in ("PM2.5", "PM10", "NO2", "O3", "SO2", "CO")
                     if p in r and pd.notna(r[p])}
            from aqi.cpcb_aqi import compute_aqi
            res = compute_aqi(concs, enforce_min_pollutants=False)
            latest[r["station_id"]] = {
                "aqi": res.aqi, "category": res.category,
                "pm25": float(r["PM2.5"]) if "PM2.5" in r and pd.notna(r["PM2.5"]) else None,
                "ts": pd.Timestamp(r["_ts"]).isoformat() if pd.notna(r.get("_ts")) else None,
            }
    out = []
    for s in load_stations():
        lv = latest.get(s.id, {})
        out.append({
            "id": s.id, "name": s.name, "city": s.city, "agency": s.agency,
            "lat": s.lat, "lon": s.lon, "site_type": s.site_type,
            "latest_aqi": lv.get("aqi"), "latest_category": lv.get("category"),
            "latest_pm25": lv.get("pm25"), "latest_ts": lv.get("ts"),
        })
    return out


@router.get("/observations/{station_id}")
def observations(station_id: str, hours: int = Query(72, ge=1, le=336)):
    if STATE.observations.empty:
        return {"station_id": station_id, "series": []}
    o = STATE.observations
    o = o[o["station_id"] == station_id].copy()
    if o.empty:
        raise HTTPException(404, f"no observations for {station_id}")
    o["ts"] = pd.to_datetime(o["ts"], utc=True)
    cut = o["ts"].max() - pd.Timedelta(hours=hours)
    o = o[o["ts"] >= cut]
    series = []
    for pol, g in o.groupby("pollutant"):
        g = g.sort_values("ts")
        series.append({"pollutant": pol,
                       "ts": [t.isoformat() for t in g["ts"]],
                       "value": [None if pd.isna(v) else round(float(v), 2) for v in g["value"]]})
    return {"station_id": station_id, "series": series}


@router.get("/history/{station_id}")
def history(station_id: str, hours: int = Query(168, ge=1, le=8760)):
    """Observation history for one station. Served from Postgres when the mirror is
    enabled (retains far more than the in-memory cache), else from the live cache."""
    if station_index().get(station_id) is None:
        raise HTTPException(404, f"unknown station {station_id}")
    try:
        from api.services.store import observation_history
        h = observation_history(station_id, hours)
    except Exception:  # noqa: BLE001
        h = None

    if h is not None and not h.empty:
        h["ts"] = pd.to_datetime(h["ts"], utc=True)
        series = []
        for pol, g in h.groupby("pollutant"):
            g = g.sort_values("ts")
            series.append({"pollutant": pol,
                           "ts": [t.isoformat() for t in g["ts"]],
                           "value": [None if pd.isna(v) else round(float(v), 2) for v in g["value"]]})
        return {"station_id": station_id, "source": "postgres", "series": series}

    # fall back to the in-memory observation cache
    try:
        return {**observations(station_id, min(hours, 336)), "source": "cache"}
    except HTTPException:
        return {"station_id": station_id, "source": "cache", "series": []}


@router.get("/forecast/{station_id}")
def forecast(station_id: str):
    if STATE.forecast.empty:
        raise HTTPException(503, "forecast not ready — call POST /api/refresh")
    f = STATE.forecast[STATE.forecast["station_id"] == station_id].sort_values("horizon")
    if f.empty:
        raise HTTPException(404, f"no forecast for {station_id}")
    st = station_index().get(station_id)
    return {
        "station_id": station_id,
        "name": st.name if st else station_id,
        "issued_ts": pd.Timestamp(f["issued_ts"].iloc[0]).isoformat(),
        "dominant_pollutant": "PM2.5",
        "advisory": f["advisory"].iloc[0] if "advisory" in f else "",
        "points": [{
            "valid_ts": pd.Timestamp(r.valid_ts).isoformat(), "horizon": int(r.horizon),
            "pm25_p10": round(float(r.pm25_p10), 1), "pm25_p50": round(float(r.pm25_p50), 1),
            "pm25_p90": round(float(r.pm25_p90), 1),
            "aqi": int(r.aqi) if pd.notna(r.aqi) else None,
            "category": r.category,
        } for r in f.itertuples()],
    }


@router.get("/grid")
def grid(horizon: int = Query(0, ge=0, le=72)):
    return STATE.grid_at(horizon)


@router.get("/drivers/{station_id}")
def drivers(station_id: str):
    d = STATE.drivers.get(station_id)
    if d is None:
        raise HTTPException(404, f"no drivers for {station_id} (model may be running the naive fallback)")
    return {
        "station_id": station_id,
        "isi": d.get("isi"),
        "isi_components": d.get("isi_components", {}),
        "incoming_stubble_load": d.get("incoming_stubble_load"),
        "plume_from_bearing_deg": d.get("plume_from_bearing_deg"),
        "ventilation_index": d.get("ventilation_index"),
        "groups": [{"group": k, "contribution": v} for k, v in d.get("groups", {}).items()],
        "top_features": d.get("top_features", []),
    }


@router.get("/fires")
def fires():
    df = STATE.fires
    if df.empty:
        return {"as_of": STATE.last_refresh, "clusters": [], "plume_vector": {}}
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    recent = d[d["date"] >= d["date"].max() - pd.Timedelta(days=2)]
    clusters = [{"lat": round(float(r["lat"]), 3), "lon": round(float(r["lon"]), 3),
                 "frp_sum": round(float(r["frp_sum"]), 1), "count": int(r["count"]),
                 "date": pd.Timestamp(r["date"]).date().isoformat()}
                for r in recent.to_dict("records")]
    # aggregate plume vector from the drivers cache (mean over stations)
    bs, ls = [], []
    for dd in STATE.drivers.values():
        if dd.get("plume_from_bearing_deg") is not None:
            bs.append(dd["plume_from_bearing_deg"])
        if dd.get("incoming_stubble_load") is not None:
            ls.append(dd["incoming_stubble_load"])
    plume = {}
    if bs:
        ang = np.deg2rad(np.array(bs))
        plume = {"from_bearing_deg": round(float(np.rad2deg(np.arctan2(
            np.sin(ang).mean(), np.cos(ang).mean())) % 360), 1),
            "incoming_load": round(float(np.mean(ls)) if ls else 0.0, 1)}
    return {"as_of": STATE.last_refresh, "clusters": clusters, "plume_vector": plume}


@router.get("/alerts")
def alerts():
    return {"as_of": STATE.last_refresh, "alerts": STATE.alerts}


@router.get("/model-card")
def model_card():
    reg = SETTINGS.processed_dir.parent.parent / "models" / "registry"
    backtest = {}
    bt = reg / "backtest_metrics.json"
    if bt.exists():
        try:
            backtest = json.loads(bt.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            backtest = {}
    horizons = []
    meta = reg / "lgbm_meta.json"
    if meta.exists():
        try:
            horizons = json.loads(meta.read_text(encoding="utf-8")).get("horizons", [])
        except Exception:  # noqa: BLE001
            pass
    return {
        "model": STATE.model_name,
        "horizons": horizons or list(range(1, 73)),
        "backtest": {k: backtest.get(k) for k in ("overall", "by_horizon", "events") if k in backtest},
        "data_sources": [
            "CPCB real-time AQI (data.gov.in)", "ERA5 + GFS meteorology (Open-Meteo)",
            "NASA FIRMS fire hotspots", "OpenAQ v3 / S3 archive (historical ground truth)",
        ],
        "limitations": [
            "Live 72 h forecasts come from a fast ML emulator, not a live WRF-Chem run.",
            "Historical 925/850 hPa winds are degraded for older training seasons.",
            "AQI is derived from the PM2.5 forecast (dominant pollutant in Delhi winter).",
        ],
        "wrfchem_validation": (
            "Offline WRF-Chem run over a Delhi-NCR nest for a historical stubble-burning "
            "spike, validated against CPCB observations — see wrfchem/validate.ipynb."
        ),
        "region_bounds": list(NCR_BBOX),
    }
