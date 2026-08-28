"""The live forecast pipeline + in-memory state the API serves.

refresh():  ingest (CPCB now + GFS forecast + FIRMS) -> serving feature matrix ->
            model forecast -> AQI grid + per-station drivers + alerts -> cache + snapshot.

Every stage degrades gracefully; a failed refresh keeps the last good state and flags it stale.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import SETTINGS, station_index
from ingest.common import read_table

_UTC = dt.UTC


@dataclass
class PipelineState:
    forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    observations: pd.DataFrame = field(default_factory=pd.DataFrame)
    drivers: dict[str, dict] = field(default_factory=dict)
    fires: pd.DataFrame = field(default_factory=pd.DataFrame)
    alerts: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    model_name: str = "none"
    last_refresh: str | None = None
    stale: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)

    # ---- derived helpers -------------------------------------------------
    def latest_obs_wide(self) -> pd.DataFrame:
        if self.observations.empty:
            return pd.DataFrame()
        o = self.observations.copy()
        o["ts"] = pd.to_datetime(o["ts"], utc=True)
        last = o.sort_values("ts").groupby(["station_id", "pollutant"]).tail(1)
        wide = last.pivot_table(index="station_id", columns="pollutant", values="value").reset_index()
        wide["_ts"] = last.groupby("station_id")["ts"].max().values
        return wide

    def grid_at(self, horizon: int) -> dict:
        from api.services.grid import idw_grid

        if self.forecast.empty:
            return {"bounds": [], "cells": [], "valid_ts": None, "horizon": horizon}
        idx = station_index()
        f = self.forecast
        near = f.iloc[(f["horizon"] - horizon).abs().argsort()]
        pick_h = int(near["horizon"].iloc[0])
        sub = f[f["horizon"] == pick_h]
        lats = np.array([idx[s].lat for s in sub["station_id"] if s in idx])
        lons = np.array([idx[s].lon for s in sub["station_id"] if s in idx])
        vals = sub[sub["station_id"].isin(idx)]["aqi"].astype("float64").to_numpy()
        g = idw_grid(lats, lons, vals)
        g["valid_ts"] = sub["valid_ts"].iloc[0].isoformat() if len(sub) else None
        g["horizon"] = pick_h
        return g


STATE = PipelineState()


# ---------------------------------------------------------------------------
def _load_model():
    from models.baseline_lgbm import REGISTRY, MultiForecaster

    if (REGISTRY / "lgbm_pm25_meta.json").exists():
        try:
            return MultiForecaster.load(), "lgbm"
        except Exception:  # noqa: BLE001
            pass
    return None, "naive"


_REPO_ROOT = SETTINGS.data_dir.parent


def _snapshot_paths(read: bool = False):
    d = SETTINGS.snapshots_dir
    # for reads, fall back to the committed demo seed if no live snapshot exists yet
    if read and not (d / "api_meta.json").exists() and (_REPO_ROOT / "demo" / "snapshot" / "api_meta.json").exists():
        d = _REPO_ROOT / "demo" / "snapshot"
    return {
        "forecast": d / "api_forecast.parquet",
        "observations": d / "api_obs.parquet",
        "fires": d / "api_fires.parquet",
        "meta": d / "api_meta.json",
    }


def _save_snapshot() -> None:
    p = _snapshot_paths()
    SETTINGS.snapshots_dir.mkdir(parents=True, exist_ok=True)
    if not STATE.forecast.empty:
        STATE.forecast.to_parquet(p["forecast"], index=False)
    if not STATE.observations.empty:
        STATE.observations.to_parquet(p["observations"], index=False)
    if not STATE.fires.empty:
        STATE.fires.to_parquet(p["fires"], index=False)
    p["meta"].write_text(json.dumps({
        "drivers": STATE.drivers, "alerts": STATE.alerts, "sources": STATE.sources,
        "model_name": STATE.model_name, "last_refresh": STATE.last_refresh,
    }, indent=2), encoding="utf-8")


def load_snapshot() -> bool:
    p = _snapshot_paths(read=True)
    if not p["meta"].exists():
        return False
    try:
        meta = json.loads(p["meta"].read_text(encoding="utf-8"))
        with STATE.lock:
            if p["forecast"].exists():
                STATE.forecast = pd.read_parquet(p["forecast"])
            if p["observations"].exists():
                STATE.observations = pd.read_parquet(p["observations"])
            if p["fires"].exists():
                STATE.fires = pd.read_parquet(p["fires"])
            STATE.drivers = meta.get("drivers", {})
            STATE.alerts = meta.get("alerts", [])
            STATE.sources = meta.get("sources", [])
            STATE.model_name = meta.get("model_name", "none")
            STATE.last_refresh = meta.get("last_refresh")
            STATE.stale = True
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
def refresh(*, do_ingest: bool = True) -> dict:
    """Run one full cycle. Returns a short status dict."""
    from features.build import build_matrix
    from models.serving import forecast as run_forecast
    from models.serving import naive_forecast, peak_alerts

    status = {"ok": False, "steps": []}
    try:
        if do_ingest:
            from ingest.run_ingest import run_once
            run_once()
            status["steps"].append("ingest")

        obs = _read_or_empty(SETTINGS.processed_dir / "obs.parquet")
        met = _read_or_empty(SETTINGS.interim_dir / "weather_forecast.parquet")
        fires = _read_or_empty(SETTINGS.interim_dir / "fires_recent.parquet")
        if met.empty:
            raise RuntimeError("no forecast meteorology available")

        feat = build_matrix(obs, met, fires if not fires.empty else None)
        status["steps"].append("features")

        fc, model_name = _load_model()
        if fc is not None:
            fdf = run_forecast(fc, feat)
        else:
            fdf = naive_forecast(feat)
        status["steps"].append(f"forecast:{model_name}")

        drivers = _compute_drivers(fc, feat, fdf) if fc is not None else {}
        alerts = _decorate_alerts(peak_alerts(fdf))

        manifest = _read_manifest()

        with STATE.lock:
            STATE.forecast = fdf
            STATE.observations = obs
            STATE.fires = fires
            STATE.drivers = drivers
            STATE.alerts = alerts
            STATE.sources = manifest
            STATE.model_name = model_name
            STATE.last_refresh = dt.datetime.now(_UTC).isoformat()
            STATE.stale = False
        _save_snapshot()
        status.update(ok=True, model=model_name, stations=int(fdf["station_id"].nunique()))
    except Exception as exc:  # noqa: BLE001
        STATE.stale = True
        status["error"] = str(exc)
    return status


def _read_or_empty(path) -> pd.DataFrame:
    try:
        return read_table(path) if path.exists() else pd.DataFrame()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _read_manifest() -> list[dict]:
    p = SETTINGS.interim_dir / "ingest_manifest.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("sources", [])
        except Exception:  # noqa: BLE001
            return []
    return []


def _compute_drivers(fc, feat: pd.DataFrame, fdf: pd.DataFrame) -> dict[str, dict]:
    from models.dataset import make_supervised
    from models.drivers import explain_station_forecast

    try:
        sup, _ = make_supervised(feat, list(range(1, 73)), target="pm25",
                                 base_stride_h=1, require_target=False)
        base = feat.dropna(subset=["pm25"]).sort_values(["station_id", "ts"])
        base = base.groupby("station_id").tail(1)[["station_id", "ts"]].rename(columns={"ts": "_b"})
        sup = sup.merge(base, on="station_id").query("ts == _b").drop(columns="_b")
    except Exception:  # noqa: BLE001
        return {}

    out: dict[str, dict] = {}
    feat_now = feat.dropna(subset=["pm25"]).sort_values("ts").groupby("station_id").tail(1)
    fn = feat_now.set_index("station_id")
    for sid in fdf["station_id"].unique():
        try:
            d = explain_station_forecast(fc, fdf, sup, sid)
        except Exception:  # noqa: BLE001
            d = {"station_id": sid, "groups": {}, "top_features": []}
        if sid in fn.index:
            row = fn.loc[sid]
            d["isi"] = _f(row.get("isi"))
            d["isi_components"] = {
                k: _f(row.get(k)) for k in ("isi_theta", "isi_pbl", "isi_stagnation", "isi_radiative")
                if _f(row.get(k)) is not None
            }
            d["incoming_stubble_load"] = _f(row.get("incoming_stubble_load"))
            d["plume_from_bearing_deg"] = _f(row.get("plume_from_bearing_deg"))
            d["ventilation_index"] = _f(row.get("ventilation_index"))
        out[sid] = d
    return out


def _decorate_alerts(alerts: list[dict]) -> list[dict]:
    idx = station_index()
    for a in alerts:
        st = idx.get(a["station_id"])
        a["name"] = st.name if st else a["station_id"]
    return alerts


def _f(v):
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 3)
    except (TypeError, ValueError):
        return None
