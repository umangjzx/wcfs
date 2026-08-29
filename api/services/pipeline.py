"""The live forecast pipeline + in-memory state the API serves.

refresh():  ingest (CPCB now + GFS forecast + FIRMS) -> serving feature matrix ->
            trained-model forecast -> AQI grid + per-station drivers + alerts -> in-memory cache.

Live data only: there is no bundled snapshot and no heuristic fallback forecast. Until a
refresh has ingested real CPCB/GFS/FIRMS data and the trained model has run, the API serves
nothing (503). A failed refresh keeps the last good state and flags it stale.
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
    """Load the trained forecaster from the registry, or ``(None, "none")``.

    There is no heuristic fallback — if the registry has no usable model, the
    pipeline raises and the API serves 503 rather than a synthetic forecast.
    """
    from models.baseline_lgbm import REGISTRY, MultiForecaster

    if (REGISTRY / "lgbm_pm25_meta.json").exists():
        try:
            return MultiForecaster.load(), "lgbm"
        except Exception:  # noqa: BLE001
            pass
    return None, "none"


# ---------------------------------------------------------------------------
def refresh(*, do_ingest: bool = True) -> dict:
    """Run one full cycle. Returns a short status dict."""
    from features.build import build_matrix
    from models.serving import forecast as run_forecast
    from models.serving import peak_alerts

    status = {"ok": False, "steps": []}
    try:
        if do_ingest:
            from ingest.run_ingest import run_once
            run_once()
            status["steps"].append("ingest")

        obs = _read_or_empty(SETTINGS.processed_dir / "obs.parquet")
        met = _read_or_empty(SETTINGS.interim_dir / "weather_forecast.parquet")
        fires = _read_or_empty(SETTINGS.interim_dir / "fires_recent.parquet")
        if obs.empty:
            raise RuntimeError("no observations ingested yet — run a live ingest first")
        if met.empty:
            raise RuntimeError("no forecast meteorology available")

        feat = build_matrix(obs, met, fires if not fires.empty else None)
        status["steps"].append("features")

        fc, model_name = _load_model()
        if fc is None:
            raise RuntimeError("no trained model in models/registry — train one first")
        fdf = run_forecast(fc, feat)
        if fdf.empty:
            raise RuntimeError("model produced no forecast rows")
        status["steps"].append(f"forecast:{model_name}")

        drivers = _compute_drivers(fc, feat, fdf)
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
        try:
            from api.services.store import write_refresh
            wrote = write_refresh(STATE)
            if wrote is not None:
                status["db"] = wrote
        except Exception as exc:  # noqa: BLE001 - the Postgres mirror is best-effort
            status["db_error"] = str(exc)
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
