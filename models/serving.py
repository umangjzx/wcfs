"""Turn the live serving feature frame into a 72-hour hourly forecast per station.

Uses the trained ``LGBMForecaster`` with the GFS forecast meteorology (and advected fire
load) as the known-future inputs. Output is what the API serves.
"""

from __future__ import annotations

import pandas as pd

from aqi.cpcb_aqi import aqi_category, health_advisory
from config.settings import FORECAST_HORIZON_HOURS
from models.baseline_lgbm import MultiForecaster
from models.dataset import make_supervised


def _latest_base_rows(feat: pd.DataFrame) -> pd.DataFrame:
    """The most recent row per station that has an observed PM2.5 (the forecast origin t0)."""
    f = feat.dropna(subset=["pm25"]).sort_values(["station_id", "ts"])
    return f.groupby("station_id", as_index=False).tail(1)


def forecast(
    fc: MultiForecaster,
    feat_serving: pd.DataFrame,
    *,
    horizon_h: int = FORECAST_HORIZON_HOURS,
) -> pd.DataFrame:
    """Return station_id, issued_ts, valid_ts, horizon, pm25_p10/p50/p90, aqi, category, advisory."""
    feat_serving = feat_serving.sort_values(["station_id", "ts"]).copy()
    feat_serving["ts"] = pd.to_datetime(feat_serving["ts"], utc=True)
    horizons = list(range(1, horizon_h + 1))

    sup, _ = make_supervised(feat_serving, horizons, target="pm25", base_stride_h=1,
                             require_target=False)
    base = _latest_base_rows(feat_serving)[["station_id", "ts"]].rename(columns={"ts": "_base_ts"})
    sup = sup.merge(base, on="station_id", how="inner")
    sup = sup[sup["ts"] == sup["_base_ts"]].drop(columns="_base_ts")
    if sup.empty:
        return pd.DataFrame()

    pred = fc.predict(sup).rename(columns={"ts0": "issued_ts"})
    pred["aqi"] = pd.to_numeric(pred["aqi"], errors="coerce").round().astype("Int64")
    pred["category"] = pred["aqi"].map(lambda a: aqi_category(a) if pd.notna(a) else "Unknown")
    pred["advisory"] = pred["category"].map(health_advisory)
    keep = ["station_id", "issued_ts", "valid_ts", "horizon",
            "pm25_p10", "pm25_p50", "pm25_p90", "aqi", "category", "advisory",
            "dominant_pollutant"]
    for extra in ("pm10_p50", "no2_p50"):
        if extra in pred:
            keep.append(extra)
    return pred[keep].sort_values(["station_id", "horizon"]).reset_index(drop=True)


def peak_alerts(forecast_df: pd.DataFrame, thresholds: dict[str, int] | None = None) -> list[dict]:
    """Soonest crossing of each AQI threshold per station, for the alerts banner."""
    from aqi.cpcb_aqi import sub_index_series
    from config.settings import ALERT_THRESHOLDS
    from models.baseline_lgbm import event_score

    thresholds = thresholds or ALERT_THRESHOLDS
    if forecast_df.empty:
        return []
    f = forecast_df.copy()
    # alert on the ~P75 event-decision score, not the median (precision/recall for hazards)
    f["_ev_aqi"] = sub_index_series(
        "PM2.5", event_score(f["pm25_p50"].to_numpy(), f["pm25_p90"].to_numpy()))
    alerts = []
    for sid, g in f.groupby("station_id"):
        g = g.sort_values("horizon")
        for label, thr in thresholds.items():
            hit = g[g["_ev_aqi"].fillna(0) >= thr]
            if not hit.empty:
                r = hit.iloc[0]
                alerts.append({
                    "station_id": sid, "level": label, "lead_hours": int(r["horizon"]),
                    "valid_ts": r["valid_ts"].isoformat(),
                    "aqi": int(r["aqi"]) if pd.notna(r["aqi"]) else int(r["_ev_aqi"]),
                })
    return sorted(alerts, key=lambda a: (a["lead_hours"], -a["aqi"]))
