"""Turn the live serving feature frame into a 72-hour hourly forecast per station.

Uses the trained ``LGBMForecaster`` with the GFS forecast meteorology (and advected fire
load) as the known-future inputs. Output is what the API serves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqi.cpcb_aqi import aqi_category, health_advisory
from config.settings import FORECAST_HORIZON_HOURS
from models.baseline_lgbm import LGBMForecaster
from models.dataset import make_supervised


def _latest_base_rows(feat: pd.DataFrame) -> pd.DataFrame:
    """The most recent row per station that has an observed PM2.5 (the forecast origin t0)."""
    f = feat.dropna(subset=["pm25"]).sort_values(["station_id", "ts"])
    return f.groupby("station_id", as_index=False).tail(1)


def forecast(
    fc: LGBMForecaster,
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

    pred = fc.predict(sup)
    pred = pred.rename(columns={"ts0": "issued_ts"})
    pred["aqi"] = pred["aqi_p50"].round().astype("Int64")
    pred["category"] = pred["aqi"].map(lambda a: aqi_category(a) if pd.notna(a) else "Unknown")
    pred["advisory"] = pred["category"].map(health_advisory)
    pred["dominant_pollutant"] = "PM2.5"
    cols = ["station_id", "issued_ts", "valid_ts", "horizon",
            "pm25_p10", "pm25_p50", "pm25_p90", "aqi", "category", "advisory",
            "dominant_pollutant"]
    return pred[cols].sort_values(["station_id", "horizon"]).reset_index(drop=True)


_DELHI_WINTER_DIURNAL = np.array([  # relative PM2.5 by local hour (peak pre-dawn / evening)
    1.18, 1.22, 1.24, 1.25, 1.24, 1.20, 1.10, 0.95, 0.82, 0.74, 0.70, 0.70,
    0.72, 0.74, 0.76, 0.80, 0.88, 1.00, 1.10, 1.15, 1.16, 1.16, 1.16, 1.17,
])


def naive_forecast(feat_serving: pd.DataFrame, *, horizon_h: int = FORECAST_HORIZON_HOURS
                   ) -> pd.DataFrame:
    """Persistence-to-diurnal-climatology fallback when no trained model is available."""
    from aqi.cpcb_aqi import aqi_category, health_advisory, sub_index_series
    from ingest.common import IST

    base = _latest_base_rows(feat_serving)
    rows = []
    for _, b in base.iterrows():
        t0 = pd.Timestamp(b["ts"]).tz_convert("UTC")
        pm0 = float(b["pm25"])
        h0 = t0.tz_convert(IST).hour
        anchor = pm0 / _DELHI_WINTER_DIURNAL[h0]
        for h in range(1, horizon_h + 1):
            vt = t0 + pd.Timedelta(hours=h)
            hh = vt.tz_convert(IST).hour
            decay = np.exp(-h / 30.0)
            p50 = decay * pm0 + (1 - decay) * anchor * _DELHI_WINTER_DIURNAL[hh]
            spread = 0.35 * p50 * (1 - decay) + 6
            rows.append({
                "station_id": b["station_id"], "issued_ts": t0, "valid_ts": vt, "horizon": h,
                "pm25_p10": max(p50 - spread, 0), "pm25_p50": p50, "pm25_p90": p50 + spread,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["aqi"] = sub_index_series("PM2.5", df["pm25_p50"].to_numpy()).round()
    df["aqi"] = df["aqi"].astype("Int64")
    df["category"] = df["aqi"].map(lambda a: aqi_category(a) if pd.notna(a) else "Unknown")
    df["advisory"] = df["category"].map(health_advisory)
    df["dominant_pollutant"] = "PM2.5"
    return df.sort_values(["station_id", "horizon"]).reset_index(drop=True)


def peak_alerts(forecast_df: pd.DataFrame, thresholds: dict[str, int] | None = None) -> list[dict]:
    """Soonest crossing of each AQI threshold per station, for the alerts banner."""
    from config.settings import ALERT_THRESHOLDS

    thresholds = thresholds or ALERT_THRESHOLDS
    if forecast_df.empty:
        return []
    alerts = []
    for sid, g in forecast_df.groupby("station_id"):
        g = g.sort_values("horizon")
        for label, thr in thresholds.items():
            hit = g[g["aqi"].fillna(0) >= thr]
            if not hit.empty:
                r = hit.iloc[0]
                alerts.append({
                    "station_id": sid, "level": label, "lead_hours": int(r["horizon"]),
                    "valid_ts": r["valid_ts"].isoformat(), "aqi": int(r["aqi"]),
                })
    return sorted(alerts, key=lambda a: (a["lead_hours"], -a["aqi"]))
