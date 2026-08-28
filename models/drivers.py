"""Forecast explanations — SHAP attributions grouped into the coupling story.

Every driver feature is mapped to one of six human-readable groups so the dashboard can
say *why* a forecast is high: inversion trapping, upwind stubble transport, local
emissions / persistence, wind ventilation, other meteorology, time & season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.baseline_lgbm import LGBMForecaster
from models.dataset import encode_categoricals

DRIVER_GROUPS: dict[str, tuple[str, ...]] = {
    "inversion_trapping": (
        "isi", "isi_theta", "isi_pbl", "isi_stagnation", "isi_lag24",
        "ventilation_index", "pbl_height", "dtheta_surface", "self_trapping",
        "pbl_suppression", "pm25_x_blh_lag24", "blh", "blh_lag24", "f_blh",
        "f_isi", "f_isi_pbl", "f_isi_stagnation", "f_ventilation_index",
    ),
    "stubble_transport": (
        "incoming_stubble_load", "stubble_index", "plume_from_bearing_deg",
        "fire_frp_active", "nearest_fire_km",
        "f_incoming_stubble_load", "f_stubble_index",
    ),
    "local_emissions_persistence": (
        "pm25", "pm10", "no2", "aqi", "aqi_inst", "aqi_lag24",
        "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag12", "pm25_lag24",
        "pm25_roll6", "pm25_roll24", "pm25_tend_6h", "pm25_tend_24h",
        "pm10_lag6", "no2_lag6", "aod_proxy", "aod_proxy_lag24",
    ),
    "wind_ventilation": (
        "wind_speed10", "wind_u10", "wind_v10", "wind_u850", "wind_v850",
        "wind_speed10_lag24", "f_wind_speed10", "f_wind_u10", "f_wind_v10",
        "f_wind_u850", "f_wind_v850",
    ),
    "other_meteorology": (
        "t2m", "d2m", "rh2m", "surface_pressure", "precip", "solar", "cloud",
        "t2m_lag24", "radiative_dimming",
        "f_t2m", "f_rh2m", "f_precip", "f_solar", "f_cloud",
    ),
    "time_season": (
        "local_hour", "local_month", "is_weekend", "is_stubble_season",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "days_into_stubble_season", "diwali_proximity", "horizon",
        "f_hour_sin", "f_hour_cos", "f_doy_sin", "f_doy_cos", "f_is_weekend",
        "f_diwali_proximity", "f_is_stubble_season", "f_days_into_stubble_season",
    ),
}
_FEATURE_GROUP = {f: g for g, feats in DRIVER_GROUPS.items() for f in feats}


def _shap_values(fc: LGBMForecaster, X: pd.DataFrame) -> np.ndarray:
    booster = fc.models["median"].booster_
    contrib = booster.predict(X, pred_contrib=True)  # (n, n_features + 1) — last col = bias
    return np.asarray(contrib)[:, :-1]


def explain(fc: LGBMForecaster, sup: pd.DataFrame, top_k: int = 6) -> list[dict]:
    """Per supervised row: grouped SHAP contributions (ug/m3 of PM2.5) + top features."""
    X, _ = encode_categoricals(sup[fc.feature_cols], fc.categorical)
    shap = _shap_values(fc, X)
    feats = list(fc.feature_cols)

    group_idx: dict[str, list[int]] = {g: [] for g in DRIVER_GROUPS}
    group_idx["unclassified"] = []
    for i, f in enumerate(feats):
        group_idx.setdefault(_FEATURE_GROUP.get(f, "unclassified"), []).append(i)

    rows = []
    for r in range(len(sup)):
        gc = {g: float(np.sum(shap[r, idx])) for g, idx in group_idx.items() if idx}
        order = np.argsort(-np.abs(shap[r]))[:top_k]
        top = [{"feature": feats[i], "group": _FEATURE_GROUP.get(feats[i], "unclassified"),
                "contribution": round(float(shap[r, i]), 2)} for i in order]
        rows.append({
            "groups": {g: round(v, 2) for g, v in sorted(gc.items(), key=lambda kv: -abs(kv[1]))},
            "top_features": top,
        })
    return rows


def explain_station_forecast(fc: LGBMForecaster, serving_pred: pd.DataFrame,
                             sup: pd.DataFrame, station_id: str) -> dict:
    """Aggregate driver groups across a station's 72 h forecast for the API `/drivers`."""
    mask = sup["station_id"] == station_id
    if not mask.any():
        return {"station_id": station_id, "groups": {}, "top_features": []}
    per_row = explain(fc, sup[mask])
    agg: dict[str, float] = {}
    for row in per_row:
        for g, v in row["groups"].items():
            agg[g] = agg.get(g, 0.0) + v
    n = len(per_row)
    agg = {g: round(v / n, 2) for g, v in sorted(agg.items(), key=lambda kv: -abs(kv[1]))}
    feat_scores: dict[str, float] = {}
    for row in per_row:
        for tf in row["top_features"]:
            feat_scores[tf["feature"]] = feat_scores.get(tf["feature"], 0.0) + abs(tf["contribution"])
    top = sorted(feat_scores.items(), key=lambda kv: -kv[1])[:8]
    return {
        "station_id": station_id,
        "groups": agg,
        "top_features": [{"feature": f, "group": _FEATURE_GROUP.get(f, "unclassified"),
                          "importance": round(s / n, 2)} for f, s in top],
    }
