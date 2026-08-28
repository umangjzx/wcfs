"""Stubble-plume transport vector — the other half of the coupled link.

For every (station, hour) we advect the upwind Punjab/Haryana fire field toward the station
using that hour's transport-level wind, and accumulate an FRP-weighted "incoming stubble
load" plus a 2-D plume vector (for the map arrow and as model features).

Per fire cluster c and station s at time t:
    ê        unit vector from c to s
    v_in     wind projected onto ê  (m/s; > 0 means wind carries smoke toward s)
    align    clip(v_in / |wind|, 0, 1)                     -- directional match
    t_tr     d / max(v_in, floor) + cluster_age            -- transport time (h)
    w_c      FRP_c * align * exp(-t_tr / tau) * dilution(d)

    incoming_stubble_load = sum_c w_c
    (plume_u, plume_v)     = sum_c w_c * ê                  -- mean transport direction

Falls back to all-zero features when no fire data is available (no FIRMS key).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    PLUME_DECAY_TAU_HOURS,
    PLUME_MIN_INBOUND_SPEED_MS,
    Station,
    load_stations,
)

_KM_PER_DEG_LAT = 110.57
_KM_PER_DEG_LON = 111.32
_LOAD_SCALE = 2000.0        # FRP-weighted load that maps to stubble_index ~0.63
_DILUTION_SCALE_KM = 60.0
_CARRYOVER_DAYS = 1         # also use yesterday's fires, aged by 24 h

STUBBLE_FEATURE_COLUMNS = [
    "incoming_stubble_load", "stubble_index", "plume_u", "plume_v",
    "plume_from_bearing_deg", "fire_frp_active", "fire_count_active", "nearest_fire_km",
]


def _zero_frame(keys: pd.DataFrame) -> pd.DataFrame:
    out = keys[["station_id", "ts"]].copy()
    for c in STUBBLE_FEATURE_COLUMNS:
        out[c] = 0.0
    out["fire_count_active"] = out["fire_count_active"].astype("int64")
    out["nearest_fire_km"] = np.nan
    out["plume_from_bearing_deg"] = np.nan
    return out


def compute_stubble_features(
    met: pd.DataFrame,
    fires: pd.DataFrame | None,
    stations: list[Station] | None = None,
) -> pd.DataFrame:
    """Return one row of stubble features per (station_id, ts) in ``met``."""
    keys = met[["station_id", "ts"]].drop_duplicates().reset_index(drop=True)
    if fires is None or fires.empty:
        return _zero_frame(keys)

    st_by_id = {s.id: s for s in (stations or load_stations())}
    f = fires.copy()
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    fires_by_date: dict[pd.Timestamp, pd.DataFrame] = dict(tuple(f.groupby("date")))

    m = met.copy()
    m["ts"] = pd.to_datetime(m["ts"], utc=True)
    m["date"] = m["ts"].dt.normalize().dt.tz_localize(None)
    # transport wind: 850 hPa, fall back to 10 m
    u = m["wind_u850"].fillna(m["wind_u10"]).to_numpy("float64")
    v = m["wind_v850"].fillna(m["wind_v10"]).to_numpy("float64")
    m["_u"], m["_v"] = u, v

    results: list[pd.DataFrame] = []
    for sid, g in m.groupby("station_id", sort=False):
        st = st_by_id.get(sid)
        if st is None:
            continue
        lat0 = np.deg2rad(st.lat)
        for date, gd in g.groupby("date", sort=False):
            clusters = []
            for age_days in range(0, _CARRYOVER_DAYS + 1):
                key = pd.Timestamp(date) - pd.Timedelta(days=age_days)
                fc = fires_by_date.get(key)
                if fc is not None and not fc.empty:
                    fc = fc.copy()
                    fc["_age_h"] = 24.0 * age_days
                    clusters.append(fc)
            gd = gd.sort_values("ts")
            if not clusters:
                results.append(_zero_frame(gd))
                continue
            fc = pd.concat(clusters, ignore_index=True)
            # geometry: vector from cluster -> station, in km (east, north)
            ex_km = (st.lon - fc["lon"].to_numpy("float64")) * _KM_PER_DEG_LON * np.cos(lat0)
            ey_km = (st.lat - fc["lat"].to_numpy("float64")) * _KM_PER_DEG_LAT
            d_km = np.hypot(ex_km, ey_km)
            d_safe = np.where(d_km < 1.0, 1.0, d_km)
            ex, ey = ex_km / d_safe, ey_km / d_safe
            frp = fc["frp_sum"].to_numpy("float64")
            age_h = fc["_age_h"].to_numpy("float64")
            dilution = 1.0 / (1.0 + (d_km / _DILUTION_SCALE_KM) ** 2)

            uu = gd["_u"].to_numpy("float64")
            vv = gd["_v"].to_numpy("float64")
            spd = np.hypot(uu, vv)
            v_in = uu[:, None] * ex[None, :] + vv[:, None] * ey[None, :]           # (H, K)
            align = np.clip(v_in / np.maximum(spd[:, None], 1e-6), 0.0, 1.0)
            v_in_pos = np.maximum(v_in, PLUME_MIN_INBOUND_SPEED_MS)
            t_tr = d_km[None, :] * 1000.0 / v_in_pos / 3600.0 + age_h[None, :]
            decay = np.where(v_in > 0.0, np.exp(-t_tr / PLUME_DECAY_TAU_HOURS), 0.0)
            w = frp[None, :] * align * decay * dilution[None, :]                    # (H, K)

            load = w.sum(axis=1)
            plume_u = (w * ex[None, :]).sum(axis=1)
            plume_v = (w * ey[None, :]).sum(axis=1)
            from_bearing = (np.degrees(np.arctan2(-plume_u, -plume_v)) % 360.0)
            from_bearing[load <= 0] = np.nan

            today = fires_by_date.get(pd.Timestamp(date))
            frp_active = float(today["frp_sum"].sum()) if today is not None else 0.0
            count_active = int(len(today)) if today is not None else 0

            results.append(
                pd.DataFrame(
                    {
                        "station_id": sid,
                        "ts": gd["ts"].to_numpy(),
                        "incoming_stubble_load": load,
                        "stubble_index": 1.0 - np.exp(-load / _LOAD_SCALE),
                        "plume_u": plume_u,
                        "plume_v": plume_v,
                        "plume_from_bearing_deg": from_bearing,
                        "fire_frp_active": frp_active,
                        "fire_count_active": count_active,
                        "nearest_fire_km": float(np.min(d_km)),
                    }
                )
            )

    if not results:
        return _zero_frame(keys)
    out = pd.concat(results, ignore_index=True)
    return keys.merge(out, on=["station_id", "ts"], how="left").fillna(
        {c: 0.0 for c in STUBBLE_FEATURE_COLUMNS if c not in ("plume_from_bearing_deg", "nearest_fire_km")}
    )
