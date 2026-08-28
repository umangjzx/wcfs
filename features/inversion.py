"""Inversion Strength Index (ISI) — one half of the coupled met->chem link.

A shallow, stable boundary layer traps PM2.5 near the surface. ISI in [0, 1] blends four
observable proxies for that trapping, each also exposed individually for explainability:

  isi_theta       low-level potential-temperature increase (925 hPa vs surface) -> stability
  isi_pbl         boundary-layer shallowness (exp decay in blh)
  isi_stagnation  near-surface calm (low 10 m wind)
  isi_radiative   clear-sky + calm + night -> radiative surface inversion

Also returns the ventilation index (blh * wind) — the classic dispersion metric.

Input: a met frame with columns t2m, t925, t850, surface_pressure, blh, wind_speed10,
cloud, solar (see ingest.common.MET_COLUMNS). Temperatures in degrees Celsius.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_RCP = 0.286  # R/cp for dry air
_KELVIN = 273.15

# tuning scales
_THETA_SCALE_K = 8.0       # dtheta (K) that saturates the stability term
_PBL_EFOLD_M = 400.0       # blh e-folding scale
_CALM_WIND_MS = 3.0        # 10 m wind at/below which air is "stagnant"
_ISI_WEIGHTS = {"theta": 0.35, "pbl": 0.30, "stagnation": 0.20, "radiative": 0.15}


def _theta(temp_c: pd.Series, pressure_hpa: float | pd.Series) -> pd.Series:
    """Potential temperature (K) referenced to 1000 hPa."""
    return (temp_c + _KELVIN) * (1000.0 / pressure_hpa) ** _RCP


def compute_inversion_features(met: pd.DataFrame) -> pd.DataFrame:
    """Return ISI + components, indexed the same as ``met`` (keeps station_id, ts, kind).

    Robust to missing pressure-level temperatures: when t925/t850 are absent (some
    historical reanalysis pulls), the potential-temperature term is dropped and ISI is
    renormalized over the components that are available.
    """
    m = met.copy()
    sp = m["surface_pressure"].where(m["surface_pressure"].between(800, 1100), 1000.0)

    theta_sfc = _theta(m["t2m"], sp)
    theta_925 = _theta(m["t925"], 925.0)
    theta_850 = _theta(m["t850"], 850.0)

    dtheta_surface = theta_925 - theta_sfc          # surface..~760 m
    dtheta_lower = theta_850 - theta_925            # ~760 m..~1500 m
    theta_available = m["t925"].notna() & m["t2m"].notna()
    isi_theta = np.clip(dtheta_surface / _THETA_SCALE_K, 0.0, 1.0)

    blh = m["blh"].clip(lower=10.0)
    isi_pbl = np.exp(-blh / _PBL_EFOLD_M)

    wind = m["wind_speed10"].clip(lower=0.0)
    calm = np.clip((_CALM_WIND_MS - wind) / _CALM_WIND_MS, 0.0, 1.0)
    isi_stagnation = calm

    # night: prefer the solar column; fall back to "blh very low & no solar column"
    if "solar" in m and m["solar"].notna().any():
        night = (m["solar"].fillna(0.0) < 5.0).astype("float64")
    else:  # pragma: no cover - solar is always present in our met schema
        night = (m.get("local_hour", pd.Series(0, index=m.index)).isin([*range(19, 24), *range(0, 6)])
                 ).astype("float64")
    clear = np.clip(1.0 - m["cloud"].fillna(50.0) / 100.0, 0.0, 1.0)
    isi_radiative = night * clear * calm

    w = _ISI_WEIGHTS
    comps = {
        "theta": (isi_theta.fillna(0.0), theta_available.astype("float64") * w["theta"]),
        "pbl": (isi_pbl.fillna(0.0), (m["blh"].notna()).astype("float64") * w["pbl"]),
        "stagnation": (isi_stagnation.fillna(0.0),
                       (m["wind_speed10"].notna()).astype("float64") * w["stagnation"]),
        "radiative": (isi_radiative.fillna(0.0), pd.Series(w["radiative"], index=m.index)),
    }
    num = sum(val * wt for val, wt in comps.values())
    den = sum(wt for _, wt in comps.values()).replace(0.0, np.nan)
    isi = (num / den).clip(0.0, 1.0)

    out = pd.DataFrame(index=m.index)
    for col in ("station_id", "ts", "kind", "lead_h"):
        if col in m:
            out[col] = m[col]
    out["isi"] = isi
    out["isi_theta"] = isi_theta
    out["isi_pbl"] = isi_pbl
    out["isi_stagnation"] = isi_stagnation
    out["isi_radiative"] = isi_radiative
    out["dtheta_surface"] = dtheta_surface
    out["dtheta_lower"] = dtheta_lower
    out["pbl_height"] = m["blh"]
    out["ventilation_index"] = (m["blh"].clip(lower=10.0) * wind)
    return out


INVERSION_FEATURE_COLUMNS = [
    "isi", "isi_theta", "isi_pbl", "isi_stagnation", "isi_radiative",
    "dtheta_surface", "dtheta_lower", "pbl_height", "ventilation_index",
]
