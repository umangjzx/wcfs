"""Inverse-distance-weighted interpolation of station AQI to a raster for the map layer."""

from __future__ import annotations

import numpy as np

from config.settings import NCR_BBOX


def idw_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    *,
    bounds: tuple[float, float, float, float] = NCR_BBOX,
    nx: int = 48,
    ny: int = 40,
    power: float = 2.5,
    smoothing_km: float = 3.0,
) -> dict:
    """Return {bounds, cells:[{lat,lon,aqi}]} — a coarse NCR raster."""
    lat_min, lon_min, lat_max, lon_max = bounds
    gy = np.linspace(lat_min, lat_max, ny)
    gx = np.linspace(lon_min, lon_max, nx)
    mesh_lat, mesh_lon = np.meshgrid(gy, gx, indexing="ij")

    ok = np.isfinite(values)
    lats, lons, values = lats[ok], lons[ok], values[ok]
    if len(values) == 0:
        return {"bounds": list(bounds), "cells": []}

    # squared great-circle-ish distance in km (small-angle)
    dlat = (mesh_lat[..., None] - lats[None, None, :]) * 111.0
    dlon = (mesh_lon[..., None] - lons[None, None, :]) * 111.0 * np.cos(np.deg2rad(mesh_lat[..., None]))
    d = np.sqrt(dlat ** 2 + dlon ** 2) + smoothing_km
    w = 1.0 / d ** power
    grid = np.sum(w * values[None, None, :], axis=2) / np.sum(w, axis=2)

    cells = [
        {"lat": round(float(mesh_lat[i, j]), 4), "lon": round(float(mesh_lon[i, j]), 4),
         "aqi": round(float(grid[i, j]), 1)}
        for i in range(ny) for j in range(nx)
    ]
    return {"bounds": list(bounds), "cells": cells}
