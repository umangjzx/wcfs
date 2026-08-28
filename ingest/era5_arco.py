"""Historical meteorology from ARCO-ERA5 (Analysis-Ready, Cloud-Optimized ERA5).

Public Zarr on Google Cloud, no account or key required. Provides the 925/850 hPa winds &
temperatures the Open-Meteo archive omits, so the stubble-plume transport wind and the ISI
theta-term are full fidelity for training history. Writes the same schema/target as
``ingest.weather --history`` (``data/processed/met_history.parquet``) so ``features.build``
is unchanged.

STATUS (2026-08): from a Windows box this path is currently impractical — the consolidated
``1959-2023_01_10`` store has a corrupt ``level`` coordinate (all zeros) and a non-unique
lat/lon index, and the ``full_37-...zarr-v3`` store takes ~2 min just to open. Left in the
tree because it is the right long-term answer and a Linux VM / different xarray+zarr build
may handle these stores cleanly. Training history currently runs on the Open-Meteo ERA5
archive (surface + BLH + cloud) with the pressure-level terms degrading gracefully; see
``.planning/PROJECT.md`` "historical meteorology" decision.

    pip install -e ".[era5]"
    python -m ingest.era5_arco --start 2021-10-01 --end 2023-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from config.settings import SETTINGS, Station, load_stations
from ingest.common import MET_COLUMNS, SourceResult, write_table

ARCO_STORES = {
    "default": (
        "gs://gcp-public-data-arco-era5/ar/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr",
        True,   # consolidated
    ),
    "v3": (
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        False,
    ),
}
LEVELS = (1000, 925, 850)

# canonical field -> candidate ARCO variable names (first match wins)
_SURFACE = {
    "t2m": ["2m_temperature"],
    "d2m": ["2m_dewpoint_temperature"],
    "wind_u10": ["10m_u_component_of_wind"],
    "wind_v10": ["10m_v_component_of_wind"],
    "surface_pressure": ["surface_pressure"],
    "precip": ["total_precipitation"],
    "cloud": ["total_cloud_cover"],
    "blh": ["boundary_layer_height"],
    "solar": ["surface_solar_radiation_downwards", "surface_net_solar_radiation"],
}
_LEVEL_VARS = {
    "temperature": ["temperature"],
    "u": ["u_component_of_wind"],
    "v": ["v_component_of_wind"],
}


def _pick(ds, names: list[str]) -> str | None:
    for n in names:
        if n in ds.data_vars:
            return n
    return None


def _magnus_rh(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    def es(x):
        return 6.112 * np.exp(17.67 * x / (x + 243.5))

    return np.clip(100.0 * es(td_c) / es(t_c), 0.0, 100.0)


def open_arco(store: str = "default"):  # pragma: no cover - network / heavy dep
    try:
        import xarray as xr
    except ImportError as exc:
        raise SystemExit('ARCO-ERA5 needs extra deps: pip install -e ".[era5]"') from exc
    url, consolidated = ARCO_STORES[store]
    return xr.open_zarr(url, chunks={}, consolidated=consolidated,
                        storage_options={"token": "anon"})


def fetch_reanalysis_arco(
    stations: list[Station] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    *,
    ds=None,
) -> tuple[pd.DataFrame, SourceResult]:
    """Point-extract ERA5 for each station over [start, end); returns the MET schema frame."""
    import xarray as xr

    stations = stations or load_stations()
    start = start or (dt.date.today() - dt.timedelta(days=40))
    end = end or (dt.date.today() - dt.timedelta(days=6))
    ds = ds if ds is not None else open_arco()

    lats = xr.DataArray([s.lat for s in stations], dims="station",
                        coords={"station": [s.id for s in stations]})
    lons = xr.DataArray([s.lon % 360 for s in stations], dims="station",
                        coords={"station": [s.id for s in stations]})
    tsel = slice(pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(hours=1))

    surf_names = {k: _pick(ds, v) for k, v in _SURFACE.items()}
    lvl_names = {k: _pick(ds, v) for k, v in _LEVEL_VARS.items()}
    keep = [n for n in list(surf_names.values()) + list(lvl_names.values()) if n]
    sub = (
        ds[keep]
        .sel(time=tsel)
        .sel(latitude=lats, longitude=lons, method="nearest")
    )
    if lvl_names["temperature"]:
        sub = sub.sel(level=list(LEVELS))
    sub = sub.load()

    times = pd.to_datetime(sub.time.values, utc=True)
    n = len(times)

    def _v(g, name):
        return np.asarray(g[name].values, dtype="float64") if name else np.full(n, np.nan)

    frames = []
    for i, st in enumerate(stations):
        col = {"station_id": st.id, "ts": times, "kind": "reanalysis",
               "lead_h": pd.NA, "source": "arco-era5"}
        g = sub.isel(station=i)

        t2m = _v(g, surf_names["t2m"]) - 273.15
        d2m = _v(g, surf_names["d2m"]) - 273.15
        col["t2m"], col["d2m"] = t2m, d2m
        col["rh2m"] = _magnus_rh(t2m, d2m)
        u10, v10 = _v(g, surf_names["wind_u10"]), _v(g, surf_names["wind_v10"])
        col["wind_u10"], col["wind_v10"] = u10, v10
        col["wind_speed10"] = np.hypot(u10, v10)
        col["wind_dir10"] = (np.degrees(np.arctan2(-u10, -v10)) % 360.0)
        col["surface_pressure"] = _v(g, surf_names["surface_pressure"]) / 100.0  # Pa -> hPa
        col["precip"] = _v(g, surf_names["precip"]) * 1000.0                      # m -> mm
        col["cloud"] = _v(g, surf_names["cloud"]) * 100.0                        # 0..1 -> %
        col["blh"] = _v(g, surf_names["blh"])
        col["solar"] = _v(g, surf_names["solar"]) / 3600.0                       # J/m2 -> W/m2

        if lvl_names["temperature"]:
            tlev = g[lvl_names["temperature"]]
            for lv, name in ((1000, "t1000"), (925, "t925"), (850, "t850")):
                col[name] = np.asarray(tlev.sel(level=lv).values, dtype="float64") - 273.15
            col["wind_u850"] = np.asarray(g[lvl_names["u"]].sel(level=850).values, dtype="float64")
            col["wind_v850"] = np.asarray(g[lvl_names["v"]].sel(level=850).values, dtype="float64")
        frames.append(pd.DataFrame(col))

    df = pd.concat(frames, ignore_index=True)
    for c in MET_COLUMNS:
        if c not in df:
            df[c] = np.nan
    df = df[MET_COLUMNS]
    return df, SourceResult("arco-era5", ok=not df.empty, rows=len(df),
                            message=f"{len(stations)} stations {start}..{end}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ingest.era5_arco", description=__doc__)
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--end", default=(dt.date.today() - dt.timedelta(days=6)).isoformat())
    ap.add_argument("--append", action="store_true",
                    help="merge into existing met_history.parquet instead of overwriting")
    args = ap.parse_args(argv)
    SETTINGS.ensure_dirs()

    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    ds = open_arco()
    out_path = SETTINGS.processed_dir / "met_history.parquet"

    # month-by-month so a long pull is resumable and prints progress
    cursor = start
    parts = []
    while cursor < end:
        nxt = min(end, (cursor.replace(day=1) + dt.timedelta(days=32)).replace(day=1))
        df, res = fetch_reanalysis_arco(start=cursor, end=nxt, ds=ds)
        print(f"  {cursor}..{nxt}: {res.rows} rows")
        parts.append(df)
        cursor = nxt

    new = pd.concat(parts, ignore_index=True)
    if args.append and out_path.exists():
        prior = pd.read_parquet(out_path)
        new = (
            pd.concat([prior, new], ignore_index=True)
            .drop_duplicates(["station_id", "ts"], keep="last")
            .sort_values(["station_id", "ts"])
        )
    write_table(new, out_path)
    print(f"wrote {out_path}: {len(new)} rows, {new['station_id'].nunique()} stations")


if __name__ == "__main__":
    main()
