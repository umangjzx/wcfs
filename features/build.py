"""Assemble the per-station hourly feature matrix.

Pipeline: pivot observations -> hourly grid joined with meteorology -> CPCB AQI ->
Inversion Strength Index -> stubble-plume transport -> aerosol->PBL feedback ->
calendar features -> lags & rolling stats.

    python -m features.build --history --start 2021-10-01 --end 2024-02-29
    python -m features.build --serving
    python -m features.build --stats
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from aqi.cpcb_aqi import aqi_category, sub_index_series
from config.settings import SETTINGS, load_stations, station_index
from features.calendar_feats import CALENDAR_FEATURE_COLUMNS, add_calendar_features
from features.feedback import FEEDBACK_FEATURE_COLUMNS, compute_feedback_features
from features.inversion import INVERSION_FEATURE_COLUMNS, compute_inversion_features
from features.stubble import STUBBLE_FEATURE_COLUMNS, compute_stubble_features
from ingest.common import read_table, sanitize_observations

POLLUTANT_COLS = {
    "PM2.5": "pm25", "PM10": "pm10", "NO2": "no2", "O3": "o3",
    "SO2": "so2", "CO": "co", "NH3": "nh3",
}
_AVG24 = ["pm25", "pm10", "no2", "so2", "nh3"]
_AVG8MAX = ["o3", "co"]

_MET_COLS = [
    "t2m", "d2m", "rh2m", "wind_speed10", "wind_dir10", "wind_u10", "wind_v10",
    "surface_pressure", "precip", "solar", "cloud", "blh",
    "t1000", "t925", "t850", "wind_u850", "wind_v850",
]

_LAG_COLS = ["pm25", "pm10", "no2", "o3", "so2", "co", "aqi",
             "t2m", "blh", "wind_speed10", "solar", "cloud",
             "isi", "self_trapping", "ventilation_index",
             "incoming_stubble_load", "stubble_index", "fire_frp_active"]
_LAGS = [1, 2, 3, 4, 6, 8, 12, 24, 36, 48]
_ROLL_COLS = ["pm25", "pm10", "no2", "blh", "isi", "ventilation_index",
              "incoming_stubble_load", "wind_speed10"]
_ROLL_WINDOWS = [6, 24]
_ROLL_EXTRA = {"pm25": ["std", "max"], "no2": ["max"]}  # {col: [aggs]} in addition to mean


# ---------------------------------------------------------------------------
def pivot_observations(obs_long: pd.DataFrame) -> pd.DataFrame:
    """Long (station_id, ts, pollutant, value) -> wide with lowercase pollutant columns, hourly."""
    if obs_long.empty:
        return pd.DataFrame(columns=["station_id", "ts", *POLLUTANT_COLS.values()])
    o = sanitize_observations(obs_long)
    o["ts"] = pd.to_datetime(o["ts"], utc=True).dt.floor("h")
    o["pollutant"] = o["pollutant"].map(POLLUTANT_COLS)
    o = o.dropna(subset=["pollutant"])
    wide = (
        o.pivot_table(index=["station_id", "ts"], columns="pollutant", values="value",
                      aggfunc="mean")
        .reset_index()
    )
    wide.columns.name = None
    for c in POLLUTANT_COLS.values():
        if c not in wide:
            wide[c] = np.nan
    return wide


def hourly_grid(obs_wide: pd.DataFrame, met: pd.DataFrame) -> pd.DataFrame:
    """Per-station continuous hourly index; left-join meteorology and forward-fill obs (<=3 h)."""
    met = met.copy()
    met["ts"] = pd.to_datetime(met["ts"], utc=True).dt.floor("h")
    met = met.sort_values(["station_id", "ts"]).drop_duplicates(["station_id", "ts"], keep="last")

    frames = []
    stations = sorted(set(met["station_id"]) | set(obs_wide.get("station_id", pd.Series(dtype=str))))
    for sid in stations:
        m = met[met["station_id"] == sid]
        o = obs_wide[obs_wide["station_id"] == sid] if not obs_wide.empty else obs_wide
        if m.empty and (o is None or o.empty):
            continue
        lo = min([x["ts"].min() for x in (m, o) if x is not None and not x.empty])
        hi = max([x["ts"].max() for x in (m, o) if x is not None and not x.empty])
        idx = pd.date_range(lo, hi, freq="h", tz="UTC")
        base = pd.DataFrame({"station_id": sid, "ts": idx})
        base = base.merge(m.drop(columns=["station_id"]), on="ts", how="left")
        if o is not None and not o.empty:
            base = base.merge(o.drop(columns=["station_id"]), on="ts", how="left")
            for c in POLLUTANT_COLS.values():
                if c in base:
                    base[c] = base[c].ffill(limit=3)
        frames.append(base)
    grid = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for c in [*POLLUTANT_COLS.values(), *_MET_COLS]:
        if c not in grid:
            grid[c] = np.nan
    return grid


def attach_station_static(grid: pd.DataFrame) -> pd.DataFrame:
    idx = station_index()
    grid = grid.copy()
    grid["lat"] = grid["station_id"].map(lambda s: idx[s].lat if s in idx else np.nan)
    grid["lon"] = grid["station_id"].map(lambda s: idx[s].lon if s in idx else np.nan)
    grid["site_type"] = grid["station_id"].map(lambda s: idx[s].site_type if s in idx else "mixed")
    grid["city"] = grid["station_id"].map(lambda s: idx[s].city if s in idx else "NCR")
    return grid


def add_aqi(grid: pd.DataFrame) -> pd.DataFrame:
    """Add instantaneous sub-indices and the proper (rolling-averaged) CPCB AQI + category."""
    g = grid.sort_values(["station_id", "ts"]).copy()
    grp = g.groupby("station_id", sort=False)

    # proper averaging windows
    avg: dict[str, pd.Series] = {}
    for c in _AVG24:
        avg[c] = grp[c].transform(lambda s: s.rolling(24, min_periods=6).mean())
    for c in _AVG8MAX:
        roll8 = grp[c].transform(lambda s: s.rolling(8, min_periods=3).mean())
        avg[c] = roll8.groupby(g["station_id"]).transform(
            lambda s: s.rolling(24, min_periods=6).max()
        )

    sub_cols = {}
    for canon, col in [("PM2.5", "pm25"), ("PM10", "pm10"), ("NO2", "no2"), ("O3", "o3"),
                       ("SO2", "so2"), ("CO", "co"), ("NH3", "nh3")]:
        g[f"si_{col}_inst"] = sub_index_series(canon, g[col].to_numpy())
        sub_cols[col] = sub_index_series(canon, avg[col].to_numpy())

    names = np.array(list(sub_cols.keys()))
    safe = np.where(np.isnan(np.vstack(list(sub_cols.values()))), -np.inf, np.vstack(list(sub_cols.values())))
    mx = safe.max(axis=0)
    all_nan = ~np.isfinite(mx)
    dom_idx = safe.argmax(axis=0)
    g["aqi"] = np.where(all_nan, np.nan, mx)
    g["aqi_dominant"] = np.where(all_nan, None, names[dom_idx])
    g["aqi_category"] = [aqi_category(v) if np.isfinite(v) else "Unknown" for v in g["aqi"]]

    inst = np.where(
        np.isnan(np.vstack([g[f"si_{c}_inst"].to_numpy() for c in POLLUTANT_COLS.values()])),
        -np.inf,
        np.vstack([g[f"si_{c}_inst"].to_numpy() for c in POLLUTANT_COLS.values()]),
    ).max(axis=0)
    g["aqi_inst"] = np.where(np.isfinite(inst), inst, np.nan)
    return g


def add_lags(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    g = df.groupby("station_id", sort=False)
    new = {f"{c}_lag{L}": g[c].shift(L) for c in cols if c in df for L in lags}
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_rollings(df: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    """Backward-looking rolling stats (shifted 1 h so no target leakage)."""
    new = {}
    g = df.groupby("station_id", sort=False)
    for c in cols:
        if c not in df:
            continue
        aggs = ["mean", *_ROLL_EXTRA.get(c, [])]
        for w in windows:
            for agg in aggs:
                suffix = "roll" if agg == "mean" else f"roll{agg}"
                new[f"{c}_{suffix}{w}"] = g[c].transform(
                    lambda s, w=w, agg=agg: getattr(
                        s.shift(1).rolling(w, min_periods=max(2, w // 3)), agg
                    )()
                )
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction + regime features assembled from the merged grid (pre-lag)."""
    d = df.copy()
    g = d.groupby("station_id", sort=False)

    d["isi_x_pm25"] = d["isi"] * d["pm25"]
    d["stubble_x_isi"] = d.get("stubble_index", 0.0) * d["isi"]
    d["pm25_over_pm10"] = d["pm25"] / d["pm10"].where(d["pm10"] > 1)

    # boundary-layer / ventilation tendencies
    d["blh_tend_6h"] = d["blh"] - g["blh"].shift(6)
    d["vent_tend_6h"] = d["ventilation_index"] - g["ventilation_index"].shift(6)

    # near-surface wind steadiness over 6 h (vector mean / scalar mean); low => stagnant
    umean = g["wind_u10"].transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
    vmean = g["wind_v10"].transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
    spdmean = g["wind_speed10"].transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
    d["wind_steadiness_6h"] = (np.hypot(umean, vmean) / spdmean.where(spdmean > 0.1)).clip(0, 1)

    # hours since meaningful rain (scavenging); resets on precip > 0.2 mm/h
    def _since_rain(s: pd.Series) -> pd.Series:
        wet = s.fillna(0) > 0.2
        grp = wet.cumsum()
        return (~wet).groupby(grp).cumsum()

    d["hours_since_rain"] = g["precip"].transform(_since_rain).clip(upper=240)

    lh = d["local_hour"]
    d["is_morning_rush"] = lh.between(7, 10).astype("int8")
    d["is_evening_peak"] = lh.between(18, 22).astype("int8")
    return d.copy()  # de-fragment before the lag/roll concats


# ---------------------------------------------------------------------------
def build_matrix(obs_long: pd.DataFrame, met: pd.DataFrame,
                 fires: pd.DataFrame | None) -> pd.DataFrame:
    if met is None or met.empty:
        raise ValueError("meteorology frame is empty — run ingest.weather first")
    stations = load_stations()

    grid = hourly_grid(pivot_observations(obs_long), met)
    grid = attach_station_static(grid)
    grid = add_aqi(grid)

    inv = compute_inversion_features(grid)[["station_id", "ts", *INVERSION_FEATURE_COLUMNS]]
    grid = grid.merge(inv, on=["station_id", "ts"], how="left")

    stub = compute_stubble_features(grid, fires, stations)
    grid = grid.merge(stub, on=["station_id", "ts"], how="left")

    fb = compute_feedback_features(grid)
    grid = grid.merge(fb, on=["station_id", "ts"], how="left")

    grid = add_calendar_features(grid)
    grid = grid.sort_values(["station_id", "ts"]).reset_index(drop=True)
    grid = add_derived_features(grid)
    grid = add_lags(grid, _LAG_COLS, _LAGS)
    grid = add_rollings(grid, _ROLL_COLS, _ROLL_WINDOWS)
    if "pm25_roll24" in grid:
        grid["pm25_anom"] = grid["pm25"] - grid["pm25_roll24"]
    return grid.copy()  # de-fragment after the many column adds


FEATURE_GROUPS = {
    "meteorology": _MET_COLS,
    "inversion": INVERSION_FEATURE_COLUMNS,
    "stubble": STUBBLE_FEATURE_COLUMNS,
    "feedback": FEEDBACK_FEATURE_COLUMNS,
    "calendar": CALENDAR_FEATURE_COLUMNS,
    "pollutants": list(POLLUTANT_COLS.values()),
}


# ---------------------------------------------------------------------------
def _load_obs_history(start: dt.date, end: dt.date) -> pd.DataFrame:
    parts = []
    for name in ("obs_history.parquet", "obs.parquet"):
        p = SETTINGS.processed_dir / name
        if p.exists():
            parts.append(read_table(p))
    if not parts:
        return pd.DataFrame(columns=["station_id", "ts", "pollutant", "value", "source"])
    obs = pd.concat(parts, ignore_index=True)
    obs["ts"] = pd.to_datetime(obs["ts"], utc=True)
    mask = (obs["ts"] >= pd.Timestamp(start, tz="UTC")) & (obs["ts"] < pd.Timestamp(end, tz="UTC"))
    return obs[mask]


def _cmd_history(start: str, end: str) -> None:
    SETTINGS.ensure_dirs()
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    obs = _load_obs_history(s, e)
    met_p = SETTINGS.processed_dir / "met_history.parquet"
    if not met_p.exists():
        raise SystemExit("data/processed/met_history.parquet missing — run "
                         "`python -m ingest.weather --history` first")
    met = read_table(met_p)
    met = met[(pd.to_datetime(met["ts"], utc=True) >= pd.Timestamp(s, tz="UTC"))
              & (pd.to_datetime(met["ts"], utc=True) < pd.Timestamp(e, tz="UTC"))]
    fires_p = SETTINGS.processed_dir / "fires_history.parquet"
    fires = read_table(fires_p) if fires_p.exists() else None

    feats = build_matrix(obs, met, fires)
    out = SETTINGS.processed_dir / "features.parquet"
    feats.to_parquet(out, index=False)
    print(f"wrote {out}: {feats.shape[0]} rows x {feats.shape[1]} cols, "
          f"{feats['station_id'].nunique()} stations, "
          f"{feats['ts'].min()} .. {feats['ts'].max()}")


def _cmd_serving() -> None:
    SETTINGS.ensure_dirs()
    obs_p = SETTINGS.processed_dir / "obs.parquet"
    obs = read_table(obs_p) if obs_p.exists() else pd.DataFrame(
        columns=["station_id", "ts", "pollutant", "value", "source"])
    if not obs.empty:
        obs["ts"] = pd.to_datetime(obs["ts"], utc=True)
        obs = obs[obs["ts"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)]
    met_p = SETTINGS.interim_dir / "weather_forecast.parquet"
    if not met_p.exists():
        raise SystemExit("data/interim/weather_forecast.parquet missing — run "
                         "`python -m ingest.weather --forecast` first")
    met = read_table(met_p)
    fires_p = SETTINGS.interim_dir / "fires_recent.parquet"
    fires = read_table(fires_p) if fires_p.exists() else None

    feats = build_matrix(obs, met, fires)
    out = SETTINGS.interim_dir / "features_serving.parquet"
    feats.to_parquet(out, index=False)
    print(f"wrote {out}: {feats.shape[0]} rows x {feats.shape[1]} cols")


def _cmd_stats() -> None:
    p = SETTINGS.processed_dir / "features.parquet"
    if not p.exists():
        print("no data/processed/features.parquet yet")
        return
    df = pd.read_parquet(p)
    print(f"rows={len(df)} cols={df.shape[1]} stations={df['station_id'].nunique()}")
    for grp, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in df]
        cov = df[present].notna().mean().mean() if present else 0
        print(f"  {grp:12s} {len(present):2d} cols  mean non-null {cov:.0%}")
    if "isi" in df and "pm25" in df:
        sub = df[["isi", "incoming_stubble_load", "pm25"]].dropna()
        if len(sub) > 100:
            print(f"  corr(isi, pm25)                 = {sub['isi'].corr(sub['pm25']):+.3f}")
            print(f"  corr(incoming_stubble_load,pm25) = "
                  f"{sub['incoming_stubble_load'].corr(sub['pm25']):+.3f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="features.build", description=__doc__)
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--serving", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    args = ap.parse_args(argv)

    if args.serving:
        _cmd_serving()
    elif args.stats:
        _cmd_stats()
    else:
        _cmd_history(args.start, args.end)


if __name__ == "__main__":
    main()
