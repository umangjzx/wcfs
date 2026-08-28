"""NASA FIRMS fire-hotspot ingestion for the Punjab/Haryana stubble-burning belt.

Uses the FIRMS "area" CSV API (requires a free MAP_KEY). Individual detections are
aggregated into daily 0.1-degree clusters (summed FRP, detection count) which the
stubble-plume transport feature (Phase 3) advects toward Delhi using the wind field.

CLI:
    python -m ingest.firms --once                       # last --days days, all sensors
    python -m ingest.firms --history --start 2021-10-01 --end 2023-12-15
    python -m ingest.firms --stats
"""

from __future__ import annotations

import argparse
import datetime as dt
import io

import pandas as pd

from config.settings import SETTINGS, STUBBLE_BBOX
from ingest.common import (
    FIRE_COLUMNS,
    SourceResult,
    http_get,
    load_snapshot,
    save_snapshot,
    write_table,
)

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# NRT for recent (<~2 months), SP (standard processing) for the archive.
NRT_SOURCES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"]
ARCHIVE_SOURCES = ["VIIRS_SNPP_SP", "MODIS_SP"]

_CLUSTER_RES = 0.1  # degrees


def _bbox_param() -> str:
    lat_min, lon_min, lat_max, lon_max = STUBBLE_BBOX
    return f"{lon_min},{lat_min},{lon_max},{lat_max}"


def _confidence_to_num(series: pd.Series) -> pd.Series:
    m = {"l": 20.0, "n": 60.0, "h": 90.0, "low": 20.0, "nominal": 60.0, "high": 90.0}
    s = series.astype(str).str.strip().str.lower()
    num = pd.to_numeric(s, errors="coerce")
    return num.fillna(s.map(m)).fillna(60.0)


def parse_firms_csv(text: str, source: str) -> pd.DataFrame:
    """Parse one FIRMS area CSV response into raw detections."""
    if not text or "latitude" not in text[:200].lower():
        return pd.DataFrame(columns=["lat", "lon", "acq_date", "frp", "confidence", "source"])
    raw = pd.read_csv(io.StringIO(text))
    if raw.empty:
        return pd.DataFrame(columns=["lat", "lon", "acq_date", "frp", "confidence", "source"])
    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(raw["latitude"], errors="coerce"),
            "lon": pd.to_numeric(raw["longitude"], errors="coerce"),
            "acq_date": pd.to_datetime(raw["acq_date"], errors="coerce"),
            "frp": pd.to_numeric(raw.get("frp", 0), errors="coerce").fillna(0.0),
            "confidence": _confidence_to_num(raw.get("confidence", pd.Series(dtype=str))),
            "source": source,
        }
    )
    return out.dropna(subset=["lat", "lon", "acq_date"])


def cluster_daily(detections: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw detections into daily 0.1-degree grid clusters (FIRE_COLUMNS schema)."""
    if detections.empty:
        return pd.DataFrame(columns=FIRE_COLUMNS)
    d = detections.copy()
    d["date"] = d["acq_date"].dt.date
    d["glat"] = (d["lat"] / _CLUSTER_RES).round() * _CLUSTER_RES
    d["glon"] = (d["lon"] / _CLUSTER_RES).round() * _CLUSTER_RES
    g = (
        d.groupby(["date", "glat", "glon"])
        .agg(frp_sum=("frp", "sum"), count=("frp", "size"), confidence_mean=("confidence", "mean"))
        .reset_index()
    )
    g["cluster_id"] = (
        g["date"].astype(str) + "_" + g["glat"].round(2).astype(str) + "_" + g["glon"].round(2).astype(str)
    )
    g = g.rename(columns={"glat": "lat", "glon": "lon"})
    g["source"] = "firms"
    g["date"] = pd.to_datetime(g["date"])
    return g[FIRE_COLUMNS].sort_values(["date", "frp_sum"], ascending=[True, False]).reset_index(drop=True)


def _fetch_sources(map_key: str, sources: list[str], days: int, start: dt.date | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for src in sources:
        url = f"{FIRMS_AREA_URL}/{map_key}/{src}/{_bbox_param()}/{days}"
        if start is not None:
            url += f"/{start.isoformat()}"
        try:
            resp = http_get(url)
            frames.append(parse_firms_csv(resp.text, src))
        except Exception:  # noqa: BLE001 - one sensor down shouldn't stop the rest
            continue
    if not frames:
        return pd.DataFrame(columns=["lat", "lon", "acq_date", "frp", "confidence", "source"])
    return pd.concat(frames, ignore_index=True)


def fetch_recent(days: int = 3, map_key: str | None = None) -> tuple[pd.DataFrame, SourceResult]:
    map_key = map_key or SETTINGS.firms_map_key
    if not map_key:
        snap = load_snapshot("firms_clusters")
        return (
            snap if snap is not None else pd.DataFrame(columns=FIRE_COLUMNS),
            SourceResult("firms", ok=snap is not None, stale=True,
                         rows=0 if snap is None else len(snap),
                         message="no FIRMS_MAP_KEY; used snapshot"),
        )
    det = _fetch_sources(map_key, NRT_SOURCES, min(days, 10), None)
    clusters = cluster_daily(det)
    if not clusters.empty:
        save_snapshot("firms_clusters", clusters)
    return clusters, SourceResult("firms", ok=not clusters.empty, rows=len(clusters),
                                  message=f"{len(det)} detections -> {len(clusters)} daily clusters")


def fetch_history(start: dt.date, end: dt.date, map_key: str | None = None
                  ) -> tuple[pd.DataFrame, SourceResult]:
    map_key = map_key or SETTINGS.firms_map_key
    if not map_key:
        return pd.DataFrame(columns=FIRE_COLUMNS), SourceResult(
            "firms", ok=False, message="no FIRMS_MAP_KEY")
    all_det: list[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        span = min(10, (end - cursor).days + 1)
        all_det.append(_fetch_sources(map_key, ARCHIVE_SOURCES, span, cursor))
        cursor += dt.timedelta(days=span)
    det = pd.concat(all_det, ignore_index=True) if all_det else pd.DataFrame()
    clusters = cluster_daily(det)
    return clusters, SourceResult("firms", ok=not clusters.empty, rows=len(clusters),
                                  message=f"{len(det)} detections {start}..{end}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ingest.firms", description=__doc__)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args(argv)
    SETTINGS.ensure_dirs()

    if args.history:
        df, res = fetch_history(dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end))
        print(res.as_dict())
        if not df.empty:
            write_table(df, SETTINGS.processed_dir / "fires_history.parquet")
    elif args.stats:
        p = SETTINGS.processed_dir / "fires_history.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            print(f"rows={len(df)} dates={df['date'].nunique()} total_frp={df['frp_sum'].sum():.0f}")
        else:
            print("no fires_history.parquet yet")
    else:
        df, res = fetch_recent(args.days)
        print(res.as_dict())
        if not df.empty:
            write_table(df, SETTINGS.interim_dir / "fires_recent.parquet")


if __name__ == "__main__":
    main()
