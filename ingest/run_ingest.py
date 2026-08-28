"""One-shot ingestion orchestrator.

``--once``    : a live refresh cycle — current CPCB obs, GFS forecast met, recent FIRMS fires.
``--history`` : the training bulk pull — CPCB/CAMS history, ERA5 reanalysis, FIRMS archive.

Every source is wrapped so one failure never aborts the cycle; results are written to
``data/interim/ingest_manifest.json`` for the API's status endpoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pandas as pd

from config.settings import SETTINGS
from ingest.common import merge_observations, write_table
from ingest.cpcb import fetch_history as cpcb_history
from ingest.cpcb import fetch_realtime
from ingest.firms import fetch_history as firms_history
from ingest.firms import fetch_recent
from ingest.weather import fetch_forecast, fetch_reanalysis


def _write_manifest(results: list, mode: str) -> None:
    manifest = {
        "mode": mode,
        "run_at": dt.datetime.now(dt.UTC).isoformat(),
        "sources": [r.as_dict() for r in results],
        "ok": all(r.ok for r in results),
    }
    path = SETTINGS.interim_dir / "ingest_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def run_once() -> None:
    SETTINGS.ensure_dirs()
    results = []

    obs, r_obs = fetch_realtime()
    results.append(r_obs)
    if not obs.empty:
        proc = SETTINGS.processed_dir / "obs.parquet"
        prior = pd.read_parquet(proc) if proc.exists() else None
        write_table(merge_observations(prior, obs), proc)
        write_table(obs, SETTINGS.interim_dir / "cpcb_realtime.parquet")

    met, r_met = fetch_forecast()
    results.append(r_met)
    if not met.empty:
        write_table(met, SETTINGS.interim_dir / "weather_forecast.parquet")

    fires, r_fire = fetch_recent(days=3)
    results.append(r_fire)
    if not fires.empty:
        write_table(fires, SETTINGS.interim_dir / "fires_recent.parquet")

    _write_manifest(results, "once")


def run_history(start: dt.date, end: dt.date) -> None:
    SETTINGS.ensure_dirs()
    results = []

    obs, r_obs = cpcb_history(start=start, end=end)
    results.append(r_obs)
    if not obs.empty:
        write_table(obs, SETTINGS.processed_dir / "obs_history.parquet")

    met, r_met = fetch_reanalysis(start=start, end=end)
    results.append(r_met)
    if not met.empty:
        write_table(met, SETTINGS.processed_dir / "met_history.parquet")

    fires, r_fire = firms_history(start=start, end=end)
    results.append(r_fire)
    if not fires.empty:
        write_table(fires, SETTINGS.processed_dir / "fires_history.parquet")

    _write_manifest(results, "history")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ingest.run_ingest", description=__doc__)
    ap.add_argument("--once", action="store_true", help="live refresh cycle (default)")
    ap.add_argument("--history", action="store_true", help="training bulk pull")
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--end", default=(dt.date.today() - dt.timedelta(days=6)).isoformat())
    args = ap.parse_args(argv)

    if args.history:
        run_history(dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end))
    else:
        run_once()


if __name__ == "__main__":
    main()
