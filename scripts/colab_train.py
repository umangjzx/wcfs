"""End-to-end training run, designed for a fresh Colab (T4) VM.

    python scripts/colab_train.py --start 2025-10-01 --end 2026-02-15

Pulls data (OpenAQ S3 + Open-Meteo, keyless; FIRMS + data.gov.in if keys in env),
builds the feature matrix, trains the multi-pollutant forecaster, runs the walk-forward
backtest, and zips models/registry/ -> vayucast_model.zip for download / commit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(mod: str, *args: str, check: bool = True) -> int:
    print(f"\n$ python -m {mod} {' '.join(args)}", flush=True)
    return subprocess.run([sys.executable, "-m", mod, *args], check=check, cwd=ROOT).returncode


def _run_retry(mod: str, *args: str, tries: int = 4, wait: int = 150) -> None:
    """Retry a step that hits shared-IP rate limits (Open-Meteo on Colab)."""
    for k in range(1, tries + 1):
        if _run(mod, *args, check=False) == 0:
            return
        if k < tries:
            print(f"  [{mod}] attempt {k}/{tries} failed — waiting {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"{mod} still failing after {tries} attempts — rerun this cell later")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-02-15")
    ap.add_argument("--stride", type=int, default=3)   # base-row stride; 3-4 keeps Colab RAM in bounds
    ap.add_argument("--num-boost", type=int, default=400)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args(argv)
    s, e = args.start, args.end

    proc = ROOT / "data" / "processed"

    def _rows(name: str) -> int:
        p = proc / name
        if not p.exists():
            return 0
        import pyarrow.parquet as pq
        return pq.ParquetFile(p).metadata.num_rows

    if not args.skip_ingest:
        # each step is idempotent + skipped if already done, so re-running the cell after
        # a Colab rate-limit stall resumes cheaply
        if _rows("obs_history.parquet") < 200_000:
            _run("ingest.openaq", "--start", s, "--end", e)          # real CPCB (S3, keyless)
            _run("ingest.cpcb", "--history", "--source", "cams", "--append", "--start", s, "--end", e)
        else:
            print("obs_history.parquet already populated — skipping OpenAQ/CAMS")
        if _rows("met_history.parquet") < 150_000:
            _run_retry("ingest.weather", "--history", "--strict", "--start", s, "--end", e)
        else:
            print("met_history.parquet already populated — skipping weather")
        if _rows("fires_history.parquet") == 0:
            if _run("ingest.firms", "--history", "--start", s, "--end", e, check=False) != 0:
                print("FIRMS skipped (no/blocked key) — stubble transport uses the wind fallback")

    met = ROOT / "data" / "processed" / "met_history.parquet"
    if not met.exists():
        raise SystemExit(f"{met} missing — the weather ingest never succeeded; rerun the cell")

    _run("features.build", "--history", "--start", s, "--end", e)
    _run("models.train", "--stride", str(args.stride), "--num-boost", str(args.num_boost))
    _run("models.backtest", "--folds", str(args.folds), "--stride", str(args.stride))

    reg = ROOT / "models" / "registry"
    out = ROOT / "vayucast_model"
    shutil.make_archive(str(out), "zip", reg)
    print(f"\n✅ {out}.zip  ({(out.with_suffix('.zip')).stat().st_size/1e6:.1f} MB)")
    metrics = reg / "backtest_metrics.json"
    if metrics.exists():
        m = json.loads(metrics.read_text())["overall"]
        print(f"   backtest MAE model={m['model']['MAE']}  persistence={m['persistence']['MAE']}  "
              f"climatology={m['climatology']['MAE']}")
    dt.datetime.now()  # timestamp in logs


if __name__ == "__main__":
    main()
