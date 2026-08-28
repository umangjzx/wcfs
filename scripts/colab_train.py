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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(mod: str, *args: str) -> None:
    print(f"\n$ python -m {mod} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-m", mod, *args], check=True, cwd=ROOT)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-02-15")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--num-boost", type=int, default=400)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args(argv)
    s, e = args.start, args.end

    if not args.skip_ingest:
        _run("ingest.openaq", "--start", s, "--end", e)          # real CPCB ground truth (S3, keyless)
        _run("ingest.cpcb", "--history", "--source", "cams", "--append", "--start", s, "--end", e)
        _run("ingest.weather", "--history", "--start", s, "--end", e)
        try:
            _run("ingest.firms", "--history", "--start", s, "--end", e)   # needs FIRMS_MAP_KEY
        except subprocess.CalledProcessError:
            print("FIRMS skipped (no key) — stubble transport uses the wind fallback")

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
