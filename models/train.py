"""Train and persist the production forecaster (PM2.5 + PM10 + NO2 -> real AQI).

    python -m models.train                 # all targets
    python -m models.train --fast          # PM2.5 only (quick)
"""

from __future__ import annotations

import argparse

import pandas as pd

from config.settings import SETTINGS
from models.baseline_lgbm import REGISTRY
from models.baseline_lgbm import train as train_lgbm
from models.conformal import coverage_report
from models.dataset import make_supervised


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="models.train", description=__doc__)
    ap.add_argument("--model", choices=["lgbm"], default="lgbm")
    ap.add_argument("--features", default=str(SETTINGS.processed_dir / "features.parquet"))
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fast", action="store_true", help="PM2.5 target only")
    ap.add_argument("--num-boost", type=int, default=None)
    args = ap.parse_args(argv)

    feat = pd.read_parquet(args.features)
    print(f"features: {feat.shape[0]} rows x {feat.shape[1]} cols, "
          f"{feat['station_id'].nunique()} stations, {feat['ts'].min()} .. {feat['ts'].max()}",
          flush=True)

    mf = train_lgbm(feat, targets=["pm25"] if args.fast else None,
                    base_stride_h=args.stride, num_boost=args.num_boost)
    mf.save()

    sup, _ = make_supervised(feat, mf.horizons, target="pm25", base_stride_h=6)
    cov = coverage_report(mf.by_target["pm25"].predict(sup), sup["target"].to_numpy())
    print(f"saved {list(mf.by_target)} -> {REGISTRY}")
    print(f"pm25 P10-P90 conformal coverage: {cov['overall']:.0%} (target 80%)")


if __name__ == "__main__":
    main()
