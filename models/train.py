"""Train and persist the production forecaster.

    python -m models.train --model lgbm --features data/processed/features.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd

from config.settings import SETTINGS
from models.baseline_lgbm import REGISTRY
from models.baseline_lgbm import train as train_lgbm


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="models.train", description=__doc__)
    ap.add_argument("--model", choices=["lgbm", "tft"], default="lgbm")
    ap.add_argument("--features", default=str(SETTINGS.processed_dir / "features.parquet"))
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args(argv)

    feat = pd.read_parquet(args.features)
    print(f"features: {feat.shape[0]} rows x {feat.shape[1]} cols, "
          f"{feat['station_id'].nunique()} stations, "
          f"{feat['ts'].min()} .. {feat['ts'].max()}")

    if args.model == "lgbm":
        fc = train_lgbm(feat, base_stride_h=args.stride)
        fc.save()
        print(f"saved LGBM forecaster -> {REGISTRY}")
    else:
        raise SystemExit("TFT training lands in plan 04-02")


if __name__ == "__main__":
    main()
