"""LightGBM multi-horizon quantile baseline for the 72 h PM2.5 forecast.

Three boosters (P10 / P50 / P90), each with ``horizon`` as an input feature so one model
covers the whole 1..72 h range. Fast to train, honest baseline for the TFT to beat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import QUANTILES, SETTINGS
from models.dataset import (
    DEFAULT_HORIZONS,
    add_predicted_aqi,
    encode_categoricals,
    make_supervised,
)

REGISTRY = Path(__file__).resolve().parent / "registry"

_LGB_PARAMS = dict(
    objective="quantile",
    n_estimators=350,
    learning_rate=0.06,
    num_leaves=31,
    max_bin=127,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    force_col_wise=True,
    n_jobs=-1,
    verbosity=-1,
)


@dataclass
class LGBMForecaster:
    models: dict[float, object]
    feature_cols: list[str]
    categorical: list[str]
    target: str
    horizons: list[int]

    # -- persistence -----------------------------------------------------
    def save(self, path: Path | str = REGISTRY) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for q, m in self.models.items():
            m.booster_.save_model(str(path / f"lgbm_pm25_q{int(q * 100):02d}.txt"))
        (path / "lgbm_meta.json").write_text(json.dumps({
            "feature_cols": self.feature_cols,
            "categorical": self.categorical,
            "target": self.target,
            "horizons": self.horizons,
            "quantiles": list(self.models.keys()),
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str = REGISTRY) -> LGBMForecaster:
        import lightgbm as lgb

        path = Path(path)
        meta = json.loads((path / "lgbm_meta.json").read_text(encoding="utf-8"))
        models = {}
        for q in meta["quantiles"]:
            b = lgb.Booster(model_file=str(path / f"lgbm_pm25_q{int(q * 100):02d}.txt"))
            models[float(q)] = _BoosterWrap(b)
        return cls(models, meta["feature_cols"], meta["categorical"],
                   meta["target"], meta["horizons"])

    # -- inference -----------------------------------------------------
    def predict(self, sup: pd.DataFrame) -> pd.DataFrame:
        """``sup`` = rows from make_supervised (or the serving equivalent). Returns
        station_id, ts0, horizon, valid_ts, pm25_p10/p50/p90, aqi_p50."""
        X, _ = encode_categoricals(sup[self.feature_cols], self.categorical)
        out = sup[["station_id"]].copy()
        out["ts0"] = sup["ts"] if "ts" in sup else sup.get("ts0")
        out["horizon"] = sup["horizon"]
        out["valid_ts"] = out["ts0"] + pd.to_timedelta(out["horizon"], unit="h")
        for q, m in self.models.items():
            out[f"pm25_p{int(q * 100):02d}"] = np.clip(m.predict(X), 0, None)
        if "pm25_p50" in out:
            out["aqi_p50"] = add_predicted_aqi(out["pm25_p50"].to_numpy())
        # enforce quantile monotonicity
        qcols = [c for c in out.columns if c.startswith("pm25_p")]
        out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
        return out


class _BoosterWrap:
    """Give a loaded Booster the same .predict signature as the sklearn estimator."""

    def __init__(self, booster):
        self.booster_ = booster

    def predict(self, X):
        return self.booster_.predict(X)


def train(
    feat: pd.DataFrame,
    *,
    horizons: list[int] | None = None,
    quantiles=QUANTILES,
    target: str = "pm25",
    base_stride_h: int = 2,
    num_boost: int | None = None,
) -> LGBMForecaster:
    import lightgbm as lgb

    horizons = horizons or DEFAULT_HORIZONS
    sup, cols = make_supervised(feat, horizons, target=target, base_stride_h=base_stride_h)
    sup, cat = encode_categoricals(sup, [c for c in cols])
    X = sup[cols]
    y = sup["target"].to_numpy("float64")

    params = dict(_LGB_PARAMS)
    if num_boost:
        params["n_estimators"] = num_boost

    models: dict[float, object] = {}
    for q in quantiles:
        est = lgb.LGBMRegressor(alpha=q, **params)
        est.fit(X, y, categorical_feature=cat)
        models[float(q)] = est
    return LGBMForecaster(models, cols, cat, target, horizons)


def _cmd(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="models.baseline_lgbm")
    ap.add_argument("--features", default=str(SETTINGS.processed_dir / "features.parquet"))
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args(argv)
    feat = pd.read_parquet(args.features)
    fc = train(feat, base_stride_h=args.stride)
    fc.save()
    print(f"trained LGBM P10/P50/P90 on {len(feat)} feature rows; saved to {REGISTRY}")


if __name__ == "__main__":
    _cmd()
