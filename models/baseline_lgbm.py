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
        keymap = {"median": "median", 0.1: "q10", 0.9: "q90"}
        for k, m in self.models.items():
            m.booster_.save_model(str(path / f"lgbm_pm25_{keymap[k]}.txt"))
        (path / "lgbm_meta.json").write_text(json.dumps({
            "feature_cols": self.feature_cols,
            "categorical": self.categorical,
            "target": self.target,
            "horizons": self.horizons,
            "interval_k": self.interval_k,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str = REGISTRY) -> LGBMForecaster:
        import lightgbm as lgb

        path = Path(path)
        meta = json.loads((path / "lgbm_meta.json").read_text(encoding="utf-8"))
        models = {
            "median": _BoosterWrap(lgb.Booster(model_file=str(path / "lgbm_pm25_median.txt"))),
            0.1: _BoosterWrap(lgb.Booster(model_file=str(path / "lgbm_pm25_q10.txt"))),
            0.9: _BoosterWrap(lgb.Booster(model_file=str(path / "lgbm_pm25_q90.txt"))),
        }
        fc = cls(models, meta["feature_cols"], meta["categorical"],
                 meta["target"], meta["horizons"])
        fc.interval_k = meta.get("interval_k", 1.0)
        return fc

    # spread multiplier so P10-P90 covers ~80% (set by calibrate(); 1.0 until then)
    interval_k: float = 1.0
    # horizons at/below which the P50 is blended toward persistence
    _persist_blend_h: int = 6

    # -- inference -----------------------------------------------------
    def predict(self, sup: pd.DataFrame) -> pd.DataFrame:
        """``sup`` = rows from make_supervised (or the serving equivalent). Returns
        station_id, ts0, horizon, valid_ts, pm25_p10/p50/p90, aqi_p50."""
        X, _ = encode_categoricals(sup[self.feature_cols], self.categorical)
        out = sup[["station_id"]].copy()
        out["ts0"] = sup["ts"] if "ts" in sup else sup.get("ts0")
        out["horizon"] = sup["horizon"].to_numpy()
        out["valid_ts"] = out["ts0"] + pd.to_timedelta(out["horizon"], unit="h")

        p50 = np.clip(self.models["median"].predict(X), 0, None)
        p10 = np.clip(self.models[0.1].predict(X), 0, None)
        p90 = np.clip(self.models[0.9].predict(X), 0, None)

        # blend toward persistence for the first few hours (current value is hard to beat)
        if "pm25" in sup.columns:
            h = out["horizon"].to_numpy()
            w = np.clip(h / self._persist_blend_h, 0, 1)
            p50 = w * p50 + (1 - w) * sup["pm25"].to_numpy()

        # widen the predictive interval by the calibrated factor, keep it centred on p50
        p10 = p50 - self.interval_k * np.maximum(p50 - p10, 0)
        p90 = p50 + self.interval_k * np.maximum(p90 - p50, 0)

        out["pm25_p10"] = np.clip(p10, 0, None)
        out["pm25_p50"] = p50
        out["pm25_p90"] = p90
        out["aqi_p50"] = add_predicted_aqi(p50)
        out[["pm25_p10", "pm25_p50", "pm25_p90"]] = np.sort(
            out[["pm25_p10", "pm25_p50", "pm25_p90"]].to_numpy(), axis=1)
        return out

    def calibrate(self, sup_cal: pd.DataFrame) -> LGBMForecaster:
        """Set ``interval_k`` so P10-P90 empirically covers ~80% on a calibration slice."""
        self.interval_k = 1.0
        pred = self.predict(sup_cal)
        y = sup_cal["target"].to_numpy()
        half = np.maximum(pred["pm25_p90"].to_numpy() - pred["pm25_p50"].to_numpy(), 1e-6)
        resid = np.abs(y - pred["pm25_p50"].to_numpy())
        self.interval_k = float(np.clip(np.quantile(resid / half, 0.8), 0.5, 6.0))
        return self


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
    sup, cat = encode_categoricals(sup, cols)
    X = sup[cols]
    y = sup["target"].to_numpy("float64")

    params = dict(_LGB_PARAMS)
    if num_boost:
        params["n_estimators"] = num_boost

    models: dict = {}
    # median: L1 regression (unbiased-ish, better MAE than pinball@0.5)
    m = lgb.LGBMRegressor(objective="regression_l1", **params)
    m.fit(X, y, categorical_feature=cat)
    models["median"] = m
    # lower / upper quantiles for the predictive interval
    for q in (0.1, 0.9):
        est = lgb.LGBMRegressor(objective="quantile", alpha=q, **params)
        est.fit(X, y, categorical_feature=cat)
        models[float(q)] = est

    fc = LGBMForecaster(models, cols, cat, target, horizons)
    fc.calibrate(sup)
    return fc


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
