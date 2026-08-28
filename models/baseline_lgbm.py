"""LightGBM multi-horizon quantile forecaster(s).

Per pollutant target (PM2.5, PM10, NO2): an L1-regression median + P10/P90 quantile
boosters, ``horizon`` as an input feature so one model spans 1..72 h. Rare high-pollution
rows are up-weighted; the P10-P90 interval is calibrated by split-conformal (CQR) on a
held-out slice. ``MultiForecaster`` runs all three and derives the real CPCB AQI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from aqi.cpcb_aqi import aqi_category, sub_index_series
from config.settings import SETTINGS
from models.conformal import apply_margins, coverage_report, cqr_margins
from models.dataset import DEFAULT_HORIZONS, TARGETS, encode_categoricals, make_supervised

REGISTRY = Path(__file__).resolve().parent / "registry"

_LGB_PARAMS = dict(
    n_estimators=320,
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
_PERSIST_BLEND_H = 6
_CAL_FRACTION = 0.15  # tail of the training window held out for conformal calibration


class _BoosterWrap:
    def __init__(self, booster):
        self.booster_ = booster

    def predict(self, X):
        return self.booster_.predict(X)


@dataclass
class LGBMForecaster:
    models: dict
    feature_cols: list[str]
    categorical: list[str]
    target: str
    horizons: list[int]
    conformal: dict[str, float] = field(default_factory=lambda: {"_global": 0.0})

    def _cols(self) -> tuple[str, str, str]:
        t = self.target
        return f"{t}_p10", f"{t}_p50", f"{t}_p90"

    def predict(self, sup: pd.DataFrame) -> pd.DataFrame:
        X, _ = encode_categoricals(sup[self.feature_cols], self.categorical)
        hz = sup["horizon"].to_numpy()
        out = pd.DataFrame({"station_id": sup["station_id"].to_numpy(), "horizon": hz})
        out["ts0"] = (sup["ts"] if "ts" in sup else sup["ts0"]).to_numpy()
        out["valid_ts"] = pd.to_datetime(out["ts0"], utc=True) + pd.to_timedelta(hz, unit="h")

        p50 = np.clip(self.models["median"].predict(X), 0, None)
        p10 = np.clip(self.models[0.1].predict(X), 0, None)
        p90 = np.clip(self.models[0.9].predict(X), 0, None)

        if self.target in sup.columns:
            w = np.clip(hz / _PERSIST_BLEND_H, 0, 1)
            p50 = w * p50 + (1 - w) * sup[self.target].to_numpy()
            p10 = np.minimum(p10, p50)
            p90 = np.maximum(p90, p50)

        p10, p90 = apply_margins(p10, p90, hz, self.conformal)
        c10, c50, c90 = self._cols()
        stk = np.sort(np.vstack([p10, p50, p90]).T, axis=1)
        out[c10], out[c50], out[c90] = stk[:, 0], stk[:, 1], stk[:, 2]
        return out

    def calibrate(self, sup_cal: pd.DataFrame) -> LGBMForecaster:
        Xc = encode_categoricals(sup_cal[self.feature_cols], self.categorical)[0]
        raw10 = self.models[0.1].predict(Xc)
        raw90 = self.models[0.9].predict(Xc)
        self.conformal = cqr_margins(
            sup_cal["target"].to_numpy(), np.clip(raw10, 0, None), np.clip(raw90, 0, None),
            sup_cal["horizon"].to_numpy(),
        )
        return self

    # -- persistence ---------------------------------------------------
    def save(self, path: Path | str = REGISTRY) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        km = {"median": "median", 0.1: "q10", 0.9: "q90"}
        for k, m in self.models.items():
            m.booster_.save_model(str(path / f"lgbm_{self.target}_{km[k]}.txt"))
        (path / f"lgbm_{self.target}_meta.json").write_text(json.dumps({
            "feature_cols": self.feature_cols, "categorical": self.categorical,
            "target": self.target, "horizons": self.horizons, "conformal": self.conformal,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, target: str = "pm25", path: Path | str = REGISTRY) -> LGBMForecaster:
        import lightgbm as lgb

        path = Path(path)
        meta = json.loads((path / f"lgbm_{target}_meta.json").read_text(encoding="utf-8"))
        models = {
            "median": _BoosterWrap(lgb.Booster(model_file=str(path / f"lgbm_{target}_median.txt"))),
            0.1: _BoosterWrap(lgb.Booster(model_file=str(path / f"lgbm_{target}_q10.txt"))),
            0.9: _BoosterWrap(lgb.Booster(model_file=str(path / f"lgbm_{target}_q90.txt"))),
        }
        return cls(models, meta["feature_cols"], meta["categorical"], meta["target"],
                   meta["horizons"], meta.get("conformal", {"_global": 0.0}))


def _sample_weight(y: np.ndarray) -> np.ndarray:
    """Up-weight high-pollution rows so Very Poor / Severe episodes aren't averaged away."""
    return np.clip(1.0 + (np.clip(y, 0, None) / 120.0) ** 1.6, 1.0, 8.0)


def train_target(
    feat: pd.DataFrame, target: str, *, horizons: list[int] | None = None,
    base_stride_h: int = 3, num_boost: int | None = None,
) -> LGBMForecaster:
    import lightgbm as lgb

    horizons = horizons or DEFAULT_HORIZONS
    sup, cols = make_supervised(feat, horizons, target=target, base_stride_h=base_stride_h)
    sup, cat = encode_categoricals(sup, cols)

    # time-ordered split: tail slice held out for conformal calibration
    cut = sup["ts"].quantile(1 - _CAL_FRACTION)
    fit = sup[sup["ts"] <= cut]
    cal = sup[sup["ts"] > cut]
    Xf, yf = fit[cols], fit["target"].to_numpy("float64")
    w = _sample_weight(yf)

    params = dict(_LGB_PARAMS)
    if num_boost:
        params["n_estimators"] = num_boost

    models: dict = {}
    m = lgb.LGBMRegressor(objective="regression_l1", **params)
    m.fit(Xf, yf, sample_weight=w, categorical_feature=cat)
    models["median"] = m
    for q in (0.1, 0.9):
        est = lgb.LGBMRegressor(objective="quantile", alpha=q, **params)
        est.fit(Xf, yf, sample_weight=w, categorical_feature=cat)
        models[float(q)] = est

    fc = LGBMForecaster(models, cols, cat, target, horizons)
    fc.calibrate(cal if len(cal) > 500 else sup)
    return fc


@dataclass
class MultiForecaster:
    by_target: dict[str, LGBMForecaster]

    @property
    def horizons(self) -> list[int]:
        return next(iter(self.by_target.values())).horizons

    @property
    def feature_cols(self) -> list[str]:
        return next(iter(self.by_target.values())).feature_cols

    @property
    def categorical(self) -> list[str]:
        return next(iter(self.by_target.values())).categorical

    def predict(self, sup: pd.DataFrame) -> pd.DataFrame:
        # each fc.predict returns rows in sup order -> combine columns positionally
        base = None
        for t, fc in self.by_target.items():
            p = fc.predict(sup).reset_index(drop=True)
            if base is None:
                base = p
            else:
                for c in (f"{t}_p10", f"{t}_p50", f"{t}_p90"):
                    base[c] = p[c].to_numpy()
        # real CPCB AQI from the P50 of each pollutant
        canon = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2"}
        sub = np.vstack([
            sub_index_series(canon[t], base[f"{t}_p50"].to_numpy())
            for t in self.by_target if f"{t}_p50" in base
        ])
        with np.errstate(invalid="ignore"):
            aqi = np.nanmax(np.where(np.isnan(sub), -np.inf, sub), axis=0)
        base["aqi"] = np.where(np.isfinite(aqi), np.round(aqi), np.nan)
        base["aqi_p50"] = base["aqi"]
        names = np.array([canon[t] for t in self.by_target if f"{t}_p50" in base])
        dom = names[np.argmax(np.where(np.isnan(sub), -np.inf, sub), axis=0)]
        base["dominant_pollutant"] = np.where(np.isfinite(aqi), dom, "PM2.5")
        base["category"] = [aqi_category(a) if np.isfinite(a) else "Unknown" for a in base["aqi"]]
        return base

    def save(self, path: Path | str = REGISTRY) -> None:
        for fc in self.by_target.values():
            fc.save(path)
        Path(path).joinpath("registry_index.json").write_text(
            json.dumps({"targets": list(self.by_target)}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str = REGISTRY) -> MultiForecaster:
        idx = Path(path) / "registry_index.json"
        targets = json.loads(idx.read_text(encoding="utf-8"))["targets"] if idx.exists() else ["pm25"]
        return cls({t: LGBMForecaster.load(t, path) for t in targets})


def train(feat: pd.DataFrame, *, targets: list[str] | None = None,
          base_stride_h: int = 3, num_boost: int | None = None,
          horizons: list[int] | None = None) -> MultiForecaster:
    targets = targets or TARGETS
    return MultiForecaster({
        t: train_target(feat, t, horizons=horizons, base_stride_h=base_stride_h, num_boost=num_boost)
        for t in targets
    })


def _cmd(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="models.baseline_lgbm")
    ap.add_argument("--features", default=str(SETTINGS.processed_dir / "features.parquet"))
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args(argv)
    feat = pd.read_parquet(args.features)
    mf = train(feat, base_stride_h=args.stride)
    mf.save()
    sup, _ = make_supervised(feat, mf.horizons, target="pm25", base_stride_h=6)
    cov = coverage_report(mf.by_target["pm25"].predict(sup), sup["target"].to_numpy())
    print(f"trained {list(mf.by_target)} -> {REGISTRY}; pm25 P10-P90 coverage {cov['overall']:.0%}")


if __name__ == "__main__":
    _cmd()
