"""Walk-forward backtest for the PM2.5 / AQI forecaster.

Expanding train window, fixed test window, a gap in between so lag/rolling features never
straddle the split. Reports per-horizon MAE/RMSE/bias, quantile pinball loss + P10-P90
coverage, AQI-category accuracy, and POD/FAR/CSI for "Very Poor" (AQI>=301) and "Severe"
(AQI>=401) — against persistence and hour-of-year climatology baselines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aqi.cpcb_aqi import aqi_category, sub_index_series
from config.settings import SETTINGS
from models.baseline_lgbm import REGISTRY, train
from models.dataset import DEFAULT_HORIZONS, make_supervised

_EVENTS = {"very_poor": 301, "severe": 401}


@dataclass
class Fold:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_folds(feat: pd.DataFrame, n_folds: int = 3, test_days: int = 21,
                       gap_days: int = 2, min_train_days: int = 21) -> list[Fold]:
    ts = pd.to_datetime(feat["ts"], utc=True)
    t0, t1 = ts.min(), ts.max()
    span_days = (t1 - t0).days
    need = min_train_days + gap_days + n_folds * test_days
    if span_days < need:
        n_folds = max(1, (span_days - min_train_days - gap_days) // test_days)
    folds = []
    for k in range(n_folds):
        test_end = t1 - pd.Timedelta(days=test_days * (n_folds - 1 - k))
        test_start = test_end - pd.Timedelta(days=test_days)
        train_end = test_start - pd.Timedelta(days=gap_days)
        if train_end - t0 < pd.Timedelta(days=min_train_days):
            continue
        folds.append(Fold(train_end, test_start, test_end))
    return folds


def _pinball(y, q_pred, q):
    d = y - q_pred
    return np.mean(np.maximum(q * d, (q - 1) * d))


def _event_scores(pred_flag: np.ndarray, obs_flag: np.ndarray) -> dict:
    hits = int(np.sum(pred_flag & obs_flag))
    miss = int(np.sum(~pred_flag & obs_flag))
    fa = int(np.sum(pred_flag & ~obs_flag))
    pod = hits / (hits + miss) if hits + miss else np.nan
    far = fa / (hits + fa) if hits + fa else np.nan
    csi = hits / (hits + miss + fa) if hits + miss + fa else np.nan
    return {"POD": round(pod, 3), "FAR": round(far, 3), "CSI": round(csi, 3),
            "n_events": hits + miss}


def _climatology(train_feat: pd.DataFrame, target: str) -> pd.Series:
    t = train_feat.copy()
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    key = [t["station_id"], t["ts"].dt.month, t["ts"].dt.hour]
    return t.groupby(key)[target].mean()


def evaluate_fold(feat: pd.DataFrame, fold: Fold, horizons: list[int], target: str,
                  stride: int) -> pd.DataFrame:
    ts = pd.to_datetime(feat["ts"], utc=True)
    tr = feat[ts <= fold.train_end]
    te = feat[(ts > fold.test_start) & (ts <= fold.test_end)]
    if len(tr) < 500 or len(te) < 200:
        return pd.DataFrame()

    print(f"  fold train<= {fold.train_end.date()} ({len(tr)} feat rows) "
          f"test {fold.test_start.date()}..{fold.test_end.date()}", flush=True)
    fc = train(tr, horizons=horizons, target=target, base_stride_h=stride, num_boost=250)
    sup_te, _ = make_supervised(te, horizons, target=target, base_stride_h=2)
    pred = fc.predict(sup_te)

    df = pred.copy()
    df["y"] = sup_te["target"].to_numpy()
    df["persist"] = sup_te[target].to_numpy()  # t0 value carried forward

    clim = _climatology(tr, target)
    m = pd.to_datetime(df["ts0"], utc=True).dt.month
    h = pd.to_datetime(df["valid_ts"], utc=True).dt.hour
    df["clim"] = [clim.get((s, mm, hh), np.nan)
                  for s, mm, hh in zip(df["station_id"], m, h, strict=False)]
    df["clim"] = df["clim"].fillna(df["persist"])
    return df


def summarize(df: pd.DataFrame) -> dict:
    out: dict = {"n": int(len(df)), "by_horizon": {}, "overall": {}, "events": {}}
    y = df["y"].to_numpy()
    for name, col in [("model", "pm25_p50"), ("persistence", "persist"), ("climatology", "clim")]:
        p = df[col].to_numpy()
        err = p - y
        out["overall"][name] = {
            "MAE": round(float(np.mean(np.abs(err))), 2),
            "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 2),
            "bias": round(float(np.mean(err)), 2),
        }
    out["overall"]["model"]["pinball_p10"] = round(_pinball(y, df["pm25_p10"].to_numpy(), 0.1), 2)
    out["overall"]["model"]["pinball_p90"] = round(_pinball(y, df["pm25_p90"].to_numpy(), 0.9), 2)
    out["overall"]["model"]["cov_p10_p90"] = round(float(np.mean(
        (y >= df["pm25_p10"].to_numpy()) & (y <= df["pm25_p90"].to_numpy()))), 3)

    for hz, g in df.groupby("horizon"):
        e_m = g["pm25_p50"].to_numpy() - g["y"].to_numpy()
        e_p = g["persist"].to_numpy() - g["y"].to_numpy()
        out["by_horizon"][int(hz)] = {
            "MAE_model": round(float(np.mean(np.abs(e_m))), 2),
            "MAE_persist": round(float(np.mean(np.abs(e_p))), 2),
            "skill_vs_persist": round(1 - np.mean(np.abs(e_m)) / max(np.mean(np.abs(e_p)), 1e-6), 3),
        }

    y_aqi = sub_index_series("PM2.5", y)
    p_aqi = df["aqi_p50"].to_numpy()
    valid = np.isfinite(y_aqi) & np.isfinite(p_aqi)
    out["overall"]["aqi_category_acc"] = round(float(np.mean(
        [aqi_category(a) == aqi_category(b) for a, b in zip(p_aqi[valid], y_aqi[valid], strict=False)]
    )), 3)
    for ev, thr in _EVENTS.items():
        out["events"][ev] = {
            "model": _event_scores(p_aqi >= thr, y_aqi >= thr),
            "persistence": _event_scores(
                sub_index_series("PM2.5", df["persist"].to_numpy()) >= thr, y_aqi >= thr),
        }
    return out


def run(feat: pd.DataFrame, *, horizons: list[int] | None = None, target: str = "pm25",
        n_folds: int = 3, stride: int = 2) -> dict:
    horizons = horizons or DEFAULT_HORIZONS
    feat = feat.sort_values(["station_id", "ts"]).reset_index(drop=True)
    folds = walk_forward_folds(feat, n_folds=n_folds)
    if not folds:
        raise SystemExit("not enough history for a walk-forward backtest — widen the window")
    frames = [evaluate_fold(feat, f, horizons, target, stride) for f in folds]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("backtest produced no evaluable rows")
    allrows = pd.concat(frames, ignore_index=True)
    summary = summarize(allrows)
    summary["folds"] = [
        {"train_end": str(f.train_end), "test": [str(f.test_start), str(f.test_end)]}
        for f in folds
    ]
    return summary


def _print(summary: dict) -> None:
    o = summary["overall"]
    print(f"\nrows evaluated: {summary['n']}")
    print("\n            MAE     RMSE    bias")
    for k in ("model", "persistence", "climatology"):
        v = o[k]
        print(f"  {k:12s}{v['MAE']:7.1f}{v['RMSE']:8.1f}{v['bias']:8.1f}")
    print(f"\n  quantile pinball P10={o['model']['pinball_p10']}  P90={o['model']['pinball_p90']}"
          f"  |  P10-P90 coverage {o['model']['cov_p10_p90']:.0%} (target 80%)")
    print(f"  AQI category accuracy: {o['aqi_category_acc']:.0%}")
    print("\n  horizon  MAE_model  MAE_persist  skill")
    for hz, r in summary["by_horizon"].items():
        print(f"  {hz:5d}h {r['MAE_model']:9.1f} {r['MAE_persist']:11.1f} {r['skill_vs_persist']:+7.2f}")
    print("\n  event         model POD/FAR/CSI        persist POD/FAR/CSI     n")
    for ev, r in summary["events"].items():
        m, p = r["model"], r["persistence"]
        print(f"  {ev:12s} {m['POD']}/{m['FAR']}/{m['CSI']}      "
              f"{p['POD']}/{p['FAR']}/{p['CSI']}   {m['n_events']}")


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="models.backtest", description=__doc__)
    ap.add_argument("--features", default=str(SETTINGS.processed_dir / "features.parquet"))
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args(argv)

    feat = pd.read_parquet(args.features)
    summary = run(feat, n_folds=args.folds, stride=args.stride)
    _print(summary)
    (REGISTRY).mkdir(parents=True, exist_ok=True)
    (REGISTRY / "backtest_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved -> {REGISTRY / 'backtest_metrics.json'}")


if __name__ == "__main__":
    main()
