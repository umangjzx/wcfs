"""Conformalized Quantile Regression (Romano, Patterson & Candès, 2019).

Split-conformal calibration of the LightGBM P10/P90 interval on a held-out slice so the
P10-P90 band achieves ~80% marginal coverage, per forecast horizon.

score  E_i = max( q_lo(x_i) - y_i ,  y_i - q_hi(x_i) )
margin Q_h = ceil((n+1)(1-alpha)) / n  empirical quantile of {E_i} within horizon h
interval   [ q_lo - Q_h ,  q_hi + Q_h ]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cqr_margins(
    y: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    horizon: np.ndarray,
    *,
    alpha: float = 0.15,  # target ~85% on the calibration split -> ~80% out-of-sample
    min_per_bin: int = 40,
) -> dict[str, float]:
    """Return {"<horizon>": margin, "_global": margin}. Falls back to the global
    margin for horizons with too few calibration points."""
    y = np.asarray(y, "float64")
    lo = np.asarray(q_lo, "float64")
    hi = np.asarray(q_hi, "float64")
    hz = np.asarray(horizon)
    scores = np.maximum(lo - y, y - hi)
    ok = np.isfinite(scores)
    scores, hz = scores[ok], hz[ok]
    if scores.size == 0:
        return {"_global": 0.0}

    def _q(s: np.ndarray) -> float:
        n = s.size
        level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        return float(np.quantile(s, level, method="higher"))

    out: dict[str, float] = {"_global": max(_q(scores), 0.0)}
    for h in np.unique(hz):
        s = scores[hz == h]
        out[str(int(h))] = max(_q(s), 0.0) if s.size >= min_per_bin else out["_global"]
    return out


def apply_margins(
    q_lo: np.ndarray, q_hi: np.ndarray, horizon: np.ndarray, margins: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    m = np.array([margins.get(str(int(h)), margins.get("_global", 0.0)) for h in np.asarray(horizon)])
    return np.clip(q_lo - m, 0, None), q_hi + m


def coverage_report(pred: pd.DataFrame, y: np.ndarray) -> dict:
    """P10-P90 empirical coverage overall and by horizon (for logging)."""
    y = np.asarray(y, "float64")
    lo = pred["pm25_p10"].to_numpy()
    hi = pred["pm25_p90"].to_numpy()
    inside = (y >= lo) & (y <= hi)
    by_h = (
        pd.DataFrame({"h": pred["horizon"].to_numpy(), "in": inside})
        .groupby("h")["in"].mean().round(3).to_dict()
    )
    return {"overall": round(float(np.mean(inside)), 3), "by_horizon": by_h}
