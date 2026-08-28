"""VayuCast forecasting models.

- dataset.make_supervised          -> multi-horizon supervised table from the feature matrix
- baseline_lgbm.LGBMForecaster     -> per-target LightGBM P10/P50/P90 (L1 median, CQR interval)
- baseline_lgbm.MultiForecaster    -> PM2.5 + PM10 + NO2 -> real CPCB AQI
- baseline_lgbm.train              -> fit + conformal-calibrate all targets
- conformal.cqr_margins            -> split-conformal (CQR) interval calibration
- backtest.run                     -> walk-forward evaluation vs persistence + climatology
"""

from models.baseline_lgbm import LGBMForecaster, MultiForecaster, train
from models.conformal import coverage_report, cqr_margins
from models.dataset import DEFAULT_HORIZONS, TARGETS, make_supervised
from models.drivers import explain, explain_station_forecast
from models.serving import forecast, naive_forecast, peak_alerts

__all__ = [
    "LGBMForecaster", "MultiForecaster", "train", "make_supervised", "DEFAULT_HORIZONS",
    "TARGETS", "cqr_margins", "coverage_report",
    "explain", "explain_station_forecast", "forecast", "naive_forecast", "peak_alerts",
]
