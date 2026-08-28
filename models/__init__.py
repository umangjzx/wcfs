"""VayuCast forecasting models.

- dataset.make_supervised     -> multi-horizon supervised table from the feature matrix
- baseline_lgbm.LGBMForecaster -> LightGBM P10/P50/P90 baseline (train / save / load / predict)
- backtest.run                -> walk-forward evaluation vs persistence + climatology
"""

from models.baseline_lgbm import LGBMForecaster, train
from models.dataset import DEFAULT_HORIZONS, make_supervised
from models.drivers import explain, explain_station_forecast
from models.serving import forecast, peak_alerts

__all__ = [
    "LGBMForecaster", "train", "make_supervised", "DEFAULT_HORIZONS",
    "explain", "explain_station_forecast", "forecast", "peak_alerts",
]
