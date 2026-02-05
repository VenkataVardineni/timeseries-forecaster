from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def fit_arima(
    y_train: np.ndarray,
    order: Tuple[int, int, int] = (2, 1, 2),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> ARIMA:
    """
    Fit ARIMA model on training data.

    Args:
        y_train: 1D array of target values
        order: (p, d, q) order
        seasonal_order: (P, D, Q, s) seasonal order

    Returns:
        Fitted ARIMA model
    """
    model = ARIMA(y_train, order=order, seasonal_order=seasonal_order)
    fitted = model.fit()
    return fitted


def forecast_arima(
    model: ARIMA,
    horizon: int,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate multi-step forecast with quantiles.

    Args:
        model: Fitted ARIMA model
        horizon: Number of steps ahead to forecast
        alpha: Confidence level for intervals (e.g., 0.1 for 10th/90th percentiles)

    Returns:
        y_mean: [horizon] mean forecast
        y_p10: [horizon] 10th percentile
        y_p90: [horizon] 90th percentile
    """
    forecast_result = model.get_forecast(steps=horizon)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=alpha)

    # Handle both pandas Series and numpy array
    if hasattr(forecast, 'values'):
        y_mean = forecast.values
    else:
        y_mean = np.array(forecast)
    
    # Handle both pandas DataFrame and numpy array for confidence intervals
    if hasattr(conf_int, 'iloc'):
        y_p10 = conf_int.iloc[:, 0].values
        y_p90 = conf_int.iloc[:, 1].values
    else:
        conf_int = np.array(conf_int)
        y_p10 = conf_int[:, 0]
        y_p90 = conf_int[:, 1]

    return y_mean, y_p10, y_p90


def predict_fold_arima(
    train_series: np.ndarray,
    test_contexts: np.ndarray,
    order: Tuple[int, int, int] = (2, 1, 2),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    horizon: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Fit ARIMA on training series and predict test horizons.

    Args:
        train_series: 1D array of target values for training
        test_contexts: [num_test, context_length] context windows for test samples
        order: ARIMA order
        seasonal_order: Seasonal order
        horizon: Forecast horizon

    Returns:
        y_pred_mean: [num_test, horizon] mean forecasts
        y_pred_p10: [num_test, horizon] 10th percentile
        y_pred_p90: [num_test, horizon] 90th percentile
        diagnostics: Dict with model parameters and diagnostics
    """
    num_test = test_contexts.shape[0]
    y_pred_mean = []
    y_pred_p10 = []
    y_pred_p90 = []

    diagnostics = {
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "aic": None,
        "bic": None,
    }

    # Fit ARIMA on training series
    model = fit_arima(train_series, order=order, seasonal_order=seasonal_order)
    diagnostics["aic"] = float(model.aic)
    diagnostics["bic"] = float(model.bic)

    # For each test sample, extend training series with its context and forecast
    for i in range(num_test):
        # Extend training series with test context (last values)
        # Use the last context_length values from test context
        test_context = test_contexts[i, :]
        extended_series = np.concatenate([train_series, test_context])

        # Refit model with extended series (more accurate but slower)
        # For speed, we could use the existing model, but refitting is more correct
        extended_model = fit_arima(extended_series, order=order, seasonal_order=seasonal_order)

        # Forecast horizon steps ahead
        y_mean, y_p10, y_p90 = forecast_arima(extended_model, horizon=horizon, alpha=0.1)

        y_pred_mean.append(y_mean)
        y_pred_p10.append(y_p10)
        y_pred_p90.append(y_p90)

    y_pred_mean = np.array(y_pred_mean)
    y_pred_p10 = np.array(y_pred_p10)
    y_pred_p90 = np.array(y_pred_p90)

    return y_pred_mean, y_pred_p10, y_pred_p90, diagnostics

