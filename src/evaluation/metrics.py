from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def per_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Compute MAE/RMSE per horizon step.

    Args:
        y_true: [num_samples, horizon]
        y_pred: [num_samples, horizon]
    """
    assert y_true.shape == y_pred.shape
    horizon = y_true.shape[1]
    rows = []
    for h in range(horizon):
        m_mae = mae(y_true[:, h], y_pred[:, h])
        m_rmse = rmse(y_true[:, h], y_pred[:, h])
        rows.append({"horizon_step": h + 1, "mae": m_mae, "rmse": m_rmse})
    return pd.DataFrame(rows)


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q: float,
) -> float:
    diff = y_true - y_pred
    loss = np.maximum(q * diff, (q - 1) * diff)
    return float(np.mean(loss))


def interval_coverage(
    y_true: np.ndarray,
    y_p10: np.ndarray,
    y_p90: np.ndarray,
) -> float:
    inside = (y_true >= y_p10) & (y_true <= y_p90)
    return float(np.mean(inside))


