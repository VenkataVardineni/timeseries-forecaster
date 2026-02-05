from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Scalers:
    target_scaler: StandardScaler
    feature_scaler: StandardScaler


def normalize_train_test(
    df: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, Scalers]:
    n_train = int(len(df) * train_ratio)
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    feature_cols = [c for c in df.columns if c != target_col]

    target_scaler = StandardScaler()
    feature_scaler = StandardScaler()

    train_target = train_df[[target_col]].values
    train_features = train_df[feature_cols].values

    train_df[target_col] = target_scaler.fit_transform(train_target)
    train_df[feature_cols] = feature_scaler.fit_transform(train_features)

    test_df[target_col] = target_scaler.transform(test_df[[target_col]].values)
    test_df[feature_cols] = feature_scaler.transform(test_df[feature_cols].values)

    scalers = Scalers(target_scaler=target_scaler, feature_scaler=feature_scaler)
    return train_df, test_df, scalers


def build_windows(
    df: pd.DataFrame,
    target_col: str,
    context_length: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build supervised windows for multi-horizon forecasting.

    Returns:
        X: [num_windows, context_length, num_features]
        y: [num_windows, horizon]
        timestamps: [num_windows, horizon] (end timestamps of each horizon step)
    """
    feature_cols = list(df.columns)
    num_features = len(feature_cols)

    values = df[feature_cols].values.astype(np.float32)
    target_values = df[target_col].values.astype(np.float32)
    index = df.index.to_numpy()

    X_list = []
    y_list = []
    ts_list = []

    max_start = len(df) - context_length - horizon + 1
    for start in range(max_start):
        end_context = start + context_length
        end_horizon = end_context + horizon
        x_window = values[start:end_context, :]
        y_window = target_values[end_context:end_horizon]
        ts_window = index[end_context:end_horizon]
        X_list.append(x_window)
        y_list.append(y_window)
        ts_list.append(ts_window)

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    timestamps = np.stack(ts_list, axis=0)
    return X, y, timestamps


