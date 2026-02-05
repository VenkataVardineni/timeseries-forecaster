from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    idx = df.index if df.index.name == timestamp_col else pd.to_datetime(df[timestamp_col])
    df = df.copy()
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags: Sequence[int]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    rolling: List[dict],
) -> pd.DataFrame:
    df = df.copy()
    for spec in rolling:
        window = spec.get("window")
        stats = spec.get("stats", [])
        if window is None:
            continue
        roll = df[target_col].rolling(window=window, min_periods=1)
        for stat in stats:
            if stat == "mean":
                df[f"{target_col}_rollmean_{window}"] = roll.mean()
            elif stat == "std":
                df[f"{target_col}_rollstd_{window}"] = roll.std().fillna(0.0)
    return df


def build_features(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    calendar: bool = True,
    lags: Sequence[int] | None = None,
    rolling: List[dict] | None = None,
) -> pd.DataFrame:
    if calendar:
        df = add_calendar_features(df, timestamp_col)
    if lags:
        df = add_lag_features(df, target_col, lags)
    if rolling:
        df = add_rolling_features(df, target_col, rolling)

    # Drop initial rows where lags are NaN
    df = df.dropna()
    return df


def save_features(df: pd.DataFrame, path: str) -> None:
    if path.endswith(".parquet"):
        df.to_parquet(path, index=True)
    elif path.endswith(".csv"):
        df.to_csv(path, index=True)
    else:
        raise ValueError("Unsupported feature output format; use .parquet or .csv")


