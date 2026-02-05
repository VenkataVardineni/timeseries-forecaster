from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_time_series(
    csv_path: str | Path,
    timestamp_col: str,
    target_col: str,
    freq: str = "D",
) -> Tuple[pd.DataFrame, str]:
    """
    Load a univariate time series from CSV and enforce a monotonic datetime index.

    - Parses timestamp column
    - Sorts by timestamp
    - Reindexes to a complete date range at the given frequency
    - Forward-fills missing values
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if timestamp_col not in df.columns:
        raise ValueError(f"timestamp_col '{timestamp_col}' not in CSV columns {df.columns.tolist()}")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in CSV columns {df.columns.tolist()}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    df = df.set_index(timestamp_col)

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_index)

    # Forward-fill target and covariates
    df[target_col] = df[target_col].ffill()
    df = df.ffill()

    df.index.name = timestamp_col
    return df, target_col


