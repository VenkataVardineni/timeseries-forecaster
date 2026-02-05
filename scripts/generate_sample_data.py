"""
Generate a sample time series dataset for testing the forecasting toolkit.
Creates a realistic daily time series with trend, seasonality, and noise.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_data(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    output_path: str = "data/raw/example_daily.csv",
):
    """
    Generate a synthetic daily time series with:
    - Trend component
    - Weekly seasonality
    - Monthly seasonality
    - Random noise
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    # Trend: linear increase
    trend = np.linspace(100, 150, n)

    # Weekly seasonality (7-day cycle)
    weekly = 10 * np.sin(2 * np.pi * np.arange(n) / 7)

    # Monthly seasonality (30-day cycle)
    monthly = 5 * np.sin(2 * np.pi * np.arange(n) / 30)

    # Random noise
    noise = np.random.RandomState(42).normal(0, 3, n)

    # Combine components
    y = trend + weekly + monthly + noise

    # Ensure positive values
    y = np.maximum(y, 10)

    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": dates,
        "y": y,
    })

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated sample dataset with {len(df)} rows")
    print(f"Saved to: {output_path}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Target range: {df['y'].min():.2f} to {df['y'].max():.2f}")


if __name__ == "__main__":
    generate_sample_data()

