from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data.loaders import load_time_series
from src.data.features import build_features, save_features
from src.data.prepare import build_windows, normalize_train_test
from src.data.splits import WalkForwardFold, make_walk_forward_folds
from src.evaluation.metrics import per_horizon_metrics, pinball_loss, interval_coverage
from src.models.arima import predict_fold_arima
from src.utils.config import load_config
from src.utils.io import create_run_id, get_run_dir, save_json


def train_arima(config_path: str) -> str:
    """
    Train ARIMA model with walk-forward validation.

    Returns:
        run_id: Identifier for this training run
    """
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    feat_cfg = cfg.get("features", {})
    win_cfg = cfg["windows"]
    model_cfg = cfg["model"]
    walk_cfg = cfg.get("walk_forward", {"n_folds": 5})
    reports_cfg = cfg.get("reports", {"base_dir": "reports"})

    # Load and prepare data
    df, target_col = load_time_series(
        csv_path=data_cfg["csv_path"],
        timestamp_col=data_cfg["timestamp_col"],
        target_col=data_cfg["target_col"],
        freq=data_cfg.get("freq", "D"),
    )

    df = build_features(
        df,
        timestamp_col=data_cfg["timestamp_col"],
        target_col=target_col,
        calendar=feat_cfg.get("calendar", True),
        lags=feat_cfg.get("lags", []),
        rolling=feat_cfg.get("rolling", []),
    )

    processed_path = Path("data/processed/features.parquet")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    save_features(df, str(processed_path))

    train_df, test_df, scalers = normalize_train_test(
        df, target_col=target_col, train_ratio=data_cfg.get("train_ratio", 0.8)
    )

    full_df = pd.concat([train_df, test_df], axis=0)
    X, y, timestamps = build_windows(
        full_df,
        target_col=target_col,
        context_length=win_cfg["context_length"],
        horizon=win_cfg["horizon"],
    )

    num_windows = X.shape[0]
    folds = make_walk_forward_folds(
        num_windows=num_windows,
        horizon=win_cfg["horizon"],
        n_folds=walk_cfg.get("n_folds", 5),
    )

    # Setup run directory
    run_name = cfg.get("run", {}).get("name", "arima_baseline")
    run_id = create_run_id(run_name)
    run_dir = get_run_dir(reports_cfg.get("base_dir", "reports"), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Extract target values from context windows for ARIMA
    target_idx = list(full_df.columns).index(target_col)
    y_context = X[:, :, target_idx]  # [num_windows, context_length]

    # Reconstruct full time series from windows for ARIMA
    # ARIMA needs a continuous univariate series
    target_series = full_df[target_col].values

    # Walk-forward evaluation
    all_metrics = []
    all_predictions = []
    fold_metadata = []

    for fold_idx, fold in enumerate(folds):
        train_start, train_end = fold.train_idx
        test_start, test_end = fold.test_idx

        # Leakage check
        train_max_ts = timestamps[train_start:train_end, -1].max()
        test_min_ts = timestamps[test_start:test_end, 0].min()
        assert train_max_ts < test_min_ts, "Leakage detected!"

        # Get training series: use original time series up to the end of training windows
        # Find the last timestamp in training windows
        train_window_end_idx = train_end + win_cfg["context_length"] - 1
        train_series = target_series[:train_window_end_idx]

        X_test_fold = X[test_start:test_end]
        y_test_fold = y[test_start:test_end]  # Horizon targets
        test_contexts = y_context[test_start:test_end]  # Context windows for test

        # Predict with ARIMA
        order = tuple(model_cfg.get("order", [2, 1, 2]))
        seasonal_order = tuple(model_cfg.get("seasonal_order", [0, 0, 0, 0]))

        y_pred_mean, y_pred_p10, y_pred_p90, diagnostics = predict_fold_arima(
            train_series,
            test_contexts,
            order=order,
            seasonal_order=seasonal_order,
            horizon=win_cfg["horizon"],
        )

        # Compute metrics
        metrics_df = per_horizon_metrics(y_test_fold, y_pred_mean)
        metrics_df["fold"] = fold_idx
        all_metrics.append(metrics_df)

        # Coverage and pinball loss
        coverage = interval_coverage(y_test_fold, y_pred_p10, y_pred_p90)
        pinball_p10 = pinball_loss(y_test_fold, y_pred_p10, q=0.1)
        pinball_p50 = pinball_loss(y_test_fold, y_pred_mean, q=0.5)
        pinball_p90 = pinball_loss(y_test_fold, y_pred_p90, q=0.9)

        # Store predictions
        for i in range(len(y_test_fold)):
            for h in range(win_cfg["horizon"]):
                all_predictions.append(
                    {
                        "fold": fold_idx,
                        "sample": i,
                        "horizon_step": h + 1,
                        "timestamp": str(timestamps[test_start + i, h]),
                        "y_true": float(y_test_fold[i, h]),
                        "y_pred_p10": float(y_pred_p10[i, h]),
                        "y_pred_p50": float(y_pred_mean[i, h]),
                        "y_pred_p90": float(y_pred_p90[i, h]),
                    }
                )

        fold_metadata.append(
            {
                "fold": fold_idx,
                "train_idx": [int(train_start), int(train_end)],
                "test_idx": [int(test_start), int(test_end)],
                "coverage": float(coverage),
                "pinball_p10": float(pinball_p10),
                "pinball_p50": float(pinball_p50),
                "pinball_p90": float(pinball_p90),
                "diagnostics": diagnostics,
            }
        )

    # Aggregate metrics
    metrics_combined = pd.concat(all_metrics, ignore_index=True)
    metrics_summary = (
        metrics_combined.groupby("horizon_step")[["mae", "rmse"]].mean().reset_index()
    )

    # Save outputs
    metrics_combined.to_csv(run_dir / "metrics_arima.csv", index=False)
    metrics_summary.to_csv(run_dir / "metrics_summary.csv", index=False)

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(run_dir / "predictions.csv", index=False)

    save_json({"folds": fold_metadata}, run_dir / "folds.json")
    save_json({"config": cfg}, run_dir / "config.json")

    print(f"ARIMA training complete. Results saved to {run_dir}")
    print(f"Summary metrics:\n{metrics_summary.head(10)}")

    return run_id


def main():
    parser = argparse.ArgumentParser(description="Train forecasting models.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_type = cfg["model"]["type"]

    if model_type == "arima":
        train_arima(args.config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main()

