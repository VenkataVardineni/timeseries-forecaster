from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.loaders import load_time_series
from src.data.features import build_features, save_features
from src.data.prepare import build_windows, normalize_train_test
from src.data.splits import Fold, make_walk_forward_folds
from src.evaluation.metrics import per_horizon_metrics, pinball_loss, interval_coverage
from src.evaluation.plots import generate_forecast_lab_report
from src.models.arima import predict_fold_arima
from src.models.seq2seq_attention import Seq2SeqAttention
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.losses import QuantileLoss
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
        context_length=win_cfg["context_length"],
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

        # Leakage check: ensure training windows don't overlap with test windows
        # Last training window uses data up to: (train_end - 1) + context_length
        # First test window uses data starting at: test_start
        # We need: (train_end - 1) + context_length < test_start
        last_train_data_idx = (train_end - 1) + win_cfg["context_length"]
        first_test_data_idx = test_start
        if last_train_data_idx >= first_test_data_idx:
            raise AssertionError(
                f"Leakage detected in fold {fold_idx}! Train data extends to window {last_train_data_idx}, "
                f"but test starts at window {first_test_data_idx}"
            )

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

    # Generate forecast lab report
    predictions_df = pd.DataFrame(all_predictions)
    generate_forecast_lab_report(
        run_dir, predictions_df, metrics_combined, model_name="ARIMA"
    )

    print(f"ARIMA training complete. Results saved to {run_dir}")
    print(f"Summary metrics:\n{metrics_summary.head(10)}")

    return run_id


def train_seq2seq(config_path: str) -> str:
    """
    Train Seq2Seq LSTM with attention model with walk-forward validation.

    Returns:
        run_id: Identifier for this training run
    """
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    feat_cfg = cfg.get("features", {})
    win_cfg = cfg["windows"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("training", {})
    walk_cfg = cfg.get("walk_forward", {"n_folds": 5})
    reports_cfg = cfg.get("reports", {"base_dir": "reports"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        context_length=win_cfg["context_length"],
    )

    # Setup run directory
    run_name = cfg.get("run", {}).get("name", "seq2seq_attention")
    run_id = create_run_id(run_name)
    run_dir = get_run_dir(reports_cfg.get("base_dir", "reports"), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Model parameters
    input_size = X.shape[2]
    hidden_size = model_cfg.get("hidden_size", 64)
    num_layers = model_cfg.get("num_layers", 2)
    dropout = model_cfg.get("dropout", 0.1)
    quantiles = model_cfg.get("quantiles", [0.1, 0.5, 0.9])

    # Training parameters
    batch_size = int(train_cfg.get("batch_size", 64))
    max_epochs = int(train_cfg.get("max_epochs", 50))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    gradient_clip_val = float(train_cfg.get("gradient_clip_val", 1.0))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 5))

    # Walk-forward evaluation
    all_metrics = []
    all_predictions = []
    fold_metadata = []

    for fold_idx, fold in enumerate(folds):
        train_start, train_end = fold.train_idx
        test_start, test_end = fold.test_idx

        # Leakage check: ensure training windows don't overlap with test windows
        last_train_data_idx = (train_end - 1) + win_cfg["context_length"]
        first_test_data_idx = test_start
        if last_train_data_idx >= first_test_data_idx:
            raise AssertionError(
                f"Leakage detected in fold {fold_idx}! Train data extends to window {last_train_data_idx}, "
                f"but test starts at window {first_test_data_idx}"
            )

        X_train_fold = X[train_start:train_end]
        y_train_fold = y[train_start:train_end]
        X_test_fold = X[test_start:test_end]
        y_test_fold = y[test_start:test_end]

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_fold).to(device)
        y_train_tensor = torch.FloatTensor(y_train_fold).to(device)
        X_test_tensor = torch.FloatTensor(X_test_fold).to(device)
        y_test_tensor = torch.FloatTensor(y_test_fold).to(device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        model = Seq2SeqAttention(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=win_cfg["horizon"],
            quantiles=quantiles,
            dropout=dropout,
        ).to(device)

        # Loss and optimizer
        criterion = QuantileLoss(quantiles=quantiles)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Callbacks
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        checkpoint = ModelCheckpoint(checkpoint_dir / f"fold_{fold_idx}")

        # Training loop
        model.train()
        best_val_loss = float("inf")

        for epoch in range(max_epochs):
            epoch_losses = []
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                # Forward pass with teacher forcing
                y_pred = model(batch_X, y_prev=batch_y, teacher_forcing=True)

                # Reshape for loss: [batch_size, horizon, num_quantiles]
                # Loss expects: [batch_size, horizon] vs [batch_size, horizon, num_quantiles]
                loss = criterion(batch_y, y_pred)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)

            # Validation on test set (for early stopping)
            model.eval()
            with torch.no_grad():
                y_pred_val = model(X_test_tensor, teacher_forcing=False)
                val_loss = criterion(y_test_tensor, y_pred_val).item()
            model.train()

            checkpoint(epoch, model, val_loss, optimizer)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if early_stopping(val_loss, model):
                print(f"Early stopping at epoch {epoch}")
                break

            if (epoch + 1) % 10 == 0:
                print(f"Fold {fold_idx}, Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test_tensor, teacher_forcing=False)
            # y_pred_test: [batch_size, horizon, num_quantiles]

        # Extract quantiles
        q_idx_p10 = quantiles.index(0.1)
        q_idx_p50 = quantiles.index(0.5)
        q_idx_p90 = quantiles.index(0.9)

        y_pred_p10 = y_pred_test[:, :, q_idx_p10].cpu().numpy()
        y_pred_p50 = y_pred_test[:, :, q_idx_p50].cpu().numpy()
        y_pred_p90 = y_pred_test[:, :, q_idx_p90].cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()

        # Compute metrics
        metrics_df = per_horizon_metrics(y_test_np, y_pred_p50)
        metrics_df["fold"] = fold_idx
        all_metrics.append(metrics_df)

        # Coverage and pinball loss
        coverage = interval_coverage(y_test_np, y_pred_p10, y_pred_p90)
        pinball_p10 = pinball_loss(y_test_np, y_pred_p10, q=0.1)
        pinball_p50 = pinball_loss(y_test_np, y_pred_p50, q=0.5)
        pinball_p90 = pinball_loss(y_test_np, y_pred_p90, q=0.9)

        # Store predictions
        for i in range(len(y_test_np)):
            for h in range(win_cfg["horizon"]):
                all_predictions.append(
                    {
                        "fold": fold_idx,
                        "sample": i,
                        "horizon_step": h + 1,
                        "timestamp": str(timestamps[test_start + i, h]),
                        "y_true": float(y_test_np[i, h]),
                        "y_pred_p10": float(y_pred_p10[i, h]),
                        "y_pred_p50": float(y_pred_p50[i, h]),
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
                "best_val_loss": float(best_val_loss),
            }
        )

    # Aggregate metrics
    metrics_combined = pd.concat(all_metrics, ignore_index=True)
    metrics_summary = (
        metrics_combined.groupby("horizon_step")[["mae", "rmse"]].mean().reset_index()
    )

    # Save outputs
    metrics_combined.to_csv(run_dir / "metrics_seq2seq.csv", index=False)
    metrics_summary.to_csv(run_dir / "metrics_summary.csv", index=False)

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(run_dir / "predictions.csv", index=False)

    save_json({"folds": fold_metadata}, run_dir / "folds.json")
    save_json({"config": cfg}, run_dir / "config.json")

    # Generate forecast lab report
    predictions_df = pd.DataFrame(all_predictions)
    generate_forecast_lab_report(
        run_dir, predictions_df, metrics_combined, model_name="Seq2Seq-Attention"
    )

    print(f"Seq2Seq training complete. Results saved to {run_dir}")
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
    elif model_type == "seq2seq_attention":
        train_seq2seq(args.config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main()

