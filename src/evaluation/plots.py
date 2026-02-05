from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def plot_forecast_vs_actual(
    y_true: np.ndarray,
    y_pred_p50: np.ndarray,
    y_pred_p10: Optional[np.ndarray] = None,
    y_pred_p90: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    title: str = "Forecast vs Actual",
    save_path: Optional[Path] = None,
):
    """
    Plot actual vs forecast with uncertainty intervals.

    Args:
        y_true: [num_samples, horizon] or [num_samples * horizon]
        y_pred_p50: [num_samples, horizon] or [num_samples * horizon]
        y_pred_p10: Optional [num_samples, horizon] or [num_samples * horizon]
        y_pred_p90: Optional [num_samples, horizon] or [num_samples * horizon]
        timestamps: Optional timestamps for x-axis
        title: Plot title
        save_path: Optional path to save figure
    """
    # Flatten if needed
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
        y_pred_p50 = y_pred_p50.flatten()
        if y_pred_p10 is not None:
            y_pred_p10 = y_pred_p10.flatten()
        if y_pred_p90 is not None:
            y_pred_p90 = y_pred_p90.flatten()

    n = len(y_true)
    x = np.arange(n) if timestamps is None else timestamps

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot uncertainty interval
    if y_pred_p10 is not None and y_pred_p90 is not None:
        ax.fill_between(
            x, y_pred_p10, y_pred_p90, alpha=0.3, color="blue", label="80% Prediction Interval"
        )

    # Plot predictions and actuals
    ax.plot(x, y_pred_p50, label="Forecast (p50)", color="blue", linewidth=2)
    ax.plot(x, y_true, label="Actual", color="red", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Time Step" if timestamps is None else "Timestamp")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_horizon_errors(
    metrics_df: pd.DataFrame,
    metric: str = "mae",
    title: str = "Error vs Horizon Step",
    save_path: Optional[Path] = None,
):
    """
    Plot error metric vs horizon step.

    Args:
        metrics_df: DataFrame with columns ['horizon_step', metric, ...]
        metric: Metric name to plot (e.g., 'mae', 'rmse')
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if "fold" in metrics_df.columns:
        # Plot per-fold and mean
        for fold in metrics_df["fold"].unique():
            fold_data = metrics_df[metrics_df["fold"] == fold]
            ax.plot(
                fold_data["horizon_step"],
                fold_data[metric],
                alpha=0.3,
                color="gray",
                linewidth=1,
            )

        # Plot mean across folds
        mean_metrics = metrics_df.groupby("horizon_step")[metric].mean()
        ax.plot(
            mean_metrics.index,
            mean_metrics.values,
            color="blue",
            linewidth=2,
            marker="o",
            label=f"Mean {metric.upper()}",
        )
    else:
        ax.plot(
            metrics_df["horizon_step"],
            metrics_df[metric],
            color="blue",
            linewidth=2,
            marker="o",
            label=metric.upper(),
        )

    ax.set_xlabel("Horizon Step")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals Plot",
    save_path: Optional[Path] = None,
):
    """
    Plot residuals (errors) for diagnostic purposes.

    Args:
        y_true: [num_samples, horizon] or [num_samples * horizon]
        y_pred: [num_samples, horizon] or [num_samples * horizon]
        title: Plot title
        save_path: Optional path to save figure
    """
    # Flatten if needed
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals over time
    axes[0].plot(residuals, alpha=0.6)
    axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals Over Time")
    axes[0].grid(True, alpha=0.3)

    # Residuals histogram
    axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor="black")
    axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residuals Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def generate_forecast_lab_report(
    run_dir: Path,
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    model_name: str = "model",
):
    """
    Generate comprehensive forecast lab report with all plots and metrics.

    Args:
        run_dir: Directory to save report artifacts
        predictions_df: DataFrame with predictions (columns: fold, sample, horizon_step, y_true, y_pred_p50, etc.)
        metrics_df: DataFrame with metrics (columns: horizon_step, mae, rmse, fold)
        model_name: Name of the model for titles
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Forecast vs Actual for each fold
    for fold in predictions_df["fold"].unique():
        fold_preds = predictions_df[predictions_df["fold"] == fold]
        fold_preds = fold_preds.sort_values(["sample", "horizon_step"])

        y_true = fold_preds["y_true"].values
        y_pred_p50 = fold_preds["y_pred_p50"].values
        y_pred_p10 = fold_preds.get("y_pred_p10", None)
        y_pred_p90 = fold_preds.get("y_pred_p90", None)

        if y_pred_p10 is not None:
            y_pred_p10 = y_pred_p10.values
        if y_pred_p90 is not None:
            y_pred_p90 = y_pred_p90.values

        plot_forecast_vs_actual(
            y_true,
            y_pred_p50,
            y_pred_p10,
            y_pred_p90,
            title=f"{model_name} - Fold {fold}: Forecast vs Actual",
            save_path=run_dir / f"forecast_plot_fold_{fold}.png",
        )

    # Plot 2: Horizon error plot
    plot_horizon_errors(
        metrics_df,
        metric="mae",
        title=f"{model_name} - MAE vs Horizon Step",
        save_path=run_dir / "horizon_error_mae.png",
    )

    plot_horizon_errors(
        metrics_df,
        metric="rmse",
        title=f"{model_name} - RMSE vs Horizon Step",
        save_path=run_dir / "horizon_error_rmse.png",
    )

    # Plot 3: Residuals plot (using first fold as example)
    if len(predictions_df) > 0:
        first_fold = predictions_df[predictions_df["fold"] == predictions_df["fold"].min()]
        y_true_resid = first_fold["y_true"].values
        y_pred_resid = first_fold["y_pred_p50"].values

        plot_residuals(
            y_true_resid,
            y_pred_resid,
            title=f"{model_name} - Residuals Plot (Fold {first_fold['fold'].iloc[0]})",
            save_path=run_dir / "residuals_plot.png",
        )

    print(f"Forecast lab report generated in {run_dir}")

