from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.loaders import load_time_series
from src.data.features import build_features, save_features
from src.data.prepare import build_windows, normalize_train_test
from src.data.splits import make_walk_forward_folds
from src.evaluation.metrics import per_horizon_metrics
from src.utils.config import load_config
from src.utils.io import create_run_id, get_run_dir, save_json


def run_walk_forward(config_path: str) -> str:
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    feat_cfg = cfg.get("features", {})
    win_cfg = cfg["windows"]
    walk_cfg = cfg.get("walk_forward", {"n_folds": 5})
    reports_cfg = cfg.get("reports", {"base_dir": "reports"})

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

    # Optionally save processed features
    processed_path = Path("data/processed/features.parquet")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    save_features(df, str(processed_path))

    train_df, test_df, scalers = normalize_train_test(
        df, target_col=target_col, train_ratio=data_cfg.get("train_ratio", 0.8)
    )

    # Use full normalized series for window building
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

    run_name = cfg.get("run", {}).get("name", "walk_forward")
    run_id = create_run_id(run_name)
    run_dir = get_run_dir(reports_cfg.get("base_dir", "reports"), run_id)

    # For now, walk-forward only prepares splits and leakage checks;
    # model-specific training/prediction happens in training scripts.
    folds_meta: List[Dict] = []
    for i, fold in enumerate(folds):
        train_start, train_end = fold.train_idx
        test_start, test_end = fold.test_idx

        # Leakage guard: ensure train windows end strictly before test windows start
        train_max_ts = timestamps[train_start:train_end].max()
        test_min_ts = timestamps[test_start:test_end].min()
        if not (train_max_ts < test_min_ts):
            raise AssertionError("Leakage detected: training timestamps overlap test period.")

        folds_meta.append(
            {
                "fold": i,
                "train_idx": [int(train_start), int(train_end)],
                "test_idx": [int(test_start), int(test_end)],
                "train_max_timestamp": str(train_max_ts.max()),
                "test_min_timestamp": str(test_min_ts.min()),
            }
        )

    save_json({"folds": folds_meta}, run_dir / "folds.json")

    return run_id


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward data preparation and leakage checks.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    run_id = run_walk_forward(args.config)
    print(f"Walk-forward splits created under run_id={run_id}")


if __name__ == "__main__":
    main()


