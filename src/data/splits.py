from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Fold:
    train_idx: Tuple[int, int]
    test_idx: Tuple[int, int]


def make_walk_forward_folds(
    num_windows: int,
    horizon: int,
    n_folds: int,
    context_length: int = 0,
) -> List[Fold]:
    """
    Create rolling-origin walk-forward folds over window indices.

    Each fold uses windows [0..train_end) for training and the **next horizon**
    windows for testing where possible. Ensures no data leakage between train and test.

    Args:
        num_windows: Total number of windows
        horizon: Forecast horizon (number of steps ahead)
        n_folds: Number of folds to create
        context_length: Length of context window (for leakage checking)
    """
    folds: List[Fold] = []
    # Reserve space for test windows
    # Each window uses context_length + horizon data points
    # To avoid leakage, test_start must be >= train_end (windows don't overlap)
    # But we also need to ensure the data used by training doesn't overlap with test data
    # Window i uses data indices [i, i+context_length+horizon-1]
    # So we need: (train_end-1) + context_length + horizon < test_start
    # Which simplifies to: train_end + context_length + horizon - 1 < test_start
    # Or: test_start >= train_end + context_length + horizon - 1
    
    # For simplicity, we'll use test_start = train_end (adjacent windows)
    # But we need to account for the fact that windows overlap in data usage
    # Actually, if windows are adjacent (test_start = train_end), then:
    # - Last train window (train_end-1) uses data up to: (train_end-1) + context_length + horizon - 1
    # - First test window (test_start) uses data from: test_start
    # So we need: (train_end-1) + context_length + horizon - 1 < test_start
    # Since test_start = train_end, we need: train_end - 1 + context_length + horizon - 1 < train_end
    # Which simplifies to: context_length + horizon - 2 < 0, which is false for typical values
    
    # So we need to add a gap. Let's use: test_start = train_end + gap
    # Where gap ensures no overlap. Minimum gap needed: context_length + horizon - 1
    
    min_gap = max(0, context_length + horizon - 1) if context_length > 0 else 0
    max_train_end = num_windows - horizon - min_gap
    
    if max_train_end <= 0:
        raise ValueError(f"Not enough windows for walk-forward validation. Need at least {horizon + min_gap + 1} windows, got {num_windows}")
    
    # Create evenly spaced split points
    split_points = np.linspace(0.3, 0.9, n_folds) * max_train_end
    split_points = np.unique(split_points.astype(int))
    split_points = split_points[split_points >= horizon]  # Ensure minimum training size

    for split in split_points:
        train_start = 0
        train_end = split
        # Add gap to prevent leakage
        test_start = train_end + min_gap
        test_end = min(test_start + horizon, num_windows)
        
        if test_end <= test_start or test_end > num_windows:
            continue
        folds.append(Fold(train_idx=(train_start, train_end), test_idx=(test_start, test_end)))

    return folds


