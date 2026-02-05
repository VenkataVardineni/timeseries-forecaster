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
) -> List[Fold]:
    """
    Create rolling-origin walk-forward folds over window indices.

    Each fold uses windows [0..train_end) for training and the **next horizon**
    windows for testing where possible.
    """
    folds: List[Fold] = []
    # Reserve at least horizon windows for the final test block
    max_train_end = num_windows - horizon
    split_points = np.linspace(0.2, 0.8, n_folds) * max_train_end
    split_points = np.unique(split_points.astype(int))

    for split in split_points:
        train_start = 0
        train_end = max(split, horizon)  # ensure enough context
        test_start = train_end
        test_end = min(test_start + horizon, num_windows)
        if test_end <= test_start:
            continue
        folds.append(Fold(train_idx=(train_start, train_end), test_idx=(test_start, test_end)))

    return folds


