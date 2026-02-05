from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


class EarlyStopping:
    """
    Early stopping callback that monitors validation loss.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_weights: Optional[dict] = None
        self.early_stop = False

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score (lower is better for 'min' mode)
            model: Model to save weights from

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self._save_weights(model)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self._save_weights(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self._restore_weights(model)

        return self.early_stop

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta

    def _save_weights(self, model: torch.nn.Module):
        self.best_weights = model.state_dict().copy()

    def _restore_weights(self, model: torch.nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ModelCheckpoint:
    """
    Save model checkpoints during training.
    """

    def __init__(self, checkpoint_dir: Path, save_best_only: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_score: Optional[float] = None

    def __call__(
        self,
        epoch: int,
        model: torch.nn.Module,
        score: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Save checkpoint if conditions are met.

        Args:
            epoch: Current epoch number
            model: Model to save
            score: Current validation score
            optimizer: Optional optimizer state
        """
        if self.save_best_only:
            if self.best_score is None or score < self.best_score:
                self.best_score = score
                self._save_checkpoint(epoch, model, optimizer, is_best=True)
        else:
            self._save_checkpoint(epoch, model, optimizer, is_best=False)

    def _save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        is_best: bool,
    ):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")
        torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")

