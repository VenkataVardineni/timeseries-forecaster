from __future__ import annotations

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """
    Quantile/pinball loss for probabilistic forecasting.

    L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """

    def __init__(self, quantiles: list[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_true: [batch_size, horizon]
            y_pred: [batch_size, horizon, num_quantiles]

        Returns:
            loss: Scalar tensor
        """
        assert y_pred.shape[-1] == len(self.quantiles), "Quantile count mismatch"

        losses = []
        for i, q in enumerate(self.quantiles):
            y_pred_q = y_pred[:, :, i]  # [batch_size, horizon]
            diff = y_true - y_pred_q
            loss_q = torch.maximum(q * diff, (q - 1) * diff)
            losses.append(loss_q)

        # Average across quantiles and samples
        total_loss = torch.stack(losses, dim=0).mean()
        return total_loss


class PinballLoss(nn.Module):
    """
    Pinball loss for a single quantile (alias for QuantileLoss with one quantile).
    """

    def __init__(self, quantile: float):
        super().__init__()
        self.quantile = quantile

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_true: [batch_size, horizon]
            y_pred: [batch_size, horizon]

        Returns:
            loss: Scalar tensor
        """
        diff = y_true - y_pred
        loss = torch.maximum(self.quantile * diff, (self.quantile - 1) * diff)
        return loss.mean()

