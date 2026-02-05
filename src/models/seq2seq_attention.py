from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """
    Bahdanau-style attention mechanism.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: [batch_size, hidden_size] current decoder hidden state
            encoder_outputs: [batch_size, seq_len, hidden_size] encoder outputs

        Returns:
            context: [batch_size, hidden_size] attention-weighted context
            attention_weights: [batch_size, seq_len] attention weights
        """
        # Compute attention scores
        # decoder_hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]

        # Expand decoder_hidden for broadcasting
        decoder_expanded = decoder_hidden.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # Compute attention energies
        energies = self.V(torch.tanh(self.W(decoder_expanded) + self.W(encoder_outputs)))
        # energies: [batch_size, seq_len, 1]

        energies = energies.squeeze(-1)  # [batch_size, seq_len]
        attention_weights = torch.softmax(energies, dim=1)  # [batch_size, seq_len]

        # Compute weighted context
        context = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs
        )  # [batch_size, 1, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size]

        return context, attention_weights


class Seq2SeqAttention(nn.Module):
    """
    Seq2Seq LSTM model with attention for multi-horizon forecasting with quantiles.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 30,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        # Encoder
        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention
        self.attention = AttentionLayer(hidden_size)

        # Decoder
        self.decoder = nn.LSTM(
            hidden_size + 1,  # +1 for previous target value
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection: predict quantiles for each horizon step
        self.output_proj = nn.Linear(hidden_size, self.num_quantiles)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        y_prev: Optional[torch.Tensor] = None,
        teacher_forcing: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch_size, context_length, input_size] input features
            y_prev: [batch_size, horizon] previous target values (for teacher forcing)
            teacher_forcing: Whether to use teacher forcing during training

        Returns:
            y_pred: [batch_size, horizon, num_quantiles] quantile predictions
        """
        batch_size = x.shape[0]

        # Encode input sequence
        encoder_outputs, (hidden, cell) = self.encoder(x)
        # encoder_outputs: [batch_size, context_length, hidden_size]
        # hidden, cell: [num_layers, batch_size, hidden_size]

        # Initialize decoder hidden state from encoder
        decoder_hidden = hidden[-1]  # Use last layer: [batch_size, hidden_size]
        decoder_cell = cell[-1]  # [batch_size, hidden_size]

        # Decode to produce horizon predictions
        predictions = []

        # Start with last value from context (or zero if not available)
        if y_prev is not None and teacher_forcing:
            decoder_input = y_prev[:, 0].unsqueeze(1)  # [batch_size, 1]
        else:
            # Use last value from encoder input (target column)
            decoder_input = x[:, -1, 0:1]  # [batch_size, 1] (assuming first col is target)

        for step in range(self.horizon):
            # Compute attention context
            context, _ = self.attention(decoder_hidden, encoder_outputs)
            # context: [batch_size, hidden_size]

            # Concatenate context with previous target value
            decoder_input_expanded = decoder_input.unsqueeze(1)  # [batch_size, 1, 1]
            context_expanded = context.unsqueeze(1)  # [batch_size, 1, hidden_size]
            decoder_input_with_context = torch.cat(
                [decoder_input_expanded, context_expanded], dim=2
            )  # [batch_size, 1, hidden_size + 1]

            # Decode one step
            # Need to reshape hidden/cell for LSTM: [num_layers, batch_size, hidden_size]
            decoder_hidden_expanded = decoder_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
            decoder_cell_expanded = decoder_cell.unsqueeze(0).repeat(self.num_layers, 1, 1)

            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_input_with_context, (decoder_hidden_expanded, decoder_cell_expanded)
            )
            # decoder_output: [batch_size, 1, hidden_size]
            # decoder_hidden, decoder_cell: [num_layers, batch_size, hidden_size]

            decoder_hidden = decoder_hidden[-1]  # Use last layer: [batch_size, hidden_size]
            decoder_cell = decoder_cell[-1]  # [batch_size, hidden_size]

            # Project to quantiles
            output = self.output_proj(decoder_output.squeeze(1))  # [batch_size, num_quantiles]
            predictions.append(output)

            # Prepare next input (teacher forcing or use prediction)
            if teacher_forcing and y_prev is not None and step < self.horizon - 1:
                decoder_input = y_prev[:, step + 1].unsqueeze(1)  # [batch_size, 1]
            else:
                # Use median quantile (p50) as next input
                decoder_input = output[:, self.quantiles.index(0.5)].unsqueeze(1)  # [batch_size, 1]

        # Stack predictions: [batch_size, horizon, num_quantiles]
        y_pred = torch.stack(predictions, dim=1)
        return y_pred

