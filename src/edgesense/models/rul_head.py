"""Small MLP head that predicts Remaining Useful Life from a USAD latent.

Sits on top of `USADConv1d.encode(...)`: takes (batch, latent_channels,
T_latent), pools over time, and regresses a scalar RUL value.
"""

from __future__ import annotations

import torch
from torch import nn


class RULHead(nn.Module):
    """Time-pooled MLP regressing a scalar RUL from the encoder output."""

    def __init__(
        self,
        latent_channels: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """encoder_output: (batch, latent_channels, T_latent) -> (batch,)."""

        if encoder_output.ndim != 3:
            raise ValueError(
                f"RULHead expects 3D encoder output, got {encoder_output.shape}."
            )
        pooled = encoder_output.mean(dim=2)
        return self.mlp(pooled).squeeze(-1)
