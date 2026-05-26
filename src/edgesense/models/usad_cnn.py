"""1D-CNN USAD model for multivariate time-series reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class USADConv1dConfig:
    """Configuration for the 1D-CNN USAD model.

    Attributes:
        in_features: Number of sensor channels (features).
        base_channels: Base channel width for convolutional blocks.
        latent_channels: Channel width of the latent representation.
        kernel_size: Kernel size for convolutional layers (odd for symmetric padding).
        downsample_layers: Number of stride-2 convolutional layers in the encoder.
        dropout: Optional dropout applied after convolutional blocks.
    """

    in_features: int
    base_channels: int = 32
    latent_channels: int = 64
    kernel_size: int = 3
    downsample_layers: int = 2
    dropout: float = 0.0

    @property
    def downsample_factor(self) -> int:
        """Total temporal downsampling factor introduced by the encoder."""

        return 2**self.downsample_layers


class ConvEncoder1d(nn.Module):
    """Convolutional encoder that compresses temporal windows."""

    def __init__(self, config: USADConv1dConfig) -> None:
        super().__init__()
        if config.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal alignment.")
        if config.in_features <= 0:
            raise ValueError("in_features must be a positive integer.")
        if config.downsample_layers <= 0:
            raise ValueError("downsample_layers must be at least 1.")

        padding = config.kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(config.in_features, config.base_channels, config.kernel_size, padding=padding),
            nn.ReLU(),
        ]
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))

        in_channels = config.base_channels
        for _ in range(config.downsample_layers):
            out_channels = min(config.latent_channels, in_channels * 2)
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        config.kernel_size,
                        stride=2,
                        padding=padding,
                    ),
                    nn.ReLU(),
                ]
            )
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_channels = out_channels

        if in_channels != config.latent_channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, config.latent_channels, config.kernel_size, padding=padding),
                    nn.ReLU(),
                ]
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor to a latent representation."""

        return self.net(x)


class ConvDecoder1d(nn.Module):
    """Convolutional decoder that reconstructs temporal windows."""

    def __init__(self, config: USADConv1dConfig) -> None:
        super().__init__()
        padding = config.kernel_size // 2

        self.blocks = nn.ModuleList()
        in_channels = config.latent_channels
        for _ in range(config.downsample_layers):
            out_channels = max(config.base_channels, in_channels // 2)
            self.blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv1d(in_channels, out_channels, config.kernel_size, padding=padding),
                    nn.ReLU(),
                )
            )
            in_channels = out_channels

        self.output_conv = nn.Conv1d(in_channels, config.in_features, config.kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """Decode latent tensor into the original temporal length."""

        for block in self.blocks:
            x = block(x)

        if x.shape[-1] != target_length:
            x = F.interpolate(x, size=target_length, mode="nearest")

        return self.output_conv(x)


class USADConv1d(nn.Module):
    """USAD-inspired 1D convolutional autoencoder with dual decoders.

    Expected input shape: (batch_size, sequence_length, num_features)
    Output reconstructions follow the same shape.
    """

    def __init__(self, config: USADConv1dConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ConvEncoder1d(config)
        self.decoder1 = ConvDecoder1d(config)
        self.decoder2 = ConvDecoder1d(config)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input batch into the latent space."""

        # Input: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        return self.encoder(x)

    def decode(self, z: torch.Tensor, target_length: int) -> torch.Tensor:
        """Decode a latent tensor to a reconstruction using decoder1."""

        # Latent: (B, C, T_latent) -> (B, F, T)
        recon = self.decoder1(z, target_length=target_length)
        return recon.transpose(1, 2)

    def reconstruct_via_decoder2(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input by passing through encoder and decoder2.

        Args:
            x: Input tensor shaped (batch_size, sequence_length, num_features).

        Returns:
            Reconstruction from decoder2 with shape (batch_size, sequence_length, num_features).
        """

        if x.ndim != 3:
            raise ValueError("Input must have shape (batch_size, sequence_length, num_features).")
        if x.shape[2] != self.config.in_features:
            raise ValueError("Input feature dimension does not match config.in_features.")

        sequence_length = x.shape[1]
        z = self.encoder(x.transpose(1, 2))
        recon = self.decoder2(z, target_length=sequence_length)
        return recon.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstructions and latent representation.

        Args:
            x: Input tensor shaped (batch_size, sequence_length, num_features).

        Returns:
            Tuple of (reconstruction_1, reconstruction_2, latent).
        """

        if x.ndim != 3:
            raise ValueError("Input must have shape (batch_size, sequence_length, num_features).")
        if x.shape[2] != self.config.in_features:
            raise ValueError("Input feature dimension does not match config.in_features.")
        if x.shape[1] < self.config.downsample_factor:
            raise ValueError("Sequence length is too short for the configured downsampling.")

        sequence_length = x.shape[1]
        # Input: (B, T, F) -> (B, F, T)
        x_channels = x.transpose(1, 2)
        latent = self.encoder(x_channels)

        recon_1 = self.decoder1(latent, target_length=sequence_length)
        recon_2 = self.decoder2(latent, target_length=sequence_length)

        # Output: (B, F, T) -> (B, T, F)
        return recon_1.transpose(1, 2), recon_2.transpose(1, 2), latent
