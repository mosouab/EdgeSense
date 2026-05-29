"""Anomaly scoring and thresholding utilities for USAD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from .models import USADConv1d
from .training import create_dataloader


@dataclass(frozen=True)
class ScoringConfig:
    """Configuration for USAD anomaly scoring.

    Attributes:
        alpha: Weight for AE1 reconstruction error.
        beta: Weight for AE2(AE1(x)) reconstruction error.
        batch_size: Batch size used during scoring.
        device: Device used for scoring.
    """

    alpha: float = 0.3
    beta: float = 0.7
    batch_size: int = 256
    device: str = "cpu"

    def validate(self) -> None:
        """Validate scoring configuration values."""

        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative.")
        if self.alpha + self.beta <= 0:
            raise ValueError("alpha and beta must sum to a positive value.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")


@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for threshold computation.

    Attributes:
        method: Thresholding method name ('percentile' or 'mean_std').
        value: Percentile (0-100) or k for mean_std (mean + k * std).
    """

    method: str = "percentile"
    value: float = 99.0

    def validate(self) -> None:
        """Validate threshold configuration values."""

        if self.method not in {"percentile", "mean_std"}:
            raise ValueError("method must be 'percentile' or 'mean_std'.")
        if self.method == "percentile" and not (0.0 < self.value < 100.0):
            raise ValueError("Percentile value must be in (0, 100).")
        if self.method == "mean_std" and self.value <= 0:
            raise ValueError("Mean-std multiplier must be positive.")


def compute_usad_scores(
    model: USADConv1d,
    windows: np.ndarray,
    config: ScoringConfig,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute USAD anomaly scores for each window.

    Args:
        model: Trained USADConv1d model.
        windows: Windowed dataset of shape (num_windows, window_size, num_features).
        config: Scoring configuration.
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        Array of anomaly scores per window.
    """

    config.validate()
    dataloader = create_dataloader(
        windows,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = model.to(config.device)
    model.eval()

    scores: list[float] = []
    with torch.no_grad():
        for (batch_windows,) in tqdm(
            dataloader, desc="Scoring windows", leave=False, disable=not show_progress
        ):
            batch_windows = batch_windows.to(config.device)
            recon1, _, _ = model(batch_windows)
            recon2_from_recon1 = model.reconstruct_via_decoder2(recon1)

            mse_ae1 = _mse_per_window(batch_windows, recon1)
            mse_ae2 = _mse_per_window(batch_windows, recon2_from_recon1)

            batch_scores = config.alpha * mse_ae1 + config.beta * mse_ae2
            scores.extend(batch_scores.cpu().numpy().tolist())

    return np.asarray(scores, dtype=np.float32)


def compute_threshold(scores: np.ndarray, config: ThresholdConfig) -> float:
    """Compute a scalar anomaly threshold from healthy scores.

    Args:
        scores: Array of healthy-window scores.
        config: Threshold configuration.

    Returns:
        Threshold value used for anomaly detection.
    """

    config.validate()
    if scores.size == 0:
        raise ValueError("Scores array must not be empty.")

    if config.method == "percentile":
        return float(np.percentile(scores, config.value))

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    return mean + config.value * std


def flag_anomalies(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Flag anomaly windows based on a threshold.

    Args:
        scores: Array of anomaly scores.
        threshold: Scalar threshold.

    Returns:
        Boolean array where True indicates anomalous windows.
    """

    if scores.ndim != 1:
        raise ValueError("Scores must be a 1D array.")
    return scores > threshold


def _mse_per_window(inputs: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error per window."""

    if inputs.shape != recon.shape:
        raise ValueError("Input and reconstruction shapes must match.")
    return torch.mean((inputs - recon) ** 2, dim=(1, 2))
