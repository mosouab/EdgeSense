"""Training utilities for the USAD 1D-CNN model."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import USADConv1d


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for USAD training.

    Attributes:
        batch_size: Number of windows per batch.
        epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        seed: Random seed for deterministic training.
        shuffle: Whether to shuffle windows each epoch.
        drop_last: Whether to drop the last partial batch.
    """

    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    shuffle: bool = True
    drop_last: bool = True


@dataclass(frozen=True)
class TrainingHistory:
    """Container for tracking training losses per epoch."""

    ae1_losses: list[float]
    ae2_losses: list[float]
    val_recon_losses: list[float] | None = None


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration for early stopping.

    Attributes:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum improvement required to reset patience.
        max_epochs: Maximum epochs to train regardless of early stopping.
        val_fraction: Fraction of windows reserved for validation (0-1).
    """

    patience: int = 5
    min_delta: float = 1e-4
    max_epochs: int = 50
    val_fraction: float = 0.1


def seed_all(seed: int) -> None:
    """Seed RNGs for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloader(
    windows: np.ndarray,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    """Create a DataLoader for windowed numpy arrays."""

    if windows.ndim != 3:
        raise ValueError("Windows must have shape (num_windows, window_size, num_features).")

    tensor = torch.tensor(windows, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def split_train_validation(
    windows: np.ndarray,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split windows into train/validation sets, preserving temporal order.

    Args:
        windows: Windowed dataset of shape (num_windows, window_size, num_features).
        val_fraction: Fraction of windows reserved for validation (0-1).

    Returns:
        Tuple of (train_windows, val_windows).
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")
    num_windows = windows.shape[0]
    split_index = int(num_windows * (1.0 - val_fraction))
    if split_index <= 0 or split_index >= num_windows:
        raise ValueError("val_fraction yields an empty train or validation split.")
    return windows[:split_index], windows[split_index:]


def train_usad(
    model: USADConv1d,
    windows: np.ndarray,
    config: TrainingConfig,
    device: torch.device | str = "cpu",
) -> TrainingHistory:
    """Train USAD with the two-phase loss schedule described in the paper.

    The loss weights evolve per epoch:
      - w_rec = 1 / epoch
      - w_adv = 1 - 1 / epoch

    Args:
        model: USADConv1d model instance.
        windows: Windowed dataset of shape (num_windows, window_size, num_features).
        config: Training configuration.
        device: Training device.

    Returns:
        TrainingHistory with AE1 and AE2 losses per epoch.
    """

    seed_all(config.seed)
    model = model.to(device)
    model.train()

    dataloader = create_dataloader(
        windows,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
    )

    ae1_params = list(model.encoder.parameters()) + list(model.decoder1.parameters())
    ae2_params = list(model.encoder.parameters()) + list(model.decoder2.parameters())
    optimizer_ae1 = torch.optim.Adam(ae1_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer_ae2 = torch.optim.Adam(ae2_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    mse = nn.MSELoss()

    ae1_epoch_losses: list[float] = []
    ae2_epoch_losses: list[float] = []

    for epoch in range(1, config.epochs + 1):
        w_rec = 1.0 / float(epoch)
        w_adv = 1.0 - w_rec

        ae1_running = 0.0
        ae2_running = 0.0
        batch_count = 0

        for (batch_windows,) in dataloader:
            batch_windows = batch_windows.to(device)

            # Phase 1/2: update AE1 (encoder + decoder1)
            optimizer_ae1.zero_grad()
            recon1, _, _ = model(batch_windows)
            recon2_from_recon1 = model.reconstruct_via_decoder2(recon1)
            loss_ae1 = w_rec * mse(batch_windows, recon1) + w_adv * mse(batch_windows, recon2_from_recon1)
            loss_ae1.backward()
            optimizer_ae1.step()

            # Phase 2: update AE2 (encoder + decoder2)
            optimizer_ae2.zero_grad()
            _, recon2, _ = model(batch_windows)
            recon2_from_recon1 = model.reconstruct_via_decoder2(recon1.detach())
            loss_ae2 = w_rec * mse(batch_windows, recon2) - w_adv * mse(batch_windows, recon2_from_recon1)
            loss_ae2.backward()
            optimizer_ae2.step()

            ae1_running += loss_ae1.item()
            ae2_running += loss_ae2.item()
            batch_count += 1

        ae1_epoch_losses.append(ae1_running / max(batch_count, 1))
        ae2_epoch_losses.append(ae2_running / max(batch_count, 1))

    return TrainingHistory(ae1_losses=ae1_epoch_losses, ae2_losses=ae2_epoch_losses)


def train_usad_with_validation(
    model: USADConv1d,
    windows: np.ndarray,
    train_config: TrainingConfig,
    early_stopping: EarlyStoppingConfig,
    device: torch.device | str = "cpu",
) -> TrainingHistory:
    """Train USAD with early stopping based on validation reconstruction loss.

    Args:
        model: USADConv1d model instance.
        windows: Windowed dataset of shape (num_windows, window_size, num_features).
        train_config: Training configuration.
        early_stopping: Early stopping configuration.
        device: Training device.

    Returns:
        TrainingHistory including validation reconstruction losses.
    """

    seed_all(train_config.seed)
    train_windows, val_windows = split_train_validation(windows, early_stopping.val_fraction)

    history = TrainingHistory(ae1_losses=[], ae2_losses=[], val_recon_losses=[])
    best_val = float("inf")
    patience_counter = 0

    model = model.to(device)
    mse = nn.MSELoss()

    for epoch in range(1, early_stopping.max_epochs + 1):
        epoch_config = TrainingConfig(
            batch_size=train_config.batch_size,
            epochs=1,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            seed=train_config.seed,
            shuffle=train_config.shuffle,
            drop_last=train_config.drop_last,
        )
        epoch_history = train_usad(model, train_windows, epoch_config, device=device)
        history.ae1_losses.append(epoch_history.ae1_losses[-1])
        history.ae2_losses.append(epoch_history.ae2_losses[-1])

        val_loss = _evaluate_reconstruction_loss(model, val_windows, train_config.batch_size, device)
        history.val_recon_losses.append(val_loss)

        if best_val - val_loss >= early_stopping.min_delta:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping.patience:
            break

    return history


def _evaluate_reconstruction_loss(
    model: USADConv1d,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device | str,
) -> float:
    """Compute average AE1 reconstruction loss for validation windows."""

    dataloader = create_dataloader(windows, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    mse = nn.MSELoss()
    running = 0.0
    count = 0
    with torch.no_grad():
        for (batch_windows,) in dataloader:
            batch_windows = batch_windows.to(device)
            recon1, _, _ = model(batch_windows)
            loss = mse(batch_windows, recon1)
            running += loss.item()
            count += 1
    model.train()
    return running / max(count, 1)
