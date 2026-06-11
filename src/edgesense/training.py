"""Training utilities for the USAD 1D-CNN model."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
        adv_ramp_epochs: Number of epochs over which w_adv ramps linearly from 0 to its cap.
            The original USAD `w_adv = 1 - 1/epoch` schedule grows too fast on small datasets
            and the encoder destabilizes before the adversarial phase. A linear ramp + cap
            keeps the reconstruction objective dominant long enough to converge first.
        adv_max_weight: Upper bound on w_adv (so w_rec >= 1 - adv_max_weight is always active).
        grad_clip_norm: Max L2 norm for gradient clipping (None disables clipping).
    """

    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    shuffle: bool = True
    drop_last: bool = True
    adv_ramp_epochs: int = 20
    adv_max_weight: float = 0.5
    grad_clip_norm: float | None = 1.0


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
    show_progress: bool = True,
    val_windows: np.ndarray | None = None,
    early_stopping: EarlyStoppingConfig | None = None,
) -> TrainingHistory:
    """Train USAD with a stabilised two-phase adversarial loss schedule.

    The adversarial weight ramps linearly from 0 to `config.adv_max_weight`
    over `config.adv_ramp_epochs`, and reconstruction takes the remainder:
      - w_adv = min(epoch / adv_ramp_epochs, 1) * adv_max_weight
      - w_rec = 1 - w_adv
    (This replaces the original paper's w_adv = 1 - 1/epoch, which ramps too
    fast and destabilises training on small calibration sets.)

    Optimizer state and the data loader are created once and reused across
    every epoch, so Adam moments and shuffle order are preserved end-to-end.

    Args:
        model: USADConv1d model instance.
        windows: Windowed training dataset of shape (num_windows, window_size, num_features).
        config: Training configuration. `config.epochs` is used when `early_stopping` is None.
        device: Training device.
        show_progress: Whether to show tqdm progress bars.
        val_windows: Optional validation windows for tracking val reconstruction loss.
        early_stopping: Optional early-stopping config. When provided, training runs
            for up to `early_stopping.max_epochs` and stops once `val_recon_loss`
            fails to improve by `min_delta` for `patience` epochs. Requires `val_windows`.

    Returns:
        TrainingHistory with AE1, AE2, and (if val_windows) validation losses per epoch.
    """

    if early_stopping is not None and val_windows is None:
        raise ValueError("early_stopping requires val_windows to be provided.")

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
    val_recon_losses: list[float] | None = [] if val_windows is not None else None

    max_epochs = early_stopping.max_epochs if early_stopping is not None else config.epochs
    best_val = float("inf")
    patience_counter = 0
    best_state_dict: dict | None = None
    best_epoch: int = 0

    ramp_epochs = max(config.adv_ramp_epochs, 1)
    adv_cap = float(config.adv_max_weight)
    pbar_epoch = tqdm(range(1, max_epochs + 1), desc="Training USAD", disable=not show_progress)
    for epoch in pbar_epoch:
        # Linear ramp of w_adv from 0 to adv_max_weight over ramp_epochs.
        # Reconstruction always carries weight (1 - w_adv) >= (1 - adv_cap) > 0.
        w_adv = min(float(epoch) / ramp_epochs, 1.0) * adv_cap
        w_rec = 1.0 - w_adv

        model.train()
        ae1_running = 0.0
        ae2_running = 0.0
        batch_count = 0

        pbar_batch = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, disable=not show_progress)
        for (batch_windows,) in pbar_batch:
            batch_windows = batch_windows.to(device)

            # Phase 1/2: update AE1 (encoder + decoder1)
            optimizer_ae1.zero_grad()
            recon1, _, _ = model(batch_windows)
            recon2_from_recon1 = model.reconstruct_via_decoder2(recon1)
            loss_ae1 = w_rec * mse(batch_windows, recon1) + w_adv * mse(batch_windows, recon2_from_recon1)
            loss_ae1.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(ae1_params, config.grad_clip_norm)
            optimizer_ae1.step()

            # Phase 2: update AE2 (encoder + decoder2)
            optimizer_ae2.zero_grad()
            _, recon2, _ = model(batch_windows)
            recon2_from_recon1 = model.reconstruct_via_decoder2(recon1.detach())
            loss_ae2 = w_rec * mse(batch_windows, recon2) - w_adv * mse(batch_windows, recon2_from_recon1)
            loss_ae2.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(ae2_params, config.grad_clip_norm)
            optimizer_ae2.step()

            ae1_running += loss_ae1.item()
            ae2_running += loss_ae2.item()
            batch_count += 1

        ae1_epoch_losses.append(ae1_running / max(batch_count, 1))
        ae2_epoch_losses.append(ae2_running / max(batch_count, 1))

        if val_windows is not None:
            val_loss = _evaluate_reconstruction_loss(model, val_windows, config.batch_size, device)
            val_recon_losses.append(val_loss)

            # Track the best checkpoint by val_recon_loss so the saved model
            # reflects the optimum, not whatever state we landed on at early-stop.
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_epoch = epoch
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

            pbar_epoch.set_postfix(
                {
                    "w_adv": f"{w_adv:.2f}",
                    "AE1": f"{ae1_epoch_losses[-1]:.4f}",
                    "Val": f"{val_loss:.4f}",
                    "Best": f"{best_val:.4f}@{best_epoch}",
                }
            )

            if early_stopping is not None:
                if improved and (best_val == val_loss):
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stopping.patience:
                    pbar_epoch.set_description("Early stopping triggered")
                    break
        else:
            pbar_epoch.set_postfix(
                {"w_adv": f"{w_adv:.2f}", "AE1": f"{ae1_epoch_losses[-1]:.4f}"}
            )

    # Restore best checkpoint so downstream scoring uses the optimum,
    # not whatever (often degraded) state we landed on at early-stop time.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return TrainingHistory(
        ae1_losses=ae1_epoch_losses,
        ae2_losses=ae2_epoch_losses,
        val_recon_losses=val_recon_losses,
    )


def train_usad_with_validation(
    model: USADConv1d,
    windows: np.ndarray,
    train_config: TrainingConfig,
    early_stopping: EarlyStoppingConfig,
    device: torch.device | str = "cpu",
) -> TrainingHistory:
    """Train USAD with early stopping based on validation reconstruction loss.

    Thin wrapper that splits `windows` into train/val and delegates to `train_usad`.

    Args:
        model: USADConv1d model instance.
        windows: Windowed dataset of shape (num_windows, window_size, num_features).
        train_config: Training configuration.
        early_stopping: Early stopping configuration.
        device: Training device.

    Returns:
        TrainingHistory including validation reconstruction losses.
    """

    train_windows, val_windows = split_train_validation(windows, early_stopping.val_fraction)
    return train_usad(
        model,
        train_windows,
        train_config,
        device=device,
        show_progress=True,
        val_windows=val_windows,
        early_stopping=early_stopping,
    )


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
