"""Generate training-loss and latent-space projection plots."""

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from edgesense.data_ingestion import load_failure_reports, load_metropt_dataset
from edgesense.evaluation import label_windows_by_failures
from edgesense.models import USADConv1d, USADConv1dConfig
from edgesense.preprocessing import MetroPTPreprocessor, build_healthy_mask
from edgesense.training import EarlyStoppingConfig, TrainingConfig, train_usad_with_validation
from edgesense.windowing import build_window_mask, create_sliding_windows


def main() -> None:
    """Train USAD and produce training/latent-space plots."""

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_metropt_dataset()
    failures = load_failure_reports()

    # Train on 30 days of healthy data before the first failure.
    first_failure = failures.iloc[0]
    train_end = pd.to_datetime(first_failure["start_time"]) - pd.Timedelta(hours=1)
    train_start = train_end - pd.Timedelta(days=30)
    train_mask = dataset.data[dataset.timestamp_col].between(train_start, train_end, inclusive="both")
    train_data = dataset.data.loc[train_mask].reset_index(drop=True)

    train_dataset = dataset.__class__(
        data=train_data,
        feature_columns=dataset.feature_columns,
        timestamp_col=dataset.timestamp_col,
        sampling_interval_seconds=dataset.sampling_interval_seconds,
        start_time=train_data[dataset.timestamp_col].iloc[0],
        end_time=train_data[dataset.timestamp_col].iloc[-1],
    )

    preprocessor = MetroPTPreprocessor(
        feature_columns=train_dataset.feature_columns,
        timestamp_col=train_dataset.timestamp_col,
    )
    scaled_train = preprocessor.fit_transform(train_dataset, failures)

    row_mask = build_healthy_mask(train_dataset.data, train_dataset.timestamp_col, failures)
    window_mask = build_window_mask(row_mask, window_size=20, stride=10)
    train_windows = create_sliding_windows(
        scaled_train,
        window_size=20,
        stride=10,
        window_mask=window_mask,
    )

    model = USADConv1d(
        USADConv1dConfig(
            in_features=scaled_train.shape[1],
            base_channels=32,
            latent_channels=64,
            downsample_layers=2,
        )
    )
    train_config = TrainingConfig(batch_size=128, epochs=1, learning_rate=1e-3)
    stop_config = EarlyStoppingConfig(patience=5, min_delta=1e-4, max_epochs=50, val_fraction=0.1)

    history = train_usad_with_validation(model, train_windows.windows, train_config, stop_config)
    plot_training_losses(
        history.ae1_losses,
        history.ae2_losses,
        history.val_recon_losses or [],
        output_dir / "training_losses.png",
    )

    # Build evaluation windows around the first failure for latent projection.
    eval_start = pd.to_datetime(first_failure["start_time"]) - pd.Timedelta(hours=12)
    eval_end = pd.to_datetime(first_failure["end_time"]) + pd.Timedelta(hours=12)
    eval_mask = dataset.data[dataset.timestamp_col].between(eval_start, eval_end, inclusive="both")
    eval_data = dataset.data.loc[eval_mask].reset_index(drop=True)

    eval_dataset = dataset.__class__(
        data=eval_data,
        feature_columns=dataset.feature_columns,
        timestamp_col=dataset.timestamp_col,
        sampling_interval_seconds=dataset.sampling_interval_seconds,
        start_time=eval_data[dataset.timestamp_col].iloc[0],
        end_time=eval_data[dataset.timestamp_col].iloc[-1],
    )

    scaled_eval = preprocessor.transform(eval_dataset)
    eval_windows = create_sliding_windows(
        scaled_eval,
        window_size=20,
        stride=10,
        timestamps=eval_dataset.data[eval_dataset.timestamp_col],
    )

    labels = label_windows_by_failures(
        eval_windows.start_times,
        eval_windows.end_times,
        failures,
    )
    latent_vectors, latent_labels = sample_latent_vectors(
        model,
        eval_windows.windows,
        labels,
        max_normal=2000,
        seed=42,
    )
    plot_latent_projection(
        latent_vectors,
        latent_labels,
        output_dir / "latent_pca.png",
    )

    print(f"Saved plots to: {output_dir.resolve()}")


def plot_training_losses(
    ae1_losses: list[float],
    ae2_losses: list[float],
    val_losses: list[float],
    output_path: Path,
) -> None:
    """Plot training and validation losses."""

    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(ae1_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, ae1_losses, label="AE1 Loss")
    plt.plot(epochs, ae2_losses, label="AE2 Loss")
    if val_losses:
        plt.plot(epochs, val_losses, label="Val Recon Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("USAD Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def sample_latent_vectors(
    model: USADConv1d,
    windows: np.ndarray,
    labels: np.ndarray,
    max_normal: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract latent vectors for a balanced subset of windows."""

    if windows.ndim != 3:
        raise ValueError("Windows must have shape (num_windows, window_size, num_features).")

    normal_indices = np.where(~labels)[0]
    anomaly_indices = np.where(labels)[0]
    random.seed(seed)

    if normal_indices.size > max_normal:
        normal_indices = np.array(random.sample(list(normal_indices), max_normal))

    selected_indices = np.concatenate([normal_indices, anomaly_indices])
    selected_windows = windows[selected_indices]
    selected_labels = labels[selected_indices]

    model.eval()
    device = next(model.parameters()).device
    batch_size = 256
    latent_vectors: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, selected_windows.shape[0], batch_size):
            batch = torch.tensor(
                selected_windows[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            latent = model.encode(batch)
            latent_mean = latent.mean(dim=2)
            latent_vectors.append(latent_mean.cpu().numpy())

    return np.vstack(latent_vectors), selected_labels


def plot_latent_projection(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """Plot a 2D PCA projection of latent vectors."""

    import matplotlib.pyplot as plt

    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(latent_vectors)

    plt.figure(figsize=(6, 6))
    plt.scatter(
        projected[~labels, 0],
        projected[~labels, 1],
        s=10,
        alpha=0.6,
        label="Normal",
    )
    if labels.any():
        plt.scatter(
            projected[labels, 0],
            projected[labels, 1],
            s=12,
            alpha=0.8,
            label="Anomaly",
            color="red",
        )
    plt.title("Latent Space Projection (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
