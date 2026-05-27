"""Run full evaluation, generate plots, and save model artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys
import random

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from edgesense.data_ingestion import load_failure_reports, load_metropt_dataset
from edgesense.evaluation import (
    apply_median_filter,
    apply_temporal_persistence,
    compute_optimal_f1_threshold,
    point_adjust_predictions,
    label_windows_by_failures,
)
from edgesense.models import USADConv1d, USADConv1dConfig
from edgesense.preprocessing import MetroPTPreprocessor, build_healthy_mask
from edgesense.scoring import ScoringConfig, compute_usad_scores
from edgesense.training import EarlyStoppingConfig, TrainingConfig, train_usad_with_validation
from edgesense.windowing import build_window_mask, create_sliding_windows


def main() -> None:
    """Train USAD, evaluate, and generate artifacts."""

    print("--- EdgeSense Full Evaluation Pipeline ---")
    
    output_dir = Path("reports") / "full_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading and Preprocessing data...")
    dataset = load_metropt_dataset()
    failures = load_failure_reports()

    preprocessor = MetroPTPreprocessor(
        feature_columns=dataset.feature_columns,
        timestamp_col=dataset.timestamp_col,
    )
    scaled_full = preprocessor.fit_transform(dataset, failures)
    preprocessor.save(output_dir / "preprocessor.pkl")

    window_size = 100
    stride = 50

    row_mask = build_healthy_mask(dataset.data, dataset.timestamp_col, failures)
    window_mask = build_window_mask(row_mask, window_size=window_size, stride=stride)

    train_windows = create_sliding_windows(
        scaled_full,
        window_size=window_size,
        stride=stride,
        window_mask=window_mask,
    )
    all_windows = create_sliding_windows(
        scaled_full,
        window_size=window_size,
        stride=stride,
        timestamps=dataset.data[dataset.timestamp_col],
    )

    model_config = USADConv1dConfig(
        in_features=scaled_full.shape[1],
        base_channels=32,
        latent_channels=64,
        downsample_layers=2,
    )
    model = USADConv1d(model_config)

    train_config = TrainingConfig(batch_size=256, epochs=1, learning_rate=1e-3)
    stop_config = EarlyStoppingConfig(patience=5, min_delta=1e-4, max_epochs=50, val_fraction=0.1)
    history = train_usad_with_validation(model, train_windows.windows, train_config, stop_config)

    torch.save(model.state_dict(), output_dir / "usad_conv1d.pt")
    with (output_dir / "model_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(model_config), handle, indent=2)

    scores_raw = compute_usad_scores(
        model,
        all_windows.windows,
        ScoringConfig(alpha=0.3, beta=0.7, batch_size=512),
    )
    labels = label_windows_by_failures(
        all_windows.start_times,
        all_windows.end_times,
        failures,
    )

    scores_smoothed = apply_median_filter(scores_raw, window_size=11)
    optimal_threshold = compute_optimal_f1_threshold(labels, scores_smoothed)
    predictions = scores_smoothed >= optimal_threshold
    adjusted_predictions = point_adjust_predictions(labels, predictions)
    persistence_predictions = apply_temporal_persistence(predictions, min_consecutive=25)
    persistence_adjusted = point_adjust_predictions(labels, persistence_predictions)

    metrics = compute_metrics(labels, predictions, scores_smoothed)
    adjusted_metrics = compute_metrics(labels, adjusted_predictions, scores_smoothed, include_auc=False)
    persistence_metrics = compute_metrics(labels, persistence_predictions, scores_smoothed, include_auc=False)
    persistence_adjusted_metrics = compute_metrics(labels, persistence_adjusted, scores_smoothed, include_auc=False)

    metrics_payload = {
        "optimal_threshold": optimal_threshold,
        "raw": metrics,
        "point_adjusted": adjusted_metrics,
        "persistence": persistence_metrics,
        "persistence_point_adjusted": persistence_adjusted_metrics,
        "training": {
            "epochs": len(history.ae1_losses),
            "final_ae1_loss": history.ae1_losses[-1],
            "final_ae2_loss": history.ae2_losses[-1],
            "final_val_recon_loss": history.val_recon_losses[-1] if history.val_recon_losses else None,
        },
        "windowing": {"window_size": window_size, "stride": stride},
        "score_smoothing": {"median_window": 11},
        "temporal_persistence": {"min_consecutive": 25},
        "train_windows": int(train_windows.windows.shape[0]),
        "eval_windows": int(all_windows.windows.shape[0]),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    save_training_history(history, output_dir / "training_history.csv")
    save_score_timeline(
        all_windows.start_times,
        all_windows.end_times,
        scores_raw,
        scores_smoothed,
        labels,
        predictions,
        adjusted_predictions,
        persistence_predictions,
        persistence_adjusted,
        output_dir / "scores_timeline.csv",
    )
    save_pr_curve(
        labels,
        scores_smoothed,
        output_dir / "precision_recall_curve.png",
        output_dir / "pr_curve.csv",
    )
    save_training_plot(history, output_dir / "training_losses.png")
    save_latent_projection(
        model,
        all_windows.windows,
        labels,
        output_dir / "latent_pca.png",
        max_normal=2000,
        seed=42,
    )
    save_anomaly_timeline(
        all_windows.start_times,
        all_windows.end_times,
        scores_smoothed,
        failures,
        optimal_threshold,
        output_dir / "anomaly_score_timeline.png",
    )

    print(f"Saved evaluation artifacts to: {output_dir.resolve()}")


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    include_auc: bool = True,
) -> dict[str, float | None]:
    """Compute precision/recall/F1/accuracy and optional AUC."""

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    auc = None
    if include_auc and labels.any() and (~labels).any():
        auc = roc_auc_score(labels, scores)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "auc": float(auc) if auc is not None else None,
    }


def save_training_history(history, output_path: Path) -> None:
    """Save training history to CSV."""

    data = {
        "epoch": list(range(1, len(history.ae1_losses) + 1)),
        "ae1_loss": history.ae1_losses,
        "ae2_loss": history.ae2_losses,
        "val_recon_loss": history.val_recon_losses or [],
    }
    pd.DataFrame(data).to_csv(output_path, index=False)


def save_training_plot(history, output_path: Path) -> None:
    """Plot training losses."""

    epochs = np.arange(1, len(history.ae1_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history.ae1_losses, label="AE1 Loss")
    plt.plot(epochs, history.ae2_losses, label="AE2 Loss")
    if history.val_recon_losses:
        plt.plot(epochs, history.val_recon_losses, label="Val Recon Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("USAD Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_latent_projection(
    model: USADConv1d,
    windows: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    max_normal: int,
    seed: int,
) -> None:
    """Plot PCA projection of latent vectors."""

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

    latent_vectors = np.vstack(latent_vectors)
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(latent_vectors)

    plt.figure(figsize=(6, 6))
    plt.scatter(
        projected[~selected_labels, 0],
        projected[~selected_labels, 1],
        s=10,
        alpha=0.6,
        label="Normal",
    )
    if selected_labels.any():
        plt.scatter(
            projected[selected_labels, 0],
            projected[selected_labels, 1],
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


def save_anomaly_timeline(
    start_times: pd.Series,
    end_times: pd.Series,
    scores: np.ndarray,
    failures: pd.DataFrame,
    threshold: float,
    output_path: Path,
) -> None:
    """Plot score timeline with failure intervals."""

    start_series = pd.to_datetime(start_times, errors="raise")
    end_series = pd.to_datetime(end_times, errors="raise")
    mid_times = start_series + (end_series - start_series) / 2

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(mid_times, scores, linewidth=0.7, color="#2c7fb8", label="Anomaly score")
    ax.axhline(threshold, color="#d95f0e", linestyle="--", label="Optimal threshold")

    for _, row in failures.iterrows():
        start = pd.to_datetime(row["start_time"])
        end = pd.to_datetime(row["end_time"])
        ax.axvspan(start, end, color="#e34a33", alpha=0.2, label="Failure interval")

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            unique_handles.append(handle)
            unique_labels.append(label)
            seen.add(label)

    ax.set_title("USAD Anomaly Scores with Failure Intervals")
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    ax.legend(unique_handles, unique_labels, loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_pr_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    plot_path: Path,
    csv_path: Path,
) -> None:
    """Save PR curve plot and CSV data."""

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(thresholds, np.nan),
        }
    )
    pr_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color="#1f78b4", linewidth=1.5)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def save_score_timeline(
    start_times: pd.Series,
    end_times: pd.Series,
    scores_raw: np.ndarray,
    scores_smoothed: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    adjusted_predictions: np.ndarray,
    persistence_predictions: np.ndarray,
    persistence_adjusted: np.ndarray,
    output_path: Path,
) -> None:
    """Save scores and labels per window to CSV."""

    start_series = pd.to_datetime(start_times, errors="raise")
    end_series = pd.to_datetime(end_times, errors="raise")
    mid_times = start_series + (end_series - start_series) / 2

    df = pd.DataFrame(
        {
            "window_start": start_series,
            "window_end": end_series,
            "window_mid": mid_times,
            "score_raw": scores_raw,
            "score_smoothed": scores_smoothed,
            "label": labels,
            "prediction": predictions,
            "adjusted_prediction": adjusted_predictions,
            "persistence_prediction": persistence_predictions,
            "persistence_adjusted_prediction": persistence_adjusted,
        }
    )
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
