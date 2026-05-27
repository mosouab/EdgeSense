"""Compute PR curve, optimal F1 threshold, and point-adjusted metrics."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from edgesense.data_ingestion import load_failure_reports, load_metropt_dataset
from edgesense.evaluation import label_windows_by_failures
from edgesense.models import USADConv1d, USADConv1dConfig
from edgesense.preprocessing import MetroPTPreprocessor, build_healthy_mask
from edgesense.scoring import ScoringConfig, compute_usad_scores
from edgesense.training import EarlyStoppingConfig, TrainingConfig, train_usad_with_validation
from edgesense.windowing import build_window_mask, create_sliding_windows


def main() -> None:
    """Run full PR analysis with point adjustment."""

    dataset = load_metropt_dataset()
    failures = load_failure_reports()

    preprocessor = MetroPTPreprocessor(
        feature_columns=dataset.feature_columns,
        timestamp_col=dataset.timestamp_col,
    )
    scaled_full = preprocessor.fit_transform(dataset, failures)

    window_size = 20
    stride = 10

    # Train on healthy windows only.
    row_mask = build_healthy_mask(dataset.data, dataset.timestamp_col, failures)
    window_mask = build_window_mask(row_mask, window_size=window_size, stride=stride)
    train_windows = create_sliding_windows(
        scaled_full,
        window_size=window_size,
        stride=stride,
        window_mask=window_mask,
    )

    model = USADConv1d(
        USADConv1dConfig(
            in_features=scaled_full.shape[1],
            base_channels=32,
            latent_channels=64,
            downsample_layers=2,
        )
    )
    train_config = TrainingConfig(batch_size=256, epochs=1, learning_rate=1e-3)
    stop_config = EarlyStoppingConfig(patience=5, min_delta=1e-4, max_epochs=50, val_fraction=0.1)
    train_usad_with_validation(model, train_windows.windows, train_config, stop_config)

    # Score all windows.
    all_windows = create_sliding_windows(
        scaled_full,
        window_size=window_size,
        stride=stride,
        timestamps=dataset.data[dataset.timestamp_col],
    )
    labels = label_windows_by_failures(
        all_windows.start_times,
        all_windows.end_times,
        failures,
    )

    scores = compute_usad_scores(
        model,
        all_windows.windows,
        ScoringConfig(alpha=0.5, beta=0.5, batch_size=512),
    )

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = _f1_from_pr(precision, recall)

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])

    preds = scores >= best_threshold
    adjusted_preds = point_adjust_predictions(labels, preds)

    raw_metrics = {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, scores) if labels.any() and (~labels).any() else float("nan"),
    }
    adjusted_metrics = {
        "precision": precision_score(labels, adjusted_preds, zero_division=0),
        "recall": recall_score(labels, adjusted_preds, zero_division=0),
        "f1": f1_score(labels, adjusted_preds, zero_division=0),
    }

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "precision_recall_curve.png"
    plot_pr_curve(recall, precision, plot_path)

    print("Precision-Recall Curve computed.")
    print(f"Optimal threshold (max F1): {best_threshold:.6f}")
    print(
        "Raw metrics at optimal threshold -> "
        f"Precision: {raw_metrics['precision']:.4f} | "
        f"Recall: {raw_metrics['recall']:.4f} | "
        f"F1: {raw_metrics['f1']:.4f} | "
        f"AUC: {raw_metrics['auc']:.4f}"
    )
    print(
        "Point-adjusted metrics -> "
        f"Precision: {adjusted_metrics['precision']:.4f} | "
        f"Recall: {adjusted_metrics['recall']:.4f} | "
        f"F1: {adjusted_metrics['f1']:.4f}"
    )
    print(f"Saved PR curve plot to: {plot_path.resolve()}")


def _f1_from_pr(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """Compute F1 scores aligned with precision/recall from PR curve.

    The last PR point has no corresponding threshold, so it is excluded.
    """

    precision = precision[:-1]
    recall = recall[:-1]
    return (2 * precision * recall) / (precision + recall + 1e-12)


def point_adjust_predictions(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Apply USAD-style point adjustment over contiguous anomaly segments."""

    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape.")

    adjusted = predictions.copy()
    idx = 0
    while idx < len(labels):
        if labels[idx]:
            start = idx
            while idx < len(labels) and labels[idx]:
                idx += 1
            end = idx
            if predictions[start:end].any():
                adjusted[start:end] = True
        else:
            idx += 1
    return adjusted


def plot_pr_curve(recall: np.ndarray, precision: np.ndarray, output_path: Path) -> None:
    """Plot and save the precision-recall curve."""

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color="#1f78b4", linewidth=1.5)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
