"""Plot anomaly scores over time with failure intervals highlighted."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from edgesense.data_ingestion import load_failure_reports, load_metropt_dataset
from edgesense.models import USADConv1d, USADConv1dConfig
from edgesense.preprocessing import MetroPTPreprocessor, build_healthy_mask
from edgesense.scoring import ScoringConfig, ThresholdConfig, compute_threshold, compute_usad_scores
from edgesense.training import EarlyStoppingConfig, TrainingConfig, train_usad_with_validation
from edgesense.windowing import build_window_mask, create_sliding_windows


def main() -> None:
    """Train USAD and generate anomaly score timeline plot."""

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

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
    scoring_config = ScoringConfig(alpha=0.5, beta=0.5, batch_size=512)
    scores = compute_usad_scores(model, all_windows.windows, scoring_config)

    healthy_scores = scores[window_mask]
    threshold = compute_threshold(healthy_scores, ThresholdConfig(method="percentile", value=99.0))

    # Use window midpoints for plotting.
    start_times = pd.to_datetime(all_windows.start_times)
    end_times = pd.to_datetime(all_windows.end_times)
    mid_times = start_times + (end_times - start_times) / 2

    plot_path = output_dir / "anomaly_score_timeline.png"
    plot_scores(mid_times, scores, failures, threshold, plot_path)
    print(f"Saved plot to: {plot_path.resolve()}")


def plot_scores(
    times: pd.Series,
    scores: pd.Series | list[float],
    failures: pd.DataFrame,
    threshold: float,
    output_path: Path,
) -> None:
    """Plot score timeline with failure intervals."""

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, scores, linewidth=0.7, color="#2c7fb8", label="Anomaly score")
    ax.axhline(threshold, color="#d95f0e", linestyle="--", label="Threshold (99th pct)")

    for _, row in failures.iterrows():
        start = pd.to_datetime(row["start_time"])
        end = pd.to_datetime(row["end_time"])
        ax.axvspan(start, end, color="#e34a33", alpha=0.2, label="Failure interval")

    # Deduplicate legend entries.
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


if __name__ == "__main__":
    main()
