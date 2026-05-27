"""Plot binary anomaly predictions with failure intervals highlighted."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from edgesense.data_ingestion import load_failure_reports


def main() -> None:
    """Generate a binary anomaly timeline plot."""

    reports_dir = Path("reports") / "full_evaluation"
    timeline_path = reports_dir / "scores_timeline.csv"
    if not timeline_path.exists():
        raise FileNotFoundError(f"Missing {timeline_path}. Run run_full_evaluation.py first.")

    timeline = pd.read_csv(timeline_path, parse_dates=["window_mid"])
    failures = load_failure_reports()

    time_series = timeline["window_mid"]
    binary = timeline["persistence_prediction"].astype(int)

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "binary_fault_timeline.png"

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.step(time_series, binary, where="post", color="#2c7fb8", label="Prediction (binary)")

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

    ax.set_title("Binary Anomaly Predictions with Failure Intervals")
    ax.set_xlabel("Time")
    ax.set_ylabel("Prediction")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Fault"])
    ax.legend(unique_handles, unique_labels, loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved plot to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
