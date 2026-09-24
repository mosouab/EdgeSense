"""Generate the figure set for the multi-dataset chapters of the README.

Produces:
    figures/09_hydraulic_per_component.png  bar chart of AUC / F1 per component
    figures/12_metropt_health_score.png     Metro.PT health score timeline
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from edgesense.health import health_score

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

C_HEALTHY = "#2c7fb8"
C_ANOMALY = "#d7301f"
C_THRESHOLD = "#fdae61"
C_ORACLE = "#1a9850"
C_NEUTRAL = "#525252"


def main() -> None:
    print("[1/2] Hydraulic per-component bar chart...")
    plot_hydraulic_components()

    print("[2/2] Metro.PT health score timeline...")
    plot_metropt_health_score()

    print("Done.")


def plot_hydraulic_components() -> None:
    payload = json.loads(Path("reports/hydraulic_evaluation/metrics.json").read_text())
    components = list(payload["components"].keys())
    auc = [payload["components"][c]["raw"]["auc"] for c in components]
    f1 = [payload["components"][c]["raw"]["f1"] for c in components]

    x = np.arange(len(components))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars_auc = ax.bar(x - width / 2, auc, width, color=C_HEALTHY, label="ROC-AUC")
    bars_f1 = ax.bar(x + width / 2, f1, width, color=C_ANOMALY, label="F1 (p99 threshold)")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in components])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Hydraulic systems: per-component fault detection (one USAD per component)",
        fontsize=12,
        fontweight="bold",
    )
    ax.axhline(0.5, color=C_NEUTRAL, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    for bars in [bars_auc, bars_f1]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_hydraulic_per_component.png", dpi=150)
    plt.close(fig)


def plot_metropt_health_score() -> None:
    """Use the existing Metro.PT scores to compute a Health Score timeline."""

    timeline_path = Path("reports/full_evaluation/scores_timeline.csv")
    metrics_path = Path("reports/full_evaluation/metrics.json")
    if not timeline_path.exists() or not metrics_path.exists():
        print("  Metro.PT artifacts missing; skipping health score figure.")
        return

    timeline = pd.read_csv(timeline_path, parse_dates=["window_start", "window_end", "window_mid"])
    metrics = json.loads(metrics_path.read_text())
    threshold = float(metrics["thresholds"]["headline_recalibrated"])
    recal_end = pd.Timestamp(metrics["split"]["recal_end"])

    # Healthy reference: the recalibration window (label-free).
    recal_scores = timeline[timeline["window_start"] < recal_end]["score_smoothed"].to_numpy()
    test = timeline[timeline["window_start"] >= recal_end].copy()
    test["health"] = health_score(test["score_smoothed"].to_numpy(), recal_scores, threshold)

    from edgesense.datasets.metropt import load_metropt_failures
    failures = load_metropt_failures()

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(test["window_mid"], test["health"], color=C_HEALTHY, linewidth=0.7, label="Health Score")
    ax.fill_between(test["window_mid"], 0, test["health"], color=C_HEALTHY, alpha=0.15)
    failure_handle = None
    for _, row in failures.iterrows():
        start = pd.to_datetime(row["start_time"])
        end = pd.to_datetime(row["end_time"])
        failure_handle = ax.axvspan(start, end, color=C_ANOMALY, alpha=0.18)

    handles, labels = ax.get_legend_handles_labels()
    if failure_handle is not None:
        handles.append(failure_handle)
        labels.append("Labeled failure interval")
    ax.legend(handles, labels, loc="lower left")

    ax.set_ylim(-2, 102)
    ax.set_ylabel("Health Score (%)")
    ax.set_xlabel("Date")
    ax.set_title(
        "Metro.PT — Health Score on the test horizon (100 = recal-window healthy, 0 = at alert threshold)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "12_metropt_health_score.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
