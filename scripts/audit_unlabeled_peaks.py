"""Inspect the top unlabeled high-score plateaus on the test horizon.

For each plateau, plot the raw sensor traces alongside a reference healthy day
from the training period. The result lets a human classify each plateau as
(a) a real undocumented anomaly, or (b) a model drift artifact.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from edgesense.data_ingestion import load_failure_reports, load_metropt_dataset

OUTPUT = Path("figures") / "08_unlabeled_plateau_audit.png"

KEY_SENSORS = ["TP2", "Oil_temperature", "Motor_current", "Reservoirs"]
SENSOR_LABELS = {
    "TP2": "TP2 (bar)",
    "Oil_temperature": "Oil temp (°C)",
    "Motor_current": "Motor (A)",
    "Reservoirs": "Reservoirs (bar)",
}

# Reference healthy day from training period (well before any failure).
REFERENCE_DAY = pd.Timestamp("2020-02-20")

# Plateau periods identified by aggregating runs of score > 3 on the test horizon
# (figures shown are the top 3 by duration * peak score).
PLATEAUS = [
    {
        "label": "May 26–28 (42 h, peak 9.9)",
        "start": pd.Timestamp("2020-05-26 06:00"),
        "end": pd.Timestamp("2020-05-28 06:00"),
    },
    {
        "label": "Jun 22–25 (62 h, peak 8.9)",
        "start": pd.Timestamp("2020-06-22 12:00"),
        "end": pd.Timestamp("2020-06-25 09:00"),
    },
    {
        "label": "Apr 20–21 (21 h, peak 6.7)",
        "start": pd.Timestamp("2020-04-20 00:00"),
        "end": pd.Timestamp("2020-04-21 06:00"),
    },
]


def main() -> None:
    print("Loading raw dataset...")
    dataset = load_metropt_dataset()
    df = dataset.data
    ts_col = dataset.timestamp_col
    df[ts_col] = pd.to_datetime(df[ts_col])

    # Slice the reference day (24 hours from REFERENCE_DAY).
    ref_slice = _slice(df, ts_col, REFERENCE_DAY, REFERENCE_DAY + pd.Timedelta(days=1))

    # Slice each plateau (plus a small pad on each side so the onset is visible).
    plateau_slices = [
        (p["label"], _slice(df, ts_col, p["start"] - pd.Timedelta(hours=2), p["end"] + pd.Timedelta(hours=2)))
        for p in PLATEAUS
    ]

    print("Rendering audit figure...")
    n_cols = 1 + len(plateau_slices)
    n_rows = len(KEY_SENSORS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols + 1, 2.2 * n_rows),
        sharey="row",
    )

    column_titles = ["REFERENCE — Feb 20 (healthy)"] + [p["label"] for p in PLATEAUS]
    column_colors = ["#2c7fb8"] + ["#d7301f"] * len(plateau_slices)

    for col_idx, (title, color) in enumerate(zip(column_titles, column_colors)):
        ax = axes[0, col_idx]
        ax.set_title(title, fontsize=10, fontweight="bold", color=color)

    for row_idx, sensor in enumerate(KEY_SENSORS):
        # Reference column
        ax = axes[row_idx, 0]
        ax.plot(ref_slice[ts_col], ref_slice[sensor], color=column_colors[0], linewidth=0.5)
        ax.set_ylabel(SENSOR_LABELS[sensor], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        # Plateau columns
        for col_offset, (label, segment) in enumerate(plateau_slices, start=1):
            ax = axes[row_idx, col_offset]
            ax.plot(segment[ts_col], segment[sensor], color=column_colors[col_offset], linewidth=0.5)
            # Shade the actual plateau interval (excluding pad)
            plateau = PLATEAUS[col_offset - 1]
            ax.axvspan(plateau["start"], plateau["end"], color="#d7301f", alpha=0.10)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)

        if row_idx < n_rows - 1:
            for ax in axes[row_idx, :]:
                ax.tick_params(axis="x", labelbottom=False)

    fig.suptitle(
        "Audit: top unlabeled high-score plateaus vs. reference healthy day",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT.resolve()}")

    print("\nSummary statistics per sensor per slice:")
    summary_rows = []
    for sensor in KEY_SENSORS:
        ref_stats = ref_slice[sensor].agg(["mean", "std", "min", "max"])
        row = {"sensor": sensor, "reference_mean": ref_stats["mean"], "reference_std": ref_stats["std"]}
        for label, segment in plateau_slices:
            row[f"{label[:12]}_mean"] = segment[sensor].mean()
            row[f"{label[:12]}_std"] = segment[sensor].std()
        summary_rows.append(row)
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def _slice(df: pd.DataFrame, ts_col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df[ts_col] >= start) & (df[ts_col] < end)]


if __name__ == "__main__":
    main()
