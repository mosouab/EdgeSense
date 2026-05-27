"""Generate a zoomed-in binary timeline for the June failure."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from edgesense.data_ingestion import load_failure_reports

def main() -> None:
    reports_dir = Path("reports") / "full_evaluation"
    timeline_path = reports_dir / "scores_timeline.csv"
    
    # Load data
    timeline = pd.read_csv(timeline_path, parse_dates=["window_mid"])
    failures = load_failure_reports()
    
    # Target June failure (Failure ID 3)
    june_failure = failures.iloc[2]
    start_zoom = pd.to_datetime(june_failure["start_time"]) - pd.Timedelta(hours=12)
    end_zoom = pd.to_datetime(june_failure["end_time"]) + pd.Timedelta(hours=12)
    
    # Filter timeline
    mask = timeline["window_mid"].between(start_zoom, end_zoom)
    subset = timeline[mask].copy()
    
    # Convert binary columns
    subset["persistence_prediction"] = subset["persistence_prediction"].astype(int)
    
    output_path = Path("figures/june_failure_zoom.png")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot Step function for binary predictions
    ax.step(subset["window_mid"], subset["persistence_prediction"], 
            where="post", color="#2c7fb8", linewidth=2, label="EdgeSense Prediction")
    
    # Highlight failure interval
    ax.axvspan(june_failure["start_time"], june_failure["end_time"], 
               color="#e34a33", alpha=0.3, label="Actual Failure (Air Leak)")
    
    ax.set_title("Detailed Performance: June Air Leak Event", fontsize=14)
    ax.set_xlabel("Date/Time", fontsize=12)
    ax.set_ylabel("System State", fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Fault Identified"])
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.legend(loc="upper left")
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Zoomed plot saved to {output_path.resolve()}")

if __name__ == "__main__":
    main()
