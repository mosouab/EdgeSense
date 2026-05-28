"""Generate the canonical figure set for the EdgeSense POC.

This script reads the artifacts produced by `scripts/run_full_evaluation.py`
(plus the raw dataset and the trained model) and renders a structured,
numbered figure set into `figures/`:

    01_sensor_overview.png         Raw sensor traces (healthy vs failure day)
    02_training_curves.png         Loss curves + w_adv ramp + best epoch
    03_anomaly_score_timeline.png  Headline detection result on eval period
    04_score_distribution.png      Histogram of healthy vs failure scores
    05_precision_recall_curve.png  PR curve with operating points
    06_latent_pca.png              Latent space projection
    07_june_failure_zoom.png       Zoomed view of one failure event
"""

from __future__ import annotations

from pathlib import Path
import json
import random
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from edgesense.data_ingestion import load_failure_reports, load_metropt_dataset
from edgesense.models import USADConv1d, USADConv1dConfig
from edgesense.preprocessing import MetroPTPreprocessor
from edgesense.windowing import create_sliding_windows

OUTPUT_DIR = Path("figures")
ARTIFACTS_DIR = Path("reports") / "full_evaluation"
TRAIN_END = pd.Timestamp("2020-04-01")

C_HEALTHY = "#2c7fb8"
C_ANOMALY = "#d7301f"
C_THRESHOLD = "#fdae61"
C_ORACLE = "#1a9850"
C_NEUTRAL = "#525252"
C_BEST = "#7a0177"

KEY_SENSORS = ["TP2", "Oil_temperature", "Motor_current", "Reservoirs"]
SENSOR_LABELS = {
    "TP2": "TP2: Compressed air (bar)",
    "Oil_temperature": "Oil temperature (°C)",
    "Motor_current": "Motor current (A)",
    "Reservoirs": "Reservoirs pressure (bar)",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading artifacts...")
    timeline = pd.read_csv(
        ARTIFACTS_DIR / "scores_timeline.csv",
        parse_dates=["window_start", "window_end", "window_mid"],
    )
    history = pd.read_csv(ARTIFACTS_DIR / "training_history.csv")
    metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
    pr_data = pd.read_csv(ARTIFACTS_DIR / "pr_curve.csv")
    failures = load_failure_reports()

    headline_threshold = metrics["thresholds"]["headline_recalibrated"]
    training_threshold = metrics["thresholds"].get("supplementary_training_quantile")
    oracle_threshold = metrics["thresholds"]["reference_oracle_pr_optimal"]
    recal_end = pd.Timestamp(metrics["split"]["recal_end"])

    print("[1/7] Sensor overview (loads raw CSV, ~10s)...")
    dataset = load_metropt_dataset()
    plot_sensor_overview(dataset, failures)

    print("[2/7] Training curves...")
    plot_training_curves(history, metrics)

    print("[3/7] Anomaly score timeline...")
    plot_score_timeline(timeline, failures, headline_threshold, training_threshold, recal_end, metrics)

    print("[4/7] Score distribution...")
    plot_score_distribution(timeline, headline_threshold, oracle_threshold, recal_end)

    print("[5/7] Precision-recall curve...")
    plot_pr_curve(pr_data, metrics, headline_threshold, oracle_threshold)

    print("[6/7] Latent PCA (re-windows eval data + encodes)...")
    plot_latent_pca(dataset, failures, metrics)

    print("[7/7] June failure zoom...")
    plot_june_zoom(timeline, failures, headline_threshold)

    print(f"\nFigures saved to: {OUTPUT_DIR.resolve()}")


# ---------- Individual figure builders ----------

def plot_sensor_overview(dataset, failures: pd.DataFrame) -> None:
    """Side-by-side raw traces from a healthy day and a failure day."""

    df = dataset.data.copy()
    df[dataset.timestamp_col] = pd.to_datetime(df[dataset.timestamp_col])

    healthy_day = pd.Timestamp("2020-02-15")
    failure_day = pd.Timestamp("2020-06-06")  # mid-window of failure #3

    healthy = df[
        (df[dataset.timestamp_col] >= healthy_day)
        & (df[dataset.timestamp_col] < healthy_day + pd.Timedelta(days=1))
    ]
    failure = df[
        (df[dataset.timestamp_col] >= failure_day)
        & (df[dataset.timestamp_col] < failure_day + pd.Timedelta(days=1))
    ]

    fig, axes = plt.subplots(len(KEY_SENSORS), 2, figsize=(13, 9), sharex="col")
    for row, sensor in enumerate(KEY_SENSORS):
        for col, (subset, label, color) in enumerate(
            [
                (healthy, f"Healthy day — {healthy_day.date()}", C_HEALTHY),
                (failure, f"Failure day — {failure_day.date()} (Air Leak #3)", C_ANOMALY),
            ]
        ):
            ax = axes[row, col]
            ax.plot(
                subset[dataset.timestamp_col],
                subset[sensor],
                color=color,
                linewidth=0.6,
            )
            ax.set_ylabel(SENSOR_LABELS[sensor], fontsize=9)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(label, fontsize=11, fontweight="bold")
            if row == len(KEY_SENSORS) - 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                ax.set_xlabel("Time of day")

    fig.suptitle(
        "Metro.PT Air Compressor — raw sensor traces, healthy vs failure",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_DIR / "01_sensor_overview.png", dpi=150)
    plt.close(fig)


def plot_training_curves(history: pd.DataFrame, metrics: dict) -> None:
    """AE1 / AE2 / validation loss with the w_adv schedule overlaid."""

    epochs = history["epoch"].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(epochs, history["ae1_loss"], color=C_HEALTHY, label="AE1 training loss", linewidth=1.8)
    ax.plot(epochs, history["ae2_loss"], color=C_ANOMALY, label="AE2 training loss", linewidth=1.4, alpha=0.7)
    if "val_recon_loss" in history.columns and history["val_recon_loss"].notna().any():
        ax.plot(
            epochs,
            history["val_recon_loss"],
            color=C_NEUTRAL,
            label="Validation reconstruction loss",
            linewidth=2.0,
            linestyle="--",
        )

    # Mark best epoch (lowest val loss)
    if "val_recon_loss" in history.columns:
        best_idx = int(np.argmin(history["val_recon_loss"].to_numpy()))
        best_epoch = int(history["epoch"].iloc[best_idx])
        best_val = float(history["val_recon_loss"].iloc[best_idx])
        ax.axvline(best_epoch, color=C_BEST, linestyle=":", linewidth=1.5)
        ax.annotate(
            f"best @ epoch {best_epoch}\nval = {best_val:.3f}",
            xy=(best_epoch, best_val),
            xytext=(best_epoch + 1, best_val + 0.25),
            color=C_BEST,
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1),
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Overlay w_adv schedule on a secondary axis.
    schedule_cfg = metrics.get("adversarial_schedule", {})
    ramp_epochs = int(schedule_cfg.get("adv_ramp_epochs", 20))
    adv_cap = float(schedule_cfg.get("adv_max_weight", 0.5))
    schedule = np.minimum(epochs / ramp_epochs, 1.0) * adv_cap
    ax2 = ax.twinx()
    ax2.plot(epochs, schedule, color=C_ORACLE, linewidth=1.5, alpha=0.6, label="w_adv (right axis)")
    ax2.set_ylabel("Adversarial weight w_adv", color=C_ORACLE)
    ax2.tick_params(axis="y", colors=C_ORACLE)
    ax2.set_ylim(0, adv_cap * 1.1)
    ax2.legend(loc="upper right")

    ax.set_title("USAD training dynamics — loss curves and adversarial ramp", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_training_curves.png", dpi=150)
    plt.close(fig)


def plot_score_timeline(
    timeline: pd.DataFrame,
    failures: pd.DataFrame,
    threshold: float,
    training_threshold: float | None,
    recal_end: pd.Timestamp,
    metrics: dict,
) -> None:
    """The headline figure: smoothed anomaly score across the eval period."""

    fig, ax = plt.subplots(figsize=(14, 4.5))

    times = timeline["window_mid"]
    ax.plot(
        times,
        timeline["score_smoothed"],
        color=C_HEALTHY,
        linewidth=0.7,
        label="Anomaly score (median-smoothed)",
    )

    # Shade the on-site recalibration window so viewers see which slice was
    # used to refit the threshold versus which slice was actually evaluated.
    recal_start = pd.to_datetime(metrics["split"]["recal_start"])
    recal_span = ax.axvspan(
        recal_start,
        recal_end,
        color=C_ORACLE,
        alpha=0.10,
    )

    ax.axhline(
        threshold,
        color=C_THRESHOLD,
        linestyle="--",
        linewidth=1.7,
        label=f"Recalibrated threshold (p{metrics['thresholds']['headline_quantile']} of recal-window scores) = {threshold:.2f}",
    )
    if training_threshold is not None:
        ax.axhline(
            training_threshold,
            color=C_NEUTRAL,
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
            label=f"Training-period threshold (no recal) = {training_threshold:.2f}",
        )

    failure_handle = None
    for _, row in failures.iterrows():
        start = pd.to_datetime(row["start_time"])
        end = pd.to_datetime(row["end_time"])
        span = ax.axvspan(start, end, color=C_ANOMALY, alpha=0.20)
        failure_handle = span

    handles, labels = ax.get_legend_handles_labels()
    if recal_span is not None:
        handles.append(recal_span)
        labels.append(f"Recalibration window ({recal_start.date()} → {recal_end.date()})")
    if failure_handle is not None:
        handles.append(failure_handle)
        labels.append("Labeled failure interval")
    ax.legend(handles, labels, loc="upper right", fontsize=8)

    headline = metrics["headline_recalibrated"]["raw"]
    persistence = metrics["headline_recalibrated"]["persistence"]
    test_start = pd.to_datetime(metrics["split"]["test_start"]).date()
    test_end = pd.to_datetime(metrics["split"]["test_end"]).date()
    ax.set_title(
        f"Anomaly score timeline — test horizon {test_start} → {test_end}\n"
        f"Raw recall {headline['recall']:.0%}, precision {headline['precision']:.0%} | "
        f"+ persistence: recall {persistence['recall']:.0%}, precision {persistence['precision']:.0%} | "
        f"AUC {headline['auc']:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Smoothed anomaly score")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_anomaly_score_timeline.png", dpi=150)
    plt.close(fig)


def plot_score_distribution(
    timeline: pd.DataFrame,
    threshold: float,
    oracle_threshold: float,
    recal_end: pd.Timestamp,
) -> None:
    """Histogram of healthy vs failure scores on the test horizon only."""

    test = timeline[timeline["window_start"] >= recal_end]
    scores = test["score_smoothed"].to_numpy()
    labels = test["label"].astype(bool).to_numpy()

    healthy_scores = scores[~labels]
    failure_scores = scores[labels]

    # Use log scale; clip the upper tail for readability
    upper = float(np.percentile(scores, 99.9))
    healthy_clipped = healthy_scores[healthy_scores <= upper]
    failure_clipped = failure_scores[failure_scores <= upper]
    bins = np.linspace(0, upper, 80)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(
        healthy_clipped,
        bins=bins,
        color=C_HEALTHY,
        alpha=0.7,
        label=f"Healthy windows  (n={len(healthy_scores):,})",
    )
    ax.hist(
        failure_clipped,
        bins=bins,
        color=C_ANOMALY,
        alpha=0.7,
        label=f"Failure windows  (n={len(failure_scores):,})",
    )
    ax.axvline(threshold, color=C_THRESHOLD, linestyle="--", linewidth=2, label=f"Recalibrated threshold = {threshold:.2f}")
    ax.axvline(oracle_threshold, color=C_ORACLE, linestyle=":", linewidth=2, label=f"Oracle PR-optimal = {oracle_threshold:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Smoothed anomaly score")
    ax.set_ylabel("Window count (log scale)")
    ax.set_title("Score distribution — healthy vs failure windows (test horizon)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_score_distribution.png", dpi=150)
    plt.close(fig)


def plot_pr_curve(
    pr_data: pd.DataFrame, metrics: dict, headline_threshold: float, oracle_threshold: float
) -> None:
    """Precision-recall curve with deployable and oracle operating points."""

    precision = pr_data["precision"].to_numpy()
    recall = pr_data["recall"].to_numpy()
    thresholds = pr_data["threshold"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(recall, precision, color=C_HEALTHY, linewidth=2.0)

    # Mark operating points by snapping the threshold to nearest available point.
    def _snap(target: float) -> tuple[float, float]:
        valid = ~np.isnan(thresholds)
        idx = int(np.argmin(np.abs(thresholds[valid] - target)))
        return float(recall[valid][idx]), float(precision[valid][idx])

    h_r, h_p = _snap(headline_threshold)
    o_r, o_p = _snap(oracle_threshold)

    ax.scatter(
        [h_r], [h_p], color=C_THRESHOLD, s=120, zorder=5, edgecolor="black", linewidth=1.0,
        label=f"Recalibrated threshold\n(recall {h_r:.0%}, precision {h_p:.0%})",
    )
    ax.scatter(
        [o_r], [o_p], color=C_ORACLE, s=120, zorder=5, edgecolor="black", linewidth=1.0, marker="^",
        label=f"Oracle PR-optimal\n(recall {o_r:.0%}, precision {o_p:.0%})",
    )

    auc = metrics["headline_recalibrated"]["raw"]["auc"]
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Precision-Recall curve   (ROC-AUC = {auc:.3f})", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_precision_recall_curve.png", dpi=150)
    plt.close(fig)


def plot_latent_pca(dataset, failures: pd.DataFrame, metrics: dict) -> None:
    """PCA of the model's latent space, colored by healthy/failure label."""

    timestamps = pd.to_datetime(dataset.data[dataset.timestamp_col], errors="raise")
    eval_mask = timestamps >= TRAIN_END
    eval_df = dataset.data.loc[eval_mask].reset_index(drop=True)

    # Refit-free transform: load preprocessor saved at evaluation time.
    preprocessor = MetroPTPreprocessor.load(ARTIFACTS_DIR / "preprocessor.pkl")
    from edgesense.data_ingestion import MetroPTDataset
    eval_dataset = MetroPTDataset(
        data=eval_df,
        feature_columns=dataset.feature_columns,
        timestamp_col=dataset.timestamp_col,
        sampling_interval_seconds=dataset.sampling_interval_seconds,
        start_time=eval_df[dataset.timestamp_col].iloc[0],
        end_time=eval_df[dataset.timestamp_col].iloc[-1],
    )
    scaled_eval = preprocessor.transform(eval_dataset)
    window_size = int(metrics["windowing"]["window_size"])
    stride = int(metrics["windowing"]["stride"])
    eval_windows = create_sliding_windows(
        scaled_eval,
        window_size=window_size,
        stride=stride,
        timestamps=eval_df[dataset.timestamp_col],
    )

    # Load model
    model_config = USADConv1dConfig(**json.loads((ARTIFACTS_DIR / "model_config.json").read_text()))
    model = USADConv1d(model_config)
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "usad_conv1d.pt", map_location="cpu"))
    model.eval()

    # Window labels
    from edgesense.evaluation import label_windows_by_failures
    labels = label_windows_by_failures(eval_windows.start_times, eval_windows.end_times, failures)

    # Subsample healthy windows for visual clarity
    rng = np.random.default_rng(42)
    healthy_idx = np.where(~labels)[0]
    failure_idx = np.where(labels)[0]
    if healthy_idx.size > 2000:
        healthy_idx = rng.choice(healthy_idx, 2000, replace=False)
    selected = np.concatenate([healthy_idx, failure_idx])
    selected_labels = labels[selected]
    windows_subset = eval_windows.windows[selected]

    latent_vectors: list[np.ndarray] = []
    batch_size = 256
    with torch.no_grad():
        for start in range(0, windows_subset.shape[0], batch_size):
            batch = torch.tensor(windows_subset[start : start + batch_size], dtype=torch.float32)
            latent = model.encode(batch)
            latent_vectors.append(latent.mean(dim=2).numpy())
    latent_arr = np.vstack(latent_vectors)

    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(latent_arr)
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(
        projected[~selected_labels, 0],
        projected[~selected_labels, 1],
        s=10,
        alpha=0.55,
        color=C_HEALTHY,
        label=f"Healthy ({(~selected_labels).sum():,} pts, subsampled)",
    )
    if selected_labels.any():
        ax.scatter(
            projected[selected_labels, 0],
            projected[selected_labels, 1],
            s=14,
            alpha=0.85,
            color=C_ANOMALY,
            label=f"Failure ({selected_labels.sum():,} pts)",
        )
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% var)")
    ax.set_title("Latent space — PCA of encoder output, eval period", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_latent_pca.png", dpi=150)
    plt.close(fig)


def plot_june_zoom(
    timeline: pd.DataFrame, failures: pd.DataFrame, threshold: float
) -> None:
    """Zoomed score view around the June Air Leak (failure #3)."""

    june_failure = failures.iloc[2]
    start = pd.to_datetime(june_failure["start_time"])
    end = pd.to_datetime(june_failure["end_time"])
    pad = pd.Timedelta(hours=24)

    mask = timeline["window_mid"].between(start - pad, end + pad)
    subset = timeline.loc[mask]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        subset["window_mid"],
        subset["score_raw"],
        color=C_NEUTRAL,
        alpha=0.45,
        linewidth=0.7,
        label="Raw score",
    )
    ax.plot(
        subset["window_mid"],
        subset["score_smoothed"],
        color=C_HEALTHY,
        linewidth=1.8,
        label="Smoothed score",
    )
    ax.axhline(threshold, color=C_THRESHOLD, linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.2f}")
    ax.axvspan(start, end, color=C_ANOMALY, alpha=0.18, label="Labeled failure")

    # Mark the first time the smoothed score crosses the threshold within the failure interval.
    inside = subset["window_mid"].between(start, end)
    first_cross = subset.loc[inside & (subset["score_smoothed"] >= threshold), "window_mid"]
    if not first_cross.empty:
        first = first_cross.iloc[0]
        latency = first - start
        ax.axvline(first, color=C_BEST, linestyle=":", linewidth=1.5)
        ax.annotate(
            f"first cross\n+{latency}",
            xy=(first, threshold),
            xytext=(first + pd.Timedelta(hours=3), threshold * 1.4),
            color=C_BEST,
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1),
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Anomaly score")
    ax.set_title(
        f"June Air Leak (failure #3) — {start.date()} to {end.date()}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_june_failure_zoom.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
