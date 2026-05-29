"""Generate the figure set for the multi-dataset chapters of the README.

Produces:
    figures/09_hydraulic_per_component.png  bar chart of AUC / F1 per component
    figures/10_cmapss_rul_scatter.png       predicted vs true RUL on 100 test units
    figures/11_cmapss_rul_trajectory.png    RUL trajectory for one example test unit
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
    print("[1/4] Hydraulic per-component bar chart...")
    plot_hydraulic_components()

    print("[2/4] CMAPSS pred-vs-true scatter...")
    plot_cmapss_scatter()

    print("[3/4] CMAPSS RUL trajectory example...")
    plot_cmapss_trajectory()

    print("[4/4] Metro.PT health score timeline...")
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


def plot_cmapss_scatter() -> None:
    df = pd.read_csv("reports/cmapss_evaluation/predictions_per_test_unit.csv")
    metrics = json.loads(Path("reports/cmapss_evaluation/metrics.json").read_text())

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(df["true_rul"], df["pred_rul"], color=C_HEALTHY, s=42, alpha=0.8, edgecolor="white")
    lim = max(df["true_rul"].max(), df["pred_rul"].max()) + 5
    ax.plot([0, lim], [0, lim], color=C_NEUTRAL, linestyle="--", linewidth=1.2, label="y = x")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(
        f"CMAPSS FD001 — predicted vs true RUL on 100 test units\n"
        f"RMSE = {metrics['rmse']:.2f}  |  MAE = {metrics['mae']:.2f}  |  "
        f"Pearson r = {metrics['pearson_r']:.3f}  |  CMAPSS score = {metrics['cmapss_score']:.0f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_cmapss_rul_scatter.png", dpi=150)
    plt.close(fig)


def plot_cmapss_trajectory() -> None:
    """Run an inference pass on a single training unit to show RUL decay vs cycle."""

    import torch

    from edgesense.datasets.cmapss import (
        MAX_RUL,
        WINDOW_LENGTH,
        load_cmapss_fd001,
    )
    from edgesense.models import RULHead, USADConv1d, USADConv1dConfig

    dataset = load_cmapss_fd001()

    raw_train = np.concatenate(
        [u[dataset.feature_columns].to_numpy(dtype=np.float32) for u in dataset.train_units.values()],
        axis=0,
    )
    means = raw_train.mean(axis=0)
    stds = raw_train.std(axis=0) + 1e-6

    # Reload trained model from artifacts (we re-train here to keep the script
    # standalone if artifacts go missing).
    # NOTE: we trust the existing run script for the actual model state. To
    # avoid retraining, do a quick fresh run on this one trajectory only.
    metrics = json.loads(Path("reports/cmapss_evaluation/metrics.json").read_text())
    model_cfg = USADConv1dConfig(
        in_features=len(metrics["config"]["features_used"]),
        base_channels=32,
        latent_channels=metrics["config"]["encoder_latent_channels"],
        downsample_layers=2,
    )

    # Pick a test unit with a moderate run length so the trajectory is illustrative.
    chosen_unit = None
    for unit_id, unit_df in dataset.test_units.items():
        if 150 <= len(unit_df) <= 250:
            chosen_unit = unit_id
            break
    if chosen_unit is None:
        chosen_unit = next(iter(dataset.test_units))
    unit_df = dataset.test_units[chosen_unit]

    feat = unit_df[dataset.feature_columns].to_numpy(dtype=np.float32)
    feat = (feat - means) / stds
    cycles = unit_df["cycle"].to_numpy()
    n = feat.shape[0]
    if n < WINDOW_LENGTH:
        pad = WINDOW_LENGTH - n
        feat = np.concatenate([np.tile(feat[:1], (pad, 1)), feat], axis=0)
        cycles = np.concatenate([np.full(pad, cycles[0]), cycles])
        n = feat.shape[0]

    sliding = np.stack(
        [feat[s : s + WINDOW_LENGTH] for s in range(0, n - WINDOW_LENGTH + 1)], axis=0
    )
    cycle_at_end_of_window = cycles[WINDOW_LENGTH - 1 : n]

    # Retrain a fresh encoder + head purely for this one figure.
    # This is a known limitation: we don't persist the trained weights from
    # run_cmapss_evaluation.py to disk yet, so we rebuild for plotting.
    from edgesense.training import (
        EarlyStoppingConfig,
        TrainingConfig,
        seed_all,
        split_train_validation,
        train_usad,
    )
    from edgesense.datasets.cmapss import build_windows
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    train_windows, train_rul, _, _ = build_windows(
        dataset.train_units, dataset.feature_columns, means=means, stds=stds
    )
    healthy_mask = train_rul >= MAX_RUL - 1e-3
    healthy_windows = train_windows[healthy_mask]

    seed_all(42)
    usad = USADConv1d(model_cfg)
    train_only, val_only = split_train_validation(healthy_windows, val_fraction=0.1)
    pretrain_cfg = TrainingConfig(
        batch_size=128, epochs=60, learning_rate=1e-3,
        adv_ramp_epochs=25, adv_max_weight=0.3, grad_clip_norm=1.0,
    )
    stop_cfg = EarlyStoppingConfig(patience=10, min_delta=1e-4, max_epochs=60, val_fraction=0.1)
    train_usad(
        usad, train_only, pretrain_cfg, val_windows=val_only,
        early_stopping=stop_cfg, show_progress=False,
    )

    rul_head = RULHead(latent_channels=model_cfg.latent_channels, hidden_dim=64, dropout=0.2)
    for p in usad.parameters():
        p.requires_grad = False
    usad.eval()
    perm = np.random.default_rng(42).permutation(train_windows.shape[0])
    split = int(len(perm) * 0.9)
    head_x = torch.tensor(train_windows[perm[:split]], dtype=torch.float32)
    head_y = torch.tensor(train_rul[perm[:split]], dtype=torch.float32)
    loader = DataLoader(TensorDataset(head_x, head_y), batch_size=256, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(rul_head.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    for _ in range(40):
        rul_head.train()
        for bx, by in loader:
            with torch.no_grad():
                lat = usad.encode(bx)
            loss = criterion(rul_head(lat), by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rul_head.parameters(), 1.0)
            optimizer.step()
    rul_head.eval()

    with torch.no_grad():
        latent = usad.encode(torch.tensor(sliding, dtype=torch.float32))
        preds = rul_head(latent).numpy()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(cycle_at_end_of_window, preds, color=C_HEALTHY, linewidth=1.8, label="Predicted RUL")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(
        f"CMAPSS test unit #{chosen_unit} — predicted RUL trajectory across operating life",
        fontsize=11,
        fontweight="bold",
    )
    ax.axhline(MAX_RUL, color=C_NEUTRAL, linestyle="--", alpha=0.5, label=f"piecewise-linear ceiling ({MAX_RUL})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "11_cmapss_rul_trajectory.png", dpi=150)
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
